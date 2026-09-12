import argparse
import os
import sys
import traceback
from pathlib import Path

from src import outputs
from src.config import (
    AUDIO_FORMATS,
    DEFAULT_PRESET,
    INPUT_DIR,
    OUTPUT_DIR,
    OUTPUT_FORMATS,
    resolve_runtime_settings,
)
from src.transcriber import load_model, transcribe_audio
from src.utils import (
    ensure_directories,
    get_cuda_status,
    scan_audio_files,
    sanitize_output_filename,
    validate_output_path,
    write_log,
)


def _should_pause_for_debug():
    return os.environ.get("TRANSCRIBE_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _build_parser():
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("paths", nargs="*", help="Audio file(s) to transcribe")
    parser.add_argument(
        "--preset",
        default=None,
        choices=["default", "safe"],
        help=f"Runtime preset. Defaults to WHISPER_PRESET or '{DEFAULT_PRESET}'.",
    )
    parser.add_argument("--model", default=None, help="Whisper model name")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument("--compute-type", default=None, help="Optional compute type override")
    parser.add_argument("--language", default=None, help="Language code override")
    parser.add_argument("--beam-size", type=int, default=None, help="Beam size override")
    parser.add_argument("--temperature", type=float, default=None, help="Decoder temperature")
    parser.add_argument("--no-speech-threshold", type=float, default=None)
    parser.add_argument("--logprob-threshold", type=float, default=None)
    parser.add_argument("--compression-ratio-threshold", type=float, default=None)
    parser.add_argument(
        "--vad-filter",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable VAD filtering",
    )
    parser.add_argument("--initial-prompt", default=None, help="Optional initial prompt")
    parser.add_argument("--prefix", default=None, help="Optional decode prefix")
    parser.add_argument("--hotwords", default=None, help="Optional hotword hints")
    parser.add_argument(
        "--condition-on-previous-text",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Condition each decode window on previous text",
    )
    parser.add_argument(
        "--mixed-script-mode",
        default=None,
        choices=["off", "flag", "reject"],
        help="How to handle mixed-script output after inference",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Optional custom output directory (defaults to 'output').",
    )
    return parser


def _collect_targets(raw_paths):
    if raw_paths:
        return [Path(path) for path in raw_paths], True
    return scan_audio_files(INPUT_DIR), False


def _validate_requested_targets(targets, explicit_paths_requested):
    if not explicit_paths_requested:
        return [Path(path) for path in targets]

    valid_targets = []
    for raw_target in targets:
        audio_path = Path(raw_target)
        if not audio_path.exists():
            write_log(f"File not found: {audio_path}", level="WARNING")
            print(f"{audio_path} not found")
            continue

        if audio_path.suffix.lower() not in AUDIO_FORMATS:
            write_log(f"Unsupported format skipped: {audio_path}", level="WARNING")
            print(f"Unsupported format skipped: {audio_path}")
            continue

        valid_targets.append(audio_path)

    return valid_targets


def main(args_list=None):
    parser = _build_parser()
    args = parser.parse_args(args_list)

    target_output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else OUTPUT_DIR
    )
    target_output_dir.mkdir(parents=True, exist_ok=True)

    ensure_directories()

    settings = resolve_runtime_settings(
        preset=args.preset,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        temperature=args.temperature,
        no_speech_threshold=args.no_speech_threshold,
        logprob_threshold=args.logprob_threshold,
        compression_ratio_threshold=args.compression_ratio_threshold,
        initial_prompt=args.initial_prompt,
        prefix=args.prefix,
        hotwords=args.hotwords,
        condition_on_previous_text=args.condition_on_previous_text,
        mixed_script_mode=args.mixed_script_mode,
    )

    requested_paths, explicit_paths_requested = _collect_targets(args.paths)

    if not requested_paths:
        print("No supported audio files found in the input directory.")
        return 0

    targets = _validate_requested_targets(requested_paths, explicit_paths_requested)
    if explicit_paths_requested and not targets:
        write_log("All requested files were invalid, missing, or unsupported.", level="ERROR")
        return 1

    write_log(
        "Runtime settings resolved: "
        f"preset={settings['preset']}, "
        f"model={settings['model_name']}, "
        f"requested_device={settings['device']}, "
        f"requested_compute_type={settings['compute_type'] or 'auto'}"
    )
    print(f"Selected preset: {settings['preset']}")
    print(f"Requested device before model load: {settings['device']}")

    try:
        model, compute_type, model_device = load_model(
            model_path=settings["model_path"],
            device=settings["device"],
            compute_type=settings["compute_type"],
        )
    except Exception as exc:
        write_log(f"Failed to load model: {exc}")
        print(f"Failed to load model: {exc}", file=sys.stderr)
        traceback.print_exc()

        if _should_pause_for_debug():
            input("\nPress Enter to exit...")

        return 1

    status = get_cuda_status()
    selected_device = model_device or settings["device"]

    print(f"Selected device: {selected_device}")
    print(f"Compute type: {compute_type}")
    print(f"Language: {settings['language']}")
    print(f"VAD enabled: {settings['vad_filter']}")
    print(f"Mixed-script mode: {settings['mixed_script_mode']}")
    print(f"CUDA available: {status['cuda_available']}")
    print(f"CTranslate2 CUDA device count: {status['ctranslate2_cuda_device_count']}")
    print(f"GPU name: {status['gpu_name']}")

    exit_code = 0
    successful_targets = 0

    for audio_path in targets:
        audio_path = Path(audio_path)

        try:
            print(f"\nProcessing: {audio_path.name}")
            result = transcribe_audio(
                audio_path,
                model,
                device=selected_device,
                language=settings["language"],
                beam_size=settings["beam_size"],
                vad_filter=settings["vad_filter"],
                temperature=settings["temperature"],
                no_speech_threshold=settings["no_speech_threshold"],
                logprob_threshold=settings["logprob_threshold"],
                compression_ratio_threshold=settings["compression_ratio_threshold"],
                model_path=settings["model_path"],
                initial_prompt=settings["initial_prompt"],
                prefix=settings["prefix"],
                hotwords=settings["hotwords"],
                condition_on_previous_text=settings["condition_on_previous_text"],
                mixed_script_mode=settings["mixed_script_mode"],
            )

            for warning in result.get("warnings", []):
                print(f"Warning: {warning}")

            # Sanitize the output filename to remove invalid characters
            sanitized_stem = sanitize_output_filename(audio_path.stem)

            for fmt in OUTPUT_FORMATS:
                output_path = target_output_dir / f"{sanitized_stem}.{fmt}"

                # Validate that output path is within target_output_dir (no path traversal)
                if not validate_output_path(output_path, target_output_dir):
                    write_log(f"ERROR: Invalid output path for {audio_path.name}: {output_path}")
                    print(f"Error: Invalid output path for {audio_path.name}", file=sys.stderr)
                    raise ValueError(f"Output path outside allowed directory: {output_path}")

                if fmt == "txt":
                    outputs.save_txt(result, output_path)
                elif fmt == "docx":
                    outputs.save_docx(result, output_path)
                elif fmt == "srt":
                    outputs.save_srt(result, output_path)
                elif fmt == "json":
                    outputs.save_json(result, output_path, audio_path=audio_path)

            write_log(f"Completed successfully: {audio_path.name}")
            print(f"{audio_path.name} SUCCESS")
            successful_targets += 1
        except Exception as exc:
            write_log(f"FAILED: {audio_path.name} -> {exc}")
            print(f"Error processing {audio_path.name}: {exc}", file=sys.stderr)
            traceback.print_exc()

            if _should_pause_for_debug():
                input("\nPress Enter to exit...")

            exit_code = 1

    if targets and successful_targets == 0:
        return 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
