from faster_whisper import WhisperModel
import os
import time
import traceback
import unicodedata
from pathlib import Path

from tqdm import tqdm

from src.config import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_CPU_THREADS,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_LOGPROB_THRESHOLD,
    DEFAULT_NO_SPEECH_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_VAD_FILTER,
    resolve_model_path,
)
from src.utils import GPUMonitor, correct_persian_spelling, detect_device, write_log


def _is_gpu_memory_error(exc):
    message = str(exc).lower()
    return "out of memory" in message or "cuda failed" in message or (
        "cuda" in message and "memory" in message
    )


def _get_supported_compute_types(device_name):
    """Query CTranslate2 for hardware-supported compute types on this device."""
    try:
        import ctranslate2
        if hasattr(ctranslate2, "get_supported_compute_types"):
            return set(ctranslate2.get_supported_compute_types(device_name))
    except Exception:
        pass
    return None


def _build_compute_type_candidates(device_name, requested_compute_type=None):
    device_name = (device_name or "").lower()
    candidates = []

    supported = _get_supported_compute_types(device_name)

    if requested_compute_type:
        candidates.append(requested_compute_type)

    if device_name == "cuda":
        # Preferred order for NVIDIA GPUs from fastest/optimal to legacy:
        # float16 (modern GPUs) -> bfloat16 (Ampere+) -> int8_float16 -> int8_float32 -> int8 -> float32
        preferred_cuda = ["float16", "bfloat16", "int8_float16", "int8_float32", "int8", "float32"]
        if supported is not None:
            # Add preferred types supported by this specific GPU architecture
            for c_type in preferred_cuda:
                if c_type in supported and c_type not in candidates:
                    candidates.append(c_type)
            # If for some reason none of preferred matched, add all reported supported
            for c_type in supported:
                if c_type not in candidates:
                    candidates.append(c_type)
        else:
            candidates.extend(preferred_cuda)

    elif device_name == "cpu":
        # Preferred order for CPUs: int8 (fast vectorized) -> int8_float32 -> float32
        preferred_cpu = ["int8", "int8_float32", "float32"]
        if supported is not None:
            for c_type in preferred_cpu:
                if c_type in supported and c_type not in candidates:
                    candidates.append(c_type)
            for c_type in supported:
                if c_type not in candidates:
                    candidates.append(c_type)
        else:
            candidates.extend(preferred_cpu)

    else:
        candidates.append(DEFAULT_COMPUTE_TYPE or "float32")

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def load_model(model_path=None, device=None, compute_type=None, cpu_threads=None):
    resolved_model_path = resolve_model_path(model_path=model_path or None)
    device_name = (device or DEFAULT_DEVICE or "auto").lower()
    if device_name not in {"cuda", "cpu"}:
        device_name = detect_device().lower()

    if cpu_threads is None:
        cpu_threads = DEFAULT_CPU_THREADS

    compute_type_candidates = _build_compute_type_candidates(device_name, compute_type)
    last_error = None

    for candidate_compute_type in compute_type_candidates:
        write_log(
            f"Attempting to load model from {resolved_model_path} with device={device_name}, compute_type={candidate_compute_type}"
        )
        try:
            model_kwargs = {}
            if cpu_threads > 0:
                model_kwargs["cpu_threads"] = cpu_threads
            model = WhisperModel(
                str(resolved_model_path),
                device=device_name,
                compute_type=candidate_compute_type,
                **model_kwargs,
            )
            write_log(
                f"Model loaded successfully using device={device_name}, compute_type={candidate_compute_type}"
            )
            return model, candidate_compute_type, device_name
        except Exception as exc:
            last_error = exc
            write_log(
                f"Model load failed for compute_type={candidate_compute_type} device={device_name}: {exc}"
            )
            # Try next compute type if available

    if device_name == "cuda":
        fallback_device = "cpu"
        fallback_candidates = _build_compute_type_candidates(fallback_device)
        write_log(
            f"Falling back to CPU for model loading after CUDA failure: {last_error}"
        )
        for fallback_compute_type in fallback_candidates:
            try:
                model_kwargs = {}
                if cpu_threads > 0:
                    model_kwargs["cpu_threads"] = cpu_threads
                model = WhisperModel(
                    str(resolved_model_path),
                    device=fallback_device,
                    compute_type=fallback_compute_type,
                    **model_kwargs,
                )
                write_log(
                    f"Model loaded successfully using device={fallback_device}, compute_type={fallback_compute_type}"
                )
                return model, fallback_compute_type, fallback_device
            except Exception as fallback_exc:
                write_log(
                    f"CPU fallback candidate compute_type={fallback_compute_type} failed: {fallback_exc}"
                )
                last_error = fallback_exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to initialize Whisper model")


def _script_name(char):
    if not char.isalpha():
        return None

    char_name = unicodedata.name(char, "")
    if "ARABIC" in char_name:
        return "arabic"
    if "LATIN" in char_name:
        return "latin"
    if "CYRILLIC" in char_name:
        return "cyrillic"
    return "other"


def _expected_script(language):
    normalized_language = str(language or "").strip().lower()
    if normalized_language in {"fa", "ar", "ur", "ps"}:
        return "arabic"
    if normalized_language in {"en", "fr", "de", "es", "it", "pt"}:
        return "latin"
    if normalized_language in {"ru", "uk", "bg"}:
        return "cyrillic"
    return None


def _analyze_segment_scripts(segments, language):
    expected = _expected_script(language)
    flagged_segments = []

    for index, segment in enumerate(segments, start=1):
        text = (segment.get("text", "") or "").strip()
        scripts = {script for script in (_script_name(char) for char in text) if script}
        unexpected_scripts = []

        if expected:
            unexpected_scripts = sorted(script for script in scripts if script != expected)
            is_flagged = bool(unexpected_scripts)
        else:
            is_flagged = len(scripts) > 1

        segment["detected_scripts"] = sorted(scripts)
        segment["unexpected_scripts"] = unexpected_scripts
        segment["mixed_script"] = is_flagged

        if is_flagged:
            flagged_segments.append(
                {
                    "index": index,
                    "text": text,
                    "scripts": sorted(scripts),
                    "unexpected_scripts": unexpected_scripts,
                }
            )

    return flagged_segments


def _apply_mixed_script_policy(result, language, mixed_script_mode):
    mode = str(mixed_script_mode or "off").strip().lower()
    if mode == "off":
        return result

    flagged_segments = _analyze_segment_scripts(result.get("segments", []), language)
    if not flagged_segments:
        return result

    warnings = list(result.get("warnings", []))
    for flagged in flagged_segments[:10]:
        warning = (
            f"Mixed-script segment {flagged['index']} detected "
            f"(scripts={','.join(flagged['scripts'])}): {flagged['text']}"
        )
        warnings.append(warning)
        write_log(warning, level="WARNING")

    result["warnings"] = warnings
    result["mixed_script_segments"] = flagged_segments

    if mode == "reject":
        raise RuntimeError(
            f"Rejected transcript because {len(flagged_segments)} mixed-script segment(s) were detected."
        )

    return result


def _process_segments(segments, info, audio_path, start_time, monitor=None, on_progress=None):
    total_duration = float(getattr(info, "duration", 0.0) or 0.0)
    if total_duration <= 0:
        total_duration = 1.0

    processed_duration = 0.0
    pbar = None
    if on_progress is None:
        pbar = tqdm(total=100, desc=audio_path.name, leave=False)
    segments_data = []

    try:
        for segment in segments:
            segment_start = float(getattr(segment, "start", 0.0) or 0.0)
            segment_end = float(getattr(segment, "end", 0.0) or 0.0)
            segment_text = (getattr(segment, "text", "") or "").strip()

            if segment_end > processed_duration:
                processed_duration = segment_end

            segments_data.append(
                {"start": segment_start, "end": segment_end, "text": segment_text}
            )

            percent = min(100.0, (processed_duration / total_duration) * 100.0)
            elapsed = time.time() - start_time
            speed = processed_duration / elapsed if elapsed > 0 else 0.0

            if monitor is not None:
                try:
                    stats = monitor.get_stats() or {}
                except Exception:
                    stats = {}
            else:
                stats = {}

            if on_progress is not None:
                event = {
                    "percent": percent,
                    "processed_duration": processed_duration,
                    "total_duration": total_duration,
                    "speed": speed,
                    "elapsed": elapsed,
                    "segments_count": len(segments_data),
                    "vram_mb": str(stats.get("memory", "n/a")),
                    "gpu_util": str(stats.get("utilization", "n/a")),
                    "current_text": segment_text,
                }
                try:
                    on_progress(event)
                except Exception as cb_exc:
                    write_log(f"on_progress callback error for {audio_path.name}: {cb_exc}", level="WARNING")
            else:
                pbar.n = int(percent)
                pbar.set_postfix_str(f"{processed_duration:.1f}/{total_duration:.1f}s")
                pbar.refresh()

                print(
                    f"\rProcessing {audio_path.name}: "
                    f"{percent:5.1f}% | "
                    f"{processed_duration:6.1f}/{total_duration:6.1f}s | "
                    f"Elapsed {int(elapsed)//60:02d}:{int(elapsed)%60:02d} | "
                    f"Speed {speed:4.2f}x | "
                    f"Segments {len(segments_data)} | "
                    f"GPU {stats.get('utilization', 'n/a')}% | "
                    f"VRAM {stats.get('memory', 'n/a')} MB",
                    end="",
                    flush=True,
                )
    finally:
        if pbar is not None:
            pbar.close()
            print()

    text = "\n".join([item["text"] for item in segments_data if item.get("text")]).strip()
    elapsed_total = time.time() - start_time
    write_log(f"Completed transcription for {audio_path.name} in {elapsed_total:.2f}s")

    return {
        "text": text,
        "segments": segments_data,
        "filename": audio_path.name,
        "duration": total_duration,
    }


def transcribe_audio(
    audio_file,
    model,
    device=None,
    language=None,
    beam_size=None,
    vad_filter=None,
    temperature=None,
    no_speech_threshold=None,
    logprob_threshold=None,
    compression_ratio_threshold=None,
    model_path=None,
    initial_prompt=None,
    prefix=None,
    hotwords=None,
    condition_on_previous_text=None,
    mixed_script_mode="off",
    enable_spell_correction=False,
    on_progress=None,
):
    audio_path = Path(audio_file).expanduser().resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if device is None:
        device = detect_device()

    if language is None:
        language = DEFAULT_LANGUAGE

    if beam_size is None:
        beam_size = DEFAULT_BEAM_SIZE

    if vad_filter is None:
        vad_filter = DEFAULT_VAD_FILTER
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    if no_speech_threshold is None:
        no_speech_threshold = DEFAULT_NO_SPEECH_THRESHOLD
    if logprob_threshold is None:
        logprob_threshold = DEFAULT_LOGPROB_THRESHOLD
    if compression_ratio_threshold is None:
        compression_ratio_threshold = DEFAULT_COMPRESSION_RATIO_THRESHOLD
    if condition_on_previous_text is None:
        condition_on_previous_text = False

    monitor = None
    try:
        monitor = GPUMonitor()
        monitor.start()
    except Exception as exc:
        write_log(f"GPU monitor unavailable: {exc}")
        monitor = None

    start_time = time.time()

    try:
        if on_progress is None:
            print(f"Loading transcription for: {audio_path.name}")

        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            temperature=temperature,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            initial_prompt=initial_prompt,
            prefix=prefix,
            hotwords=hotwords,
            condition_on_previous_text=condition_on_previous_text,
        )

        result = _process_segments(segments, info, audio_path, start_time, monitor, on_progress=on_progress)
        if enable_spell_correction:
            for seg in result.get("segments", []):
                seg["text"] = correct_persian_spelling(seg.get("text", ""))
            result["text"] = "\n".join(seg["text"] for seg in result["segments"] if seg.get("text")).strip()
        return _apply_mixed_script_policy(result, language, mixed_script_mode)
    except Exception as exc:
        if str(device).lower() == "cuda" and _is_gpu_memory_error(exc):
            fallback_device = "cpu"
            write_log(
                f"GPU transcription failed for {audio_path.name}: {exc}. Falling back to CPU."
            )
            if on_progress is None:
                print(
                    f"GPU transcription failed for {audio_path.name}: {exc}. Retrying on CPU...",
                    flush=True,
                )
            try:
                fallback_model, _, _ = load_model(
                    model_path=model_path or resolve_model_path(),
                    device=fallback_device,
                    compute_type=None,
                )
                segments, info = fallback_model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    temperature=temperature,
                    no_speech_threshold=no_speech_threshold,
                    log_prob_threshold=logprob_threshold,
                    compression_ratio_threshold=compression_ratio_threshold,
                    initial_prompt=initial_prompt,
                    prefix=prefix,
                    hotwords=hotwords,
                    condition_on_previous_text=condition_on_previous_text,
                )
                result = _process_segments(segments, info, audio_path, start_time, monitor, on_progress=on_progress)
                if enable_spell_correction:
                    for seg in result.get("segments", []):
                        seg["text"] = correct_persian_spelling(seg.get("text", ""))
                    result["text"] = "\n".join(seg["text"] for seg in result["segments"] if seg.get("text")).strip()
                return _apply_mixed_script_policy(result, language, mixed_script_mode)
            except Exception as fallback_exc:
                write_log(f"CPU fallback failed for {audio_path.name}: {fallback_exc}")
                traceback.print_exc()
                raise fallback_exc from exc

        write_log(f"Error transcribing {audio_path.name}: {exc}")
        traceback.print_exc()
        raise
    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception as stop_exc:
                write_log(f"GPU monitor stop failed: {stop_exc}")
