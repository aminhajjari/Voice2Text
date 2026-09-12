"""
Diagnostic Test #2: A/B test for condition_on_previous_text.

Runs the same audio file twice with identical settings except for
condition_on_previous_text, and saves detailed output for comparison.

Usage:
    python diagnostic_test2.py [--audio input/voice.mp3]

Produces:
    output/diagnostic_test2_A_True.json
    output/diagnostic_test2_B_False.json
    output/diagnostic_test2_comparison.txt
"""

import argparse
import copy
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.transcriber import load_model, _analyze_segment_scripts
from src.config import resolve_runtime_settings
from src.utils import ensure_directories, get_cuda_status, write_log
from src import outputs


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


def _count_anomalous_chars(text, expected_script="arabic"):
    """Count chars not in the expected script."""
    latin = cyrillic = other = 0
    for ch in text:
        if not ch.isalpha():
            continue
        s = _script_name(ch)
        if s == "latin":
            latin += 1
        elif s == "cyrillic":
            cyrillic += 1
        elif s == "other" and s != expected_script:
            other += 1
    return {"latin": latin, "cyrillic": cyrillic, "other": other}


def _detect_hallucinations(text):
    """Detect obvious hallucinations like repeated single chars, 'تتتتت', etc."""
    words = text.split()
    hallu_indicators = []
    for w in words:
        # Repeated single character
        if len(w) >= 4 and len(set(w)) == 1:
            hallu_indicators.append(w)
        # Very long word with no vowels (Arabic context)
        if len(w) > 15:
            # Check if it looks like a hallucinated repetition
            for i in range(len(w) - 3):
                if w[i] == w[i+1] == w[i+2] == w[i+3]:
                    hallu_indicators.append(w)
                    break
    return hallu_indicators


def run_single_test(audio_path, settings, condition_value, output_label):
    """Run transcription with given condition_on_previous_text value."""
    settings = dict(settings)
    settings["condition_on_previous_text"] = condition_value

    print(f"\n{'='*60}")
    print(f"Test {output_label}: condition_on_previous_text = {condition_value}")
    print(f"{'='*60}")

    model, compute_type, model_device = load_model(
        model_path=settings["model_path"],
        device=settings["device"],
        compute_type=settings.get("compute_type"),
    )

    status = get_cuda_status()
    selected_device = model_device or settings["device"]

    config_summary = {
        "condition_on_previous_text": condition_value,
        "audio_file": str(audio_path),
        "model": settings["model_name"],
        "model_path": settings["model_path"],
        "device": selected_device,
        "compute_type": compute_type,
        "language": settings["language"],
        "beam_size": settings["beam_size"],
        "vad_filter": settings["vad_filter"],
        "initial_prompt": settings.get("initial_prompt"),
        "prefix": settings.get("prefix"),
        "hotwords": settings.get("hotwords"),
        "mixed_script_mode": settings["mixed_script_mode"],
        "cuda_available": status["cuda_available"],
        "gpu_name": status["gpu_name"],
    }

    print(f"\nConfiguration:")
    for k, v in config_summary.items():
        print(f"  {k}: {v}")

    start_time = time.time()

    segments_result, info = model.transcribe(
        str(audio_path),
        language=settings["language"],
        beam_size=settings["beam_size"],
        vad_filter=settings["vad_filter"],
        initial_prompt=settings.get("initial_prompt"),
        prefix=settings.get("prefix"),
        hotwords=settings.get("hotwords"),
        condition_on_previous_text=condition_value,
    )

    segments_list = []
    for segment in segments_result:
        seg_start = float(getattr(segment, "start", 0.0) or 0.0)
        seg_end = float(getattr(segment, "end", 0.0) or 0.0)
        seg_text = (getattr(segment, "text", "") or "").strip()
        segments_list.append({
            "segment_number": len(segments_list) + 1,
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "text_length": len(seg_text),
            "anomaly_chars": _count_anomalous_chars(seg_text),
            "hallucinations": _detect_hallucinations(seg_text),
        })

    elapsed = time.time() - start_time
    full_text = "\n".join(s["text"] for s in segments_list if s["text"]).strip()

    # Analyze mixed scripts across all segments
    result_dict = {
        "text": full_text,
        "segments": segments_list,
    }
    flagged = _analyze_segment_scripts(result_dict["segments"], settings["language"])

    # Count anomalies
    total_latin = sum(s["anomaly_chars"]["latin"] for s in segments_list)
    total_cyrillic = sum(s["anomaly_chars"]["cyrillic"] for s in segments_list)

    # Collect all foreign words
    foreign_words = []
    for s in segments_list:
        words = s["text"].split()
        for w in words:
            scripts_in_word = set()
            for ch in w:
                if not ch.isalpha():
                    continue
                sn = _script_name(ch)
                if sn and sn != "arabic":
                    scripts_in_word.add(sn)
            if scripts_in_word:
                foreign_words.append({
                    "word": w,
                    "scripts": list(scripts_in_word),
                    "segment": s["segment_number"],
                    "start": s["start"],
                    "end": s["end"],
                })

    result = {
        "config": config_summary,
        "elapsed_seconds": elapsed,
        "total_segments": len(segments_list),
        "total_text_length": len(full_text),
        "full_text": full_text,
        "segments": segments_list,
        "flagged_segments": flagged,
        "mixed_script_segment_count": len(flagged),
        "anomaly_summary": {
            "latin_chars": total_latin,
            "cyrillic_chars": total_cyrillic,
        },
        "foreign_words": foreign_words,
        "foreign_word_count": len(foreign_words),
        "all_hallucinations": list(set(
            h for s in segments_list for h in s["hallucinations"]
        )),
    }

    # Save JSON
    output_dir = Path(project_root) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"diagnostic_test2_{output_label}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Also save the full text
    txt_path = output_dir / f"diagnostic_test2_{output_label}.txt"
    txt_path.write_text(full_text, encoding="utf-8")

    print(f"\nResults for Test {output_label}:")
    print(f"  Segments: {len(segments_list)}")
    print(f"  Mixed-script flagged segments: {len(flagged)}")
    print(f"  Latin chars: {total_latin}")
    print(f"  Cyrillic chars: {total_cyrillic}")
    print(f"  Foreign word occurrences: {len(foreign_words)}")
    print(f"  Hallucinations: {result['all_hallucinations']}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Saved to: {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic Test #2: A/B test for condition_on_previous_text"
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Audio file path (default: first .mp3 in input/)",
    )
    args = parser.parse_args()

    ensure_directories()

    if args.audio:
        audio_path = Path(args.audio)
    else:
        input_dir = project_root / "input"
        mp3_files = list(input_dir.glob("*.mp3"))
        if not mp3_files:
            print("No MP3 files found in input/ and no --audio specified.")
            return 1
        audio_path = mp3_files[0]

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return 1

    print(f"Audio file: {audio_path}")

    # Resolve base settings (using default preset to get True default)
    base_settings = resolve_runtime_settings(
        preset="default",
        language="fa",
        mixed_script_mode="flag",  # Enable flagging for diagnostics
    )

    print(f"Base settings loaded.")
    print(f"Default condition_on_previous_text: {base_settings['condition_on_previous_text']}")

    # Test A: condition_on_previous_text = True
    result_a = run_single_test(
        audio_path, base_settings, condition_value=True, output_label="A_True"
    )

    # Test B: condition_on_previous_text = False
    result_b = run_single_test(
        audio_path, base_settings, condition_value=False, output_label="B_False"
    )

    # ---- COMPARISON ----
    print(f"\n\n{'='*60}")
    print(f"COMPARISON: True vs False")
    print(f"{'='*60}")

    # 1. Foreign words in both runs
    fw_a_set = set(fw["word"] for fw in result_a["foreign_words"])
    fw_b_set = set(fw["word"] for fw in result_b["foreign_words"])

    foreign_in_both = fw_a_set & fw_b_set
    foreign_only_a = fw_a_set - fw_b_set
    foreign_only_b = fw_b_set - fw_a_set

    print(f"\n1. Foreign words in BOTH runs ({len(foreign_in_both)}):")
    for w in sorted(foreign_in_both):
        print(f"   - {w}")

    print(f"\n2. Foreign words ONLY when True ({len(foreign_only_a)}):")
    for w in sorted(foreign_only_a):
        print(f"   - {w}")

    print(f"\n3. Foreign words ONLY when False ({len(foreign_only_b)}):")
    for w in sorted(foreign_only_b):
        print(f"   - {w}")

    # 4. Cascading errors analysis
    print(f"\n4. Cascading error analysis:")
    # Check if a word in segment N with True is followed by errors in segment N+1
    seg_a = {s["segment_number"]: s for s in result_a["segments"]}
    seg_b = {s["segment_number"]: s for s in result_b["segments"]}

    cascade_true = 0
    for i in range(1, len(seg_a)):
        current = seg_a.get(i, {})
        next_seg = seg_a.get(i+1, {})
        if current.get("anomaly_chars", {}).get("latin", 0) > 0:
            if next_seg.get("anomaly_chars", {}).get("latin", 0) > 0:
                cascade_true += 1

    cascade_false = 0
    for i in range(1, len(seg_b)):
        current = seg_b.get(i, {})
        next_seg = seg_b.get(i+1, {})
        if current.get("anomaly_chars", {}).get("latin", 0) > 0:
            if next_seg.get("anomaly_chars", {}).get("latin", 0) > 0:
                cascade_false += 1

    print(f"   Consecutive segments with anomalies (True): {cascade_true}")
    print(f"   Consecutive segments with anomalies (False): {cascade_false}")

    # 5-6. Summary
    print(f"\n5. Summary statistics:")

    table = f"""
| Metric                          | True    | False   |
| ------------------------------- | ------- | ------- |
| Total segments                  | {result_a['total_segments']:>7} | {result_b['total_segments']:>7} |
| Segments with mixed scripts     | {result_a['mixed_script_segment_count']:>7} | {result_b['mixed_script_segment_count']:>7} |
| English/Latin chars             | {result_a['anomaly_summary']['latin_chars']:>7} | {result_b['anomaly_summary']['latin_chars']:>7} |
| Russian/Cyrillic chars          | {result_a['anomaly_summary']['cyrillic_chars']:>7} | {result_b['anomaly_summary']['cyrillic_chars']:>7} |
| Foreign word occurrences        | {result_a['foreign_word_count']:>7} | {result_b['foreign_word_count']:>7} |
| Unique foreign words            | {len(fw_a_set):>7} | {len(fw_b_set):>7} |
| Obvious hallucinations          | {len(result_a['all_hallucinations']):>7} | {len(result_b['all_hallucinations']):>7} |
| Consecutive anomaly segments    | {cascade_true:>7} | {cascade_false:>7} |
"""
    print(table)

    # Save comparison
    comparison = {
        "test_A_True": {
            "total_segments": result_a["total_segments"],
            "mixed_script_segments": result_a["mixed_script_segment_count"],
            "latin_chars": result_a["anomaly_summary"]["latin_chars"],
            "cyrillic_chars": result_a["anomaly_summary"]["cyrillic_chars"],
            "foreign_word_count": result_a["foreign_word_count"],
            "unique_foreign_words": len(fw_a_set),
            "hallucinations": result_a["all_hallucinations"],
        },
        "test_B_False": {
            "total_segments": result_b["total_segments"],
            "mixed_script_segments": result_b["mixed_script_segment_count"],
            "latin_chars": result_b["anomaly_summary"]["latin_chars"],
            "cyrillic_chars": result_b["anomaly_summary"]["cyrillic_chars"],
            "foreign_word_count": result_b["foreign_word_count"],
            "unique_foreign_words": len(fw_b_set),
            "hallucinations": result_b["all_hallucinations"],
        },
        "shared_foreign_words": sorted(foreign_in_both),
        "foreign_words_only_when_True": sorted(foreign_only_a),
        "foreign_words_only_when_False": sorted(foreign_only_b),
        "consecutive_anomaly_segments_True": cascade_true,
        "consecutive_anomaly_segments_False": cascade_false,
    }

    comp_path = Path(project_root) / "output" / "diagnostic_test2_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # Print final table
    print(f"\nFinal Comparison Table:")
    print(table)

    # Conclusion
    delta_segments = result_a["mixed_script_segment_count"] - result_b["mixed_script_segment_count"]
    delta_foreign = result_a["foreign_word_count"] - result_b["foreign_word_count"]
    delta_cascade = cascade_true - cascade_false

    print(f"Conclusions:")
    if delta_segments > 0 and delta_foreign > 0:
        print("  Strong evidence that condition_on_previous_text=True causes/amplifies the problem")
    elif delta_segments > 0 or delta_foreign > 0:
        print("  Some evidence")
    else:
        print("  No meaningful evidence")

    print(f"\n  Delta (True - False):")
    print(f"    Mixed-script segments: {delta_segments:+d}")
    print(f"    Foreign word occurrences: {delta_foreign:+d}")
    print(f"    Consecutive anomaly segments: {delta_cascade:+d}")

    print(f"\nDetailed JSON results saved to output/")
    return 0


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    sys.exit(main())
