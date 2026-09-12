"""
Simple A/B test runner for condition_on_previous_text.
Runs both conditions sequentially and dumps all output to files.
"""
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Redirect stdout/stderr to log file immediately
log_path = os.path.join(project_root, "output", "ab_test_log.txt")
os.makedirs(os.path.join(project_root, "output"), exist_ok=True)
sys.stdout = open(log_path, "w", encoding="utf-8", buffering=1)
sys.stderr = sys.stdout

print("=" * 60)
print(f"DIAGNOSTIC TEST #2: condition_on_previous_text A/B Test")
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

from src.transcriber import load_model
from src.config import resolve_runtime_settings
from src.utils import get_cuda_status

# 1. Resolve settings
settings = resolve_runtime_settings(
    preset="default",
    language="fa",
    mixed_script_mode="flag",
)
print(f"\nConfiguration resolved:")
for k, v in settings.items():
    print(f"  {k}: {v}")

# 2. Load model once
print(f"\nLoading model from: {settings['model_path']}")
print(f"  device: {settings['device']}")
print(f"  compute_type: {settings.get('compute_type') or 'auto'}")
sys.stdout.flush()

model, compute_type, model_device = load_model(
    model_path=settings["model_path"],
    device=settings["device"],
    compute_type=settings.get("compute_type"),
)
print(f"Model loaded: device={model_device}, compute_type={compute_type}")

status = get_cuda_status()
print(f"CUDA: {status['cuda_available']}, GPU: {status['gpu_name']}")

audio_path = os.path.join(project_root, "input", "voice.mp3")
print(f"\nAudio file: {audio_path}")

def run_test(condition_value, label):
    print(f"\n{'=' * 50}")
    print(f"RUN {label}: condition_on_previous_text = {condition_value}")
    print(f"{'=' * 50}")
    sys.stdout.flush()

    start_time = time.time()

    segments_gen, info = model.transcribe(
        audio_path,
        language=settings["language"],
        beam_size=settings["beam_size"],
        vad_filter=settings["vad_filter"],
        initial_prompt=settings.get("initial_prompt"),
        prefix=settings.get("prefix"),
        hotwords=settings.get("hotwords"),
        condition_on_previous_text=condition_value,
    )

    segments = []
    for seg in segments_gen:
        s = {
            "segment": len(segments) + 1,
            "start": round(float(getattr(seg, "start", 0) or 0), 2),
            "end": round(float(getattr(seg, "end", 0) or 0), 2),
            "text": (getattr(seg, "text", "") or "").strip(),
        }
        segments.append(s)
        print(f"  Seg {s['segment']:3d}: {s['start']:6.2f}-{s['end']:6.2f} | {s['text']}")
        sys.stdout.flush()

    elapsed = time.time() - start_time
    full_text = "\n".join(s["text"] for s in segments if s["text"]).strip()

    result = {
        "condition_on_previous_text": condition_value,
        "label": label,
        "elapsed_seconds": round(elapsed, 2),
        "total_segments": len(segments),
        "full_text": full_text,
        "segments": segments,
    }

    # Save JSON
    json_path = os.path.join(project_root, "output", f"ab_test_{label}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save text
    txt_path = os.path.join(project_root, "output", f"ab_test_{label}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\n  Completed in {elapsed:.2f}s")
    print(f"  {len(segments)} segments")
    print(f"  Saved: {json_path}")
    sys.stdout.flush()

    return result

# Test A: True
result_a = run_test(condition_value=True, label="A_True")

# Test B: False
result_b = run_test(condition_value=False, label="B_False")

# COMPARISON
print(f"\n\n{'=' * 60}")
print(f"COMPARISON: True vs False")
print(f"{'=' * 60}")

# Analyze scripts - simple function
import unicodedata

def script_name(char):
    if not char.isalpha():
        return None
    cn = unicodedata.name(char, "")
    if "ARABIC" in cn: return "arabic"
    if "LATIN" in cn: return "latin"
    if "CYRILLIC" in cn: return "cyrillic"
    return "other"

def analyze_segments(segments):
    results = []
    for s in segments:
        text = s["text"]
        latin_count = 0
        cyrillic_count = 0
        foreign_words_list = []
        for w in text.split():
            has_latin = any(script_name(c) == "latin" for c in w if c.isalpha())
            has_cyrillic = any(script_name(c) == "cyrillic" for c in w if c.isalpha())
            if has_latin or has_cyrillic:
                foreign_words_list.append(w)
            if has_latin: latin_count += len([c for c in w if script_name(c) == "latin"])
            if has_cyrillic: cyrillic_count += len([c for c in w if script_name(c) == "cyrillic"])
        results.append({
            "segment": s["segment"],
            "start": s["start"],
            "end": s["end"],
            "text": text,
            "latin": latin_count,
            "cyrillic": cyrillic_count,
            "foreign_words": foreign_words_list,
        })
    return results

ana_a = analyze_segments(result_a["segments"])
ana_b = analyze_segments(result_b["segments"])

# Count metrics
for label, ana, res in [("True", ana_a, result_a), ("False", ana_b, result_b)]:
    mixed_script_segs = sum(1 for a in ana if a["latin"] > 0 or a["cyrillic"] > 0)
    all_foreign_words = set()
    for a in ana:
        all_foreign_words.update(a["foreign_words"])
    total_latin = sum(a["latin"] for a in ana)
    total_cyrillic = sum(a["cyrillic"] for a in ana)
    total_foreign_occurrences = sum(len(a["foreign_words"]) for a in ana)
    
    print(f"\n--- {label} ---")
    print(f"  Segments: {res['total_segments']}")
    print(f"  Mixed-script segments: {mixed_script_segs}")
    print(f"  Latin chars: {total_latin}")
    print(f"  Cyrillic chars: {total_cyrillic}")
    print(f"  Foreign word occurrences: {total_foreign_occurrences}")
    print(f"  Unique foreign words: {len(all_foreign_words)}")
    print(f"  Foreign words: {sorted(all_foreign_words)}")

# Cascading analysis
def cascade_count(ana):
    count = 0
    for i in range(len(ana) - 1):
        if (ana[i]["latin"] + ana[i]["cyrillic"]) > 0 and (ana[i+1]["latin"] + ana[i+1]["cyrillic"]) > 0:
            count += 1
    return count

cascade_true = cascade_count(ana_a)
cascade_false = cascade_count(ana_b)

print(f"\nCascading consecutive anomaly segments:")
print(f"  True: {cascade_true}")
print(f"  False: {cascade_false}")

# DETAILED SEGMENT COMPARISON
print(f"\n\n--- DETAILED SEGMENT COMPARISON ---")
max_segs = max(len(ana_a), len(ana_b))
for i in range(max_segs):
    a = ana_a[i] if i < len(ana_a) else None
    b = ana_b[i] if i < len(ana_b) else None
    if a and b:
        flag_a = "!" if a["latin"] > 0 or a["cyrillic"] > 0 else " "
        flag_b = "!" if b["latin"] > 0 or b["cyrillic"] > 0 else " "
        if flag_a != flag_b or a["text"] != b["text"]:
            print(f"\nSeg {i+1} ({a['start']:.2f}-{a['end']:.2f}):")
            print(f"  T [{flag_a}] {a['text']}")
            print(f"  F [{flag_b}] {b['text']}")
    elif a:
        print(f"\nSeg {i+1} ({a['start']:.2f}-{a['end']:.2f}):")
        print(f"  T [!] {a['text']}")
        print(f"  F [ ] (no segment)")
    elif b:
        print(f"\nSeg {i+1} ({b['start']:.2f}-{b['end']:.2f}):")
        print(f"  T [ ] (no segment)")
        print(f"  F [!] {b['text']}")

# Save comparison summary
foreign_a = set(w for a in ana_a for w in a["foreign_words"])
foreign_b = set(w for a in ana_b for w in a["foreign_words"])

comparison = {
    "True": {
        "total_segments": result_a["total_segments"],
        "mixed_script_segments": sum(1 for a in ana_a if a["latin"] > 0 or a["cyrillic"] > 0),
        "latin_chars": sum(a["latin"] for a in ana_a),
        "cyrillic_chars": sum(a["cyrillic"] for a in ana_a),
        "foreign_word_occurrences": sum(len(a["foreign_words"]) for a in ana_a),
        "unique_foreign_words": sorted(foreign_a),
        "cascade_count": cascade_true,
    },
    "False": {
        "total_segments": result_b["total_segments"],
        "mixed_script_segments": sum(1 for a in ana_b if a["latin"] > 0 or a["cyrillic"] > 0),
        "latin_chars": sum(a["latin"] for a in ana_b),
        "cyrillic_chars": sum(a["cyrillic"] for a in ana_b),
        "foreign_word_occurrences": sum(len(a["foreign_words"]) for a in ana_b),
        "unique_foreign_words": sorted(foreign_b),
        "cascade_count": cascade_false,
    },
    "foreign_in_both": sorted(foreign_a & foreign_b),
    "foreign_only_True": sorted(foreign_a - foreign_b),
    "foreign_only_False": sorted(foreign_b - foreign_a),
}

comp_path = os.path.join(project_root, "output", "ab_test_comparison.json")
with open(comp_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)

print(f"\nComparison saved: {comp_path}")
print(f"\n{'=' * 60}")
print(f"TEST COMPLETE")
print(f"{'=' * 60}")
