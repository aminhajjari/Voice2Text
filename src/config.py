import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "transcription.log"

AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4"]
OUTPUT_FORMATS = ["txt", "docx", "srt", "json"]

VALID_PRESETS = {"default", "safe"}
VALID_MIXED_SCRIPT_MODES = {"off", "flag", "reject"}


def _getenv_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_optional_text(value):
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_preset_name(value):
    preset_name = (_normalize_optional_text(value) or "default").lower()
    if preset_name not in VALID_PRESETS:
        raise ValueError(
            f"Unsupported preset '{preset_name}'. Expected one of: {sorted(VALID_PRESETS)}"
        )
    return preset_name


def normalize_mixed_script_mode(value):
    mode = (_normalize_optional_text(value) or "off").lower()
    if mode not in VALID_MIXED_SCRIPT_MODES:
        raise ValueError(
            "Unsupported mixed-script mode "
            f"'{mode}'. Expected one of: {sorted(VALID_MIXED_SCRIPT_MODES)}"
        )
    return mode


DEFAULT_PRESET = normalize_preset_name(os.getenv("WHISPER_PRESET", "default"))
DEFAULT_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "medium")
DEFAULT_DEVICE = (_normalize_optional_text(os.getenv("WHISPER_DEVICE", "auto")) or "auto").lower()
DEFAULT_COMPUTE_TYPE = _normalize_optional_text(os.getenv("WHISPER_COMPUTE_TYPE"))

DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "fa")
DEFAULT_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
DEFAULT_VAD_FILTER = _getenv_bool("WHISPER_VAD_FILTER", True)
DEFAULT_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
DEFAULT_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))
DEFAULT_LOGPROB_THRESHOLD = float(os.getenv("WHISPER_LOGPROB_THRESHOLD", "-1.0"))
DEFAULT_COMPRESSION_RATIO_THRESHOLD = float(
    os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")
)
DEFAULT_INITIAL_PROMPT = _normalize_optional_text(os.getenv("WHISPER_INITIAL_PROMPT"))
DEFAULT_PREFIX = _normalize_optional_text(os.getenv("WHISPER_PREFIX"))
DEFAULT_HOTWORDS = _normalize_optional_text(os.getenv("WHISPER_HOTWORDS"))
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = _getenv_bool(
    "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False
)
DEFAULT_MIXED_SCRIPT_MODE = normalize_mixed_script_mode(
    os.getenv("WHISPER_MIXED_SCRIPT_MODE", "off")
)

ENABLE_SPELL_CORRECTION = _getenv_bool("ENABLE_SPELL_CORRECTION", False)

PRESET_OVERRIDES = {
    "default": {},
    "safe": {
        "model_name": "base",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 3,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "mixed_script_mode": "flag",
    },
}


def _candidate_model_names(model_name=None):
    names = []
    if model_name:
        names.append(model_name)
    if model_name != DEFAULT_MODEL_NAME:
        names.append(DEFAULT_MODEL_NAME)
    names.extend(["medium", "base"])

    seen = set()
    for raw_name in names:
        if not raw_name:
            continue
        normalized = str(raw_name).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.startswith("faster-whisper-"):
            yield normalized
            # Also yield without prefix so both naming styles are checked
            yield normalized[len("faster-whisper-"):]
        else:
            # Yield the exact folder name first, then the prefixed variant
            yield normalized
            yield f"faster-whisper-{normalized}"


def resolve_model_path(model_path=None, model_name=None):
    if model_path:
        provided_path = Path(str(model_path)).expanduser()
        if not provided_path.is_absolute():
            provided_path = (BASE_DIR / provided_path).resolve()
        return str(provided_path)

    for candidate_name in _candidate_model_names(model_name):
        candidate_path = Path(candidate_name)
        if candidate_path.is_absolute():
            if candidate_path.exists():
                return str(candidate_path)
            continue

        resolved_path = (BASE_DIR / "models" / candidate_name).resolve()
        if resolved_path.exists():
            return str(resolved_path)

    default_name = (
        DEFAULT_MODEL_NAME
        if str(DEFAULT_MODEL_NAME).startswith("faster-whisper-")
        else f"faster-whisper-{DEFAULT_MODEL_NAME}"
    )
    return str((BASE_DIR / "models" / default_name).resolve())


DEFAULT_MODEL_PATH = resolve_model_path()


def resolve_runtime_settings(
    *,
    preset=None,
    model_name=None,
    device=None,
    compute_type=None,
    language=None,
    beam_size=None,
    vad_filter=None,
    temperature=None,
    no_speech_threshold=None,
    logprob_threshold=None,
    compression_ratio_threshold=None,
    initial_prompt=None,
    prefix=None,
    hotwords=None,
    condition_on_previous_text=None,
    mixed_script_mode=None,
):
    preset_name = normalize_preset_name(preset or DEFAULT_PRESET)
    preset_values = PRESET_OVERRIDES.get(preset_name, {})

    resolved_model_name = (
        _normalize_optional_text(model_name)
        or preset_values.get("model_name")
        or DEFAULT_MODEL_NAME
    )
    resolved_device = (
        (_normalize_optional_text(device) or preset_values.get("device") or DEFAULT_DEVICE or "auto")
        .lower()
    )
    resolved_compute_type = (
        _normalize_optional_text(compute_type)
        or preset_values.get("compute_type")
        or DEFAULT_COMPUTE_TYPE
    )
    resolved_language = _normalize_optional_text(language) or DEFAULT_LANGUAGE
    resolved_beam_size = int(
        beam_size
        if beam_size is not None
        else preset_values.get("beam_size", DEFAULT_BEAM_SIZE)
    )
    resolved_vad_filter = (
        bool(vad_filter)
        if vad_filter is not None
        else bool(preset_values.get("vad_filter", DEFAULT_VAD_FILTER))
    )
    resolved_temperature = float(
        temperature if temperature is not None else DEFAULT_TEMPERATURE
    )
    resolved_no_speech_threshold = float(
        no_speech_threshold
        if no_speech_threshold is not None
        else DEFAULT_NO_SPEECH_THRESHOLD
    )
    resolved_logprob_threshold = float(
        logprob_threshold
        if logprob_threshold is not None
        else DEFAULT_LOGPROB_THRESHOLD
    )
    resolved_compression_ratio_threshold = float(
        compression_ratio_threshold
        if compression_ratio_threshold is not None
        else DEFAULT_COMPRESSION_RATIO_THRESHOLD
    )
    resolved_initial_prompt = (
        _normalize_optional_text(initial_prompt)
        or preset_values.get("initial_prompt")
        or DEFAULT_INITIAL_PROMPT
    )
    resolved_prefix = (
        _normalize_optional_text(prefix)
        or preset_values.get("prefix")
        or DEFAULT_PREFIX
    )
    resolved_hotwords = (
        _normalize_optional_text(hotwords)
        or preset_values.get("hotwords")
        or DEFAULT_HOTWORDS
    )
    resolved_condition_on_previous_text = (
        bool(condition_on_previous_text)
        if condition_on_previous_text is not None
        else bool(
            preset_values.get(
                "condition_on_previous_text",
                DEFAULT_CONDITION_ON_PREVIOUS_TEXT,
            )
        )
    )
    resolved_mixed_script_mode = normalize_mixed_script_mode(
        mixed_script_mode
        or preset_values.get("mixed_script_mode")
        or DEFAULT_MIXED_SCRIPT_MODE
    )
    resolved_model_path = resolve_model_path(model_name=resolved_model_name)

    return {
        "preset": preset_name,
        "model_name": resolved_model_name,
        "model_path": resolved_model_path,
        "device": resolved_device,
        "compute_type": resolved_compute_type,
        "language": resolved_language,
        "beam_size": resolved_beam_size,
        "vad_filter": resolved_vad_filter,
        "temperature": resolved_temperature,
        "no_speech_threshold": resolved_no_speech_threshold,
        "logprob_threshold": resolved_logprob_threshold,
        "compression_ratio_threshold": resolved_compression_ratio_threshold,
        "initial_prompt": resolved_initial_prompt,
        "prefix": resolved_prefix,
        "hotwords": resolved_hotwords,
        "condition_on_previous_text": resolved_condition_on_previous_text,
        "mixed_script_mode": resolved_mixed_script_mode,
    }
