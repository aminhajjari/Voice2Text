import logging
import shutil
import subprocess
import threading
from pathlib import Path

from .config import AUDIO_FORMATS, INPUT_DIR, LOG_DIR, LOG_FILE, OUTPUT_DIR


def ensure_directories():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _configure_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("transcribe_app")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


LOGGER = _configure_logger()


def write_log(message, level="INFO"):
    if level.upper() == "INFO":
        LOGGER.info(message)
    elif level.upper() == "ERROR":
        LOGGER.error(message)
    elif level.upper() == "WARNING":
        LOGGER.warning(message)
    else:
        LOGGER.info(message)


def detect_device():
    try:
        import ctranslate2
    except Exception:
        return "cpu"
    # handle different ctranslate2 versions/signatures safely
    try:
        if hasattr(ctranslate2, "get_cuda_device_count"):
            cuda_count = int(ctranslate2.get_cuda_device_count() or 0)
        else:
            # some versions expose get_device_count() with no args
            try:
                cuda_count = int(ctranslate2.get_device_count("cuda") or 0)
            except TypeError:
                try:
                    cuda_count = int(ctranslate2.get_device_count() or 0)
                except Exception:
                    cuda_count = 0
    except Exception:
        cuda_count = 0

    return "cuda" if cuda_count > 0 else "cpu"


def scan_audio_files(input_dir=None):
    if input_dir is None:
        input_dir = INPUT_DIR

    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []

    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_FORMATS
    ]


def sanitize_output_filename(filename):
    """Remove/replace characters invalid in Windows/POSIX filenames."""
    # Invalid characters in Windows: < > : " / \ | ? *
    invalid_chars = '<>:"|?*\\'
    sanitized = filename
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    # Avoid null character and control characters
    sanitized = ''.join(c if ord(c) >= 32 else '_' for c in sanitized)
    return sanitized.strip()


def validate_output_path(output_path, base_dir):
    """Validate that output_path is within base_dir (no path traversal)."""
    output_path = Path(output_path).resolve()
    base_dir = Path(base_dir).resolve()
    try:
        output_path.relative_to(base_dir)
        return True
    except ValueError:
        return False


def format_timestamp(seconds):
    # normalize and avoid rounding that yields 1000 milliseconds
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    fraction = seconds - int(seconds)
    millis = int(fraction * 1000)  # floor to avoid 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_cuda_status():
    cuda_available = False
    ctranslate2_cuda_device_count = 0
    gpu_name = "unavailable"

    try:
        import ctranslate2

        if hasattr(ctranslate2, "get_cuda_device_count"):
            ctranslate2_cuda_device_count = int(ctranslate2.get_cuda_device_count() or 0)
        else:
            try:
                ctranslate2_cuda_device_count = int(ctranslate2.get_device_count("cuda") or 0)
            except TypeError:
                try:
                    ctranslate2_cuda_device_count = int(ctranslate2.get_device_count() or 0)
                except Exception:
                    ctranslate2_cuda_device_count = 0

        cuda_available = ctranslate2_cuda_device_count > 0
    except Exception:
        ctranslate2_cuda_device_count = 0
        cuda_available = False

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass

    return {
        "cuda_available": cuda_available,
        "ctranslate2_cuda_device_count": ctranslate2_cuda_device_count,
        "gpu_name": gpu_name,
    }


class GPUMonitor:
    def __init__(self):
        self._stop_event = threading.Event()
        self._stats = {"utilization": "unavailable", "memory": "unavailable"}
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def get_stats(self):
        return self._stats.copy()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                if shutil.which("nvidia-smi"):
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        line = result.stdout.strip().splitlines()[0]
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) >= 2:
                            self._stats = {"utilization": parts[0], "memory": parts[1]}
                            self._stop_event.wait(1.0)
                            continue
            except Exception:
                pass

            self._stats = {"utilization": "unavailable", "memory": "unavailable"}
            self._stop_event.wait(1.0)