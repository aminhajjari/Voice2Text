import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.transcriber import transcribe_audio


class TestGpuFallback(unittest.TestCase):
    def test_transcribe_audio_retries_on_cpu_when_gpu_oom(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"fake audio")
            temp_path = Path(handle.name)

        class DummyModel:
            def transcribe(self, *args, **kwargs):
                raise RuntimeError("CUDA failed with error out of memory")

        class DummyMonitor:
            def start(self):
                return None

            def stop(self):
                return None

            def get_stats(self):
                return {}

        fallback_model = type(
            "FallbackModel",
            (),
            {
                "transcribe": lambda self, *args, **kwargs: (
                    [],
                    type("Info", (), {"duration": 1.0})(),
                )
            },
        )()

        with patch("src.transcriber.GPUMonitor", return_value=DummyMonitor()), patch(
            "src.transcriber.load_model", return_value=(fallback_model, "int8", "cpu")
        ) as mock_load_model:
            result = transcribe_audio(
                temp_path,
                DummyModel(),
                device="cuda",
                model_path="models/faster-whisper-medium",
            )

        self.assertEqual(result["text"], "")
        self.assertEqual(mock_load_model.call_count, 1)
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
