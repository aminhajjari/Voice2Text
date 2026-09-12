import unittest
from pathlib import Path
from unittest.mock import patch

import src.transcriber as transcriber
from src.transcriber import transcribe_audio


class TestTranscriber(unittest.TestCase):
    def test_transcribe_audio_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            transcribe_audio("path/to/invalid/audio/file.wav", model=None)

    def test_transcribe_audio_pathlike_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            transcribe_audio(Path("path/to/invalid/audio/file.wav"), model=None)

    def test_build_compute_type_candidates_prefers_cuda_fallbacks(self):
        candidates = transcriber._build_compute_type_candidates("cuda")
        self.assertEqual(candidates, ["float16", "int8_float16", "float32"])

    def test_transcribe_audio_passes_hallucination_controls(self):
        class DummyModel:
            def transcribe(self, *args, **kwargs):
                self.kwargs = kwargs
                return [], type("Info", (), {"duration": 1.0})()

        class DummyMonitor:
            def start(self):
                return None

            def stop(self):
                return None

            def get_stats(self):
                return {}

        with patch("src.transcriber.GPUMonitor", return_value=DummyMonitor()):
            model = DummyModel()
            temp_path = Path("tests/decoder-controls.wav")
            temp_path.write_bytes(b"fake audio")
            try:
                transcribe_audio(temp_path, model, device="cpu", language="fa")
            finally:
                temp_path.unlink(missing_ok=True)

        self.assertEqual(model.kwargs["language"], "fa")
        self.assertEqual(model.kwargs["temperature"], 0.0)
        self.assertEqual(model.kwargs["no_speech_threshold"], 0.6)
        self.assertEqual(model.kwargs["log_prob_threshold"], -1.0)
        self.assertEqual(model.kwargs["compression_ratio_threshold"], 2.4)
        self.assertFalse(model.kwargs["condition_on_previous_text"])

    def test_transcribe_audio_progress_callback(self):
        class DummySegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        class DummyModel:
            def transcribe(self, *args, **kwargs):
                return (
                    [DummySegment(0.0, 1.5, "سلام"), DummySegment(1.5, 3.0, "دنیا")],
                    type("Info", (), {"duration": 3.0})(),
                )

        class DummyMonitor:
            def start(self):
                return None

            def stop(self):
                return None

            def get_stats(self):
                return {"utilization": "15", "memory": "1200"}

        events = []

        def callback(event):
            events.append(event)

        with patch("src.transcriber.GPUMonitor", return_value=DummyMonitor()):
            model = DummyModel()
            temp_path = Path("tests/progress-callback.wav")
            temp_path.write_bytes(b"fake audio")
            try:
                result = transcribe_audio(
                    temp_path,
                    model,
                    device="cpu",
                    language="fa",
                    on_progress=callback,
                )
            finally:
                temp_path.unlink(missing_ok=True)

        self.assertEqual(len(events), 2)
        # Verify first event
        self.assertIn("percent", events[0])
        self.assertIn("processed_duration", events[0])
        self.assertIn("total_duration", events[0])
        self.assertIn("speed", events[0])
        self.assertIn("elapsed", events[0])
        self.assertIn("segments_count", events[0])
        self.assertIn("vram_mb", events[0])
        self.assertIn("gpu_util", events[0])
        self.assertIn("current_text", events[0])

        self.assertEqual(events[0]["segments_count"], 1)
        self.assertEqual(events[0]["current_text"], "سلام")
        self.assertEqual(events[0]["vram_mb"], "1200")
        self.assertEqual(events[0]["gpu_util"], "15")

        self.assertEqual(events[1]["segments_count"], 2)
        self.assertEqual(events[1]["current_text"], "دنیا")
        self.assertAlmostEqual(events[1]["percent"], 100.0)

        # Result dictionary keys must remain identical
        self.assertIn("text", result)
        self.assertIn("segments", result)
        self.assertIn("filename", result)
        self.assertIn("duration", result)
        self.assertEqual(result["text"], "سلام\nدنیا")

    def test_transcribe_audio_progress_callback_error_does_not_abort(self):
        class DummySegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        class DummyModel:
            def transcribe(self, *args, **kwargs):
                return (
                    [DummySegment(0.0, 1.0, "تست")],
                    type("Info", (), {"duration": 1.0})(),
                )

        class DummyMonitor:
            def start(self):
                return None

            def stop(self):
                return None

            def get_stats(self):
                return {}

        def faulty_callback(event):
            raise RuntimeError("UI crashed")

        with patch("src.transcriber.GPUMonitor", return_value=DummyMonitor()):
            model = DummyModel()
            temp_path = Path("tests/progress-callback-err.wav")
            temp_path.write_bytes(b"fake audio")
            try:
                # Callback error is logged as warning and does not corrupt or abort transcription
                result = transcribe_audio(
                    temp_path,
                    model,
                    device="cpu",
                    language="fa",
                    on_progress=faulty_callback,
                )
            finally:
                temp_path.unlink(missing_ok=True)

        self.assertEqual(result["text"], "تست")


if __name__ == "__main__":
    unittest.main()