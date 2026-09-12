import unittest
from pathlib import Path

from src.config import DEFAULT_MODEL_PATH, resolve_model_path


class TestConfig(unittest.TestCase):
    def test_default_model_path_points_to_local_model_directory(self):
        self.assertIsInstance(DEFAULT_MODEL_PATH, str)
        self.assertTrue(Path(DEFAULT_MODEL_PATH).name)
        self.assertTrue(Path(DEFAULT_MODEL_PATH).exists() or "models" in str(DEFAULT_MODEL_PATH))

    def test_resolve_model_path_distil_large_v3(self):
        resolved = resolve_model_path(model_name="distil-large-v3")
        resolved_path = Path(resolved)
        self.assertEqual(resolved_path.name, "distil-large-v3")
        self.assertTrue(resolved_path.exists())

    def test_resolve_model_path_faster_whisper_medium(self):
        resolved = resolve_model_path(model_name="medium")
        resolved_path = Path(resolved)
        self.assertEqual(resolved_path.name, "faster-whisper-medium")
        self.assertTrue(resolved_path.exists())


if __name__ == "__main__":
    unittest.main()
