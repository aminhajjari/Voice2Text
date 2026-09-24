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


    def test_audio_formats_includes_aac_and_opus(self):
        from src.config import AUDIO_FORMATS
        self.assertIn(".aac", AUDIO_FORMATS)
        self.assertIn(".opus", AUDIO_FORMATS)
        self.assertIn(".wav", AUDIO_FORMATS)
        self.assertIn(".mp3", AUDIO_FORMATS)


    def test_spell_correction_configuration_flow(self):
        from src.config import resolve_runtime_settings
        # Default is False
        settings = resolve_runtime_settings()
        self.assertFalse(settings["enable_spell_correction"])

        # Override via parameter
        settings_enabled = resolve_runtime_settings(enable_spell_correction=True)
        self.assertTrue(settings_enabled["enable_spell_correction"])

    def test_user_selected_model_is_not_silently_downgraded(self):
        # When a model name is explicitly requested that doesn't exist locally,
        # it must return the requested model identifier (so faster-whisper can load/download or fail)
        # and NOT silently fall back to "medium" or "base".
        resolved = resolve_model_path(model_name="large-v3")
        self.assertEqual(resolved, "large-v3")

        resolved_custom = resolve_model_path(model_name="my-custom-model")
        self.assertEqual(resolved_custom, "my-custom-model")

    def test_default_model_used_when_none_specified(self):
        # When no model is specified, it should resolve to DEFAULT_MODEL_PATH
        resolved = resolve_model_path()
        self.assertIn("faster-whisper-medium", resolved)


if __name__ == "__main__":
    unittest.main()
