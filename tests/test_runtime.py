import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.config as config_module
import transcribe
from src.transcriber import transcribe_audio


class TestRuntimeSettings(unittest.TestCase):
    def test_safe_preset_exposes_safer_defaults(self):
        settings = config_module.resolve_runtime_settings(preset="safe")

        self.assertEqual(settings["preset"], "safe")
        self.assertEqual(settings["model_name"], "base")
        self.assertEqual(settings["device"], "cpu")
        self.assertEqual(settings["compute_type"], "int8")
        self.assertEqual(settings["beam_size"], 3)
        self.assertTrue(settings["vad_filter"])
        self.assertFalse(settings["condition_on_previous_text"])
        self.assertEqual(settings["temperature"], 0.0)
        self.assertEqual(settings["no_speech_threshold"], 0.6)
        self.assertEqual(settings["logprob_threshold"], -1.0)
        self.assertEqual(settings["compression_ratio_threshold"], 2.4)
        self.assertEqual(settings["mixed_script_mode"], "flag")

    def test_environment_variables_flow_into_runtime_settings(self):
        with patch.dict(
            os.environ,
            {
                "WHISPER_DEVICE": "cpu",
                "WHISPER_COMPUTE_TYPE": "int8",
                "WHISPER_HOTWORDS": "ایران, جمهوری اسلامی",
            },
            clear=False,
        ):
            reloaded = importlib.reload(config_module)
            settings = reloaded.resolve_runtime_settings()

        self.assertEqual(settings["device"], "cpu")
        self.assertEqual(settings["compute_type"], "int8")
        self.assertEqual(settings["hotwords"], "ایران, جمهوری اسلامی")
        importlib.reload(config_module)


class TestCliFlow(unittest.TestCase):
    def test_main_passes_effective_model_device_to_transcriber(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"fake audio")
            temp_path = Path(handle.name)

        with patch("transcribe.ensure_directories"), patch(
            "transcribe.get_cuda_status",
            return_value={
                "cuda_available": False,
                "ctranslate2_cuda_device_count": 0,
                "gpu_name": "unavailable",
            },
        ), patch("transcribe.load_model", return_value=(object(), "int8", "cpu")), patch(
            "transcribe.transcribe_audio",
            return_value={"text": "ok", "segments": [], "warnings": []},
        ) as mock_transcribe, patch("transcribe.outputs.save_txt"), patch(
            "transcribe.outputs.save_docx"
        ), patch("transcribe.outputs.save_srt"), patch("transcribe.outputs.save_json"):
            exit_code = transcribe.main([str(temp_path)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_transcribe.call_args.kwargs["device"], "cpu")
        temp_path.unlink(missing_ok=True)

    def test_main_returns_non_zero_when_all_explicit_paths_invalid(self):
        with patch("transcribe.ensure_directories"), patch("transcribe.load_model") as mock_load:
            exit_code = transcribe.main(["missing.wav", "notes.txt"])

        self.assertEqual(exit_code, 1)
        mock_load.assert_not_called()

    def test_main_supports_custom_output_dir(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"fake audio")
            temp_path = Path(handle.name)

        with tempfile.TemporaryDirectory() as custom_out_dir:
            custom_out_path = Path(custom_out_dir)
            captured_paths = []

            def fake_save_txt(result, path):
                captured_paths.append(Path(path))

            with patch("transcribe.ensure_directories"), patch(
                "transcribe.get_cuda_status",
                return_value={
                    "cuda_available": False,
                    "ctranslate2_cuda_device_count": 0,
                    "gpu_name": "unavailable",
                },
            ), patch("transcribe.load_model", return_value=(object(), "int8", "cpu")), patch(
                "transcribe.transcribe_audio",
                return_value={"text": "ok", "segments": [], "warnings": []},
            ), patch("transcribe.outputs.save_txt", side_effect=fake_save_txt), patch(
                "transcribe.outputs.save_docx"
            ), patch("transcribe.outputs.save_srt"), patch("transcribe.outputs.save_json"):
                exit_code = transcribe.main([str(temp_path), "--output-dir", str(custom_out_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(len(captured_paths) > 0)
            self.assertEqual(captured_paths[0].parent, custom_out_path.resolve())

        temp_path.unlink(missing_ok=True)


class TestMixedScriptHandling(unittest.TestCase):
    def test_transcribe_audio_flags_mixed_script_segments(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"fake audio")
            temp_path = Path(handle.name)

        class DummySegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        class DummyModel:
            def transcribe(self, *args, **kwargs):
                return (
                    [DummySegment(0.0, 1.0, "سلام hello")],
                    type("Info", (), {"duration": 1.0})(),
                )

        class DummyMonitor:
            def start(self):
                return None

            def stop(self):
                return None

            def get_stats(self):
                return {}

        with patch("src.transcriber.GPUMonitor", return_value=DummyMonitor()):
            result = transcribe_audio(
                temp_path,
                DummyModel(),
                device="cpu",
                language="fa",
                mixed_script_mode="flag",
                condition_on_previous_text=False,
            )

        self.assertTrue(result["segments"][0]["mixed_script"])
        self.assertTrue(result["warnings"])
        temp_path.unlink(missing_ok=True)

    def test_transcribe_audio_rejects_mixed_script_segments(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"fake audio")
            temp_path = Path(handle.name)

        class DummySegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        class DummyModel:
            def transcribe(self, *args, **kwargs):
                return (
                    [DummySegment(0.0, 1.0, "سلام hello")],
                    type("Info", (), {"duration": 1.0})(),
                )

        class DummyMonitor:
            def start(self):
                return None

            def stop(self):
                return None

            def get_stats(self):
                return {}

        with patch("src.transcriber.GPUMonitor", return_value=DummyMonitor()):
            with self.assertRaises(RuntimeError):
                transcribe_audio(
                    temp_path,
                    DummyModel(),
                    device="cpu",
                    language="fa",
                    mixed_script_mode="reject",
                    condition_on_previous_text=False,
                )

        temp_path.unlink(missing_ok=True)


class TestFilenameSanitization(unittest.TestCase):
    def test_sanitize_output_filename_preserves_valid_names(self):
        from src.utils import sanitize_output_filename
        self.assertEqual(sanitize_output_filename("audio"), "audio")
        self.assertEqual(sanitize_output_filename("test_audio"), "test_audio")
        self.assertEqual(sanitize_output_filename("test audio"), "test audio")
        self.assertEqual(sanitize_output_filename("meeting-01"), "meeting-01")
        self.assertEqual(sanitize_output_filename("meeting.test"), "meeting.test")
        self.assertEqual(sanitize_output_filename("جلسه"), "جلسه")
        self.assertEqual(sanitize_output_filename("جلسه 1"), "جلسه 1")
        self.assertEqual(sanitize_output_filename("audio (final)"), "audio (final)")

    def test_sanitize_output_filename_fallback_for_empty_stems(self):
        from src.utils import sanitize_output_filename
        self.assertEqual(sanitize_output_filename(""), "transcript")
        self.assertEqual(sanitize_output_filename("..."), "transcript")
        self.assertEqual(sanitize_output_filename("   "), "transcript")
        self.assertEqual(sanitize_output_filename(":::"), "___")
        self.assertEqual(sanitize_output_filename(". . ."), "transcript")


class TestSpellCorrection(unittest.TestCase):
    def test_correct_persian_spelling_variants(self):
        from src.utils import correct_persian_spelling
        # Arabic to Persian normalization
        self.assertEqual(correct_persian_spelling("كتاب يادگيري"), "کتاب یادگیری")
        # Affixes and ZWNJ
        self.assertEqual(correct_persian_spelling("مي رود"), "می‌رود")
        self.assertEqual(correct_persian_spelling("نمي دانم"), "نمی‌دانم")
        self.assertEqual(correct_persian_spelling("صدا ها"), "صدا‌ها")
        self.assertEqual(correct_persian_spelling("بزرگ ترين"), "بزرگ‌ترین")
        self.assertEqual(correct_persian_spelling("خانه اش"), "خانه‌اش")
        # Deduplication of repeated character hallucinations
        self.assertEqual(correct_persian_spelling("تتتتتست"), "تتست")

    def test_persian_and_arabic_indic_numbers_preserved(self):
        from src.utils import correct_persian_spelling
        # Mandatory: Persian numbers must not be truncated
        self.assertEqual(correct_persian_spelling("مبلغ ۱۰۰۰ تومان"), "مبلغ ۱۰۰۰ تومان")
        self.assertEqual(correct_persian_spelling("مبلغ ۱۰۰۰۰۰۰ ریال"), "مبلغ ۱۰۰۰۰۰۰ ریال")
        self.assertEqual(correct_persian_spelling("شماره ۰۹۱۲۰۰۰۱۲۳۴"), "شماره ۰۹۱۲۰۰۰۱۲۳۴")
        self.assertEqual(correct_persian_spelling("سال ۱۴۰۴"), "سال ۱۴۰۴")
        # Arabic-Indic digits
        self.assertEqual(correct_persian_spelling("رقم ١٠٠٠"), "رقم ١٠٠٠")

    def test_punctuation_preserved(self):
        from src.utils import correct_persian_spelling
        self.assertEqual(correct_persian_spelling("،،،"), "،،،")
        self.assertEqual(correct_persian_spelling("؟؟؟"), "؟؟؟")
        self.assertEqual(correct_persian_spelling("!!!"), "!!!")
        self.assertEqual(correct_persian_spelling("..."), "...")

    def test_english_french_german_preserved(self):
        from src.utils import correct_persian_spelling
        # English
        self.assertEqual(
            correct_persian_spelling("Hello everyone OpenAI Meeting Management Software Engineering"),
            "Hello everyone OpenAI Meeting Management Software Engineering",
        )
        # French
        self.assertEqual(
            correct_persian_spelling("Bonjour Merci beaucoup Paris Université"),
            "Bonjour Merci beaucoup Paris Université",
        )
        # German
        self.assertEqual(
            correct_persian_spelling("Guten Morgen Danke Universität Berlin"),
            "Guten Morgen Danke Universität Berlin",
        )

    def test_arabic_text_preservation_and_normalization(self):
        from src.utils import correct_persian_spelling
        # Legitimate Arabic: characters are not deleted or corrupted; Persian orthography applied intentionally
        self.assertEqual(correct_persian_spelling("مرحبا"), "مرحبا")
        self.assertEqual(correct_persian_spelling("جامعة"), "جامعه")

    def test_mixed_language_transcript(self):
        from src.utils import correct_persian_spelling
        text = "سلام everyone، امروز درباره OpenAI و Universität Berlin صحبت می‌کنیم."
        expected = "سلام everyone، امروز درباره OpenAI و Universität Berlin صحبت می‌کنیم."
        self.assertEqual(correct_persian_spelling(text), expected)

    def test_persian_spell_correction_preserved(self):
        from src.utils import correct_persian_spelling
        self.assertEqual(correct_persian_spelling("می رود"), "می‌رود")
        self.assertEqual(correct_persian_spelling("نمی دانم"), "نمی‌دانم")
        self.assertEqual(correct_persian_spelling("کتاب ها"), "کتاب‌ها")
        self.assertEqual(correct_persian_spelling("بزرگ تر"), "بزرگ‌تر")
        self.assertEqual(correct_persian_spelling("خانه ام"), "خانه‌ام")
        self.assertEqual(correct_persian_spelling("دسته اش"), "دسته‌اش")

    def test_idempotence(self):
        from src.utils import correct_persian_spelling
        samples = [
            "سلام everyone، امروز درباره OpenAI و Universität Berlin صحبت می‌کنیم.",
            "مبلغ ۱۰۰۰ تومان و ۱۰۰۰۰۰۰ ریال",
            "کتاب های آموزشی می روند",
            "Hello everyone, testing 123",
            "Bonjour Merci beaucoup Paris",
        ]
        for s in samples:
            self.assertEqual(correct_persian_spelling(correct_persian_spelling(s)), correct_persian_spelling(s))


if __name__ == "__main__":
    unittest.main()
