import unittest

from aura.config import (
    COMPUTE_TYPE,
    DEFAULT_LIVE_PROMPT,
    DEFAULT_PROMPT,
    DEVICE,
    DIARIZATION_MODEL_ID,
    MODEL_ID,
    SUMMARY_MODEL_ID,
)
from aura.settings import DEFAULT_SETTINGS, AppSettings
from aura.ui.messages import UIStrings


class SettingsTests(unittest.TestCase):
    def test_default_settings_preserve_runtime_constants(self):
        self.assertEqual(DEFAULT_SETTINGS.model_id, MODEL_ID)
        self.assertEqual(DEFAULT_SETTINGS.device, DEVICE)
        self.assertEqual(DEFAULT_SETTINGS.compute_type, COMPUTE_TYPE)
        self.assertEqual(DEFAULT_SETTINGS.device, "cuda")
        self.assertEqual(DEFAULT_SETTINGS.compute_type, "int8")
        self.assertEqual(DEFAULT_SETTINGS.file_initial_prompt, DEFAULT_PROMPT)
        self.assertEqual(DEFAULT_SETTINGS.live_initial_prompt, DEFAULT_LIVE_PROMPT)
        self.assertEqual(DEFAULT_SETTINGS.denoise_preset, "off")
        self.assertFalse(DEFAULT_SETTINGS.speaker_diarization_enabled)
        self.assertEqual(DEFAULT_SETTINGS.speaker_min_speakers, 2)
        self.assertEqual(DEFAULT_SETTINGS.speaker_max_speakers, 6)
        self.assertEqual(DEFAULT_SETTINGS.speaker_diarization_model, DIARIZATION_MODEL_ID)
        self.assertFalse(DEFAULT_SETTINGS.llm_summary_enabled)
        self.assertEqual(DEFAULT_SETTINGS.llm_summary_model, SUMMARY_MODEL_ID)
        self.assertEqual(DEFAULT_SETTINGS.llm_summary_quantization, "int8")

    def test_custom_settings_can_override_runtime_defaults(self):
        settings = AppSettings(device="cpu", compute_type="int8", language=None, target_dbfs=-18.0)

        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.compute_type, "int8")
        self.assertIsNone(settings.language)
        self.assertEqual(settings.target_dbfs, -18.0)

    def test_ui_strings_format_dynamic_status_messages(self):
        strings = UIStrings()

        self.assertEqual(strings.model_ready("cpu", "int8"), "✅ Model is ready (cpu/int8)")
        self.assertIn("v1.5.0", strings.update_found("1.5.0"))
        self.assertEqual(
            strings.splitter_status("meeting.wav", "/tmp/out"),
            "Source: meeting.wav | Output to: /tmp/out",
        )


if __name__ == "__main__":
    unittest.main()
