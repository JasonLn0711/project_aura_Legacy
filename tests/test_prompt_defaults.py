import unittest

from aura.asr.file_pipeline import build_transcribe_kwargs, resolve_initial_prompt
from aura.asr.threads import FileTranscriberThread, TranscriberThread
from aura.config import DEFAULT_LIVE_PROMPT, DEFAULT_PROMPT


class PromptDefaultTests(unittest.TestCase):
    def test_file_thread_uses_default_prompt_when_not_supplied(self):
        thread = FileTranscriberThread(model=object(), file_path="input.wav")

        self.assertEqual(thread.initial_prompt, DEFAULT_PROMPT)
        self.assertFalse(thread.enable_denoise)

    def test_live_thread_starts_with_live_default_prompt(self):
        thread = TranscriberThread()

        self.assertEqual(thread.live_initial_prompt, DEFAULT_LIVE_PROMPT)

    def test_live_update_without_prompt_keeps_live_default(self):
        thread = TranscriberThread()

        thread.update_live_settings()

        self.assertEqual(thread.live_initial_prompt, DEFAULT_LIVE_PROMPT)

    def test_explicit_empty_prompt_remains_empty(self):
        self.assertEqual(resolve_initial_prompt("", DEFAULT_PROMPT), "")

    def test_transcribe_kwargs_include_default_prompt(self):
        kwargs = build_transcribe_kwargs(
            beam_size=5,
            language="zh",
            initial_prompt=DEFAULT_PROMPT,
            condition_on_previous_text=True,
        )

        self.assertEqual(kwargs["initial_prompt"], DEFAULT_PROMPT)
        self.assertEqual(kwargs["language"], "zh")
        self.assertEqual(kwargs["beam_size"], 5)
        self.assertTrue(kwargs["condition_on_previous_text"])

    def test_transcribe_kwargs_omit_language_for_auto_detect(self):
        kwargs = build_transcribe_kwargs(
            beam_size=5,
            language=None,
            initial_prompt=DEFAULT_PROMPT,
            condition_on_previous_text=True,
        )

        self.assertNotIn("language", kwargs)
        self.assertEqual(kwargs["initial_prompt"], DEFAULT_PROMPT)


if __name__ == "__main__":
    unittest.main()
