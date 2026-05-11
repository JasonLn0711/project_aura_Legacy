import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aura.asr.threads import FileTranscriberThread
from aura.system import runtime_paths


class RuntimePathTests(unittest.TestCase):
    def test_runtime_paths_use_configured_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {runtime_paths.RUNTIME_DIR_ENV: tmpdir}):
                self.assertEqual(runtime_paths.runtime_dir(), Path(tmpdir))
                self.assertEqual(runtime_paths.transcript_backup_path(), Path(tmpdir) / "temp_transcript.txt")
                self.assertEqual(runtime_paths.temp_normalized_path("abc"), Path(tmpdir) / "normalized_abc.wav")

    def test_append_and_remove_transcript_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {runtime_paths.RUNTIME_DIR_ENV: tmpdir}):
                runtime_paths.append_transcript_backup("[00:00:01] hello")

                backup = runtime_paths.transcript_backup_path()
                self.assertEqual(backup.read_text(encoding="utf-8"), "[00:00:01] hello\n")

                runtime_paths.remove_transcript_backup()
                self.assertFalse(backup.exists())

    def test_file_transcriber_cancellation_flag(self):
        thread = FileTranscriberThread(model=object(), file_path="input.wav")

        thread.request_cancel()

        self.assertTrue(thread.cancel_requested)
        with self.assertRaises(RuntimeError):
            thread._raise_if_cancelled()


if __name__ == "__main__":
    unittest.main()
