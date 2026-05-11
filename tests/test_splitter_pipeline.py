import tempfile
import unittest
from pathlib import Path

from pydub import AudioSegment

from aura.audio.splitter_pipeline import (
    SplitterSettings,
    choose_cut_point,
    export_format_for_ext,
    output_extension,
    split_audio_file,
)


def export_silence(path: Path, duration_ms: int = 100):
    with path.open("wb") as target:
        AudioSegment.silent(duration=duration_ms, frame_rate=16000).export(target, format="wav")


class SplitterPipelineTests(unittest.TestCase):
    def test_output_extension_preserves_supported_extension(self):
        self.assertEqual(output_extension("/tmp/meeting.FLAC"), "flac")

    def test_output_extension_falls_back_to_mp3_for_unsupported_extension(self):
        self.assertEqual(output_extension("/tmp/meeting.mov"), "mp3")

    def test_export_format_maps_container_specific_formats(self):
        self.assertEqual(export_format_for_ext("m4a"), "ipod")
        self.assertEqual(export_format_for_ext("aac"), "adts")
        self.assertEqual(export_format_for_ext("wav"), "wav")

    def test_choose_cut_point_uses_silence_inside_search_window(self):
        audio = AudioSegment.silent(duration=1200, frame_rate=16000)
        settings = SplitterSettings(target_minutes=0.01, tolerance_minutes=0.003, min_silence_len=50)

        decision = choose_cut_point(audio, current_pos=0, settings=settings)

        self.assertTrue(decision.used_silence)
        self.assertGreaterEqual(decision.cut_point, decision.search_start)
        self.assertLessEqual(decision.cut_point, decision.search_end)

    def test_split_audio_file_exports_final_short_segment_and_reports_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            output_dir = Path(tmpdir) / "chunks"
            output_dir.mkdir()
            export_silence(source)
            logs = []
            progress = []

            result = split_audio_file(
                file_path=str(source),
                output_dir=str(output_dir),
                settings=SplitterSettings(target_minutes=1, tolerance_minutes=1),
                log_callback=logs.append,
                progress_callback=progress.append,
            )

            self.assertEqual(result.output_paths, [output_dir / "meeting_part01.wav"])
            self.assertTrue(result.output_paths[0].exists())
            self.assertEqual(progress[0], 5)
            self.assertEqual(progress[-1], 100)
            self.assertIn("All segments have been split and saved.", logs[-1])

    def test_split_audio_file_exports_multiple_chunks_for_longer_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "lecture.wav"
            output_dir = Path(tmpdir) / "chunks"
            output_dir.mkdir()
            export_silence(source, duration_ms=1500)

            result = split_audio_file(
                file_path=str(source),
                output_dir=str(output_dir),
                settings=SplitterSettings(target_minutes=0.01, tolerance_minutes=0.003, min_silence_len=50),
            )

            self.assertEqual(
                result.output_paths,
                [
                    output_dir / "lecture_part01.wav",
                    output_dir / "lecture_part02.wav",
                    output_dir / "lecture_part03.wav",
                ],
            )
            self.assertTrue(all(path.exists() for path in result.output_paths))

    def test_split_audio_file_rejects_zero_target_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            export_silence(source)

            with self.assertRaisesRegex(ValueError, "target_minutes"):
                split_audio_file(
                    file_path=str(source),
                    output_dir=tmpdir,
                    settings=SplitterSettings(target_minutes=0),
                )


if __name__ == "__main__":
    unittest.main()
