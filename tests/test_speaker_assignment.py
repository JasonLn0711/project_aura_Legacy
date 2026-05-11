import unittest

from aura.diarization.pyannote_pipeline import DiarizationSettings, pipeline_kwargs
from aura.diarization.speaker_assignment import (
    UNKNOWN_SPEAKER,
    SpeakerTurn,
    TranscriptSegment,
    assign_speakers,
    speaker_for_segment,
)


class SpeakerAssignmentTests(unittest.TestCase):
    def test_assigns_speaker_with_largest_overlap(self):
        segment = TranscriptSegment(start=10.0, end=16.0, text="hello")
        turns = [
            SpeakerTurn(start=9.0, end=11.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=11.0, end=17.0, speaker="SPEAKER_01"),
        ]

        self.assertEqual(speaker_for_segment(segment, turns), "SPEAKER_01")

    def test_assigns_unknown_when_no_turn_matches(self):
        segment = TranscriptSegment(start=30.0, end=32.0, text="hello")
        turns = [SpeakerTurn(start=9.0, end=11.0, speaker="SPEAKER_00")]

        self.assertEqual(speaker_for_segment(segment, turns), UNKNOWN_SPEAKER)

    def test_assign_speakers_preserves_transcript_order(self):
        segments = [
            TranscriptSegment(start=0.0, end=2.0, text="a"),
            TranscriptSegment(start=3.0, end=4.0, text="b"),
        ]
        turns = [
            SpeakerTurn(start=0.0, end=2.5, speaker="SPEAKER_00"),
            SpeakerTurn(start=2.5, end=5.0, speaker="SPEAKER_01"),
        ]

        labels = assign_speakers(segments, turns)

        self.assertEqual([item.speaker for item in labels], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual([item.transcript.text for item in labels], ["a", "b"])

    def test_pipeline_kwargs_uses_exact_count_when_min_equals_max(self):
        settings = DiarizationSettings(enabled=True, min_speakers=3, max_speakers=3)

        self.assertEqual(pipeline_kwargs(settings), {"num_speakers": 3})

    def test_pipeline_kwargs_uses_range_when_bounds_differ(self):
        settings = DiarizationSettings(enabled=True, min_speakers=2, max_speakers=5)

        self.assertEqual(pipeline_kwargs(settings), {"min_speakers": 2, "max_speakers": 5})

    def test_rejects_invalid_speaker_range(self):
        with self.assertRaises(ValueError):
            DiarizationSettings(enabled=True, min_speakers=4, max_speakers=2)


if __name__ == "__main__":
    unittest.main()
