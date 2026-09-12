import json
import unittest
from datetime import datetime, UTC

from src.models.meeting import MeetingDocument, TranscriptSegment


class TestMeetingModel(unittest.TestCase):
    def test_meeting_document_creation_from_transcript_result(self):
        result = {
            "text": "hello world",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "hello"},
                {"start": 1.5, "end": 3.0, "text": "world"},
            ],
            "filename": "sample.wav",
            "duration": 3.0,
            "warnings": ["mixed script"],
        }

        meeting = MeetingDocument.from_transcription_result(result, file_path="/tmp/sample.wav")

        self.assertEqual(meeting.source_audio.file_path, "/tmp/sample.wav")
        self.assertEqual(meeting.metadata.language, "und")
        self.assertEqual(len(meeting.transcript_segments), 2)
        self.assertEqual(meeting.transcript_segments[0].text, "hello")
        self.assertEqual(meeting.summary.text, "hello world")
        self.assertEqual(meeting.status.overall, "completed")
        self.assertEqual(meeting.errors, [])

    def test_transcript_segment_requires_valid_range(self):
        with self.assertRaises(ValueError):
            TranscriptSegment(id="seg-1", start_time=2.0, end_time=1.0, text="bad")

    def test_json_round_trip_preserves_metadata_and_timestamps(self):
        meeting = MeetingDocument(
            id="meeting-123",
            source_audio={
                "file_path": "/tmp/sample.wav",
                "original_name": "sample.wav",
                "size_bytes": 42,
                "duration_seconds": 12.5,
                "format": "wav",
            },
            metadata={
                "title": "Weekly sync",
                "date": "2026-08-22",
                "language": "en",
                "participants": ["A", "B"],
            },
            transcript_segments=[
                TranscriptSegment(id="seg-1", start_time=0.0, end_time=2.0, text="Hi", source_model="faster-whisper", language="en"),
            ],
            created_at=datetime(2026, 8, 22, 10, 0, 0),
            updated_at=datetime(2026, 8, 22, 10, 1, 0),
            processed_at=datetime(2026, 8, 22, 10, 2, 0),
        )

        payload = meeting.to_json()
        loaded = MeetingDocument.from_json(payload)

        self.assertEqual(loaded.id, meeting.id)
        self.assertEqual(loaded.transcript_segments[0].text, "Hi")
        self.assertEqual(loaded.metadata.title, "Weekly sync")
        self.assertEqual(loaded.created_at.isoformat(), meeting.created_at.isoformat())
        self.assertEqual(loaded.processed_at.isoformat(), meeting.processed_at.isoformat())
        self.assertEqual(loaded.errors, [])
        self.assertEqual(loaded.topics, [])

    def test_empty_optional_collections_are_preserved(self):
        meeting = MeetingDocument(
            id="meeting-2",
            source_audio={
                "file_path": "/tmp/sample.wav",
                "original_name": "sample.wav",
                "size_bytes": 1,
                "duration_seconds": 1.0,
                "format": "wav",
            },
            metadata={},
            speakers=[],
            transcript_segments=[],
            topics=[],
            decisions=[],
            action_items=[],
            follow_up_items=[],
            entities=[],
            summary={},
            outputs=[],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            processed_at=datetime.now(UTC),
        )

        payload = json.loads(meeting.to_json())
        self.assertEqual(payload["topics"], [])
        self.assertEqual(payload["decisions"], [])
        self.assertEqual(payload["action_items"], [])
        self.assertEqual(payload["follow_up_items"], [])
        self.assertEqual(payload["entities"], [])
        self.assertEqual(payload["outputs"], [])


if __name__ == "__main__":
    unittest.main()
