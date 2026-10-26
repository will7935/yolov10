"""Offline tests for ByteTrack-ID state labels and track records."""
from __future__ import annotations

import unittest

from industrial_inspection.tools.slot_track_display import slot_track_label, slot_track_record


class ByteTrackStateDisplayTests(unittest.TestCase):
    def test_provisional_track_label_uses_byte_track_id_and_maintained_state(self):
        snapshot = {"slot_numbering_status": "provisional", "slots": [{
            "group_id": 2, "slot_id": 1, "byte_track_id": 41,
            "state": "unknown", "last_confirmed_state": "empty",
            "observation": "unknown", "reason": "hand_occlusion",
            "confirmation_source": "yolo"}]}
        detection = {"track_id": 41, "class_name": "wafer_slot_filled", "confidence": .91}

        label = slot_track_label(detection, snapshot)
        record = slot_track_record(detection, snapshot)

        self.assertIn("BT:41", label)
        self.assertIn("empty*", label)
        self.assertIn("det:filled", label)
        self.assertNotIn("T1", label)
        self.assertEqual(record["slot_state"], "unknown")
        self.assertEqual(record["last_confirmed_state"], "empty")
        self.assertIsNone(record["slot_id"])
        self.assertIsNone(record["formal_slot_id"])

    def test_unmapped_track_records_detection_without_claiming_slot_state(self):
        snapshot = {"slot_numbering_status": "provisional", "slots": []}
        detection = {"track_id": 52, "class_name": "wafer_slot_empty", "confidence": .8}

        record = slot_track_record(detection, snapshot)

        self.assertIsNone(record["slot_id"])
        self.assertEqual(record["slot_state"], "unknown")
        self.assertEqual(record["observation"], "empty")
        self.assertEqual(record["state_reason"], "unmapped_track")

    def test_final_numbering_keeps_formal_slot_mapping_alongside_track_id(self):
        snapshot = {"slot_numbering_status": "final", "slots": [{
            "group_id": 3, "slot_id": 4, "byte_track_id": 77,
            "state": "filled", "last_confirmed_state": "filled",
            "observation": "filled", "reason": "confirmed",
            "confirmation_source": "yolo"}]}
        detection = {"track_id": 77, "class_name": "wafer_slot_filled", "confidence": .95}

        label = slot_track_label(detection, snapshot)

        self.assertIn("BT:77", label)
        self.assertIn("G3-S4", label)
        self.assertIn("filled", label)


if __name__ == "__main__":
    unittest.main()
