"""Regression tests for the provisional S4 contact identity lock."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker


CENTERS = np.array([[100., 100.], [200., 100.], [100., 200.], [200., 200.]])


def detection(index, state="filled", centers=CENTERS, track_id=None):
    x, y = centers[index - 1]
    result = {
        "class_name": "wafer_slot_" + state,
        "confidence": .9,
        "xyxy": [x - 20, y - 20, x + 20, y + 20],
    }
    if track_id is not None:
        result["track_id"] = track_id
    return result


class ProvisionalS4LockTests(unittest.TestCase):
    def make_tracker(self):
        tracker = SlotGroupTracker(
            group_id=2, phase="unload", confirm_seconds=0., minimum_frames=1,
            initialization_seconds=1., initialization_frames=4,
            load_hand_contact_seconds=0., load_hand_contact_min_frames=2,
        )
        tracker.start_incremental_entry(40.)
        return tracker

    @staticmethod
    def hand():
        return {"class_name": "hand", "confidence": .9,
                "xyxy": [180., 180., 220., 220.]}

    def test_pending_contact_keeps_same_slot_as_s4_when_geometry_order_changes(self):
        tracker = self.make_tracker()
        full = [detection(index, track_id=100 + index) for index in range(1, 5)]
        tracker.update(full, 0.)
        tracker.update(full, .1)
        tracker.update(full + [self.hand()], .2)
        result = tracker.update(full + [self.hand()], .3)

        self.assertEqual(result["s4_contact_lock"]["internal_slot_index"], 4)
        locked_slot = tracker._entry_locked_s4_slot
        self.assertIsNotNone(locked_slot)

        # Make the geometric S4 candidate move to another temporary index.
        # The contact-bearing SlotState must remain the selected S4.
        tracker._entry_points[3] = np.array([200., 50.])
        self.assertNotEqual(int(tracker._numbering_order(np.asarray(tracker._entry_points))[3]), 3)
        self.assertEqual(tracker._entry_s4_index(), 3)
        self.assertIs(tracker._entry_locked_s4_slot, locked_slot)

    def test_visual_confirmation_releases_the_lock(self):
        tracker = self.make_tracker()
        full = [detection(index, track_id=100 + index) for index in range(1, 5)]
        tracker.update(full, 0.)
        tracker.update(full, .1)
        tracker.update(full + [self.hand()], .2)
        tracker.update(full + [self.hand()], .3)

        empty_s4 = [detection(index, track_id=100 + index) for index in range(1, 4)]
        empty_s4.append(detection(4, state="empty", track_id=104))
        result = tracker.update(empty_s4, .4)

        self.assertIsNone(result["s4_contact_lock"])
        self.assertIsNone(tracker._entry_locked_s4_slot)
        self.assertEqual(result["slots"][3]["last_confirmed_state"], "empty")

    def test_contact_cancellation_releases_the_lock(self):
        tracker = self.make_tracker()
        full = [detection(index, track_id=100 + index) for index in range(1, 5)]
        tracker.update(full, 0.)
        tracker.update(full, .1)
        tracker.update(full + [self.hand()], .2)
        tracker.update(full + [self.hand()], .3)

        locked_slot = tracker._entry_locked_s4_slot
        self.assertIsNotNone(locked_slot)
        tracker._clear_load_contact(locked_slot)
        self.assertIsNone(tracker._entry_locked_s4_slot)

    def test_track_loss_releases_the_lock_after_the_loss_window(self):
        tracker = self.make_tracker()
        full = [detection(index, track_id=100 + index) for index in range(1, 5)]
        tracker.update(full, 0.)
        tracker.update(full, .1)
        tracker.update(full + [self.hand()], .2)
        tracker.update(full + [self.hand()], .3)
        self.assertIsNotNone(tracker._entry_locked_s4_slot)

        short_gap = tracker.update([], .8)
        self.assertIsNotNone(short_gap["s4_contact_lock"])

        result = tracker.update([], 2.0)

        self.assertIsNone(result["s4_contact_lock"])
        self.assertIsNone(tracker._entry_locked_s4_slot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
