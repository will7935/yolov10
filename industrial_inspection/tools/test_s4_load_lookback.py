"""Regression coverage for delayed visual confirmation of early S4 hand contact."""
import unittest

import numpy as np

from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_s4_load_lookback import S4LoadLookback


POSE = np.array([[100, 100], [100, 200], [200, 100], [200, 200]], dtype=float)
ROI = np.float32([[60, 60], [240, 60], [240, 240], [60, 240]])
HAND = {"xyxy": [180, 180, 222, 220], "confidence": .9}


def track(now, center):
    x, y = center
    return {"last_seen": now, "reliable_since": now, "center": center,
            "xyxy": [x-20, y-18, x+20, y+18]}


class S4LoadLookbackTests(unittest.TestCase):
    def setUp(self):
        self.tracker = SlotGroupTracker(POSE, phase="load", load_hand_contact_seconds=0.,
                                        visual_reversal_review_seconds=.5)
        self.tracker.byte_track_ids = [10, 20, 30, None]
        self.review = S4LoadLookback(ROI, 40)

    def record(self, now, hands, *, both=False):
        records = {10: track(now, [100, 100]), 30: track(1., [200, 100])}
        if both:
            records[30] = track(now, [200, 100])
        return self.review.record(self.tracker, now, hands, records, POSE, {10})

    def test_hint_is_available_before_visual_confirmation(self):
        self.assertFalse(self.record(1., [HAND], both=True))
        self.assertTrue(self.record(1.1, [HAND]))
        self.record(1.2, [])
        tracker = self.tracker
        tracker.s4_load_contact_lookback = self.review.resolve(tracker, 1.2, empty_at=.1)
        self.assertEqual(tracker.events, [])
        slot = tracker.slots[3]
        slot.state = slot.confirmed = "filled"
        event = {"slot_id": 4, "from": "empty", "to": "filled", "action": "load",
                 "time_lower": .1, "time_upper": 2., "confirmed_at": 2.1, "reported_at": 2.1,
                 "recovered_before_initialization": False}
        self.assertEqual(tracker._review_visual_events(slot, [], event, 2.1), [])
        confirmed, = tracker._review_visual_events(slot, [], None, 2.7)
        self.assertEqual((confirmed["time_upper"], confirmed["visual_time_upper"]), (1.1, 2.))
        self.assertEqual(confirmed["confirmed_at"], 2.1)
        self.assertEqual(confirmed["hand_contact_frames"], 2)
        self.assertIsNone(tracker.s4_load_contact_lookback)

    def test_ambiguous_contacts_do_not_create_pending_evidence(self):
        for at in (1., 1.1, 1.2, 1.3, 1.4, 1.5):
            self.record(at, [] if at in (1.2, 1.5) else [HAND], both=True)
        self.assertIsNone(self.review.resolve(self.tracker, 1.5, empty_at=.1))

    def test_later_visual_empty_invalidates_contact(self):
        self.record(1., [HAND], both=True)
        self.record(1.1, [HAND])
        self.record(1.2, [])
        tracker = self.tracker
        tracker.s4_load_contact_lookback = self.review.resolve(tracker, 1.2, empty_at=.1)
        self.assertIsNotNone(tracker.s4_load_contact_lookback)
        slot = tracker.slots[3]
        slot.state = slot.confirmed = "empty"
        slot.confirmation_source = "yolo"
        tracker._review_visual_events(slot, [], None, 1.3)
        self.assertIsNone(tracker.s4_load_contact_lookback)
