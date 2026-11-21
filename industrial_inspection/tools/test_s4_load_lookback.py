"""Regression coverage for delayed visual confirmation of early S4 hand contact."""
import unittest

import numpy as np

from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_s4_load_lookback import S4LoadContact, S4LoadLookback


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

    def test_crop_reach_requires_consecutive_frames_and_missing_s4(self):
        points = np.array([[100., 180.], [160., 180.], [130., 140.], [190., 140.]])
        tracker = SlotGroupTracker(points, phase="load", load_hand_contact_seconds=0.)
        tracker.byte_track_ids = [10, 20, 30, 40]
        s4_hand = {"xyxy": [140., 126., 196., 154.]}
        s2_hand = {"xyxy": [116., 166., 164., 194.]}
        for name, observed, hand_sequence in (
            ("occluded", False, [s4_hand, s4_hand]),
            ("still visible", True, [s4_hand, s4_hand]),
            ("separate frames", False, [s4_hand, None, s4_hand]),
            ("other slot", False, [s2_hand, s2_hand]),
        ):
            with self.subTest(name=name):
                self.review.clear()
                for index, hand in enumerate([*hand_sequence, None]):
                    at = 1.+index*.05
                    anchors = {10: track(at, points[0]), 30: track(at, points[2])}
                    if observed:
                        anchors[40] = track(at, points[3])
                    self.review.record(tracker, at, [], anchors, None, (), reach_hand=hand)
                contact = self.review.resolve_reach(tracker, at, empty_at=.1)
                if name == "occluded":
                    self.assertEqual((contact.first_at, contact.last_at, contact.frames), (1., 1.05, 2))
                    self.assertTrue(contact.require_bound_visual and contact.from_reach_crop)
                else:
                    self.assertIsNone(contact)

    def test_crop_reach_survives_visual_review_but_expires_before_late_confirmation(self):
        tracker = self.tracker
        tracker._region_formal_numbering = True
        tracker.byte_track_ids[3] = 40
        slot = tracker.slots[3]
        slot.state = slot.confirmed = "filled"
        slot.confirmation_source = "yolo"
        event = {"slot_id": 4, "from": "empty", "to": "filled", "action": "load",
                 "time_lower": .5, "time_upper": 1.9, "confirmed_at": 2.,
                 "recovered_before_initialization": False, "byte_track_id": 40}
        for deadline in (2.1, 1.95):
            with self.subTest(deadline=deadline):
                tracker.s4_load_contact_lookback = S4LoadContact(
                    1., 1.05, 2, deadline, require_bound_visual=True, from_reach_crop=True)
                self.assertFalse(tracker._review_visual_events(slot, [], event, 2.))
                finalized, = tracker._review_visual_events(slot, [], None, 2.6)
                self.assertEqual(finalized["time_upper"], 1.05 if deadline > 2. else 1.9)
                self.assertEqual(finalized.get("hand_contact_source"), "entry_reach_crop" if deadline > 2. else None)

    def test_ordinary_entry_releases_only_after_same_s4_visual_review(self):
        points = np.array([[100., 180.], [160., 180.], [130., 140.], [190., 140.]])
        tracker = SlotGroupTracker(group_id=3, phase="load", slot2_below_slot3=True,
            allow_three_point_initialization=True, initialization_seconds=.1, minimum_frames=2,
            confirm_seconds=0., load_hand_contact_seconds=0., visual_reversal_review_seconds=.3)
        tracker._load_numbering_reference = points.copy()
        sequence = RoiGroupSequence(tracker, ROI)
        unloaded = SlotGroupTracker(points, group_id=3)
        unloaded.slots[3].confirmed = "empty"
        unloaded.slots[3].confirmation_source = "yolo"
        unloaded.slots[3].evidence_time = -1.
        sequence.closed_groups.append({"phase": "unload", "group_id": 3, "snapshot": unloaded.snapshot()})
        tracker.start_incremental_entry(32.)
        sequence.state = "entering"
        slots = [{"class_name": "wafer_slot_empty", "confidence": .95, "track_id": i,
                  "xyxy": [x-20, y-20, x+20, y+20]} for i, (x, y) in enumerate(points, start=1)]
        hand = {"class_name": "hand", "confidence": .95, "xyxy": [171., 122., 208., 158.]}
        for tick in range(7):
            result = sequence.update([slots[0], slots[2]] + ([hand] if tick in (4, 5) else []), tick/10)
            self.assertFalse(result["new_events"])
        for now in (.7, .8, .9):
            result = sequence.update(slots[:3], now)
        self.assertEqual(result["slot_numbering_status"], "final")
        self.assertTrue(tracker.s4_load_contact_lookback.require_bound_visual)
        self.assertFalse(sequence.all_events)

        filled = dict(slots[3], class_name="wafer_slot_filled")
        for now in (1., 1.1, 1.2, 1.3):
            self.assertFalse(sequence.update(slots[:3] + [filled], now)["new_events"])
        event, = sequence.update(slots[:3] + [filled], 1.41)["new_events"]
        self.assertEqual(event["reason"], "visual_after_s4_contact_lookback")
        self.assertEqual((event["slot_id"], event["time_upper"], event["visual_time_upper"],
                          event["confirmed_at"], event["reported_at"]), (4, .5, 1., 1.1, 1.41))
        self.assertEqual(sequence.all_events, [event])
