"""An interior G1 quartet must displace a neighboring carrier's edge candidate."""
import unittest

import numpy as np

from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_track_state import InitialEdgeSlot


ROI = np.array([[279, 144], [628, 228], [633, 384], [300, 318]], dtype=np.float32)
PEERS = {1: (482., 222.), 53: (471., 283.), 63: (548., 275.)}
EDGE = (550., 207.)
INTERIOR = (386., 217.)


def detection(track_id, center):
    x, y = center
    return {"class_name": "wafer_slot_filled", "confidence": .95, "track_id": track_id,
            "xyxy": [x-10, y-8, x+10, y+8]}


class InitialEdgeReplacementTests(unittest.TestCase):
    def make(self):
        tracker = SlotGroupTracker(phase="unload", group_id=1,
            initialization_seconds=.1, geometry_initialization_seconds=.3,
            initialization_frames=2, require_unload_event_before_numbering=True)
        sequence = RoiGroupSequence(tracker, ROI, edge_evidence_px=8,
                                    require_first_unload_event_for_g1=True)
        points = np.array([*PEERS.values(), EDGE])
        provisional = SlotGroupTracker(points, phase="unload", group_id=1)
        provisional.byte_track_ids = [*PEERS, 44]
        provisional._region_formal_numbering = False
        provisional.track_store = sequence.track_store
        for slot in provisional.slots:
            slot.confirmed = "filled"
        tracker._provisional = provisional
        tracker._initial_seen = set(range(4))
        tracker._pending_numbering_points = points.copy()
        tracker._pending_numbering_order = np.arange(4)
        sequence._g1_edge_slot = InitialEdgeSlot(np.array(EDGE),
            {tid: np.array(point) for tid, point in PEERS.items()}, 0.,
            frames=10, track_ids={44}, filled_at=0.)
        return sequence

    def objects(self, new_id=87):
        return [*(detection(tid, point) for tid, point in PEERS.items()),
                detection(new_id, INTERIOR), detection(44, EDGE)]

    def test_stable_interior_quartet_replaces_only_edge_and_preserves_peer_evidence(self):
        sequence = self.make()
        parent, provisional = sequence.tracker, sequence.tracker._provisional
        peer_slots = provisional.slots[:3].copy()
        event = {"slot_id": 3, "byte_track_id": 63, "action": "unload", "confirmed_at": -.1}
        wrong_event = {"slot_id": 4, "byte_track_id": 44, "action": "unload", "confirmed_at": -.1}
        pending = {"action": "unload", "confirmed_at": 0.}
        provisional.slots[2].pending_visual_event = pending
        provisional.events = [event, wrong_event]
        for now in (0., .2, .4):
            result = sequence._admit_initial_edge_slot(self.objects(), (), now)
        self.assertEqual(provisional.byte_track_ids, [1, 53, 63, 87])
        self.assertEqual(provisional.events, [event])
        self.assertIs(provisional.slots[2].pending_visual_event, pending)
        for previous, retained in zip(peer_slots, provisional.slots[:3]):
            self.assertIs(previous, retained)
        self.assertEqual(provisional.slots[3].confirmed, "unknown")
        np.testing.assert_array_equal(provisional.points[3], INTERIOR)
        self.assertIsNone(parent._pending_numbering_order)
        self.assertIsNone(parent._pending_numbering_points)
        self.assertIsNone(sequence._g1_edge_slot)
        self.assertEqual(sequence._g1_rejected_edge_ids, {44})
        self.assertNotIn(44, [obj.get("track_id") for obj in result])
        self.assertEqual(parent.initialization["initial_edge_replacement"]["preserved_event_count"], 1)
        # An occluded real slot must not allow the rejected edge ID to return.
        later = sequence._admit_initial_edge_slot(self.objects()[:-2]+[detection(44, EDGE)], (), .6)
        self.assertNotIn(44, [obj.get("track_id") for obj in later])
        # The exclusion applies only to this G1, so later carriers can use that ID.
        sequence.tracker = sequence._new_tracker(2, phase="unload")
        self.assertIn(44, [obj.get("track_id") for obj in
                          sequence._admit_initial_edge_slot(self.objects(), (), .8)])

    def test_transient_quartet_does_not_accumulate_across_missing_support(self):
        sequence = self.make()
        sequence._admit_initial_edge_slot(self.objects(), (), 0.)
        sequence._admit_initial_edge_slot(self.objects()[:-2]+[detection(44, EDGE)], (), .2)
        for now in (.4, .6):
            sequence._admit_initial_edge_slot(self.objects(), (), now)
            self.assertEqual(sequence.tracker._provisional.byte_track_ids[3], 44)
        sequence._admit_initial_edge_slot(self.objects(), (), .8)
        self.assertEqual(sequence.tracker._provisional.byte_track_ids[3], 87)

    def test_quartet_requires_same_three_peer_ids_real_geometry_and_unoccluded_tracks(self):
        for case in ("changed_peer", "changed_fourth", "invalid_geometry", "occluded", "local_crop"):
            with self.subTest(case=case):
                sequence = self.make()
                for frame, now in enumerate((0., .2, .4, .6)):
                    objects = self.objects()
                    if case == "changed_peer":
                        objects[0] = detection(99, PEERS[1])
                    elif case == "changed_fourth":
                        objects[3] = detection(87+frame, INTERIOR)
                    elif case == "invalid_geometry":
                        objects[3] = detection(87, (500., 250.))
                    elif case == "occluded":
                        objects.append({"class_name": "hand", "confidence": .95,
                                        "xyxy": [370, 200, 400, 235]})
                    elif case == "local_crop":
                        objects[3]["detection_source"] = "local_crop"
                    sequence._admit_initial_edge_slot(objects, (), now)
                self.assertNotIn(87, sequence.tracker._provisional.byte_track_ids)
                self.assertFalse(sequence._g1_rejected_edge_ids)

    def test_formal_slot_numbers_are_not_reassigned_by_this_review(self):
        sequence = self.make()
        sequence.tracker.points = sequence.tracker._provisional.points.copy()
        for now in (0., .2, .4):
            sequence._admit_initial_edge_slot(self.objects(), (), now)
        self.assertEqual(sequence.tracker._provisional.byte_track_ids[3], 44)
        self.assertFalse(sequence._g1_rejected_edge_ids)

    def test_nearby_new_id_preserves_the_existing_edge_evidence_for_normal_recovery(self):
        sequence = self.make()
        original = sequence._g1_edge_slot
        for now in (0., .2, .4, .6):
            objects = self.objects()
            objects[3] = detection(87, (550., 215.))
            sequence._admit_initial_edge_slot(objects, (), now)
        self.assertIs(sequence._g1_edge_slot, original)
        self.assertEqual(sequence.tracker._provisional.byte_track_ids[3], 44)
        self.assertFalse(sequence._g1_rejected_edge_ids)


if __name__ == "__main__":
    unittest.main()
