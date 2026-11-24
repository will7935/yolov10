"""Two measured slots can number a group only after a unique stable template match."""
import unittest

import numpy as np

from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_perspective_numbering import SlotPerspectiveNumbering
from industrial_inspection.tools.slot_two_point_contact import TwoPointHandContact
from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence


ROI = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
POINTS = np.array([[100, 100], [100, 200], [200, 100], [200, 200]], dtype=float)


def calibration():
    return SlotPerspectiveNumbering({"image_size": [400, 400], "homography": np.eye(3).tolist(),
        "unload_to_load_clockwise_degrees": 0., "angle_tolerance_degrees": 25.,
        "maximum_error_ratio": .45}, (400, 400))


def objects(shift=(0, 0)):
    return [{"class_name": "wafer_slot_empty", "confidence": .95, "track_id": 11+i,
             "xyxy": [x-200/3, y-20, x+200/3, y+20]}
            for i, (x, y) in enumerate(POINTS[:2]+shift)]


def tracker(reference=True):
    result = SlotGroupTracker(group_id=1, phase="load", confirm_seconds=0., minimum_frames=3,
        initialization_seconds=.2, geometry_initialization_seconds=.2, initialization_frames=2,
        slot2_below_slot3=True, allow_three_point_initialization=True)
    result._load_perspective_numbering = calibration()
    result._load_numbering_reference = POINTS.copy() if reference else None
    result.two_point_hand_contact = TwoPointHandContact(1.5, [(ROI[0], ROI[1])], ROI)
    result.start_incremental_entry(40)
    return result


class TwoPointNumberingTests(unittest.TestCase):
    def test_first_pair_frame_shows_two_unnumbered_predictions_without_state_changes(self):
        t = tracker(False)
        result = t.update(objects(), 0., roi_contour=ROI)
        self.assertEqual(result["slot_numbering_status"], "provisional")
        self.assertEqual(len(result["two_point_hand_contacts"]), 2)
        self.assertTrue(all(p["slot_id"] is None and not p["contact_enabled"]
                            for p in result["two_point_hand_contacts"]))
        np.testing.assert_allclose([p["center_xy"] for p in result["two_point_hand_contacts"]], POINTS[2:])
        self.assertFalse(t.events)

    def test_unique_pair_numbers_group_and_retains_two_predictions_as_unknown(self):
        t = tracker()
        for now in (0., .1):
            result = t.update(objects(), now, roi_contour=ROI)
            self.assertEqual(result["slot_numbering_status"], "provisional")
        result = t.update(objects(), .2, roi_contour=ROI)
        self.assertEqual(result["slot_numbering_status"], "final")
        self.assertEqual(t.byte_track_ids, [11, 12, None, None])
        self.assertEqual(t.initialization["inferred_slot_ids"], [3, 4])
        self.assertEqual(t.initialization["required_slot_count"], 2)
        self.assertEqual(t.initialization["two_point_numbering"]["measured_support"], 2)
        self.assertEqual(t.inferred_load_contact_indices(), (2, 3))
        self.assertEqual([t.slots[i].confirmed for i in (2, 3)], ["unknown", "unknown"])
        self.assertFalse(t.events)
        np.testing.assert_allclose(t.points, POINTS)
        self.assertEqual([p['slot_id'] for p in result['two_point_hand_contacts']], [3, 4])
        t.update(objects(), .3, roi_contour=ROI)
        self.assertEqual(t.two_point_hand_contact.formal_slot_ids, [3, 4])

    def test_verified_two_point_group_can_build_g1_template_without_inventing_states(self):
        t = tracker()
        sequence = RoiGroupSequence(t, ROI, load_numbering_calibration=t._load_perspective_numbering)
        for now in (0., .1, .2, .3, .4):
            result = t.update(objects(), now, roi_contour=ROI)
            sequence.state = 'active' if t.points is not None else 'entering'
            sequence._record_g1_load_layout(result, now)
        np.testing.assert_allclose(sequence._g1_load_layout, POINTS)
        self.assertEqual(sequence._g1_load_layout_inferred_slot_ids, (3, 4))
        self.assertFalse(t.events)
        self.assertEqual([t.slots[i].confirmed for i in (2, 3)], ['unknown', 'unknown'])
        later = sequence._new_tracker(2, phase='load')
        self.assertIsNotNone(later._g1_load_numbering_reference)

    def test_no_reference_keeps_preview_and_contact_but_never_formalizes(self):
        t = tracker(False)
        hand = {"class_name": "hand", "confidence": .95, "xyxy": [175, 75, 225, 125]}
        for frame in range(12):
            result = t.update(objects()+[hand], frame/10, roi_contour=ROI)
        self.assertEqual(result["slot_numbering_status"], "provisional")
        self.assertEqual(len(result["two_point_hand_contacts"]), 2)
        self.assertTrue(t.two_point_hand_contact.pending[0].ready)
        self.assertFalse(t.events)

    def test_moving_pair_cannot_accumulate_numbering_stability(self):
        t = tracker()
        for frame in range(7):
            result = t.update(objects((frame*12, 0)), frame/10, roi_contour=ROI)
            self.assertEqual(result["slot_numbering_status"], "provisional")
            self.assertEqual(len(result["two_point_hand_contacts"]), 2)

    def test_released_contacts_remain_independent_for_both_predicted_slots(self):
        t = tracker(False)
        t.load_hand_contact_seconds, t.load_hand_contact_min_frames = .2, 3
        hands = [{"class_name": "hand", "confidence": .95, "xyxy": [175, y-25, 225, y+25]}
                 for y in (100, 200)]
        for frame in range(13):
            hand = [hands[0]] if 3 <= frame <= 5 else [hands[1]] if 8 <= frame <= 10 else []
            t.update(objects()+hand, frame/10, roi_contour=ROI)
        self.assertEqual(len(t.two_point_hand_contact.released_loads), 2)
        t._load_numbering_reference = POINTS.copy()
        for frame in range(13, 16):
            t.update(objects(), frame/10, roi_contour=ROI)
        self.assertIsNotNone(t.points)
        first = t.two_point_hand_contact.take_inferred_load(t, 2, -.1, 1.5)
        second = t.two_point_hand_contact.take_inferred_load(t, 3, -.1, 1.5)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertAlmostEqual(first.since, .3)
        self.assertAlmostEqual(second.since, .8)
        self.assertFalse(t.events)

    def test_missing_support_breaks_continuous_numbering_verification(self):
        t = tracker()
        for now in (0., .1):
            t.update(objects(), now, roi_contour=ROI)
        t.update(objects()[:1], .2, roi_contour=ROI)
        for now in (.3, .4):
            result = t.update(objects(), now, roi_contour=ROI)
            self.assertEqual(result["slot_numbering_status"], "provisional")
        self.assertEqual(t.update(objects(), .5, roi_contour=ROI)["slot_numbering_status"], "final")

    def test_pausing_observations_restarts_pair_numbering_verification(self):
        t = tracker()
        for now in (0., .1):
            t.update(objects(), now, roi_contour=ROI)
        t.update(objects(), .2, roi_contour=ROI, observations_enabled=False)
        for now in (.3, .4):
            self.assertEqual(t.update(objects(), now, roi_contour=ROI)["slot_numbering_status"], "provisional")
        self.assertEqual(t.update(objects(), .5, roi_contour=ROI)["slot_numbering_status"], "final")

    def test_template_geometry_and_roi_reject_wrong_side_or_large_scale(self):
        m = calibration()
        layout = POINTS[[0, 1, 2, 3]]
        roles, info = m.match_pair(POINTS, layout, np.array([1., 0.]), ROI)
        self.assertEqual(roles, (0, 1, 2, 3))
        self.assertEqual(info["candidate_count"], 1)
        wrong = layout.copy()
        wrong[2:] += [150, 0]
        roles, info = m.match_pair(POINTS, wrong, np.array([1., 0.]), ROI)
        self.assertIsNone(roles)
        self.assertEqual(info["status"], "no_compatible_pair")

    def test_multiple_compatible_correspondences_remain_unnumbered(self):
        m = SlotPerspectiveNumbering({"image_size": [400, 400], "homography": np.eye(3).tolist(),
            "unload_to_load_clockwise_degrees": 25., "angle_tolerance_degrees": 44.,
            "maximum_error_ratio": .95}, (400, 400))
        reference = np.array([[100., 100.], [100., 200.], [170., 130.], [170., 230.]])
        roles, info = m.match_pair(reference, POINTS, np.array([1., 0.]), ROI)
        self.assertIsNone(roles)
        self.assertEqual(info["status"], "ambiguous")
        self.assertEqual(info["candidate_count"], 2)

    def test_confirmed_upper_s3_lower_s1_anchor_numbers_without_rotation_template_match(self):
        for reverse_ids in (False, True):
            with self.subTest(reverse_ids=reverse_ids):
                t = tracker(False)
                t.two_point_hand_contact.upper_slot_three = True
                detected = objects()
                if reverse_ids:
                    detected[0]['track_id'], detected[1]['track_id'] = 12, 11
                for now in (0., .1, .2):
                    result = t.update(detected, now, roi_contour=ROI)
                self.assertEqual(result['slot_numbering_status'], 'final')
                self.assertEqual(t.byte_track_ids[2], detected[0]['track_id'])
                self.assertEqual(t.byte_track_ids[0], detected[1]['track_id'])
                self.assertEqual(t.initialization['inferred_slot_ids'], [2, 4])
                self.assertEqual(t.initialization['two_point_numbering']['reference'],
                                 'configured_upper_s3_lower_s1')
                np.testing.assert_allclose(t.points, POINTS[[1, 3, 0, 2]])
                self.assertFalse(t.events)

    def test_explicit_upper_s3_rule_is_only_applied_to_loading_g1(self):
        for group_id, phase in ((2, 'load'), (1, 'unload')):
            with self.subTest(group_id=group_id, phase=phase):
                t = tracker(False)
                t.group_id, t.phase = group_id, phase
                t.two_point_hand_contact.upper_slot_three = True
                for now in (0., .1, .2, .3):
                    result = t.update(objects(), now, roi_contour=ROI)
                self.assertEqual(result['slot_numbering_status'], 'provisional')

    def test_explicit_anchor_source_setting_is_scoped_to_confirmed_video(self):
        from pathlib import Path
        from types import SimpleNamespace
        from industrial_inspection.tools.predict_slot_groups import make_sequence
        for filename, expected in (('camera_B_20260924_062208.avi', True), ('other.mp4', False)):
            sequence = make_sequence(ROI/400, (400, 400), args=SimpleNamespace(source=Path(filename)))
            self.assertEqual(sequence.tracker.two_point_hand_contact.upper_slot_three, expected)
            self.assertEqual(sequence._new_tracker(1, phase='load').two_point_hand_contact.upper_slot_three,
                             expected)


if __name__ == "__main__":
    unittest.main()
