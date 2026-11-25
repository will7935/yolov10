"""Corner selection follows physical slot identity rather than maximum image x."""
import itertools
import json
from pathlib import Path
import unittest

import numpy as np

from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_perspective_numbering import SlotPerspectiveNumbering
from industrial_inspection.tools.slot_process_config import LOAD_NUMBERING_CALIBRATION_FILE
from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence


REFERENCE = np.array([[444.50, 294.71], [563.62, 337.20], [469.61, 251.11], [580.97, 288.88]])
OBSERVED = np.array([[482.28, 261.36], [465.54, 307.18], [594.36, 299.46]])
ROI = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
VIDEO7_REFERENCE = np.array([[408.64, 251.45], [507.38, 315.36], [474.28, 234.77], [584.72, 300.3]])
VIDEO7_SUPPORT = np.array([[486.05, 248.81], [434.37, 275.01], [551.48, 310.74]])


class ThreePointTemplateTests(unittest.TestCase):
    def setUp(self):
        config = json.loads(Path(LOAD_NUMBERING_CALIBRATION_FILE).read_text(encoding='utf-8'))
        self.matcher = SlotPerspectiveNumbering(config, (640, 480))

    def test_video12_missing_s2_is_lower_right_for_every_detection_order(self):
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                point, info = self.matcher.missing_load_point(REFERENCE, OBSERVED[list(order)], ROI)
                np.testing.assert_allclose(point, [577.62, 345.28], atol=.01)
                self.assertEqual(info['missing_slot_id'], 2)
                self.assertEqual(info['slot_ids_in_observation_order'], np.array([3, 1, 4])[list(order)].tolist())
                self.assertEqual(info['measured_support'], 3)

    def test_template_can_also_select_an_upper_right_corner(self):
        observed = REFERENCE[[0, 1, 2]]+[10, -5]
        point, info = self.matcher.missing_load_point(REFERENCE, observed, ROI)
        self.assertIsNotNone(point)
        self.assertEqual(info['missing_slot_id'], 4)
        self.assertLess(point[1], observed[1, 1])

    def test_missing_reference_does_not_enable_rightmost_fallback(self):
        tracker = SlotGroupTracker(phase='load', group_id=2)
        tracker._load_perspective_numbering = self.matcher
        point, groups = tracker._infer_entry_s4(OBSERVED, ROI)
        self.assertIsNone(point)
        self.assertEqual(groups, [])
        self.assertEqual(tracker.initialization['three_point_template_prediction']['status'], 'waiting_for_reference')

    def test_no_compatible_in_roi_corner_waits(self):
        tiny_roi = np.float32([[400, 200], [500, 200], [500, 300], [400, 300]])
        point, info = self.matcher.missing_load_point(REFERENCE, OBSERVED, tiny_roi)
        self.assertIsNone(point)
        self.assertEqual(info['status'], 'no_compatible_corner')

    def test_validated_g1_pose_is_cached_immediately_despite_current_occlusion(self):
        tracker = SlotGroupTracker(REFERENCE, phase='load', group_id=1)
        sequence = RoiGroupSequence(tracker, ROI, load_numbering_calibration=self.matcher)
        tracker.last_time = 108.1
        tracker.initialization.update(perspective_load_numbering={'status': 'matched'},
            numbering_reference_centers_xy=REFERENCE.tolist(), inferred_slot_ids=[])
        sequence._record_g1_load_layout({'slot_numbering_status': 'final'}, 108.1)
        np.testing.assert_allclose(sequence._g1_load_layout, REFERENCE)
        loaded = sequence._new_tracker(2, phase='load')
        point, _ = loaded._infer_entry_s4(OBSERVED, ROI)
        np.testing.assert_allclose(point, [577.62, 345.28], atol=.01)

    def test_unverified_g1_pose_is_not_cached(self):
        tracker = SlotGroupTracker(REFERENCE, phase='load', group_id=1)
        sequence = RoiGroupSequence(tracker, ROI, load_numbering_calibration=self.matcher)
        tracker.last_time = 108.1
        tracker.initialization['numbering_reference_centers_xy'] = REFERENCE.tolist()
        sequence._record_g1_load_layout({'slot_numbering_status': 'final'}, 108.1)
        self.assertIsNone(sequence._g1_load_layout)

    def test_g1_uses_unload_rotation_before_its_loading_template_exists(self):
        config = self.matcher.config
        reference = np.array(config['anchor']['unload_centers_xy'])
        loaded = np.array(config['anchor']['load_centers_xy_before_correction'])[[1, 3, 0, 2]]
        observed = loaded[[2, 0, 3]]
        expected = observed[1]+observed[2]-observed[0]
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                tracker = SlotGroupTracker(phase='load', group_id=1)
                tracker._load_perspective_numbering = self.matcher
                tracker._load_numbering_reference = reference
                point, groups = tracker._infer_entry_s4(observed[list(order)], ROI)
                np.testing.assert_allclose(point, expected)
                self.assertEqual(groups, [1])
                info = tracker.initialization['three_point_template_prediction']
                self.assertEqual(info['reference'], 'same_group_unload')
                self.assertEqual(info['missing_slot_id'], 2)
                self.assertAlmostEqual(info['expected_clockwise_degrees'], self.matcher.rotation_degrees)

    def test_g1_without_unload_reference_does_not_revert_to_rightmost(self):
        tracker = SlotGroupTracker(phase='load', group_id=1)
        tracker._load_perspective_numbering = self.matcher
        point, groups = tracker._infer_entry_s4(OBSERVED, ROI)
        self.assertIsNone(point)
        self.assertEqual(groups, [])
        self.assertEqual(tracker.initialization['three_point_template_prediction']['reference'], 'same_group_unload')

    def test_video7_shape_advantage_is_not_cancelled_by_angle_score(self):
        point, info = self.matcher.missing_load_point(VIDEO7_REFERENCE, VIDEO7_SUPPORT, ROI)
        self.assertEqual(info['missing_slot_id'], 4)
        np.testing.assert_allclose(point, [603.16, 284.54])

    def test_short_ambiguity_retains_roles_and_updates_position_for_same_tracks(self):
        tracker = SlotGroupTracker(phase='load', group_id=2)
        tracker._load_perspective_numbering = self.matcher
        tracker._g1_load_numbering_reference = VIDEO7_REFERENCE
        tracker.last_time = 1.
        tracker._infer_entry_s4(VIDEO7_SUPPORT, ROI, support_track_ids=[490, 493, 516])
        current = VIDEO7_SUPPORT.copy()
        current[2, 0] = 555.
        self.assertIsNone(self.matcher.missing_load_point(VIDEO7_REFERENCE, current, ROI)[0])
        tracker.last_time = 1.1
        point, _ = tracker._infer_entry_s4(current[[2, 0, 1]], ROI, support_track_ids=[516, 490, 493])
        np.testing.assert_allclose(point, [606.68, 284.54])
        self.assertEqual(tracker.initialization['three_point_template_prediction']['selection'],
                         'previous_verified_roles')
        tracker.last_time = 2.
        self.assertIsNone(tracker._infer_entry_s4(current, ROI, support_track_ids=[490, 493, 516])[0])

    def test_changed_support_ids_cannot_inherit_prediction_identity(self):
        tracker = SlotGroupTracker(phase='load', group_id=2)
        tracker._load_perspective_numbering = self.matcher
        tracker._g1_load_numbering_reference = VIDEO7_REFERENCE
        tracker.last_time = 1.
        tracker._infer_entry_s4(VIDEO7_SUPPORT, ROI, support_track_ids=[490, 493, 516])
        current = VIDEO7_SUPPORT.copy()
        current[2, 0] = 555.
        tracker.last_time = 1.1
        self.assertIsNone(tracker._infer_entry_s4(current, ROI, support_track_ids=[490, 493, 999])[0])

    def test_rotated_fourth_real_slot_upgrades_shape_in_original_pose(self):
        original = REFERENCE.copy()
        initial = original.copy()
        initial[3] += [15., 6.]
        tracker = SlotGroupTracker(initial, phase='load', group_id=1)
        sequence = RoiGroupSequence(tracker, ROI, load_numbering_calibration=self.matcher)
        tracker.last_time = 0.
        tracker.initialization.update(perspective_load_numbering={'status': 'matched'},
            numbering_reference_centers_xy=initial.tolist(), inferred_slot_ids=[4])
        sequence._record_g1_load_layout({'slot_numbering_status': 'final'}, 0.)
        plane = self.matcher.project(original)
        radians = np.radians(34.)
        rotation = np.array([[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]])
        rotated = (plane-plane.mean(0)) @ rotation*.9+plane.mean(0)+[10., -20.]
        homogeneous = np.column_stack((rotated, np.ones(4))) @ np.linalg.inv(self.matcher.homography).T
        tracker.points = homogeneous[:, :2]/homogeneous[:, 2:]
        tracker.byte_track_ids = [1, 2, 3, 4]
        for slot in tracker.slots:
            slot.geometry_observed = True
            slot.geometry_inferred = False
            slot.observation = 'filled'
        for now in (1., 1.1, 1.2):
            tracker.last_time = now
            sequence._record_g1_load_layout({'slot_numbering_status': 'final'}, now)
        np.testing.assert_allclose(sequence._g1_load_layout, original, atol=1e-8)
        self.assertEqual(sequence._g1_load_layout_inferred_slot_ids, ())
        self.assertAlmostEqual(tracker.initialization['g1_load_template']['pose_alignment']['clockwise_degrees'], 34.)
        self.assertEqual(tracker.events, [])

    def test_video8_fixed_reference_selects_right_hand_missing_corner(self):
        reference = np.array([[422.38, 267.24], [522.39, 330.52], [492.86, 250.78], [590.73, 313.45]])
        observed = np.array([[426.01, 262.56], [499.91, 252.59], [520.50, 329.85]])
        point, info = self.matcher.missing_load_point(reference, observed, ROI)
        self.assertEqual(info['missing_slot_id'], 4)
        np.testing.assert_allclose(point, [594.40, 319.88], atol=.01)


if __name__ == '__main__':
    unittest.main()
