"""Physical slot identity across rotation, missing detections, and group handoff."""
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_perspective_numbering import SlotPerspectiveNumbering
from industrial_inspection.tools.slot_process_config import LOAD_NUMBERING_CALIBRATION_FILE


class PerspectiveNumberingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(LOAD_NUMBERING_CALIBRATION_FILE).read_text(encoding="utf-8"))
        self.matcher = SlotPerspectiveNumbering(self.config, (640, 480))
        self.unload = np.array(self.config["anchor"]["unload_centers_xy"])
        self.load = np.array(self.config["anchor"]["load_centers_xy_before_correction"])
        self.expected = np.array([1, 3, 0, 2])

    def test_user_anchor_resolves_quarter_turn_ambiguity(self):
        order, info = self.matcher.match(self.unload, self.load)
        np.testing.assert_array_equal(order, self.expected)
        self.assertEqual(order[3], 2)  # unload S4 is the previous load S3.
        self.assertAlmostEqual(info["clockwise_degrees"], -31.82615583)

    def test_permuting_detections_preserves_physical_slots(self):
        permutation = np.array([3, 0, 2, 1])
        order, _ = self.matcher.match(self.unload, self.load[permutation])
        np.testing.assert_array_equal(self.load[permutation][order], self.load[self.expected])

    def test_three_real_points_match_without_using_prediction_as_angle_evidence(self):
        shifted = self.load.copy()
        shifted[3] += [5, 4]
        order, info = self.matcher.match(self.unload, shifted, inferred_indices={3})
        baseline, _ = self.matcher.match(self.unload, self.load, inferred_indices={3})
        np.testing.assert_array_equal(order, baseline)
        self.assertEqual(info["measured_support"], 3)
        _, baseline_info = self.matcher.match(self.unload, self.load, inferred_indices={3})
        self.assertAlmostEqual(info["clockwise_degrees"], baseline_info["clockwise_degrees"])

    def test_two_real_points_cannot_formalize_slot_numbers(self):
        order, info = self.matcher.match(self.unload, self.load, inferred_indices={2, 3})
        self.assertIsNone(order)
        self.assertEqual(info["status"], "insufficient_measured_support")

    def test_wrong_rotation_cannot_fall_back_to_a_quarter_turn(self):
        config = dict(self.config, unload_to_load_clockwise_degrees=10.)
        order, info = SlotPerspectiveNumbering(config, (640, 480)).match(self.unload, self.load)
        self.assertIsNone(order)
        self.assertEqual(info["status"], "rotation_mismatch")

    def test_subsequent_groups_match_g1_load_with_translation(self):
        reference = self.load[self.expected]
        for current in (self.load, reference, self.load[[2, 0, 3, 1]]):
            order, _ = self.matcher.match(reference, current+[12, -3], same_load_pose=True)
            np.testing.assert_allclose(current[order], reference)

    def test_resolution_scaling_preserves_anchor(self):
        matcher = SlotPerspectiveNumbering(self.config, (1280, 960))
        order, _ = matcher.match(self.unload*2, self.load*2)
        np.testing.assert_array_equal(order, self.expected)

    def test_all_source_names_use_shared_calibration(self):
        for source in ("camera_B_20260924_062208.avi", "camera_B_20260924_021033.avi", "new_video.mp4"):
            matcher = SlotPerspectiveNumbering.for_source(LOAD_NUMBERING_CALIBRATION_FILE, source, (640, 480))
            self.assertIsNotNone(matcher)
            order, _ = matcher.match(self.unload, self.load)
            np.testing.assert_array_equal(order, self.expected)

    def test_calibration_can_be_disabled_explicitly(self):
        self.assertIsNone(SlotPerspectiveNumbering.for_source(
            None, "camera_B_20260924_062208.avi", (640, 480)))

    def test_video_runs_enable_calibration_and_keep_independent_templates(self):
        from industrial_inspection.tools.predict_slot_groups import make_sequence
        polygon = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
        first = make_sequence(polygon, (640, 480), args=SimpleNamespace(source=Path("first.avi")))
        second = make_sequence(polygon, (640, 480), args=SimpleNamespace(source=Path("second.avi")))
        self.assertIsNotNone(first.load_numbering_calibration)
        self.assertIsNotNone(second.load_numbering_calibration)
        first._g1_load_layout = self.load.copy()
        first._unload_layouts[1] = [self.unload.copy()]
        self.assertIsNone(second._g1_load_layout)
        self.assertEqual(second._unload_layouts, {})

    def test_calibration_collects_unload_reference_when_previous_s4_prediction_is_disabled(self):
        tracker = SlotGroupTracker(self.unload, phase="unload", group_id=1)
        tracker.byte_track_ids = [1, 2, 3, 4]
        sequence = RoiGroupSequence(tracker, [[0, 0], [640, 0], [640, 480], [0, 480]],
                                    use_previous_s4_layouts=False, load_numbering_calibration=self.matcher)
        for slot in tracker.slots:
            slot.geometry_observed = True
            slot.observation = "empty"
        for now in range(3):
            tracker.last_time = now
            sequence._record_unload_layout({"slot_numbering_status": "final"}, now)
        loaded = sequence._new_tracker(1, phase="load")
        np.testing.assert_allclose(loaded._load_numbering_reference, self.unload)
        np.testing.assert_array_equal(loaded._entry_numbering_order(self.load), self.expected)

    def test_layout_translation_and_initial_rotation_do_not_change_relative_rotation(self):
        def image_points(top_down):
            homogeneous = np.column_stack((top_down, np.ones(len(top_down)))) @ np.linalg.inv(self.matcher.homography).T
            return homogeneous[:, :2]/homogeneous[:, 2:]
        reference = np.array([[0, 0], [100, 100], [100, 0], [0, 100]], dtype=float)
        relative_angle = np.radians(self.matcher.rotation_degrees)
        relative = np.array([[np.cos(relative_angle), np.sin(relative_angle)],
                             [-np.sin(relative_angle), np.cos(relative_angle)]])
        for angle in (0., 45., 120.):
            radians = np.radians(angle)
            initial = np.array([[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]])
            unloaded = reference @ initial + [120, 230]
            loaded = reference @ initial @ relative + [240, 350]
            order, _ = self.matcher.match(image_points(unloaded), image_points(loaded))
            np.testing.assert_array_equal(order, np.arange(4))

    def test_calibrated_numbering_waits_instead_of_using_machine_anchor(self):
        tracker = SlotGroupTracker(phase="load", group_id=1, machine_center=(637, 266))
        tracker._load_perspective_numbering = self.matcher
        self.assertIsNone(tracker._entry_numbering_order(self.load))
        self.assertEqual(tracker.initialization["perspective_load_numbering"]["status"],
                         "waiting_for_reference")
        tracker._load_numbering_reference = self.unload
        np.testing.assert_array_equal(tracker._entry_numbering_order(self.load), self.expected)

    def test_three_real_slots_create_g1_template_for_three_point_matching_then_upgrade(self):
        tracker = SlotGroupTracker(self.load[self.expected], phase="load", group_id=1)
        tracker.byte_track_ids = [336, 338, 334, 333]
        tracker.initialization["perspective_load_numbering"] = {"status": "matched"}
        sequence = RoiGroupSequence(tracker, [[0, 0], [640, 0], [640, 480], [0, 480]],
                                    load_numbering_calibration=self.matcher)
        snapshot = {"slot_numbering_status": "final"}
        for slot in tracker.slots:
            slot.geometry_observed = True
            slot.observation = "empty"
        tracker.slots[1].geometry_inferred = True
        tracker.slots[1].geometry_observed = False
        tracker.slots[1].observation = "unknown"
        tracker.byte_track_ids[1] = None
        tracker.initialization["inferred_slot_ids"] = [2]
        # Formal initialization records the inferred slot even if the per-frame set is empty.
        tracker._predicted_indices = set()
        for now in range(3):
            tracker.last_time = now
            sequence._record_g1_load_layout(snapshot, now)
        np.testing.assert_allclose(sequence._g1_load_layout, tracker.points, atol=1e-9)
        self.assertEqual(sequence._g1_load_layout_inferred_slot_ids, (2,))
        self.assertEqual(tracker.slots[1].confirmed, "unknown")
        self.assertEqual(tracker.events, [])
        second = sequence._new_tracker(2, phase="load")
        second._entry_inferred_indices = {3}
        np.testing.assert_array_equal(second._entry_numbering_order(self.load), self.expected)
        diagnostic = second.initialization["perspective_load_numbering"]
        self.assertEqual(diagnostic["reference"], "g1_load")
        self.assertEqual(diagnostic["reference_measured_support"], 3)
        self.assertEqual(diagnostic["reference_inferred_slot_ids"], [2])
        self.assertEqual(diagnostic["measured_support"], 3)
        tracker.slots[1].geometry_inferred = False
        tracker.slots[1].geometry_observed = True
        tracker.slots[1].observation = "empty"
        tracker.byte_track_ids[1] = 338
        for now in range(3, 6):
            tracker.last_time = now
            sequence._record_g1_load_layout(snapshot, now)
        np.testing.assert_allclose(sequence._g1_load_layout, tracker.points, atol=1e-9)
        self.assertEqual(sequence._g1_load_layout_inferred_slot_ids, ())
        third = sequence._new_tracker(3, phase="load")
        np.testing.assert_array_equal(third._entry_numbering_order(self.load), self.expected)
        self.assertEqual(third.initialization["perspective_load_numbering"]["reference_measured_support"], 4)
        self.assertIsNone(sequence._new_tracker(2, phase="unload")._load_perspective_numbering)

    def test_g1_template_rejects_two_real_slots_and_a_held_box(self):
        tracker = SlotGroupTracker(self.load[self.expected], phase="load", group_id=1)
        tracker.byte_track_ids = [336, None, 334, None]
        tracker.initialization.update(perspective_load_numbering={"status": "matched"},
                                      inferred_slot_ids=[2, 4])
        tracker._predicted_indices = {1, 3}
        sequence = RoiGroupSequence(tracker, [[0, 0], [640, 0], [640, 480], [0, 480]],
                                    load_numbering_calibration=self.matcher)
        for i, slot in enumerate(tracker.slots):
            slot.geometry_observed = i in (0, 2)
            slot.geometry_inferred = i in (1, 3)
            slot.observation = "empty" if slot.geometry_observed else "unknown"
        for now in range(3):
            tracker.last_time = now
            sequence._record_g1_load_layout({"slot_numbering_status": "final"}, now)
        self.assertIsNone(sequence._g1_load_layout)
        tracker.slots[3].geometry_observed = True
        tracker.slots[3].geometry_inferred = False
        tracker.slots[3].observation = "empty"
        tracker.byte_track_ids[3] = 333
        # A last-seen box alone cannot substitute for a three-point geometric prediction.
        tracker.slots[1].geometry_inferred = False
        for now in range(3, 6):
            tracker.last_time = now
            sequence._record_g1_load_layout({"slot_numbering_status": "final"}, now)
        self.assertIsNone(sequence._g1_load_layout)


if __name__ == "__main__":
    unittest.main()
