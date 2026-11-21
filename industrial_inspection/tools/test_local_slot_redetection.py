"""验证局部补检坐标、槽位归属、遮挡及补检结果进入实际取放事件保存流程。"""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from industrial_inspection.tools import predict_slot_groups as predict
from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_local_redetection import LocalRedetectionPolicy, recover_load_preview_support
from industrial_inspection.tools.slot_two_point_contact import _contact_areas, two_point_slot_layout
from industrial_inspection.tools.test_group_prediction_pipeline import SyntheticModel
from industrial_inspection.tools.test_roi_group_sequence import POINTS, ROI, boxes
from industrial_inspection.tools.test_slot_group_tracker import detection


class TwoPointContactTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.visible = [
            {"class_name": "wafer_slot_filled", "confidence": .95,
             "track_id": i, "xyxy": [100, y-20, 140, y+20]}
            for i, y in enumerate((140, 200), start=1)]

    def test_shared_extension_respects_roi_direction_and_multiplier(self):
        tracker = SlotGroupTracker()
        measured = tracker._candidates(self.visible)
        measured[1]["xyxy"] = [98, 180, 142, 220]
        direction = np.array([.8, .6])
        for multiplier in (1.5, 2.):
            layout, extension = two_point_slot_layout(measured, direction, multiplier, ROI)
            self.assertAlmostEqual(extension, 26.25 * multiplier)
            np.testing.assert_allclose(layout[2:] - layout[:2],
                                       np.tile(extension * direction, (2, 1)))

    def test_rejects_diagonal_and_disproportionate_input(self):
        tracker = SlotGroupTracker()
        different = {**self.visible[1], "xyxy": [110, 180, 130, 220]}
        diagonal = {**self.visible[1], "xyxy": [170, 180, 210, 220]}
        for candidate in (different, diagonal):
            layout = two_point_slot_layout(tracker._candidates([self.visible[0], candidate]),
                                           np.array([1., 0.]), 1.5, ROI)
            self.assertIsNone(layout)

    def test_load_reference_verifies_rightward_projection_before_numbering(self):
        reference = np.array([[168., 150.], [204., 159.], [100., 100.], [136., 109.]])
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1)
        tracker._load_numbering_reference = reference
        roi = np.float32([[90, 82], [222, 115], [222, 175], [90, 142]])
        sequence = RoiGroupSequence(tracker, roi, two_point_extension_multiplier=1.8)
        detections = [detection(i, "filled", centers=reference+[4, 3], track_id=i) for i in (1, 3)]
        for now in (0., .05, .1, .15):
            result = sequence.update(detections, now)
        contacts = result["two_point_hand_contacts"]
        self.assertEqual([c["reference_slot_id"] for c in contacts], [2, 4])
        self.assertEqual([c["reference_anchor_slot_ids"] for c in contacts], [[1, 3], [1, 3]])
        self.assertEqual([c["slot_id"] for c in contacts], [None, None])
        np.testing.assert_allclose([c["center_xy"] for c in contacts], reference[[1, 3]]+[4, 3])
        np.testing.assert_allclose(np.asarray([c["center_xy"] for c in contacts])-reference[[0, 2]]-[4, 3],
                                   [[36., 9.], [36., 9.]])

    def test_load_preview_follows_clockwise_when_both_predictions_are_nearer_s4(self):
        reference = np.array([[65., 160.], [105., 160.], [100., 100.], [140., 100.]])
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1)
        tracker._load_numbering_reference = reference
        roi = np.float32([[35, 70], [270, 70], [270, 200], [35, 200]])
        sequence = RoiGroupSequence(tracker, roi, two_point_extension_multiplier=6.5)
        detections = [detection(i, "filled", centers=reference, track_id=i) for i in (1, 3)]

        for now in (0., .05, .1, .15):
            result = sequence.update(detections, now)

        contacts = result["two_point_hand_contacts"]
        self.assertEqual([c["reference_slot_id"] for c in contacts], [2, 4])
        centers = np.array([c["center_xy"] for c in contacts])
        np.testing.assert_allclose(centers, [[195., 160.], [230., 100.]])
        nearest = np.argmin(np.linalg.norm(centers[:, None]-reference[[1, 3]], axis=2), axis=1)
        np.testing.assert_array_equal(nearest, [1, 1])

    def test_plausible_load_preview_is_unnumbered_with_unverified_supports(self):
        reference = np.array([[100., 100.], [145., 165.], [190., 110.], [235., 175.]])
        observed = reference.copy()
        observed[[0, 2]] = [[130., 180.], [130., 120.]]
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1)
        tracker._load_numbering_reference = reference
        roi = np.float32([[70, 70], [240, 70], [240, 240], [70, 240]])
        sequence = RoiGroupSequence(tracker, roi, two_point_extension_multiplier=1.8)
        detections = [detection(i, "filled", centers=observed, track_id=i) for i in (1, 3)]

        for now in (0., .05, .1, .15):
            result = sequence.update(detections, now)

        contacts = result["two_point_hand_contacts"]
        self.assertEqual([c["reference_slot_id"] for c in contacts], [None, None])
        self.assertEqual([c["slot_id"] for c in contacts], [None, None])

    def test_load_reference_rejects_horizontal_extension_for_opposite_side_anchors(self):
        reference = np.array([[100., 200.], [200., 200.], [100., 100.], [200., 100.]])
        tracker = SlotGroupTracker(reference, phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1)
        tracker._load_numbering_reference = reference
        tracker.byte_track_ids = [1, 2, 3, 4]
        roi = np.float32([[5, 5], [330, 5], [330, 325], [5, 325]])
        sequence = RoiGroupSequence(tracker, roi, two_point_extension_multiplier=1.8)
        detections = [detection(i, "filled", centers=reference, track_id=i) for i in (2, 3)]
        for now in (0., .05, .1, .15, .2):
            result = sequence.update(detections, now)
            self.assertFalse(result["two_point_hand_contacts"])

    def test_load_reference_uses_formal_roles_and_only_touches_inferred_slot(self):
        reference = np.array([[168., 135.], [204., 135.], [100., 100.], [136., 100.]])
        tracker = SlotGroupTracker(reference, phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1, load_hand_contact_seconds=0.)
        tracker._load_numbering_reference = reference
        tracker.initialization["inferred_slot_ids"] = [2]
        tracker.byte_track_ids = [1, 2, 3, 4]
        roi = np.float32([[70, 70], [240, 70], [240, 240], [70, 240]])
        sequence = RoiGroupSequence(tracker, roi, two_point_extension_multiplier=1.8)
        detections = [detection(i, "filled", centers=reference, track_id=i) for i in (1, 3)]
        hand = {"class_name": "hand", "confidence": .9, "xyxy": [125, 89, 147, 111]}
        for now in (0., .05, .1, .15):
            result = sequence.update(detections+[hand], now)
        contacts = result["two_point_hand_contacts"]
        self.assertEqual([c["slot_id"] for c in contacts], [2, 4])
        self.assertEqual([c["status"] for c in contacts], ["prediction", "prediction"])
        hand["xyxy"] = [193, 124, 215, 146]
        for now in (.2, .25):
            result = sequence.update(detections+[hand], now)
        self.assertEqual([c["status"] for c in result["two_point_hand_contacts"]],
                         ["contact", "prediction"])

    def test_split_overlap_and_suspend_large_overlap(self):
        centers = np.array([[180., 140.], [180., 169.]])
        boxes = np.hstack((centers-20., centers+20.))
        scores = _contact_areas(boxes, centers, [{"xyxy": [160, 155, 200, 159]}])
        self.assertAlmostEqual(scores[0], 0.)
        self.assertGreater(scores[1], 0.)
        centers[1, 1] = 158.
        self.assertIsNone(_contact_areas(np.hstack((centers-20., centers+20.)), centers, []))

    def test_overlay_marks_both_prediction_centers_over_the_boxes(self):
        contacts = [
            {"center_xy": [180.2, 140.1], "box_xyxy": [150, 120, 210, 160],
             "status": "prediction", "slot_id": None},
            {"center_xy": [204.3, 161.2], "box_xyxy": [174, 141, 234, 181],
             "status": "contact", "slot_id": None, "reference_slot_id": 2, "held": True},
        ]
        with patch.object(predict.cv2, "putText", wraps=cv2.putText) as draw_text:
            predict.draw_two_point_hand_contacts(self.frame, contacts)
            self.assertEqual([call.args[1] for call in draw_text.call_args_list],
                             ["2PT PRED", "2PT ~S2 HOLD TOUCH"])
        for x, y in ((180, 140), (204, 161)):
            self.assertGreater(int(self.frame[y, x].min()), 235)
        blue, green, red = map(int, self.frame[120, 165])
        self.assertGreater(min(green, red), blue+120)
        blue, green, red = map(int, self.frame[181, 221])
        self.assertGreater(min(blue, red), green+100)

    def test_held_boxes_accumulate_contact_when_a_support_is_occluded_or_missing(self):
        tracker = SlotGroupTracker(phase="load", slot2_below_slot3=True,
            initialization_seconds=.1, geometry_initialization_seconds=.1,
            minimum_frames=2, confirm_seconds=0., load_hand_contact_seconds=0.)
        sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.)
        for now in (0., .1, .2, .3):
            measured = sequence.update(self.visible, now)["two_point_hand_contacts"]
        hand = {"class_name": "hand", "confidence": .9, "xyxy": [117, 120, 198, 156]}
        first = sequence.update(self.visible[1:] + [hand], .4)
        second = sequence.update(self.visible + [hand], .5)
        for frame in (first, second):
            contacts = frame["two_point_hand_contacts"]
            self.assertEqual([c["box_xyxy"] for c in contacts], [c["box_xyxy"] for c in measured])
            self.assertTrue(all(c["held"] for c in contacts))
            self.assertGreater(contacts[0]["coverage"], .15)
            self.assertFalse(frame["new_events"])
        self.assertEqual(first["two_point_hand_contacts"][0]["status"], "collecting")
        self.assertEqual(second["two_point_hand_contacts"][0]["status"], "contact")
        released = sequence.update(self.visible, .6)["two_point_hand_contacts"]
        self.assertTrue(released[0]["held"])
        self.assertEqual(released[0]["status"], "waiting_visual")
        right = [{**self.visible[0], "track_id": i, "xyxy": [160, y-20, 200, y+20]}
                 for i, y in ((3, 140), (4, 200))]
        for now in (.7, .8, .9, 1.):
            sequence.update(self.visible+right, now)
        self.assertEqual(tracker.initialization["inferred_slot_ids"], [])
        self.assertFalse(sequence.all_events)

    def test_hand_center_must_be_inside_roi_for_contact_and_prediction_hold(self):
        roi = np.array([[0, 0], [300, 0], [300, 210], [0, 210]], dtype=np.float32)
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1, load_hand_contact_seconds=0.)
        sequence = RoiGroupSequence(tracker, roi, two_point_extension_multiplier=3.)
        for now in (0., .1, .2, .3):
            sequence.update(self.visible, now)
        outside_hand = {"class_name": "hand", "confidence": .99, "xyxy": [164, 187, 196, 253]}
        for now in (.4, .5):
            contacts = sequence.update(self.visible+[outside_hand], now)["two_point_hand_contacts"]
            self.assertEqual([c["coverage"] for c in contacts], [0., 0.])
            self.assertTrue(all(c["status"] == "prediction" for c in contacts))
        outside_covering_anchor = {**outside_hand, "xyxy": [117, 190, 198, 255]}
        result = sequence.update(self.visible[:1]+[outside_covering_anchor], .6)
        self.assertTrue(all(c["held"] for c in result["two_point_hand_contacts"]))
        self.assertEqual([c["coverage"] for c in result["two_point_hand_contacts"]], [0., 0.])

        # A real hand can extend past the lower edge and still pass.
        inside_covering_anchor = {**outside_hand, "xyxy": [117, 185, 198, 223]}
        first = sequence.update(self.visible[:1]+[inside_covering_anchor], .7)
        second = sequence.update(self.visible[:1]+[inside_covering_anchor], .8)
        for result in (first, second):
            contacts = result["two_point_hand_contacts"]
            self.assertTrue(all(c["held"] for c in contacts))
            self.assertGreater(contacts[1]["coverage"], .15)
        self.assertEqual(second["two_point_hand_contacts"][1]["status"], "contact")

    def test_held_boxes_expire_or_clear_upon_motion_pause(self):
        hand = {"class_name": "hand", "confidence": .9, "xyxy": [117, 120, 198, 156]}
        for stop_for_motion in (False, True):
            with self.subTest(stop_for_motion=stop_for_motion):
                tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                           geometry_initialization_seconds=.1)
                sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.,
                                            two_point_hand_hold_seconds=.25)
                for now in (0., .1, .2, .3):
                    sequence.update(self.visible, now)
                held = sequence.update([hand], .4)["two_point_hand_contacts"]
                self.assertTrue(held[0]["held"])
                self.assertAlmostEqual(held[0]["prediction_age_seconds"], .1)
                if stop_for_motion:
                    result = sequence.update(self.visible[1:] + [hand], .45, observations_enabled=False)
                    self.assertFalse(result["two_point_hand_contacts"])
                    check_at = .5
                else:
                    check_at = .56
                result = sequence.update([hand], check_at)
                self.assertFalse(result["two_point_hand_contacts"])
                self.assertFalse(sequence.all_events)

    def test_contact_continues_after_missing_support_and_single_anchor_jitter(self):
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1, load_hand_contact_seconds=0.)
        sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.)
        for now in (0., .1, .2, .3):
            sequence.update(self.visible, now)
        hand_away_from_supports = {"class_name": "hand", "confidence": .9, "xyxy": [164, 124, 196, 156]}
        result = sequence.update(self.visible[1:] + [hand_away_from_supports], .4)
        self.assertEqual(result["two_point_hand_contacts"][0]["status"], "collecting")
        moved = {**self.visible[1], "xyxy": [112, 180, 152, 220]}
        result = sequence.update([moved, hand_away_from_supports], .5)
        self.assertTrue(all(c["held"] for c in result["two_point_hand_contacts"]))
        self.assertEqual(result["two_point_hand_contacts"][0]["status"], "contact")

    def test_old_pair_serves_contact_until_both_new_positions_stabilize(self):
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1, load_hand_contact_seconds=0.)
        sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.)
        for now in (0., .1, .2, .3):
            contacts = sequence.update(self.visible, now)["two_point_hand_contacts"]
        old_centers = [c["center_xy"] for c in contacts]
        shifted = [dict(obj, xyxy=(np.array(obj["xyxy"])+[dx, 0, dx, 0]).tolist())
                   for obj, dx in zip(self.visible, (10, -10))]
        hand = {"class_name": "hand", "confidence": .9, "xyxy": [164, 120, 198, 156]}
        for now, status in ((.4, "collecting"), (.45, "contact")):
            contacts = sequence.update(shifted+[hand], now)["two_point_hand_contacts"]
            self.assertEqual([c["center_xy"] for c in contacts], old_centers)
            self.assertTrue(all(c["held"] for c in contacts))
            self.assertEqual(contacts[0]["status"], status)
        contacts = sequence.update(shifted+[hand], .51)["two_point_hand_contacts"]
        self.assertFalse(any(c["held"] for c in contacts))
        self.assertEqual([c["center_xy"] for c in contacts], [[190., 140.], [170., 200.]])
        self.assertEqual(contacts[0]["status"], "contact")
        self.assertAlmostEqual(contacts[0]["contact_since"], .4)

    def test_coherent_motion_for_consecutive_frames_clears_old_boxes(self):
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1)
        sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.)
        for now in (0., .1, .2, .3):
            sequence.update(self.visible, now)
        for dx, now in ((10, .4), (20, .45)):
            shifted = [dict(obj, xyxy=(np.array(obj["xyxy"])+[dx, 0, dx, 0]).tolist())
                       for obj in self.visible]
            contacts = sequence.update(shifted, now)["two_point_hand_contacts"]
            if dx == 10:
                self.assertTrue(contacts and all(c["held"] for c in contacts))
            else:
                self.assertFalse(contacts)

    def test_one_trusted_anchor_can_extend_hold_until_it_is_lost(self):
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1)
        sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.,
                                    two_point_hand_hold_seconds=.25)
        for now in (0., .1, .2, .3):
            sequence.update(self.visible, now)
        for now in (.4, .55, .7):
            contacts = sequence.update(self.visible[1:], now)["two_point_hand_contacts"]
            self.assertTrue(contacts and all(c["held"] for c in contacts))
            self.assertEqual(contacts[0]["prediction_support_age_seconds"], 0.)
        self.assertGreater(contacts[0]["prediction_age_seconds"], .25)
        self.assertFalse(sequence.update([], .96)["two_point_hand_contacts"])

    def test_contact_waits_for_vision_and_only_then_records_one_operation(self):
        for phase, target in (("load", "filled"), ("unload", "empty")):
            for entering in (False, True):
                with self.subTest(phase=phase, entering=entering):
                    tracker = SlotGroupTracker(phase=phase, slot2_below_slot3=True,
                        initialization_seconds=.1, geometry_initialization_seconds=.1,
                        minimum_frames=2, confirm_seconds=0., load_hand_contact_seconds=0.,
                        load_hand_contact_min_frames=2)
                    sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.)
                    if entering:
                        tracker.start_incremental_entry(40.)
                        sequence.state = "entering"
                    hand = {"class_name": "hand", "confidence": .9, "xyxy": [164, 124, 196, 156]}
                    right = [{"class_name": f"wafer_slot_{target}", "confidence": .95,
                              "track_id": track_id, "xyxy": [160, y-20, 200, y+20]}
                             for track_id, y in ((3, 140), (4, 200))]
                    for tick in range(11):
                        objects = self.visible + ([hand] if tick in (4, 5) else right if tick >= 7 else [])
                        result = sequence.update(objects, tick / 10)
                        if tick < 7:
                            self.assertFalse(result["new_events"])
                        if tick == 5:
                            self.assertEqual([c["status"] for c in result["two_point_hand_contacts"]],
                                             ["contact", "prediction"])
                        if tick == 6:
                            self.assertEqual(result["two_point_hand_contacts"][0]["status"], "waiting_visual")
                    if phase == "load":
                        self.assertEqual(tracker.initialization["inferred_slot_ids"], [])
                        self.assertFalse(sequence.all_events)
                        continue
                    event, = sequence.all_events
                    self.assertEqual((event["slot_id"], event["action"], event["to"]), (3, phase, target))
                    self.assertEqual(event["reason"], "visual_after_two_point_hand_contact")
                    self.assertAlmostEqual(event["hand_contact_since"], .4)
                    self.assertAlmostEqual(event["time_upper"], .6)
                    self.assertAlmostEqual(event["visual_time_upper"], .7)

    def test_ambiguous_hand_never_arms_both_predictions(self):
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1)
        sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.)
        hand = {"class_name": "hand", "confidence": .9, "xyxy": [164, 120, 196, 220]}
        for tick in range(7):
            result = sequence.update(self.visible + ([hand] if tick in (4, 5) else []), tick / 10)
            if tick == 5:
                self.assertEqual([c["status"] for c in result["two_point_hand_contacts"]],
                                 ["ambiguous", "ambiguous"])
        self.assertFalse(tracker.two_point_hand_contact.pending)
        self.assertFalse(sequence.all_events)

    def test_unique_center_resolves_overlap_and_both_centers_pause_contact(self):
        tracker = SlotGroupTracker(phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1,
                                   load_hand_contact_seconds=0., load_hand_contact_min_frames=2)
        sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.)
        for now in (0., .1, .2, .3):
            sequence.update(self.visible, now)

        hand = {"class_name": "hand", "confidence": .9, "xyxy": [164, 139, 196, 198]}
        first = sequence.update(self.visible + [hand], .4)["two_point_hand_contacts"]
        second = sequence.update(self.visible + [hand], .5)["two_point_hand_contacts"]
        self.assertTrue(all(c["coverage"] >= .15 for c in second))
        self.assertEqual([c["center_covered"] for c in second], [True, False])
        self.assertEqual(first[0]["status"], "collecting")
        self.assertEqual(second[0]["status"], "contact")

        hand["xyxy"] = [164, 120, 196, 201]
        paused = sequence.update(self.visible + [hand], .6)["two_point_hand_contacts"]
        self.assertEqual([c["status"] for c in paused], ["ambiguous", "ambiguous"])
        self.assertEqual([c["center_covered"] for c in paused], [True, True])
        contact = tracker.two_point_hand_contact.pending[0]
        self.assertEqual(contact.frames, 2)
        self.assertIsNone(contact.released_at)
        self.assertTrue(contact.requires_visual_load)

        released = sequence.update(self.visible, .7)["two_point_hand_contacts"]
        self.assertEqual(released[0]["status"], "waiting_visual")
        self.assertEqual(released[0]["released_at"], .7)

    def _enter_load_with_two_prediction_contacts(self, *, contact_s4=True, empty_baseline=True,
                                                 dual_center_pause=False, with_reference=False):
        points = np.array([[100, 180], [160, 180], [130, 140], [190, 140]], dtype=float)
        slots = [{"class_name": "wafer_slot_empty", "confidence": .95, "track_id": i,
                  "xyxy": [x-20, y-20, x+20, y+20]} for i, (x, y) in enumerate(points, start=1)]
        tracker = SlotGroupTracker(group_id=4, phase="load", slot2_below_slot3=True,
            allow_three_point_initialization=True, initialization_seconds=.1,
            minimum_frames=2, confirm_seconds=0., load_hand_contact_seconds=0.)
        if with_reference:
            tracker._load_numbering_reference = points.copy()
        sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.,
                                    two_point_hand_hold_seconds=.15)
        previous = SlotGroupTracker(points, group_id=4)
        if empty_baseline:
            previous.slots[3].confirmed = "empty"
            previous.slots[3].evidence_time = -1.
            previous.slots[3].confirmation_source = "yolo"
        sequence.closed_groups.append({"phase": "unload", "group_id": 4, "snapshot": previous.snapshot()})
        tracker.start_incremental_entry(32.)
        sequence.state = "entering"
        anchors = [slots[0], slots[2]]
        hands = [{"class_name": "hand", "confidence": .95,
                  "xyxy": box} for box in ([170, 122, 208, 158], [150, 122, 208, 173])]
        if not contact_s4:
            hands[0]["xyxy"] = hands[1]["xyxy"] = [141, 161, 178, 196]
        for now in (0., .1, .2, .3):
            sequence.update(anchors, now)
        sequence.update(anchors + hands[:1], .4)
        touch = sequence.update(anchors + hands[1:], .5)
        self.assertEqual(sum(c["status"] == "contact" for c in touch["two_point_hand_contacts"]), 1)
        if contact_s4:
            self.assertTrue(all(c["coverage"] > .15 for c in touch["two_point_hand_contacts"]))
        if dual_center_pause:
            hand = {"class_name": "hand", "confidence": .95, "xyxy": [147, 119, 212, 200]}
            contacts = sequence.update(anchors + [hand], .55)["two_point_hand_contacts"]
            self.assertEqual([c["status"] for c in contacts], ["ambiguous", "ambiguous"])
        sequence.update(anchors, .6)
        sequence.update([], .7)
        expired = sequence.update([], .8)
        self.assertFalse(expired["two_point_hand_contacts"])
        for now in (.9, 1., 1.1):
            mapped = sequence.update(slots[:3], now)
        self.assertEqual(mapped["slot_numbering_status"], "final")
        return sequence, mapped, slots

    def test_late_mapping_recovers_s4_contact_and_locks_its_filled_state(self):
        sequence, mapped, slots = self._enter_load_with_two_prediction_contacts()
        event, = mapped["new_events"]
        self.assertEqual((event["slot_id"], event["reason"]), (4, "two_point_hand_contact_inferred_load"))
        self.assertEqual((event["hand_contact_since"], event["time_upper"]), (.4, .6))
        self.assertTrue(event["inferred"])
        self.assertEqual(mapped["confirmed_event_sequence"], [4])
        self.assertTrue(mapped["slots"][3]["load_state_locked"])
        first = sequence.update(slots, 1.2)
        self.assertEqual(first["confirmed_event_sequence"], [4])
        second = sequence.update(slots, 1.3)
        self.assertEqual(second["confirmed_event_sequence"], [4])
        self.assertEqual(second["slots"][3]["last_confirmed_state"], "filled")
        self.assertEqual(second["new_events"], [])
        self.assertEqual(sequence.all_events, [event])
        self.assertFalse(second["new_group_events"])

    def test_late_mapping_does_not_infer_s4_from_s2_contact_or_without_empty_baseline(self):
        for contact_s4, empty_baseline in ((False, True), (True, False)):
            with self.subTest(contact_s4=contact_s4, empty_baseline=empty_baseline):
                sequence, mapped, _ = self._enter_load_with_two_prediction_contacts(
                    contact_s4=contact_s4, empty_baseline=empty_baseline)
                self.assertFalse(mapped["new_events"])
                self.assertFalse(sequence.all_events)

    def test_dual_center_contact_waits_for_visual_fill_after_numbering(self):
        self._assert_visual_load_after_entry_contact(dual_center_pause=True)

    def test_clockwise_preview_contact_waits_for_visual_fill_after_numbering(self):
        self._assert_visual_load_after_entry_contact(with_reference=True)

    def _assert_visual_load_after_entry_contact(self, **kwargs):
        sequence, mapped, slots = self._enter_load_with_two_prediction_contacts(**kwargs)
        self.assertEqual(sequence.tracker.initialization["inferred_slot_ids"], [4])
        self.assertFalse(mapped["new_events"])
        self.assertEqual(sequence.tracker.slots[3].prediction_contact.entry_preview_source,
                         "two_point_preview")

        filled = dict(slots[3], class_name="wafer_slot_filled")
        for now in (1.2, 1.3, 1.4, 1.5):
            sequence.update(slots[:3] + [filled], now)
        event, = sequence.all_events
        self.assertEqual((event["slot_id"], event["from"], event["to"]), (4, "empty", "filled"))
        self.assertEqual(event["reason"], "visual_after_two_point_preview_contact")
        self.assertFalse(event.get("inferred"))
        self.assertEqual((event["time_upper"], event["visual_time_upper"]), (.6, 1.2))

    def test_formal_contact_does_not_infer_state_without_vision_or_leak_to_next_group(self):
        points = np.array([[120, 140], [120, 200], [180, 140], [180, 200]], dtype=float)
        tracker = SlotGroupTracker(points, phase="load", initialization_seconds=.1,
                                   geometry_initialization_seconds=.1, minimum_frames=2,
                                   confirm_seconds=0., load_hand_contact_seconds=0., load_hand_contact_min_frames=2,
                                   lost_seconds=1.)
        tracker.initialization["inferred_slot_ids"] = [3]
        sequence = RoiGroupSequence(tracker, ROI, two_point_extension_multiplier=3.)
        hand = {"class_name": "hand", "confidence": .9, "xyxy": [164, 124, 196, 156]}
        for tick in range(13):
            result = sequence.update(self.visible + ([hand] if tick in (4, 5) else []), tick / 10)
        self.assertEqual(result["slots"][2]["last_confirmed_state"], "unknown")
        self.assertEqual(result["two_point_hand_contacts"][0]["status"], "waiting_visual")
        self.assertEqual(result["two_point_hand_contacts"][0]["slot_id"], 3)
        self.assertFalse(sequence.all_events)

        new_tracker = sequence._new_tracker(2)
        self.assertFalse(new_tracker.two_point_hand_contact.pending)
        self.assertTrue(tracker.two_point_hand_contact.pending)

        right = {"class_name": "wafer_slot_filled", "confidence": .95,
                 "track_id": 3, "xyxy": [160, 120, 200, 160]}
        sequence.update(self.visible+[right], 1.3)
        result = sequence.update(self.visible+[right], 1.4)
        event, = result["new_events"]
        self.assertEqual(event["slot_id"], 3)
        self.assertEqual(event["reason"], "visual_after_two_point_hand_contact")
        self.assertFalse(sequence.update(self.visible+[right], 1.5)["new_events"])

    def test_main_pipeline_uses_full_frames_without_two_point_crop_inference(self):
        hand = {"class_name": "hand", "confidence": .9, "xyxy": [164, 124, 196, 156]}
        right = [{"class_name": "wafer_slot_filled", "confidence": .95, "xyxy": [160, y-20, 200, y+20]}
                 for y in (140, 200)]
        packets = [self.visible + ([hand] if tick in (4, 5) else right if tick >= 7 else [])
                   for tick in range(14)]

        class FullFrameOnlyModel(SyntheticModel):
            def predict(model, image, **kwargs):
                self.assertEqual(image.shape[:2], (480, 640))
                return super().predict(image, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"source.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10., (640, 480))
            self.assertTrue(writer.isOpened())
            for _ in packets:
                writer.write(self.frame)
            writer.release()
            args = argparse.Namespace(source=source, weights=root/"model.pt", output_dir=root/"results",
                conf=.35, imgsz=640, device="cpu", stride=1, show=False, max_frames=0, nms_iou=.7,
                write_video_output=False, run_regression_tests=False)
            context = {"polygon": ROI/[640, 480], "size": (640, 480), "start_frame": 0}
            model = FullFrameOnlyModel(packets)
            with (patch.object(predict.base, "load_detection_model", return_value=(model, "synthetic")),
                  patch.multiple(predict, PROCESS_PHASE="load", HATCH_CONTROL_ENABLED=False,
                      TWO_POINT_EXTENSION_MULTIPLIER=3., LOCAL_REDETECT_ENABLED=True,
                      CONFIRM_SECONDS=0., MIN_CONFIRM_FRAMES=2, INITIAL_GROUP_STABLE_SECONDS=.1)):
                summary = predict.run_prediction(args, roi_polygon=context)
            with (args.output_dir/"slot_states.jsonl").open(encoding="utf8") as records:
                rows = [json.loads(row) for row in records]
        self.assertTrue(any(c["status"] == "waiting_visual" for row in rows
                            for c in row["two_point_hand_contacts"]))
        self.assertTrue(all(not row["local_redetection"] for row in rows))
        self.assertFalse(summary["events"])


class LocalRedetectionTests(unittest.TestCase):
    def setUp(self):
        hatch_control = patch.object(predict, 'HATCH_CONTROL_ENABLED', False)
        hatch_control.start()
        self.addCleanup(hatch_control.stop)
        self.frame=np.zeros((480,640,3),dtype=np.uint8)
        self.args=argparse.Namespace(conf=.35,device='cpu',nms_iou=.7)
        self.sequence=RoiGroupSequence(SlotGroupTracker(POINTS,group_id=2),ROI,
                                     exit_min_slots=1,partial_exit_slots=1)
        self.sequence.tracker._point_tracking_gate=40
        self.config=patch.multiple(predict,LOCAL_REDETECT_ENABLED=True,
                                   LOCAL_REDETECT_CROP_SIZE=(180,140),LOCAL_REDETECT_IMAGE_SIZE=320)
        self.config.start()
        self.addCleanup(self.config.stop)

    def recover(self, objects, local):
        model=SyntheticModel([local])
        model.predict=unittest.mock.Mock(wraps=model.predict)
        result,attempts=predict.recover_missing_slots(model,self.frame,objects,self.sequence,self.args)
        return result,attempts,model

    def test_complete_full_frame_needs_no_extra_inference(self):
        result,attempts,model=self.recover(boxes(),[])
        self.assertEqual(result,boxes())
        self.assertEqual(attempts,[])
        model.predict.assert_not_called()

    def test_missing_s1_to_s3_never_trigger_crops(self):
        result, attempts, model = self.recover(boxes(visible=(3,)), [])
        self.assertEqual(result, boxes(visible=(3,)))
        self.assertEqual(attempts, [])
        model.predict.assert_not_called()

    def test_only_s4_is_cropped_when_multiple_slots_are_missing(self):
        local = [{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[80,60,100,80]}]
        result, attempts, model = self.recover(boxes(visible=(0,)), local)
        self.assertEqual([attempt['slot_id'] for attempt in attempts], [4])
        self.assertEqual(result[-1]['xyxy'], [190,190,210,210])
        model.predict.assert_called_once()

    def test_local_hand_threshold_does_not_lower_slot_evidence_threshold(self):
        local=[{'class_name':'hand','confidence':.025,'xyxy':[80,60,100,80]},
               {'class_name':'wafer_slot_filled','confidence':.25,'xyxy':[80,60,100,80]}]
        with patch.object(predict,'HAND_DETECTION_CONFIDENCE',.02):
            result,_,model=self.recover(boxes(visible=(0,1,2)),local)
        self.assertEqual(model.predict.call_args.kwargs['conf'],.02)
        additions=result[3:]
        self.assertEqual(additions, [])

    def test_crop_coordinates_return_to_source_and_wrong_slot_is_rejected(self):
        # 槽4中心(200,200)，裁剪原点(110,130)；第二个框属于邻槽3，不能用于槽4。
        local=[{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[80,60,100,80]},
               {'class_name':'wafer_slot_empty','confidence':.99,'xyxy':[80,0,100,20]}]
        result,attempts,model=self.recover(boxes(visible=(0,1,2)),local)
        self.assertEqual(len(result),4)
        self.assertEqual(result[-1]['xyxy'],[190,190,210,210])
        self.assertEqual(result[-1]['detection_source'],'local_crop')
        self.assertEqual(attempts[0]['crop_xyxy'],[110,130,290,270])
        self.assertEqual(attempts[0]['slot_id'],4)
        self.assertEqual(model.predict.call_args.args[0].shape,(140,180,3))
        self.assertEqual(model.predict.call_args.kwargs['conf'],.35)

    def test_crop_at_image_edge_is_clipped_and_coordinates_stay_correct(self):
        points=POINTS.copy()
        points[3]=[630,460]
        self.sequence=RoiGroupSequence(SlotGroupTracker(points),
            np.array([[0,0],[640,0],[640,480],[0,480]],dtype=np.float32))
        self.sequence.tracker._point_tracking_gate=40
        local=[{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[80,60,100,80]}]
        result,attempts,model=self.recover(boxes(points,visible=(0,1,2)),local)
        self.assertEqual(result[-1]['xyxy'],[620,450,640,470])
        self.assertEqual(attempts[0]['crop_xyxy'],[540,390,640,480])
        self.assertEqual(model.predict.call_args.args[0].shape,(90,100,3))

    def test_detected_hand_in_crop_still_blocks_state_confirmation(self):
        tracker=self.sequence.tracker
        for now in [0.,.2,.4]:
            self.sequence.update(boxes(filled=(0,1,2,3)),now)
        local=[{'class_name':'wafer_slot_empty','confidence':.99,'xyxy':[80,60,100,80]},
               {'class_name':'hand','confidence':.95,'xyxy':[75,55,105,85]}]
        result,attempts,_=self.recover(boxes(visible=(0,1,2)),local)
        snapshot=self.sequence.update(result,.5)
        self.assertTrue(attempts[0]['accepted_detections'])
        self.assertEqual(snapshot['slots'][3]['reason'],'hand_occlusion')
        self.assertEqual(snapshot['slots'][3]['last_confirmed_state'],'filled')
        self.assertEqual(snapshot['new_events'],[])

    def test_formal_loading_does_not_run_local_redetection(self):
        self.sequence.tracker.phase='load'
        visible = boxes(visible=(0, 1, 2))
        local = [{'class_name': 'wafer_slot_filled', 'confidence': .95, 'xyxy': [80, 60, 100, 80]}]
        result, attempts, model = self.recover(visible, local)
        self.assertEqual(result, visible)
        self.assertEqual(attempts, [])
        model.predict.assert_not_called()

    def test_loading_crop_reuses_recent_support_only_for_preview(self):
        tracker = SlotGroupTracker(group_id=2, phase="load", allow_three_point_initialization=True,
                                   geometry_initialization_seconds=.3, initialization_frames=3)
        sequence = RoiGroupSequence(tracker, ROI)
        tracker.start_incremental_entry(40)
        sequence.state = "entering"
        visible = [dict(obj, track_id=10+i) for i, obj in enumerate(boxes(visible=(0,1,2)))]
        left = [visible[0], visible[2]]
        sequence.update(visible, 0.)
        recovered = {"class_name": "wafer_slot_empty", "confidence": .9, "xyxy": [80,60,100,80]}
        model = SyntheticModel([[recovered]])
        calls = []

        def resolve(current, anchors, hands, candidates):
            support, attempt = recover_load_preview_support(
                model, self.frame, current, anchors, hands, candidates, sequence, self.args,
                predict.resolve_detection_thresholds(self.args), LocalRedetectionPolicy(True, (180,140), 320))
            calls.append(attempt)
            return support

        result = sequence.update(left, .1, load_support_resolver=resolve)
        attempt, = calls
        self.assertEqual(attempt["crop_xyxy"], [110,30,290,170])
        self.assertEqual(attempt["purpose"], "third_preview_support")
        preview = result["three_point_preview"]
        self.assertEqual(preview["center_xy"], [200,200])
        self.assertEqual(preview["local_support_box_xyxy"], [190,90,210,110])
        self.assertEqual(preview["support_byte_track_ids"], [10,11,12])
        self.assertEqual(result["initialization"]["eligible_slot_count"], 2)
        self.assertEqual(result["slot_numbering_status"], "provisional")
        self.assertEqual(result["new_events"], [])

    def test_loading_crop_can_use_same_group_layout_before_third_track_is_seen(self):
        tracker = SlotGroupTracker(group_id=2, phase="load", allow_three_point_initialization=True)
        roi = np.float32([[80,80], [300,80], [300,280], [80,280]])
        sequence = RoiGroupSequence(tracker, roi)
        tracker.start_incremental_entry(40)
        tracker._load_numbering_reference = POINTS.copy()
        sequence.state = "entering"
        anchors = [dict(obj, track_id=10+i) for i, obj in enumerate(boxes(visible=(0,2)))]
        sequence.update(anchors, 0.)
        third = {"class_name": "wafer_slot_empty", "confidence": .9, "xyxy": [105,10,125,30]}
        fourth = {**third, "xyxy": [105,110,125,130]}
        model = SyntheticModel([[third], [third, fourth]])
        attempts = []

        def resolve(current, support, hands, candidates):
            observation, attempt = recover_load_preview_support(
                model, self.frame, current, support, hands, candidates, sequence, self.args,
                predict.resolve_detection_thresholds(self.args), LocalRedetectionPolicy(True, (180,140), 320))
            attempts.append(attempt)
            return observation

        result = sequence.update(anchors, .1, load_support_resolver=resolve)
        preview = result["three_point_preview"]
        self.assertEqual(preview["center_xy"], [200,200])
        self.assertEqual(preview["local_support_box_xyxy"], [190,90,210,110])
        self.assertEqual(preview["support_byte_track_ids"], [10,11])
        self.assertNotIn("hand_contact", preview)
        result = sequence.update(anchors, .2, load_support_resolver=resolve)
        self.assertEqual(attempts[-1]["accepted_detections"], [])
        self.assertIsNone(result["three_point_preview"])

    def test_other_group_stages_and_lost_identity_never_run_crops(self):
        for state in ('initializing','entering','exiting','waiting_next','sequence_complete','identity_uncertain'):
            with self.subTest(state=state):
                self.sequence.state=state
                _,attempts,model=self.recover([],[])
                self.assertEqual(attempts,[])
                model.predict.assert_not_called()
        self.sequence.state='active'
        self.sequence.tracker.identity_lost=True
        _,_,model=self.recover([],[])
        model.predict.assert_not_called()

    def test_crop_center_shift_does_not_move_track_or_change_group(self):
        for now in [0.,.2,.4]:
            self.sequence.update(boxes(filled=(0,1,2,3)),now)
        original=self.sequence.tracker.reference_points.copy()
        # 局部框中心向左偏25px，但它只提供槽4的空满状态。
        local=[{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[55,60,75,80]}]
        for now in [.5,.7,.9]:
            result,_,_=self.recover(boxes(visible=(0,1,2),filled=(0,1,2)),local)
            snapshot=self.sequence.update(result,now)
        np.testing.assert_allclose(self.sequence.tracker.reference_points,original)
        self.assertEqual(snapshot['slots'][3]['center_xy'],[200.,200.])
        self.assertEqual(snapshot['slots'][3]['state'],'empty')
        self.assertEqual(snapshot['group_id'],2)
        self.assertEqual(snapshot['group_status'],'active')
        self.assertTrue(snapshot['s4_unload_review_pending'])
        self.sequence.finish()
        self.assertEqual([(e['slot_id'],e['action']) for e in self.sequence.all_events],[(4,'unload')])

    def test_outside_roi_and_distant_boxes_do_not_fill_missing_slot(self):
        self.sequence.contour=np.array([[80,80],[225,80],[225,300],[80,300]],dtype=np.float32)
        # S4 裁剪原点为 (110,130)。首个框中心距 S4 30px 但越出 ROI，第二个距 S4 45px。
        local=[{'class_name':'wafer_slot_empty','confidence':.99,'xyxy':[110,60,130,80]},
               {'class_name':'wafer_slot_empty','confidence':.99,'xyxy':[80,105,100,125]}]
        result,attempts,_=self.recover(boxes(visible=(0,1,2)),local)
        self.assertEqual(len(result),3)
        self.assertEqual(attempts[0]['accepted_detections'],[])

    def test_fallback_uses_alternate_gate_but_keeps_geometric_center(self):
        tracker = self.sequence.tracker
        tracker._s4_before_recalculation = np.array([175., 200.])
        local = {'class_name': 'wafer_slot_filled', 'confidence': .9, 'xyxy': [60, 60, 80, 80]}
        visible = boxes(visible=(0, 1, 2), filled=(0, 1, 2))
        model = SyntheticModel([[], [local]])
        observations, attempts = predict.recover_missing_slots(
            model, self.frame, visible, self.sequence, self.args)
        self.assertEqual([a['crop_xyxy'] for a in attempts], [[110, 130, 290, 270], [85, 130, 265, 270]])
        self.assertEqual(attempts[1]['accepted_detections'][0]['xyxy'], [145, 190, 165, 210])
        result = self.sequence.update(observations, .1)
        self.assertEqual(result['slots'][3]['observation'], 'filled')
        result = self.sequence.update(visible, .2)
        self.assertEqual(result['slots'][3]['temporary_contact_box_xyxy'], [145., 190., 165., 210.])
        self.assertEqual(result['slots'][3]['center_xy'], [200., 200.])
        np.testing.assert_allclose(tracker.reference_points[3], [200., 200.])
        self.assertIsNone(result['slots'][3]['byte_track_id'])

    def test_old_group_crop_is_not_used_to_initialize_next_group(self):
        roi=np.array([[80,80],[410,80],[410,310],[80,310]],dtype=np.float32)
        sequence=RoiGroupSequence(SlotGroupTracker(POINTS,group_id=2),roi,
                                  exit_min_slots=1,partial_exit_slots=1)
        for now in [0.,.2,.4]:
            sequence.update(boxes(),now)
        moved=POINTS.copy()
        moved[0]=[75,100]
        old_crop={'class_name':'wafer_slot_filled','confidence':.99,
                  'xyxy':[290,190,310,210],'detection_source':'local_crop'}
        snapshot=sequence.update(boxes(moved)+[old_crop],.5)
        self.assertEqual((snapshot['group_id'],snapshot['group_status']),(3,'entering'))
        self.assertEqual(snapshot['initialization']['eligible_slot_count'],0)
        self.assertTrue(all(slot['center_xy'] is None for slot in snapshot['slots']))

    def test_local_empty_detection_is_confirmed_and_saved_as_one_unload(self):
        full_frames=[boxes(filled=(0,1,2,3))]*12+[boxes(visible=(0,1,2),filled=(0,1,2))]*8
        local=[{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[80,60,100,80]}]

        class MixedModel(SyntheticModel):
            def __init__(self):
                super().__init__(full_frames)
                self.local_calls=0

            def predict(self,image,**kwargs):
                if image.shape[:2] == (480,640):
                    return super().predict(image,**kwargs)
                self.local_calls+=1
                return SyntheticModel([local]).predict(image,**kwargs)

        with tempfile.TemporaryDirectory(prefix='wafer_crop_recovery_') as directory:
            root=Path(directory)
            source=root/'source.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in full_frames:
                writer.write(self.frame)
            writer.release()
            args=argparse.Namespace(source=source,weights=root/'model.pt',output_dir=root/'results',
                conf=.35,imgsz=1280,device='cpu',stride=1,show=False,max_frames=0,nms_iou=.7)
            context={'polygon':ROI/[640,480],'size':(640,480),'start_frame':0}
            model=MixedModel()
            with (patch.object(predict.base,'load_detection_model',return_value=(model,'synthetic')),
                  patch.object(predict,'CONFIRM_SECONDS',.2),patch.object(predict,'MIN_CONFIRM_FRAMES',3)):
                summary=predict.run_prediction(args,roi_polygon=context)
            self.assertEqual(model.local_calls,8)
            self.assertEqual([(e['group_id'],e['slot_id'],e['action']) for e in summary['events']],[(1,4,'unload')])
            self.assertAlmostEqual(summary['events'][0]['confirmed_at'],1.4)
            saved=[json.loads(line) for line in (args.output_dir/'slot_states.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(saved[-1]['slots'][3]['state'],'empty')
            self.assertEqual(saved[-1]['local_redetection'][0]['group_id'],1)
            self.assertEqual(saved[-1]['local_redetection'][0]['accepted_detections'][0]['xyxy'],[190,190,210,210])


if __name__ == '__main__':
    unittest.main(verbosity=2)
