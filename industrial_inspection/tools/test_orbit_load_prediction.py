"""The orbit trial predicts geometry; each hidden slot still needs independent contact evidence."""
import unittest

import numpy as np

from industrial_inspection.tools.test_two_point_numbering import tracker, objects, POINTS, ROI


def orbit_tracker():
    t = tracker()
    t.machine_center = np.array([300., 150.])
    t.two_point_hand_contact.orbit_degrees = -40.
    target = POINTS[[1, 3, 0, 2]]
    t._orbit_unload_reference = t._load_perspective_numbering.rotate_about_machine(target, t.machine_center, 40.)
    t._load_numbering_reference = t._orbit_unload_reference.copy()
    return t


def baseline(t):
    for now in (0., .1, .2):
        result = t.update(objects(), now, roi_contour=ROI)
    assert result['slot_numbering_status'] == 'final'
    for index in (1, 3):
        slot = t.slots[index]
        slot.confirmed = slot.state = 'empty'
        slot.confirmed_since = slot.confirmed_at = slot.evidence_time = -.1
        slot.confirmation_source = 'previous_unload_yolo'
    t._load_verified_empty_slots.update((2, 4))
    return result


class OrbitLoadPredictionTests(unittest.TestCase):
    def test_inverse_rotation_matches_debug_geometry_and_labels_hidden_s2_s4(self):
        t = orbit_tracker()
        result = baseline(t)
        np.testing.assert_allclose(t.points, POINTS[[1, 3, 0, 2]], atol=1e-10)
        self.assertEqual(t.byte_track_ids, [12, None, 11, None])
        self.assertEqual(t.initialization['inferred_slot_ids'], [2, 4])
        self.assertEqual(t.initialization['two_point_numbering']['reference'], 'unload_orbit_rotation')
        self.assertEqual(t.initialization['two_point_numbering']['clockwise_degrees'], -40.)
        self.assertFalse(t.events)

    def test_no_hand_does_not_convert_geometry_to_filled(self):
        t = orbit_tracker()
        baseline(t)
        for frame in range(3, 20):
            t.update(objects(), frame/10, roi_contour=ROI)
        self.assertFalse(t.events)
        self.assertEqual([t.slots[i].confirmed for i in (1, 3)], ['empty', 'empty'])

    def test_independent_center_contacts_infer_both_hidden_loads_after_release(self):
        t = orbit_tracker()
        t.load_hand_contact_min_frames, t.load_hand_contact_seconds = 2, 0.
        baseline(t)
        for frame in range(3, 24):
            center_y = 100 if frame in (3, 4) else 200 if frame in (13, 14) else None
            hand = [] if center_y is None else [{'class_name': 'hand', 'confidence': .95,
                'xyxy': [175., center_y-25, 225., center_y+25]}]
            t.update(objects()+hand, frame/10, roi_contour=ROI)
        self.assertEqual([(e['slot_id'], e['action']) for e in t.events], [(4, 'load'), (2, 'load')])
        self.assertTrue(all(e.get('inferred') for e in t.events))
        self.assertEqual([t.slots[i].confirmed for i in (1, 3)], ['filled', 'filled'])

    def test_covering_both_centers_cannot_start_two_loads(self):
        t = orbit_tracker()
        baseline(t)
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [175., 75., 225., 225.]}
        for frame in range(3, 15):
            t.update(objects()+([hand] if frame < 8 else []), frame/10, roi_contour=ROI)
        self.assertFalse(t.events)
        self.assertFalse(any(t.slots[i].load_contact_pending for i in (1, 3)))

    def test_wrong_visible_pair_cannot_force_an_orbit_mapping(self):
        t = orbit_tracker()
        for frame in range(10):
            result = t.update(objects((50, 0)), frame/10, roi_contour=ROI)
        self.assertEqual(result['slot_numbering_status'], 'provisional')
        self.assertFalse(t.events)

    def test_explicit_anchors_tolerate_projection_bias_and_keep_orbit_predictions(self):
        t = orbit_tracker()
        t.two_point_hand_contact.upper_slot_three = True
        for frame in range(5):
            result = t.update(objects((50, 0)), frame/10, roi_contour=ROI)
        self.assertEqual(result['slot_numbering_status'], 'final')
        self.assertEqual(t.byte_track_ids, [12, None, 11, None])
        np.testing.assert_allclose(t.reference_points[[1, 3]], POINTS[[3, 2]])
        self.assertEqual([p['slot_id'] for p in result['two_point_hand_contacts']], [4, 2])
        self.assertFalse(t.events)

    def test_orbit_trial_leaves_subsequent_groups_on_g1_loading_template(self):
        t = orbit_tracker()
        self.assertTrue(t.two_point_hand_contact.uses_orbit(t))
        t.group_id = 2
        self.assertFalse(t.two_point_hand_contact.uses_orbit(t))

    def test_existing_three_point_numbering_keeps_its_original_contact_policy(self):
        t = orbit_tracker()
        baseline(t)
        t.initialization['two_point_numbering'] = {'reference': 'same_group_unload'}
        t.initialization['inferred_slot_ids'] = [4]
        self.assertFalse(t.two_point_hand_contact.uses_orbit(t))
        self.assertTrue(t._is_hand_contact_slot(3))
        self.assertFalse(t._is_hand_contact_slot(1))

    def test_main_entry_enables_orbit_only_for_trial_source(self):
        from pathlib import Path
        from types import SimpleNamespace
        from industrial_inspection.tools.predict_slot_groups import make_sequence
        for filename, expected in (('camera_B_20260924_062208.avi', -40.), ('other.avi', None)):
            with self.subTest(filename=filename):
                seq = make_sequence(ROI/400, (400, 400), args=SimpleNamespace(source=Path(filename)))
                self.assertEqual(seq._new_tracker(1, phase='load').two_point_hand_contact.orbit_degrees,
                                 expected)


if __name__ == '__main__':
    unittest.main()
