import json
import unittest

import numpy as np

from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_hand_trajectory import LoadHandTrajectory
from industrial_inspection.tools.slot_operation_alarms import group_operation_alarms


POINTS = np.array([[435.22, 281.18], [535.73, 328.82], [498.94, 248.25], [600.91, 307.18]])
SAMPLES = [(140.587, [470.39, 268.71, 524.35, 317.47]),
           (140.641, [496.27, 273.40, 547.50, 322.75]),
           (140.695, [513.83, 276.32, 556.36, 323.51]),
           (140.749, [524.73, 276.07, 615.82, 342.80])]


def targets():
    result = {sid: {'center': POINTS[sid-1], 'box': [*(POINTS[sid-1]-[50., 30.]),
                                                 *(POINTS[sid-1]+[50., 30.])],
                    'baseline_at': 140.3, 'bound': sid != 2} for sid in (1, 2, 4)}
    result[4]['box'] = [561.82, 269.71, 640., 344.65]
    return result


def sample_review():
    review = LoadHandTrajectory()
    for now, box in SAMPLES:
        review.observe([{'confidence': .9, 'xyxy': box}], targets(), [2], now)
    return review


def visual_event(slot_id=4, **kwargs):
    return dict({'slot_id': slot_id, 'action': 'load', 'from': 'empty', 'to': 'filled',
                 'time_lower': 140.4, 'time_upper': 148.128, 'confirmed_at': 148.29,
                 'reported_at': 148.83, 'byte_track_id': 738, 'expected_action': True}, **kwargs)


class TrajectoryReviewTests(unittest.TestCase):
    def test_passed_prediction_is_deferred_and_one_observed_contact_is_not_four(self):
        review = sample_review()
        self.assertTrue(review.blocks_inference(2))
        self.assertEqual(set(review.pending), {4})
        self.assertEqual(review.pending[4]['status'], 'waiting_for_visual')
        self.assertEqual(review.pending[4]['trajectory_sample_count'], 4)
        self.assertEqual(review.pending[4]['observed_contact_frames'], 1)

    def test_visual_confirmation_backfills_action_but_keeps_confirmation_time(self):
        review = sample_review()
        event = review.confirm(visual_event(), 148.83)
        self.assertEqual(event['time_upper'], 140.749)
        self.assertEqual(event['visual_time_upper'], 148.128)
        self.assertEqual(event['confirmed_at'], 148.29)
        self.assertEqual(event['reported_at'], 148.83)
        self.assertTrue(event['action_time_inferred'])
        self.assertNotIn('inferred', event)
        self.assertFalse(review.pending)
        self.assertEqual(review.confirm(event, 148.9), event)

    def test_numpy_slot_ids_can_be_written_to_the_frame_report(self):
        review = LoadHandTrajectory()
        numpy_targets = {np.int32(sid): value for sid, value in targets().items()}
        for now, box in SAMPLES:
            review.observe([{'confidence': .9, 'xyxy': box}], numpy_targets, [np.int32(2)], now)
        self.assertIn('waiting_for_visual', json.dumps(review.snapshot()))

    def test_the_passed_slots_own_visual_load_is_never_renamed(self):
        review = sample_review()
        source = visual_event(slot_id=2, time_upper=143.6)
        self.assertEqual(review.confirm(source, 143.8), source)
        self.assertIn(4, review.pending)

    def test_visual_evidence_that_origin_loaded_first_cancels_alternative(self):
        review = sample_review()
        source = visual_event(slot_id=2, time_upper=140.7)
        self.assertEqual(review.confirm(source, 141.), source)
        self.assertNotIn(4, review.pending)

    def test_inferred_or_unbound_visual_event_does_not_confirm(self):
        for kwargs in ({'inferred': True}, {'byte_track_id': None}):
            review = sample_review()
            event = visual_event(**kwargs)
            self.assertEqual(review.confirm(event, 148.83), event)
            self.assertIn(4, review.pending)

    def test_later_empty_observation_cancels_but_gaps_do_not_count(self):
        review = sample_review()
        for now in (141., 141.1):
            review.observe_empty(4, now, 4)
        review.break_empty_run(4)
        for now in (141.3, 141.4):
            review.observe_empty(4, now, 4)
        self.assertIn(4, review.pending)
        for now in (141.5, 141.6):
            review.observe_empty(4, now, 4)
        self.assertNotIn(4, review.pending)

    def test_missing_hands_do_not_create_motion_or_contact(self):
        review = LoadHandTrajectory()
        for now, box in SAMPLES[:3]:
            review.observe([{'confidence': .9, 'xyxy': box}], targets(), [2], now)
        review.observe([], targets(), [2], 140.749)
        review.observe([{'confidence': .9, 'xyxy': SAMPLES[-1][1]}], targets(), [2], 141.2)
        self.assertFalse(review.pending)

    def test_unknown_alternative_identity_does_not_assign_a_slot(self):
        review = LoadHandTrajectory()
        unbound = targets()
        unbound[4]['bound'] = False
        for now, box in SAMPLES:
            review.observe([{'confidence': .9, 'xyxy': box}], unbound, [2], now)
        self.assertFalse(review.pending)

    def test_s2_dwell_prevents_treating_departure_as_pass_through(self):
        review = LoadHandTrajectory()
        path = [(0., [485., 300.]), (.05, [520., 320.]), (.1, [535., 329.]),
                (.16, [536., 329.]), (.22, [535., 330.]), (.27, [570., 309.])]
        for now, center in path:
            p = np.asarray(center)
            review.observe([{'confidence': .9, 'xyxy': [*(p-20), *(p+20)]}], targets(), [2], now)
        self.assertFalse(review.pending)

    def test_a_later_dwell_allows_the_passed_slots_own_contact_again(self):
        review = sample_review()
        for now in (141.2, 141.25, 141.3, 141.36):
            p = POINTS[1]
            review.observe([{'confidence': .9, 'xyxy': [*(p-20), *(p+20)]}], targets(), [2], now)
        self.assertFalse(review.blocks_inference(2))
        self.assertIn(4, review.pending)

    def test_timeout_and_group_reset_do_not_reuse_evidence(self):
        review = sample_review()
        review.observe([], {}, [], 154.)
        self.assertFalse(review.pending)
        tracker = SlotGroupTracker(POINTS, phase='load', load_trajectory_enabled=True)
        tracker.load_trajectory = sample_review()
        tracker.reset_group(2)
        self.assertFalse(tracker.load_trajectory.pending)
        self.assertTrue(tracker.load_trajectory_enabled)

    def test_pending_defers_only_active_order_alarm(self):
        snapshot = {'phase': 'load', 'group_id': 4, 'slot_numbering_status': 'final',
                    'order_result': 'fail', 'hand_trajectory_review_pending': True,
                    'slots': [{'slot_id': i, 'last_confirmed_state': 'empty'} for i in range(1, 5)]}
        events = [{'slot_id': i, 'time_upper': i, 'expected_action': True} for i in (1, 2, 3)]
        self.assertEqual(group_operation_alarms(snapshot, events, {}, 1.), [])
        closed = group_operation_alarms(snapshot, events, {}, 2., close_reason='observed_roi_exit')
        self.assertTrue(any(a['type'] == 'group_incomplete_alarm' for a in closed))


class TrackerIntegrationTests(unittest.TestCase):
    def test_disabling_hand_contact_also_disables_trajectory_review(self):
        tracker = SlotGroupTracker(POINTS, phase='load', load_hand_contact_enabled=False,
                                   load_trajectory_enabled=True)
        tracker.initialization['inferred_slot_ids'] = [2]
        for slot in tracker.slots:
            slot.confirmed = 'empty'
            slot.box_size_wh = (100., 60.)
        tracker.byte_track_ids = [654, None, 652, 697]
        for now, box in SAMPLES:
            tracker._observe_load_trajectory([{'confidence': .9, 'xyxy': box}], now, True)
        self.assertFalse(tracker.load_trajectory.pending)

    def test_trajectory_prevents_early_state_toggle_without_fabricating_s4_load(self):
        tracker = SlotGroupTracker(POINTS, phase='load', confirm_seconds=0., minimum_frames=2,
            load_hand_contact_seconds=0., load_hand_contact_min_frames=2,
            load_hand_contact_competing_slot_enabled=False, load_hand_contact_overlap_ratio=.03,
            load_trajectory_enabled=True)
        tracker.initialization['inferred_slot_ids'] = [2]
        tracker.slots[1].geometry_inferred = True
        tracker._load_verified_empty_slots = {1, 2, 4}
        tracker.byte_track_ids = [654, None, 652, 697]
        for slot in tracker.slots:
            slot.confirmed = slot.state = 'filled' if slot.slot_id == 3 else 'empty'
            slot.evidence_time = slot.confirmed_since = slot.confirmed_at = 140.3
            slot.box_size_wh = (100., 60.)
        for now, box in SAMPLES:
            # Feed real measurements until the hand obscures them, as in the target video.
            tracker.last_time = now
            tracker.slots[1].temporary_contact_box_xyxy = targets()[2]['box']
            tracker._observe_load_trajectory([{'confidence': .9, 'xyxy': box}], now, True)
            if tracker.load_trajectory.blocks_inference(2):
                tracker._hand_contact(1, [{'confidence': .9, 'xyxy': box}])
        self.assertEqual(tracker.slots[1].confirmed, 'empty')
        self.assertEqual(tracker.slots[3].confirmed, 'empty')
        self.assertFalse(tracker.slots[1].load_contact_pending)
        self.assertEqual(tracker.events, [])
        self.assertTrue(tracker.snapshot()['hand_trajectory_review_pending'])


if __name__ == '__main__':
    unittest.main()
