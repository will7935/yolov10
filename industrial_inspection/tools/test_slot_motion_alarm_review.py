"""Stationary alarm gates must not change tracking or hide post-stop errors."""
from copy import deepcopy
import unittest

from industrial_inspection.tools.slot_operation_alarms import SlotMotionAlarmReview, group_operation_alarms
from industrial_inspection.tools.test_slot_operation_alarms import event


def frame(now, shift=0., group=2, count=3, observation='filled'):
    return {'phase': 'unload', 'group_id': group, 'slot_numbering_status': 'final',
            'order_result': 'fail', 'initialization': {'provisional_started_at': 0.},
            'slots': [{'slot_id': i+1, 'byte_track_id': 100+i,
                       'center_xy': [100.+i*80+shift, 150.+i%2*60+shift],
                       'geometry_inferred': False, 'last_seen_at': now, 'observation': observation,
                       'last_confirmed_state': 'empty', 'state': observation,
                       'tracking_box_xyxy': [70.+i*80+shift, 125., 130.+i*80+shift, 175.]}
                      for i in range(count)]}


def settle(review):
    for i in range(20):
        now = i/20
        review.observe_group(frame(now, min(i*5., 50.)), now)
    return frame(.95, 50.)


class StationaryAlarmTests(unittest.TestCase):
    def test_no_order_alarm_before_verified_stop(self):
        review = SlotMotionAlarmReview()
        s = frame(.1)
        review.observe_group(s, .1)
        self.assertTrue(review.paused)
        self.assertEqual(group_operation_alarms(s, [event(3, 'load', .1)], {}, .1,
                                               motion_review=review), [])

    def test_rotation_history_never_reappears_after_stop(self):
        review = SlotMotionAlarmReview()
        s = settle(review)
        self.assertFalse(review.paused)
        original = [event(3, 'load', .3, historical=True), event(3, 'unload', .9)]
        copy = deepcopy(original)
        self.assertEqual(group_operation_alarms(s, original, {}, 1., motion_review=review), [])
        self.assertEqual(original, copy)
        self.assertEqual(len(s['alarm_ignored_motion_events']), 1)

    def test_early_false_unload_load_pair_does_not_remove_real_later_unload(self):
        review = SlotMotionAlarmReview()
        s = settle(review)
        events = [event(3, 'unload', .2), event(3, 'load', .4), event(3, 'unload', .9)]
        kept, ignored = review.review(s, events)
        self.assertEqual([e['time_upper'] for e in kept], [.9])
        self.assertEqual(len(ignored), 2)
        self.assertEqual(group_operation_alarms(s, events, {}, 1., motion_review=review), [])

    def test_genuine_reverse_after_stop_still_alarms(self):
        review = SlotMotionAlarmReview()
        s = settle(review)
        alarms = group_operation_alarms(s, [event(3, 'load', .9)], {}, 1., motion_review=review)
        self.assertEqual(alarms[0]['reason'], 'unexpected_action')

    def test_incomplete_alarm_survives_unverified_motion(self):
        review = SlotMotionAlarmReview()
        s = frame(.1)
        s['order_result'] = 'pending'
        s['slots'][0]['last_confirmed_state'] = 'filled'
        review.observe_group(s, .1)
        alarms = group_operation_alarms(s, [], {}, .1, close_reason='hatch_closed', motion_review=review)
        self.assertEqual([a['type'] for a in alarms], ['group_incomplete_alarm'])

    def test_single_point_cannot_verify_stop(self):
        review = SlotMotionAlarmReview()
        for i in range(10):
            review.observe_group(frame(i/10, count=1), i/10)
        self.assertTrue(review.paused)

    def test_new_group_starts_with_its_own_closed_gate(self):
        review = SlotMotionAlarmReview()
        settle(review)
        review.observe_group(frame(1., group=3), 1.)
        self.assertTrue(review.paused)
        self.assertIsNotNone(review.groups['unload', 2]['ranges'][-1]['end'])

    def test_real_coherent_motion_recloses_gate(self):
        review = SlotMotionAlarmReview()
        settle(review)
        for i in range(10):
            review.observe_group(frame(1.+i/20, 50.+i*6.), 1.+i/20)
        self.assertTrue(review.paused)

    def test_observation_does_not_mutate_states_or_numbering(self):
        review = SlotMotionAlarmReview()
        s = frame(0.)
        original = deepcopy(s)
        review.observe_group(s, 0.)
        self.assertEqual(s, original)

    def test_order_alarms_pause_while_coherent_motion_is_being_verified(self):
        review = SlotMotionAlarmReview()
        settle(review)
        review.observe_group(frame(1., 60.), 1.)
        self.assertTrue(review.paused)
        self.assertTrue(review.groups['unload', 2]['stationary'])

    def test_state_class_changes_do_not_look_like_common_carrier_motion(self):
        review = SlotMotionAlarmReview()
        settle(review)
        for i in range(5):
            s = frame(1.+i/20, 60., observation='empty')
            review.observe_group(s, 1.+i/20)
        self.assertFalse(review.paused)


if __name__ == '__main__':
    unittest.main()
