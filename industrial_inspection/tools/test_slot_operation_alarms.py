"""Entry-history alarm review must preserve real operations and completion alarms."""
from copy import deepcopy
import unittest

from industrial_inspection.tools.slot_operation_alarms import SlotMotionAlarmReview, group_operation_alarms


def event(slot, action, when, *, historical=False):
    return dict(slot_id=slot, action=action, expected_action=action == 'unload',
                **{'from': 'empty' if action == 'load' else 'filled',
                   'to': 'filled' if action == 'load' else 'empty'},
                time_lower=when-.3, time_upper=when, confirmed_at=when+.16,
                byte_track_id=159 if slot == 3 else 160+slot,
                recovered_before_initialization=historical,
                reason='track_history_before_slot_mapping' if historical else 'detected')


def snapshot():
    return dict(phase='unload', group_id=2, slot_numbering_status='final', order_result='fail',
                initialization={'provisional_started_at': 31.457},
                slots=[dict(slot_id=i, last_confirmed_state='empty') for i in range(1, 5)])


def stationary_review():
    review = SlotMotionAlarmReview()
    for now in (32.5, 32.6, 32.7, 32.8):
        sample = snapshot()
        sample['slots'] = [dict(slot_id=i+1, byte_track_id=i, center_xy=[100+i*80, 100],
                                last_seen_at=now, observation='filled') for i in range(3)]
        review.observe_group(sample, now)
    return review


class EntryAlarmTests(unittest.TestCase):
    def test_video11_initial_reverse_is_baseline_without_changing_raw_events(self):
        s = snapshot()
        events = [event(3, 'load', 31.403, historical=True), event(3, 'unload', 34.689, historical=True)]
        original = deepcopy(events)
        self.assertEqual(group_operation_alarms(s, events, {3: 5}, 35.497,
                                               motion_review=stationary_review()), [])
        self.assertEqual(events, original)
        self.assertEqual(s['order_result'], 'fail')  # Original audit result is preserved.
        self.assertEqual(s['alarm_order_result'], 'pending')
        self.assertEqual(len(s['alarm_ignored_motion_events']), 1)

    def test_full_correct_unload_passes_after_baseline_review(self):
        s = snapshot()
        events = [event(3, 'load', 31.403, historical=True)]
        events += [event(i, 'unload', 34.689+j*2) for j, i in enumerate((3, 4, 2, 1))]
        self.assertEqual(group_operation_alarms(s, events, {}, 44., close_reason='observed_roi_exit',
                                               motion_review=stationary_review()), [])
        self.assertEqual(s['alarm_order_result'], 'pass')

    def test_reverse_during_entry_is_still_an_alarm(self):
        s = snapshot()
        alarms = group_operation_alarms(s, [event(3, 'load', 32., historical=True)], {}, 35.5)
        self.assertEqual(alarms[0]['reason'], 'unexpected_action')

    def test_live_reverse_and_g1_are_not_filtered(self):
        for group, historical in ((2, False), (1, True)):
            s = snapshot()
            s['group_id'] = group
            self.assertEqual(group_operation_alarms(s, [event(3, 'load', 31., historical=historical)],
                                                   {}, 35.5)[0]['reason'], 'unexpected_action')

    def test_reverse_after_a_valid_operation_is_not_baseline(self):
        events = [event(3, 'unload', 30.5, historical=True), event(4, 'load', 31., historical=True)]
        self.assertEqual(group_operation_alarms(snapshot(), events, {}, 35.5)[0]['reason'], 'unexpected_action')

    def test_bad_clockwise_order_still_alarms_after_filtering(self):
        events = [event(3, 'load', 31.403, historical=True)]
        events += [event(i, 'unload', 35.+j) for j, i in enumerate((3, 2, 4))]
        alarm = group_operation_alarms(snapshot(), events, {}, 38., motion_review=stationary_review())[0]
        self.assertEqual(alarm['reason'], 'not_clockwise')
        self.assertEqual(alarm['slot_sequence'], [3, 2, 4])

    def test_repeated_real_operation_still_alarms_after_filtering(self):
        events = [event(3, 'load', 31.403, historical=True),
                  event(3, 'unload', 34.689), event(3, 'unload', 36.)]
        alarm = group_operation_alarms(snapshot(), events, {}, 37., motion_review=stationary_review())[0]
        self.assertEqual(alarm['reason'], 'repeated_slot')
        self.assertEqual(alarm['slot_sequence'], [3, 3])

    def test_incomplete_alarm_is_not_suppressed(self):
        s = snapshot()
        s['slots'][0]['last_confirmed_state'] = 'filled'
        events = [event(3, 'load', 31.403, historical=True), event(3, 'unload', 34.689, historical=True)]
        alarms = group_operation_alarms(s, events, {}, 36., close_reason='hatch_closed',
                                        motion_review=stationary_review())
        self.assertEqual([a['type'] for a in alarms], ['group_incomplete_alarm'])
        self.assertEqual(alarms[0]['slots'][0]['slot_id'], 1)

    def test_missing_entry_time_keeps_original_alarm(self):
        s = snapshot()
        s['initialization'] = {}
        self.assertEqual(group_operation_alarms(s, [event(3, 'load', 31., historical=True)],
                                               {}, 35.5)[0]['reason'], 'unexpected_action')

    def test_video12_empty_end_states_do_not_hide_the_missing_s4_operation(self):
        s = snapshot()
        s['group_id'] = 4
        events = [event(3, 'unload', 31.), event(3, 'load', 31.5)]
        events += [event(i, 'unload', 35.+j) for j, i in enumerate((3, 2, 1))]
        review = stationary_review()
        review.groups[('unload', 4)] = review.groups[('unload', 2)]
        alarms = group_operation_alarms(s, events, {3: 13, 4: 14, 2: 15, 1: 16}, 40.,
            close_reason='observed_roi_exit', motion_review=review)
        alarm, = alarms
        self.assertEqual(s['alarm_order_result'], 'pending')
        self.assertEqual(alarm['type'], 'slot_operation_record_alarm')
        self.assertEqual(alarm['missing_slot_ids'], [4])
        self.assertEqual(alarm['confirmed_count'], 3)
        self.assertEqual(alarm['slot_sequence'], [3, 3, 2, 1])
        self.assertIn('S4', alarm['message'])

    def test_record_audit_waits_for_group_close(self):
        s = snapshot()
        s['order_result'] = 'pending'
        events = [event(i, 'unload', 35.+j) for j, i in enumerate((3, 2, 1))]
        self.assertEqual(group_operation_alarms(s, events, {}, 40.), [])

    def test_early_real_operation_counts_toward_record_coverage(self):
        s = snapshot()
        events = [event(3, 'unload', 31., historical=True)]
        events += [event(i, 'unload', 35.+j) for j, i in enumerate((4, 2, 1))]
        alarms = group_operation_alarms(s, events, {}, 40.,
            close_reason='observed_roi_exit', motion_review=stationary_review())
        self.assertEqual(alarms, [])

    def test_hatch_close_audits_missing_load_records_too(self):
        s = snapshot()
        s.update(phase='load', order_result='pending', hand_trajectory_review_pending=True)
        for slot in s['slots']:
            slot['last_confirmed_state'] = 'filled'
        events = [dict(event(i, 'load', 35.+j), expected_action=True)
                  for j, i in enumerate((3, 2, 1))]
        alarm, = group_operation_alarms(s, events, {}, 40., close_reason='hatch_closed')
        self.assertEqual(alarm['missing_slot_ids'], [4])
        self.assertIn('上料操作记录不完整', alarm['message'])


if __name__ == '__main__':
    unittest.main()
