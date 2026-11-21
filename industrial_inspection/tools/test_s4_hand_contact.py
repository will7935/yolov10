"""验证区域门控、实际接触和待确认状态的诊断含义不同。"""
import unittest

import numpy as np

from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.test_slot_group_tracker import CENTERS, detection


class ContactDiagnosticsTests(unittest.TestCase):
    def tracker(self, *, inferred=True):
        tracker = SlotGroupTracker(CENTERS,phase='load',confirm_seconds=0.,minimum_frames=2,
            load_hand_contact_seconds=0.,load_hand_contact_min_frames=2,
            load_hand_contact_competing_slot_enabled=False,
            s4_hand_region=np.asarray([[180,80],[350,80],[350,220],[180,220]],dtype=np.float32))
        tracker.initialization['inferred_slot_ids'] = [4] if inferred else []
        for now in (0.,.05):
            tracker.update([detection(i,'filled' if i<4 else 'empty') for i in range(1,5)],now)
        tracker.events=[{'slot_id':i,'expected_action':True,'time_upper':j,'time_lower':0.}
                        for j,i in enumerate((2,1,3))]
        return tracker

    def test_region_gate_does_not_mean_contact_accepted(self):
        tracker = self.tracker()
        hand = {'class_name':'hand','confidence':.9,'xyxy':[190,90,210,110]}
        self.assertFalse(tracker._hand_contact(3,[hand]))
        diagnostic = tracker.s4_region_diagnostic
        self.assertEqual(diagnostic['status'],'eligible')
        self.assertFalse(diagnostic['contact_accepted'])
        self.assertEqual(diagnostic['decision_reason'],'other_slot_center_covered')
        self.assertEqual(tracker.s4_contact_diagnostic['candidates'][0]['covered_slot_ids'],[2])

    def test_two_frames_release_and_state_confirmation_have_distinct_status(self):
        tracker = self.tracker()
        visible = [detection(i,'filled') for i in (1,2,3)]
        hand = {'class_name':'hand','confidence':.9,'xyxy':[300,185,320,195]}
        result = tracker.update(visible+[hand],.1)
        self.assertEqual(result['s4_contact_diagnostic']['accumulation_status'],'collecting_contact')
        self.assertFalse(result['s4_contact_diagnostic']['evidence_ready'])
        result = tracker.update(visible+[hand],.15)
        self.assertEqual(result['s4_contact_diagnostic']['accumulation_status'],'pending_contact')
        self.assertTrue(result['s4_contact_diagnostic']['evidence_ready'])
        result = tracker.update(visible,.2)
        self.assertEqual(result['s4_contact_diagnostic']['accumulation_status'],'released_waiting_verification')
        self.assertEqual(len(tracker.events),3)
        result = tracker.update(visible,.8)
        self.assertEqual(result['s4_contact_diagnostic']['accumulation_status'],'event_confirmed')
        self.assertFalse(result['s4_contact_diagnostic']['evidence_ready'])
        self.assertEqual(result['new_events'][0]['reason'],'hand_contact_inferred_load')

    def test_contact_interrupt_is_reported_and_does_not_count_sparse_frames(self):
        tracker = self.tracker()
        visible = [detection(i,'filled') for i in (1,2,3)]
        hand = {'class_name':'hand','confidence':.9,'xyxy':[300,185,320,195]}
        tracker.update(visible+[hand],.1)
        result = tracker.update(visible,.15)
        self.assertEqual(result['s4_contact_diagnostic']['accumulation_status'],
                         'contact_interrupted_before_confirmation')
        self.assertEqual(result['s4_contact_diagnostic']['contact_count'],0)
        result = tracker.update(visible+[hand],.2)
        self.assertFalse(result['s4_contact_diagnostic']['evidence_ready'])

    def test_inferred_s4_load_requires_visual_empty_to_reverse(self):
        tracker = self.tracker()
        tracker.minimum_frames = 4
        visible = [detection(i, 'filled') for i in (1, 2, 3)]
        hand = {'class_name': 'hand', 'confidence': .9, 'xyxy': [185, 185, 215, 215]}
        for now in (.1, .15):
            tracker.update(visible+[hand], now)
        tracker.update(visible, .2)
        result = tracker.update(visible, .8)
        load, = result['new_events']
        self.assertEqual((load['action'], load['reason']), ('load', 'hand_contact_inferred_load'))

        for now in (.85, .9):
            result = tracker.update(visible+[hand], now)
        self.assertTrue(result['slots'][3]['load_contact_pending'])
        tracker.update(visible, .95)
        result = tracker.update(visible, 1.5)
        self.assertEqual(result['new_events'], [])
        self.assertEqual(result['slots'][3]['last_confirmed_state'], 'filled')
        self.assertFalse(result['slots'][3]['load_contact_pending'])

        for now in (1.55, 1.6, 1.65):
            result = tracker.update(visible+[detection(4, 'empty')], now)
            self.assertEqual(result['new_events'], [])
        result = tracker.update(visible+[detection(4, 'empty')], 1.7)
        unload, = result['new_events']
        self.assertEqual((unload['slot_id'], unload['action'], unload['expected_action']), (4, 'unload', False))
        self.assertFalse(unload.get('inferred', False))
        self.assertEqual(result['order_result'], 'fail')

    def test_box_edge_contact_waits_for_filled_detection(self):
        tracker = self.tracker()
        visible = [detection(i, 'filled') for i in (1, 2, 3)]
        hand = {'class_name': 'hand', 'confidence': .9, 'xyxy': [200, 180, 220, 198]}
        for now in (.1, .15):
            result = tracker.update(visible+[hand], now)
        self.assertEqual(result['s4_contact_diagnostic']['source'], 's4_box')
        self.assertTrue(result['s4_contact_diagnostic']['visual_confirmation_required'])

        tracker.update(visible, .2)
        result = tracker.update(visible, .8)
        self.assertEqual(result['new_events'], [])
        self.assertEqual(result['slots'][3]['last_confirmed_state'], 'empty')

        for now in (.85, .9):
            result = tracker.update(visible+[detection(4, 'filled')], now)
        event, = result['new_events']
        self.assertEqual(event['reason'], 'detected_after_hand_contact')
        self.assertFalse(event['inferred'])
        self.assertEqual(event['time_upper'], .85)

    def test_measured_s4_does_not_use_box_or_fixed_region_contact(self):
        tracker = self.tracker(inferred=False)
        visible = [detection(i, 'filled') for i in (1, 2, 3)]
        for hand_box in ([184, 184, 216, 216], [300, 185, 320, 195]):
            with self.subTest(hand_box=hand_box):
                hand = {'class_name': 'hand', 'confidence': .9, 'xyxy': hand_box}
                self.assertFalse(tracker._hand_contact(3, [hand]))
        for now in (.1, .15):
            result = tracker.update(visible+[hand], now)
        self.assertIsNone(result['s4_contact_diagnostic'])
        self.assertFalse(result['slots'][3]['load_contact_pending'])


if __name__ == '__main__':
    unittest.main()
