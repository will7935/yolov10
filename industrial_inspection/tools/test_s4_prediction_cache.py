"""Regression coverage for S4 inferred before its first real detection."""
import unittest

import numpy as np

from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.test_slot_group_tracker import CENTERS, detection


class S4UnloadReviewTests(unittest.TestCase):
    def test_quick_visual_reversal_cancels_unload_and_refill(self):
        tracker = SlotGroupTracker(CENTERS, confirm_seconds=0., minimum_frames=2,
                                   load_hand_contact_seconds=0.)
        support = [detection(i, 'filled') for i in (1, 2, 3)]
        for now, state in ((0., 'filled'), (.1, 'filled'), (.2, 'empty'), (.3, 'empty')):
            result = tracker.update(support + [detection(4, state)], now)
        self.assertEqual(result['new_events'], [])
        self.assertTrue(result['s4_unload_review_pending'])

        tracker.update(support + [detection(4, 'filled')], .4)
        result = tracker.update(support + [detection(4, 'filled')], .5)
        self.assertEqual(result['new_events'], [])
        self.assertEqual(tracker.events, [])
        self.assertFalse(result['s4_unload_review_pending'])
        self.assertEqual(result['s4_contact_diagnostic']['unload_review'], 'visual_reversal_cancelled')

        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [188, 188, 212, 212]}
        for now in (.6, .7):
            tracker.update(support + [hand], now)
        for now in (.8, .9):
            result = tracker.update(support + [detection(4, 'empty')], now)
        event, = result['new_events']
        self.assertEqual(event['time_upper'], .8)
        self.assertEqual(event['reported_at'], .9)
        self.assertEqual(tracker.events, [event])

    def test_finishing_publishes_pending_unload_with_original_times(self):
        tracker = SlotGroupTracker(CENTERS, confirm_seconds=0., minimum_frames=2)
        roi = np.float32([[40, 40], [300, 40], [300, 300], [40, 300]])
        sequence = RoiGroupSequence(tracker, roi)
        for now, state in ((0., 'filled'), (.1, 'filled'), (.2, 'empty'), (.3, 'empty')):
            result = sequence.update([detection(4, state)], now)
        self.assertTrue(result['s4_unload_review_pending'])
        result = sequence.finish()
        event, = result['new_events']
        self.assertEqual((event['time_upper'], event['confirmed_at']), (.2, .3))
        self.assertEqual(sequence.all_events, [event])
        self.assertEqual(sequence.finish()['new_events'], [])

    def test_review_timeout_publishes_visual_unload_once(self):
        tracker = SlotGroupTracker(CENTERS, confirm_seconds=0., minimum_frames=2)
        for now, state in ((0., 'filled'), (.1, 'filled'), (.2, 'empty'), (.3, 'empty'), (1., 'empty')):
            result = tracker.update([detection(4, state)], now)
            self.assertEqual(result['new_events'], [])
        result = tracker.update([detection(4, 'empty')], 1.15)
        event, = result['new_events']
        self.assertEqual((event['time_upper'], event['confirmed_at'], event['reported_at']), (.2, .3, 1.15))
        self.assertEqual(tracker.update([detection(4, 'empty')], 1.2)['new_events'], [])


class S4PredictionCacheTests(unittest.TestCase):
    def tracker(self):
        tracker = SlotGroupTracker(CENTERS, confirm_seconds=0., minimum_frames=2,
            load_hand_contact_seconds=0., load_hand_contact_min_frames=2,
            load_hand_contact_overlap_ratio=.03,
            s4_contact_prediction_seconds=.8)
        tracker.slots[3].geometry_inferred = True
        tracker.update([detection(i, 'filled') for i in (1, 2, 3)], 0.)
        return tracker

    def test_box_only_overlap_waits_until_post_release_empty_is_confirmed(self):
        tracker = self.tracker()
        # Overlap S4's left edge without covering its center.
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [178, 185, 190, 215]}
        tracker.update([detection(1, 'filled'), hand], .05)
        gap = tracker.update([detection(1, 'filled')], .075)
        self.assertIsNone(gap['slots'][3]['prediction_contact'])
        for now in (.1, .2):
            result = tracker.update([detection(1, 'filled'), hand], now)
            slot = result['slots'][3]
            self.assertEqual(slot['tracking_box_xyxy'], [180., 180., 220., 220.])
            self.assertEqual(slot['tracking_box_xyxy'],
                             result['s4_contact_diagnostic']['temporary_contact_box_xyxy'])
            self.assertEqual(slot['last_valid_box_at'], 0.)
            self.assertTrue(result['s4_contact_diagnostic']['temporary_contact_detected'])
            self.assertFalse(result['s4_contact_diagnostic']['contact_accepted'])
        self.assertFalse(slot['load_contact_pending'])
        self.assertEqual(slot['load_contact_count'], 0)
        self.assertTrue(slot['prediction_contact']['pending'])
        self.assertEqual(slot['prediction_contact']['frames'], 2)
        self.assertEqual(slot['prediction_contact']['since'], .1)
        for now in (.3, .9):
            result = tracker.update([detection(1, 'filled')], now)
        self.assertTrue(result['slots'][3]['prediction_contact']['pending'])
        self.assertEqual(tracker.events, [])
        self.assertEqual(tracker.slots[3].confirmed, 'unknown')
        for now in (1., 1.1):
            result = tracker.update([detection(1, 'filled'), detection(4, 'empty')], now)
        event, = tracker.events
        self.assertEqual((event['slot_id'], event['from'], event['to']), (4, 'unknown', 'empty'))
        self.assertTrue(event['inferred'])
        self.assertEqual(event['time_upper'], 1.)
        self.assertIsNone(result['slots'][3]['prediction_contact'])
        self.assertEqual(result['s4_contact_diagnostic']['prediction_contact']['status'],
                         'visual_empty_after_prediction_contact')

    def test_prediction_edge_touch_does_not_arm_visual_inference(self):
        tracker = self.tracker()
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [170, 190, 182.5, 210]}
        for now in (.1, .2):
            result = tracker.update([detection(1, 'filled'), hand], now)
            self.assertTrue(result['s4_contact_diagnostic']['temporary_contact_detected'])
            self.assertIsNone(result['slots'][3]['prediction_contact'])
        for now in (.3, .4):
            tracker.update([detection(1, 'filled'), detection(4, 'empty')], now)
        self.assertEqual(tracker.events, [])

    def tracker_with_prediction_contact(self, before_state):
        tracker = self.tracker()
        support = [detection(i, 'filled') for i in (1, 2, 3)]
        for now in (.05, .1):
            tracker.update(support+[detection(4, before_state)], now)
        gray = detection(4, 'filled', centers=CENTERS+[-45, 0], score=.08)
        for now in (.15, .2, .25):
            tracker.update(support, now, s4_position_candidates=[gray])
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [145, 185, 165, 215]}
        for now in (.3, .35):
            tracker.update(support+[hand], now)
        self.assertTrue(tracker.slots[3].prediction_contact.pending)
        return tracker, support

    def test_prediction_contact_is_attached_to_a_real_visual_transition(self):
        tracker, support = self.tracker_with_prediction_contact('filled')
        for now in (.4, .45):
            result = tracker.update(support+[detection(4, 'empty')], now)
        event, = result['new_events']
        self.assertEqual((event['slot_id'], event['from'], event['to']), (4, 'filled', 'empty'))
        self.assertFalse(event.get('inferred', False))
        self.assertEqual(event['time_upper'], .4)
        self.assertEqual(event['prediction_contact']['since'], .3)
        self.assertEqual(event['prediction_contact']['status'], 'visual_change_confirmed')
        self.assertIsNone(result['slots'][3]['prediction_contact'])

    def test_already_empty_slot_stays_empty_without_unload_event(self):
        tracker, support = self.tracker_with_prediction_contact('empty')
        for now in (.4, .45):
            result = tracker.update(support+[detection(4, 'empty')], now)
        self.assertEqual(tracker.events, [])
        self.assertEqual(result['s4_contact_diagnostic']['prediction_contact']['status'],
                         'visual_state_unchanged')
        self.assertIsNone(result['slots'][3]['prediction_contact'])

    def test_prediction_expiry_keeps_original_center_contact(self):
        tracker = self.tracker()
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [185, 185, 215, 215]}
        result = tracker.update([detection(1, 'filled'), hand], .81)
        slot = result['slots'][3]
        self.assertIsNone(slot['tracking_box_xyxy'])
        self.assertEqual(slot['temporary_contact_box_reason'], 'prediction_expired')
        self.assertEqual(slot['load_contact_count'], 1)
        self.assertTrue(result['s4_contact_diagnostic']['contact_accepted'])

    def test_fresh_prediction_renews_cache_and_visual_replaces_it(self):
        tracker = self.tracker()
        tracker.update([detection(i, 'filled') for i in (1, 2, 3)], 1.)
        self.assertEqual(tracker.slots[3].last_valid_box_at, 1.)
        tracker.update([detection(i, 'filled') for i in range(1, 5)], 1.1)
        slot = tracker.slots[3]
        self.assertEqual(slot.last_valid_box_source, 'detection')
        self.assertEqual(slot.last_valid_center_xy, (200., 200.))
        self.assertIsNone(slot.temporary_contact_box_xyxy)

    def test_hand_departure_clears_box_and_waits_for_verification(self):
        tracker = self.tracker()
        center_hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [185, 185, 215, 215]}
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [178, 185, 190, 215]}
        for now in (.1, .15):
            tracker.update([detection(1, 'filled'), center_hand], now)
        result = tracker.update([detection(1, 'filled'), hand], .2)
        self.assertEqual(result['slots'][3]['load_contact_count'], 2)
        self.assertIsNone(result['slots'][3]['load_contact_released_at'])
        self.assertEqual(result['s4_contact_diagnostic']['accumulation_status'],
                         'pending_contact_during_prediction')
        result = tracker.update([detection(1, 'filled')], .3)
        self.assertIsNone(result['slots'][3]['tracking_box_xyxy'])
        self.assertFalse(result['s4_contact_diagnostic']['temporary_contact_box_active'])
        self.assertTrue(result['slots'][3]['load_contact_pending'])
        for now in (.4, .5):
            result = tracker.update([detection(1, 'filled'), detection(4, 'empty')], now)
        self.assertEqual(result['new_events'][0]['slot_id'], 4)
        self.assertEqual(result['new_events'][0]['action'], 'unload')

    def test_expiry_during_held_contact_waits_for_vision_without_starting_release(self):
        tracker = self.tracker()
        center_hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [185, 185, 215, 215]}
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [178, 185, 190, 215]}
        for now in (.1, .15):
            tracker.update([detection(1, 'filled'), center_hand], now)
        tracker.update([detection(1, 'filled'), hand], .75)
        for now in (.81, 1.4):
            result = tracker.update([detection(1, 'filled'), hand], now)
            self.assertIsNone(result['slots'][3]['tracking_box_xyxy'])
            self.assertIsNone(result['slots'][3]['load_contact_released_at'])
            self.assertEqual(result['s4_contact_diagnostic']['accumulation_status'],
                             'pending_contact_waiting_for_visual')
        self.assertEqual(tracker.events, [])
        tracker.update([detection(1, 'filled')], 2.)
        self.assertFalse(tracker.slots[3].load_contact_pending)
        self.assertEqual(tracker.events, [])

    def test_candidate_group_with_internal_points_cannot_use_cache(self):
        tracker = SlotGroupTracker(require_unload_event_before_numbering=True,
                                   initialization_seconds=0., initialization_frames=1)
        tracker.update([detection(i, 'filled') for i in range(1, 5)], 0.)
        self.assertIsNone(tracker.points)
        candidate = tracker._provisional
        self.assertIsNotNone(candidate.points)
        self.assertIsNone(candidate.slots[3].last_valid_box_at)
        tracker.update([detection(i, 'filled') for i in (1, 2, 3)], .1)
        self.assertIsNone(candidate.slots[3].temporary_contact_box_xyxy)

    def test_identity_loss_and_group_reset_discard_prediction(self):
        tracker = self.tracker()
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [178, 185, 190, 215]}
        for now in (.05, .1):
            tracker.update([detection(1, 'filled'), hand], now)
        self.assertTrue(tracker.slots[3].prediction_contact.pending)
        tracker.identity_lost = True
        result = tracker.update([], .15)
        self.assertIsNone(result['slots'][3]['tracking_box_xyxy'])
        self.assertIsNone(result['slots'][3]['prediction_contact'])
        self.assertEqual(result['slots'][3]['temporary_contact_box_reason'], 'identity_lost')
        tracker.reset_group(2)
        self.assertIsNone(tracker.slots[3].last_valid_box_at)


class S4PositionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tracker = SlotGroupTracker(CENTERS, match_distance_ratio=.4)
        self.support = [detection(i, 'filled', track_id=i) for i in (1, 2, 3)]
        self.gray_box = detection(4, 'empty', centers=CENTERS+[-45, 0], score=.08)

    def test_stable_gray_box_updates_position_only_and_survives_a_gap(self):
        roi = np.float32([[50, 50], [250, 50], [250, 250], [50, 250]])
        sequence = RoiGroupSequence(self.tracker, roi)
        for now in (0., .05):
            result = sequence.update(self.support, now, s4_position_candidates=[self.gray_box])
            self.assertNotEqual(result['slots'][3]['last_valid_box_source'], 'unassigned_detection')
        result = sequence.update(self.support, .1, s4_position_candidates=[self.gray_box])
        slot = result['slots'][3]
        self.assertEqual(slot['center_xy'], [155., 200.])
        self.assertEqual(slot['temporary_contact_box_xyxy'], [135., 180., 175., 220.])
        self.assertEqual(slot['last_valid_box_source'], 'unassigned_detection')
        self.assertIsNone(slot['byte_track_id'])
        self.assertEqual(slot['last_confirmed_state'], 'unknown')
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [145, 185, 165, 215]}
        for now in (.15, .2):
            result = sequence.update(self.support+[hand], now)
            self.assertEqual(result['slots'][3]['center_xy'], [155., 200.])
            self.assertTrue(result['s4_contact_diagnostic']['temporary_contact_detected'])
            self.assertEqual(result['slots'][3]['load_contact_count'], 0)
        self.assertEqual(sequence.all_events, [])

    def test_unassigned_tracked_box_requires_consecutive_observations(self):
        tracked_box = {**self.gray_box, 'track_id': 14, 'confidence': .9}
        for now in (0., .05):
            self.tracker.update(self.support+[tracked_box], now)
        self.tracker.update(self.support, .1)
        for now in (.15, .2):
            result = self.tracker.update(self.support+[tracked_box], now)
            self.assertNotEqual(result['slots'][3]['last_valid_box_source'], 'unassigned_detection')
        result = self.tracker.update(self.support+[tracked_box], .25)
        self.assertEqual(result['slots'][3]['center_xy'], [155., 200.])
        self.assertIsNone(result['slots'][3]['byte_track_id'])

    def test_real_s4_takes_priority_over_gray_box(self):
        for now in (0., .05, .1):
            result = self.tracker.update(self.support+[detection(4, 'filled', track_id=4)], now,
                                         s4_position_candidates=[self.gray_box])
        self.assertEqual(result['slots'][3]['center_xy'], [200., 200.])
        self.assertEqual(result['slots'][3]['last_valid_box_source'], 'detection')
        self.assertIsNone(result['slots'][3]['temporary_contact_box_xyxy'])

    def test_gray_box_overlapping_old_group_track_is_excluded(self):
        tracker = SlotGroupTracker(CENTERS, group_id=2, match_distance_ratio=.4)
        roi = np.float32([[50, 50], [250, 50], [250, 250], [50, 250]])
        sequence = RoiGroupSequence(tracker, roi)
        old_box = {**self.gray_box, 'track_id': 14, 'confidence': .9}
        old_group = ('unload', 1)
        sequence.track_store.observe([old_box], -.05)
        self.assertTrue(sequence.track_store.bind(14, old_group, 4))
        sequence.track_store.closed.add(old_group)
        for now in (0., .05, .1):
            result = sequence.update(self.support+[old_box], now,
                                     s4_position_candidates=[self.gray_box])
        self.assertNotEqual(result['slots'][3]['last_valid_box_source'], 'unassigned_detection')
        self.assertIsNone(result['slots'][3]['byte_track_id'])


class S4MeasuredLayoutTests(unittest.TestCase):
    def setUp(self):
        self.layout = np.array([[100, 100], [100, 200], [200, 100], [184, 190]], dtype=float)
        tracker = SlotGroupTracker(self.layout, slot2_below_slot3=True,
            confirm_seconds=0., minimum_frames=1, initialization_frames=2,
            unload_entry_numbering_seconds=.2)
        self.roi = np.float32([[40, 40], [300, 40], [300, 300], [40, 300]])
        self.sequence = RoiGroupSequence(tracker, self.roi, entry_seconds=.05,
                                         use_previous_s4_layouts=True)

    def test_collects_stable_real_slots_and_projects_before_numbering(self):
        all_slots = [detection(i, 'filled', centers=self.layout, track_id=i)
                     for i in range(1, 5)]
        self.sequence.update(all_slots, 0.)
        self.sequence.update(all_slots[:3], .05)
        self.assertNotIn(1, self.sequence._unload_layouts)
        for now in (.1, .15, .2):
            self.sequence.update(all_slots, now)
        np.testing.assert_allclose(np.median(self.sequence._unload_layouts[1], axis=0), self.layout)
        self.sequence._unload_layouts[2] = [self.layout+[-15, 8]]

        result = self.enter_with_three_slots()
        self.assertEqual(result['slot_numbering_status'], 'provisional')
        self.assertEqual(result['initialization']['s4_reference_group_ids'], [1, 2])
        self.assertEqual(result['slots'][3]['center_xy'], [209., 202.])
        self.assertTrue(result['slots'][3]['position_is_prediction'])

    def test_disagreeing_references_keep_three_point_estimate(self):
        different_layout = self.layout.copy()
        different_layout[3] = [220., 225.]
        self.sequence._unload_layouts.update({1: [self.layout], 2: [different_layout]})
        result = self.enter_with_three_slots()
        self.assertEqual(result['initialization']['s4_reference_group_ids'], [])
        self.assertEqual(result['slots'][3]['center_xy'], [225., 212.])

    def test_recalculates_once_after_three_real_slots_stabilize_again(self):
        tracker = SlotGroupTracker(initialization_seconds=.05, initialization_frames=2,
            unload_entry_numbering_seconds=10.)
        tracker.start_incremental_entry(40.)
        layout = np.array([[100, 100], [100, 200], [200, 100], [200, 200]], dtype=float)
        visible = [detection(i, 'filled', centers=layout, track_id=i) for i in (1, 2, 3)]
        tracker.update(visible, 0.)
        initial = tracker.update(visible, .05)
        self.assertEqual(initial['slots'][3]['center_xy'], [200., 200.])
        tracker.update([visible[0], visible[2]], .10)

        layout[1, 1] = 230.
        visible[1] = detection(2, 'filled', centers=layout, track_id=20)
        recovered = tracker.update(visible, .15)
        self.assertNotIn('s4_last_recalculated_at', recovered['initialization'])
        stable = tracker.update(visible, .20)
        self.assertEqual(stable['initialization']['s4_last_recalculated_at'], .20)
        self.assertEqual(stable['slots'][3]['center_xy'], [200., 230.])
        self.assertTrue(stable['slots'][3]['position_is_prediction'])
        self.assertIsNone(stable['slots'][3]['byte_track_id'])
        follow = tracker.update(visible, .25)
        self.assertEqual(follow['initialization']['s4_last_recalculated_at'], .20)

    def enter_with_three_slots(self):
        entry = self.sequence._new_tracker(3, entry=True)
        entry.start_incremental_entry(40.)
        current = self.layout+[25., 12.]
        visible = [detection(i, 'filled', centers=current, track_id=i+10) for i in (1, 2, 3)]
        entry.update(visible, 1., roi_contour=self.roi)
        return entry.update(visible, 1.06, roi_contour=self.roi)


if __name__ == '__main__':
    unittest.main()
