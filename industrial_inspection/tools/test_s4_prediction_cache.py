"""Regression coverage for S4 inferred before its first real detection."""
import unittest

import numpy as np

from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_track_state import SlotTrackStateStore
from industrial_inspection.tools.test_slot_group_tracker import CENTERS, detection


class EntryS4ContactTests(unittest.TestCase):
    centers = CENTERS[[0, 2, 1, 3]]
    roi = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

    def setUp(self):
        self.tracker = SlotGroupTracker(None, group_id=3, slot2_below_slot3=True,
            confirm_seconds=.08, minimum_frames=2, initialization_frames=2,
            initialization_seconds=.05, geometry_initialization_seconds=.05,
            unload_entry_numbering_seconds=.6, load_hand_contact_seconds=0.)
        self.tracker.start_incremental_entry(40)
        self.tracker.track_store = SlotTrackStateStore(self.tracker, self.roi)
        filled = [self.box(i, 'filled', i) for i in range(1, 5)]
        for now in (0., .1):
            self.frame(filled, now)
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [188, 188, 212, 212]}
        for now in (.2, .25):
            self.frame(filled[:3] + [hand], now)
        self.frame(filled, .3)
        # The old S4 ID now tracks S3. The new S4 ID has no filled history.
        self.switched = filled[:2] + [self.box(3, 'filled', 4), self.box(4, 'empty', 40)]

    def box(self, slot, state, track_id):
        return detection(slot, state, self.centers, track_id=track_id)

    def frame(self, objects, now):
        self.tracker.track_store.observe(objects, now)
        return self.tracker.update(objects, now, roi_contour=self.roi)

    def test_recovers_early_unload_when_new_track_is_formally_numbered_s4(self):
        for now in (.4, .5, .6, .7, .8, .9, 1.):
            self.assertEqual(self.frame(self.switched, now)['new_events'], [])
        result = self.frame(self.switched, 1.1)
        event, = result['new_events']
        self.assertEqual(self.tracker.byte_track_ids, [1, 2, 4, 40])
        self.assertEqual((event['slot_id'], event['contact_byte_track_id'], event['byte_track_id']), (4, 4, 40))
        self.assertEqual((event['time_upper'], event['confirmed_at'], event['reported_at']), (.5, .6, 1.1))
        self.assertEqual(self.frame(self.switched, 1.2)['new_events'], [])

    def test_ambiguous_frames_interrupt_empty_confirmation_across_numbering(self):
        distractor = {**self.box(4, 'empty', 41), 'xyxy': [195, 180, 235, 220]}
        for tick in range(4, 12):
            objects = self.switched + ([distractor] if tick % 2 else [])
            self.assertEqual(self.frame(objects, tick / 10)['new_events'], [])
        self.assertEqual(self.tracker.byte_track_ids, [1, 2, 4, 40])
        self.assertEqual(self.frame(self.switched, 1.2)['new_events'], [])
        event, = self.frame(self.switched, 1.3)['new_events']
        self.assertEqual((event['slot_id'], event['byte_track_id']), (4, 40))
        self.assertEqual((event['time_upper'], event['confirmed_at'], event['reported_at']), (1.2, 1.3, 1.3))

    def test_expired_contact_does_not_recover_a_late_empty_track(self):
        for now in (1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3):
            self.assertEqual(self.frame(self.switched, now)['new_events'], [])
        self.assertEqual(self.tracker.byte_track_ids, [1, 2, 4, 40])
        self.assertEqual(self.tracker.slots[3].confirmed, 'empty')


class S4UnloadReviewTests(unittest.TestCase):
    def test_second_contact_cannot_reverse_unload_inferred_before_numbering(self):
        tracker = SlotGroupTracker(CENTERS, confirm_seconds=0., minimum_frames=2,
                                   load_hand_contact_seconds=0.)
        support = [detection(i, 'filled') for i in (1, 2, 3)]
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [188, 188, 212, 212]}
        for now in (0., .1):
            tracker.update(support + [detection(4, 'filled')], now)
        for now in (.2, .3):
            tracker.update(support + [hand], now)
        for now in (.4, .7, .95):
            tracker.update(support, now)
        event, = tracker.events
        event['recovered_before_initialization'] = True

        for now in (1., 1.1):
            tracker.update(support + [hand], now)
        for now in (1.2, 1.5, 1.75):
            result = tracker.update(support, now)
        self.assertEqual(result['new_events'], [])
        self.assertEqual(tracker.events, [event])
        self.assertEqual(result['slots'][3]['last_confirmed_state'], 'empty')

        tracker.update(support + [detection(4, 'filled')], 1.8)
        result = tracker.update(support + [detection(4, 'filled')], 1.9)
        refill, = result['new_events']
        self.assertEqual((refill['slot_id'], refill['action']), (4, 'load'))

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

    def test_held_prediction_keeps_original_center_contact(self):
        tracker = self.tracker()
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [185, 185, 215, 215]}
        result = tracker.update([detection(1, 'filled'), hand], .81)
        slot = result['slots'][3]
        self.assertEqual(slot['tracking_box_xyxy'], [180., 180., 220., 220.])
        self.assertTrue(slot['position_is_held'])
        self.assertEqual(slot['temporary_contact_box_xyxy'], slot['tracking_box_xyxy'])
        self.assertEqual(slot['temporary_contact_box_reason'], 'held_position')
        self.assertEqual(slot['load_contact_count'], 1)
        self.assertTrue(result['s4_contact_diagnostic']['contact_accepted'])

    def test_fresh_prediction_renews_cache_and_visual_replaces_it(self):
        tracker = self.tracker()
        tracker.update([detection(i, 'filled') for i in (1, 2, 3)], 1.)
        self.assertEqual(tracker.slots[3].last_valid_box_at, 1.)
        moved = CENTERS + [8., 6.]
        tracker.update([detection(i, 'filled', centers=moved) for i in range(1, 5)], 1.1)
        slot = tracker.slots[3]
        self.assertEqual(slot.last_valid_box_source, 'detection')
        self.assertEqual(slot.last_valid_center_xy, (208., 206.))
        self.assertIsNone(slot.temporary_contact_box_xyxy)

    def test_hand_departure_keeps_box_and_waits_for_verification(self):
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
        self.assertEqual(result['slots'][3]['tracking_box_xyxy'], [180., 180., 220., 220.])
        self.assertTrue(result['slots'][3]['position_is_held'])
        self.assertTrue(result['s4_contact_diagnostic']['temporary_contact_box_active'])
        self.assertTrue(result['slots'][3]['load_contact_pending'])
        for now in (.4, .5):
            result = tracker.update([detection(1, 'filled'), detection(4, 'empty')], now)
        self.assertEqual(result['new_events'][0]['slot_id'], 4)
        self.assertEqual(result['new_events'][0]['action'], 'unload')

    def test_held_contact_uses_cached_box_until_the_hand_leaves(self):
        tracker = self.tracker()
        center_hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [185, 185, 215, 215]}
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [178, 185, 190, 215]}
        for now in (.1, .15):
            tracker.update([detection(1, 'filled'), center_hand], now)
        tracker.update([detection(1, 'filled'), hand], .75)
        for now in (.81, 1.4):
            result = tracker.update([detection(1, 'filled'), hand], now)
            self.assertEqual(result['slots'][3]['temporary_contact_box_xyxy'], [180., 180., 220., 220.])
            self.assertIsNone(result['slots'][3]['load_contact_released_at'])
            self.assertEqual(result['s4_contact_diagnostic']['accumulation_status'],
                             'pending_contact_during_prediction')
        self.assertEqual(tracker.events, [])
        result = tracker.update([detection(1, 'filled')], 1.5)
        self.assertEqual(result['slots'][3]['load_contact_released_at'], 1.5)
        self.assertTrue(result['slots'][3]['position_is_held'])
        self.assertEqual(result['slots'][3]['temporary_contact_box_xyxy'], [180., 180., 220., 220.])
        self.assertEqual(tracker.events, [])

    def test_prediction_only_contact_can_complete_after_expiry(self):
        tracker = self.tracker()
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [178, 185, 190, 215]}
        for now in (.75, .81, .9):
            result = tracker.update([detection(1, 'filled'), hand], now)
        slot = result['slots'][3]
        self.assertEqual(slot['last_valid_box_at'], 0.)
        self.assertEqual(slot['prediction_contact']['frames'], 3)
        self.assertTrue(slot['prediction_contact']['pending'])
        self.assertEqual(slot['prediction_contact']['since'], .75)
        self.assertEqual(tracker.events, [])
        released = tracker.update([detection(1, 'filled')], .95)['slots'][3]
        self.assertEqual(released['prediction_contact']['released_at'], .95)
        self.assertEqual(released['temporary_contact_box_xyxy'], slot['tracking_box_xyxy'])
        self.assertEqual(released['tracking_box_xyxy'], slot['tracking_box_xyxy'])
        for now in (1., 1.1):
            tracker.update([detection(1, 'filled'), detection(4, 'empty')], now)
        event, = tracker.events
        self.assertEqual(event['reason'], 'empty_after_prediction_contact')

    def test_held_box_can_start_contact_and_still_needs_visual_confirmation(self):
        tracker = self.tracker()
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [178, 185, 190, 215]}
        for now in (.81, .9):
            slot = tracker.update([detection(1, 'filled'), hand], now)['slots'][3]
            self.assertTrue(slot['position_is_held'])
            self.assertEqual(slot['temporary_contact_box_xyxy'], slot['tracking_box_xyxy'])
            self.assertEqual(slot['last_valid_box_at'], 0.)
        self.assertTrue(slot['prediction_contact']['pending'])
        self.assertTrue(slot['prediction_contact']['started_on_held_box'])
        for now in (.95, 1.5):
            tracker.update([detection(1, 'filled')], now)
            self.assertEqual(tracker.events, [])
        for now in (1.6, 1.7):
            tracker.update([detection(1, 'filled'), detection(4, 'empty')], now)
        event, = tracker.events
        self.assertEqual(event['time_upper'], .95)
        self.assertEqual(event['visual_time_upper'], 1.6)

    def test_held_box_contact_overlapping_s3_requires_that_exact_unload_to_survive_review(self):
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [95, 183, 193, 217]}
        for reverses in (False, True):
            with self.subTest(reverses=reverses):
                tracker = self.tracker()
                tracker.visual_reversal_review_seconds = .5
                tracker.update([detection(i, 'filled') for i in (1, 2, 3)], .1)
                for now in (.2, .3):
                    tracker.update([detection(1, 'empty'), detection(2, 'filled'), detection(3, 'filled')], now)
                tracker.update([detection(2, 'filled'), hand], .85)
                self.assertIsNone(tracker.slots[3].prediction_contact)
                for now in (.91, .97):
                    tracker.update([detection(2, 'filled'), detection(3, 'empty')], now)
                for now in (1.11, 1.16):
                    result = tracker.update([detection(2, 'filled'), hand], now)
                evidence = result['slots'][3]['prediction_contact']
                self.assertTrue(evidence['pending'])
                self.assertEqual(evidence['competing_unloads'], ((3, .97),))
                support = [detection(2, 'filled')]
                if reverses:
                    support.append(detection(3, 'filled'))
                for now in (1.2, 1.25, 1.3):
                    result = tracker.update(support+[detection(4, 'empty')], now)
                self.assertEqual([e['slot_id'] for e in tracker.events], [1])
                status = result['s4_contact_diagnostic']['prediction_contact']['status'] if not reverses else None
                if not reverses:
                    self.assertEqual(status, 'waiting_for_competing_unloads')
                tracker.update(support+[detection(4, 'empty')], 1.5)
                expected = [1] if reverses else [1, 3, 4]
                self.assertEqual([e['slot_id'] for e in tracker.events], expected)

    def test_confirmed_motion_clears_cached_position_and_extended_contact(self):
        tracker = self.tracker()
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [178, 185, 190, 215]}
        tracker.update([detection(1, 'filled'), hand], .75)
        result = tracker.update([detection(i, 'filled') for i in (1, 2, 3)] + [hand], .81,
                                before_observe=lambda *_: False)
        slot = result['slots'][3]
        self.assertIsNone(slot['last_valid_box_at'])
        self.assertIsNone(slot['center_xy'])
        self.assertIsNone(slot['tracking_box_xyxy'])
        self.assertIsNone(slot['prediction_contact'])
        self.assertEqual(slot['temporary_contact_box_reason'], 'rotation_or_transition')

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
        self.assertIsNone(result['slots'][3]['last_valid_center_xy'])
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

    def test_load_binds_recovered_s4_after_three_consecutive_frames_with_three_matched_anchors(self):
        tracker = SlotGroupTracker(CENTERS, phase='load', match_distance_ratio=.4,
            confirm_seconds=0., minimum_frames=2)
        tracker.slots[3].geometry_inferred = True
        roi = np.float32([[50, 50], [250, 50], [250, 250], [50, 250]])
        sequence = RoiGroupSequence(tracker, roi)
        measured = {**self.gray_box, 'track_id': 14, 'confidence': .9}

        for now in (0., .05, .1):
            result = sequence.update(self.support[:2]+[measured], now)
            self.assertIsNone(result['slots'][3]['byte_track_id'])
        self.assertEqual(result['slots'][3]['last_valid_box_source'], 'unassigned_detection')

        sequence.update(self.support, .15)
        for now in (.2, .25):
            objects = self.support[:2]+[measured]
            result = sequence.update(objects, now)
            self.assertIsNone(result['slots'][3]['byte_track_id'])
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [85, 85, 115, 115]}
        result = sequence.update(self.support+[measured, hand], .3)
        self.assertEqual(result['slots'][3]['byte_track_id'], 14)
        self.assertEqual(result['slots'][3]['center_xy'], [155., 200.])
        self.assertFalse(result['slots'][3]['geometry_inferred'])
        self.assertEqual(sequence.track_store.owner(14), ('load', 1))
        self.assertFalse(any(o['track_id'] == 14 for o in result['unassigned_slot_detections']))

        sequence.update(self.support+[measured], .35)
        filled = dict(measured, class_name='wafer_slot_filled')
        for now in (.4, .45):
            result = sequence.update(self.support+[filled], now)
        event, = result['new_events']
        self.assertEqual((event['slot_id'], event['action'], event['byte_track_id']), (4, 'load', 14))
        self.assertFalse(event.get('inferred'))

    def test_binding_s4_does_not_import_unbound_load_history(self):
        for already_reported in (False, True):
            with self.subTest(already_reported=already_reported):
                tracker = SlotGroupTracker(CENTERS, phase='load', match_distance_ratio=.4,
                    confirm_seconds=0., minimum_frames=2)
                slot = tracker.slots[3]
                slot.geometry_inferred = True
                roi = np.float32([[50, 50], [250, 50], [250, 250], [50, 250]])
                sequence = RoiGroupSequence(tracker, roi)
                measured = {**self.gray_box, 'track_id': 14, 'confidence': .9}
                for now in (0., .05):
                    sequence.update(self.support[:2]+[measured], now)
                slot.confirmed = slot.state = 'filled' if already_reported else 'empty'
                slot.confirmation_source = 'hand_contact' if already_reported else 'previous_unload_yolo'
                slot.evidence_time = .05
                if already_reported:
                    tracker.events.append({'group_id': 1, 'phase': 'load', 'slot_id': 4, 'action': 'load',
                        'from': 'empty', 'to': 'filled', 'expected_action': True, 'inferred': True,
                        'time_lower': .05, 'time_upper': .1, 'confirmed_at': .15})
                filled = dict(measured, class_name='wafer_slot_filled')
                for now in (.1, .15):
                    sequence.update(self.support[:2]+[filled], now)
                history = sequence.track_store.tracks[14]['events']
                self.assertEqual([event['action'] for event in history], ['load'])

                result = sequence.update(self.support+[filled], .2)
                self.assertEqual(result['slots'][3]['byte_track_id'], 14)
                self.assertEqual(result['new_events'], [])
                for now in (.25, .3):
                    result = sequence.update(self.support+[filled], now)
                event, = tracker.events
                self.assertEqual(event['time_upper'], .1 if already_reported else .2)
                self.assertEqual(bool(event.get('inferred')), already_reported)

    def test_load_binds_the_missing_s2_without_using_s4_contact(self):
        tracker = SlotGroupTracker(CENTERS, phase='load', match_distance_ratio=.4,
            confirm_seconds=0., minimum_frames=2)
        tracker._load_numbering_reference = CENTERS.copy()
        tracker.slots[1].geometry_inferred = True
        support = [detection(i, 'empty', track_id=i) for i in (1, 3, 4)]
        measured = detection(2, 'empty', centers=CENTERS+[-45, 0], track_id=14)
        for now in (0., .05, .1, .15):
            result = tracker.update(support+[measured], now)
        self.assertEqual([s['byte_track_id'] for s in result['slots']], [1, 14, 3, 4])
        self.assertIsNone(result['slots'][1]['temporary_contact_box_xyxy'])
        self.assertEqual(result['slots'][1]['last_confirmed_state'], 'empty')
        filled = dict(measured, class_name='wafer_slot_filled')
        for now in (.2, .25):
            result = tracker.update(support+[filled], now)
        event, = result['new_events']
        self.assertEqual((event['slot_id'], event['action'], event['byte_track_id']), (2, 'load', 14))
        self.assertFalse(event.get('inferred'))

    def test_load_hand_inference_follows_the_missing_slot(self):
        tracker = SlotGroupTracker(CENTERS, phase='load', confirm_seconds=0.,
            minimum_frames=2, load_hand_contact_seconds=.05, load_hand_contact_min_frames=2,
            s4_hand_region=np.float32([[175, 175], [225, 175], [225, 225], [175, 225]]))
        tracker._load_numbering_reference = CENTERS.copy()
        tracker.slots[1].geometry_inferred = True
        support = [detection(i, 'empty', track_id=i) for i in (1, 3, 4)]
        s2_hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [184, 81, 196, 98]}
        s4_hand = {**s2_hand, 'xyxy': [180, 180, 220, 220]}
        for now in (0., .05):
            tracker.update(support+[detection(2, 'empty', track_id=2)], now)
        tracker.slots[1].geometry_inferred = True
        for now in (.1, .15):
            result = tracker.update(support+[s2_hand], now)
        self.assertEqual(result['slots'][1]['load_contact_count'], 0)

        tracker._load_verified_empty_slots = {2}
        for now in (.2, .25):
            result = tracker.update(support+[s4_hand], now)
        self.assertEqual(result['slots'][1]['load_contact_count'], 0)
        self.assertEqual(result['slots'][3]['load_contact_count'], 0)
        for now in (.3, .35):
            result = tracker.update(support+[s2_hand], now)
        self.assertTrue(result['slots'][1]['load_contact_pending'])
        self.assertEqual(result['s4_contact_diagnostic']['target_slot_id'], 2)
        self.assertEqual(result['s4_contact_diagnostic']['source'], 'predicted_slot_box')
        self.assertEqual(result['slots'][1]['temporary_contact_box_xyxy'], [180., 80., 220., 120.])

        tracker.update(support, .4)
        result = tracker.update(support, .9)
        event, = result['new_events']
        self.assertEqual((event['slot_id'], event['action'], event['reason']),
                         (2, 'load', 'hand_contact_inferred_load'))
        self.assertTrue(event['inferred'])
        for now in (1., 1.05, 1.1, 1.65):
            tracker.update(support+([s2_hand] if now <= 1.05 else []), now)
        self.assertEqual(len(tracker.events), 1)

    def test_load_contact_box_moves_with_the_inferred_slot_point(self):
        tracker = SlotGroupTracker(CENTERS, phase='load')
        tracker.slots[1].geometry_inferred = True
        for now, shift in ((0., [0., 0.]), (.05, [9., 5.])):
            support = [detection(i, 'empty', centers=CENTERS+shift, track_id=i) for i in (1, 3, 4)]
            slot = tracker.update(support, now)['slots'][1]
            np.testing.assert_allclose(slot['center_xy'], CENTERS[1]+shift, atol=.01)
            self.assertTrue(slot['position_is_prediction'])
            box = np.asarray(slot['temporary_contact_box_xyxy']).reshape(2, 2)
            np.testing.assert_allclose(box.mean(axis=0), slot['center_xy'], atol=.01)
            np.testing.assert_allclose(box[1]-box[0], [40, 40], atol=.01)
            self.assertEqual(slot['tracking_box_xyxy'], slot['temporary_contact_box_xyxy'])

    def test_load_resumes_prediction_when_an_observed_slot_goes_missing(self):
        tracker = SlotGroupTracker(CENTERS, phase='load', s4_contact_prediction_seconds=.8,
            confirm_seconds=0., minimum_frames=2, load_hand_contact_seconds=0.)
        tracker._load_numbering_reference = CENTERS.copy()
        tracker._load_verified_empty_slots = {2}
        all_slots = [detection(i, 'empty', track_id=i) for i in range(1, 5)]
        for now in (0., .05):
            slot = tracker.update(all_slots, now)['slots'][1]
        self.assertFalse(slot['position_is_prediction'])
        self.assertEqual(slot['byte_track_id'], 2)

        moved = [detection(i, 'empty', centers=CENTERS+[8., 4.], track_id=i) for i in range(1, 5)]
        result = tracker.update([moved[0], moved[2], moved[3]], .1)
        slot = result['slots'][1]
        self.assertEqual(slot['center_xy'], [208., 104.])
        self.assertEqual(slot['temporary_contact_box_xyxy'], [188., 84., 228., 124.])
        self.assertTrue(slot['position_is_prediction'])
        self.assertFalse(slot['position_is_held'])

        support = [moved[0], moved[2]]
        held = tracker.update(support, .2)['slots'][1]
        self.assertTrue(held['position_is_held'])
        self.assertEqual(held['temporary_contact_box_xyxy'], slot['temporary_contact_box_xyxy'])
        hand = {'class_name': 'hand', 'confidence': .95, 'xyxy': [193, 89, 223, 119]}
        for now in (.25, .3):
            result = tracker.update(support+[hand], now)
        self.assertEqual(result['s4_contact_diagnostic']['target_slot_id'], 2)
        self.assertTrue(result['slots'][1]['prediction_contact']['pending'])
        self.assertFalse(result['slots'][1]['load_contact_pending'])
        tracker.update(support, .35)
        expired = tracker.update(support, 1.)['slots'][1]
        self.assertFalse(expired['position_is_prediction'])
        self.assertIsNone(expired['temporary_contact_box_xyxy'])
        self.assertEqual(expired['last_confirmed_state'], 'empty')
        self.assertEqual(tracker.events, [])

        moved[1] = dict(moved[1], class_name='wafer_slot_filled')
        tracker.update(moved, 1.05)
        result = tracker.update(moved, 1.1)
        observed = result['slots'][1]
        self.assertFalse(observed['position_is_prediction'])
        self.assertEqual(observed['byte_track_id'], 2)
        self.assertIsNone(observed['temporary_contact_box_xyxy'])
        event, = result['new_events']
        self.assertEqual((event['slot_id'], event['reason']), (2, 'visual_after_prediction_contact'))
        self.assertEqual((event['time_upper'], event['visual_time_upper']), (.35, 1.05))

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

    def test_load_numbering_keeps_measured_s4_when_s2_is_hidden(self):
        tracker = SlotGroupTracker(phase='load', slot2_below_slot3=True,
            machine_center=(637, 266), initialization_seconds=.05, initialization_frames=2,
            allow_three_point_initialization=True)
        tracker._load_numbering_reference = self.layout.copy()
        tracker.start_incremental_entry(40)
        load = np.array([[463, 308], [573, 331], [434, 244], [547, 270]], dtype=float)
        visible = [detection(i, 'empty', centers=load, track_id=i+100) for i in (1, 3, 4)]
        for now in (0., .05, .1, .15, .2, .25, .3):
            result = tracker.update(visible, now)
        self.assertEqual(result['slot_numbering_status'], 'final')
        self.assertEqual(result['initialization']['inferred_slot_ids'], [2])
        self.assertEqual([s['byte_track_id'] for s in result['slots']], [101, None, 103, 104])

    def test_load_projects_same_group_layout_when_local_guess_leaves_roi(self):
        tracker = SlotGroupTracker(phase='load', machine_center=(637, 266))
        layout = np.array([[441, 289], [547, 323], [456, 242], [543, 267]], dtype=float)
        tracker._load_numbering_reference = np.array(
            [[381.8, 220.2], [501., 274.2], [430.4, 196.8], [530.9, 240.]], dtype=float)
        visible = layout[[0, 1, 3]]
        roi = np.float32([[279, 144], [628, 228], [633, 384], [300, 318]])
        point, groups = tracker._infer_entry_s4(visible, roi)
        np.testing.assert_allclose(point, layout[2], atol=2.)
        self.assertEqual(groups, [1])

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
