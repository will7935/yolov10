"""Offline regression tests for tracks missing at an ROI group handoff."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker


POINTS = np.array([[100, 100], [200, 100], [100, 200], [200, 200]], dtype=float)
ROI = np.array([[80, 80], [410, 80], [410, 310], [80, 310]], dtype=np.float32)


def slot(track_id, center):
    x, y = center
    return dict(class_name='wafer_slot_empty', confidence=.95, track_id=track_id,
                xyxy=[x-10, y-10, x+10, y+10])


def group(points=POINTS, visible=(0, 1, 2, 3)):
    return [slot(101+i, points[i]) for i in visible]


class TransitionQuarantineTests(unittest.TestCase):
    def make(self, *, enabled=True, single_exit=True):
        tracker = SlotGroupTracker(POINTS, confirm_seconds=0, minimum_frames=2,
            lost_seconds=1.2, initialization_seconds=.1, initialization_frames=2)
        sequence = RoiGroupSequence(tracker, ROI, exit_min_slots=1 if single_exit else 3,
            partial_exit_slots=1 if single_exit else 2, exit_direction=(0, -1),
            entry_seconds=.1, reject_transition_roi_detections=enabled)
        for now in (0., .1, .2):
            sequence.update(group(), now)
        return sequence

    def close(self, sequence, now=.5):
        points = POINTS.copy()
        points[0] = [100, 75]
        result = sequence.update(group(points), now)
        self.assertEqual(result['group_id'], 2)
        self.assertEqual(len(sequence.closed_groups), 1)
        return result

    def test_records_all_roi_ids_including_unassigned_and_keeps_closed_history(self):
        sequence = self.make()
        hand = dict(class_name='hand', confidence=.9, track_id=999, xyxy=[330, 130, 370, 170])
        result = sequence.update(group()+[slot(283, [350, 150]), slot(900, [450, 150]), hand], .3)
        self.assertEqual(result['roi_observed_track_ids'], [101, 102, 103, 104, 283])
        self.assertIsNone(sequence.track_store.owner(283))
        self.assertFalse(sequence.is_transition_quarantined_track(283))
        self.close(sequence)
        self.assertEqual(sequence.closed_groups[0]['roi_observed_track_ids'], [101, 102, 103, 104, 283])
        result = sequence.update([slot(309, [310, 270])], .6)
        self.assertEqual(result['roi_observed_track_ids'], [309])
        self.assertIn(283, sequence.closed_groups[0]['roi_observed_track_ids'])

    def test_same_group_missing_track_restores_original_slot_without_quarantine(self):
        sequence = self.make()
        before = sequence.tracker.byte_track_ids.index(101)
        sequence.update(group(visible=(1, 2, 3)), .3)
        result = sequence.update(group(), .7)
        self.assertEqual(result['group_id'], 1)
        self.assertEqual(sequence.tracker.byte_track_ids.index(101), before)
        self.assertEqual(sequence.track_store.owner(101), ('unload', 1))
        record = next(r for r in result['track_states'] if r['byte_track_id'] == 101)
        self.assertEqual(record['mapping_reason'], 'mapped')
        self.assertFalse(sequence.is_transition_quarantined_track(101))
        self.assertEqual(sequence.closed_groups, [])

    def test_unassigned_track_missing_at_handoff_never_enters_established_next_group(self):
        sequence = self.make()
        sequence.update(group()+[slot(283, [350, 150])], .3)
        self.close(sequence)
        self.assertTrue(sequence.is_transition_quarantined_track(283))
        incoming = [slot(309, [300, 270]), slot(311, [390, 270])]
        for now in (.6, .7, .8, .9, 1., 2.):
            result = sequence.update(incoming+[slot(283, [350, 150])], now)
            record = next(r for r in result['track_states'] if r['byte_track_id'] == 283)
            self.assertEqual(record['mapping_reason'], 'transition_roi_quarantine')
            self.assertIsNone(record['group_id'])
            self.assertIsNone(record['formal_slot_id'])
        self.assertGreaterEqual(len(sequence.tracker._entry_points), 2)
        self.assertIsNone(sequence.track_store.owner(283))
        self.assertIsNone(result['transition_quarantine_until'])

    def test_long_missing_unassigned_id_is_recorded_without_being_quarantined(self):
        sequence = self.make()
        sequence.update(group()+[slot(283, [350, 150])], .3)
        for now in (.8, 1.3, 1.9):
            sequence.update(group(), now)
        self.close(sequence, now=2.)
        self.assertIn(283, sequence.closed_groups[0]['roi_observed_track_ids'])
        self.assertFalse(sequence.is_transition_quarantined_track(283))

    def test_latest_real_position_outside_roi_prevents_stale_inside_quarantine(self):
        sequence = self.make()
        sequence.update(group()+[slot(283, [350, 150])], .3)
        sequence.update(group()+[slot(283, [430, 150])], .4)
        self.close(sequence)
        self.assertIn(283, sequence.closed_groups[0]['roi_observed_track_ids'])
        self.assertFalse(sequence.is_transition_quarantined_track(283))

    def test_missing_track_only_quarantines_its_id_not_its_old_position(self):
        sequence = self.make()
        sequence.update(group()+[slot(283, [350, 270])], .3)
        self.close(sequence)
        result = sequence.update([slot(309, [350, 270])], .6)
        self.assertTrue(sequence.is_transition_quarantined_track(283))
        self.assertEqual(sequence.track_store.owner(309), ('unload', 2))
        record = next(r for r in result['track_states'] if r['byte_track_id'] == 309)
        self.assertEqual(record['mapping_reason'], 'waiting_slot_numbering')

    def test_disabled_quarantine_still_records_roi_ids(self):
        sequence = self.make(enabled=False)
        sequence.update(group()+[slot(283, [350, 150])], .3)
        self.close(sequence)
        self.assertIn(283, sequence.closed_groups[0]['roi_observed_track_ids'])
        self.assertFalse(sequence.is_transition_quarantined_track(283))

    def test_closed_owner_is_excluded_even_after_long_missing_and_geometry_expiry(self):
        sequence = self.make()
        for now in (.3, .6, .9, 1.2, 1.5, 1.8):
            sequence.update(group(visible=(0, 2, 3)), now)
        moved = POINTS.copy()
        moved[0] = [100, 75]
        result = sequence.update(group(moved, visible=(0, 2, 3)), 1.9)
        self.assertEqual(result['group_id'], 2)
        result = sequence.update([slot(102, POINTS[1])], 3.2)
        record = next(r for r in result['track_states'] if r['byte_track_id'] == 102)
        self.assertEqual(record['mapping_reason'], 'previous_group_track')
        self.assertEqual(sequence.track_store.owner(102), ('unload', 1))
        self.assertNotIn(102, sequence.tracker._entry_track_ids)

    def test_waiting_entry_filters_closed_owners_without_live_exit_geometry(self):
        sequence = self.make(single_exit=False)
        sequence._close(.3)
        sequence.departing_points = None
        sequence._transition_quarantine_track_ids.clear()
        sequence._transition_quarantine_centers.clear()
        result = sequence.update([slot(102, POINTS[1]), slot(309, [350, 270])], .4)
        record = next(r for r in result['track_states'] if r['byte_track_id'] == 102)
        self.assertEqual(record['mapping_reason'], 'previous_group_track')
        self.assertNotIn(102, sequence.pending_tracker.byte_track_ids)
        self.assertIn(309, sequence._group_roi_tracks[('unload', 2)])


if __name__ == '__main__':
    unittest.main()
