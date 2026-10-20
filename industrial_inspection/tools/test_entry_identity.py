"""旧尾槽隔离、进入姿态变化及临时证据映射的回归测试。"""
from __future__ import annotations

import unittest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.test_roi_group_sequence import boxes


PHYSICAL = np.array([[100, 100], [100, 200], [200, 100], [200, 200]], dtype=float)
CAMERA_ROI = np.array([[279, 157], [628, 241], [633, 387], [300, 321]], dtype=np.float32)


def rotated(degrees):
    angle = np.radians(degrees)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (PHYSICAL-150) @ matrix.T+150


class EntryIdentityTests(unittest.TestCase):
    def tracker(self):
        tracker = SlotGroupTracker(group_id=2, confirm_seconds=.1, minimum_frames=2,
            initialization_seconds=.3, initialization_frames=2, slot2_below_slot3=True)
        tracker.start_incremental_entry(40)
        return tracker

    def test_x_crossing_during_entry_waits_for_pose_and_preserves_early_event(self):
        tracker = self.tracker()
        now = 0.
        self.assertGreater(rotated(-50)[1, 0], rotated(-50)[2, 0])
        for index, angle in enumerate(range(-50, 1, 5)):
            result = tracker.update(boxes(rotated(angle), filled=(1,) if index >= 2 else ()), now)
            self.assertEqual(result['slot_numbering_status'], 'provisional')
            self.assertEqual(result['new_events'], [])
            now += .1
        events = []
        for _ in range(5):
            result = tracker.update(boxes(PHYSICAL, filled=(1,)), now)
            events.extend(result['new_events'])
            now += .1
        self.assertEqual(result['slot_numbering_status'], 'final')
        np.testing.assert_allclose(tracker.points, PHYSICAL)
        self.assertEqual([(e['slot_id'], e['action']) for e in events], [(2, 'load')])
        self.assertLess(events[0]['confirmed_at'], events[0]['reported_at'])

    def test_numbered_tracks_keep_ids_when_x_order_reverses(self):
        tracker = self.tracker()
        for index in range(5):
            tracker.update(boxes(PHYSICAL), index*.1)
        for index, angle in enumerate(range(-10, -51, -10), start=1):
            points = rotated(angle)
            result = tracker.update(boxes(points), .4+index*.1)
        self.assertGreater(result['slots'][1]['center_xy'][0], result['slots'][2]['center_xy'][0])
        np.testing.assert_allclose(result['slots'][1]['center_xy'], points[1], atol=.01)
        np.testing.assert_allclose(result['slots'][2]['center_xy'], points[2], atol=.01)

    def test_live_entry_tracks_keep_early_events_until_late_numbering(self):
        tracker = SlotGroupTracker(group_id=1, phase='load', confirm_seconds=.1,
            minimum_frames=2, initialization_seconds=.3, initialization_frames=2,
            slot2_below_slot3=True)
        tracker.start_incremental_entry(40)
        # 第四槽遮挡超过历史窗口，前三槽一直可跟踪；实际按2、1、3上料。
        for tick in range(110):
            filled = tuple(i for i, start in ((1, 10), (0, 30), (2, 60)) if tick >= start)
            result = tracker.update(boxes(PHYSICAL, visible=(0, 1, 2), filled=filled), tick*.1)
            self.assertEqual(result['new_events'], [])
        self.assertEqual([e['slot_id'] for e in tracker._entry_events], [2, 1, 3])
        early = [dict(e) for e in tracker._entry_events]
        result = tracker.update(boxes(PHYSICAL, filled=(0, 1, 2)), 11.)
        self.assertEqual(result['slot_numbering_status'], 'provisional')
        for tick in range(111, 114):
            result = tracker.update(boxes(PHYSICAL, filled=(0, 1, 2)), tick*.1)
        self.assertEqual(result['slot_numbering_status'], 'final')
        self.assertEqual(result['confirmed_event_sequence'], [2, 1, 3])
        for before, after in zip(early, result['new_events']):
            for key in ('slot_id', 'time_lower', 'time_upper', 'confirmed_at'):
                self.assertEqual(after[key], before[key])
        self.assertLess(result['new_events'][0]['confirmed_at'], 2.)
        tracker.update(boxes(PHYSICAL, filled=(0, 1, 2)), 11.4)
        tracker.update(boxes(PHYSICAL, filled=(0, 1, 2, 3)), 12.)
        result = tracker.update(boxes(PHYSICAL, filled=(0, 1, 2, 3)), 12.1)
        self.assertEqual(result['confirmed_event_sequence'], [2, 1, 3, 4])
        self.assertEqual(result['order_result'], 'pass')

    def test_fourth_slot_seen_after_motion_uses_existing_stable_reference(self):
        tracker = self.tracker()
        for index in range(5):
            tracker.update(boxes(PHYSICAL, visible=(0, 1, 2)), index*.1)
        self.assertIsNotNone(tracker._entry_reference_points)
        for index, angle in enumerate((5, 10, 15), start=5):
            tracker.update(boxes(rotated(angle), visible=(0, 1, 2)), index*.1)
        result = tracker.update(boxes(rotated(20)), .8)
        self.assertEqual(result['slot_numbering_status'], 'provisional')
        for index, angle in enumerate((25, 30, 35), start=9):
            result = tracker.update(boxes(rotated(angle)), index*.1)
        self.assertEqual(result['slot_numbering_status'], 'final')
        np.testing.assert_allclose(tracker.points, rotated(35), atol=.01)
        self.assertEqual(result['initialization']['reason'], 'stable_reference_numbering')

    def test_isolated_fourth_point_cannot_be_preserved_by_three_real_tracks(self):
        tracker = self.tracker()
        for tick in range(5):
            tracker.update(boxes(PHYSICAL, visible=(0, 1, 2)), tick*.1)
        # 两次离散闪现不能借其余三条稳定轨迹获得永久身份。
        for tick in range(5, 24):
            visible = (0, 1, 2, 3) if tick in (5, 9) else (0, 1, 2)
            result = tracker.update(boxes(PHYSICAL, visible=visible), tick*.1)
            self.assertEqual(result['slot_numbering_status'], 'provisional')
            self.assertNotIn(3, tracker._initial_seen)
        self.assertEqual(len(tracker._entry_points), 3)

    def test_stale_entry_track_rejects_neighbor_moving_against_group(self):
        tracker = self.tracker()
        for tick in range(5):
            tracker.update(boxes(PHYSICAL, visible=(0, 1, 2)), tick*.1)
        for tick in range(5, 10):
            moved = PHYSICAL-(tick-4)*5
            objects = boxes(moved, visible=(0, 1))
            if tick == 9:
                neighbor = PHYSICAL.copy()
                neighbor[2] = [215, 115]
                objects += boxes(neighbor, visible=(2,))
            tracker.update(objects, tick*.1)
        # 两点仅能否决反向跳点，隐藏点仍保持最后真实位置。
        np.testing.assert_allclose(tracker._entry_points[2], [200, 100])
        moved = PHYSICAL-25
        neighbor = PHYSICAL.copy()
        neighbor[2] = [215, 115]
        tracker.update(boxes(moved, visible=(0, 1, 2))+boxes(neighbor, visible=(2,)), 1.)
        np.testing.assert_allclose(tracker._entry_points[2], moved[2])

    def test_recorded_g5_rejects_transient_fourth_and_preserves_late_slot_identity(self):
        fixture = Path(__file__).with_name('fixtures')/'g5_loading_entry.json'
        data = json.loads(fixture.read_text(encoding='utf-8'))
        tracker = SlotGroupTracker(group_id=5, phase='load', confirm_seconds=.2,
            minimum_frames=3, initialization_seconds=.3, initialization_frames=2,
            slot2_below_slot3=True)
        tracker.start_incremental_entry(40)
        roi = np.asarray(data['roi_pixels'], dtype=np.float32)
        for row in data['frames']:
            result = tracker.update(row['objects'], row['frame']/data['fps'], roi_contour=roi)
            if row['frame'] < 3067:
                self.assertEqual(result['slot_numbering_status'], 'provisional')
            if row['frame'] == 3025:
                # 旧版在此用隐藏偏移框凑齐四槽；现在只保留三个可靠实体。
                self.assertEqual(len(tracker._entry_points), 3)
        self.assertEqual(result['slot_numbering_status'], 'final')
        np.testing.assert_allclose(tracker.points[2], [513.78, 226.76], atol=.02)
        np.testing.assert_allclose(tracker.points[3], [547.30, 280.34], atol=.02)
        self.assertTrue(all(slot['last_confirmed_state'] == 'filled' for slot in result['slots']))
        # 早先已上料的S3不能把状态转移给刚露出的S4。
        self.assertLess(result['slots'][2]['confirmed_state_since'], 161.)
        self.assertGreater(result['slots'][3]['confirmed_state_since'], 165.)

    def test_orphan_track_expires_without_transferring_its_state(self):
        tracker = self.tracker()
        for now in (0., .1):
            tracker.update(boxes(PHYSICAL, visible=(0,)), now)
        for now in (.2, .3):
            tracker.update(boxes(PHYSICAL, visible=(0,), filled=(0,)), now)
        self.assertEqual(len(tracker._entry_events), 1)
        other = PHYSICAL+[300, 0]
        result = tracker.update(boxes(other, visible=(0,), filled=(0,)), 5.4)
        self.assertEqual(len(tracker._entry_points), 1)
        self.assertEqual(result['slots'][0]['center_xy'], [400., 100.])
        self.assertEqual(result['slots'][0]['last_confirmed_state'], 'unknown')
        self.assertEqual(tracker._entry_events, [])

    def test_shifted_duplicate_cannot_become_fourth_slot(self):
        tracker = self.tracker()
        tracker.update(boxes(PHYSICAL, visible=(0, 1, 2)), 0.)
        invalid = PHYSICAL.copy()
        # 使用沿顶边的重复框，确保不是四槽凸四边形。
        invalid[3] = [150, 100]
        result = tracker.update(boxes(invalid), .1)
        self.assertEqual(result['slot_numbering_status'], 'provisional')
        self.assertEqual(len(tracker._entry_points), 3)

    def test_old_tail_inside_match_gate_cannot_start_next_group(self):
        tracker = SlotGroupTracker(group_id=3)
        sequence = RoiGroupSequence(tracker, CAMERA_ROI, exit_min_slots=1,
            partial_exit_slots=1, exit_direction=(-1, -1))
        tracker.start_incremental_entry(40)
        sequence.state = 'entering'
        old = np.array([[427.33, 183.96], [550.87, 197.03], [362.85, 204.56], [524.52, 224.89]])
        sequence.departing_points = old.copy()
        sequence.departing_seen = np.zeros(4)
        sequence.departing_gate = 40
        tail = old.copy()
        tail[0] = [504.26, 257.32]  # 旧S4参考点先接到邻槽，真实尾槽距其38.24px。
        result = sequence.update(boxes(old)+boxes(tail, visible=(0,)), .1)
        self.assertTrue(all(s['center_xy'] is None for s in result['slots']))
        self.assertTrue(result['entry_rejected_slot_detections'])
        # 即使旧轨迹失联超时，该出口侧点也不能突然变成新组槽。
        result = sequence.update(boxes(tail, visible=(0,)), 2.)
        self.assertTrue(all(s['center_xy'] is None for s in result['slots']))
        incoming = tail.copy()
        incoming[0] = [494, 315]
        result = sequence.update(boxes(incoming, visible=(0,)), 2.1)
        self.assertEqual(result['slots'][0]['center_xy'], [494., 315.])

    def test_closing_frame_unmatched_old_tail_is_preserved_for_exclusion(self):
        old = np.array([[427.33, 183.96], [550.87, 197.03], [362.85, 204.56], [524.52, 224.89]])
        sequence = RoiGroupSequence(SlotGroupTracker(old, group_id=3), CAMERA_ROI,
            exit_min_slots=1, partial_exit_slots=1, exit_direction=(-1, -1))
        tail = old.copy()
        tail[0] = [504.26, 257.32]
        objects = boxes(old)+boxes(tail, visible=(0,))
        sequence._close(0.)
        sequence._start_immediate_next(0., objects)
        self.assertEqual(len(sequence.departing_points), 5)
        result = sequence.update(objects, .1)
        self.assertTrue(all(s['center_xy'] is None for s in result['slots']))

    def test_partial_group_can_exit_without_inventing_formal_slot_numbers(self):
        tracker = self.tracker()
        roi = np.array([[80, 80], [210, 80], [210, 310], [80, 310]], dtype=np.float32)
        sequence = RoiGroupSequence(tracker, roi, exit_min_slots=1, partial_exit_slots=1)
        sequence.state = 'entering'
        sequence.update(boxes(PHYSICAL, visible=(0, 1, 2)), 0.)
        shifted = PHYSICAL.copy()
        shifted[0] = [75, 100]
        result = sequence.update(boxes(shifted, visible=(0, 1, 2)), .1)
        self.assertEqual(result['group_id'], 3)
        self.assertEqual(len(sequence.closed_groups), 1)
        self.assertEqual(sequence.closed_groups[0]['snapshot']['slot_numbering_status'], 'provisional')
        self.assertEqual(sequence.closed_groups[0]['operation_result'], 'pending')

    def test_recorded_loading_transitions_exclude_all_four_reported_old_tails(self):
        fixture = Path(__file__).with_name('fixtures')/'loading_tail_transitions.json'
        data = json.loads(fixture.read_text(encoding='utf-8'))
        for case in data['cases']:
            with self.subTest(next_group=case['old_group_id']+1):
                tracker = SlotGroupTracker(case['initial_centers'], group_id=case['old_group_id'],
                    phase='load', confirm_seconds=.2, minimum_frames=3, match_distance_ratio=.46,
                    initialization_frames=2, slot2_below_slot3=True)
                sequence = RoiGroupSequence(tracker, data['roi_pixels'], exit_min_slots=1,
                    partial_exit_slots=1, exit_direction=(-1, -1), entry_seconds=.3)
                maximum_visible = 0
                for row in case['frames']:
                    result = sequence.update(row['objects'], row['frame']/data['fps'])
                    if row['frame'] == case['old_tail_frame']:
                        self.assertEqual(result['group_id'], case['old_group_id']+1)
                        for slot in result['slots']:
                            if slot['center_xy'] is not None:
                                self.assertGreater(np.linalg.norm(np.asarray(slot['center_xy'])
                                    -case['old_tail_xy']), 3)
                    if result['group_id'] == case['old_group_id']+1:
                        maximum_visible = max(maximum_visible,
                            sum(slot['center_xy'] is not None for slot in result['slots']))
                self.assertGreaterEqual(maximum_visible, 3)  # 不能靠拒绝整组来“通过”。


if __name__ == '__main__':
    unittest.main()
