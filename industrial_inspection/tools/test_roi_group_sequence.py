"""无需模型和视频：验证真实越界换组、停稳、遮挡/旋转拒绝和五组边界。"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np

from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker

POINTS = np.array([[100,100],[200,100],[100,200],[200,200]],dtype=float)
ROI = np.array([[0,0],[300,0],[300,300],[0,300]],dtype=np.float32)


def boxes(points=POINTS,visible=(0,1,2,3),filled=()):
    return [{"class_name": "wafer_slot_"+("filled" if i in filled else "empty"),
             "confidence": .95,"xyxy": [points[i,0]-10,points[i,1]-10,points[i,0]+10,points[i,1]+10]}
            for i in visible]


class GroupSequenceTests(unittest.TestCase):
    def setUp(self):
        self.now = -.1

    def make(self,group_id=1,roi=ROI,**kwargs):
        tracker = SlotGroupTracker(POINTS,group_id=group_id,confirm_seconds=.1,minimum_frames=2,
                                   initialization_seconds=.1,initialization_frames=2)
        return RoiGroupSequence(tracker,roi,exit_seconds=.2,entry_seconds=.3,**kwargs)

    def step(self,s,objects,**kwargs):
        self.now = round(self.now+.1,6)
        return s.update(objects,self.now,**kwargs)

    def baseline(self,s):
        self.step(s,boxes())
        return self.step(s,boxes())

    def exit(self,s,visible=(0,1,2,3)):
        for delta in range(10,261,10):
            result = self.step(s,boxes(POINTS+[delta,0],visible=visible))
            if result["group_status"] in ("waiting_next","sequence_complete"):
                return result
        self.fail(f"未确认退出: {result['group_status']} {result['crossed_slot_ids']}")

    def enter(self,s,filled=()):
        result = None
        for _ in range(8):
            result = self.step(s,boxes(filled=filled))
        return result

    def test_exit_then_stable_entry_increments_once(self):
        s=self.make()
        self.baseline(s)
        result=self.exit(s)
        self.assertEqual(result['group_id'],1)
        self.assertEqual(result['pending_group_id'],2)
        self.assertEqual(len(s.closed_groups),1)
        self.assertEqual(s.closed_groups[0]['operation_result'],'pending')
        result=self.enter(s)
        self.assertEqual((result['group_id'],result['group_status']),(2,'active'))
        for _ in range(20): result=self.step(s,boxes())
        self.assertEqual(result['group_id'],2)
        self.assertEqual(len(s.closed_groups),1)
        self.assertEqual(s.tracker.events,[])
        json.dumps(result,allow_nan=False)

    def test_load_hand_contact_without_filled_stays_unconfirmed_at_end(self):
        tracker = SlotGroupTracker(POINTS,phase='load',confirm_seconds=.1,minimum_frames=2,
                                   load_hand_contact_seconds=.15)
        s = RoiGroupSequence(tracker,ROI,exit_min_slots=1,partial_exit_slots=1,
                             exit_seconds=.2,entry_seconds=.3)
        self.baseline(s)
        hand={"class_name":"hand","confidence":.95,"xyxy":[185,185,215,215]}
        for objects in (boxes(visible=(0,1,2))+[hand],
                        boxes(visible=(0,1,2))+[hand],
                        boxes(visible=(0,1,2))+[hand]):
            self.step(s,objects)
        for _ in range(4):
            self.step(s,boxes(visible=(0,1,2)))
        self.assertEqual(s.all_events,[])
        s.finish()
        self.assertEqual(s.all_events,[])
        self.assertEqual(s.tracker.slots[3].confirmed,'empty')

    def test_initialization_also_uses_only_roi(self):
        t=SlotGroupTracker(initialization_seconds=.1,initialization_frames=2)
        s=RoiGroupSequence(t,ROI)
        outside=boxes(POINTS+[400,0])
        for _ in range(3): result=self.step(s,outside)
        self.assertEqual(result['group_status'],'initializing')
        self.assertIsNone(s.tracker.points)
        for _ in range(3): result=self.step(s,boxes())
        self.assertEqual(result['group_status'],'active')

    def test_disappearance_does_not_count_as_exit(self):
        s=self.make()
        self.baseline(s)
        for _ in range(20): result=self.step(s,[])
        self.assertEqual(result['group_id'],1)
        self.assertEqual(result['group_status'],'identity_uncertain')
        self.assertEqual(s.closed_groups,[])
        for _ in range(8): result=self.step(s,boxes(filled=(0,1,2,3)))
        self.assertEqual(result['group_status'],'identity_uncertain')

    def test_hand_occlusion_does_not_count_as_exit(self):
        s=self.make()
        self.baseline(s)
        hand={"class_name":"hand","confidence":.9,"xyxy":[0,0,600,300]}
        for delta in range(10,241,10):
            result=self.step(s,boxes(POINTS+[delta,0])+[hand])
        self.assertEqual(result['group_id'],1)
        self.assertEqual(s.closed_groups,[])

    def test_rotation_crossing_roi_keeps_group(self):
        roi=np.array([[90,90],[210,90],[210,210],[90,210]],dtype=np.float32)
        s=self.make(roi=roi)
        self.baseline(s)
        for angle in range(5,91,5):
            a=np.radians(angle)
            matrix=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            moved=(POINTS-[150,150])@matrix.T+[150,150]
            result=self.step(s,boxes(moved))
        self.assertEqual(result['group_id'],1)
        self.assertEqual(result['group_status'],'active')
        self.assertEqual(s.closed_groups,[])

    def test_small_jitter_does_not_switch(self):
        s=self.make()
        self.baseline(s)
        for frame in range(30):
            result=self.step(s,boxes(POINTS+[2*(frame%2),0]))
        self.assertEqual(result['group_status'],'active')
        self.assertEqual(s.closed_groups,[])

    def test_three_visible_slots_can_exit_with_partial_rule(self):
        s=self.make()
        self.baseline(s)
        for _ in range(7): self.step(s,boxes(visible=(0,1,2)))
        result=self.exit(s,visible=(0,1,2))
        self.assertEqual(result['required_exit_slots'],2)
        self.assertEqual(len(s.closed_groups),1)

    def test_single_point_outside_roi_triggers_group_exit(self):
        """单点换组配置下，任意一个真实槽中心越过ROI即开始换组。"""
        roi=np.array([[80,80],[210,80],[210,210],[80,210]],dtype=np.float32)
        s=self.make(roi=roi,exit_min_slots=1,partial_exit_slots=1)
        self.baseline(s)
        moved=POINTS.copy()
        moved[0] = [75,100]  # 距ROI外侧5像素，仍在跟踪门限内
        result=self.step(s,boxes(moved))
        self.assertEqual((result['group_id'], result['group_status']),(2,'entering'))
        self.assertEqual(s.closed_groups[0]['crossed_slot_ids'],[1])
        self.assertEqual(result['required_exit_slots'],1)
        self.assertEqual(result['new_group_events'][0]['type'],'group_exiting')
        self.assertEqual([e['type'] for e in result['new_group_events']],
                         ['group_exiting','group_closed','group_started'])
        self.assertIsNone(result['pending_group_id'])
        result=self.step(s,boxes(moved))
        result=self.step(s,boxes(moved))
        self.assertEqual((result['group_id'], result['group_status']),(2,'entering'))
        self.assertEqual(result['initialization']['eligible_slot_count'],0)

    def test_single_exit_tracks_new_point_and_full_state_before_four_slots_appear(self):
        roi=np.array([[80,80],[410,80],[410,310],[80,310]],dtype=np.float32)
        s=self.make(roi=roi,exit_min_slots=1,partial_exit_slots=1)
        self.baseline(s)
        old=POINTS.copy()
        old[0]=[75,100]
        incoming=POINTS+[180,80]
        before=boxes(old)+boxes(incoming,visible=(0,),filled=(0,))
        result=self.step(s,before)
        self.assertEqual((result['group_id'],result['group_status']),(2,'entering'))
        self.assertEqual(result['slots'][0]['observation'],'filled')
        result=self.step(s,before)
        self.assertEqual(result['slots'][0]['state'],'filled')
        self.assertEqual(result['slot_numbering_status'],'provisional')
        self.assertEqual([x['center_xy'] for x in result['slots'][1:]],[None]*3)
        for _ in range(2):
            result=self.step(s,boxes(old)+boxes(incoming,visible=(0,)))
        self.assertEqual(result['slots'][0]['state'],'empty')
        self.assertEqual(result['initialization']['buffered_event_count'],1)
        self.assertEqual(result['new_events'],[])
        # 首槽已遮挡，其余三槽才出现：不要求四槽同帧，三槽支持停稳后正式编号。
        for _ in range(4):
            result=self.step(s,boxes(old)+boxes(incoming,visible=(1,2,3),filled=(1,2,3)))
        self.assertEqual((result['group_id'],result['group_status']),(2,'active'))
        self.assertEqual(result['slot_numbering_status'],'final')
        event=result['new_events'][0]
        self.assertEqual((event['group_id'],event['slot_id'],event['action']),(2,1,'unload'))
        self.assertTrue(event['recovered_before_initialization'])
        self.assertEqual(s.all_events,[event])
        self.assertEqual(s.closed_groups[0]['events'],[])
        self.assertEqual(self.step(s,boxes(incoming))['new_events'],[])

    def test_early_point_state_and_event_follow_final_slot_number_mapping(self):
        roi=np.array([[80,80],[410,80],[410,310],[80,310]],dtype=np.float32)
        s=self.make(roi=roi,exit_min_slots=1,partial_exit_slots=1)
        self.baseline(s)
        old=POINTS.copy()
        old[0]=[75,100]
        incoming=POINTS+[180,80]
        for _ in range(2):
            result=self.step(s,boxes(old)+boxes(incoming,visible=(3,),filled=(3,)))
        self.assertEqual(result['slots'][0]['state'],'filled')
        for _ in range(2):
            self.step(s,boxes(old)+boxes(incoming,visible=(3,)))
        for _ in range(4):
            result=self.step(s,boxes(old)+boxes(incoming,visible=(0,1,2)))
        self.assertEqual(result['new_events'][0]['slot_id'],4)
        self.assertEqual(result['slots'][3]['last_confirmed_state'],'empty')
        np.testing.assert_allclose(s.tracker.points,incoming[[0,2,1,3]])

    def test_old_point_match_does_not_hide_a_distinct_new_point_within_match_gate(self):
        roi=np.array([[80,80],[410,80],[410,310],[80,310]],dtype=np.float32)
        s=self.make(roi=roi,exit_min_slots=1,partial_exit_slots=1)
        self.baseline(s)
        old=POINTS.copy()
        old[0]=[75,100]
        incoming=POINTS.copy()
        incoming[0]=[230,200]  # 距真实旧点30px；匹配门限40px，二者同帧可区分。
        for _ in range(2):
            result=self.step(s,boxes(old)+boxes(incoming,visible=(0,),filled=(0,)))
        self.assertEqual(result['initialization']['eligible_slot_count'],1)
        self.assertEqual(result['slots'][0]['center_xy'],[230.,200.])
        self.assertEqual(result['slots'][0]['state'],'filled')

    def test_missing_old_tail_cannot_take_an_established_incoming_track(self):
        """旧点漏检后，新点仍应归属新组，临时号和正式号阶段都不能被旧点吞掉。"""
        roi=np.array([[80,80],[410,80],[410,310],[80,310]],dtype=np.float32)
        s=self.make(roi=roi,exit_min_slots=1,partial_exit_slots=1)
        self.baseline(s)
        old=POINTS.copy()
        old[0]=[75,100]
        incoming=np.array([[230,200],[280,100],[380,100],[380,200]],dtype=float)
        for _ in range(2):
            self.step(s,boxes(old)+boxes(incoming,visible=(0,),filled=(0,)))
        old_seen=s.departing_seen[3]
        # 距新点30px的旧点不再出现；不能用新点续接它，也不能刷新它的存活时间。
        for _ in range(2):
            result=self.step(s,boxes(old,visible=(0,1,2))
                             +boxes(incoming,visible=(0,),filled=(0,)))
            self.assertEqual(result['slots'][0]['state'],'filled')
            self.assertEqual(result['slots'][0]['center_xy'],[230.,200.])
            self.assertEqual(s.departing_seen[3],old_seen)
        for _ in range(4):
            result=self.step(s,boxes(old,visible=(0,1,2))+boxes(incoming,filled=(0,1,2,3)))
        self.assertEqual((result['group_id'],result['group_status']),(2,'active'))
        self.assertTrue(all(slot['state']=='filled' for slot in result['slots']))
        for _ in range(2):
            result=self.step(s,boxes(old,visible=(0,1,2))+boxes(incoming,filled=(1,2,3)))
        self.assertEqual(result['slots'][0]['state'],'empty')
        self.assertEqual([(e['group_id'],e['slot_id'],e['action']) for e in s.all_events],
                         [(2,1,'unload')])
        self.assertEqual(s.departing_seen[3],old_seen)

    def test_two_crossings_when_four_visible_are_not_enough(self):
        s=self.make()
        self.baseline(s)
        for delta in range(10,131,10):
            result=self.step(s,boxes(POINTS+[delta,0]))
        self.assertEqual(len(result['crossed_slot_ids']),2)
        for _ in range(8): result=self.step(s,[])
        self.assertEqual(s.closed_groups,[])
        self.assertEqual(result['group_id'],1)

    def test_stationary_external_slots_do_not_create_crossings(self):
        s=self.make()
        self.baseline(s)
        for _ in range(10): result=self.step(s,boxes()+boxes(POINTS+[300,0]))
        self.assertEqual(result['group_status'],'active')
        self.assertEqual(result['crossed_slot_ids'],[])

    def test_no_entry_with_three_or_more_than_four_candidates(self):
        s=self.make()
        self.baseline(s)
        self.exit(s)
        for _ in range(10): result=self.step(s,boxes(visible=(0,1,2)))
        self.assertEqual(result['group_id'],1)
        extra={"class_name":"wafer_slot_empty","confidence":.9,"xyxy":[230,240,250,260]}
        for _ in range(10): result=self.step(s,boxes()+[extra])
        self.assertEqual(result['group_id'],1)
        self.assertEqual(result['group_status'],'waiting_next')

    def test_moving_entry_does_not_initialize_until_stopped(self):
        s=self.make()
        self.baseline(s)
        self.exit(s)
        for delta in range(-60,41,10):
            result=self.step(s,boxes(POINTS+[delta,0]))
        self.assertEqual(result['group_id'],1)
        for _ in range(7): result=self.step(s,boxes(POINTS+[40,0]))
        self.assertEqual(result['group_id'],2)

    def test_group_events_and_slot_events_are_isolated(self):
        s=self.make()
        self.baseline(s)
        for _ in range(2): self.step(s,boxes(filled=(0,)))
        self.assertEqual(len(s.all_events),1)
        self.exit(s)
        archived=json.dumps(s.closed_groups)
        self.enter(s,filled=(0,1,2,3))
        self.assertEqual(s.tracker.events,[])
        self.assertEqual(len(s.all_events),1)
        self.assertEqual(s.all_events[0]['group_id'],1)
        self.assertEqual(json.dumps(s.closed_groups),archived)

    def test_return_to_original_position_cancels_exit(self):
        s=self.make()
        self.baseline(s)
        for delta in (10,20,30,20,10,0,0,0,0):
            result=self.step(s,boxes(POINTS+[delta,0]))
        self.assertEqual(result['group_status'],'active')
        self.assertEqual(result['group_id'],1)
        self.assertEqual(s.closed_groups,[])

    def test_configured_direction_rejects_opposite_exit(self):
        s=self.make(exit_direction=(-1,0))
        self.baseline(s)
        for delta in range(10,261,10): result=self.step(s,boxes(POINTS+[delta,0]))
        self.assertEqual(result['group_id'],1)
        self.assertEqual(s.closed_groups,[])

    def test_group_five_finishes_without_wrapping(self):
        s=self.make(group_id=5)
        self.baseline(s)
        result=self.exit(s)
        self.assertEqual(result['group_status'],'sequence_complete')
        result=self.enter(s)
        self.assertEqual(result['group_id'],5)
        self.assertEqual(result['group_status'],'sequence_complete')
        self.assertIsNone(result['pending_group_id'])

    def test_disabled_mode_never_switches(self):
        s=self.make(enabled=False)
        self.baseline(s)
        for delta in range(10,261,10): result=self.step(s,boxes(POINTS+[delta,0]))
        self.assertEqual(result['group_id'],1)
        self.assertEqual(s.closed_groups,[])

    def test_outside_roi_does_not_confirm_fill_state(self):
        s=self.make(enabled=False,roi=np.array([[90,90],[210,90],[210,210],[90,210]],dtype=np.float32))
        self.baseline(s)
        moved=POINTS+[15,0]
        for _ in range(3): result=self.step(s,boxes(moved,filled=(1,3)))
        self.assertEqual(result['slots'][1]['reason'],'outside_roi')
        self.assertEqual(result['slots'][1]['state'],'unknown')
        self.assertEqual(s.tracker.events,[])

    def test_non_monotonic_time_is_rejected(self):
        s=self.make()
        self.step(s,boxes())
        with self.assertRaises(ValueError): s.update(boxes(),self.now)

    def test_small_reposition_stopping_inside_keeps_operation_active(self):
        s=self.make()
        self.baseline(s)
        for delta in (10,20,30): self.step(s,boxes(POINTS+[delta,0]))
        for _ in range(10): result=self.step(s,boxes(POINTS+[30,0]))
        self.assertEqual(result['group_status'],'active')
        self.assertEqual(result['group_id'],1)
        self.assertEqual([slot['state'] for slot in result['slots']],['empty']*4)

    def test_entire_five_group_sequence(self):
        s=self.make()
        self.baseline(s)
        for group_id in range(1,6):
            result=self.exit(s)
            self.assertEqual(result['group_id'],group_id)
            if group_id < 5:
                result=self.enter(s)
                self.assertEqual(result['group_id'],group_id+1)
        self.assertEqual(result['group_status'],'sequence_complete')
        self.assertEqual([g['group_id'] for g in s.closed_groups],[1,2,3,4,5])

    def test_jump_outside_without_tracked_crossing_keeps_group_uncertain(self):
        s=self.make()
        self.baseline(s)
        for _ in range(20): result=self.step(s,boxes(POINTS+[500,0]))
        self.assertEqual(result['group_id'],1)
        self.assertEqual(result['group_status'],'identity_uncertain')
        self.assertEqual(s.closed_groups,[])

    def test_transition_does_not_generate_load_events(self):
        s=self.make()
        self.baseline(s)
        for delta in range(10,251,10):
            self.step(s,boxes(POINTS+[delta,0],filled=(0,1,2,3)))
            if s.state == 'waiting_next': break
        self.assertEqual(s.all_events,[])

    def test_old_tail_and_incoming_group_overlap_does_not_require_empty_roi(self):
        s=self.make()
        self.baseline(s)
        # 仅上方左槽和右侧两槽可见，两个右槽越界后，左槽尾部还在ROI中。
        for _ in range(7): self.step(s,boxes(visible=(0,1,3)))
        result=self.exit(s,visible=(0,1,3))
        self.assertEqual(result['required_exit_slots'],2)
        for frame in range(12):
            tail=s.departed_points+s.departed_velocity*(frame+1)*.1
            result=self.step(s,boxes()+boxes(tail,visible=(0,)))
        self.assertEqual(result['group_id'],2)
        self.assertEqual(result['group_status'],'active')

    def test_next_entry_hand_occlusion_blocks_count(self):
        s=self.make()
        self.baseline(s)
        self.exit(s)
        hand={"class_name":"hand","confidence":.9,"xyxy":[85,85,115,115]}
        for _ in range(10): result=self.step(s,boxes()+[hand])
        self.assertEqual(result['group_id'],1)
        self.assertEqual(result['pending_group_id'],2)

    def test_recorded_first_transition_tracks_perspective_motion(self):
        """人工确认40秒首次换组；回放真实框，防止参考位置冻结后误接新组。"""
        path=Path(__file__).with_name('fixtures')/'first_roi_transition.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        t=SlotGroupTracker(data['initial_centers'],confirm_seconds=.1,minimum_frames=3,
                          initialization_seconds=.3,initialization_frames=3)
        s=RoiGroupSequence(t,data['roi_pixels'],exit_direction=(-1,-1))
        group_events=[]
        for row in data['frames']:
            result=s.update(row['objects'],row['frame']/data['fps'])
            group_events.extend(result['new_group_events'])
            if row['frame']/data['fps'] < 40:
                self.assertEqual(result['group_id'],1)
                self.assertEqual(s.closed_groups,[])
        closed=[e for e in group_events if e['type']=='group_closed']
        started=[e for e in group_events if e['type']=='group_started']
        self.assertEqual(len(closed),1)
        self.assertEqual(closed[0]['group_id'],1)
        self.assertGreaterEqual(closed[0]['at'],42)
        self.assertLess(closed[0]['at'],43)
        self.assertEqual(closed[0]['crossed_slot_ids'],[1,2,3])
        self.assertEqual(len(started),1)
        self.assertEqual(started[0]['group_id'],2)
        self.assertGreater(started[0]['at'],closed[0]['at'])
        self.assertLess(started[0]['at'],48)
        self.assertEqual(result['group_id'],2)

    def test_recorded_single_exit_captures_second_group_full_states_before_four_slots(self):
        data=json.loads((Path(__file__).with_name('fixtures')/'first_roi_transition.json').read_text(encoding='utf-8'))
        t=SlotGroupTracker(data['initial_centers'],confirm_seconds=.1,minimum_frames=3,match_distance_ratio=.46)
        s=RoiGroupSequence(t,data['roi_pixels'],exit_min_slots=1,partial_exit_slots=1,exit_direction=(-1,-1))
        first_full=None
        all_full=None
        for row in data['frames']:
            now=row['frame']/data['fps']
            result=s.update(row['objects'],now)
            if result['group_id']==2:
                if first_full is None and any(x['state']=='filled' for x in result['slots']):
                    first_full=now
                    self.assertEqual(result['group_status'],'entering')
                    self.assertEqual(sum(x['center_xy'] is not None for x in result['slots']),1)
                if all_full is None and all(x['last_confirmed_state']=='filled' for x in result['slots']):
                    all_full=now
                self.assertNotEqual(result['group_status'],'identity_uncertain')
        self.assertIsNotNone(first_full)
        self.assertLess(first_full,42)
        self.assertIsNotNone(all_full)
        self.assertLess(all_full,44.2)
        self.assertAlmostEqual(s.closed_groups[0]['at'],40.883382709399406)
        self.assertEqual([(e['group_id'],e['slot_id'],e['action']) for e in s.all_events],[(2,2,'unload'),(2,1,'unload')])

    def test_recorded_second_exit_keeps_old_duplicate_out_of_third_group(self):
        data=json.loads((Path(__file__).with_name('fixtures')/'second_roi_transition.json').read_text(encoding='utf-8'))
        t=SlotGroupTracker(data['initial_centers'],group_id=2,confirm_seconds=.2,minimum_frames=3,
                           match_distance_ratio=.46,slot2_below_slot3=True)
        s=RoiGroupSequence(t,data['roi_pixels'],exit_min_slots=1,partial_exit_slots=1,
                           exit_direction=(-1,-1),entry_seconds=.3)
        first_full=None
        all_full=None
        for row in data['frames']:
            now=row['frame']/data['fps']
            result=s.update(row['objects'],now)
            if result['group_id']==3:
                if first_full is None and any(x['state']=='filled' for x in result['slots']):
                    first_full=now
                    self.assertEqual(result['group_status'],'entering')
                if all_full is None and all(x['last_confirmed_state']=='filled' for x in result['slots']):
                    all_full=now
                self.assertNotEqual(result['group_status'],'identity_uncertain')
        self.assertIsNotNone(first_full)
        self.assertLess(first_full,55)
        self.assertIsNotNone(all_full)
        self.assertLess(all_full,57.5)
        self.assertEqual([(e['slot_id'],e['action']) for e in s.all_events if e['group_id']==3],[(2,'unload')])

    def test_next_group_recovers_early_load_and_unload_without_duplicates(self):
        for phase in ('load','unload'):
            with self.subTest(phase=phase):
                self.now=-.1
                s=self.make()
                s.tracker.phase=phase
                s.entry_seconds=.6
                self.baseline(s)
                self.exit(s)
                archived=json.dumps(s.closed_groups)
                hand={"class_name":"hand","confidence":.9,"xyxy":[85,85,115,115]}
                before=boxes(filled=(0,1,2,3) if phase=='unload' else ())
                after=boxes(filled=(1,2,3) if phase=='unload' else (0,))
                first_time=self.now+.1
                for objects in (before,before,before+[hand],before+[hand],after,after):
                    result=self.step(s,objects)
                    self.assertEqual(result['group_id'],1)
                    self.assertEqual(result['new_events'],[])
                self.assertEqual(result['pending_initialization']['buffered_event_count'],1)
                result=self.step(s,after)
                self.assertEqual((result['group_id'],result['group_status']),(2,'active'))
                self.assertEqual(len(result['new_events']),1)
                event=result['new_events'][0]
                self.assertEqual((event['group_id'],event['slot_id'],event['action']),(2,1,phase))
                self.assertAlmostEqual(event['time_lower'],first_time+.1)
                self.assertAlmostEqual(event['time_upper'],first_time+.4)
                self.assertTrue(event['recovered_before_initialization'])
                self.assertEqual(s.all_events,[event])
                self.assertEqual(json.dumps(s.closed_groups),archived)
                for _ in range(5): self.assertEqual(self.step(s,after)['new_events'],[])
                self.assertEqual(s.all_events,[event])

    def test_next_group_can_finish_identity_confirmation_with_one_hidden_slot(self):
        s=self.make()
        self.baseline(s)
        self.exit(s)
        for _ in range(2): self.step(s,boxes(filled=(0,1,2,3)))
        for _ in range(3): result=self.step(s,boxes(visible=(1,2,3),filled=(0,1,2,3)))
        self.assertEqual((result['group_id'],result['group_status']),(2,'active'))
        self.assertEqual(result['slots'][0]['state'],'unknown')
        self.assertEqual(result['slots'][0]['last_confirmed_state'],'filled')
        self.assertEqual(s.all_events,[])

    def test_template_drift_without_observed_slot_motion_does_not_pause_unload(self):
        s=self.make()
        self.baseline(s)
        # 参考布局可能受空满框形状变化影响；实际四个中心并没有一起外移。
        s.tracker.reference_points=POINTS+[-25,-5]
        candidates=s.tracker._candidates(boxes())
        matches={i:i for i in range(4)}
        enabled=s._observe_motion(s.tracker,candidates,matches,self.now+.1)
        self.assertTrue(enabled)
        self.assertEqual(s.state,'active')
        self.assertEqual(s.closed_groups,[])

    def test_recorded_first_unload_from_initialization_keeps_fourth_event(self):
        """实际视频中槽4空框偏移曾使39.91秒提前暂停，导致第一组只记到三次取片。"""
        path=Path(__file__).with_name('fixtures')/'first_unload.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        tracker=SlotGroupTracker(confirm_seconds=.1,minimum_frames=3,
                                 initialization_seconds=.3,initialization_frames=3)
        s=RoiGroupSequence(tracker,data['roi_pixels'],entry_seconds=.3,exit_direction=(-1,-1))
        events=[]
        for row in data['frames']:
            result=s.update(row['objects'],row['frame']/data['fps'])
            events.extend(result['new_group_events'])
        self.assertEqual(len(s.closed_groups),1)
        first=s.closed_groups[0]
        self.assertEqual(first['group_id'],1)
        self.assertEqual(first['snapshot']['confirmed_event_sequence'],[2,1,3,4])
        self.assertEqual(first['operation_result'],'pass')
        fourth=[e for e in first['events'] if e['slot_id']==4]
        self.assertEqual(len(fourth),1)
        self.assertEqual(fourth[0]['action'],'unload')
        self.assertLess(fourth[0]['confirmed_at'],40.5)
        exiting=[e for e in events if e['type']=='group_exiting' and e['group_id']==1]
        self.assertEqual(len(exiting),1)
        self.assertGreater(exiting[0]['at'],fourth[0]['confirmed_at'])
        self.assertGreater(exiting[0]['at'],40.4)
        self.assertGreaterEqual(len(exiting[0]['moving_slot_ids']),2)
        self.assertGreater(first['at'],42)
        self.assertLess(first['at'],43)

    def test_two_observed_centers_with_consistent_motion_can_start_exit(self):
        s=self.make()
        self.baseline(s)
        moved=POINTS+[-30,0]
        s.tracker.reference_points=moved.copy()
        candidates=s.tracker._candidates(boxes(moved,visible=(0,1)))
        matches={0:0,1:1}
        self.assertFalse(s._observe_motion(s.tracker,candidates,matches,.2))
        self.assertEqual(s.state,'exiting')
        self.assertEqual(s._group_events[-1]['moving_slot_ids'],[1,2])

    def test_two_inconsistent_center_shifts_do_not_start_exit(self):
        s=self.make()
        self.baseline(s)
        moved=POINTS+[-30,0]
        s.tracker.reference_points=moved.copy()
        moved[1]+=[-40,0]
        candidates=s.tracker._candidates(boxes(moved,visible=(0,1)))
        matches={0:0,1:1}
        self.assertTrue(s._observe_motion(s.tracker,candidates,matches,.2))
        self.assertEqual(s.state,'active')


if __name__ == '__main__':
    unittest.main(verbosity=2)
