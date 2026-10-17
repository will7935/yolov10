"""运行本文件验证编号、遮挡、空满确认和环形顺序；无需GPU或视频。"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np

from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker

CENTERS = np.array([[100,100],[200,100],[100,200],[200,200]],dtype=float)


def detection(index,state="empty",centers=CENTERS,score=.9):
    x,y = centers[index-1]
    return {"class_name": "wafer_slot_"+state,"class_id": 3 if state=="empty" else 4,
            "confidence": score,"xyxy": [x-20,y-20,x+20,y+20]}


def tracker(**kwargs):
    return SlotGroupTracker(CENTERS,confirm_seconds=.1,minimum_frames=2,**kwargs)


class TrackingTests(unittest.TestCase):
    def test_baseline_does_not_create_events(self):
        t=tracker()
        objects=[detection(i,"filled") for i in range(1,5)]
        t.update(objects,0)
        result=t.update(list(reversed(objects)),.1)
        self.assertEqual([s['state'] for s in result['slots']],['filled']*4)
        self.assertEqual(t.events,[])
        self.assertEqual([s['slot_id'] for s in result['slots']],[1,2,3,4])

    def test_missing_fourth_is_unknown_and_group_stays(self):
        t=tracker(group_id=3)
        for now in (0,.1):
            result=t.update([detection(i) for i in (1,2,3)],now)
        self.assertEqual(result['slots'][3]['state'],'unknown')
        self.assertEqual(result['group_id'],3)
        self.assertEqual(result['confirmed_event_sequence'],[])

    def test_occlusion_keeps_last_confirmed_but_current_unknown(self):
        t=tracker()
        for now in (0,.1):
            t.update([detection(1)],now)
        s=t.update([], .2)['slots'][0]
        self.assertEqual(s['state'],'unknown')
        self.assertEqual(s['last_confirmed_state'],'empty')

    def test_hand_occlusion_blocks_confirmation(self):
        t=tracker()
        hand={'class_name':'hand','class_id':0,'confidence':.9,'xyxy':[85,85,115,115]}
        for now in (0,.1):
            result=t.update([detection(1),hand],now)
        self.assertEqual(result['slots'][0]['reason'],'hand_occlusion')
        self.assertEqual(result['slots'][0]['state'],'unknown')

    def test_conflicting_classes_do_not_create_event(self):
        t=tracker()
        for now in (0,.1):
            result=t.update([detection(1,'empty',score=.9),detection(1,'filled',score=.85)],now)
        self.assertEqual(result['slots'][0]['reason'],'empty_filled_conflict')
        self.assertEqual(t.events,[])

    def test_small_rotation_preserves_ids_and_infers_missing_anchor(self):
        t=tracker()
        t.update([detection(i) for i in range(1,5)],0)
        angle=np.radians(5)
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        moved=(CENTERS-[150,150])@rotation.T+[153,151]
        t.update([detection(i,centers=moved) for i in (3,1,2)],.1)
        self.assertTrue(np.allclose(t.points,moved,atol=.05))
        self.assertEqual(t.snapshot()['slots'][3]['state'],'unknown')

    def test_two_visible_jittering_boxes_do_not_drift_hidden_slots(self):
        t=tracker()
        t.update([detection(i) for i in range(1,5)],0)
        # 右侧两框轻微抖动，不应逐帧变换左侧未检出的中心。
        for frame in range(1,15):
            moved=CENTERS.copy()
            moved[1]+=[0,2*(frame%2)]
            moved[3]+=[3*(frame%2),0]
            t.update([detection(i,centers=moved) for i in (2,4)],frame*.1)
        np.testing.assert_allclose(t.points[[0,2]],CENTERS[[0,2]])
        for now in (1.5,1.6):
            result=t.update([detection(i) for i in range(1,5)],now)
        self.assertEqual([s['state'] for s in result['slots']],['empty']*4)

    def test_single_box_cannot_walk_reference_to_neighbouring_slot(self):
        t=tracker()
        t.update([detection(i) for i in range(1,5)],0)
        for frame in range(1,10):
            moved=CENTERS.copy()
            moved[0]+=[frame*5,0]
            t.update([detection(1,centers=moved)],frame*.1)
        np.testing.assert_allclose(t.reference_points,CENTERS)
        self.assertEqual(t._spacing(),100)
        self.assertEqual(t.snapshot()['slots'][0]['reason'],'not_detected')
        for now in (1.,1.1):
            result=t.update([detection(1)],now)
        self.assertEqual(result['slots'][0]['state'],'empty')

    def test_identity_timeout_does_not_reassign_to_new_group(self):
        t=tracker(group_id=2)
        t.update([detection(1)],0)
        t.update([], .5)
        result=t.update([detection(1,'filled')],1.5)
        self.assertEqual(result['identity_status'],'lost')
        self.assertEqual(result['group_id'],2)
        self.assertEqual(result['slots'][0]['state'],'unknown')
        # 即使四点再出现也不按x重排或解除身份锁，避免给下一组继承编号。
        result=t.update([detection(i) for i in range(1,5)],1.6)
        self.assertEqual(result['identity_status'],'lost')

    def test_rotation_interval_suspends_events(self):
        t=tracker()
        for now in (0,.1):
            t.update([detection(1)],now)
        for now in (.2,.3):
            result=t.update([detection(1,'filled')],now,observations_enabled=False)
        self.assertEqual(t.events,[])
        self.assertEqual(result['slots'][0]['reason'],'rotation_or_transition')

    def _cycle(self,phase,order):
        t=tracker(phase=phase)
        states=['filled' if phase=='unload' else 'empty']*4
        now=0
        for _ in range(2):
            t.update([detection(i,states[i-1]) for i in range(1,5)],now)
            now+=.1
        for slot_id in order:
            states[slot_id-1]='empty' if phase=='unload' else 'filled'
            for _ in range(2):
                t.update([detection(i,states[i-1]) for i in range(1,5)],now)
                now+=.1
        return t

    def test_loading_and_unloading_use_same_clockwise_cycle(self):
        for phase in ('load','unload'):
            for order in ((1,3,4,2),(2,1,3,4)):
                with self.subTest(phase=phase,order=order):
                    t=self._cycle(phase,order)
                    self.assertEqual(t.order_result(),'pass')
                    self.assertEqual(t.snapshot()['last_verified_target_count'],4)

    def test_reverse_order_fails(self):
        self.assertEqual(self._cycle('unload',(1,2,4,3)).order_result(),'fail')

    def test_simultaneous_transitions_are_uncertain(self):
        t=tracker(phase='load')
        for now in (0,.1):
            t.update([detection(i) for i in range(1,5)],now)
        for now in (.2,.3):
            t.update([detection(i,'filled') for i in range(1,5)],now)
        self.assertEqual(len(t.events),4)
        self.assertEqual(t.order_result(),'pending')
        self.assertTrue(t.snapshot()['all_four_target_previously_verified'])

    def test_load_hand_contact_alone_never_fills_hidden_s4(self):
        t=tracker(phase='load',load_hand_contact_seconds=.15)
        hand={'class_name':'hand','class_id':0,'confidence':.9,'xyxy':[185,185,215,215]}
        full=[detection(i,'empty') for i in range(1,5)]
        for now in (0.,.1):
            t.update(full,now)
        for now in (.2,.3,.4):
            result=t.update([detection(i,'empty') for i in (1,2,3)]+[hand],now)
        self.assertEqual(result['slots'][3]['state'],'unknown')
        self.assertTrue(result['slots'][3]['load_contact_pending'])
        result=t.update([detection(i,'empty') for i in (1,2,3)],.5)
        self.assertEqual(result['new_events'],[])
        result=t.update([detection(i,'empty') for i in (1,2,3)],.7)
        self.assertEqual(result['new_events'],[])
        self.assertEqual(t.events,[])
        self.assertEqual(t.snapshot()['slots'][3]['last_confirmed_state'],'empty')

    def test_load_hand_contact_prefers_later_filled_detection(self):
        t=tracker(phase='load',load_hand_contact_seconds=.15)
        hand={'class_name':'hand','class_id':0,'confidence':.9,'xyxy':[185,185,215,215]}
        for now in (0.,.1):
            t.update([detection(i,'empty') for i in range(1,5)],now)
        for now in (.2,.3,.4):
            t.update([detection(i,'empty') for i in (1,2,3)]+[hand],now)
        for now in (.5,.6):
            result=t.update([detection(i,'empty') for i in (1,2,3)]+[detection(4,'filled')],now)
        self.assertEqual([(e['slot_id'],e['reason'],e['inferred']) for e in t.events],
                         [(4,'detected_after_hand_contact',False)])
        self.assertEqual(result['slots'][3]['last_confirmed_state'],'filled')

    def test_short_load_hand_pass_does_not_fill_slot(self):
        t=tracker(phase='load',load_hand_contact_seconds=.15)
        hand={'class_name':'hand','class_id':0,'confidence':.9,'xyxy':[185,185,215,215]}
        for now in (0.,.1):
            t.update([detection(i,'empty') for i in range(1,5)],now)
        t.update([detection(i,'empty') for i in (1,2,3)]+[hand],.2)
        result=t.update([detection(i,'empty') for i in (1,2,3)],.6)
        self.assertEqual(result['slots'][3]['last_confirmed_state'],'empty')
        self.assertEqual(t.events,[])

    def test_unload_ignores_hand_contact_inference(self):
        t=tracker(phase='unload')
        hand={'class_name':'hand','class_id':0,'confidence':.9,'xyxy':[185,185,215,215]}
        for now in (0.,.1):
            t.update([detection(i,'filled') for i in range(1,5)],now)
        for now in (.2,.3,.4,.7):
            t.update([detection(i,'filled') for i in (1,2,3)]+([hand] if now < .5 else []),now)
        self.assertEqual(t.events,[])

    def test_hidden_event_is_not_imputed_and_partial_sequence_not_failed(self):
        t=tracker(phase='load')
        states=['empty']*4
        now=0
        for _ in range(2):
            t.update([detection(i,states[i-1]) for i in range(1,5)],now)
            now+=.1
        for slot_id in (1,3,2):
            states[slot_id-1]='filled'
            for _ in range(2):
                t.update([detection(i,states[i-1]) for i in (1,2,3)],now)
                now+=.1
        self.assertEqual(t.snapshot()['confirmed_event_sequence'],[1,3,2])
        self.assertEqual(t.order_result(),'pending')
        self.assertEqual(t.snapshot()['slots'][3]['state'],'unknown')

    def test_one_detection_cannot_fill_two_slots(self):
        t=tracker()
        result=t.update([detection(1)],0)
        self.assertEqual(sum(s['observation']!='unknown' for s in result['slots']),1)


class AutomaticInitializationTests(unittest.TestCase):
    def make_auto(self,**kwargs):
        return SlotGroupTracker(confirm_seconds=.1,minimum_frames=2,
            initialization_seconds=.1,initialization_frames=2,**kwargs)

    def test_no_detections_does_not_timeout_or_invent_centers(self):
        t=self.make_auto()
        for now in (0,2,10):
            result=t.update([],now)
        self.assertEqual(result['identity_status'],'waiting_initialization')
        self.assertEqual([s['center_xy'] for s in result['slots']],[None]*4)
        self.assertEqual([s['state'] for s in result['slots']],['unknown']*4)
        json.dumps(result,allow_nan=False)

    def test_three_slots_wait_until_four_appear(self):
        t=self.make_auto(group_id=4)
        for now in (0,.1):
            result=t.update([detection(i) for i in (1,2,3)],now)
        self.assertIsNone(t.points)
        self.assertEqual(result['initialization']['eligible_slot_count'],3)
        objects=[detection(i,'filled') for i in (4,2,1,3)]
        t.update(objects,.2)
        result=t.update(objects,.3)
        self.assertEqual(result['identity_status'],'tracking')
        expected=CENTERS[[0,2,1,3]]  # 按x排序，x相同再按y排序。
        np.testing.assert_allclose(t.points,expected)
        self.assertEqual(result['group_id'],4)
        self.assertEqual(t.events,[])
        result=t.update(objects,.4)
        self.assertEqual([s['state'] for s in result['slots']],['filled']*4)
        self.assertEqual(t.events,[])

    def test_duration_and_frame_count_both_required(self):
        t=SlotGroupTracker(initialization_seconds=.2,initialization_frames=3)
        objects=[detection(i) for i in range(1,5)]
        for now in (0,.01,.02):
            t.update(objects,now)
        self.assertIsNone(t.points)
        t.update(objects,.2)
        self.assertIsNotNone(t.points)

    def test_provisional_states_are_visible_before_identity_is_ready(self):
        t=SlotGroupTracker(initialization_seconds=.5,initialization_frames=3,
                           confirm_seconds=.1,minimum_frames=2)
        full=[detection(i,'filled') for i in range(1,5)]
        for now in (0,.1):
            result=t.update(full,now)
        self.assertIsNone(t.points)
        self.assertEqual(result['identity_status'],'waiting_initialization')
        self.assertEqual([s['state'] for s in result['slots']],['filled']*4)
        empty_first=[detection(i,'empty' if i==1 else 'filled') for i in range(1,5)]
        for now in (.2,.3):
            result=t.update(empty_first,now)
        self.assertIsNone(t.points)
        self.assertEqual(result['slots'][0]['state'],'empty')
        self.assertEqual(result['slots'][0]['last_confirmed_state'],'empty')
        self.assertEqual(result['new_events'],[])
        self.assertEqual(result['initialization']['buffered_event_count'],1)

    def test_extra_detection_resets_stability_no_top_four_selection(self):
        t=self.make_auto()
        objects=[detection(i) for i in range(1,5)]
        extra={'class_name':'wafer_slot_empty','confidence':.8,'xyxy':[280,280,320,320]}
        t.update(objects,0)
        result=t.update(objects+[extra],.1)
        self.assertEqual(result['initialization']['reason'],'too_many_slot_detections')
        t.update(objects,.2)
        self.assertIsNone(t.points)
        t.update(objects,.3)
        self.assertIsNotNone(t.points)

    def test_unseen_occluded_slot_and_transition_block_initialization(self):
        objects=[detection(i) for i in range(1,5)]
        hand={'class_name':'hand','confidence':.9,'xyxy':[85,85,115,115]}
        for additions,enabled,reason in (([hand],True,'waiting_for_slot_geometry'),
            ([],False,'rotation_or_transition')):
            with self.subTest(reason=reason):
                t=self.make_auto()
                for now in (0,.1):
                    result=t.update(objects+additions,now,observations_enabled=enabled)
                self.assertIsNone(t.points)
                self.assertEqual(result['initialization']['reason'],reason)

    def test_class_conflict_only_blocks_that_slots_state(self):
        t=self.make_auto()
        objects=[detection(i) for i in range(1,5)]+[detection(1,'filled',score=.85)]
        for now in (0,.1):
            result=t.update(objects,now)
        self.assertIsNotNone(t.points)
        self.assertEqual(result['slots'][0]['reason'],'empty_filled_conflict')
        self.assertEqual([s['state'] for s in result['slots']],['unknown','empty','empty','empty'])
        self.assertEqual(t.events,[])

    def test_fast_unload_before_identity_confirmation_is_reported_once(self):
        t=SlotGroupTracker(initialization_seconds=.6,initialization_frames=3,
                           confirm_seconds=.1,minimum_frames=2)
        full=[detection(i,'filled') for i in range(1,5)]
        empty_first=[detection(i,'empty' if i==1 else 'filled') for i in range(1,5)]
        hand={'class_name':'hand','confidence':.9,'xyxy':[85,85,115,115]}
        for now,objects in ((0,full),(.1,full),(.2,full+[hand]),(.3,full+[hand]),
                            (.4,empty_first),(.5,empty_first)):
            result=t.update(objects,now)
            self.assertIsNone(t.points)
            self.assertEqual(result['new_events'],[])
        result=t.update(empty_first,.6)
        self.assertEqual(result['identity_status'],'tracking')
        self.assertEqual(len(result['new_events']),1)
        event=result['new_events'][0]
        self.assertEqual((event['slot_id'],event['action']),(1,'unload'))
        self.assertEqual((event['time_lower'],event['time_upper']),(.1,.4))
        self.assertEqual((event['confirmed_at'],event['reported_at']),(.5,.6))
        self.assertTrue(event['recovered_before_initialization'])
        self.assertEqual(result['slots'][0]['state'],'empty')
        self.assertEqual(t.update(empty_first,.7)['new_events'],[])
        self.assertEqual(len(t.events),1)

    def test_three_visible_slots_can_confirm_previously_seen_four_slot_group(self):
        t=SlotGroupTracker(initialization_seconds=.4,initialization_frames=3,
                           confirm_seconds=.1,minimum_frames=2)
        for now in (0,.1): t.update([detection(i,'filled') for i in range(1,5)],now)
        for now in (.2,.3,.4): result=t.update([detection(i,'filled') for i in (2,3,4)],now)
        self.assertEqual(result['identity_status'],'tracking')
        self.assertEqual(result['slots'][0]['state'],'unknown')
        self.assertEqual(result['slots'][0]['last_confirmed_state'],'filled')
        self.assertEqual(t.events,[])
        for now in (.5,.6): result=t.update([detection(1)],now)
        self.assertEqual([e['slot_id'] for e in result['new_events']],[1])

    def test_entire_clockwise_unload_before_identity_confirmation_keeps_order(self):
        t=SlotGroupTracker(initialization_seconds=1.,initialization_frames=3,
                           confirm_seconds=.1,minimum_frames=2)
        states=['filled']*4
        for now in (0,.1): t.update([detection(i,states[i-1]) for i in range(1,5)],now)
        frame=2
        # 自动排序后槽号1,3,4,2，对应原输入点1,2,4,3。
        for index in (1,2,4,3):
            states[index-1]='empty'
            for _ in range(2):
                result=t.update([detection(i,states[i-1]) for i in range(1,5)],frame/10)
                frame+=1
                self.assertEqual(result['new_events'],[])
        result=t.update([detection(i,'empty') for i in range(1,5)],1.)
        self.assertEqual(result['confirmed_event_sequence'],[1,3,4,2])
        self.assertEqual(result['order_result'],'pass')
        self.assertEqual(len(result['new_events']),4)
        self.assertTrue(all(e['recovered_before_initialization'] for e in result['new_events']))

    def test_bad_box_under_hand_does_not_erase_other_tracks_or_prior_full_state(self):
        t=SlotGroupTracker(initialization_seconds=.6,initialization_frames=3,
                           confirm_seconds=.1,minimum_frames=2)
        full=[detection(i,'filled') for i in range(1,5)]
        for now in (0,.1): t.update(full,now)
        moved=CENTERS.copy()
        moved[0]+=[60,0] # 遮挡下槽1中心漂移，已超出匹配门限。
        hand={'class_name':'hand','confidence':.9,'xyxy':[75,75,175,125]}
        for now in (.2,.3):
            t.update([detection(i,'filled',centers=moved) for i in range(1,5)]+[hand],now)
        for now in (.4,.5,.6):
            result=t.update([detection(i,'empty' if i==1 else 'filled') for i in range(1,5)],now)
        self.assertEqual(result['identity_status'],'tracking')
        self.assertEqual(len(result['new_events']),1)
        self.assertEqual(result['new_events'][0]['slot_id'],1)
        self.assertEqual(result['new_events'][0]['time_lower'],.1)
        np.testing.assert_allclose(t.reference_points,CENTERS[[0,2,1,3]])

    def test_continuous_entry_motion_keeps_early_operation_evidence(self):
        t=SlotGroupTracker(initialization_seconds=.4,initialization_frames=3,
                           confirm_seconds=.1,minimum_frames=2)
        for frame,delta in enumerate((0,10,20,30,40,40,40,40,40)):
            objects=[detection(i,'filled' if frame<2 or i!=1 else 'empty',centers=CENTERS+[delta,0])
                     for i in range(1,5)]
            result=t.update(objects,round(frame*.1,6))
            if frame<8: self.assertIsNone(t.points)
        self.assertEqual(result['identity_status'],'tracking')
        self.assertEqual(len(result['new_events']),1)
        self.assertEqual(result['new_events'][0]['time_upper'],.2)

    def test_initial_empty_is_not_assumed_to_be_a_completed_unload(self):
        t=self.make_auto()
        for now in (0,.1,.2): result=t.update([detection(i) for i in range(1,5)],now)
        self.assertEqual(t.events,[])
        self.assertEqual(result['order_result'],'pending')

    def test_extra_candidate_discards_unpublished_operations(self):
        t=SlotGroupTracker(initialization_seconds=.5,initialization_frames=3,
                           confirm_seconds=.1,minimum_frames=2)
        full=[detection(i,'filled') for i in range(1,5)]
        empty=[detection(i) for i in range(1,5)]
        for now,objects in ((0,full),(.1,full),(.2,empty),(.3,empty)):
            t.update(objects,now)
        extra={'class_name':'wafer_slot_empty','confidence':.8,'xyxy':[280,280,320,320]}
        t.update(empty+[extra],.4)
        for frame in range(5,11): result=t.update(empty,frame/10)
        self.assertEqual(result['identity_status'],'tracking')
        self.assertEqual(t.events,[])

    def test_provisional_timeout_discards_prior_baseline(self):
        t=SlotGroupTracker(initialization_seconds=.5,initialization_frames=3,
                           confirm_seconds=.1,minimum_frames=2)
        for now in (0,.1): t.update([detection(i,'filled') for i in range(1,5)],now)
        for now in (2.,2.1,2.2,2.3,2.4,2.5): result=t.update([detection(i) for i in range(1,5)],now)
        self.assertEqual(result['identity_status'],'tracking')
        self.assertEqual(t.events,[])

    def test_history_window_drops_old_events_and_stale_baselines(self):
        t=SlotGroupTracker(initialization_seconds=1.,initialization_frames=3,
            initialization_history_seconds=.3,confirm_seconds=.1,minimum_frames=2)
        for frame in range(11):
            # 槽1的旧事件过期；槽2从第2帧起不可见，恢复时不能用过期满槽基线。
            visible=range(1,5) if frame<2 or frame>=8 else (1,3,4)
            objects=[detection(i,'filled' if frame<2 else 'empty') for i in visible]
            result=t.update(objects,frame/10)
        self.assertEqual(result['identity_status'],'tracking')
        self.assertEqual(t.events,[])

    def test_exact_duplicate_is_one_candidate(self):
        t=self.make_auto()
        objects=[detection(i) for i in range(1,5)]+[detection(1,score=.8)]
        t.update(objects,0)
        result=t.update(objects,.1)
        self.assertEqual(result['initialization']['eligible_slot_count'],4)
        self.assertIsNotNone(t.points)

    def test_roi_buffer_cannot_supply_fourth_initial_slot(self):
        t=self.make_auto()
        objects=[detection(i) for i in range(1,5)]
        for now in (0,.1):
            result=t.update(objects,now,initialization_objects=objects[:3])
        self.assertIsNone(t.points)
        self.assertEqual(result['initialization']['eligible_slot_count'],3)

    def test_discontinuous_geometry_restarts_initialization(self):
        t=self.make_auto()
        t.update([detection(i) for i in range(1,5)],0)
        moved=CENTERS+[200,0]
        t.update([detection(i,centers=moved) for i in range(1,5)],.1)
        self.assertIsNone(t.points)
        t.update([detection(i,centers=moved) for i in range(1,5)],.2)
        np.testing.assert_allclose(t.points,moved[[0,2,1,3]])

    def test_rotation_keeps_initial_ids_instead_of_sorting_x_again(self):
        t=self.make_auto()
        objects=[detection(i) for i in range(1,5)]
        t.update(objects,0)
        t.update(list(reversed(objects)),.1)
        angle=np.radians(5)
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        moved=(CENTERS-[150,150])@rotation.T+[150,150]
        result=t.update([detection(i,centers=moved) for i in (4,3,1,2)],.2)
        np.testing.assert_allclose(t.points,moved[[0,2,1,3]],atol=.05)
        self.assertGreater(result['slots'][0]['center_xy'][0],result['slots'][1]['center_xy'][0])

    def test_reset_group_drops_old_centers_and_events(self):
        t=self.make_auto()
        objects=[detection(i) for i in range(1,5)]
        t.update(objects,0)
        t.update(objects,.1)
        t.reset_group(2)
        self.assertIsNone(t.points)
        self.assertEqual(t.group_id,2)
        self.assertEqual(t.events,[])


class IncrementalEntryTests(unittest.TestCase):
    def make(self):
        t=SlotGroupTracker(group_id=2,confirm_seconds=.1,minimum_frames=2,initialization_seconds=10)
        t.start_incremental_entry(40)
        return t

    def test_two_visible_points_move_without_extrapolating_hidden_points(self):
        t=self.make()
        for now in (0,.1):
            t.update([detection(i,'filled') for i in range(1,5)],now)
        reference=t.reference_points.copy()
        moved=CENTERS+[-30,0]
        result=t.update([detection(i,'filled',centers=moved) for i in (1,3)],.2)
        np.testing.assert_allclose(t.reference_points[:2],reference[:2]+[-30,0])
        np.testing.assert_allclose(t.reference_points[2:],reference[2:])
        self.assertEqual([x['position_is_prediction'] for x in result['slots']],[False,False,True,True])
        self.assertEqual(result['identity_status'],'tracking')

    def test_state_change_with_shifted_box_uses_entry_pixel_gate(self):
        t=self.make()
        centers=CENTERS*.5
        for now in (0,.1):
            t.update([detection(i,'filled',centers=centers) for i in range(1,5)],now)
        moved=centers.copy()
        moved[0]+=[-30,0]  # 超过50px槽间距×0.4，但仍在逐点40px门限内。
        for now in (.2,.3):
            result=t.update([detection(1,'empty',centers=moved)],now)
        self.assertEqual(result['slots'][0]['state'],'empty')
        self.assertEqual([(e['slot_id'],e['action']) for e in t.events],[(1,'unload')])

    def test_occluded_entry_does_not_create_a_full_baseline(self):
        t=self.make()
        hand={'class_name':'hand','confidence':.9,'xyxy':[85,85,115,115]}
        for now in (0,.1,.2):
            result=t.update([detection(1,'filled'),hand],now)
        self.assertEqual(result['slots'][0]['state'],'unknown')
        for now in (.3,.4):
            result=t.update([detection(1,'empty')],now)
        self.assertEqual(result['slots'][0]['state'],'empty')
        self.assertEqual(result['initialization']['buffered_event_count'],0)


if __name__=='__main__':
    unittest.main(verbosity=2)
