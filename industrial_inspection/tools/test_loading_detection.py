"""上料视觉状态变化、手部遮挡、退出后核验和输出一致性的回归。"""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from industrial_inspection.tools import predict_slot_groups as predict
from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.test_group_prediction_pipeline import SyntheticModel
from industrial_inspection.tools.test_roi_group_sequence import POINTS, boxes

ROI = np.array([[80,80],[330,80],[330,330],[80,330]],dtype=np.float32)
HAND = {"class_name":"hand","confidence":.95,"xyxy":[185,185,215,215]}


def tracker(**kwargs):
    return SlotGroupTracker(POINTS,phase="load",confirm_seconds=.1,minimum_frames=2,**kwargs)


def contact(t):
    for now in (0.,.1):
        t.update(boxes(),now)
    for now in (.2,.3,.4):
        t.update(boxes(visible=(0,1,2))+[HAND],now)


class LoadingEvidenceTests(unittest.TestCase):
    def test_unknown_baseline_and_initial_filled_do_not_invent_a_load(self):
        t=tracker()
        for now in (0.,.1,.2):
            t.update(boxes(visible=(0,1,2))+[HAND],now)
        for now in (.3,.4):
            t.update(boxes(filled=(3,)),now)
        self.assertEqual(t.events,[])

    def test_interrupted_contact_is_not_accumulated(self):
        t=tracker()
        for now in (0.,.1):
            t.update(boxes(),now)
        for frame in range(2,10):
            t.update(boxes(visible=(0,1,2))+([HAND] if frame%2 == 0 else []),frame/10)
        self.assertEqual(t.events,[])

    def test_ambiguous_hand_covering_multiple_slots_is_not_assigned(self):
        t=tracker()
        for now in (0.,.1):
            t.update(boxes(),now)
        broad={**HAND,"xyxy":[80,80,220,220]}
        for now in (.2,.3,.4):
            t.update(boxes()+[broad],now)
        t.update([], .5)
        self.assertEqual(t.events,[])

    def test_hand_still_covering_slot_never_finalizes(self):
        t=tracker()
        contact(t)
        self.assertEqual(t.events,[])
        self.assertTrue(t.slots[3].load_contact_pending)

    def test_broad_hand_after_valid_contact_still_prevents_release(self):
        t=tracker()
        contact(t)
        broad={**HAND,"xyxy":[80,80,220,220]}
        t.update(boxes(visible=(0,1,2))+[broad],.9)
        self.assertEqual(t.events,[])
        self.assertIsNone(t.slots[3].load_contact_released_at)

    def test_confirmed_empty_after_contact_cancels_pending_load(self):
        t=tracker()
        contact(t)
        for now in (.5,.6):
            t.update(boxes(),now)
        self.assertFalse(t.slots[3].load_contact_pending)
        self.assertEqual(t.events,[])

    def test_first_filled_frame_is_not_preempted_by_inference(self):
        t=tracker()
        contact(t)
        for now in (.5,.6,.7):
            t.update(boxes(visible=(0,1,2)),now)
        t.update(boxes(filled=(3,)),.8)
        self.assertEqual(t.events,[])
        t.update(boxes(filled=(3,)),.9)
        self.assertEqual(len(t.events),1)
        self.assertFalse(t.events[0]["inferred"])
        self.assertEqual(t.events[0]["time_lower"],.1)
        self.assertEqual(t.events[0]["time_upper"],.8)

    def test_visual_confirmation_works_without_detecting_a_hand(self):
        t=tracker(load_hand_contact_enabled=False)
        contact(t)
        for now in (.5,.6,.7,.8):
            t.update(boxes(visible=(0,1,2)),now)
        self.assertEqual(t.events,[])
        for now in (.9,1.):
            t.update(boxes(filled=(3,)),now)
        self.assertEqual(len(t.events),1)
        self.assertEqual(t.events[0]["action"],"load")

    def test_lost_identity_and_outside_roi_cannot_create_contact_evidence(self):
        for lost,roi in ((True,ROI),(False,ROI+400)):
            t=tracker()
            for now in (0.,.1):
                t.update(boxes(),now)
            t.identity_lost=lost
            for now in (.2,.3,.4):
                t.update(boxes(visible=(0,1,2))+[HAND],now,roi_contour=roi)
            t.update([], .5,roi_contour=roi)
            self.assertEqual(t.events,[])

    def test_later_visual_s4_keeps_operation_order(self):
        t=tracker()
        for now in (0.,.1):
            t.update(boxes(),now)
        filled=set()
        for slot,now in ((1,.2),(0,.4),(2,.6)):
            filled.add(slot)
            for delta in (0.,.1):
                t.update(boxes(filled=filled),round(now+delta,6))
        for now in (.8,.9,1.):
            t.update(boxes(visible=(0,1,2),filled=filled)+[HAND],now)
        for now in (1.1,1.2,1.3,1.4):
            t.update(boxes(visible=(0,1,2),filled=filled),now)
        for now in (1.5,1.6):
            t.update(boxes(filled=(0,1,2,3)),now)
        self.assertEqual(t.snapshot()["confirmed_event_sequence"],[2,1,3,4])
        self.assertEqual(t.order_result(),"pass")


class LoadingPipelineTests(unittest.TestCase):
    def run_case(self, *, visual_fill, group_id=1, max_frames=0):
        filled=set()
        frames=[boxes(),boxes()]
        for slot in (1,0,2):
            filled.add(slot)
            frames.extend([boxes(filled=filled)]*2)
        frames.extend([boxes(visible=(0,1,2),filled=filled)+[HAND]]*3)
        frames.extend([boxes(visible=(0,1,2),filled=filled)]*4)
        moved=POINTS.copy()
        moved[0]=[75,100]  # 单点出ROI后立即切组；S4仍留在画面中。
        frames.append(boxes(moved,visible=(0,1,2),filled=filled))
        incoming=boxes(POINTS+[180,20],visible=(0,)) if group_id < 5 else []
        tail=boxes(moved,filled=(0,1,2,3)) if visual_fill else boxes(moved,visible=(0,1,2),filled=filled)
        frames.extend([tail+incoming]*4)
        sequence=RoiGroupSequence(tracker(group_id=group_id),ROI,exit_min_slots=1,partial_exit_slots=1)
        with tempfile.TemporaryDirectory(prefix="wafer_loading_") as directory:
            root=Path(directory)
            source=root/"source.mp4"
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*"mp4v"),10.,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in frames:
                writer.write(np.zeros((480,640,3),dtype=np.uint8))
            writer.release()
            args=argparse.Namespace(source=source,weights=root/"model.pt",output_dir=root/"results",
                conf=.35,imgsz=640,device="cpu",stride=1,show=False,max_frames=max_frames,nms_iou=.7)
            logs=[]
            with (patch.object(predict.base,"load_detection_model",return_value=(SyntheticModel(frames),"synthetic")),
                  patch.object(predict,"make_sequence",return_value=sequence),
                  patch.object(predict,"PROCESS_PHASE","load"),
                  patch.object(predict,"LOCAL_REDETECT_ENABLED",False)):
                result=predict.run_prediction(args,roi_polygon={"polygon":ROI/[640,480],
                    "size":(640,480),"start_frame":0},on_log=logs.append)
            saved=[json.loads(line) for line in (args.output_dir/"slot_events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["events"],saved)
            self.assertEqual(result["closed_groups"][0]["events"],saved)
            self.assertEqual([e["slot_id"] for e in saved[:3]],[2,1,3])
            return result,logs

    def test_rotation_visible_fill_belongs_to_old_group_and_is_reported_once(self):
        result,logs=self.run_case(visual_fill=True)
        events=result["events"]
        self.assertEqual([(e["group_id"],e["slot_id"]) for e in events],[(1,2),(1,1),(1,3),(1,4)])
        self.assertEqual(events[-1]["reason"],"detected_after_hand_contact")
        report=result["group_process_report"]["groups"][0]
        self.assertEqual(report["sequence"],[2,1,3,4])
        self.assertEqual(report["order_result"],"pass")
        self.assertEqual(logs.count("游星片1：上料完成"),1)

    def test_hidden_s4_is_not_completed_from_hand_contact_at_end(self):
        result,logs=self.run_case(visual_fill=False)
        self.assertEqual(len(result["events"]),3)
        report=result["group_process_report"]["groups"][0]
        self.assertEqual(report["slot_completed"],[1,1,1,0])
        self.assertNotIn("1*",logs[-1])
        self.assertNotIn("游星片1：上料完成",logs)

    def test_g5_exit_still_checks_rotation_visible_fill(self):
        result,_=self.run_case(visual_fill=True,group_id=5)
        self.assertEqual(result["events"][-1]["slot_id"],4)
        self.assertFalse(result["events"][-1]["inferred"])
        self.assertEqual(result["final_group_snapshot"]["group_status"],"sequence_complete")

    def test_early_stop_does_not_infer_an_incomplete_recording(self):
        result,_=self.run_case(visual_fill=False,max_frames=17)
        self.assertEqual(len(result["events"]),3)
        self.assertEqual(result["group_process_report"]["matrix"][0],[1,1,1,0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
