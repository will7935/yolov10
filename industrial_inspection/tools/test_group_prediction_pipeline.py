"""用已知检测序列验证完整视频、UI回调和五组结果保存，无需GPU和模型。"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from industrial_inspection.tools import predict_slot_groups as predict
from industrial_inspection.tools.test_roi_group_sequence import POINTS,ROI,boxes


class SyntheticModel:
    names={0:'hand',1:'hatch_handle',2:'water_gun',3:'wafer_slot_empty',4:'wafer_slot_filled'}

    def __init__(self,frames):
        self.frames=iter(frames)

    def predict(self,*args,**kwargs):
        objects=next(self.frames)
        classes={name:index for index,name in self.names.items()}
        result=SimpleNamespace(cls=np.array([classes[o['class_name']] for o in objects]),
                               conf=np.array([o['confidence'] for o in objects]),
                               xyxy=np.array([o['xyxy'] for o in objects]))
        result.cpu=lambda: result
        return [SimpleNamespace(boxes=result)]


class PipelineTest(unittest.TestCase):
    def setUp(self):
        hatch_control=patch.object(predict,'HATCH_CONTROL_ENABLED',False)
        hatch_control.start()
        self.addCleanup(hatch_control.stop)
        self.auto_load=patch.object(predict,'AUTO_LOAD_AFTER_UNLOAD',False)
        self.auto_load.start()
        self.addCleanup(self.auto_load.stop)
        # 这些整图检测序列不包含裁剪推理；局部补检由独立端到端用例覆盖。
        self.local_redetection=patch.object(predict,'LOCAL_REDETECT_ENABLED',False)
        self.local_redetection.start()
        self.addCleanup(self.local_redetection.stop)

    def test_single_exit_saves_and_displays_full_state_from_one_incoming_point(self):
        roi=np.array([[80,80],[410,80],[410,310],[80,310]],dtype=float)
        old=POINTS.copy()
        old[0]=[75,100]
        incoming=POINTS+[180,80]
        before=boxes(old)+boxes(incoming,visible=(0,),filled=(0,))
        after=boxes(old)+boxes(incoming,visible=(0,))
        frames=[boxes()]*6+[before]*2+[after]*2+[
            boxes(old)+boxes(incoming,visible=(1,2,3),filled=(1,2,3))]*5
        with tempfile.TemporaryDirectory(prefix='wafer_incremental_entry_') as directory:
            root=Path(directory)
            source=root/'synthetic.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in frames:
                writer.write(np.zeros((480,640,3),dtype=np.uint8))
            writer.release()
            args=argparse.Namespace(source=source,weights=root/'synthetic.pt',output_dir=root/'results',
                conf=.35,imgsz=640,device='cpu',stride=1,show=False,max_frames=0,nms_iou=.7)
            context={'polygon':roi/np.array([640,480]),'size':(640,480),'start_frame':0}
            ui=[]
            with (patch.object(predict.base,'load_detection_model',return_value=(SyntheticModel(frames),'synthetic')),
                  patch.object(predict,'CURRENT_GROUP_ID',1),patch.object(predict,'PROCESS_PHASE','unload'),
                  patch.object(predict,'AUTO_GROUP_SWITCH',True),patch.object(predict,'EXIT_MIN_SLOTS',1),
                  patch.object(predict,'PARTIAL_EXIT_MIN_SLOTS',1),patch.object(predict,'CONFIRM_SECONDS',.1),
                  patch.object(predict,'MIN_CONFIRM_FRAMES',2)):
                summary=predict.run_prediction(args,roi_polygon=context,on_frame=lambda image,info:ui.append(info))
            filled=[r for r in ui if r['group_id']==2 and r['group_status']=='entering'
                    and r['slots'][0]['state']=='filled']
            self.assertTrue(filled)
            self.assertEqual(filled[0]['slot_numbering_status'],'provisional')
            self.assertIn('临时点1:已填充',filled[0]['result'])
            states=[json.loads(line) for line in (args.output_dir/'slot_states.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(r['group_id']==2 and r['group_status']=='entering'
                                and r['slots'][0]['state']=='filled' for r in states))
            events=[json.loads(line) for line in (args.output_dir/'slot_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([(e['group_id'],e['slot_id'],e['action']) for e in events],[(2,1,'unload')])
            self.assertTrue(events[0]['recovered_before_initialization'])
            self.assertEqual(summary['events'],events)
            self.assertEqual(summary['closed_groups'][0]['events'],[])

    def test_early_unloads_reach_saved_events_summary_and_ui(self):
        hand={"class_name":"hand","confidence":.9,"xyxy":[85,85,115,115]}
        before=boxes(filled=(0,1,2,3))
        after=boxes(filled=(1,2,3))
        early=[before,before,before+[hand],before+[hand]]+[after]*6
        frames=early+[boxes(POINTS+[delta,0],filled=(1,2,3)) for delta in range(10,281,10)]+early
        with tempfile.TemporaryDirectory(prefix='wafer_early_events_') as directory:
            root=Path(directory)
            source=root/'synthetic.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in frames: writer.write(np.zeros((480,640,3),dtype=np.uint8))
            writer.release()
            args=argparse.Namespace(source=source,weights=root/'synthetic.pt',output_dir=root/'results',
                conf=.35,imgsz=640,device='cpu',stride=1,show=False,max_frames=0,nms_iou=.7)
            context={'polygon': ROI/np.array([640,480]),'size': (640,480),'start_frame':0}
            ui=[]
            logs=[]
            with (patch.object(predict.base,'load_detection_model',return_value=(SyntheticModel(frames),'synthetic')),
                  patch.object(predict,'CURRENT_GROUP_ID',1),patch.object(predict,'PROCESS_PHASE','unload'),
                  patch.object(predict,'AUTO_GROUP_SWITCH',True),patch.object(predict,'EXIT_DIRECTION',None),
                  patch.object(predict,'INITIALIZATION_SECONDS',.6),patch.object(predict,'NEXT_GROUP_STABLE_SECONDS',.6),
                  patch.object(predict,'EXIT_MIN_SLOTS',3),patch.object(predict,'PARTIAL_EXIT_MIN_SLOTS',2),
                  patch.object(predict,'CONFIRM_SECONDS',.1),
                  patch.object(predict,'MIN_CONFIRM_FRAMES',2)):
                summary=predict.run_prediction(args,roi_polygon=context,on_frame=lambda image,info:ui.append(info),on_log=logs.append)
            events=[json.loads(line) for line in (args.output_dir/'slot_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([(e['group_id'],e['slot_id'],e['action']) for e in events],[(1,1,'unload'),(2,1,'unload')])
            self.assertEqual(events,summary['events'])
            self.assertTrue(all(e['recovered_before_initialization'] for e in events))
            for event in events:
                self.assertLess(event['confirmed_at'],event['reported_at'])
            self.assertEqual(summary['closed_groups'][0]['events'],events[:1])
            self.assertEqual(summary['final_group_snapshot']['confirmed_event_sequence'],[1])
            for group_id in (1,2):
                visible=[info for info in ui if info['group_id']==group_id and info['group_status']=='active']
                self.assertTrue(visible)
                self.assertEqual(visible[0]['slots'][0]['state'],'empty')
            self.assertEqual(logs[:-1],['游星片1：下料未完成','游星片2：下料未完成'])
            self.assertIn('游星片槽位下料完成矩阵',logs[-1])
            self.assertEqual(summary['group_process_report']['matrix'][:2],[[1,0,0,0],[1,0,0,0]])

    def test_completed_groups_log_once_then_save_matrix_and_order_at_end(self):
        frames=[]
        for _ in range(5):
            filled={0,1,2,3}
            frames.extend([boxes(filled=filled)]*12)
            # 首次按x排序后，物理索引2、0、1、3对应槽2、1、3、4。
            for physical_slot in (2,0,1,3):
                filled.remove(physical_slot)
                frames.extend([boxes(filled=filled)]*3)
            frames.extend(boxes(POINTS+[delta,0]) for delta in range(10,281,10))
        with tempfile.TemporaryDirectory(prefix='wafer_completion_report_') as directory:
            root=Path(directory)
            source=root/'source.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in frames:
                writer.write(np.zeros((480,640,3),dtype=np.uint8))
            writer.release()
            args=argparse.Namespace(source=source,weights=root/'model.pt',output_dir=root/'results',
                conf=.35,imgsz=640,device='cpu',stride=1,show=False,max_frames=0,nms_iou=.7)
            context={'polygon':ROI/[640,480],'size':(640,480),'start_frame':0}
            logs=[]
            with (patch.object(predict.base,'load_detection_model',return_value=(SyntheticModel(frames),'synthetic')),
                  patch.object(predict,'CONFIRM_SECONDS',.1),patch.object(predict,'MIN_CONFIRM_FRAMES',2)):
                summary=predict.run_prediction(args,roi_polygon=context,on_log=logs.append)
            expected=[f'游星片{group}：下料{state}' for group in range(1,6) for state in ('未完成','完成')]
            self.assertEqual(logs[:-1],expected)
            self.assertEqual(summary['group_process_report']['matrix'],[[1,1,1,1]]*5)
            self.assertTrue(all(g['sequence']==[2,1,3,4] for g in summary['group_process_report']['groups']))
            saved=json.loads((args.output_dir/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['group_process_report'],summary['group_process_report'])
            self.assertEqual((args.output_dir/'group_process_report.txt').read_text(encoding='utf-8'),logs[-1]+'\n')
            self.assertIn('G5：2 → 1 → 3 → 4',logs[-1])

    def test_five_groups_saved_and_reported_through_ui_callbacks(self):
        frames=[]
        for _ in range(5):
            frames.extend([boxes() for _ in range(12)])
            frames.extend(boxes(POINTS+[delta,0]) for delta in range(10,281,10))
        with tempfile.TemporaryDirectory(prefix='wafer_groups_') as directory:
            root=Path(directory)
            source=root/'synthetic.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,480))
            self.assertTrue(writer.isOpened())
            for objects in frames:
                image=np.zeros((480,640,3),dtype=np.uint8)
                for obj in objects:
                    a=obj['xyxy']
                    cv2.circle(image,(int((a[0]+a[2])/2),int((a[1]+a[3])/2)),8,(255,255,255),-1)
                writer.write(image)
            writer.release()
            args=argparse.Namespace(source=source,weights=root/'synthetic.pt',output_dir=root/'results',
                conf=.35,imgsz=640,device='cpu',stride=1,show=False,max_frames=0,nms_iou=.7)
            context={'polygon': ROI/np.array([640,480]),'size': (640,480),'start_frame':0}
            statuses=[]
            def on_frame(image,info):
                self.assertEqual(image.shape,(480,640,3))
                self.assertEqual(len(info['slots']),4)
                statuses.append((info['group_id'],info['group_status']))
            with (patch.object(predict.base,'load_detection_model',return_value=(SyntheticModel(frames),'synthetic')),
                  patch.object(predict,'CURRENT_GROUP_ID',1),patch.object(predict,'AUTO_GROUP_SWITCH',True),
                  patch.object(predict,'EXIT_DIRECTION',None)):
                summary=predict.run_prediction(args,roi_polygon=context,on_frame=on_frame)
            self.assertEqual(summary['processed_frames'],len(frames))
            self.assertEqual(summary['status'],'end_of_stream')
            self.assertEqual(summary['final_group_snapshot']['group_status'],'sequence_complete')
            self.assertEqual([g['group_id'] for g in summary['closed_groups']],[1,2,3,4,5])
            self.assertEqual(sorted(set(g for g,_ in statuses)),[1,2,3,4,5])
            self.assertTrue(any(s=='entering' for _,s in statuses))
            self.assertEqual(summary['events'],[])
            output=args.output_dir
            states=[json.loads(l) for l in (output/'slot_states.jsonl').read_text(encoding='utf-8').splitlines()]
            groups=[json.loads(l) for l in (output/'group_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(states),len(frames))
            self.assertEqual(sum(g['type']=='group_closed' for g in groups),5)
            for record in states: json.dumps(record,allow_nan=False)
            cap=cv2.VideoCapture(str(output/'annotated.mp4'))
            decoded=0
            while cap.read()[0]: decoded+=1
            cap.release()
            self.assertEqual(decoded,len(frames))


if __name__ == '__main__':
    unittest.main(verbosity=2)
