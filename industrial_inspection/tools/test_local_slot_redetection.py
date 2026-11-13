"""验证局部补检坐标、槽位归属、遮挡及补检结果进入实际取放事件保存流程。"""
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
from industrial_inspection.tools.test_roi_group_sequence import POINTS, ROI, boxes


class LocalRedetectionTests(unittest.TestCase):
    def setUp(self):
        hatch_control = patch.object(predict, 'HATCH_CONTROL_ENABLED', False)
        hatch_control.start()
        self.addCleanup(hatch_control.stop)
        self.frame=np.zeros((480,640,3),dtype=np.uint8)
        self.args=argparse.Namespace(conf=.35,device='cpu',nms_iou=.7)
        self.sequence=RoiGroupSequence(SlotGroupTracker(POINTS,group_id=2),ROI,
                                     exit_min_slots=1,partial_exit_slots=1)
        self.sequence.tracker._point_tracking_gate=40
        self.config=patch.multiple(predict,LOCAL_REDETECT_ENABLED=True,
                                   LOCAL_REDETECT_CROP_SIZE=(180,140),LOCAL_REDETECT_IMAGE_SIZE=320)
        self.config.start()
        self.addCleanup(self.config.stop)

    def recover(self, objects, local):
        model=SyntheticModel([local])
        model.predict=unittest.mock.Mock(wraps=model.predict)
        result,attempts=predict.recover_missing_slots(model,self.frame,objects,self.sequence,self.args)
        return result,attempts,model

    def test_complete_full_frame_needs_no_extra_inference(self):
        result,attempts,model=self.recover(boxes(),[])
        self.assertEqual(result,boxes())
        self.assertEqual(attempts,[])
        model.predict.assert_not_called()

    def test_missing_s1_to_s3_never_trigger_crops(self):
        result, attempts, model = self.recover(boxes(visible=(3,)), [])
        self.assertEqual(result, boxes(visible=(3,)))
        self.assertEqual(attempts, [])
        model.predict.assert_not_called()

    def test_only_s4_is_cropped_when_multiple_slots_are_missing(self):
        local = [{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[80,60,100,80]}]
        result, attempts, model = self.recover(boxes(visible=(0,)), local)
        self.assertEqual([attempt['slot_id'] for attempt in attempts], [4])
        self.assertEqual(result[-1]['xyxy'], [190,190,210,210])
        model.predict.assert_called_once()

    def test_local_hand_threshold_does_not_lower_slot_evidence_threshold(self):
        local=[{'class_name':'hand','confidence':.025,'xyxy':[80,60,100,80]},
               {'class_name':'wafer_slot_filled','confidence':.25,'xyxy':[80,60,100,80]}]
        with patch.object(predict,'HAND_DETECTION_CONFIDENCE',.02):
            result,_,model=self.recover(boxes(visible=(0,1,2)),local)
        self.assertEqual(model.predict.call_args.kwargs['conf'],.02)
        additions=result[3:]
        self.assertEqual(additions, [])

    def test_crop_coordinates_return_to_source_and_wrong_slot_is_rejected(self):
        # 槽4中心(200,200)，裁剪原点(110,130)；第二个框属于邻槽3，不能用于槽4。
        local=[{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[80,60,100,80]},
               {'class_name':'wafer_slot_empty','confidence':.99,'xyxy':[80,0,100,20]}]
        result,attempts,model=self.recover(boxes(visible=(0,1,2)),local)
        self.assertEqual(len(result),4)
        self.assertEqual(result[-1]['xyxy'],[190,190,210,210])
        self.assertEqual(result[-1]['detection_source'],'local_crop')
        self.assertEqual(attempts[0]['crop_xyxy'],[110,130,290,270])
        self.assertEqual(attempts[0]['slot_id'],4)
        self.assertEqual(model.predict.call_args.args[0].shape,(140,180,3))
        self.assertEqual(model.predict.call_args.kwargs['conf'],.35)

    def test_crop_at_image_edge_is_clipped_and_coordinates_stay_correct(self):
        points=POINTS.copy()
        points[3]=[630,460]
        self.sequence=RoiGroupSequence(SlotGroupTracker(points),
            np.array([[0,0],[640,0],[640,480],[0,480]],dtype=np.float32))
        self.sequence.tracker._point_tracking_gate=40
        local=[{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[80,60,100,80]}]
        result,attempts,model=self.recover(boxes(points,visible=(0,1,2)),local)
        self.assertEqual(result[-1]['xyxy'],[620,450,640,470])
        self.assertEqual(attempts[0]['crop_xyxy'],[540,390,640,480])
        self.assertEqual(model.predict.call_args.args[0].shape,(90,100,3))

    def test_detected_hand_in_crop_still_blocks_state_confirmation(self):
        tracker=self.sequence.tracker
        for now in [0.,.2,.4]:
            self.sequence.update(boxes(filled=(0,1,2,3)),now)
        local=[{'class_name':'wafer_slot_empty','confidence':.99,'xyxy':[80,60,100,80]},
               {'class_name':'hand','confidence':.95,'xyxy':[75,55,105,85]}]
        result,attempts,_=self.recover(boxes(visible=(0,1,2)),local)
        snapshot=self.sequence.update(result,.5)
        self.assertTrue(attempts[0]['accepted_detections'])
        self.assertEqual(snapshot['slots'][3]['reason'],'hand_occlusion')
        self.assertEqual(snapshot['slots'][3]['last_confirmed_state'],'filled')
        self.assertEqual(snapshot['new_events'],[])

    def test_loading_does_not_run_local_redetection(self):
        self.sequence.tracker.phase='load'
        visible = boxes(visible=(0, 1, 2))
        local = [{'class_name': 'wafer_slot_filled', 'confidence': .95, 'xyxy': [80, 60, 100, 80]}]
        result, attempts, model = self.recover(visible, local)
        self.assertEqual(result, visible)
        self.assertEqual(attempts, [])
        model.predict.assert_not_called()

    def test_other_group_stages_and_lost_identity_never_run_crops(self):
        for state in ('initializing','entering','exiting','waiting_next','sequence_complete','identity_uncertain'):
            with self.subTest(state=state):
                self.sequence.state=state
                _,attempts,model=self.recover([],[])
                self.assertEqual(attempts,[])
                model.predict.assert_not_called()
        self.sequence.state='active'
        self.sequence.tracker.identity_lost=True
        _,_,model=self.recover([],[])
        model.predict.assert_not_called()

    def test_crop_center_shift_does_not_move_track_or_change_group(self):
        for now in [0.,.2,.4]:
            self.sequence.update(boxes(filled=(0,1,2,3)),now)
        original=self.sequence.tracker.reference_points.copy()
        # 局部框中心向左偏25px，但它只提供槽4的空满状态。
        local=[{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[55,60,75,80]}]
        for now in [.5,.7,.9]:
            result,_,_=self.recover(boxes(visible=(0,1,2),filled=(0,1,2)),local)
            snapshot=self.sequence.update(result,now)
        np.testing.assert_allclose(self.sequence.tracker.reference_points,original)
        self.assertEqual(snapshot['slots'][3]['center_xy'],[200.,200.])
        self.assertEqual(snapshot['slots'][3]['state'],'empty')
        self.assertEqual(snapshot['group_id'],2)
        self.assertEqual(snapshot['group_status'],'active')
        self.assertTrue(snapshot['s4_unload_review_pending'])
        self.sequence.finish()
        self.assertEqual([(e['slot_id'],e['action']) for e in self.sequence.all_events],[(4,'unload')])

    def test_outside_roi_and_distant_boxes_do_not_fill_missing_slot(self):
        self.sequence.contour=np.array([[80,80],[225,80],[225,300],[80,300]],dtype=np.float32)
        # S4 裁剪原点为 (110,130)。首个框中心距 S4 30px 但越出 ROI，第二个距 S4 45px。
        local=[{'class_name':'wafer_slot_empty','confidence':.99,'xyxy':[110,60,130,80]},
               {'class_name':'wafer_slot_empty','confidence':.99,'xyxy':[80,105,100,125]}]
        result,attempts,_=self.recover(boxes(visible=(0,1,2)),local)
        self.assertEqual(len(result),3)
        self.assertEqual(attempts[0]['accepted_detections'],[])

    def test_fallback_uses_alternate_gate_but_keeps_geometric_center(self):
        tracker = self.sequence.tracker
        tracker._s4_before_recalculation = np.array([175., 200.])
        local = {'class_name': 'wafer_slot_filled', 'confidence': .9, 'xyxy': [60, 60, 80, 80]}
        visible = boxes(visible=(0, 1, 2), filled=(0, 1, 2))
        model = SyntheticModel([[], [local]])
        observations, attempts = predict.recover_missing_slots(
            model, self.frame, visible, self.sequence, self.args)
        self.assertEqual([a['crop_xyxy'] for a in attempts], [[110, 130, 290, 270], [85, 130, 265, 270]])
        self.assertEqual(attempts[1]['accepted_detections'][0]['xyxy'], [145, 190, 165, 210])
        result = self.sequence.update(observations, .1)
        self.assertEqual(result['slots'][3]['observation'], 'filled')
        result = self.sequence.update(visible, .2)
        self.assertEqual(result['slots'][3]['temporary_contact_box_xyxy'], [145., 190., 165., 210.])
        self.assertEqual(result['slots'][3]['center_xy'], [200., 200.])
        np.testing.assert_allclose(tracker.reference_points[3], [200., 200.])
        self.assertIsNone(result['slots'][3]['byte_track_id'])

    def test_old_group_crop_is_not_used_to_initialize_next_group(self):
        roi=np.array([[80,80],[410,80],[410,310],[80,310]],dtype=np.float32)
        sequence=RoiGroupSequence(SlotGroupTracker(POINTS,group_id=2),roi,
                                  exit_min_slots=1,partial_exit_slots=1)
        for now in [0.,.2,.4]:
            sequence.update(boxes(),now)
        moved=POINTS.copy()
        moved[0]=[75,100]
        old_crop={'class_name':'wafer_slot_filled','confidence':.99,
                  'xyxy':[290,190,310,210],'detection_source':'local_crop'}
        snapshot=sequence.update(boxes(moved)+[old_crop],.5)
        self.assertEqual((snapshot['group_id'],snapshot['group_status']),(3,'entering'))
        self.assertEqual(snapshot['initialization']['eligible_slot_count'],0)
        self.assertTrue(all(slot['center_xy'] is None for slot in snapshot['slots']))

    def test_local_empty_detection_is_confirmed_and_saved_as_one_unload(self):
        full_frames=[boxes(filled=(0,1,2,3))]*12+[boxes(visible=(0,1,2),filled=(0,1,2))]*8
        local=[{'class_name':'wafer_slot_empty','confidence':.9,'xyxy':[80,60,100,80]}]

        class MixedModel(SyntheticModel):
            def __init__(self):
                super().__init__(full_frames)
                self.local_calls=0

            def predict(self,image,**kwargs):
                if image.shape[:2] == (480,640):
                    return super().predict(image,**kwargs)
                self.local_calls+=1
                return SyntheticModel([local]).predict(image,**kwargs)

        with tempfile.TemporaryDirectory(prefix='wafer_crop_recovery_') as directory:
            root=Path(directory)
            source=root/'source.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in full_frames:
                writer.write(self.frame)
            writer.release()
            args=argparse.Namespace(source=source,weights=root/'model.pt',output_dir=root/'results',
                conf=.35,imgsz=1280,device='cpu',stride=1,show=False,max_frames=0,nms_iou=.7)
            context={'polygon':ROI/[640,480],'size':(640,480),'start_frame':0}
            model=MixedModel()
            with (patch.object(predict.base,'load_detection_model',return_value=(model,'synthetic')),
                  patch.object(predict,'CONFIRM_SECONDS',.2),patch.object(predict,'MIN_CONFIRM_FRAMES',3)):
                summary=predict.run_prediction(args,roi_polygon=context)
            self.assertEqual(model.local_calls,8)
            self.assertEqual([(e['group_id'],e['slot_id'],e['action']) for e in summary['events']],[(1,4,'unload')])
            self.assertAlmostEqual(summary['events'][0]['confirmed_at'],1.4)
            saved=[json.loads(line) for line in (args.output_dir/'slot_states.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(saved[-1]['slots'][3]['state'],'empty')
            self.assertEqual(saved[-1]['local_redetection'][0]['group_id'],1)
            self.assertEqual(saved[-1]['local_redetection'][0]['accepted_detections'][0]['xyxy'],[190,190,210,210])


if __name__ == '__main__':
    unittest.main(verbosity=2)
