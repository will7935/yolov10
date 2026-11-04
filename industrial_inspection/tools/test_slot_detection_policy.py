"""验证职责门槛隔离，以及独立门槛确实进入预测、审计和局部补检。"""
import argparse
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from industrial_inspection.tools import predict_slot_groups as predict
from industrial_inspection.tools.slot_detection_policy import DetectionThresholds, prepare_business_detections
from industrial_inspection.tools.test_group_prediction_pipeline import SyntheticModel
from industrial_inspection.tools.test_roi_group_sequence import ROI, boxes


class ThresholdTests(unittest.TestCase):
    def test_legacy_default_and_separate_hand_override(self):
        legacy = DetectionThresholds.resolve(.1, .001)
        independent = DetectionThresholds.resolve(.1, .001, hand=.02)
        self.assertEqual(legacy.tracking_low, .05)
        for name in ('display','audit','slot_state','handle','tracking_high','tracking_low'):
            self.assertEqual(getattr(legacy, name), getattr(independent, name))
        hand = {'class_name':'hand','confidence':.025}
        self.assertEqual(prepare_business_detections([hand], legacy), [])
        self.assertEqual(prepare_business_detections([hand], independent), [hand])

    def test_tracking_threshold_does_not_change_state_hand_or_handle(self):
        policy = DetectionThresholds.resolve(.35, .001, tracking_high=.5, tracking_low=.08)
        self.assertEqual((policy.slot_state,policy.hand,policy.handle),(.35,.35,.35))
        self.assertEqual((policy.tracking_high,policy.tracking_low),(.5,.08))

    def test_slot_state_override_preserves_identity_requirement_and_raw_score(self):
        policy = DetectionThresholds.resolve(.35, .001, slot_state=.2)
        item = {'class_name':'wafer_slot_filled','confidence':.1,'tracking_confidence':.25}
        self.assertEqual(policy.admission(item)[1], 'missing_bytetrack_id')
        item['track_id'] = 7
        self.assertEqual(prepare_business_detections([item],policy)[0]['confidence'],.25)
        self.assertEqual(item['confidence'],.1)
        self.assertEqual(policy.admission({'class_name':'hand','confidence':.25})[0],False)

    def test_invalid_or_unavailable_evidence_thresholds_are_rejected(self):
        for options in ({'hand':.0001},{'hand':float('nan')},{'handle':0.},
                        {'tracking_high':.1,'tracking_low':.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DetectionThresholds.resolve(.35,.001,**options)

    def test_independent_threshold_reaches_pipeline_and_saved_audit(self):
        hand = {'class_name':'hand','confidence':.025,'xyxy':[10,10,30,30]}
        handle = {'class_name':'hatch_handle','confidence':.25,'xyxy':[20,20,40,40]}
        frames = [boxes()+[hand,handle] for _ in range(6)]
        with tempfile.TemporaryDirectory(prefix='slot_thresholds_') as directory:
            root = Path(directory)
            source = root/'source.mp4'
            writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in frames:
                writer.write(np.zeros((480,640,3),dtype=np.uint8))
            writer.release()
            args = argparse.Namespace(source=source,weights=root/'model.pt',output_dir=root/'results',
                conf=.35,audit_conf=.001,imgsz=640,device='cpu',stride=1,show=False,max_frames=0,nms_iou=.7)
            with (patch.object(predict.base,'load_detection_model',return_value=(SyntheticModel(frames),'synthetic')),
                  patch.multiple(predict,HAND_DETECTION_CONFIDENCE=.02,HATCH_HANDLE_CONFIDENCE=.3,
                      SLOT_STATE_CONFIDENCE=None,BYTE_TRACK_HIGH_CONFIDENCE=None,
                      BYTE_TRACK_LOW_CONFIDENCE_OVERRIDE=None,HATCH_CONTROL_ENABLED=False,
                      AUTO_GROUP_SWITCH=False,LOCAL_REDETECT_ENABLED=False),
                  contextlib.redirect_stdout(io.StringIO())):
                sequence = predict.make_sequence(ROI/[640,480],(640,480))
                with (patch.object(predict,'make_sequence',return_value=sequence),
                      patch.object(sequence,'update',wraps=sequence.update) as update):
                    summary = predict.run_prediction(args,roi_polygon={
                        'polygon':ROI/[640,480],'size':(640,480),'start_frame':0})
                objects = update.call_args_list[0].args[0]
                self.assertTrue(any(obj['class_name']=='hand' for obj in objects))
                self.assertFalse(any(obj['class_name']=='hatch_handle' for obj in objects))
            self.assertEqual(summary['effective_detection_thresholds']['hand'],.02)
            self.assertEqual(summary['slot_tracking']['high_confidence'],.35)
            self.assertEqual(summary['algorithm_configuration']['HAND_DETECTION_CONFIDENCE'],.02)
            with (args.output_dir/'detection_confidences.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows = list(csv.DictReader(stream))
            recorded = next(row for row in rows if row['class_name']=='hand')
            self.assertEqual(recorded['business_admitted'],'True')
            self.assertEqual(recorded['business_confidence_threshold'],'0.02')
            self.assertEqual(recorded['meets_display_confidence'],'False')
            with (args.output_dir/'slot_states.csv').open(encoding='utf-8-sig',newline='') as stream:
                self.assertIn('s4_contact_diagnostic',next(csv.reader(stream)))


if __name__ == '__main__':
    unittest.main()
