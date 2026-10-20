"""开门门控、漏检/抖动防误关及逐组二十槽统计的回归测试。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from industrial_inspection.tools import predict_slot_groups as predict
from industrial_inspection.tools.hatch_loading_inspection import (
    HatchHandleMonitor, HatchLoadingInspection, build_loading_fullness_report)
from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.test_group_prediction_pipeline import SyntheticModel
from industrial_inspection.tools.test_roi_group_sequence import POINTS, ROI, boxes
from industrial_inspection.tools.test_unload_load_cycle import group_frames


def handle(state, confidence=.95):
    return {'class_name': 'hatch_handle', 'confidence': confidence,
            'xyxy': [278, 48, 298, 68] if state == 'open' else [330, 100, 350, 140]}


def monitor():
    return HatchHandleMonitor(predict.HATCH_OPEN_POLYGON, predict.HATCH_CLOSED_POLYGON,
                              confirm_seconds=.2, minimum_frames=3)


class HatchHandleTests(unittest.TestCase):
    def test_missing_handles_never_close_confirmed_open_door(self):
        detector = monitor()
        for now in (0., .1, .2):
            events = detector.update([handle('open')], now, (640, 480))
        self.assertEqual(events[0]['type'], 'hatch_opened')
        for now in (.3, 1., 20.):
            self.assertEqual(detector.update([], now, (640, 480)), [])
            self.assertEqual(detector.state, 'open')

    def test_confirmed_closed_then_long_handle_gap_confirms_open(self):
        detector = monitor()
        for now in (0., .1, .2):
            detector.update([handle('closed')], now, (640, 480))
        self.assertEqual(detector.state, 'closed')
        self.assertEqual(detector.update([], .9, (640, 480)), [])
        self.assertEqual(detector.update([], 1.3, (640, 480)), [])
        events = detector.update([], 1.4, (640, 480))
        events += detector.update([], 1.5, (640, 480))
        self.assertEqual([e['type'] for e in events], ['hatch_opened'])
        self.assertEqual(detector.state, 'open')

    def test_distance_hysteresis_works_without_open_polygon(self):
        detector = monitor()
        for now in (0., .1, .2):
            detector.update([handle('closed')], now, (640, 480))
        far = {'class_name': 'hatch_handle', 'confidence': .95,
               'xyxy': [245, 4, 265, 24]}
        for now in (.3, .4):
            self.assertEqual(detector.update([far], now, (640, 480)), [])
        self.assertEqual(detector.update([far], .5, (640, 480))[0]['type'], 'hatch_opened')

    def test_initially_closed_door_does_not_finish_inspection(self):
        inspection = HatchLoadingInspection(monitor())
        for now in (0., .1, .2):
            inspection.observe_handle([handle('closed')], now, (640, 480))
        self.assertIsNone(inspection.started_at)
        self.assertIsNone(inspection.closed_at)
        self.assertEqual(inspection.snapshot()['status'], 'waiting_open')

    def test_conflict_short_glitch_and_low_confidence_do_not_finish(self):
        inspection = HatchLoadingInspection(monitor())
        for now in (0., .1, .2):
            inspection.observe_handle([handle('open')], now, (640, 480))
        inspection.observe_handle([handle('closed')], .3, (640, 480))
        inspection.observe_handle([handle('closed'), handle('open')], .4, (640, 480))
        inspection.observe_handle([handle('closed', .1)], .5, (640, 480))
        self.assertTrue(inspection.active)
        for now in (.6, .7, .8):
            inspection.observe_handle([handle('closed')], now, (640, 480))
        self.assertEqual(inspection.closed_at, .8)

    def test_sparse_detections_cannot_accumulate_across_long_gaps(self):
        detector = monitor()
        for now in (0., 1., 2.):
            self.assertEqual(detector.update([handle('open')], now, (640, 480)), [])
        self.assertEqual(detector.state, 'unknown')

    def test_zones_are_normalized_and_overlapping_zones_rejected(self):
        detector = monitor()
        obj = handle('open')
        obj['xyxy'] = [v*2 for v in obj['xyxy']]
        for now in (0., .1, .2):
            detector.update([obj], now, (1280, 960))
        self.assertEqual(detector.state, 'open')
        with self.assertRaises(ValueError):
            HatchHandleMonitor(predict.HATCH_OPEN_POLYGON, predict.HATCH_OPEN_POLYGON)


class FullnessTests(unittest.TestCase):
    def make_sequence(self):
        tracker = SlotGroupTracker(POINTS, phase='load', confirm_seconds=.1, minimum_frames=2)
        sequence = RoiGroupSequence(tracker, ROI)
        tracker.update(boxes(filled=range(4)), 0.)
        tracker.update(boxes(filled=range(4)), .1)
        return sequence

    def test_occupancy_does_not_require_observing_an_empty_to_filled_event(self):
        sequence = self.make_sequence()
        self.assertEqual(sequence.tracker.events, [])
        report = build_loading_fullness_report(sequence, started=True)
        self.assertEqual(report['confirmed_filled_count'], 4)
        self.assertTrue(report['groups'][0]['all_filled'])
        self.assertFalse(report['all_filled'])

    def test_archives_and_live_group_are_counted_once_and_unload_is_excluded(self):
        sequence = self.make_sequence()
        snap = sequence.snapshot()
        sequence.closed_groups = [{'phase':'load', 'group_id':i, 'snapshot':deepcopy(snap)} for i in range(1, 6)]
        sequence.closed_groups.append({'phase':'unload', 'group_id':1, 'snapshot':deepcopy(snap)})
        report = build_loading_fullness_report(sequence, started=True, finalized=True)
        self.assertEqual(report['confirmed_filled_count'], 20)
        self.assertTrue(report['all_filled'])
        self.assertTrue(report['finalized'])
        sequence.closed_groups[4]['snapshot']['slots'][3]['last_confirmed_state'] = 'empty'
        report = build_loading_fullness_report(sequence, started=True, finalized=True)
        self.assertEqual(report['confirmed_filled_count'], 19)
        self.assertEqual(report['result'], '未放满')
        self.assertEqual(report['groups'][4]['empty_slot_ids'], [4])

    def test_provisional_and_unseen_identities_do_not_count(self):
        sequence = self.make_sequence()
        snapshot = sequence.snapshot()
        snapshot['slot_numbering_status'] = 'provisional'
        sequence.closed_groups = [{'phase':'load','group_id':1,'snapshot':snapshot}]
        report = build_loading_fullness_report(sequence, started=True)
        self.assertEqual(report['confirmed_filled_count'], 0)
        self.assertEqual(report['groups'][0]['unknown_slot_ids'], [1,2,3,4])
        self.assertFalse(report['groups'][1]['observed'])
        self.assertEqual(build_loading_fullness_report(sequence, started=False)['confirmed_filled_count'], 0)

    def test_removal_updates_fullness_and_final_event_is_emitted_once(self):
        sequence = self.make_sequence()
        inspection = HatchLoadingInspection(monitor())
        inspection.started_at = 0.
        first = inspection.update_fullness(sequence, .2)
        self.assertEqual(first[0]['result'], '已放满')
        for now in (.3, .4):
            sequence.tracker.update(boxes(filled=(0,1,2)), now)
        self.assertEqual(inspection.update_fullness(sequence, .4)[0]['result'], '未放满')
        inspection.closed_at = .5
        final = inspection.update_fullness(sequence, .5)
        self.assertEqual(sum(e['type']=='loading_fullness_final' for e in final), 1)
        self.assertEqual(inspection.report['confirmed_filled_count'], 3)
        self.assertEqual(inspection.update_fullness(sequence, .6), [])


class HatchPipelineTests(unittest.TestCase):
    def run_video(self, frames, root, **configuration):
        source = root/'source.mp4'
        writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,480))
        self.assertTrue(writer.isOpened())
        for _ in frames:
            writer.write(np.zeros((480,640,3),dtype=np.uint8))
        writer.release()
        args = argparse.Namespace(source=source,weights=root/'model.pt',output_dir=root/'results',
            conf=.35,imgsz=640,device='cpu',stride=1,show=False,max_frames=0,nms_iou=.7)
        configuration = {'HATCH_CONTROL_ENABLED':True,'HATCH_CONFIRM_SECONDS':.1,
            'HATCH_MIN_CONFIRM_FRAMES':2,'LOCAL_REDETECT_ENABLED':False,
            'CONFIRM_SECONDS':.1,'MIN_CONFIRM_FRAMES':2,**configuration}
        from contextlib import ExitStack
        logs, ui = [], []
        with ExitStack() as stack:
            stack.enter_context(patch.object(predict.base,'load_detection_model',return_value=(SyntheticModel(frames),'synthetic')))
            for name,value in configuration.items():
                stack.enter_context(patch.object(predict,name,value))
            stack.enter_context(patch('sys.stdout',new=io.StringIO()))
            summary = predict.run_prediction(args,roi_polygon={'polygon':ROI/[640,480],'size':(640,480),'start_frame':0},
                on_log=logs.append,on_frame=lambda image,info:ui.append(info))
        return summary,logs,ui

    def test_full_unload_load_cycle_starts_at_open_and_finishes_at_close_with_twenty_slots(self):
        frames = [boxes(filled=range(4))+[handle('closed')]]*5
        frames += [boxes(filled=range(4))+[handle('open')]]*5
        frames += [objects+[handle('open')] for phase in ('unload','load') for _ in range(5) for objects in group_frames(phase)]
        closing = len(frames)
        frames += [boxes()+[handle('closed')]]*5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary,logs,ui = self.run_video(frames,root)
            self.assertEqual(summary['status'],'hatch_closed')
            self.assertEqual(summary['processed_frames'],closing+2)
            self.assertEqual(len(summary['closed_groups']),10)
            self.assertTrue(all(info['inspection']['status']=='waiting_open' for info in ui[:6]))
            self.assertEqual(ui[0]['slots'][0]['last_confirmed_state'],'unknown')
            report = summary['loading_fullness_report']
            self.assertTrue(report['finalized'])
            self.assertEqual(report['result'],'已放满')
            self.assertEqual(report['confirmed_filled_count'],20)
            self.assertIn('上料检测结束：已放满（20/20）',logs)
            self.assertIn('上料检测结束：已放满（20/20）',ui[-1]['status'])
            for group_id in range(1,6):
                self.assertIn(f'游星片{group_id}：已放满（4/4）',logs)
            saved = json.loads((root/'results'/'loading_fullness_report.json').read_text(encoding='utf-8'))
            self.assertEqual(saved,report)
            events = [json.loads(line) for line in (root/'results'/'inspection_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(events,summary['inspection_events'])
            self.assertEqual(sum(e['type']=='loading_fullness_final' for e in events),1)

    def test_early_close_reports_partial_current_group_and_unseen_groups(self):
        frames = [boxes(filled=(0,1,2))+[handle('open')]]*10
        frames += [boxes(filled=range(4))+[handle('closed')]]*5
        with tempfile.TemporaryDirectory() as directory:
            summary,logs,ui = self.run_video(frames,Path(directory),PROCESS_PHASE='load',AUTO_LOAD_AFTER_UNLOAD=False)
            report = summary['loading_fullness_report']
            self.assertEqual(summary['status'],'hatch_closed')
            self.assertEqual(report['confirmed_filled_count'],3)
            self.assertEqual(report['groups'][0]['empty_slot_ids'],[4])
            self.assertEqual(report['result'],'未放满')
            self.assertIn('上料检测结束：未放满（3/20）',logs)

    def test_no_open_signal_never_starts_or_reports_a_final_result(self):
        frames = [boxes(filled=range(4))+[handle('closed')]]*10
        with tempfile.TemporaryDirectory() as directory:
            summary,logs,ui = self.run_video(frames,Path(directory))
            self.assertEqual(summary['inspection_processed_frames'],0)
            self.assertEqual(summary['events'],[])
            self.assertEqual(summary['inspection']['status'],'waiting_open')
            self.assertFalse(summary['loading_fullness_report']['finalized'])
            self.assertFalse(summary['group_process_report']['groups'][0]['observed'])
            self.assertFalse(any(line.startswith('上料检测结束') for line in logs))

    def test_video_ending_with_open_door_leaves_final_result_pending(self):
        frames = [boxes(filled=range(4))+[handle('open')]]*6+[boxes(filled=range(4))]*10
        with tempfile.TemporaryDirectory() as directory:
            summary,logs,ui = self.run_video(frames,Path(directory),PROCESS_PHASE='load')
            self.assertEqual(summary['status'],'end_of_stream')
            self.assertFalse(summary['loading_fullness_report']['finalized'])
            self.assertEqual(summary['loading_fullness_report']['confirmed_filled_count'],4)
            self.assertFalse(any(line.startswith('上料检测结束') for line in logs))
            self.assertNotIn('游星片1：上料未完成',logs)


if __name__ == '__main__':
    unittest.main()
