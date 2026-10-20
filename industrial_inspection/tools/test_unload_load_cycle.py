"""五组下料自动衔接五组上料，复用空槽证据但不依赖手接触。"""
from __future__ import annotations

import argparse
import csv
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
from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.test_group_prediction_pipeline import SyntheticModel
from industrial_inspection.tools.test_roi_group_sequence import POINTS, ROI, boxes


def group_frames(phase, physical_order=(2, 0, 1, 3)):
    filled = set(range(4)) if phase == "unload" else set()
    frames = [boxes(filled=filled)] * 12
    for slot in physical_order:
        if phase == "unload":
            filled.remove(slot)
        else:
            filled.add(slot)
        frames.extend([boxes(filled=filled)] * 3)
    frames.extend(boxes(POINTS + [delta, 0], filled=filled) for delta in range(10, 281, 10))
    return frames


class CycleSequenceTests(unittest.TestCase):
    def setUp(self):
        self.now = -.1

    def make(self, group_id=1, phase="unload", exit_slots=1):
        tracker = SlotGroupTracker(group_id=group_id, phase=phase,
            confirm_seconds=.1, minimum_frames=2, initialization_seconds=.1, initialization_frames=2)
        return RoiGroupSequence(tracker, ROI, auto_load_after_unload=True,
            exit_min_slots=exit_slots, partial_exit_slots=min(2, exit_slots),
            exit_seconds=.2, entry_seconds=.3)

    def step(self, sequence, objects):
        self.now = round(self.now + .1, 6)
        return sequence.update(objects, self.now)

    def feed(self, sequence, frames):
        results = [self.step(sequence, objects) for objects in frames]
        return results

    def test_full_cycle_preserves_all_ten_groups_and_stops_after_load_g5(self):
        sequence = self.make()
        results = []
        for phase in ("unload", "load"):
            for _ in range(5):
                results.extend(self.feed(sequence, group_frames(phase)))
        self.assertEqual(sequence.state, "sequence_complete")
        keys = [(g["phase"], g["group_id"]) for g in sequence.closed_groups]
        self.assertEqual(keys, [(p, g) for p in ("unload", "load") for g in range(1, 6)])
        for group in sequence.closed_groups:
            self.assertEqual(group["operation_result"], "pass")
            self.assertEqual(group["snapshot"]["confirmed_event_sequence"], [2, 1, 3, 4])
            self.assertEqual([e["action"] for e in group["events"]], [group["phase"]] * 4)
        phase_events = [e for r in results for e in r["new_group_events"] if e["type"] == "phase_changed"]
        self.assertEqual(len(phase_events), 1)
        self.assertEqual((phase_events[0]["from"], phase_events[0]["to"]), ("unload", "load"))
        self.assertTrue(all(1 <= r["group_id"] <= 5 for r in results))
        self.feed(sequence, group_frames("unload"))
        self.assertEqual((sequence.tracker.phase, sequence.tracker.group_id), ("load", 5))
        self.assertEqual(len(sequence.closed_groups), 10)
        self.assertEqual(len(sequence.all_events), 40)

    def test_incomplete_unload_does_not_block_load_g1(self):
        sequence = self.make(group_id=5)
        results = self.feed(sequence, group_frames("unload", physical_order=()))
        self.assertEqual((sequence.tracker.phase, sequence.tracker.group_id), ("load", 1))
        self.assertFalse(predict.process_completed(sequence.closed_groups[0]["snapshot"]))
        self.assertEqual(sequence.closed_groups[0]["operation_result"], "pending")
        self.assertTrue(any(e["type"] == "phase_changed" for r in results for e in r["new_group_events"]))
        # G5的满槽不能作为G1的空槽证据；直接看到G1已满不虚构上料。
        self.feed(sequence, [boxes(filled=range(4))] * 6)
        self.assertEqual(sequence.tracker.events, [])

    def test_multi_point_switch_exposes_pending_load_phase_and_g1(self):
        sequence = self.make(group_id=5, exit_slots=3)
        results = self.feed(sequence, group_frames("unload"))
        waiting = [r for r in results if r["pending_group_id"] is not None]
        self.assertTrue(waiting)
        self.assertTrue(all((r["pending_phase"], r["pending_group_id"]) == ("load", 1) for r in waiting))
        result = self.feed(sequence, [boxes()] * 12)[-1]
        self.assertEqual((result["phase"], result["group_id"], result["group_status"]), ("load", 1, "active"))

    def test_counterclockwise_loading_fails(self):
        sequence = self.make(phase="load")
        self.feed(sequence, group_frames("load", physical_order=(2, 3, 1, 0)))
        self.assertEqual(sequence.closed_groups[0]["snapshot"]["confirmed_event_sequence"], [2, 4, 3, 1])
        self.assertEqual(sequence.closed_groups[0]["operation_result"], "fail")

    def test_removal_during_loading_is_a_reverse_action(self):
        sequence = self.make(phase="load")
        self.feed(sequence, [boxes()] * 3 + [boxes(filled=(0,))] * 3 + [boxes()] * 3)
        self.assertEqual([e["action"] for e in sequence.all_events], ["load", "unload"])
        self.assertFalse(sequence.all_events[-1]["expected_action"])
        self.assertEqual(sequence.tracker.order_result(), "fail")

    def test_rotation_confirms_s4_for_old_group_without_hand_detection(self):
        roi = np.array([[80, 80], [330, 80], [330, 330], [80, 330]], dtype=np.float32)
        sequence = self.make(phase="load")
        sequence.contour = roi
        self.feed(sequence, [boxes()] * 3)
        filled = set()
        for slot in (2, 0, 1):  # 按x、y初始化后对应正式槽2、1、3。
            filled.add(slot)
            self.feed(sequence, [boxes(filled=filled)] * 3)
        self.feed(sequence, [boxes(visible=(0, 1, 2), filled=filled)] * 3)
        moved = POINTS.copy()
        moved[0] = [75, 100]
        self.step(sequence, boxes(moved, visible=(0, 1, 2), filled=filled))
        self.assertEqual(sequence.tracker.group_id, 2)
        self.feed(sequence, [boxes(moved, filled=range(4))] * 3)
        group = sequence.closed_groups[0]
        self.assertEqual(group["snapshot"]["confirmed_event_sequence"], [2, 1, 3, 4])
        self.assertEqual(group["operation_result"], "pass")
        self.assertTrue(all(e["group_id"] == 1 for e in sequence.all_events))
        self.assertEqual(sequence.tracker.events, [])

    def previous_unload(self):
        sequence = self.make()
        self.feed(sequence, [boxes(filled=range(4))] * 3)
        self.feed(sequence, [boxes()] * 3)
        snapshot = sequence.tracker.snapshot()
        sequence.closed_groups.append({"phase": "unload", "group_id": 1, "snapshot": snapshot})
        sequence.tracker = sequence._new_tracker(1, phase="load")
        sequence.tracker.reset_group(1, phase="load", centers=POINTS)
        return sequence

    def test_restore_only_confirmed_empty_and_leave_unseen_current_state_unknown(self):
        sequence = self.previous_unload()
        old = sequence.closed_groups[0]["snapshot"]
        old["slots"][1]["last_confirmed_state"] = "unknown"
        old["slots"][2]["last_confirmed_state"] = "filled"
        before = deepcopy(sequence.closed_groups)
        result = sequence._restore_load_baseline({**sequence.tracker.snapshot(), "new_events": []}, self.now)
        self.assertEqual([s["last_confirmed_state"] for s in result["slots"]], ["empty", "unknown", "unknown", "empty"])
        self.assertTrue(all(s["state"] == "unknown" for s in result["slots"]))
        self.assertEqual(result["new_events"], [])
        self.assertEqual(sequence.closed_groups, before)

    def test_early_filled_uses_real_unload_baseline_once_and_order_stays_uncertain(self):
        sequence = self.previous_unload()
        before = deepcopy(sequence.closed_groups)
        result = self.feed(sequence, [boxes(filled=range(4))] * 3)[-1]
        self.assertEqual(sequence.tracker.events, [])
        result = sequence._restore_load_baseline(result, self.now)
        self.assertEqual(len(result["new_events"]), 4)
        self.assertTrue(all(e["reason"] == "detected_after_unload_baseline" for e in result["new_events"]))
        self.assertTrue(all(e["phase"] == "load" and e["action"] == "load" for e in result["new_events"]))
        self.assertTrue(all(e["time_lower"] < e["time_upper"] for e in result["new_events"]))
        self.assertEqual(result["order_result"], "pending")
        sequence._restore_load_baseline({**sequence.tracker.snapshot(), "new_events": []}, self.now)
        self.assertEqual(len(sequence.tracker.events), 4)
        self.assertEqual(sequence.closed_groups, before)

    def test_unknown_unload_baseline_cannot_turn_initial_filled_into_load(self):
        sequence = self.previous_unload()
        for slot in sequence.closed_groups[0]["snapshot"]["slots"]:
            slot["last_confirmed_state"], slot["last_confirmed_at"] = "unknown", None
        result = self.feed(sequence, [boxes(filled=range(4))] * 3)[-1]
        result = sequence._restore_load_baseline(result, self.now)
        self.assertEqual(sequence.tracker.events, [])
        self.assertEqual(result["new_events"], [])

    def test_restored_load_uses_first_filled_evidence_after_occlusion_and_reconfirmation(self):
        sequence = self.previous_unload()
        tracker = sequence.tracker
        # 本轮首次看见时已经满，未编号阶段无法立即与下料基线连接。
        for now in (1., 1.1):
            tracker.update(boxes(filled=(0,)), now)
        first_confirmation = tracker.slots[0].evidence_time
        tracker.update(boxes(visible=(1, 2, 3)), 1.2)
        for now in (1.3, 1.4):
            result = tracker.update(boxes(filled=(0,)), now)
        result = sequence._restore_load_baseline(result, 1.4)
        event = next(e for e in result['new_events'] if e['slot_id'] == 1)
        self.assertEqual(event['time_upper'], 1.)
        self.assertEqual(event['confirmed_at'], first_confirmation)
        self.assertEqual(event['reported_at'], 1.4)
        self.assertLessEqual(event['time_upper'], event['confirmed_at'])

    def test_incremental_entry_recovers_early_load_without_duplicating_unload_events(self):
        sequence = self.make()
        for _ in range(5):
            self.feed(sequence, group_frames("unload"))
        before = deepcopy(sequence.closed_groups)
        # T1在其他三个槽出现前已连续检测为满，正式编号后应补报上料。
        results = self.feed(sequence, [boxes(visible=(0,), filled=(0,))] * 3
                            + [boxes(filled=range(4))] * 5)
        load_events = [e for e in sequence.all_events if e["phase"] == "load"]
        self.assertEqual(len(load_events), 4)
        early = [e for e in load_events if e.get("reason") == "detected_after_unload_baseline"]
        # 停稳后才正式编号，此时四槽都可能已在T阶段确认，均须一次性补报。
        self.assertEqual([e["slot_id"] for e in early], [1, 2, 3, 4])
        self.assertTrue(all(e["recovered_before_initialization"] for e in early))
        self.assertLess(early[0]["time_upper"], min(e["time_upper"] for e in early[1:]))
        self.assertEqual(sequence.tracker.order_result(), "pending")
        self.assertEqual(sequence.closed_groups, before)
        self.assertEqual(len([e for r in results for e in r["new_events"]]), 4)

    def test_hand_contact_in_auto_load_still_cannot_confirm_s4(self):
        sequence = self.make()
        for _ in range(5):
            self.feed(sequence, group_frames("unload"))
        self.feed(sequence, [boxes()] * 4)
        hand = {"class_name": "hand", "confidence": .95, "xyxy": [185, 185, 215, 215]}
        self.feed(sequence, [boxes(visible=(0, 1, 2)) + [hand]] * 3)
        self.feed(sequence, [boxes(visible=(0, 1, 2))] * 4)
        sequence.finish()
        self.assertEqual(sequence.tracker.events, [])
        self.assertEqual(len(sequence.all_events), 20)
        self.assertEqual(sequence.tracker.slots[3].confirmed, "empty")

    def test_deferred_load_g5_updates_only_load_archive(self):
        sequence = self.make()
        for _ in range(5):
            self.feed(sequence, group_frames("unload"))
        unload_g5 = deepcopy(sequence.closed_groups[-1])
        for _ in range(4):
            self.feed(sequence, group_frames("load"))
        self.feed(sequence, [boxes()] * 4)
        filled = set()
        for slot in (2, 0, 1):
            filled.add(slot)
            self.feed(sequence, [boxes(filled=filled)] * 3)
        self.feed(sequence, [boxes(visible=(0, 1, 2), filled=filled)] * 3)
        moved = POINTS.copy()
        for delta in range(10, 111, 10):
            moved[1] = POINTS[1] + [delta, 0]
            self.step(sequence, boxes(moved, visible=(0, 1, 2), filled=filled))
        self.assertEqual(sequence.state, "sequence_complete")
        self.feed(sequence, [boxes(moved, filled=range(4))] * 3)
        sequence.finish()
        self.assertEqual(sequence.closed_groups[4], unload_g5)
        load_g5 = sequence.closed_groups[-1]
        self.assertEqual((load_g5["phase"], load_g5["group_id"]), ("load", 5))
        self.assertEqual(load_g5["operation_result"], "pass")
        self.assertEqual(load_g5["snapshot"]["confirmed_event_sequence"], [2, 1, 3, 4])
        self.assertEqual(len(sequence.all_events), 40)


class CyclePipelineTests(unittest.TestCase):
    def setUp(self):
        hatch_control = patch.object(predict, 'HATCH_CONTROL_ENABLED', False)
        hatch_control.start()
        self.addCleanup(hatch_control.stop)

    def test_default_pipeline_saves_distinct_reports_events_ui_and_logs_for_both_phases(self):
        self.assertTrue(predict.AUTO_LOAD_AFTER_UNLOAD)
        self.assertEqual(predict.MAX_FRAMES, 0)
        frames = [frame for phase in ("unload", "load") for _ in range(5) for frame in group_frames(phase)]
        with tempfile.TemporaryDirectory(prefix="wafer_unload_load_") as directory:
            root = Path(directory)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10., (640, 480))
            self.assertTrue(writer.isOpened())
            for _ in frames:
                writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
            writer.release()
            args = argparse.Namespace(source=source, weights=root / "model.pt", output_dir=root / "results",
                conf=.35, imgsz=640, device="cpu", stride=1, show=False, max_frames=0, nms_iou=.7)
            context = {"polygon": ROI / [640, 480], "size": (640, 480), "start_frame": 0}
            logs, ui = [], []
            with (patch.object(predict.base, "load_detection_model", return_value=(SyntheticModel(frames), "synthetic")),
                  patch.object(predict, "LOCAL_REDETECT_ENABLED", False),
                  patch.object(predict, "CONFIRM_SECONDS", .1), patch.object(predict, "MIN_CONFIRM_FRAMES", 2),
                  patch("sys.stdout", new=io.StringIO())):
                summary = predict.run_prediction(args, roi_polygon=context, on_log=logs.append,
                    on_frame=lambda image, info: ui.append(info))
            self.assertEqual(summary["processed_frames"], len(frames))
            self.assertEqual(summary["current_phase"], "load")
            self.assertEqual(summary["final_group_snapshot"]["group_status"], "sequence_complete")
            expected_logs = [f"游星片{g}：{action}{state}" for action in ("下料", "上料")
                             for g in range(1, 6) for state in ("未完成", "完成")]
            self.assertEqual(logs[:-1], expected_logs)
            self.assertEqual(summary["group_process_report"], summary["group_process_reports"]["unload"])
            for phase in ("unload", "load"):
                report = summary["group_process_reports"][phase]
                self.assertEqual(report["matrix"], [[1, 1, 1, 1]] * 5)
                self.assertTrue(all(g["sequence"] == [2, 1, 3, 4] and g["order_result"] == "pass" for g in report["groups"]))
            saved = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
            reports = json.loads((args.output_dir / "group_process_report.json").read_text(encoding="utf-8"))
            self.assertEqual(reports, saved["group_process_reports"])
            self.assertEqual(reports, summary["group_process_reports"])
            self.assertEqual((args.output_dir / "group_process_report.txt").read_text(encoding="utf-8"), logs[-1] + "\n")
            self.assertIn("槽位下料完成矩阵", logs[-1])
            self.assertIn("槽位上料完成矩阵", logs[-1])
            events = [json.loads(line) for line in (args.output_dir / "slot_events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, summary["events"])
            self.assertEqual(len(events), 40)
            self.assertEqual({(e["phase"], e["group_id"]) for e in events}, {(p, g) for p in ("unload", "load") for g in range(1, 6)})
            group_events = [json.loads(line) for line in (args.output_dir / "group_events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(e["type"] == "group_closed" for e in group_events), 10)
            self.assertTrue(all("phase" in e for e in group_events))
            with (args.output_dir / "slot_states.csv").open(encoding="utf-8-sig", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual({(row["phase"], int(row["group_id"])) for row in csv_rows}, {(p, g) for p in ("unload", "load") for g in range(1, 6)})
            self.assertEqual({(info["phase"], info["group_id"]) for info in ui}, {(p, g) for p in ("unload", "load") for g in range(1, 6)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
