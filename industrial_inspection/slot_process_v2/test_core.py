"""Run with python -m unittest industrial_inspection.slot_process_v2.test_core -v."""

import unittest
from copy import deepcopy

from .core import ProcessConfig, ProcessPipeline, normalize
from .tracking import CarrierTracker


def group(dx=0, track_id=1):
    return {
        "class_name": "slot_group",
        "class_id": 5,
        "confidence": 0.9,
        "xyxy": [100 + dx, 100, 320 + dx, 240],
        "track_id": track_id,
    }


def slots(dx=0, state="wafer_slot_filled"):
    return [
        {"class_name": state, "confidence": 0.9, "xyxy": [100 + x + dx, 100 + y, 125 + x + dx, 125 + y]}
        for x, y in [(30, 20), (150, 20), (150, 100), (30, 100)]
    ]


def config(**kw):
    return ProcessConfig(
        roi=((0, 0), (1, 0), (1, 1), (0, 1)),
        exit_line=((0, 0), (1, 0)),
        entry_depth_ratio=0,
        stationary_seconds=0.1,
        **kw,
    )


class PipelineTests(unittest.TestCase):
    def ready(self, cfg=None):
        p = ProcessPipeline((640, 480), cfg or config())
        for t in [0, 0.1]:
            p.process(slots() + [group()], t)
        self.assertIsNotNone(p.current)
        return p

    def test_group_box_motion_does_not_switch(self):
        p = self.ready()
        for t, dx in [(0.2, 10), (0.3, 20), (0.4, 0)]:
            p.process(slots() + [group(dx)], t)
        self.assertEqual(p.closed_groups, [])

    def test_manual_rotation_requires_two_frames_and_freezes(self):
        p = self.ready()
        p.process(slots(10) + [group()], 0.2)
        self.assertFalse(p.closed_groups)
        p.process(slots(20) + [group()], 0.3)
        self.assertFalse(p.closed_groups)
        r = p.process(slots(28) + [group()], 0.4)
        self.assertEqual(p.closed_groups[0]["close_reason"], "manual_rotation")
        self.assertEqual(r["pending_group_id"], 2)
        old = deepcopy(p.closed_groups)
        for t in [0.5, 0.6, 0.7]:
            p.process(slots() + [group()], t)
        self.assertEqual(p.closed_groups, old)
        self.assertIsNone(p.current)

    def test_subthreshold_motion_cannot_switch(self):
        p = self.ready()
        for t in [0.2, 0.3, 0.4]:
            p.process(slots(13) + [group()], t)
        self.assertFalse(p.closed_groups)

    def test_two_real_slot_centers_can_confirm_rotation(self):
        p = self.ready()
        for t, dx in [(0.2, 10), (0.3, 20), (0.4, 28)]:
            ds = slots()
            for i in [0, 1]:
                x1, y1, x2, y2 = ds[i]["xyxy"]
                ds[i]["xyxy"] = [x1 + dx, y1, x2 + dx, y2]
            r = p.process(ds + [group()], t)
            if t < 0.4:
                self.assertFalse(p.closed_groups)
        self.assertEqual(p.closed_groups[0]["close_reason"], "manual_rotation")
        self.assertEqual(r["pending_group_id"], 2)

    def test_single_center_jitter_cannot_switch(self):
        p = self.ready()
        ds = slots()
        x1, y1, x2, y2 = ds[0]["xyxy"]
        ds[0]["xyxy"] = [x1 + 20, y1, x2 + 20, y2]
        p.process(ds + [group()], 0.2)
        p.process(ds + [group()], 0.3)
        self.assertFalse(p.closed_groups)

    def test_yolo_id_can_keep_carrier_track_through_box_reposition(self):
        p = self.ready()
        p.process(slots() + [group(dx=120)], 0.2)
        self.assertEqual(len(p.carriers), 1)
        self.assertEqual(p.current.cid, 1)
        self.assertFalse(p.closed_groups)

    def test_baseline_is_never_filled_from_a_later_moving_frame(self):
        p = ProcessPipeline((640, 480), config())
        for t in [0, 0.1]:
            p.process(slots()[:2] + [group()], t)
        before = {s.tid: s.baseline.copy() if s.baseline is not None else None for s in p.current.slots}
        p.process(slots() + [group()], 0.2)
        after = {s.tid: s.baseline for s in p.current.slots}
        for tid, baseline in before.items():
            if baseline is None:
                self.assertIsNone(after[tid])
            else:
                self.assertTrue((baseline == after[tid]).all())
        self.assertTrue(all(after[tid] is None for tid in after.keys() - before.keys()))

    def test_missing_breaks_two_frame_movement_confirmation(self):
        p = self.ready()
        p.process(slots(20) + [group()], 0.2)
        p.process([group()], 0.3)
        p.process(slots(20) + [group()], 0.4)
        self.assertFalse(p.closed_groups)
        p.process(slots(28) + [group()], 0.5)
        self.assertEqual(len(p.closed_groups), 1)

    def test_missing_and_carrier_missing_remain_unknown(self):
        p = self.ready()
        p.process([group()], 0.2)
        self.assertTrue(all(s.state == "unknown" for s in p.current.slots))
        p.process(slots(), 0.3)
        self.assertTrue(all(s.reason == "carrier_box_not_observed" for s in p.current.slots))
        self.assertFalse(p.closed_groups)

    def test_conflicting_boxes_and_hand_are_unknown(self):
        p = self.ready()
        ds = slots() + [dict(slots()[0], class_name="wafer_slot_empty")]
        p.process(ds + [group()], 0.2)
        self.assertTrue(any(s.reason == "empty_filled_conflict" and s.state == "unknown" for s in p.current.slots))
        hand = {"class_name": "hand", "confidence": 0.9, "xyxy": [120, 110, 170, 160]}
        p.process(slots() + [hand, group()], 0.3)
        self.assertTrue(any(s.reason == "hand_occlusion" for s in p.current.slots))

    def test_suction_fragments_are_unknown(self):
        p = self.ready()
        fragment = deepcopy(slots()[0])
        fragment["xyxy"][2] -= 10
        p.process(slots() + [fragment, group()], 0.2)
        self.assertTrue(
            any(s.reason == "duplicate_or_suction_fragment" and s.state == "unknown" for s in p.current.slots)
        )

    def test_state_confirm_and_initial_state_is_not_action(self):
        p = self.ready()
        self.assertFalse(any(e["type"] == "slot_state_change" for e in p.events))
        for t in [0.2, 0.3]:
            p.process(slots(state="wafer_slot_empty") + [group()], t)
        changes = [e for e in p.events if e["type"] == "slot_state_change"]
        self.assertEqual(len(changes), 4)
        self.assertTrue(all(e["action"] == "unload" for e in changes))

    def test_finite_exit_uses_real_measurement(self):
        cfg = config()
        cfg.exit_line = ((140 / 640, 125 / 480), (150 / 640, 125 / 480))
        p = self.ready(cfg)
        ds = slots()
        ds[0]["xyxy"][1] -= 12
        ds[0]["xyxy"][3] -= 12
        p.process(ds + [group()], 0.2)
        self.assertEqual(p.closed_groups[0]["close_reason"], "slot_exit_line")
        cfg = config()
        cfg.exit_line = ((250 / 640, 125 / 480), (270 / 640, 125 / 480))
        p = self.ready(cfg)
        p.process(ds + [group()], 0.2)
        self.assertFalse(p.closed_groups)

    def test_default_rest_interval_is_point_six_seconds(self):
        cfg = config()
        cfg.stationary_seconds = 0.6
        p = ProcessPipeline((640, 480), cfg)
        for i in range(6):
            p.process(slots() + [group()], i * 0.1)
        self.assertIsNone(p.current)
        p.process(slots() + [group()], 0.6)
        self.assertIsNotNone(p.current)
        self.assertAlmostEqual(p.current.baseline_frozen_at, 0.6)

    def test_extra_candidates_are_not_arbitrarily_numbered(self):
        p = ProcessPipeline((640, 480), config())
        extra = {"class_name": "wafer_slot_empty", "confidence": 0.99, "xyxy": [195, 155, 220, 180]}
        for t in [0, 0.1, 0.2]:
            p.process(slots() + [extra, group()], t)
        self.assertIsNone(p.current)

    def test_two_points_do_not_invent_remaining_slots(self):
        p = ProcessPipeline((640, 480), config())
        for t in [0, 0.1]:
            p.process(slots()[:2] + [group()], t)
        snap = p.current.snapshot()
        self.assertEqual(snap["numbering_status"], "provisional")
        self.assertEqual(sum(s["reason"] == "identity_not_observed" for s in snap["slots"]), 2)

    def test_phase_wrap_and_g5_unknown_never_create_g6(self):
        p = ProcessPipeline((640, 480), config())
        t = 0
        for n in range(10):
            for dt in [0, 0.1]:
                p.process(slots() + [group(track_id=n + 1)], t + dt)
            self.assertIsNotNone(p.current)
            self.assertEqual(p.current.gid, n % 5 + 1)
            p._close(t + 0.15, {"reason": "synthetic_measured_exit"})
            if n == 4:
                self.assertEqual((p.phase, p.next_group), ("load", 1))
            t += 4
        r = p.process(slots() + [group(track_id=20)], t)
        self.assertTrue(r["finished"])
        self.assertIsNone(r["pending_group_id"])
        self.assertEqual(len(p.closed_groups), 10)
        self.assertTrue(all(g["group_id"] <= 5 for g in p.closed_groups))
        self.assertEqual(p.finalize()["status"], "ended_unverified")

    def test_circle_check_is_advisory(self):
        p = self.ready(config(circle_check_enabled=True))
        self.assertTrue(p.current.numbered)
        self.assertFalse(p.current.circle_check["affects_identity_or_switch"])

    def test_non_monotonic_cache_is_rejected(self):
        p = self.ready()
        with self.assertRaises(ValueError):
            p.process([], 0)

    def test_builtin_tracker_only_sees_group_and_keeps_detector_box(self):
        fps, buffer_seconds = 20.0, 3.0
        tracker = CarrierTracker(fps=fps, buffer_seconds=buffer_seconds)
        self.assertEqual(tracker.tracker.max_time_lost, int(fps * buffer_seconds))
        ds = slots() + [group()]
        first = tracker.update(ds)
        second = tracker.update(ds)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["class_name"], "slot_group")
        self.assertEqual(second[0]["xyxy"], group()["xyxy"])
        self.assertIn("track_id", second[0])
        tracker.update([])

    def test_bytetrack_low_score_association_keeps_existing_group_id(self):
        tracker = CarrierTracker(fps=20, high_thresh=0.35, low_thresh=0.10)
        high = group()
        high["confidence"] = 0.9
        first = tracker.update([high])
        self.assertEqual(len(first), 1)
        track_id = first[0]["track_id"]

        low = group(dx=4)
        low["confidence"] = 0.2
        second = tracker.update([low, *slots()])
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["track_id"], track_id)
        self.assertEqual(second[0]["confidence"], 0.2)

    def test_actual_exits_wrap_all_ten_groups_and_auto_finish_g5(self):
        p = ProcessPipeline((640, 480), config())
        t = 0.0
        for n in range(10):
            state = "wafer_slot_filled" if n < 5 else "wafer_slot_empty"
            for dt in (0, 0.1):
                p.process(slots(state=state) + [group(track_id=n + 1)], t + dt)
            self.assertEqual((p.current.phase, p.current.gid), ("unload" if n < 5 else "load", n % 5 + 1))
            target = "wafer_slot_empty" if n < 5 else "wafer_slot_filled"
            for dt in (0.2, 0.3, 0.4, 0.5):
                p.process(slots(state=target) + [group(track_id=n + 1)], t + dt)
            if n < 9:
                p.process(slots(10, target) + [group(track_id=n + 1)], t + 0.6)
                p.process(slots(20, target) + [group(track_id=n + 1)], t + 0.7)
                p.process(slots(28, target) + [group(track_id=n + 1)], t + 0.8)
                self.assertEqual(len(p.closed_groups), n + 1)
            t += 4.0
        result = p.process(slots() + [group(track_id=99)], t)
        self.assertTrue(result["finished"])
        self.assertIsNone(result["pending_group_id"])
        self.assertEqual(p.closed_groups[-1]["close_reason"], "g5_all_filled_confirmed")
        self.assertEqual(len(p.closed_groups), 10)
        self.assertEqual(len([e for e in p.events if e["type"] == "phase_changed"]), 1)

    def test_old_tail_rejects_class_flicker_without_changing_frozen_group(self):
        p = self.ready()
        p.process(slots(10) + [group()], 0.2)
        p.process(slots(20) + [group()], 0.3)
        p.process(slots(28) + [group()], 0.4)
        old = deepcopy(p.closed_groups)
        ds = slots(28, "wafer_slot_empty")
        p.process(ds + [group()], 0.5)
        cleaned, issues = p._clean_candidates(normalize(ds, 0.35), [], 0.5)
        self.assertEqual(cleaned, [])
        self.assertTrue(all(x["reason"] == "old_group_tail" for x in issues))
        self.assertEqual(p.closed_groups, old)

    def test_stationary_center_bias_is_not_late_rotation(self):
        p = self.ready()
        ds = slots(20)
        # Class change breaks motion support; the static offset then has no
        # recent measured travel and cannot close a group much later.
        for d in ds:
            d["class_name"] = "wafer_slot_empty"
        for i in range(2, 12):
            p.process(ds + [group()], i / 10)
        self.assertFalse(p.closed_groups)

    def test_one_continuous_measured_slot_can_confirm_rotation(self):
        p = self.ready()
        for t, dx in [(0.2, 10), (0.3, 20), (0.4, 28)]:
            ds = slots()
            ds[0]["xyxy"][0] += dx
            ds[0]["xyxy"][2] += dx
            p.process(ds + [group()], t)
            if t < 0.4:
                self.assertFalse(p.closed_groups)
        self.assertEqual(p.closed_groups[0]["close_reason"], "manual_rotation")

    def test_class_change_or_fragment_cannot_supply_exit_evidence(self):
        cfg = config()
        cfg.exit_line = ((140 / 640, 125 / 480), (150 / 640, 125 / 480))
        p = self.ready(cfg)
        ds = slots()
        ds[0]["xyxy"][1] -= 20
        ds[0]["xyxy"][3] -= 20
        ds[0]["class_name"] = "wafer_slot_empty"
        p.process(ds + [group()], 0.2)
        self.assertFalse(p.closed_groups)


if __name__ == "__main__":
    unittest.main()
