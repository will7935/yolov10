"""Independent process state machine. No model, video, or legacy-pipeline imports.

C identifies a carrier trajectory; G identifies an occurrence in unload/load 1..5.
Only real, unambiguous measurements update occupancy or provide switch evidence.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

SLOT_CLASSES = {"wafer_slot_empty": "empty", "wafer_slot_filled": "filled"}


@dataclass
class ProcessConfig:
    confidence: float = 0.35
    image_size: int = 1280
    stationary_seconds: float = 0.6
    stable_jitter_px: float = 6.0
    movement_min_px: float = 14.0
    movement_spacing_ratio: float = 0.18
    movement_confirm_frames: int = 2
    maximum_center_speed_px_s: float = 120.0
    recent_motion_window_seconds: float = 0.35
    state_confirm_frames: int = 2
    state_confirm_seconds: float = 0.10
    maximum_evidence_gap: float = 0.25
    carrier_lost_seconds: float = 3.0
    slot_lost_seconds: float = 1.2
    entry_depth_ratio: float = 0.25
    exit_margin_px: float = 4.0
    slot_match_distance_px: float = 40.0
    circle_check_enabled: bool = False
    group_count: int = 5
    roi: tuple = ((0.4359375, 0.3270833), (0.98125, 0.5020833), (0.9890625, 0.80625), (0.46875, 0.66875))
    exit_line: tuple = ((0.4359375, 0.3270833333), (0.7903906, 0.4408333333))

    def __post_init__(self):
        if self.group_count != 5 or not 0 < self.confidence <= 1:
            raise ValueError("The process requires five groups and confidence in (0,1].")
        if self.movement_confirm_frames < 2 or self.stationary_seconds < 0:
            raise ValueError("Rotation needs at least two frames; stationary time must be nonnegative.")


def center(box):
    b = np.asarray(box, dtype=float)
    return (b[:2] + b[2:]) / 2


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.prod(np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    return float(inter / max(1e-9, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter))


def contains(point, box, padding=0.0):
    return bool(np.all(point >= np.asarray(box[:2]) - padding) and np.all(point <= np.asarray(box[2:]) + padding))


def point_in_polygon(point, polygon):
    # Ray casting avoids importing OpenCV into the process core.
    x, y = point
    inside = False
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def normalize(objects, confidence):
    result = []
    for obj in objects:
        box = np.asarray(obj.get("xyxy", []), dtype=float)
        score = float(obj.get("confidence", 0))
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or np.any(box[2:] <= box[:2])
            or not math.isfinite(score)
            or score < confidence
        ):
            continue
        result.append({**obj, "xyxy": box.tolist(), "center": center(box)})
    return result


@dataclass
class Carrier:
    cid: int
    box: list
    seen_at: float
    yolo_ids: set = field(default_factory=set)
    observed: bool = True
    retired: bool = False


@dataclass
class Slot:
    tid: int
    point: np.ndarray
    box: list
    last_seen: float
    support: int = 1
    sid: int | None = None
    state: str = "unknown"
    confirmed: str = "unknown"
    confirmed_at: float | None = None
    reason: str = "confirming_identity"
    observed: bool = False
    candidate: str = "unknown"
    candidate_since: float | None = None
    candidate_count: int = 0
    last_state_seen: float | None = None
    baseline: np.ndarray | None = None
    baseline_at: float | None = None
    last_detection_class: str | None = None
    motion_reliable: bool = False
    motion_history: list = field(default_factory=list)
    stationary_anchor: np.ndarray | None = None
    stationary_since: float | None = None

    def unknown(self, reason, reset_candidate=True):
        self.state, self.reason, self.observed = "unknown", reason, False
        if reset_candidate:
            self.candidate, self.candidate_count, self.candidate_since = "unknown", 0, None

    def observe(self, detection, now, config):
        state = SLOT_CLASSES[detection["class_name"]]
        if (
            self.candidate != state
            or self.last_state_seen is None
            or now - self.last_state_seen > config.maximum_evidence_gap
        ):
            self.candidate, self.candidate_since, self.candidate_count = state, now, 0
        self.candidate_count += 1
        self.last_state_seen = now
        self.state, self.reason = "unknown", "confirming"
        if (
            self.candidate_count < config.state_confirm_frames
            or now - self.candidate_since + 1e-9 < config.state_confirm_seconds
        ):
            return None
        previous, previous_at = self.confirmed, self.confirmed_at
        self.state, self.confirmed, self.reason, self.confirmed_at = state, state, "confirmed", now
        if previous in ("empty", "filled") and state != previous:
            return {
                "type": "slot_state_change",
                "temporary_slot_id": self.tid,
                "slot_id": self.sid,
                "from": previous,
                "to": state,
                "action": "load" if state == "filled" else "unload",
                "time_lower": previous_at,
                "time_upper": self.candidate_since,
                "at": now,
            }
        return None

    def snapshot(self):
        return {
            "slot_id": self.sid,
            "temporary_slot_id": self.tid,
            "center_xy": self.point.tolist(),
            "measured_this_frame": self.observed,
            "center_is_prediction": False,
            "center_is_stale": not self.observed,
            "state": self.state,
            "last_confirmed_state": self.confirmed,
            "last_confirmed_at": self.confirmed_at,
            "reason": self.reason,
            "baseline_center_xy": None if self.baseline is None else self.baseline.tolist(),
            "motion_reliable": bool(self.motion_reliable),
            "baseline_at": self.baseline_at,
        }


@dataclass
class Group:
    gid: int
    phase: str
    cid: int
    started_at: float
    slots: list = field(default_factory=list)
    numbered: bool = False
    stable_since: float | None = None
    stable_anchor: dict = field(default_factory=dict)
    baseline_frozen_at: float | None = None
    min_spacing: float | None = None
    movement_streaks: dict = field(default_factory=dict)
    target_since: float | None = None
    frozen: bool = False
    closed_at: float | None = None
    close_reason: str | None = None
    events: list = field(default_factory=list)
    circle_check: dict = field(default_factory=lambda: {"enabled": False, "status": "disabled"})
    next_slot_track: int = 1

    def snapshot(self):
        slots = [s.snapshot() for s in self.slots]
        if self.numbered:
            slots.sort(key=lambda s: s["slot_id"])
        while len(slots) < 4:
            slots.append(
                {
                    "slot_id": None,
                    "temporary_slot_id": None,
                    "center_xy": None,
                    "measured_this_frame": False,
                    "center_is_prediction": False,
                    "state": "unknown",
                    "last_confirmed_state": "unknown",
                    "last_confirmed_at": None,
                    "reason": "identity_not_observed",
                    "baseline_center_xy": None,
                }
            )
        target = "empty" if self.phase == "unload" else "filled"
        return {
            "group_id": self.gid,
            "phase": self.phase,
            "carrier_track_id": self.cid,
            "status": "frozen" if self.frozen else "active",
            "numbering_status": "final" if self.numbered else "provisional",
            "started_at": self.started_at,
            "closed_at": self.closed_at,
            "close_reason": self.close_reason,
            "baseline_frozen_at": self.baseline_frozen_at,
            "minimum_slot_spacing_px": self.min_spacing,
            "slots": slots,
            "target_verified_count": sum(s["state"] == target for s in slots) if self.numbered else 0,
            "last_confirmed_target_count": sum(s["last_confirmed_state"] == target for s in slots)
            if self.numbered
            else 0,
            "all_targets_verified": self.numbered and all(s["state"] == target for s in slots),
            "events": deepcopy(self.events),
            "geometry_check": self.circle_check,
        }


class ProcessPipeline:
    def __init__(self, frame_size=(640, 480), config=None):
        self.config = config or ProcessConfig()
        self.width, self.height = frame_size
        scale = np.asarray(frame_size)
        self.exit_p1, self.exit_p2 = np.asarray(self.config.exit_line) * scale
        self.roi = np.asarray(self.config.roi) * scale
        self.carriers = {}
        self.next_carrier = 1
        self.current = None
        self.closed_groups = []
        self.events = []
        self.phase, self.next_group = "unload", 1
        self.finished = False
        self.last_time = None
        self.entry = {}
        self.tails = []
        self.candidates = {}
        self._new_events = []

    def _emit(self, event):
        event = deepcopy(event)
        self.events.append(event)
        self._new_events.append(event)

    def _side(self, point):
        d = self.exit_p2 - self.exit_p1
        p = np.asarray(point) - self.exit_p1
        return float(d[0] * p[1] - d[1] * p[0]) / float(np.linalg.norm(d))

    def _eligible(self, carrier):
        c = center(carrier.box)
        return (
            not carrier.retired
            and carrier.observed
            and point_in_polygon(c, self.roi)
            and self._side(c) > (carrier.box[3] - carrier.box[1]) * self.config.entry_depth_ratio
        )

    def _update_carriers(self, detections, now):
        for c in self.carriers.values():
            c.observed = False
        old = [c for c in self.carriers.values() if now - c.seen_at <= self.config.carrier_lost_seconds]
        cost = np.full((len(old), len(detections) + len(old)), 1.1)
        for i, c in enumerate(old):
            diag = float(np.linalg.norm(np.asarray(c.box[2:]) - c.box[:2]))
            for j, d in enumerate(detections):
                dist = float(np.linalg.norm(center(c.box) - d["center"]))
                overlap = iou(c.box, d["xyxy"])
                same_id = d.get("track_id") is not None and int(d["track_id"]) in c.yolo_ids
                max_dist = max(45.0, diag * 0.55)
                if same_id:
                    max_dist = max(max_dist, diag * 1.25)
                if (overlap < 0.15 and not (same_id and dist <= diag * 0.9)) or dist > max_dist:
                    continue
                cost[i, j] = 0.65 * (1 - overlap) + 0.35 * dist / max(diag, 1) - (0.65 if same_id else 0)
        used = set()
        if old:
            rs, cs = linear_sum_assignment(cost)
            for i, j in zip(rs, cs):
                if j >= len(detections) or cost[i, j] >= 0.9:
                    continue
                c, d = old[i], detections[j]
                used.add(j)
                c.box, c.seen_at, c.observed = d["xyxy"], now, True
                if d.get("track_id") is not None:
                    c.yolo_ids.add(int(d["track_id"]))
        for j, d in enumerate(detections):
            if j in used:
                continue
            c = Carrier(self.next_carrier, d["xyxy"], now)
            if d.get("track_id") is not None:
                c.yolo_ids.add(int(d["track_id"]))
            self.carriers[c.cid] = c
            self.next_carrier += 1

    def _owners(self, detections):
        """Containment and normalized box distance; close competing boxes mean unknown."""
        active = [c for c in self.carriers.values() if c.observed]
        assigned = {}
        rejected = []
        for d in detections:
            scores = []
            for c in active:
                if not contains(d["center"], c.box, 4.0):
                    continue
                half = np.maximum((np.asarray(c.box[2:]) - c.box[:2]) / 2, 1)
                score = float(np.linalg.norm((d["center"] - center(c.box)) / half))
                scores.append((score, c.cid))
            scores.sort()
            if not scores:
                rejected.append({"center_xy": d["center"].tolist(), "reason": "no_observed_carrier"})
                continue
            if len(scores) > 1 and scores[1][0] - scores[0][0] < 0.18:
                rejected.append({"center_xy": d["center"].tolist(), "reason": "ambiguous_carrier"})
                continue
            assigned.setdefault(scores[0][1], []).append(d)
        return assigned, rejected

    def _clean_candidates(self, candidates, hands, now):
        # Union overlapping slot boxes: exact same-class duplicates may be collapsed;
        # conflicting/fragment boxes remain unknown instead of choosing a winning label.
        cleaned = []
        rejected = []
        for i, d in enumerate(candidates):
            reason = None
            for j, other in enumerate(candidates):
                if i == j:
                    continue
                overlap = iou(d["xyxy"], other["xyxy"])
                near = np.linalg.norm(d["center"] - other["center"]) < 18.0
                if overlap >= 0.85 and d["class_name"] == other["class_name"]:
                    if other["confidence"] > d["confidence"] or (other["confidence"] == d["confidence"] and j < i):
                        reason = "exact_duplicate"
                        break
                    continue
                if overlap > 0.38 or near:
                    reason = (
                        "empty_filled_conflict"
                        if d["class_name"] != other["class_name"]
                        else "duplicate_or_suction_fragment"
                    )
                    break
            if reason is None:
                for hand in hands:
                    overlap = np.maximum(
                        0.0, np.minimum(d["xyxy"][2:], hand["xyxy"][2:]) - np.maximum(d["xyxy"][:2], hand["xyxy"][:2])
                    )
                    area = max(1.0, np.prod(np.asarray(d["xyxy"][2:]) - d["xyxy"][:2]))
                    if contains(d["center"], hand["xyxy"], 2.0) or np.prod(overlap) / area >= 0.12:
                        reason = "hand_occlusion"
                        break
            if reason is None:
                for tail in self.tails:
                    owner = self.carriers.get(tail["cid"])
                    retained = (owner is not None
                                and now - owner.seen_at <= self.config.carrier_lost_seconds
                                and now - tail["seen_at"] <= self.config.carrier_lost_seconds)
                    if retained and np.linalg.norm(d["center"] - tail["point"]) < tail["gate"]:
                        reason = "old_group_tail"
                        break
            if reason:
                rejected.append({"center_xy": d["center"].tolist(), "reason": reason, "detection": d})
            else:
                cleaned.append(d)
        return cleaned, rejected

    def _update_slots(self, g, candidates, rejected, now):
        if not g.numbered:
            g.slots = [
                s for s in g.slots if s.baseline is not None or now - s.last_seen <= self.config.slot_lost_seconds
            ]
        previous = {s.tid: (s.point.copy(), s.last_seen) for s in g.slots}
        for s in g.slots:
            s.unknown("not_detected", reset_candidate=False)
        # Keep debouncing across consecutive measurements; a missing frame resets it below.
        gate = min(
            self.config.slot_match_distance_px,
            max((g.min_spacing or 100.0) * 0.72, self._threshold(g) * 2.0),
        )
        pairs = []
        if g.slots:
            cost = np.full((len(g.slots), len(candidates) + len(g.slots)), gate + 1.0)
            for i, s in enumerate(g.slots):
                for j, d in enumerate(candidates):
                    anchors = [s.point] + ([] if s.baseline is None else [s.baseline])
                    cost[i, j] = min(float(np.linalg.norm(p - d["center"])) for p in anchors)
            rs, cs = linear_sum_assignment(cost)
            for i, j in zip(rs, cs):
                if j >= len(candidates) or cost[i, j] > gate:
                    continue
                alternatives = [cost[k, j] for k in range(len(g.slots)) if k != i]
                alternatives += [cost[i, k] for k in range(len(candidates)) if k != j]
                if alternatives and min(alternatives) - cost[i, j] < 5.0:
                    g.slots[i].reason = "association_ambiguous"
                    continue
                pairs.append((g.slots[i], j))
        used = {j for _, j in pairs}
        # Birth only into genuinely unoccupied space and never select an arbitrary
        # highest-score four from an overflowing candidate set.
        births = [
            j
            for j, d in enumerate(candidates)
            if j not in used and all(
                np.linalg.norm(s.point - d["center"]) > gate
                and (s.baseline is None or np.linalg.norm(s.baseline - d["center"]) > gate)
                for s in g.slots
            )
        ]
        if not g.numbered and len(g.slots) + len(births) <= 4:
            for j in sorted(births, key=lambda j: tuple(candidates[j]["center"])):
                d = candidates[j]
                s = Slot(g.next_slot_track, d["center"].copy(), d["xyxy"], now, support=0)
                g.next_slot_track += 1
                g.slots.append(s)
                pairs.append((s, j))
        measured = []
        for s, j in pairs:
            d = candidates[j]
            center_interval = now - s.last_seen
            size = np.asarray(d["xyxy"][2:]) - d["xyxy"][:2]
            old_size = np.asarray(s.box[2:]) - s.box[:2]
            same_box_scale = np.all(size >= old_size * 0.55) and np.prod(size) >= np.prod(old_size) * 0.45
            center_travel = float(np.linalg.norm(d["center"] - s.point))
            shape_consistent = np.linalg.norm(size - old_size) <= max(5.0, center_travel * 0.8)
            s.motion_reliable = (
                s.support >= 1
                and center_interval <= self.config.maximum_evidence_gap
                and s.last_detection_class == d["class_name"]
                and same_box_scale
                and shape_consistent
                and center_travel <= max(6.0, center_interval * self.config.maximum_center_speed_px_s)
            )
            if s.support >= 3 and s.last_detection_class == d["class_name"] and not same_box_scale:
                s.unknown("suction_fragment_size")
                s.motion_history = []
                continue
            if not s.motion_reliable:
                s.motion_history = []
            s.motion_history.append((now, d["center"].copy()))
            s.motion_history = [(t, p) for t, p in s.motion_history
                                if now - t <= self.config.recent_motion_window_seconds]
            s.point, s.box, s.last_seen = d["center"].copy(), d["xyxy"], now
            s.last_detection_class = d["class_name"]
            s.support += 1
            s.observed = True
            measured.append(s)
            event = s.observe(d, now, self.config)
            if event and g.gid:
                event.update(
                    group_id=g.gid, phase=g.phase, carrier_track_id=g.cid, expected_action=event["action"] == g.phase
                )
                g.events.append(event)
                self._emit(event)
        for s in g.slots:
            if s.observed:
                continue
            s.motion_reliable = False
            s.motion_history = []
            for item in rejected:
                if np.linalg.norm(s.point - np.asarray(item["center_xy"])) <= gate:
                    s.reason = item["reason"]
                    break
            # A missing or ambiguous observation breaks temporal confirmation.
            s.candidate, s.candidate_count, s.candidate_since = "unknown", 0, None
        return measured, previous

    def _number_and_freeze(self, g, measured, now):
        for s in measured:
            if s.stationary_anchor is None or np.linalg.norm(s.point - s.stationary_anchor) > self.config.stable_jitter_px:
                s.stationary_anchor, s.stationary_since = s.point.copy(), now
        for s in g.slots:
            if not s.observed and now - s.last_seen > self.config.maximum_evidence_gap:
                s.stationary_anchor, s.stationary_since = None, None
        stable = [s for s in measured if s.stationary_since is not None
                  and now - s.stationary_since + 1e-9 >= self.config.stationary_seconds]
        if len(stable) < 2:
            return
        g.stable_since = max(s.stationary_since for s in stable)
        if (
            not g.numbered
            and len(g.slots) == 4
            and all(s.support >= 2 and now - s.last_seen <= self.config.maximum_evidence_gap for s in g.slots)
        ):
            points = np.asarray([s.point for s in g.slots])
            c = points.mean(axis=0)
            ring = list(np.argsort(np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])))
            start = int(np.argmin(points[:, 0] + points[:, 1]))
            k = ring.index(start)
            ring = ring[k:] + ring[:k]
            for idx, sid in zip(ring, (1, 3, 4, 2)):
                g.slots[idx].sid = sid
            g.numbered = True
            mapping = {s.tid: s.sid for s in g.slots}
            for event in g.events:
                event["slot_id"] = mapping[event["temporary_slot_id"]]
            for event in self.events:
                if (event.get("group_id"), event.get("phase"), event.get("carrier_track_id")) == (
                    g.gid, g.phase, g.cid
                ) and event.get("temporary_slot_id") in mapping:
                    event["slot_id"] = mapping[event["temporary_slot_id"]]
            g.min_spacing = min(
                float(np.linalg.norm(a.point - b.point)) for i, a in enumerate(g.slots) for b in g.slots[:i]
            )
            if g.gid:
                self._emit(
                    {
                        "type": "slots_numbered",
                        "at": now,
                        "group_id": g.gid,
                        "phase": g.phase,
                        "carrier_track_id": g.cid,
                        "temporary_to_slot": mapping,
                    }
                )
        existing = [s for s in measured if s.baseline is not None]
        can_extend = len(existing) >= 2 and all(
            np.linalg.norm(s.point - s.baseline) <= self.config.stable_jitter_px for s in existing
        )
        new_baselines = [s for s in stable if s.baseline is None]
        if new_baselines and (g.baseline_frozen_at is None or can_extend):
            if g.baseline_frozen_at is None:
                g.baseline_frozen_at = now
            for s in new_baselines:
                s.baseline, s.baseline_at = s.point.copy(), now
            baseline_slots = [s for s in g.slots if s.baseline is not None]
            g.min_spacing = min(
                float(np.linalg.norm(a.baseline - b.baseline))
                for i, a in enumerate(baseline_slots)
                for b in baseline_slots[:i]
            )
            g.circle_check = (
                self._geometry(g)
                if g.numbered
                else {"enabled": self.config.circle_check_enabled, "status": "insufficient_observations"}
            )
            if g.gid:
                self._emit(
                    {
                        "type": "slot_baseline_frozen",
                        "at": now,
                        "stationary_since": g.stable_since,
                        "group_id": g.gid,
                        "phase": g.phase,
                        "carrier_track_id": g.cid,
                        "minimum_slot_spacing_px": g.min_spacing,
                        "threshold_px": self._threshold(g),
                        "centers": {s.tid: s.baseline.tolist() for s in baseline_slots},
                    }
                )

    def _threshold(self, g):
        return max(self.config.movement_min_px, (g.min_spacing or 0) * self.config.movement_spacing_ratio)

    def _geometry(self, g):
        if not self.config.circle_check_enabled:
            return {"enabled": False, "status": "disabled"}
        p = np.asarray([s.point for s in sorted(g.slots, key=lambda s: (1, 3, 4, 2).index(s.sid))])
        a = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(4)])
        b = np.sum(p * p, axis=1)
        fit = np.linalg.lstsq(a, b, rcond=None)[0]
        c = fit[:2]
        r = np.linalg.norm(p - c, axis=1)
        angles = np.sort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi]) * 180 / np.pi
        valid = np.std(r) / max(np.mean(r), 1e-9) < 0.30 and np.max(np.abs(gaps - 90)) < 35
        return {
            "enabled": True,
            "status": "pass" if valid else "advisory_fail",
            "center_xy": c.tolist(),
            "angular_gaps_degrees": gaps.tolist(),
            "radius_cv": float(np.std(r) / max(np.mean(r), 1e-9)),
            "affects_identity_or_switch": False,
        }

    def _switch_evidence(self, g, measured, previous, now):
        threshold = self._threshold(g)
        # Class/box-size changes or abrupt association jumps cannot count as
        # reliable motion. A measured slot must confirm in consecutive frames.
        # Perspective means image-plane pair distances need not stay rigid.
        qualifying = {
            s.tid: s
            for s in measured
            if g.baseline_frozen_at is not None
            and s.baseline is not None
            and s.motion_reliable
            and np.linalg.norm(s.point - s.baseline) >= threshold
        }
        previous_streaks = g.movement_streaks
        g.movement_streaks = {
            tid: (previous_streaks.get(tid, 0) + 1 if tid in previous_streaks else 1) for tid in qualifying
        }
        confirmed = [
            s for tid, s in qualifying.items() if g.movement_streaks[tid] >= self.config.movement_confirm_frames
        ]
        recent_travel = {
            s.tid: float(np.linalg.norm(s.point - s.motion_history[0][1])) if s.motion_history else 0.0
            for s in confirmed
        }
        if confirmed and min(recent_travel.values()) >= min(7.0, threshold * 0.5):
            return {
                "reason": "manual_rotation",
                "support_slot_ids": [s.sid for s in confirmed],
                "support_temporary_slot_ids": [s.tid for s in confirmed],
                "movement_frames": min(g.movement_streaks[s.tid] for s in confirmed),
                "threshold_px": threshold,
                "displacements_px": {s.tid: float(np.linalg.norm(s.point - s.baseline)) for s in confirmed},
                "baseline_frozen_at": g.baseline_frozen_at,
                "recent_measured_travel_px": recent_travel,
            }
        # Intersect actual consecutive measurement segment with the finite exit segment.
        for s in measured:
            if not s.motion_reliable:
                continue
            old = previous.get(s.tid)
            if old is None or now - old[1] > self.config.maximum_evidence_gap:
                continue
            a, b = old[0], s.point
            sa, sb = self._side(a), self._side(b)
            if not (sa >= 0 and sb < -self.config.exit_margin_px):
                continue
            fraction = sa / (sa - sb)
            crossing = a + fraction * (b - a)
            d = self.exit_p2 - self.exit_p1
            u = float(np.dot(crossing - self.exit_p1, d) / np.dot(d, d))
            if 0 <= u <= 1:
                return {
                    "reason": "slot_exit_line",
                    "support_slot_ids": [s.sid],
                    "crossing_xy": crossing.tolist(),
                    "line_parameter": u,
                    "previous_center_xy": a.tolist(),
                    "center_xy": b.tolist(),
                    "measurement_gap": now - old[1],
                    "motion_reliable": True,
                }
        return None

    def _close(self, now, evidence):
        g = self.current
        if g is None or g.frozen:
            return
        g.frozen, g.closed_at, g.close_reason = True, now, evidence["reason"]
        self.carriers[g.cid].retired = True
        for s in g.slots:
            self.tails.append(
                {"point": s.point.copy(), "at": now, "seen_at": s.last_seen,
                 "cid": g.cid, "gate": max(18.0, (g.min_spacing or 50) * 0.30)}
            )
        snapshot = deepcopy(g.snapshot())
        self.closed_groups.append(snapshot)
        self._emit(
            {
                "type": "group_closed",
                "at": now,
                "group_id": g.gid,
                "phase": g.phase,
                "carrier_track_id": g.cid,
                "evidence": evidence,
                "snapshot": snapshot,
            }
        )
        if g.gid == 5:
            if g.phase == "unload":
                self.phase, self.next_group = "load", 1
                self._emit({"type": "phase_changed", "at": now, "from": "unload", "to": "load", "next_group_id": 1})
            else:
                self.finished = True
                self._emit(
                    {
                        "type": "sequence_finished",
                        "at": now,
                        "g5_all_targets_verified": snapshot["all_targets_verified"],
                    }
                )
        else:
            self.next_group = g.gid + 1
        self.current = None
        self.entry = {}

    def process(self, objects, now, carrier_boxes=None):
        if not math.isfinite(now) or self.last_time is not None and now <= self.last_time:
            raise ValueError("Frame times must be finite and strictly increasing")
        gap = self.last_time is not None and now - self.last_time > self.config.maximum_evidence_gap
        self.last_time = now
        self._new_events = []
        dets = normalize(objects, self.config.confidence)
        groups = (
            normalize(carrier_boxes, self.config.confidence)
            if carrier_boxes is not None
            else [d for d in dets if d["class_name"] == "slot_group"]
        )
        groups = [d for d in groups if d["class_name"] == "slot_group"]
        self._update_carriers(groups, now)
        assigned, rejected = self._owners([d for d in dets if d["class_name"] in SLOT_CLASSES])
        hands = [d for d in dets if d["class_name"] == "hand"]
        # Tail ownership is spatial and class-independent; old empty/filled flicker
        # never creates a new group's slot. This does not modify frozen snapshots.
        for tail in self.tails:
            ds = assigned.get(tail["cid"], [])
            near = [d for d in ds if np.linalg.norm(d["center"] - tail["point"]) < tail["gate"]]
            if len(near) == 1:
                tail["point"] = near[0]["center"].copy()
                tail["seen_at"] = now
        closed_this_frame = None
        observations = []
        started_this_frame = False
        if not self.finished and self.current is None:
            eligible = [c for c in self.carriers.values() if self._eligible(c)
                        or (c.cid in self.candidates and not c.retired and c.observed
                            and point_in_polygon(center(c.box), self.roi) and self._side(center(c.box)) > 0)]
            ready = []
            for c in eligible:
                candidate = self.candidates.setdefault(c.cid, Group(0, self.phase, c.cid, now))
                cs, issues = self._clean_candidates(assigned.get(c.cid, []), hands, now)
                observations.extend({k: v for k, v in issue.items() if k != "detection"} for issue in issues)
                measured, _ = self._update_slots(candidate, cs, issues, now)
                self._number_and_freeze(candidate, measured, now)
                if candidate.baseline_frozen_at is not None:
                    ready.append(candidate)
            for cid in list(self.candidates):
                if (cid not in {c.cid for c in eligible}
                        and now - self.carriers[cid].seen_at > self.config.maximum_evidence_gap):
                    del self.candidates[cid]
            if ready:
                ready.sort(key=lambda g: self._side(center(self.carriers[g.cid].box)), reverse=True)
                self.current = ready[0]
                self.current.gid = self.next_group
                self.current.started_at = now
                started_this_frame = True
                g = self.current
                self.candidates = {}
                self._emit(
                    {
                        "type": "group_started",
                        "at": now,
                        "group_id": g.gid,
                        "phase": g.phase,
                        "carrier_track_id": g.cid,
                        "stationary_since": g.stable_since,
                        "numbering_status": "final" if g.numbered else "provisional",
                    }
                )
                self._emit(
                    {
                        "type": "slot_baseline_frozen",
                        "at": now,
                        "group_id": g.gid,
                        "phase": g.phase,
                        "carrier_track_id": g.cid,
                        "stationary_since": g.stable_since,
                        "minimum_slot_spacing_px": g.min_spacing,
                        "threshold_px": self._threshold(g),
                        "centers": {s.tid: s.baseline.tolist() for s in g.slots if s.baseline is not None},
                    }
                )
        g = self.current
        if g is not None:
            carrier = self.carriers[g.cid]
            if gap:
                g.movement_streaks = {}
                g.stable_since = None
            if carrier.observed and not started_this_frame:
                candidates, issues = self._clean_candidates(assigned.get(g.cid, []), hands, now)
                observations = [{k: v for k, v in item.items() if k != "detection"} for item in issues]
                measured, previous = self._update_slots(g, candidates, issues, now)
                self._number_and_freeze(g, measured, now)
                evidence = self._switch_evidence(g, measured, previous, now)
                if evidence:
                    closed_this_frame = g.snapshot()
                    self._close(now, evidence)
                elif g.phase == "load" and g.gid == 5 and g.numbered and all(s.state == "filled" for s in g.slots):
                    if g.target_since is None:
                        g.target_since = now
                    if now - g.target_since >= self.config.stationary_seconds:
                        closed_this_frame = g.snapshot()
                        self._close(now, {"reason": "g5_all_filled_confirmed",
                                          "confirmed_since": g.target_since,
                                          "confirmed_for_seconds": now - g.target_since})
                else:
                    g.target_since = None
            elif not carrier.observed:
                for s in g.slots:
                    s.unknown("carrier_box_not_observed")
                    s.motion_reliable, s.motion_history = False, []
                g.movement_streaks = {}
                g.stable_since = None
                g.target_since = None
        snap = (
            self.current.snapshot()
            if self.current
            else (self.closed_groups[-1] if closed_this_frame is not None else None)
        )
        return {
            "process_group_id": None if snap is None else snap["group_id"],
            "phase": self.phase if snap is None else snap["phase"],
            "carrier_track_id": None if snap is None else snap["carrier_track_id"],
            "carrier_box_observed": bool(g and self.carriers[g.cid].observed),
            "current_group": deepcopy(snap),
            "events": deepcopy(self._new_events),
            "closed_group_count": len(self.closed_groups),
            "finished": self.finished,
            "pending_group_id": None if self.finished else self.next_group,
            "pending_phase": self.phase,
            "rejected_slots": rejected + observations,
            "pending_carriers": [g.snapshot() for g in self.candidates.values()],
            "carriers": [
                {
                    "carrier_track_id": c.cid,
                    "yolo_track_ids": sorted(c.yolo_ids),
                    "xyxy": c.box,
                    "observed": c.observed,
                    "retired": c.retired,
                }
                for c in self.carriers.values()
                if now - c.seen_at <= self.config.carrier_lost_seconds
            ],
        }

    def finalize(self):
        groups = deepcopy(self.closed_groups)
        if self.current:
            groups.append(self.current.snapshot())
        observed = {(g["phase"], g["group_id"]): g for g in groups}
        summary = []
        for phase in ("unload", "load"):
            for gid in range(1, 6):
                g = observed.get((phase, gid))
                summary.append(
                    g
                    if g
                    else {
                        "phase": phase,
                        "group_id": gid,
                        "status": "not_observed",
                        "carrier_track_id": None,
                        "all_targets_verified": False,
                        "target_verified_count": 0,
                        "slots": [],
                        "events": [],
                    }
                )
        all_verified = all(g["all_targets_verified"] for g in summary)
        return {
            "status": ("complete" if all_verified else "ended_unverified") if self.finished else "incomplete",
            "sequence_finished": self.finished,
            "all_slots_verified": all_verified,
            "current_phase": self.phase,
            "next_group_id": None if self.finished else self.next_group,
            "closed_group_count": len(self.closed_groups),
            "groups": summary,
            "events": deepcopy(self.events),
            "config": asdict(self.config),
        }
