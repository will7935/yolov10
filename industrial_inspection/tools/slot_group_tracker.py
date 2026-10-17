"""单组四槽身份和状态跟踪；跨组计数由ROI分组控制器管理。"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

SLOT_STATES = {"wafer_slot_empty": "empty", "wafer_slot_filled": "filled"}


def validate_centers(centers):
    points = np.asarray(centers, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("需要四个有限的槽位中心坐标，顺序为槽1、2、3、4")
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    if np.min(distances[np.triu_indices(4, 1)]) < 2:
        raise ValueError("四个槽位中心过近，无法建立独立编号")
    return points


def box_iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - area
    return area / max(union, 1e-9)


@dataclass
class SlotState:
    slot_id: int
    confirmed: str = "unknown"
    evidence_time: float | None = None
    last_seen: float | None = None
    candidate: str = "unknown"
    candidate_since: float | None = None
    candidate_count: int = 0
    state: str = "unknown"
    observation: str = "unknown"
    confidence: float | None = None
    reason: str = "not_detected"
    geometry_observed: bool = False
    confirmation_source: str | None = None
    # 上料时手可能挡住槽位。这里仅缓存可靠的接触证据，离手后再确认，
    # 避免把手从槽位上方经过误当成已经放入晶圆。
    load_contact_since: float | None = None
    load_contact_last_seen: float | None = None
    load_contact_count: int = 0
    load_contact_pending: bool = False
    load_contact_released_at: float | None = None

    def observe(self, state, now, confidence, reason, confirm_seconds, minimum_frames):
        self.observation, self.confidence, self.reason = state, confidence, reason
        self.state = "unknown"
        if state == "unknown":
            self.candidate, self.candidate_since, self.candidate_count = "unknown", None, 0
            return None
        self.last_seen = now
        if self.candidate != state:
            self.candidate, self.candidate_since, self.candidate_count = state, now, 1
        else:
            self.candidate_count += 1
        if self.candidate_count < minimum_frames or now-self.candidate_since+1e-9 < confirm_seconds:
            self.reason = "confirming"
            return None
        previous, previous_time = self.confirmed, self.evidence_time
        self.confirmed, self.evidence_time, self.state, self.reason = state, now, state, "confirmed"
        self.confirmation_source = "yolo"
        # 初次看见满/空只建立基线，不虚构一次放入/取出事件。
        if previous in ("empty", "filled") and previous != state:
            return {"slot_id": self.slot_id, "from": previous, "to": state,
                    "action": "load" if state == "filled" else "unload",
                    "time_lower": previous_time,
                    "time_upper": self.candidate_since,
                    "confirmed_at": now}
        return None


class SlotGroupTracker:
    """用YOLO框中心自动建立四槽编号，再做一对一匹配和保守状态确认。

    group_id 是明确配置的组身份，不会因丢点或重新出现四点而自动改变。
    小幅连续运动可用已匹配槽位估计相似变换，跟随未检出的参考位置。
    长时间丢失后锁定身份，避免给新游星片继承旧编号。
    """

    def __init__(self, centers=None, *, group_id=1, group_count=5, phase="unload",
                 clockwise=(1, 3, 4, 2), match_distance_ratio=.4,
                 confirm_seconds=.3, minimum_frames=3, lost_seconds=1.2,
                 conflict_margin=.15, hand_overlap=.2, ambiguity_ratio=.08,
                 initialization_seconds=.3, initialization_frames=3,
                 initialization_stability_ratio=.12, initialization_history_seconds=5.,
                 load_hand_contact_enabled=True, load_hand_contact_seconds=.15,
                 load_hand_contact_min_frames=2):
        if group_count != 5 or not 1 <= group_id <= group_count:
            raise ValueError("当前流程固定五组，group_id 必须在1..5之间")
        if phase not in ("load", "unload") or sorted(clockwise) != [1, 2, 3, 4]:
            raise ValueError("phase 必须为load/unload；顺时针顺序必须包含槽1..4各一次")
        if not 0 < match_distance_ratio < .5 or confirm_seconds < 0 or minimum_frames < 1 or lost_seconds <= 0:
            raise ValueError("距离比例须在0..0.5之间；确认时间/帧数及丢失时限无效")
        if not 0 <= conflict_margin <= 1 or not 0 < hand_overlap <= 1 or not 0 <= ambiguity_ratio < .5:
            raise ValueError("冲突、遮挡或关联歧义阈值无效")
        if initialization_seconds < 0 or initialization_frames < 1:
            raise ValueError("初始化持续时间不能为负，帧数至少为1")
        if not 0 < initialization_stability_ratio < .5 or not np.isfinite(initialization_history_seconds) or initialization_history_seconds <= 0:
            raise ValueError("初始化停稳比例须在0..0.5之间，观测缓存秒数须为正有限数")
        if (not np.isfinite(load_hand_contact_seconds) or load_hand_contact_seconds < 0
                or load_hand_contact_min_frames < 1 or int(load_hand_contact_min_frames) != load_hand_contact_min_frames):
            raise ValueError("上料手接触确认参数无效")
        self.points = None if centers is None else validate_centers(centers).copy()
        # 编号参考位置与本帧检测中心分开。少数框的中心偏移不能改变整组关联模板。
        self.reference_points = None if self.points is None else self.points.copy()
        self.group_id, self.phase, self.clockwise = group_id, phase, tuple(clockwise)
        self.match_distance_ratio, self.confirm_seconds = match_distance_ratio, confirm_seconds
        self.minimum_frames, self.lost_seconds = minimum_frames, lost_seconds
        self.conflict_margin, self.hand_overlap, self.ambiguity_ratio = conflict_margin, hand_overlap, ambiguity_ratio
        self.slots = [SlotState(i+1) for i in range(4)]
        self.events = []
        self.last_time = None
        self.last_geometry_time = None
        self.identity_lost = False
        self.unmatched = []
        self.initialization_seconds, self.initialization_frames = initialization_seconds, initialization_frames
        self.initialization_stability_ratio = initialization_stability_ratio
        self.initialization_history_seconds = initialization_history_seconds
        self.load_hand_contact_enabled = bool(load_hand_contact_enabled)
        self.load_hand_contact_seconds = float(load_hand_contact_seconds)
        self.load_hand_contact_min_frames = int(load_hand_contact_min_frames)
        self._provisional = None
        self._entry_points = None
        self._entry_gate = None
        self._point_tracking_gate = None
        self._entry_events = []
        self._initial_seen = set()
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        self.initialization = {"status": "waiting" if centers is None else "ready",
            "method": "yolo_box_centers" if centers is None else "provided_centers",
            "reason": "waiting_for_four_detections" if centers is None else "provided_centers",
            "eligible_slot_count": 0, "stable_frames": 0, "required_slot_count": 4,
            "initialized_at": None, "initial_centers_xy": None if centers is None else self.points.tolist(),
            "provisional_started_at": None, "geometry_seen_slot_ids": [], "buffered_event_count": 0}

    def start_incremental_entry(self, match_distance):
        """新组逐点进入：先保存临时点号和状态，收齐四槽后只映射一次正式槽号。"""
        if self.last_time is not None or self.points is not None or not np.isfinite(match_distance) or match_distance <= 0:
            raise ValueError("逐点进入必须在新跟踪器首次更新前启动，匹配距离须为正")
        self._entry_points, self._entry_gate = [], float(match_distance)
        self._point_tracking_gate = float(match_distance)
        self.initialization.update(method="incremental_yolo_box_centers", reason="waiting_for_new_points")

    def _update_incremental_entry(self, objects, now, enabled, roi_contour):
        candidates = self._candidates(objects)
        hands = [o for o in objects if o["class_name"] == "hand"]
        matches = {}
        # 只按上一帧真实中心作一对一距离匹配；新点按出现顺序给临时点号。
        if self._entry_points and candidates:
            distances = np.linalg.norm(np.asarray(self._entry_points)[:, None]
                -np.asarray([c["center"] for c in candidates])[None, :], axis=2)
            cost = np.concatenate((np.where(distances <= self._entry_gate, distances, 1e6),
                np.full((len(self._entry_points), len(self._entry_points)), self._entry_gate+1)), axis=1)
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if col < len(candidates) and distances[row, col] <= self._entry_gate:
                    matches[int(row)] = int(col)
        new_indices = [j for j in range(len(candidates)) if j not in matches.values()]
        # 一个已知点附近的重复/歧义框不能另开一个点号；多于四槽时也不选高分四框。
        new_indices = [j for j in new_indices if not self._entry_points or
            min(np.linalg.norm(np.asarray(candidates[j]["center"])-point)
                for point in self._entry_points) > self._entry_gate]
        overflow = len(candidates) > 4 or len(self._entry_points)+len(new_indices) > 4
        if not overflow:
            for j in sorted(new_indices, key=lambda j: tuple(candidates[j]["center"])):
                i = len(self._entry_points)
                self._entry_points.append(np.asarray(candidates[j]["center"], dtype=float))
                matches[i] = j
        for i, col in matches.items():
            if not self._occluded(candidates[col], hands):
                self._entry_points[i] = np.asarray(candidates[col]["center"], dtype=float)
                if enabled:
                    self._initial_seen.add(i)
        cutoff = now-self.initialization_history_seconds
        for i, slot in enumerate(self.slots):
            if slot.evidence_time is not None and slot.evidence_time < cutoff:
                slot.confirmed, slot.evidence_time = "unknown", None
                slot.confirmation_source = None
            obj = candidates[matches[i]] if i in matches else None
            event = self._observe_slot(slot, obj, hands, now, enabled and not overflow, roi_contour)
            if event is not None:
                self._entry_events.append(event)
        self._entry_events = [e for e in self._entry_events if e["time_lower"] >= cutoff]
        self.unmatched = [c for j, c in enumerate(candidates) if j not in matches.values()]
        started = self.initialization["provisional_started_at"]
        self.initialization.update(status="collecting" if self._entry_points else "waiting",
            reason="too_many_slot_detections" if overflow else "collecting_slot_tracks",
            eligible_slot_count=len(candidates),
            provisional_started_at=float(now) if started is None and self._entry_points else started,
            geometry_seen_slot_ids=[i+1 for i in sorted(self._initial_seen)],
            buffered_event_count=len(self._entry_events))
        recovered = []
        if len(self._entry_points) == 4 and len(self._initial_seen) == 4 and not overflow:
            # 临时点号可以按任意顺序出现；状态和事件随点一起映射，不丢掉早期满槽基线。
            try:
                points = validate_centers(self._entry_points)
            except ValueError:
                self.initialization.update(reason="invalid_initial_geometry")
                return {**self.snapshot(), "new_events": []}
            order = np.lexsort((points[:, 1], points[:, 0]))
            mapping = {int(old)+1: new+1 for new, old in enumerate(order)}
            self.slots = [self.slots[i] for i in order]
            for i, slot in enumerate(self.slots):
                slot.slot_id = i+1
            self.points = points[order].copy()
            self.reference_points = self.points.copy()
            self.last_geometry_time = now
            self.events = self._entry_events
            for event in self.events:
                event.update(slot_id=mapping[event["slot_id"]], reported_at=float(now),
                    recovered_before_initialization=event["confirmed_at"] < now)
            recovered = list(self.events)
            self._entry_points = None
            self._entry_events = []
            self.initialization.update(status="ready", reason="four_slot_tracks_collected",
                initialized_at=float(now), initial_centers_xy=self.points.tolist(),
                geometry_seen_slot_ids=[1, 2, 3, 4])
        return {**self.snapshot(), "new_events": recovered}

    def reset_group(self, group_id, phase=None, centers=None):
        """供未来分组模块调用；切组必须显式执行，旧组摘要应由调用方先保存。"""
        # 新组不能沿用旧组中心。默认重新等待YOLO四槽初始化。
        self.__init__(centers, group_id=group_id,
                      phase=phase or self.phase, clockwise=self.clockwise,
                      match_distance_ratio=self.match_distance_ratio, confirm_seconds=self.confirm_seconds,
                      minimum_frames=self.minimum_frames, lost_seconds=self.lost_seconds,
                      conflict_margin=self.conflict_margin, hand_overlap=self.hand_overlap,
                      ambiguity_ratio=self.ambiguity_ratio,
                      initialization_seconds=self.initialization_seconds,
                      initialization_frames=self.initialization_frames,
                      initialization_stability_ratio=self.initialization_stability_ratio,
                      initialization_history_seconds=self.initialization_history_seconds,
                      load_hand_contact_enabled=self.load_hand_contact_enabled,
                      load_hand_contact_seconds=self.load_hand_contact_seconds,
                      load_hand_contact_min_frames=self.load_hand_contact_min_frames)

    def _reset_initialization(self, reason):
        """候选混组或轨迹中断时丢弃未归属的历史，不能接到另一组上。"""
        self._provisional = None
        self._initial_seen.clear()
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        self.initialization.update(status="waiting", reason=reason, stable_frames=0,
            provisional_started_at=None, initial_centers_xy=None,
            geometry_seen_slot_ids=[], buffered_event_count=0)

    def _initialize(self, candidates, hands, now, enabled, roi_contour, objects):
        """先建立候选轨迹并逐槽记录状态；身份确认后一次性发布缓存事件。"""
        self.initialization["eligible_slot_count"] = len(candidates)
        if len(candidates) > 4:
            self._reset_initialization("too_many_slot_detections")
            return []
        if self._provisional is not None:
            p = self._provisional
            matches = p._associate(candidates, p._spacing()*p.match_distance_ratio)
            unmatched_visible = any(col not in matches.values() and not self._occluded(obj,hands)
                                    for col,obj in enumerate(candidates))
            if now-p.last_geometry_time > self.lost_seconds or unmatched_visible:
                self._reset_initialization("initialization_track_discontinuity")
        if self._provisional is None:
            if len(candidates) != 4:
                self.initialization.update(reason="waiting_for_four_detections")
                return []
            try:
                points = validate_centers([c["center"] for c in candidates])
            except ValueError:
                self._reset_initialization("invalid_initial_geometry")
                return []
            # 槽号在候选轨迹开始时只排序一次；之后停稳或旋转均不重排。
            points = points[np.lexsort((points[:,1], points[:,0]))]
            self._provisional = SlotGroupTracker(points, group_id=self.group_id, phase=self.phase,
                clockwise=self.clockwise, match_distance_ratio=self.match_distance_ratio,
                confirm_seconds=self.confirm_seconds, minimum_frames=self.minimum_frames,
                lost_seconds=self.lost_seconds, conflict_margin=self.conflict_margin,
                hand_overlap=self.hand_overlap, ambiguity_ratio=self.ambiguity_ratio,
                load_hand_contact_enabled=self.load_hand_contact_enabled,
                load_hand_contact_seconds=self.load_hand_contact_seconds,
                load_hand_contact_min_frames=self.load_hand_contact_min_frames)
            self.initialization.update(provisional_started_at=float(now), initial_centers_xy=points.tolist())

        p = self._provisional
        # 缓存有界；超时的基线不能与很久以后的空/满拼成一次操作。
        cutoff = now-self.initialization_history_seconds
        for slot in p.slots:
            if slot.evidence_time is not None and slot.evidence_time < cutoff:
                slot.confirmed, slot.evidence_time = "unknown", None
                slot.confirmation_source = None
        p.events = [e for e in p.events if e["time_lower"] >= cutoff]
        matches = p._associate(candidates, p._spacing()*p.match_distance_ratio)
        reliable = {i: candidates[col] for i,col in matches.items()
                    if not self._occluded(candidates[col], hands)}
        p.update([o for o in objects if o["class_name"] in SLOT_STATES]+hands,
                 now, observations_enabled=enabled, roi_contour=roi_contour)
        # 空满冲突只影响该槽状态，不影响已匹配框的几何身份。
        if enabled:
            self._initial_seen.update(reliable)
        if enabled and len(reliable) >= 3:
            if self._initial_points is not None and any(np.linalg.norm(
                    np.asarray(obj["center"])-self._initial_points[i]) >
                    p._spacing()*self.initialization_stability_ratio for i,obj in reliable.items()):
                self._initial_points, self._initial_since, self._initial_count = None, None, 0
            if self._initial_points is None:
                self._initial_points, self._initial_since = p.points.copy(), now
            self._initial_count += 1
            reason = "confirming_group_geometry" if len(self._initial_seen) == 4 else "waiting_for_slot_geometry"
        else:
            self._initial_points, self._initial_since, self._initial_count = None, None, 0
            reason = "rotation_or_transition" if not enabled else "insufficient_visible_geometry"
        self.initialization.update(status="collecting", reason=reason, stable_frames=self._initial_count,
            geometry_seen_slot_ids=[i+1 for i in sorted(self._initial_seen)], buffered_event_count=len(p.events))
        if (len(self._initial_seen) < 4 or self._initial_count < self.initialization_frames
                or self._initial_since is None or now-self._initial_since+1e-9 < self.initialization_seconds):
            return []
        self.points, self.reference_points = p.points.copy(), p.reference_points.copy()
        self.slots, self.events, self.unmatched = p.slots, p.events, p.unmatched
        self.last_geometry_time = p.last_geometry_time
        for event in self.events:
            event.update(reported_at=float(now), recovered_before_initialization=event["confirmed_at"] < now)
        self._provisional = None
        self.initialization.update(status="ready", reason="initialized", initialized_at=float(now))
        return list(self.events)

    def _spacing(self):
        distances = np.linalg.norm(self.reference_points[:, None] - self.reference_points[None, :], axis=2)
        return float(np.min(distances[np.triu_indices(4, 1)]))

    def _candidates(self, objects):
        candidates = []
        # 高度重合候选先合并位置；分数接近的空满冲突仍保留为unknown，不按高分强判。
        for obj in sorted((o for o in objects if o["class_name"] in SLOT_STATES),
                          key=lambda o: o["confidence"], reverse=True):
            duplicate = next((c for c in candidates if box_iou(c["xyxy"], obj["xyxy"]) > .85), None)
            if duplicate is not None:
                if obj["class_name"] != duplicate["class_name"] and duplicate["confidence"]-obj["confidence"] < self.conflict_margin:
                    duplicate["conflict"] = True
                continue
            a = obj["xyxy"]
            candidates.append({**obj, "center": [(a[0]+a[2])/2, (a[1]+a[3])/2], "conflict": False})
        return candidates

    def _occluded(self, obj, hands):
        a = obj["xyxy"]
        area = max((a[2]-a[0])*(a[3]-a[1]), 1)
        cx, cy = obj["center"]
        for hand in hands:
            b = hand["xyxy"]
            overlap = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            if overlap/area >= self.hand_overlap or b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                return True
        return False

    def _associate(self, candidates, gate):
        if not candidates:
            return {}
        points = np.asarray([c["center"] for c in candidates])
        distances = np.linalg.norm(self.reference_points[:, None] - points[None, :], axis=2)
        # 加入每槽一个空匹配列，使匹配算法可以拒绝不合理的候选。
        cost = np.concatenate((np.where(distances <= gate, distances, 1e6), np.full((4, 4), gate+1)), axis=1)
        rows, cols = linear_sum_assignment(cost)
        matches = {}
        margin = self._spacing()*self.ambiguity_ratio
        for row, col in zip(rows, cols):
            if col >= len(candidates) or distances[row, col] > gate:
                continue
            alternatives = np.delete(distances[:, col], row)
            same_slot = np.delete(distances[row], col)
            if np.min(alternatives)-distances[row,col] < margin:
                continue
            # 同槽存在两个几何位置几乎一样好的非重复框时，保守拒绝归属。
            if len(same_slot) and np.min(same_slot)-distances[row,col] < margin:
                continue
            matches[int(row)] = int(col)
        return matches

    def _slot_point(self, slot_index):
        """返回当前槽的几何参考点；初始化逐点跟踪时也能使用。"""
        points = self.reference_points if self.reference_points is not None else self._entry_points
        if points is None or slot_index >= len(points):
            return None
        return np.asarray(points[slot_index], dtype=float)

    def _hand_contact(self, slot_index, hands):
        """只有手框覆盖唯一一个已跟踪槽中心才归属接触；多槽覆盖不猜测。"""
        if self.phase != "load" or not self.load_hand_contact_enabled:
            return False
        points = self.reference_points if self.reference_points is not None else self._entry_points
        if points is None or slot_index >= len(points):
            return False
        for hand in hands:
            a = hand["xyxy"]
            covered = [i for i, point in enumerate(points)
                       if a[0] <= point[0] <= a[2] and a[1] <= point[1] <= a[3]]
            if covered == [slot_index]:
                return True
        return False

    def _record_load_contact(self, slot, now, contact, enabled):
        if self.phase != "load" or not self.load_hand_contact_enabled:
            return
        if slot.confirmed != "empty" or (not enabled and not slot.load_contact_pending):
            self._clear_load_contact(slot)
            return
        if contact:
            slot.load_contact_released_at = None
            slot.load_contact_last_seen = now
            if not enabled:
                return
            if slot.load_contact_since is None:
                slot.load_contact_since = now
            slot.load_contact_count += 1
            if (slot.load_contact_count >= self.load_hand_contact_min_frames
                    and now-slot.load_contact_since+1e-9 >= self.load_hand_contact_seconds):
                slot.load_contact_pending = True
        elif slot.load_contact_pending:
            if slot.load_contact_released_at is None:
                slot.load_contact_released_at = now
        else:
            # 未达到门限前中断就从头累计，离散经过不能拼成持续接触。
            self._clear_load_contact(slot)

    def _clear_load_contact(self, slot):
        slot.load_contact_since = None
        slot.load_contact_last_seen = None
        slot.load_contact_count = 0
        slot.load_contact_pending = False
        slot.load_contact_released_at = None

    def update(self, objects, now, *, observations_enabled=True, initialization_objects=None,
               roi_contour=None, before_observe=None, translation_direction=None):
        """先关联几何，再确认状态；回调可在转位开始的同一帧暂停取放事件。"""
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("源视频时间必须严格递增")
        self.last_time = float(now)
        if self._entry_points is not None:
            initial_objects = objects if initialization_objects is None else initialization_objects
            return self._update_incremental_entry(initial_objects, now, observations_enabled, roi_contour)
        candidates = self._candidates(objects)
        hands = [o for o in objects if o["class_name"] == "hand"]
        if self.points is None:
            initial_objects = objects if initialization_objects is None else initialization_objects
            initial = self._candidates(initial_objects)
            recovered = self._initialize(initial,hands,now,observations_enabled,roi_contour,initial_objects)
            if self.points is None:
                self.unmatched = candidates
                # 候选轨迹已经能给出暂定的槽位状态；身份尚未确认时不发布
                # 正式事件，但把这些状态保留在当前帧，避免初始化等待期间
                # 的满槽检测被统一覆盖成unknown。
                if self._provisional is not None:
                    provisional = self._provisional.snapshot()
                    provisional["identity_status"] = "waiting_initialization"
                    provisional["slot_numbering_status"] = "provisional"
                    provisional["group_assignment"] = "configured"
                    provisional["initialization"] = dict(self.initialization)
                    provisional["unassigned_slot_detections"] = self.unmatched
                    provisional["new_events"] = []
                    return provisional
                for slot in self.slots:
                    slot.observe("unknown",now,None,self.initialization["reason"],self.confirm_seconds,self.minimum_frames)
                return {**self.snapshot(),"new_events": []}
            # 候选跟踪器已处理当前帧，不能在接管时重复累计确认帧数或事件。
            return {**self.snapshot(), "new_events": recovered}
        if self.last_geometry_time is None:
            self.last_geometry_time = now
        if self.last_geometry_time is not None and now-self.last_geometry_time > self.lost_seconds:
            self.identity_lost = True
        gate = self._point_tracking_gate or self._spacing()*self.match_distance_ratio
        matches = {} if self.identity_lost else self._associate(candidates, gate)
        # 局部补检只补空满观测；裁剪框的偏移不改变几何轨迹，也不作为越界证据。
        geometry_matches = {i: col for i,col in matches.items()
                            if candidates[col].get("detection_source") != "local_crop"}
        self.unmatched = [c for i,c in enumerate(candidates) if i not in matches.values()]
        if geometry_matches:
            self.last_geometry_time = now
            # 至少三个槽共同支持时才向不可见槽传播运动。
            # 两个检测框中心的抖动会被相似变换完全拟合，逐帧外推会把其余槽推离真实位置。
            indices = [i for i in geometry_matches
                       if not candidates[matches[i]]["conflict"] and not self._occluded(candidates[matches[i]],hands)]
            old = self.reference_points[indices].astype(np.float32)
            new = np.asarray([candidates[matches[i]]["center"] for i in indices], dtype=np.float32)
            if self._point_tracking_gate is not None:
                # 单点换组模式沿用逐点距离匹配，不用全组拟合外推未检出的槽。
                # 因此新组尚在转动时也能继续跟随真实中心，空满框偏移不会带走其他点。
                if indices:
                    self.reference_points[indices] = new
            elif len(indices) >= 3:
                matrix = None
                if translation_direction is not None and all(
                        not candidates[matches[i]]["conflict"] and not self._occluded(candidates[matches[i]],hands)
                        for i in indices):
                    # 转盘在斜视画面中旋转会同时出现剪切和非等比缩放。
                    # 仅转位阶段允许有界仿射拟合；操作阶段保留更严格的相似变换。
                    design = np.column_stack((old,np.ones(len(old))))
                    affine = np.linalg.lstsq(design,new,rcond=None)[0].T
                    singular = np.linalg.svd(affine[:,:2],compute_uv=False)
                    predicted = self.reference_points @ affine[:,:2].T + affine[:,2]
                    residual = np.linalg.norm(predicted[indices]-new,axis=1)
                    if (np.linalg.matrix_rank(design) == 3 and np.linalg.det(affine[:,:2]) > 0
                            and singular.min() >= .8 and singular.max() <= 1.2
                            and np.max(residual) <= gate*.4):
                        matrix = affine
                if matrix is None:
                    matrix, _ = cv2.estimateAffinePartial2D(old, new, method=cv2.LMEDS)
                if matrix is not None:
                    scale = float(np.hypot(matrix[0,0], matrix[1,0]))
                    angle = abs(float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))))
                    predicted = self.reference_points @ matrix[:,:2].T + matrix[:,2]
                    residual = np.linalg.norm(predicted[indices]-new, axis=1)
                    if .85 <= scale <= 1.15 and angle <= 15 and np.max(residual) <= gate*.4:
                        self.reference_points = predicted
            elif (len(indices) == 2 and translation_direction is not None
                  and all(not candidates[matches[i]]["conflict"] and not self._occluded(candidates[matches[i]],hands)
                          for i in indices)):
                # 已确认转位方向后，两槽只支持一致的平移，绝不拟合两点旋转/缩放。
                delta = new-old
                shift = delta.mean(axis=0)
                direction = np.asarray(translation_direction,dtype=float)
                if np.max(np.linalg.norm(delta-shift,axis=1)) <= gate*.2 and np.dot(shift,direction) >= 0:
                    self.reference_points += shift
        # 点仍显示YOLO框中心；漏检槽显示几何参考位置，关联门限不被单框抖动越缩越小。
        self.points = self.reference_points.copy()
        for i,col in geometry_matches.items():
            self.points[i] = candidates[col]["center"]
        source_enabled = observations_enabled
        if before_observe is not None:
            observations_enabled = bool(before_observe(self,candidates,geometry_matches,now)) and observations_enabled
        new_events = []
        for i,slot in enumerate(self.slots):
            obj = candidates[matches[i]] if i in matches else None
            # 转位时继续核验已有基线的槽位；无需先检测到手才能确认空满变化。
            verify_only = self.phase == "load" and not observations_enabled
            enabled = source_enabled and slot.confirmed in ("empty", "filled") if verify_only else observations_enabled
            event = self._observe_slot(slot, obj, hands, now, enabled, roi_contour,
                                       record_load_contact=not verify_only)
            if event is not None:
                self.events.append(event)
                new_events.append(event)
        result = self.snapshot()
        result["new_events"] = new_events
        return result

    def verify_departing_load(self, matches, hands, now, *, enabled=True):
        """退出点已由分组器独立关联，按实际空满核验旧组，不依赖手接触。"""
        if (self.phase != "load" or self.identity_lost
                or self.last_time is not None and now <= self.last_time):
            return []
        self.last_time = float(now)
        events = []
        for i, obj in matches.items():
            if not obj["conflict"] and not self._occluded(obj, hands):
                self.points[i] = self.reference_points[i] = obj["center"]
        for i, slot in enumerate(self.slots):
            if slot.confirmed not in ("empty", "filled"):
                continue
            obj = matches.get(i)
            event = self._observe_slot(slot, obj, hands, now, enabled, None, record_load_contact=False)
            if event is not None:
                self.events.append(event)
                events.append(event)
        return events

    def _observe_slot(self, slot, obj, hands, now, enabled, roi_contour, *, record_load_contact=True):
        """正式槽与进入中的临时点共用同一套空满确认规则。"""
        slot.geometry_observed = obj is not None and obj.get("detection_source") != "local_crop"
        contact = self._hand_contact(slot.slot_id-1, hands)
        point = self._slot_point(slot.slot_id-1)
        if slot.load_contact_pending and point is not None:
            # 多槽覆盖不能建立新证据，但已经有待核验操作时，手仍覆盖
            # 此槽就不算离手，避免宽手框被误当成接触结束。
            contact = contact or any(h["xyxy"][0] <= point[0] <= h["xyxy"][2]
                and h["xyxy"][1] <= point[1] <= h["xyxy"][3] for h in hands)
        inside = roi_contour is None or (point is not None and cv2.pointPolygonTest(
            roi_contour, tuple(point), False) >= 0)
        self._record_load_contact(slot, now, contact,
                                  enabled and record_load_contact and inside and not self.identity_lost)
        pending_contact = slot.load_contact_pending
        state, confidence, reason = "unknown", None, "not_detected"
        if self.identity_lost:
            reason = "identity_lost"
        elif obj is not None:
            confidence = obj["confidence"]
            if not enabled:
                reason = "rotation_or_transition"
            elif roi_contour is not None and cv2.pointPolygonTest(roi_contour, tuple(obj["center"]), False) < 0:
                reason = "outside_roi"
            elif obj["conflict"]:
                reason = "empty_filled_conflict"
            elif self._occluded(obj, hands) or contact:
                reason = "hand_occlusion"
            else:
                state, reason = SLOT_STATES[obj["class_name"]], "detected"
        event = slot.observe(state, now, confidence, reason, self.confirm_seconds, self.minimum_frames)
        # 接触只挂起待核验，不在操作阶段自动把漏检判成已填充。
        if pending_contact and state == "unknown" and slot.state == "unknown":
            slot.reason = "load_hand_contact_pending"
        if slot.state == "empty" and not contact and slot.load_contact_pending:
            # 手离开后仍明确检测为空，说明这次接触没有完成上料。
            self._clear_load_contact(slot)
        if event is not None:
            if pending_contact and event["action"] == "load":
                # 手接触可能是取片、放片或整理，只作辅助记录，不能用它
                # 缩窄实际空满证据的时间区间或替代视觉证据判断顺序。
                event.update(reason="detected_after_hand_contact", inferred=False)
            if event["action"] == "load":
                self._clear_load_contact(slot)
        if event is not None:
            event.update(group_id=self.group_id, phase=self.phase,
                expected_action=event["action"] == self.phase,
                reported_at=float(now), recovered_before_initialization=False)
        return event

    def order_result(self):
        """按事件时间区间检查顺序；遮挡造成先后无法区分时不给通过结论。"""
        if any(not e["expected_action"] for e in self.events):
            return "fail"
        sequence = [e["slot_id"] for e in self.events]
        if len(sequence) != len(set(sequence)) or len(sequence) > 4:
            return "fail"
        if not sequence:
            return "pending"
        legal = [self.clockwise[i:]+self.clockwise[:i] for i in range(4)]
        possible = []
        for permutation in itertools.permutations(range(len(self.events))):
            positions = {event_index: index for index,event_index in enumerate(permutation)}
            if any(a["time_upper"] < b["time_lower"] and positions[i] > positions[j]
                   for i,a in enumerate(self.events) for j,b in enumerate(self.events)):
                continue
            order = tuple(self.events[i]["slot_id"] for i in permutation)
            # 不完整序列可能漏掉了遮挡槽的事件，按环形子序列检查，不能直接判漏操作。
            possible.append(any(order == tuple(s for s in cycle if s in sequence) for cycle in legal))
        if possible and not any(possible):
            return "fail"
        if len(sequence) == 4 and possible and all(possible):
            return "pass"
        return "pending"

    def snapshot(self):
        target = "filled" if self.phase == "load" else "empty"
        slots = []
        for i,s in enumerate(self.slots):
            point = (self.points[i] if self.points is not None else
                self._entry_points[i] if self._entry_points is not None and i < len(self._entry_points) else None)
            slots.append({"group_id": self.group_id, "slot_id": s.slot_id,
                "center_xy": None if point is None else [round(float(v),2) for v in point],
                "state": s.state, "observation": s.observation, "last_confirmed_state": s.confirmed,
                "last_confirmed_at": s.evidence_time, "last_seen_at": s.last_seen,
                "confidence": s.confidence, "reason": s.reason,
                "confirmation_source": s.confirmation_source,
                "position_is_prediction": point is not None and not s.geometry_observed,
                "load_contact_pending": s.load_contact_pending,
                "load_contact_count": s.load_contact_count,
                "load_contact_since": s.load_contact_since,
                "load_contact_last_seen": s.load_contact_last_seen,
                "load_contact_released_at": s.load_contact_released_at})
        return {"group_id": self.group_id, "group_assignment": "configured",
                "slot_numbering_status": "provisional" if self.points is None else "final",
                "phase": self.phase, "identity_status": "waiting_initialization" if self.points is None else ("lost" if self.identity_lost else "tracking"),
                "initialization": dict(self.initialization),
                "slots": slots, "order_result": self.order_result(),
                "confirmed_event_sequence": [e["slot_id"] for e in sorted(self.events,
                    key=lambda e: (e["time_upper"],e["time_lower"])) if e["expected_action"]],
                "current_target_count": sum(s.state == target for s in self.slots),
                "last_verified_target_count": sum(s.confirmed == target for s in self.slots),
                "all_four_currently_confirmed": all(s.state == target for s in self.slots),
                "all_four_target_previously_verified": all(s.confirmed == target for s in self.slots),
                "unassigned_slot_detections": self.unmatched}
