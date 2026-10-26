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


def valid_slot_geometry(points):
    """四槽应形成有面积的凸四边形，不能用同槽偏移框凑出第四点。"""
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
    if len(hull) != 4:
        return False
    edges = np.roll(hull, -1, axis=0)-hull
    previous = np.roll(edges, 1, axis=0)
    sine = np.abs(edges[:, 0]*previous[:, 1]-edges[:, 1]*previous[:, 0])
    sine /= np.maximum(np.linalg.norm(edges, axis=1)*np.linalg.norm(previous, axis=1), 1e-9)
    return bool(np.min(sine) >= .35)


def infer_fourth_square_center(three_points):
    """按四个近似90度槽位的平行四边形关系，从三点推算缺失中心。"""
    points = np.asarray(three_points, dtype=float)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        return None
    distances = np.linalg.norm(points[:, None]-points[None, :], axis=2)
    pair = np.triu_indices(3, 1)
    order = np.argsort(distances[pair])
    # 长边对角线应明显长于相邻槽间距；拒绝近共线或含重复框的三点。
    largest, second = distances[pair][order[-1]], distances[pair][order[-2]]
    if largest < 2 or largest/ max(second, 1e-9) < 1.12:
        return None
    i, j = int(pair[0][order[-1]]), int(pair[1][order[-1]])
    k = next(index for index in range(3) if index not in (i, j))
    missing = points[i]+points[j]-points[k]
    four = np.vstack((points, missing))
    if not valid_slot_geometry(four):
        return None
    all_distances = np.linalg.norm(four[:, None]-four[None, :], axis=2)
    if np.min(all_distances[np.triu_indices(4, 1)]) < 2:
        return None
    return missing


@dataclass
class SlotState:
    slot_id: int
    confirmed: str = "unknown"
    evidence_time: float | None = None
    # 当前已确认状态首次成立的证据；遮挡和同状态复核不能推迟操作时间。
    confirmed_since: float | None = None
    confirmed_at: float | None = None
    last_seen: float | None = None
    candidate: str = "unknown"
    candidate_since: float | None = None
    candidate_count: int = 0
    state: str = "unknown"
    observation: str = "unknown"
    confidence: float | None = None
    reason: str = "not_detected"
    geometry_observed: bool = False
    geometry_inferred: bool = False
    box_size_wh: tuple[float, float] | None = None
    confirmation_source: str | None = None
    # 字段名保留 load_contact 以兼容现有快照；实际缓存的是上下料通用的
    # 手接触证据，必须在离手后由连续的空/满检测确认状态。
    load_contact_since: float | None = None
    load_contact_last_seen: float | None = None
    load_contact_count: int = 0
    load_contact_pending: bool = False
    load_contact_released_at: float | None = None
    load_contact_unknown_since: float | None = None
    load_contact_unknown_count: int = 0

    def observe(self, state, now, confidence, reason, confirm_seconds, minimum_frames,
                contact_evidence=None):
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
        if previous != state:
            self.confirmed_since, self.confirmed_at = self.candidate_since, now
        self.confirmed, self.evidence_time, self.state, self.reason = state, now, state, "confirmed"
        self.confirmation_source = "yolo"
        # 初次看见满/空只建立基线；有接触证据时才推断此前发生的操作。
        if previous == "unknown" and contact_evidence is not None:
            return {"slot_id": self.slot_id, "from": "unknown", "to": state,
                    "action": "load" if state == "filled" else "unload",
                    "time_lower": contact_evidence["since"],
                    "time_upper": self.candidate_since,
                    "confirmed_at": now,
                    "reason": "unknown_state_after_hand_contact",
                    "inferred": True,
                    "hand_contact_since": contact_evidence["since"],
                    "hand_contact_released_at": contact_evidence["released_at"],
                    "hand_contact_frames": contact_evidence["count"]}
        if previous in ("empty", "filled") and previous != state:
            return {"slot_id": self.slot_id, "from": previous, "to": state,
                    "action": "load" if state == "filled" else "unload",
                    "time_lower": previous_time,
                    "time_upper": self.candidate_since,
                    "confirmed_at": now}
        return None


class SlotGroupTracker:
    """组建立时用几何一次性确定S1-S4，之后以ByteTrack ID为主维护槽位状态。

    group_id 是明确配置的组身份，不会因丢点或重新出现四点而自动改变。
    轨迹ID失效时用有界几何匹配恢复；多槽共同支持时估计载体变换并跟随漏检位置。
    长时间丢失后锁定身份，避免给新游星片继承旧编号。
    """

    def __init__(self, centers=None, *, group_id=1, group_count=5, phase="unload",
                 clockwise=(1, 3, 4, 2), match_distance_ratio=.4,
                 confirm_seconds=.3, minimum_frames=3, lost_seconds=1.2,
                 conflict_margin=.15, hand_overlap=.2, ambiguity_ratio=.08,
                 initialization_seconds=.3, initialization_frames=3,
                 initialization_stability_ratio=.12, initialization_history_seconds=5.,
                 load_hand_contact_enabled=True, load_hand_contact_seconds=.15,
                 load_hand_contact_min_frames=2, slot2_below_slot3=False):
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
        # Canonical carrier pose is captured once when S1-S4 are assigned.
        # The exported transform maps this pose to the current image frame.
        self.canonical_points = None if self.points is None else self.points.copy()
        self.group_transform = np.array([[1., 0., 0.], [0., 1., 0.]]) if self.points is not None else None
        self.transform_valid_this_frame = self.points is not None
        self.transform_fit_error = 0.0 if self.points is not None else None
        self.transform_support_slot_ids = [1, 2, 3, 4] if self.points is not None else []
        self.transform_support_byte_track_ids = []
        # ByteTrack identity is an additional association cue, not the S-number
        # itself. This list is permuted together with geometry when S1-S4 are
        # assigned, so display and event records can map BT IDs to formal slots.
        self.byte_track_ids = [None] * 4
        self.track_store = None  # Shared video-wide state; assigned by RoiGroupSequence.
        self.geometry_rejected = []
        self._history_events_this_frame = []
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
        self._predicted_indices = set()
        self.initialization_seconds, self.initialization_frames = initialization_seconds, initialization_frames
        self.initialization_stability_ratio = initialization_stability_ratio
        self.initialization_history_seconds = initialization_history_seconds
        self.load_hand_contact_enabled = bool(load_hand_contact_enabled)
        self.load_hand_contact_seconds = float(load_hand_contact_seconds)
        self.load_hand_contact_min_frames = int(load_hand_contact_min_frames)
        self.slot2_below_slot3 = bool(slot2_below_slot3)
        self._provisional = None
        self._provisional_inferred_slot_ids = set()
        self._entry_points = None
        self._entry_gate = None
        self._entry_track_ids = []
        self._point_tracking_gate = None
        self._entry_events = []
        self._entry_seen_at = []
        self._entry_support_since = []
        self._entry_support_count = []
        self._entry_geometry_at_seen = []
        self._entry_inferred_indices = set()
        self._entry_reference_points = None
        self._entry_canonical_points = None
        self._entry_revision = 0
        self._initial_seen = set()
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        self.initialization = {"status": "waiting" if centers is None else "ready",
            "method": "yolo_box_centers" if centers is None else "provided_centers",
            "reason": "waiting_for_four_detections" if centers is None else "provided_centers",
            "eligible_slot_count": 0, "stable_frames": 0, "required_slot_count": 4,
            "initialized_at": None, "initial_centers_xy": None if centers is None else self.points.tolist(),
            "provisional_started_at": None, "geometry_seen_slot_ids": [], "geometry_seen_track_ids": [],
            "inferred_slot_ids": [],
            "buffered_event_count": 0}

    def start_incremental_entry(self, match_distance):
        """新组逐点进入；下料阶段三点可推算第四点，几何停稳后映射槽号。"""
        if self.last_time is not None or self.points is not None or not np.isfinite(match_distance) or match_distance <= 0:
            raise ValueError("逐点进入必须在新跟踪器首次更新前启动，匹配距离须为正")
        self._entry_points, self._entry_gate = [], float(match_distance)
        self._entry_track_ids = []
        self._point_tracking_gate = float(match_distance)
        self.initialization.update(method="incremental_yolo_box_centers", reason="waiting_for_new_points")

    def _numbering_order(self, points):
        """只在建立身份时编号；相机已知的上下关系辅助区分中间两槽。"""
        order = np.lexsort((points[:, 1], points[:, 0]))
        if self.slot2_below_slot3:
            # 固定相机下S1是左上起点，顺时针位置为1、3、4、2。
            # 使用二维环序，避免同一左/右侧的框中心轻微偏移改变x极值。
            offsets = points-points.mean(axis=0)
            ring = np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))
            start = int(np.flatnonzero(ring == np.argmin(points.sum(axis=1)))[0])
            ring = np.roll(ring, -start)
            order = ring[[0, 3, 1, 2]]
        return order

    def _entry_s4_index(self):
        """Return the temporary index that currently represents formal S4."""
        if self._entry_points is None or len(self._entry_points) < 4:
            return None
        pose = (self._entry_reference_points if self._entry_reference_points is not None
                and len(self._entry_reference_points) == 4 else np.asarray(self._entry_points))
        return int(self._numbering_order(pose)[3])

    def _s4_index(self):
        """Return the current point-array index for formal S4, if known."""
        return self._entry_s4_index() if self._entry_points is not None else 3

    def _expire_entry_tracks(self, now):
        """失联临时点不能一直占用四槽名额；历史随保留轨迹一起映射。"""
        keep = [i for i, seen in enumerate(self._entry_seen_at)
                if now-seen <= (self.initialization_history_seconds if i in self._initial_seen
                                else min(self.lost_seconds, self.initialization_history_seconds))]
        # 下料时，三个真实槽位持续存在即可保留由几何推算的第四点。
        if self.phase == "unload" and self._entry_inferred_indices:
            observed_keep = [i for i in keep if i not in self._entry_inferred_indices]
            if len(observed_keep) >= 3:
                keep = sorted(set(keep) | self._entry_inferred_indices)
        # 已确认的四条真实轨迹也可在单槽短时遮挡时互相支持。
        required_seen = 3 if self._entry_inferred_indices else 4
        if len(self._entry_points) == 4 and len(self._initial_seen) >= required_seen and len(keep) >= 3:
            return
        if len(keep) == len(self._entry_points):
            return
        if self.track_store is not None:
            for i, track_id in enumerate(self._entry_track_ids):
                if i not in keep:
                    self.track_store.release_candidate(track_id, (self.phase, self.group_id))
        mapping = {i+1: j+1 for j, i in enumerate(keep)}
        self._entry_inferred_indices = {j for j, i in enumerate(keep)
                                        if i in self._entry_inferred_indices}
        self._entry_points = [self._entry_points[i] for i in keep]
        self._entry_track_ids = [self._entry_track_ids[i] for i in keep]
        self._entry_seen_at = [self._entry_seen_at[i] for i in keep]
        self._entry_support_since = [self._entry_support_since[i] for i in keep]
        self._entry_support_count = [self._entry_support_count[i] for i in keep]
        anchors = []
        for i in keep:
            anchor = np.full((len(keep), 2), np.nan)
            previous = self._entry_geometry_at_seen[i]
            if previous is not None:
                for j, old in enumerate(keep):
                    if old < len(previous):
                        anchor[j] = previous[old]
            anchors.append(anchor)
        self._entry_geometry_at_seen = anchors
        self._initial_seen = {mapping[i+1]-1 for i in self._initial_seen if i+1 in mapping}
        self.slots = [self.slots[i] for i in keep]
        self.slots.extend(SlotState(i+1) for i in range(len(keep), 4))
        for i, slot in enumerate(self.slots):
            slot.slot_id = i+1
        self._entry_events = [e for e in self._entry_events if e["slot_id"] in mapping]
        for event in self._entry_events:
            event["slot_id"] = mapping[event["slot_id"]]
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        self._entry_reference_points = None
        self._entry_revision += 1

    def _associate_with_track_ids(self, candidates, distances, track_ids, gate):
        """Use ByteTrack IDs as physical-slot identity, then gated geometry.

        ByteTrack follows the box through group movement and short occlusion,
        so an existing unique ID remains attached to its mapped slot even when
        its center has moved beyond the stationary geometry gate. Geometry is
        only the recovery path when ByteTrack creates a new ID after a gap.
        The ROI sequence keeps BT IDs continuous and isolates closed owners.
        """
        row_count, col_count = distances.shape
        if self.track_store is not None and self.points is not None:
            distances = distances.copy()
            for col, obj in enumerate(candidates):
                record = self.track_store.tracks.get(obj.get('track_id'))
                if (record is not None and record['owner'] == (self.phase, self.group_id)
                        and record['slot_id'] is not None):
                    for row in range(row_count):
                        if row+1 != record['slot_id']:
                            distances[row, col] = 1e6
        matches, used_rows, used_cols, blocked_cols = {}, set(), set(), set()
        id_columns = {}
        for col, candidate in enumerate(candidates):
            track_id = candidate.get("track_id")
            if track_id is not None:
                id_columns.setdefault(int(track_id), []).append(col)
        for row, track_id in enumerate(track_ids):
            if track_id is None or int(track_id) not in id_columns:
                continue
            cols = id_columns[int(track_id)]
            if len(cols) != 1:
                blocked_cols.update(cols)
                continue
            col = cols[0]
            matches[row] = col
            used_rows.add(row)
            used_cols.add(col)

        remaining_rows = [row for row in range(row_count) if row not in used_rows]
        remaining_cols = [col for col in range(col_count) if col not in used_cols | blocked_cols]
        if not remaining_rows or not remaining_cols:
            return matches
        sub = distances[np.ix_(remaining_rows, remaining_cols)]
        cost = np.concatenate((np.where(sub <= gate, sub, 1e6),
                               np.full((len(remaining_rows), len(remaining_rows)), gate + 1)), axis=1)
        rows, cols = linear_sum_assignment(cost)
        points = np.asarray([candidate["center"] for candidate in candidates])
        spacing = (np.min(np.linalg.norm(points[:, None]-points[None, :], axis=2)
                    [np.triu_indices(len(points), 1)]) if len(points) > 1 else gate)
        ambiguity = spacing * self.ambiguity_ratio
        for local_row, local_col in zip(rows, cols):
            if local_col >= len(remaining_cols):
                continue
            row, col = remaining_rows[local_row], remaining_cols[local_col]
            distance = distances[row, col]
            if distance > gate:
                continue
            alternatives = [distances[other, col] for other in remaining_rows if other != row]
            same_slot = [distances[row, other] for other in remaining_cols if other != col]
            if (alternatives and min(alternatives)-distance < ambiguity
                    or same_slot and min(same_slot)-distance < ambiguity):
                continue
            matches[row] = col
        return matches

    @staticmethod
    def _remember_track_ids(matches, candidates, track_ids):
        """Update ID-to-temporary-slot links without allowing duplicate owners."""
        for row, col in matches.items():
            track_id = candidates[col].get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            for other, old_id in enumerate(track_ids):
                if other != row and old_id == track_id:
                    track_ids[other] = None
            track_ids[row] = track_id

    def _associate_entry_distances(self, distances, candidates=None):
        """可拒绝候选的一对一匹配；被运动约束否决的框不再参与歧义比较。"""
        count = len(self._entry_points)
        if candidates is not None:
            return self._associate_with_track_ids(candidates, distances,
                                                  self._entry_track_ids, self._entry_gate)
        cost = np.concatenate((np.where(distances <= self._entry_gate, distances, 1e6),
            np.full((count, count), self._entry_gate+1)), axis=1)
        rows, cols = linear_sum_assignment(cost)
        points = np.asarray(self._entry_points)
        spacing = (np.min(np.linalg.norm(points[:, None]-points[None, :], axis=2)
                    [np.triu_indices(count, 1)]) if count > 1 else self._entry_gate)
        ambiguity = spacing*self.ambiguity_ratio
        matches = {}
        for row, col in zip(rows, cols):
            if col < distances.shape[1] and distances[row, col] <= self._entry_gate:
                alternatives = np.delete(distances[:, col], row)
                same_track = np.delete(distances[row], col)
                if (len(alternatives) and np.min(alternatives)-distances[row, col] < ambiguity
                        or len(same_track) and np.min(same_track)-distances[row, col] < ambiguity):
                    continue
                matches[int(row)] = int(col)
        return matches

    def _update_incremental_entry(self, objects, now, enabled, roi_contour):
        self._predicted_indices.clear()
        self._expire_entry_tracks(now)
        self._predicted_indices.update(self._entry_inferred_indices)
        candidates = self._candidates(objects)
        hands = [o for o in objects if o["class_name"] == "hand"]
        if enabled:
            candidates = self._reconcile_entry_geometry(candidates, hands, now)
        matches = {}
        # 只按上一帧真实中心作一对一距离匹配；新点按首次建立顺序编号，
        # 同帧多个新点按中心(x,y)排序；临时点号不代表最终S号。
        if self._entry_points and candidates:
            distances = np.linalg.norm(np.asarray(self._entry_points)[:, None]
                -np.asarray([c["center"] for c in candidates])[None, :], axis=2)
            matches = self._associate_entry_distances(distances, candidates)
        # 遮挡后的旧中心可能更靠近刚出现的邻槽。两个近期真实匹配只用于
        # 否决反向跳点，不外推隐藏槽的位置，也不据此建立新的槽位。
        recent_seconds = max(self.confirm_seconds, self.initialization_seconds)
        support = [i for i, col in matches.items()
                   if now-self._entry_seen_at[i] <= recent_seconds
                   and not candidates[col]["conflict"] and not self._occluded(candidates[col], hands)]
        if len(support) >= 2:
            for i, anchor in enumerate(self._entry_geometry_at_seen):
                if anchor is None or now-self._entry_seen_at[i] <= recent_seconds:
                    continue
                indices = [j for j in support if j < len(anchor) and np.isfinite(anchor[j]).all()]
                if len(indices) < 2:
                    continue
                # 从该槽最后真实出现时累计运动，避免下一帧运动变小就又接到邻槽。
                deltas = np.asarray([np.asarray(candidates[matches[j]]["center"])-anchor[j] for j in indices])
                shift = deltas.mean(axis=0)
                distance = np.linalg.norm(shift)
                if (distance > self._entry_gate*.1
                        and np.max(np.linalg.norm(deltas-shift, axis=1)) <= self._entry_gate*.6):
                    for col, obj in enumerate(candidates):
                        delta = np.asarray(obj["center"])-self._entry_points[i]
                        if np.dot(delta, shift)/distance < -self._entry_gate*.2:
                            distances[i, col] = 1e6
            matches = self._associate_entry_distances(distances, candidates)
        new_indices = [j for j in range(len(candidates)) if j not in matches.values()]
        # 一个已知点附近的重复/歧义框不能另开一个点号；多于四槽时也不选高分四框。
        new_indices = [j for j in new_indices if not self._entry_points or
            min(np.linalg.norm(np.asarray(candidates[j]["center"])-point)
                for point in self._entry_points) > self._entry_gate]
        # 三点推算只是S4暂时不可见时的中心估计。若后续出现落在该估计附近、
        # 且能形成有效四槽几何的真实框，用它逐帧校正推算点，而非把组判成超额候选。
        if self.phase == "unload" and len(self._entry_points) == 4 and self._entry_inferred_indices:
            for i in sorted(self._entry_inferred_indices):
                for j in sorted(new_indices):
                    center = np.asarray(candidates[j]["center"], dtype=float)
                    distance = np.linalg.norm(center-self._entry_points[i])
                    other_distances = [np.linalg.norm(center-self._entry_points[k])
                                       for k in range(4) if k != i]
                    revised = np.asarray(self._entry_points).copy()
                    revised[i] = center
                    if (distance <= self._entry_gate*1.5
                            and distance < min(other_distances)
                            and valid_slot_geometry(revised)):
                        matches[i] = j
                        new_indices.remove(j)
                        break
        overflow = len(candidates) > 4 or len(self._entry_points)+len(new_indices) > 4
        if not overflow:
            for j in sorted(new_indices, key=lambda j: tuple(candidates[j]["center"])):
                if (self.slot2_below_slot3 and len(self._entry_points) == 3
                        and not valid_slot_geometry([*self._entry_points, candidates[j]["center"]])):
                    continue
                i = len(self._entry_points)
                self._entry_points.append(np.asarray(candidates[j]["center"], dtype=float))
                self._entry_track_ids.append(None)
                self._entry_seen_at.append(float(now))
                self._entry_support_since.append(None)
                self._entry_support_count.append(0)
                self._entry_geometry_at_seen.append(None)
                matches[i] = j
        inferred_non_s4 = False
        if len(self._entry_points) == 3 and self.phase == "unload":
            inferred = infer_fourth_square_center(self._entry_points)
            if inferred is not None:
                index = len(self._entry_points)
                proposed_points = np.vstack((self._entry_points, inferred))
                proposed_order = self._numbering_order(proposed_points)
                if int(proposed_order[3]) == index:
                    self._entry_points.append(inferred)
                    self._entry_track_ids.append(None)
                    self._entry_seen_at.append(float(now))
                    self._entry_support_since.append(None)
                    self._entry_support_count.append(0)
                    self._entry_geometry_at_seen.append(None)
                    self._entry_inferred_indices.add(index)
                    self._predicted_indices.add(index)
                    self.slots[index].geometry_inferred = True
                else:
                    # 业务只允许推算边缘S4。若缺失的是其他正式槽，继续等
                    # 真实检测，避免临时点编号导致状态/事件转移到错误槽位。
                    inferred_non_s4 = True
        self._remember_track_ids(matches, candidates, self._entry_track_ids)
        reliable = [i for i, col in matches.items() if not candidates[col]["conflict"]
                    and not self._occluded(candidates[col], hands)]
        if self.phase == "unload" and self._entry_inferred_indices and len(candidates) <= 4:
            # 三个可靠真实槽位已能支持该组；第四个不匹配候选先留作未分配框，
            # 不要让边缘S4的抖动/重复框反复清零几何停稳计时。
            if len(reliable) >= 3:
                overflow = False
        if len(reliable) >= 3:
            for i in self._entry_inferred_indices:
                self._entry_seen_at[i] = float(now)
        if len(self._entry_points) == 4 and len(reliable) >= 3:
            old = np.asarray(self._entry_points)[reliable].astype(np.float32)
            new = np.asarray([candidates[matches[i]]["center"] for i in reliable], dtype=np.float32)
            matrix, _ = cv2.estimateAffinePartial2D(old, new, method=cv2.LMEDS)
            if matrix is not None:
                predicted = np.asarray(self._entry_points) @ matrix[:, :2].T+matrix[:, 2]
                scale = np.hypot(matrix[0, 0], matrix[1, 0])
                if .85 <= scale <= 1.15 and np.max(np.linalg.norm(predicted[reliable]-new, axis=1)) <= self._entry_gate*.2:
                    s4_index = self._entry_s4_index()
                    if s4_index is not None and s4_index not in reliable:
                        self._entry_points[s4_index] = predicted[s4_index]
                        self._predicted_indices.add(s4_index)
        for i, col in matches.items():
            if not candidates[col]["conflict"] and not self._occluded(candidates[col], hands):
                self._entry_points[i] = np.asarray(candidates[col]["center"], dtype=float)
                # 临时点立即显示和记录状态；正式编号另需持续真实观测。
                # 漏检/遮挡期间的预测不计入帧数，离散闪现也不能拼成稳定身份。
                support_seconds = max(self.confirm_seconds, self.initialization_seconds)
                maximum_gap = min(self.lost_seconds, max(support_seconds, .25))
                if (self._entry_support_since[i] is None
                        or now-self._entry_seen_at[i] > maximum_gap):
                    self._entry_support_since[i] = float(now)
                    self._entry_support_count[i] = 0
                if enabled:
                    self._entry_support_count[i] += 1
                self._entry_seen_at[i] = float(now)
                if (enabled and self._entry_support_count[i] >= max(2, self.minimum_frames, self.initialization_frames)
                        and now-self._entry_support_since[i]+1e-9 >= support_seconds):
                    self._initial_seen.add(i)
                    if i in self._entry_inferred_indices:
                        self._entry_inferred_indices.remove(i)
                        self.slots[i].geometry_inferred = False
                if not enabled:
                    self._entry_support_since[i] = None
                    self._entry_support_count[i] = 0
        geometry = np.asarray(self._entry_points).copy()
        for i in range(len(geometry)):
            if i not in reliable:
                geometry[i] = np.nan
        for i in reliable:
            self._entry_geometry_at_seen[i] = geometry.copy()
        cutoff = now-self.initialization_history_seconds
        for i, slot in enumerate(self.slots):
            if slot.evidence_time is not None and slot.evidence_time < cutoff:
                slot.confirmed, slot.evidence_time = "unknown", None
                slot.confirmed_since, slot.confirmed_at = None, None
                slot.confirmation_source = None
            obj = candidates[matches[i]] if i in matches else None
            event = self._observe_slot(slot, obj, hands, now, enabled and not overflow, roi_contour)
            if event is not None:
                self._entry_events.append(event)
        # 已完成的状态变化随有效轨迹保留到正式编号；不能按旧状态证据的
        # 年龄清掉已确认事件。轨迹失效时由_expire_entry_tracks连同身份清理。
        self.unmatched = [c for j, c in enumerate(candidates) if j not in matches.values()]
        reliable = [i for i, col in matches.items() if not candidates[col]["conflict"]
                    and not self._occluded(candidates[col], hands)]
        # 三个可靠点即可判断进入运动是否已停止；新增第四点后重新计时。
        if enabled and not overflow and len(self._entry_points) >= 3 and len(reliable) >= 3:
            geometry = np.asarray(self._entry_points)
            spacing = np.min(np.linalg.norm(geometry[:, None]-geometry[None, :], axis=2)
                [np.triu_indices(len(geometry), 1)])
            if (self._initial_points is not None and (len(self._initial_points) != len(geometry)
                    or any(np.linalg.norm(geometry[i]-self._initial_points[i]) >
                           spacing*self.initialization_stability_ratio for i in reliable))):
                self._initial_points, self._initial_since, self._initial_count = None, None, 0
            if self._initial_points is None:
                self._initial_points, self._initial_since = geometry.copy(), now
            self._initial_count += 1
        else:
            self._initial_points, self._initial_since, self._initial_count = None, None, 0
        stable = (self._initial_count >= self.initialization_frames and self._initial_since is not None
                  and now-self._initial_since+1e-9 >= self.initialization_seconds)
        if stable:
            self._entry_reference_points = np.asarray(self._entry_points).copy()
            if self._entry_canonical_points is None:
                self._entry_canonical_points = self._entry_reference_points.copy()
        self.transform_valid_this_frame = False
        self.transform_fit_error = None
        self.transform_support_slot_ids = []
        self.transform_support_byte_track_ids = []
        if (self._entry_canonical_points is not None and len(reliable) >= 3
                and max(reliable) < len(self._entry_canonical_points)):
            source = self._entry_canonical_points[reliable].astype(np.float32)
            target = np.asarray([candidates[matches[i]]["center"] for i in reliable], dtype=np.float32)
            transform, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
            if transform is not None and np.linalg.det(transform[:, :2]) > 0:
                errors = np.linalg.norm(source @ transform[:, :2].T + transform[:, 2] - target, axis=1)
                fit_error = float(np.sqrt(np.mean(errors**2)))
                singular = np.linalg.svd(transform[:, :2], compute_uv=False)
                max_error = max(2., self._entry_gate * .2)
                if (np.isfinite(fit_error) and fit_error <= max_error
                        and singular.min() >= .8 and singular.max() <= 1.2):
                    self.group_transform = transform.astype(float)
                    self.transform_valid_this_frame = True
                    self.transform_fit_error = fit_error
                    self.transform_support_byte_track_ids = [self._entry_track_ids[i] for i in reliable]
        started = self.initialization["provisional_started_at"]
        self.initialization.update(status="collecting" if self._entry_points else "waiting",
            reason=("too_many_slot_detections" if overflow else
                    "waiting_for_detected_non_s4_slot" if inferred_non_s4 else "collecting_slot_tracks"),
            eligible_slot_count=len(candidates),
            provisional_started_at=float(now) if started is None and self._entry_points else started,
            geometry_seen_slot_ids=[],
            geometry_seen_track_ids=[self._entry_track_ids[i] for i in sorted(self._initial_seen)
                                     if self._entry_track_ids[i] is not None],
            inferred_slot_ids=sorted(i+1 for i in self._entry_inferred_indices),
            required_slot_count=3 if self._entry_inferred_indices else 4,
            buffered_event_count=len(self._entry_events), stable_frames=self._initial_count)
        recovered = []
        required_seen = 3 if self._entry_inferred_indices else 4
        if len(self._entry_points) == 4 and len(self._initial_seen) >= required_seen and not overflow:
            # 临时点号可以按任意顺序出现；状态和事件随点一起映射，不丢掉早期满槽基线。
            try:
                points = validate_centers(self._entry_points)
            except ValueError:
                self.initialization.update(reason="invalid_initial_geometry")
                return {**self.snapshot(), "new_events": []}
            self.initialization.update(reason="confirming_group_geometry", stable_frames=self._initial_count)
            numbering_points = points
            if not stable:
                # 第四槽可能直到下一次转位才解除遮挡。已有三槽的停稳姿态可
                # 作为编号参考，按同三条轨迹把新见第四槽还原到该姿态。
                reference = self._entry_reference_points
                support = [i for i in reliable if reference is not None and i < len(reference)]
                if reference is None or len(support) < 2:
                    return {**self.snapshot(), "new_events": []}
                design = np.column_stack((points[support], np.ones(len(support))))
                if len(support) == 2:
                    # 两个对应点只支持相似变换，不拟合任意仿射或更新隐藏槽轨迹。
                    transform, _ = cv2.estimateAffinePartial2D(points[support].astype(np.float32),
                        reference[support].astype(np.float32), method=cv2.LMEDS)
                    if transform is None:
                        return {**self.snapshot(), "new_events": []}
                    matrix = transform.T
                else:
                    matrix = np.linalg.lstsq(design, reference[support], rcond=None)[0]
                singular = np.linalg.svd(matrix[:2], compute_uv=False)
                if (np.linalg.matrix_rank(design) < min(len(support), 3) or np.linalg.det(matrix[:2]) <= 0
                        or singular.min() < .5 or singular.max() > 2):
                    return {**self.snapshot(), "new_events": []}
                numbering_points = np.column_stack((points, np.ones(4))) @ matrix
            spacing = np.min(np.linalg.norm(numbering_points[:, None]-numbering_points[None, :], axis=2)
                [np.triu_indices(4, 1)])
            # 二维环序结合本相机的上下约束直接消除当前姿态的编号歧义。
            # 历史停稳几何用于确认身份，不能让两点还原误差旋转整套编号。
            pose = points if self.slot2_below_slot3 else numbering_points
            order = self._numbering_order(pose)
            if self.slot2_below_slot3 and (not valid_slot_geometry(numbering_points)
                    or pose[order[1], 1]-pose[order[2], 1] < -spacing*.2):
                self.initialization.update(reason="waiting_for_numbering_pose")
                return {**self.snapshot(), "new_events": []}
            mapping = {int(old)+1: new+1 for new, old in enumerate(order)}
            self.slots = [self.slots[i] for i in order]
            inferred_slot_ids = sorted(int(mapping[i+1]) for i in self._entry_inferred_indices)
            for i, slot in enumerate(self.slots):
                slot.slot_id = i+1
                slot.geometry_inferred = i+1 in inferred_slot_ids
            self.points = points[order].copy()
            self.reference_points = self.points.copy()
            canonical_entry = self._entry_canonical_points
            if canonical_entry is not None and len(canonical_entry) == 3:
                inferred_canonical = infer_fourth_square_center(canonical_entry)
                if inferred_canonical is not None:
                    canonical_entry = np.vstack((canonical_entry, inferred_canonical))
            self.canonical_points = (canonical_entry[order].copy()
                                     if canonical_entry is not None and len(canonical_entry) == 4
                                     else self.points.copy())
            if self._entry_canonical_points is None:
                self.group_transform = np.array([[1., 0., 0.], [0., 1., 0.]])
                self.transform_valid_this_frame = True
                self.transform_fit_error = 0.0
            else:
                self.transform_valid_this_frame = self.group_transform is not None
            self.transform_support_slot_ids = [mapping[i+1] for i in reliable if i+1 in mapping]
            self.byte_track_ids = [self._entry_track_ids[i] for i in order]
            self._predicted_indices = {new for new, old in enumerate(order)
                                       if old in self._predicted_indices}
            self.last_geometry_time = now
            self.events = self._entry_events
            for event in self.events:
                event.update(slot_id=mapping[event["slot_id"]], reported_at=float(now),
                    recovered_before_initialization=event["confirmed_at"] < now)
            recovered = list(self.events)
            self._entry_points = None
            self._entry_track_ids = []
            self._entry_events = []
            self._entry_inferred_indices = set()
            self._entry_canonical_points = None
            self.initialization.update(status="ready",
                reason="four_slot_tracks_stable" if stable else "stable_reference_numbering",
                initialized_at=float(now), initial_centers_xy=self.points.tolist(),
                numbering_reference_centers_xy=numbering_points[order].tolist(),
                geometry_seen_slot_ids=sorted(int(mapping[i+1]) for i in self._initial_seen),
                geometry_seen_track_ids=[self.byte_track_ids[mapping[i+1]-1]
                                         for i in sorted(self._initial_seen)
                                         if self.byte_track_ids[mapping[i+1]-1] is not None],
                inferred_slot_ids=inferred_slot_ids,
                required_slot_count=3 if inferred_slot_ids else 4)
        return {**self.snapshot(), "new_events": recovered}

    def _select_track_geometry(self, candidates, hands, now):
        """Choose a uniquely supported carrier layout, never the top four scores."""
        self.geometry_rejected = []
        if self.track_store is None:
            return None
        reliable = []
        for obj in candidates:
            record = self.track_store.tracks.get(obj.get('track_id'))
            if (record is not None and not obj['conflict'] and not self._occluded(obj, hands)
                    and record['reliable_since'] is not None
                    and now-record['reliable_since']+1e-9 >= self.initialization_seconds
                    and record['reliable_count'] >= self.initialization_frames):
                reliable.append(obj)
        # Limit combinatorics by duration of geometric evidence, not confidence.
        reliable.sort(key=lambda obj: self.track_store.tracks[obj['track_id']]['reliable_since'])
        reliable = reliable[:10]
        layouts = []
        for subset in itertools.combinations(reliable, 4):
            points = np.asarray([obj['center'] for obj in subset], dtype=np.float32)
            if not valid_slot_geometry(points):
                continue
            hull = cv2.convexHull(points).reshape(-1, 2)
            sides = np.linalg.norm(hull-np.roll(hull, -1, axis=0), axis=1)
            spacing = float(sides.min())
            if spacing < 2 or sides.max() > spacing*4:
                continue
            # Opposite-slot diagonals should have approximately the same
            # midpoint. Allow perspective and detector box-centre displacement.
            error = float(np.linalg.norm(hull[0]+hull[2]-hull[1]-hull[3])/spacing)
            if error > .55:
                continue
            layouts.append((error, list(subset)))
        layouts.sort(key=lambda item: item[0])
        if not layouts or (len(layouts) > 1 and layouts[1][0]-layouts[0][0] < .08):
            if len(reliable) >= 4:
                reason = 'invalid_group_geometry' if not layouts else 'ambiguous_group_geometry'
                self.geometry_rejected = [{**obj, 'reason': reason} for obj in candidates]
            return None
        chosen = layouts[0][1]
        ids = {obj['track_id'] for obj in chosen}
        self.geometry_rejected = [{**obj, 'reason': 'outside_supported_group_geometry'}
                                 for obj in candidates if obj.get('track_id') not in ids]
        return chosen

    def _reconcile_entry_geometry(self, candidates, hands, now):
        """A late clear quartet may replace an incorrect provisional candidate."""
        chosen = self._select_track_geometry(candidates, hands, now)
        if chosen is None:
            return candidates
        old_ids = set(self._entry_track_ids)-{None}
        new_ids = {obj['track_id'] for obj in chosen}
        if (len(self._entry_points) == 4 and not self._entry_inferred_indices
                and old_ids == new_ids):
            return chosen
        # Rebuild provisional geometry only. Retained tracks keep their state,
        # support and events; newly accepted IDs restore independent history.
        old = {track_id: i for i, track_id in enumerate(self._entry_track_ids)
               if track_id is not None}
        for track_id in old_ids-new_ids:
            self.track_store.release_candidate(track_id, (self.phase, self.group_id))
        order = sorted(chosen, key=lambda obj: tuple(obj['center']))
        mapping = {old[obj['track_id']]+1: i+1 for i, obj in enumerate(order)
                   if obj['track_id'] in old}
        fields = ('_entry_seen_at', '_entry_support_since', '_entry_support_count',
                  '_entry_geometry_at_seen')
        saved = {name: list(getattr(self, name)) for name in fields}
        slots = self.slots
        seen = self._initial_seen.copy()
        self._entry_points = [np.asarray(obj['center'], dtype=float) for obj in order]
        self._entry_track_ids = [obj['track_id'] for obj in order]
        self.slots = []
        self._initial_seen = set()
        defaults = (now, None, 0, None)
        for name, default in zip(fields, defaults):
            setattr(self, name, [saved[name][old[obj['track_id']]]
                if obj['track_id'] in old else default for obj in order])
        for i, obj in enumerate(order):
            index = old.get(obj['track_id'])
            slot = slots[index] if index is not None else SlotState(i+1)
            slot.slot_id = i+1
            self.slots.append(slot)
            if index in seen:
                self._initial_seen.add(i)
        retained_events = []
        for event in self._entry_events:
            if event['slot_id'] in mapping:
                retained_events.append({**event, 'slot_id': mapping[event['slot_id']]})
        self._entry_events = retained_events
        self._entry_inferred_indices.clear()
        self._predicted_indices.clear()
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        self._entry_reference_points = self._entry_canonical_points = None
        # Historical co-observations are in the old index order.
        self._entry_geometry_at_seen = [None]*4
        self._entry_revision += 1
        return chosen

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
                      load_hand_contact_min_frames=self.load_hand_contact_min_frames,
                      slot2_below_slot3=self.slot2_below_slot3)

    def _reset_initialization(self, reason):
        """候选混组或轨迹中断时丢弃未归属的历史，不能接到另一组上。"""
        if self.track_store is not None and self._provisional is not None:
            for track_id in self._provisional.byte_track_ids:
                self.track_store.release_candidate(track_id, (self.phase, self.group_id))
        self._provisional = None
        self._provisional_inferred_slot_ids = set()
        self._initial_seen.clear()
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        self.initialization.update(status="waiting", reason=reason, stable_frames=0,
            provisional_started_at=None, initial_centers_xy=None,
            geometry_seen_slot_ids=[], geometry_seen_track_ids=[], inferred_slot_ids=[], required_slot_count=4,
            buffered_event_count=0)

    def _initialize(self, candidates, hands, now, enabled, roi_contour, objects):
        """先建立候选轨迹并逐槽记录状态；身份确认后一次性发布缓存事件。"""
        if enabled and len(candidates) > 4:
            selected = self._select_track_geometry(candidates, hands, now)
            if selected is not None:
                candidates = selected
                selected_ids = {obj['track_id'] for obj in selected}
                objects = [o for o in objects if o['class_name'] not in SLOT_STATES
                           or o.get('track_id') in selected_ids]
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
            inferred_missing = None
            if len(candidates) == 3 and self.phase == "unload":
                inferred_missing = infer_fourth_square_center([c["center"] for c in candidates])
            if len(candidates) == 4:
                raw_points = np.asarray([c["center"] for c in candidates], dtype=float)
            elif inferred_missing is not None:
                raw_points = np.vstack(([c["center"] for c in candidates], inferred_missing))
            else:
                reason = ("waiting_for_valid_three_slot_geometry" if len(candidates) == 3
                          and self.phase == "unload" else "waiting_for_four_detections")
                self.initialization.update(reason=reason, inferred_slot_ids=[])
                return []
            try:
                raw_points = validate_centers(raw_points)
            except ValueError:
                self._reset_initialization("invalid_initial_geometry")
                return []
            # 槽号在候选轨迹开始时只排序一次；之后停稳或旋转均不重排。
            order = self._numbering_order(raw_points)
            raw_track_ids = [candidate.get("track_id") for candidate in candidates]
            if inferred_missing is not None:
                raw_track_ids.append(None)
            missing_index = int(np.flatnonzero(order == 3)[0]) if inferred_missing is not None else None
            if inferred_missing is not None and missing_index != 3:
                self.initialization.update(reason="waiting_for_detected_non_s4_slot",
                                           inferred_slot_ids=[], required_slot_count=4)
                return []
            points = raw_points[order]
            self._provisional = SlotGroupTracker(points, group_id=self.group_id, phase=self.phase,
                clockwise=self.clockwise, match_distance_ratio=self.match_distance_ratio,
                confirm_seconds=self.confirm_seconds, minimum_frames=self.minimum_frames,
                lost_seconds=self.lost_seconds, conflict_margin=self.conflict_margin,
                hand_overlap=self.hand_overlap, ambiguity_ratio=self.ambiguity_ratio,
                load_hand_contact_enabled=self.load_hand_contact_enabled,
                load_hand_contact_seconds=self.load_hand_contact_seconds,
                load_hand_contact_min_frames=self.load_hand_contact_min_frames)
            self._provisional.byte_track_ids = [raw_track_ids[int(i)] for i in order]
            self._provisional.track_store = self.track_store
            self._provisional_inferred_slot_ids = set() if missing_index is None else {missing_index}
            if missing_index is not None:
                self._predicted_indices.add(missing_index)
                self._provisional.slots[missing_index].geometry_inferred = True
            self.initialization.update(provisional_started_at=float(now), initial_centers_xy=points.tolist(),
                inferred_slot_ids=[] if missing_index is None else [missing_index+1],
                required_slot_count=3 if missing_index is not None else 4)

        p = self._provisional
        # 过期原状态不能拼成新操作；已确认事件则随有效候选身份保留，
        # 候选几何中断时由_reset_initialization整体清理。
        cutoff = now-self.initialization_history_seconds
        for slot in p.slots:
            if slot.evidence_time is not None and slot.evidence_time < cutoff:
                slot.confirmed, slot.evidence_time = "unknown", None
                slot.confirmed_since, slot.confirmed_at = None, None
                slot.confirmation_source = None
        matches = p._associate(candidates, p._spacing()*p.match_distance_ratio)
        reliable = {i: candidates[col] for i,col in matches.items()
                    if not self._occluded(candidates[col], hands)}
        self._provisional_inferred_slot_ids.difference_update(reliable)
        self._predicted_indices.update(self._provisional_inferred_slot_ids)
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
            required_seen = 3 if self._provisional_inferred_slot_ids else 4
            reason = ("confirming_group_geometry" if len(self._initial_seen) >= required_seen
                      else "waiting_for_slot_geometry")
        else:
            self._initial_points, self._initial_since, self._initial_count = None, None, 0
            reason = "rotation_or_transition" if not enabled else "insufficient_visible_geometry"
        self.initialization.update(status="collecting", reason=reason, stable_frames=self._initial_count,
            geometry_seen_slot_ids=[],
            geometry_seen_track_ids=[p.byte_track_ids[i] for i in sorted(self._initial_seen)
                                     if p.byte_track_ids[i] is not None],
            buffered_event_count=len(p.events))
        required_seen = 3 if self._provisional_inferred_slot_ids else 4
        self.initialization["required_slot_count"] = required_seen
        self.initialization["inferred_slot_ids"] = sorted(i+1 for i in self._provisional_inferred_slot_ids)
        if (len(self._initial_seen) < required_seen or self._initial_count < self.initialization_frames
                or self._initial_since is None or now-self._initial_since+1e-9 < self.initialization_seconds):
            return []
        order = self._numbering_order(p.points) if self.slot2_below_slot3 else np.arange(4)
        self.points, self.reference_points = p.points[order].copy(), p.reference_points[order].copy()
        self.canonical_points = self.points.copy()
        self.group_transform = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.transform_valid_this_frame = True
        self.transform_fit_error = 0.0
        self.transform_support_slot_ids = [1, 2, 3, 4]
        self.byte_track_ids = [p.byte_track_ids[i] for i in order]
        self.slots, self.events, self.unmatched = [p.slots[i] for i in order], p.events, p.unmatched
        mapping = {int(old)+1: new+1 for new, old in enumerate(order)}
        for i, slot in enumerate(self.slots):
            slot.slot_id = i+1
        self.last_geometry_time = p.last_geometry_time
        for event in self.events:
            event.update(slot_id=mapping[event["slot_id"]], reported_at=float(now),
                         recovered_before_initialization=event["confirmed_at"] < now)
        self._provisional = None
        self._provisional_inferred_slot_ids = set()
        self.initialization.update(status="ready", reason="initialized", initialized_at=float(now),
            geometry_seen_slot_ids=sorted(mapping[i+1] for i in self._initial_seen),
            geometry_seen_track_ids=[self.byte_track_ids[mapping[i+1]-1] for i in sorted(self._initial_seen)
                                     if self.byte_track_ids[mapping[i+1]-1] is not None])
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
        return self._associate_with_track_ids(candidates, distances, self.byte_track_ids, gate)

    def clear_byte_track_ids(self):
        """Drop IDs when the ROI sequence starts a different process group."""
        self.byte_track_ids = [None] * 4
        self._entry_track_ids = [None] * len(self._entry_points or [])
        if self._provisional is not None:
            self._provisional.byte_track_ids = [None] * 4

    def _slot_point(self, slot_index):
        """返回当前槽的几何参考点；初始化逐点跟踪时也能使用。"""
        points = self.reference_points if self.reference_points is not None else self._entry_points
        if points is None or slot_index >= len(points):
            return None
        return np.asarray(points[slot_index], dtype=float)

    def _byte_track_id_at(self, slot_index):
        track_ids = self.byte_track_ids if self.points is not None else self._entry_track_ids
        return track_ids[slot_index] if slot_index < len(track_ids) else None

    def _hand_contact(self, slot_index, hands):
        """手接触状态推断只针对S4，避免一只手在多个槽之间造成串判。"""
        if not self.load_hand_contact_enabled or slot_index != self._s4_index():
            return False
        points = self.reference_points if self.reference_points is not None else self._entry_points
        if points is None or len(points) != 4 or slot_index >= len(points):
            return False
        for hand in hands:
            a = hand["xyxy"]
            covered = [i for i, point in enumerate(points)
                       if a[0] <= point[0] <= a[2] and a[1] <= point[1] <= a[3]]
            if covered == [slot_index]:
                return True
        return False

    def _record_hand_contact(self, slot, now, contact, enabled, slot_index):
        if slot_index != self._s4_index():
            self._clear_load_contact(slot)
            return
        if not self.load_hand_contact_enabled:
            return
        # Any previously confirmed empty/filled slot can change after contact;
        # do not limit contact evidence to the nominal phase target.
        eligible = slot.confirmed in ("unknown", "empty", "filled")
        if not eligible or (not enabled and not slot.load_contact_pending):
            self._clear_load_contact(slot)
            return
        if (slot.load_contact_pending and slot.load_contact_released_at is not None
                and now-slot.load_contact_released_at > self.lost_seconds):
            self._clear_load_contact(slot)
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
                # During unloading, an untouched/previously unseen slot is
                # assumed full once a valid hand contact starts. If YOLO stays
                # unknown after the hand leaves, the normal contact fallback
                # can then record the unload as filled -> empty.
                if self.phase == "unload" and slot.confirmed == "unknown":
                    slot.confirmed = "filled"
                    slot.confirmed_since = float(slot.load_contact_since)
                    slot.confirmed_at = float(now)
                    slot.evidence_time = float(slot.load_contact_since)
                    slot.confirmation_source = "unload_hand_contact_baseline"
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
        slot.load_contact_unknown_since = None
        slot.load_contact_unknown_count = 0

    def update(self, objects, now, *, observations_enabled=True, initialization_objects=None,
               roi_contour=None, before_observe=None, translation_direction=None):
        """先关联几何，再确认状态；回调可在转位开始的同一帧暂停取放事件。"""
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("源视频时间必须严格递增")
        self.last_time = float(now)
        self.geometry_rejected = []
        self._history_events_this_frame = []
        if self._entry_points is not None:
            initial_objects = objects if initialization_objects is None else initialization_objects
            return self._update_incremental_entry(initial_objects, now, observations_enabled, roi_contour)
        self._predicted_indices.clear()
        self.transform_valid_this_frame = False
        self.transform_support_slot_ids = []
        self.transform_support_byte_track_ids = []
        self.transform_fit_error = None
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
                    provisional["group_transform"] = None
                    provisional["transform_valid_this_frame"] = False
                    provisional["transform_fit_error"] = None
                    provisional["transform_support_slot_ids"] = []
                    for index, slot in enumerate(provisional.get("slots", []), start=1):
                        slot["slot_id"] = None
                        slot["formal_slot_id"] = None
                        slot["internal_slot_index"] = index
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
        self._remember_track_ids(matches, candidates, self.byte_track_ids)
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
            self.transform_valid_this_frame = False
            if len(indices) >= 3 and self.canonical_points is not None:
                # Fit one absolute group-to-image transform from the formal
                # S-slot correspondences. ByteTrack maintains identity; the
                # geometry fit validates the carrier and never renumbers it.
                canonical_support = self.canonical_points[indices].astype(np.float32)
                if translation_direction is not None:
                    design = np.column_stack((canonical_support, np.ones(len(indices))))
                    transform = (np.linalg.lstsq(design, new, rcond=None)[0].T
                                 if np.linalg.matrix_rank(design) == 3 else None)
                else:
                    transform, _ = cv2.estimateAffinePartial2D(
                        canonical_support, new, method=cv2.LMEDS)
                if transform is not None and np.linalg.det(transform[:, :2]) > 0:
                    projected = canonical_support @ transform[:, :2].T + transform[:, 2]
                    errors = np.linalg.norm(projected-new, axis=1)
                    fit_error = float(np.sqrt(np.mean(errors**2)))
                    max_error = max(2., self._spacing() * self.match_distance_ratio * .4)
                    singular = np.linalg.svd(transform[:, :2], compute_uv=False)
                    transform_shape_ok = (singular.min() >= .8 and singular.max() <= 1.2)
                    if (np.isfinite(fit_error) and fit_error <= max_error
                            and transform_shape_ok):
                        self.group_transform = transform.astype(float)
                        self.transform_fit_error = fit_error
                        self.transform_support_slot_ids = [i+1 for i in indices]
                        self.transform_support_byte_track_ids = [self.byte_track_ids[i] for i in indices]
                        self.transform_valid_this_frame = True
            if self._point_tracking_gate is not None:
                # 单点换组模式沿用逐点距离匹配，不用全组拟合外推未检出的槽。
                # 因此新组尚在转动时也能继续跟随真实中心，空满框偏移不会带走其他点。
                missing = set(range(4))-set(indices)
                s4_index = 3
                can_predict_s4 = self.phase == "unload" and len(indices) == 3 and missing == {s4_index}
                predicted_s4 = None
                if can_predict_s4:
                    design = np.column_stack((old, np.ones(len(old))))
                    if np.linalg.matrix_rank(design) == 3:
                        affine = np.linalg.lstsq(design, new, rcond=None)[0].T
                        singular = np.linalg.svd(affine[:, :2], compute_uv=False)
                        residual = np.linalg.norm(old @ affine[:, :2].T+affine[:, 2]-new, axis=1)
                        if (np.linalg.det(affine[:, :2]) > 0 and singular.min() >= .8
                                and singular.max() <= 1.2 and np.max(residual) <= gate*.4):
                            predicted_s4 = self.reference_points[s4_index] @ affine[:, :2].T+affine[:, 2]
                if len(indices) >= 2:
                    self.reference_points[indices] = new
                if predicted_s4 is not None:
                    self.reference_points[s4_index] = predicted_s4
                    self._predicted_indices.add(s4_index)
            elif len(indices) >= 3:
                missing = set(range(4))-set(indices)
                matrix = None
                can_predict_s4 = self.phase == "unload" and len(indices) == 3 and missing == {3}
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
                        # 内部参考几何跟随可信整体运动，维持后续匹配与出口判断；
                        # 只有缺失S4且有S1-S3支持时，才将该外推位置作为虚拟框输出。
                        self.reference_points = predicted
                        if can_predict_s4:
                            self._predicted_indices.add(3)
            elif len(indices) == 2 and translation_direction is not None and all(
                    not candidates[matches[i]]["conflict"] and not self._occluded(candidates[matches[i]],hands)
                    for i in indices):
                # 两个真实槽仅支持沿已确认转位方向的一致平移；参考几何用于
                # 后续关联和ROI出口判断，不据此生成其他槽的虚拟显示框。
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
        result["new_events"] = self._history_events_this_frame + new_events
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
        if enabled and obj is not None and self.track_store is not None:
            recovered = self.track_store.restore(slot, obj.get('track_id'),
                                                 (self.phase, self.group_id), now)
            target = self._entry_events if self._entry_points is not None else self.events
            target.extend(recovered)
            if self.points is not None:
                self._history_events_this_frame.extend(recovered)
        slot.geometry_observed = obj is not None and obj.get("detection_source") != "local_crop"
        if obj is not None:
            slot.geometry_inferred = False
            box = obj["xyxy"]
            slot.box_size_wh = (max(1., box[2]-box[0]), max(1., box[3]-box[1]))
        contact = self._hand_contact(slot.slot_id-1, hands)
        point = self._slot_point(slot.slot_id-1)
        if slot.slot_id-1 == self._s4_index() and slot.load_contact_pending and point is not None:
            # 多槽覆盖不能建立新证据，但已经有待核验操作时，手仍覆盖
            # 此槽就不算离手，避免宽手框被误当成接触结束。
            contact = contact or any(h["xyxy"][0] <= point[0] <= h["xyxy"][2]
                and h["xyxy"][1] <= point[1] <= h["xyxy"][3] for h in hands)
        inside = roi_contour is None or (point is not None and cv2.pointPolygonTest(
            roi_contour, tuple(point), False) >= 0)
        self._record_hand_contact(slot, now, contact,
                                  enabled and record_load_contact and inside and not self.identity_lost,
                                  slot.slot_id-1)
        pending_contact = slot.load_contact_pending
        contact_evidence = None
        if (pending_contact and slot.confirmed == "unknown"
                and slot.load_contact_released_at is not None):
            contact_evidence = {"since": slot.load_contact_since,
                                "released_at": slot.load_contact_released_at,
                                "count": slot.load_contact_count}
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
        prior_confirmed = slot.confirmed
        if contact_evidence is not None:
            inferred_action = "load" if state == "filled" else "unload" if state == "empty" else None
            if inferred_action != self.phase:
                contact_evidence = None
        event = slot.observe(state, now, confidence, reason, self.confirm_seconds,
                             self.minimum_frames, contact_evidence=contact_evidence)
        # 直接 YOLO 状态变化优先；目标状态仍可由视觉确认并使用正常事件路径。
        if pending_contact and state == "unknown" and slot.state == "unknown":
            slot.reason = "hand_contact_pending"
        target = "empty" if self.phase == "unload" else "filled"
        if (pending_contact and not contact and state in ("empty", "filled")
                and state != target):
            # YOLO 直接显示与当前工序目标相反的状态时，以视觉结果为准，
            # 取消接触推断，避免把整理动作/短暂触碰记成取放。
            self._clear_load_contact(slot)
            pending_contact = False
        if pending_contact and slot.load_contact_released_at is not None and not contact:
            if state == "unknown" and reason == "not_detected":
                if slot.load_contact_unknown_since is None:
                    slot.load_contact_unknown_since = float(now)
                    slot.load_contact_unknown_count = 1
                else:
                    slot.load_contact_unknown_count += 1
            else:
                slot.load_contact_unknown_since = None
                slot.load_contact_unknown_count = 0
        # 若视觉稳定确认仍是接触前状态，说明这次接触没有改变槽位状态。
        if (pending_contact and not contact and prior_confirmed in ("empty", "filled")
                and slot.state == prior_confirmed and slot.confirmed == prior_confirmed):
            self._clear_load_contact(slot)
        elif event is not None:
            if pending_contact and event["action"] == self.phase and not event.get("inferred", False):
                event.update(reason="detected_after_hand_contact", inferred=False)
            if pending_contact or event.get("inferred"):
                self._clear_load_contact(slot)
        elif (pending_contact and not contact and slot.load_contact_released_at is not None
              and prior_confirmed in ("empty", "filled") and state == "unknown"
              and reason == "not_detected" and slot.load_contact_unknown_since is not None
              and slot.load_contact_unknown_count >= 1
              and now-slot.load_contact_released_at+1e-9 >= min(
                  self.lost_seconds * .75,
                  max(.5, self.confirm_seconds * 2, self.load_hand_contact_seconds))):
            # 手接触前后YOLO都无法确认时，短暂等待视觉恢复后按已确认基线
            # 翻转状态：empty→filled、filled→empty；手接触本身不直接定状态。
            released_at = slot.load_contact_released_at
            since, count = slot.load_contact_since, slot.load_contact_count
            inferred_state = "filled" if prior_confirmed == "empty" else "empty"
            action = "load" if inferred_state == "filled" else "unload"
            slot.confirmed = slot.state = inferred_state
            slot.confirmed_since = since
            slot.confirmed_at = float(now)
            slot.evidence_time = float(now)
            slot.confirmation_source = "hand_contact"
            slot.reason = (f"hand_contact_inferred_{action}" if action == self.phase
                           else "hand_contact_unknown_state_toggle")
            event = {"slot_id": slot.slot_id, "from": prior_confirmed, "to": inferred_state,
                "action": action, "time_lower": since, "time_upper": released_at,
                "confirmed_at": float(now), "reason": slot.reason,
                "inferred": True, "hand_contact_since": since,
                "hand_contact_released_at": released_at, "hand_contact_frames": count}
            self._clear_load_contact(slot)
        if event is not None and slot.load_contact_pending:
            if pending_contact and not event.get("inferred", False):
                event.update(reason="detected_after_hand_contact", inferred=False)
            self._clear_load_contact(slot)
        if event is not None:
            event.update(group_id=self.group_id, phase=self.phase,
                byte_track_id=self._byte_track_id_at(slot.slot_id-1),
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
        sizes = [slot.box_size_wh for slot in self.slots if slot.box_size_wh is not None]
        typical_size = None if not sizes else np.median(sizes, axis=0)
        track_ids = (self.byte_track_ids if self.points is not None else self._entry_track_ids)
        for i,s in enumerate(self.slots):
            point = (self.points[i] if self.points is not None else
                self._entry_points[i] if self._entry_points is not None and i < len(self._entry_points) else None)
            size = s.box_size_wh if s.box_size_wh is not None else typical_size
            tracking_box = None if point is None or size is None else [
                round(float(point[0]-size[0]/2), 2), round(float(point[1]-size[1]/2), 2),
                round(float(point[0]+size[0]/2), 2), round(float(point[1]+size[1]/2), 2)]
            is_prediction = point is not None and i in self._predicted_indices and not s.geometry_observed
            if not s.geometry_observed and not is_prediction:
                tracking_box = None
            formal_slot_id = s.slot_id if self.points is not None else None
            slots.append({"group_id": self.group_id, "slot_id": formal_slot_id,
                "formal_slot_id": formal_slot_id, "internal_slot_index": i+1,
                "byte_track_id": track_ids[i] if i < len(track_ids) else None,
                "center_xy": None if point is None else [round(float(v),2) for v in point],
                "state": s.state, "observation": s.observation, "last_confirmed_state": s.confirmed,
                "last_confirmed_at": s.evidence_time, "last_seen_at": s.last_seen,
                "confirmed_state_since": s.confirmed_since,
                "confirmed_state_first_confirmed_at": s.confirmed_at,
                "confidence": s.confidence, "reason": s.reason,
                "confirmation_source": s.confirmation_source,
                "geometry_inferred": s.geometry_inferred,
                "position_is_prediction": is_prediction,
                "tracking_box_xyxy": tracking_box,
                "load_contact_pending": s.load_contact_pending,
                "load_contact_count": s.load_contact_count,
                "load_contact_since": s.load_contact_since,
                "load_contact_last_seen": s.load_contact_last_seen,
                "load_contact_released_at": s.load_contact_released_at,
                "load_contact_unknown_since": s.load_contact_unknown_since,
                "load_contact_unknown_count": s.load_contact_unknown_count})
        return {"group_id": self.group_id, "group_assignment": "configured",
                "slot_numbering_status": "provisional" if self.points is None else "final",
                "phase": self.phase, "identity_status": "waiting_initialization" if self.points is None else ("lost" if self.identity_lost else "tracking"),
                "group_transform": None if self.group_transform is None else self.group_transform.tolist(),
                "group_transform_definition": "initial_stable_pose_to_current_image_xy_affine_2x3",
                "transform_valid_this_frame": bool(self.transform_valid_this_frame),
                "transform_fit_error": self.transform_fit_error,
                "transform_support_slot_ids": list(self.transform_support_slot_ids),
                "transform_support_byte_track_ids": list(getattr(self, "transform_support_byte_track_ids", [])),
                "initialization": dict(self.initialization),
                "slots": slots, "order_result": self.order_result(),
                "confirmed_event_sequence": [e["slot_id"] for e in sorted(self.events,
                    key=lambda e: (e["time_upper"],e["time_lower"])) if e["expected_action"]],
                "current_target_count": sum(s.state == target for s in self.slots),
                "last_verified_target_count": sum(s.confirmed == target for s in self.slots),
                "all_four_currently_confirmed": all(s.state == target for s in self.slots),
                "all_four_target_previously_verified": all(s.confirmed == target for s in self.slots),
                "unassigned_slot_detections": self.unmatched}
