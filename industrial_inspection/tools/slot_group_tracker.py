"""单组四槽身份和状态跟踪；跨组计数由ROI分组控制器管理。"""
from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from industrial_inspection.tools.s4_hand_contact import (
    HandContactPolicy, box_overlap_ratio, centered_box, evaluate_s4_contact)
from industrial_inspection.tools.slot_process_config import (
    LOAD_ENTRY_QUARTET_REVIEW_FRAMES, LOAD_ENTRY_QUARTET_REVIEW_SECONDS, LOAD_POST_CONTACT_CONFIRM_FRAMES)
from industrial_inspection.tools.slot_hand_trajectory import LoadHandTrajectory

SLOT_STATES = {"wafer_slot_empty": "empty", "wafer_slot_filled": "filled"}
S4_UNLOAD_REVIEW_SECONDS = .8


def sort_events_by_time_upper(events):
    """仅按动作时间上界升序展示；上界相同则保留原记录顺序。"""
    return sorted(events, key=lambda event: event.get("time_upper", float("inf")))


def _event_slot_orders_by_time_upper(events):
    """上界较小的事件在先；仅上界完全相同的事件保留先后歧义。

    time_lower 可能来自上一轮空满基线，只用于审计，不参与本轮顺序判断。
    每组最多四个事件，因此可以直接枚举上界非递减的排列。
    """
    for order in itertools.permutations(events):
        if all(a["time_upper"] <= b["time_upper"] for a, b in zip(order, order[1:])):
            yield tuple(event["slot_id"] for event in order)


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


def box_containment(a, b):
    """Return the intersection as a fraction of the smaller box area."""
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(
        0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return area / max(min(area_a, area_b), 1e-9)


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


def slot_layout_error(points):
    """Return the normalized diagonal-midpoint error of a plausible layout."""
    if not valid_slot_geometry(points):
        return None
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
    sides = np.linalg.norm(hull-np.roll(hull, -1, axis=0), axis=1)
    spacing = float(sides.min())
    if spacing < 2 or sides.max() > spacing*4:
        return None
    error = float(np.linalg.norm(hull[0]+hull[2]-hull[1]-hull[3])/spacing)
    return error if error <= .55 else None


def infer_fourth_parallelogram_candidates(three_points):
    """返回由三个中心点构造的三个平行四边形第四点候选。

    三点中任意一点都可以作为两条相邻边的公共顶点，因此候选点为
    ``p[j] + p[k] - p[i]``。这里不再假定最长线段是对角线；调用方
    可以根据相机方向或历史几何选择候选点。
    """
    points = np.asarray(three_points, dtype=float)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        return []
    candidates = []
    for opposite in range(3):
        adjacent = [index for index in range(3) if index != opposite]
        missing = points[adjacent[0]] + points[adjacent[1]] - points[opposite]
        four = np.vstack((points, missing))
        if not valid_slot_geometry(four):
            continue
        all_distances = np.linalg.norm(four[:, None]-four[None, :], axis=2)
        if np.min(all_distances[np.triu_indices(4, 1)]) < 2:
            continue
        candidates.append(missing)
    return candidates


def infer_fourth_square_center(three_points):
    """从三点构造候选第四点，并取最右侧候选作为预测 S4。

    函数名保留以兼容已有调用；实际使用的是三种相邻边平行四边形
    候选，而不是“最长边为对角线”的单一假设。
    """
    candidates = infer_fourth_parallelogram_candidates(three_points)
    if not candidates:
        return None
    rightmost_x = max(float(point[0]) for point in candidates)
    rightmost = [point for point in candidates
                 if np.isclose(float(point[0]), rightmost_x, atol=1e-6)]
    if len(rightmost) == 1:
        return rightmost[0]
    # 多个候选恰好同在最右侧时，用四边形的直角程度消除并列；
    # 这不会改变“最右侧优先”，只处理像素坐标相同的对称情况。
    def rectangularity(point):
        hull = cv2.convexHull(np.vstack((np.asarray(three_points, dtype=float), point))
                              .astype(np.float32)).reshape(-1, 2)
        if len(hull) != 4:
            return float("inf")
        vectors = np.roll(hull, -1, axis=0)-hull
        cosines = []
        for index in range(4):
            incoming = vectors[index-1]
            outgoing = vectors[index]
            cosines.append(abs(float(np.dot(incoming, outgoing))) /
                          max(np.linalg.norm(incoming)*np.linalg.norm(outgoing), 1e-9))
        return max(cosines)
    return min(rightmost, key=rectangularity)


@dataclass
class PredictionContact:
    """Contact supported by a prediction, pending a real state observation."""

    since: float
    last_seen: float
    visual_baseline: str | None
    initial_state: str
    frames: int = 0
    pending: bool = False
    released_at: float | None = None
    empty_since: float | None = None
    empty_frames: int = 0
    started_on_held_box: bool = False
    competing_unloads: tuple[tuple[int, float], ...] = ()
    entry_preview_source: str | None = None

    def diagnostic(self):
        result = {"since": self.since, "last_seen": self.last_seen, "frames": self.frames,
                "pending": self.pending, "visual_baseline": self.visual_baseline,
                "initial_state": self.initial_state, "released_at": self.released_at,
                "empty_since": self.empty_since, "empty_frames": self.empty_frames,
                "started_on_held_box": self.started_on_held_box,
                "competing_unloads": self.competing_unloads,
                "status": "waiting_for_visual" if self.pending else "collecting"}
        if self.entry_preview_source:
            result["source"] = self.entry_preview_source
        return result


@dataclass
class EntryLoadPreviewContact:
    """Real-time contact with an unnumbered three-point prediction."""

    anchor_ids: tuple[int, ...]
    layout: np.ndarray
    evidence: PredictionContact


@dataclass
class EntryS4Contact:
    """Unloading evidence tied to a position while the provisional ID can change."""

    track_id: int
    center: np.ndarray
    filled_at: float
    contact_since: float
    last_contact_at: float
    released_at: float | None = None
    candidate_id: int | None = None
    empty_since: float | None = None
    empty_frames: int = 0
    confirmed_at: float | None = None
    formal_track_id: int | None = None


@dataclass
class SlotState:
    slot_id: int
    confirmed: str = "unknown"
    evidence_time: float | None = None
    # 当前已确认状态首次成立的证据；遮挡和同状态复核不能推迟操作时间。
    confirmed_since: float | None = None
    confirmed_at: float | None = None
    confirmed_visual_track_id: int | None = None
    last_seen: float | None = None
    candidate: str = "unknown"
    candidate_since: float | None = None
    candidate_count: int = 0
    candidate_visual_track_id: int | None = None
    state: str = "unknown"
    observation: str = "unknown"
    confidence: float | None = None
    reason: str = "not_detected"
    geometry_observed: bool = False
    geometry_inferred: bool = False
    box_size_wh: tuple[float, float] | None = None
    # 短时漏检时保留可靠槽框的位置及尺寸。
    last_valid_box_xyxy: tuple[float, float, float, float] | None = None
    last_valid_center_xy: tuple[float, float] | None = None
    last_valid_box_at: float | None = None
    last_valid_box_source: str | None = None
    last_valid_box_group_center_xy: tuple[float, float] | None = None
    temporary_contact_box_xyxy: tuple[float, float, float, float] | None = None
    temporary_contact_box_reason: str | None = None
    confirmation_source: str | None = None
    # 字段名保留 load_contact 以兼容现有快照；实际缓存的是上下料通用的
    # 手接触证据，必须在离手后由连续的空/满检测确认状态。
    load_contact_since: float | None = None
    load_contact_last_seen: float | None = None
    load_contact_count: int = 0
    load_contact_pending: bool = False
    load_contact_can_infer: bool = False
    load_contact_released_at: float | None = None
    load_contact_unknown_since: float | None = None
    load_contact_unknown_count: int = 0
    load_contact_prediction_seen_at: float | None = None
    entry_contact_center: np.ndarray | None = None
    entry_contact_filled_frames: int = 0
    entry_contact_saw_filled: bool = False
    prediction_contact: PredictionContact | None = None
    # 空满检测短时中断的审计状态。消失本身不构成操作证据；只有后续
    # 重新看到目标状态并完成确认时，才使用保存的最早状态证据修正事件时间。
    missing_since: float | None = None
    empty_evidence_since: float | None = None
    empty_evidence_last_seen: float | None = None
    pending_visual_event: dict | None = None

    def observe(self, state, now, confidence, reason, confirm_seconds, minimum_frames,
                contact_evidence=None, *, visual_track_id=None):
        self.observation, self.confidence, self.reason = state, confidence, reason
        self.state = "unknown"
        if state == "unknown":
            self.candidate, self.candidate_since, self.candidate_count = "unknown", None, 0
            self.candidate_visual_track_id = None
            return None
        self.last_seen = now
        if self.candidate != state:
            self.candidate, self.candidate_since, self.candidate_count = state, now, 1
            self.candidate_visual_track_id = visual_track_id
        else:
            self.candidate_count += 1
        if self.candidate_count < minimum_frames or now-self.candidate_since+1e-9 < confirm_seconds:
            self.reason = "confirming"
            return None
        previous, previous_time = self.confirmed, self.evidence_time
        if previous != state:
            self.confirmed_since, self.confirmed_at = self.candidate_since, now
            self.confirmed_visual_track_id = self.candidate_visual_track_id
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


def slot_event_order_result(events, clockwise):
    """按时间上界检查环形顺序；同上界先后不明时不给通过结论。"""
    if any(not e["expected_action"] for e in events):
        return "fail"
    sequence = [e["slot_id"] for e in events]
    if len(sequence) != len(set(sequence)) or len(sequence) > 4:
        return "fail"
    if not sequence:
        return "pending"
    legal = [clockwise[i:]+clockwise[:i] for i in range(4)]
    possible = []
    for order in _event_slot_orders_by_time_upper(events):
        # 不完整序列可能漏掉了遮挡槽的事件，按环形子序列检查，不能直接判漏操作。
        possible.append(any(order == tuple(s for s in cycle if s in sequence) for cycle in legal))
    if possible and not any(possible):
        return "fail"
    if len(sequence) == 4 and possible and all(possible):
        return "pass"
    return "pending"


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
                 geometry_initialization_seconds=None,
                 initialization_stability_ratio=.12, initialization_history_seconds=5.,
                 load_hand_contact_enabled=True, load_hand_contact_seconds=.15,
                 load_hand_contact_min_frames=2, load_hand_contact_overlap_ratio=.05,
                  load_hand_contact_competing_slot_enabled=True,
                  load_hand_contact_competing_slot_id=2,
                  load_hand_contact_competing_margin=.10,
                  load_hand_contact_competing_ratio=2.0,
                 s4_contact_prediction_seconds=None,
                 s4_prediction_contact_overlap_ratio=.15,
                 slot2_below_slot3=False,
                 allow_three_point_initialization=None, s4_hand_region=None,
                 unload_entry_numbering_seconds=None, machine_center=None,
                 require_unload_event_before_numbering=False,
                 visual_reversal_review_seconds=0.0, load_trajectory_enabled=False):
        if group_count != 5 or not 1 <= group_id <= group_count:
            raise ValueError("当前流程固定五组，group_id 必须在1..5之间")
        if phase not in ("load", "unload") or sorted(clockwise) != [1, 2, 3, 4]:
            raise ValueError("phase 必须为load/unload；顺时针顺序必须包含槽1..4各一次")
        if not 0 < match_distance_ratio < .5 or confirm_seconds < 0 or minimum_frames < 1 or lost_seconds <= 0:
            raise ValueError("距离比例须在0..0.5之间；确认时间/帧数及丢失时限无效")
        if not 0 <= conflict_margin <= 1 or not 0 < hand_overlap <= 1 or not 0 <= ambiguity_ratio < .5:
            raise ValueError("冲突、遮挡或关联歧义阈值无效")
        if (initialization_seconds < 0 or initialization_frames < 1
                or (geometry_initialization_seconds is not None
                    and geometry_initialization_seconds < 0)):
            raise ValueError("初始化持续时间不能为负，帧数至少为1")
        if (unload_entry_numbering_seconds is not None
                and (not np.isfinite(unload_entry_numbering_seconds)
                     or unload_entry_numbering_seconds <= 0)):
            raise ValueError("下料换组后的编号停稳时间须为正有限数")
        if not 0 < initialization_stability_ratio < .5 or not np.isfinite(initialization_history_seconds) or initialization_history_seconds <= 0:
            raise ValueError("初始化停稳比例须在0..0.5之间，观测缓存秒数须为正有限数")
        if (not np.isfinite(load_hand_contact_seconds) or load_hand_contact_seconds < 0
                or load_hand_contact_min_frames < 1 or int(load_hand_contact_min_frames) != load_hand_contact_min_frames
                or not 0 < load_hand_contact_overlap_ratio <= 1
                or int(load_hand_contact_competing_slot_id) != load_hand_contact_competing_slot_id
                or not 1 <= int(load_hand_contact_competing_slot_id) <= 4
                or not np.isfinite(load_hand_contact_competing_margin)
                or not 0 <= load_hand_contact_competing_margin <= 1
                or not np.isfinite(load_hand_contact_competing_ratio)
                or load_hand_contact_competing_ratio < 1):
            raise ValueError("上料手接触确认参数无效")
        if (s4_contact_prediction_seconds is not None
                and (not np.isfinite(s4_contact_prediction_seconds)
                     or s4_contact_prediction_seconds <= 0)):
            raise ValueError("S4临时接触框有效时长须为正有限数")
        if not 0 < s4_prediction_contact_overlap_ratio <= 1:
            raise ValueError("S4预测接触重叠比例须在(0,1]范围内")
        if not np.isfinite(visual_reversal_review_seconds) or visual_reversal_review_seconds < 0:
            raise ValueError("空满反转复核时间须为非负有限数")
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
        self.group_id, self.phase, self.clockwise = group_id, phase, tuple(clockwise)
        self.visual_reversal_review_seconds = float(visual_reversal_review_seconds)
        self.match_distance_ratio, self.confirm_seconds = match_distance_ratio, confirm_seconds
        self.minimum_frames, self.lost_seconds = minimum_frames, lost_seconds
        self.conflict_margin, self.hand_overlap, self.ambiguity_ratio = conflict_margin, hand_overlap, ambiguity_ratio
        self.slots = [SlotState(i+1) for i in range(4)]
        self.events = []
        self.s4_region_diagnostic = None
        self.s4_contact_diagnostic = None
        self.two_point_hand_contact = None
        self._region_formal_numbering = True
        self._pending_s4_unload = None
        self.s4_inferred_load = None
        self.cancelled_s4_inferred_load = None
        self.last_time = None
        self.last_geometry_time = None
        self.identity_lost = False
        self.unmatched = []
        self._predicted_indices = set()
        self._position_recovery_index = None
        self._position_recovery_candidate = None
        self._load_binding_frames = 0
        self._s4_before_recalculation = None
        self._s4_reference_layouts = ()
        self._load_numbering_reference = None
        self._orbit_unload_reference = None
        self._load_perspective_numbering = None
        self._g1_load_numbering_reference = None
        self._g1_load_reference_inferred_slot_ids = ()
        self._load_corner_history = None
        self._load_verified_empty_slots = set()
        self._missing_load_contact_index = None
        self.initialization_seconds, self.initialization_frames = initialization_seconds, initialization_frames
        self.geometry_initialization_seconds = (initialization_seconds if
            geometry_initialization_seconds is None else geometry_initialization_seconds)
        self.unload_entry_numbering_seconds = unload_entry_numbering_seconds
        # G1下料候选在确认首个真实下料事件前只保留为暂定轨迹，
        # 不把这段过渡位置发布成正式S1-S4。后续组仍按原有停稳规则编号。
        self.require_unload_event_before_numbering = bool(
            require_unload_event_before_numbering)
        self.machine_center = None if machine_center is None else np.asarray(machine_center, dtype=float)
        if self.machine_center is not None and (
                self.machine_center.shape != (2,) or not np.isfinite(self.machine_center).all()):
            raise ValueError("机心必须是两个有限的像素坐标")
        self._machine_center_votes = deque(maxlen=8)
        self.initialization_stability_ratio = initialization_stability_ratio
        self.initialization_history_seconds = initialization_history_seconds
        self.load_hand_contact_enabled = bool(load_hand_contact_enabled)
        self.load_hand_contact_seconds = float(load_hand_contact_seconds)
        self.load_hand_contact_min_frames = int(load_hand_contact_min_frames)
        self.load_hand_contact_overlap_ratio = float(load_hand_contact_overlap_ratio)
        self.load_hand_contact_competing_slot_enabled = bool(load_hand_contact_competing_slot_enabled)
        self.load_hand_contact_competing_slot_id = int(load_hand_contact_competing_slot_id)
        self.load_hand_contact_competing_margin = float(load_hand_contact_competing_margin)
        self.load_hand_contact_competing_ratio = float(load_hand_contact_competing_ratio)
        self.s4_contact_prediction_seconds = float(
            min(lost_seconds, .8) if s4_contact_prediction_seconds is None
            else s4_contact_prediction_seconds)
        self.s4_prediction_contact_overlap_ratio = float(s4_prediction_contact_overlap_ratio)
        self.s4_hand_region = (None if s4_hand_region is None else
                               np.asarray(s4_hand_region, dtype=np.float32))
        self.slot2_below_slot3 = bool(slot2_below_slot3)
        # Preserve the historical unload behavior when callers omit the flag;
        # the main entry point explicitly enables it for both phases.
        self.allow_three_point_initialization = (phase == "unload" if
            allow_three_point_initialization is None else bool(allow_three_point_initialization))
        self._provisional = None
        self._provisional_inferred_slot_ids = set()
        self._entry_points = None
        self._entry_gate = None
        self.three_point_preview = None
        self._entry_load_preview_contact = None
        self._entry_track_ids = []
        self._point_tracking_gate = None
        self._entry_events = []
        self._entry_seen_at = []
        self._entry_support_since = []
        self._entry_support_count = []
        self._entry_inferred_indices = set()
        self.transition_load_track_id = None
        self.transition_load_slot_index = None
        self.s4_load_contact_lookback = None
        self.load_trajectory_enabled = bool(load_trajectory_enabled)
        self.load_trajectory = LoadHandTrajectory()
        self._entry_reference_points = None
        # Once a provisional slot has accumulated enough valid S4 contact
        # evidence, keep that *slot object* as S4 until the evidence is
        # resolved.  The temporary point order can move while a new group is
        # settling; locking by list index would therefore lose the evidence
        # when the geometry is rebuilt.
        self._entry_locked_s4_slot = None
        self._entry_locked_s4_since = None
        self._entry_s4_contact = None
        self._reset_three_point_stability()
        self._entry_canonical_points = None
        self._entry_revision = 0
        self._initial_seen = set()
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        # When G1 waits for its first unload event, preserve the first stable
        # pose/order.  The candidate may move before the event; numbering must
        # still use the pose that was stable before that movement.
        self._pending_numbering_points = None
        self._pending_numbering_order = None
        # 临时四槽集合的短时漏检恢复状态。正式映射前若某个点暂时消失，
        # 不立即把新出现的无关ID当成该槽；先保留原槽的最后真实中心，
        # 在有限窗口内尝试原位换ID恢复。
        self._provisional_missing_since = {}
        self._provisional_last_real_centers = {}
        self._provisional_quarantined_track_ids = set()
        self._provisional_reidentification_events = []
        self.initialization = {"status": "waiting" if centers is None else "ready",
            "method": "yolo_box_centers" if centers is None else "provided_centers",
            "reason": "waiting_for_four_detections" if centers is None else "provided_centers",
            "eligible_slot_count": 0, "stable_frames": 0, "required_slot_count": 4,
            "geometry_stable_seconds": self.geometry_initialization_seconds,
            "initialized_at": None, "initial_centers_xy": None if centers is None else self.points.tolist(),
            "provisional_started_at": None, "geometry_seen_slot_ids": [], "geometry_seen_track_ids": [],
            "inferred_slot_ids": [],
            "buffered_event_count": 0,
            "three_point_stable_frames": 0,
            "three_point_stable_since": None,
            "provisional_quarantined_track_ids": [],
            "provisional_reidentification_events": []}

    def start_incremental_entry(self, match_distance):
        """新组逐点进入；下料阶段三点可推算第四点，几何停稳后映射槽号。"""
        if self.last_time is not None or self.points is not None or not np.isfinite(match_distance) or match_distance <= 0:
            raise ValueError("逐点进入必须在新跟踪器首次更新前启动，匹配距离须为正")
        self._entry_points, self._entry_gate = [], float(match_distance)
        self._entry_track_ids = []
        self._entry_locked_s4_slot = None
        self._entry_locked_s4_since = None
        self._point_tracking_gate = float(match_distance)
        self.initialization.update(method="incremental_yolo_box_centers", reason="waiting_for_new_points")

    def _reset_three_point_stability(self):
        self._three_point_stable_points = None
        self._three_point_stable_since = None
        self._three_point_stable_count = 0
        self._three_point_support = None
        self._three_point_prediction_applied = False

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
        if self._entry_locked_s4_slot is not None:
            # Use object identity rather than dataclass equality: two empty
            # SlotState objects can otherwise compare equal during entry.
            locked_index = next((index for index, slot in enumerate(self.slots)
                                 if slot is self._entry_locked_s4_slot), None)
            if locked_index is None or locked_index >= len(self._entry_points):
                self._clear_entry_s4_lock("temporary_slot_removed")
            elif not self._entry_locked_s4_slot.load_contact_pending:
                # Visual confirmation or an explicit contact cancellation
                # clears the pending evidence and releases the geometry lock.
                self._clear_entry_s4_lock("contact_resolved")
            else:
                return locked_index
        if len(self._entry_inferred_indices) == 1 and self._load_numbering_reference is None:
            # 三点推算阶段明确把最右侧候选定义为S4；不要再次按环序把
            # 预测点改排到其它正式槽位。
            return next(iter(self._entry_inferred_indices))
        pose = (self._entry_reference_points if self._entry_reference_points is not None
                and len(self._entry_reference_points) == 4 else np.asarray(self._entry_points))
        order = self._entry_numbering_order(pose)
        return None if order is None else int(order[3])

    def _clear_entry_s4_lock(self, reason=None):
        """Release the provisional S4 lock after evidence is resolved."""
        self._entry_locked_s4_slot = None
        self._entry_locked_s4_since = None

    def _lock_entry_s4_slot(self, slot_index, now):
        """Pin the SlotState that produced valid provisional S4 evidence."""
        if self._entry_points is None or not 0 <= slot_index < len(self.slots):
            return
        slot = self.slots[slot_index]
        if not slot.load_contact_pending:
            return
        if self._entry_locked_s4_slot is None:
            self._entry_locked_s4_slot = slot
            self._entry_locked_s4_since = float(now)
            if self.phase == "unload":
                slot.entry_contact_center = self._entry_points[slot_index].copy()
            track_id = self._byte_track_id_at(slot_index)
            if (self.phase == "unload" and track_id is not None and slot.confirmed == "filled"
                    and slot.confirmation_source == "yolo" and slot.evidence_time is not None
                    and self._entry_s4_contact is None):
                self._entry_s4_contact = EntryS4Contact(
                    track_id, self._entry_points[slot_index].copy(), slot.evidence_time,
                    slot.load_contact_since, float(now))

    def _unlock_lost_entry_s4(self, slot, slot_index, obj, now):
        """Release a lock only after the locked track is definitely gone."""
        if self._entry_points is None or self._entry_locked_s4_slot is not slot or obj is not None:
            return
        if not 0 <= slot_index < len(self._entry_seen_at):
            return
        seen_at = self._entry_seen_at[slot_index]
        if seen_at is not None and now-float(seen_at) > self.lost_seconds:
            self._clear_entry_s4_lock("track_lost")

    def _entry_s4_lock_snapshot(self):
        """Return serializable diagnostics for the active provisional lock."""
        if self._entry_points is None or self._entry_locked_s4_slot is None:
            return None
        index = next((index for index, slot in enumerate(self.slots)
                      if slot is self._entry_locked_s4_slot), None)
        if index is None or index >= len(self._entry_points):
            return None
        track_id = (self._entry_track_ids[index]
                    if index < len(self._entry_track_ids) else None)
        return {"active": True, "internal_slot_index": index+1,
                "byte_track_id": track_id,
                "locked_since": self._entry_locked_s4_since}

    def _observe_entry_s4_contact(self, candidates, matches, now, enabled, *, all_candidates=None):
        evidence = self._entry_s4_contact
        if evidence is None:
            return
        if evidence.confirmed_at is None and now-evidence.last_contact_at > self.lost_seconds+1e-9:
            self._entry_s4_contact = None
            return
        if evidence.released_at is None:
            return

        gate = self._entry_gate if self._entry_points is not None else self._point_tracking_gate
        provisional = self._entry_points is not None
        layout = np.asarray(self._entry_points) if provisional else self.points
        s4_index = None
        if layout is not None and len(layout) == 4 and valid_slot_geometry(layout):
            s4_index = int(self._numbering_order(layout)[3]) if provisional else 3
        col = matches.get(s4_index)
        candidate = candidates[col] if col is not None else None

        identity = (self.phase, self.group_id)
        other_slots = {id(candidates[index]) for slot_index, index in matches.items()
                       if slot_index != s4_index}
        nearby_count = 0
        track_id = None
        candidate_record = None
        if candidate is not None:
            for obj in candidates if all_candidates is None else all_candidates:
                obj_id = obj.get('track_id')
                if (obj_id is None or id(obj) in other_slots or obj.get('detection_source') == 'local_crop'
                        or np.linalg.norm(np.asarray(obj['center'])-evidence.center) > gate*.5):
                    continue
                record = self.track_store.tracks.get(obj_id) if self.track_store is not None else None
                if record is not None and (record['owner'] not in (None, identity)
                                           or record['slot_id'] not in (None, 4)):
                    continue
                revised = layout.copy()
                revised[s4_index] = obj['center']
                if (not valid_slot_geometry(revised)
                        or (provisional and self._numbering_order(revised)[3] != s4_index)):
                    continue
                nearby_count += 1
                if obj is candidate:
                    track_id, candidate_record = obj_id, record
        slot = self.slots[s4_index] if track_id is not None else None
        if evidence.confirmed_at is not None:
            if (track_id == evidence.candidate_id and slot.state == 'filled'
                    and slot.confirmation_source == 'yolo'):
                self._entry_s4_contact = None
            return
        # Slot observation already rejects conflicts, ROI violations and hand occlusion.
        if (not enabled or nearby_count != 1 or track_id is None or track_id == evidence.track_id
                or (candidate_record is not None and candidate_record['first_seen'] <= evidence.contact_since)
                or evidence.formal_track_id not in (None, track_id) or slot.observation != 'empty'):
            evidence.candidate_id = evidence.empty_since = None
            evidence.empty_frames = 0
            return
        if evidence.candidate_id != track_id:
            evidence.candidate_id = track_id
            evidence.empty_since = float(now)
            evidence.empty_frames = 0
        evidence.empty_frames += 1
        if (evidence.empty_frames >= self.minimum_frames
                and now-evidence.empty_since+1e-9 >= self.confirm_seconds):
            evidence.confirmed_at = float(now)

    def _recover_entry_s4_unload(self, now):
        evidence = self._entry_s4_contact
        if evidence is None:
            return None
        if (evidence.released_at is None or evidence.formal_track_id is None
                or self.byte_track_ids[3] != evidence.formal_track_id
                or evidence.formal_track_id == evidence.track_id
                or evidence.candidate_id not in (None, evidence.formal_track_id)
                or any(event['slot_id'] == 4 and event['action'] == 'unload' for event in self.events)):
            self._entry_s4_contact = None
            return None
        if (self.slots[3].pending_visual_event or {}).get('action') == 'unload':
            return None
        if evidence.confirmed_at is None:
            return None
        self._entry_s4_contact = None
        if self.slots[3].confirmed != 'empty':
            return None
        self._pending_s4_unload = None
        event = {"slot_id": 4, "group_id": self.group_id, "phase": self.phase,
            "byte_track_id": evidence.candidate_id, "contact_byte_track_id": evidence.track_id,
            "from": "filled", "to": "empty", "action": "unload", "expected_action": True,
            "time_lower": evidence.filled_at, "time_upper": evidence.empty_since,
            "confirmed_at": evidence.confirmed_at, "reported_at": float(now),
            "recovered_before_initialization": evidence.confirmed_at < now, "inferred": True,
            "hand_contact_since": evidence.contact_since, "hand_contact_released_at": evidence.released_at,
            "reason": "empty_after_provisional_s4_contact_reassociation"}
        self.events.append(event)
        return event

    def _numbering_order_with_predicted_s4(self, points, predicted_indices=None):
        """编号时保留三点推算点为正式S4。"""
        order = list(self._numbering_order(points))
        predicted_indices = (self._entry_inferred_indices if predicted_indices is None
                             else set(predicted_indices))
        if len(predicted_indices) != 1:
            return np.asarray(order, dtype=int)
        predicted = next(iter(predicted_indices))
        if predicted in order:
            order.remove(predicted)
            order.append(predicted)
        return np.asarray(order, dtype=int)

    @staticmethod
    def _numbering_from_s4(points, s4):
        offsets = points - points.mean(axis=0)
        ring = np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))
        anchor = int(np.flatnonzero(ring == s4)[0])
        return np.roll(ring, 2-anchor)[[0, 3, 1, 2]]

    def _entry_numbering_order(self, points):
        reference_order = self._load_reference_order(points, inferred_indices=self._entry_inferred_indices)
        if self._load_perspective_numbering is not None:
            return reference_order
        return reference_order if reference_order is not None else self._entry_numbering_fallback(points)

    def _entry_numbering_fallback(self, points):
        if self._load_numbering_reference is not None and self._entry_inferred_indices:
            return self._load_three_point_order(points, self._entry_inferred_indices)
        return self._numbering_order_with_predicted_s4(points)

    @staticmethod
    def _match_layout_order(reference, points, *, minimum_score, minimum_gap):
        """Match measured quadrilaterals by the directions of their clockwise edges."""
        if reference is None or not valid_slot_geometry(reference) or not valid_slot_geometry(points):
            return None
        reference_offsets = reference-reference.mean(axis=0)
        current_offsets = points-points.mean(axis=0)
        reference_ring = np.argsort(np.arctan2(reference_offsets[:, 1], reference_offsets[:, 0]))
        current_ring = np.argsort(np.arctan2(current_offsets[:, 1], current_offsets[:, 0]))
        reference_edges = np.roll(reference[reference_ring], -1, axis=0)-reference[reference_ring]
        current_edges = np.roll(points[current_ring], -1, axis=0)-points[current_ring]
        reference_edges /= np.linalg.norm(reference_edges, axis=1, keepdims=True)
        current_edges /= np.linalg.norm(current_edges, axis=1, keepdims=True)
        # Compare cyclic shifts only, so a mirrored match cannot reverse the carrier's slot order.
        scores = [float(np.sum(reference_edges*np.roll(current_edges, shift, axis=0))/4)
                  for shift in range(4)]
        runner_up, best = np.argsort(scores)[-2:]
        if scores[best] < minimum_score or scores[best]-scores[runner_up] < minimum_gap:
            return None
        order = np.empty(4, dtype=int)
        order[reference_ring] = np.roll(current_ring, int(best))
        return order, scores[best]

    def _load_reference_order(self, points, *, inferred_indices=()):
        """Match physical slots using calibration when available, otherwise directed edges."""
        if self.phase != "load":
            return None
        if self._load_perspective_numbering is not None:
            subsequent = self.group_id != 1
            reference = (self._g1_load_numbering_reference if subsequent else self._load_numbering_reference)
            order, diagnostic = self._load_perspective_numbering.match(reference, points,
                inferred_indices=inferred_indices, same_load_pose=subsequent)
            if subsequent:
                diagnostic["reference_inferred_slot_ids"] = list(self._g1_load_reference_inferred_slot_ids)
                diagnostic["reference_measured_support"] = (0 if reference is None
                    else 4-len(self._g1_load_reference_inferred_slot_ids))
            self.initialization["perspective_load_numbering"] = diagnostic
            return order
        reference = self._load_numbering_reference
        matched = self._match_layout_order(reference, points, minimum_score=.65, minimum_gap=.12)
        if matched is None:
            return None
        order, _ = matched
        if self.machine_center is not None:
            unloaded_anchor, _ = self._machine_anchor(reference, use_slot_spacing=True)
            loading_anchor, _ = self._machine_anchor(points, use_slot_spacing=True)
            # Perspective can make adjacent edges look alike across poses. Keep a clear
            # machine-side anchor when the measured unload layout also supports it.
            if unloaded_anchor == 3 and loading_anchor is not None and order[3] != loading_anchor:
                return None
        return order

    def _load_three_point_order(self, points, predicted_indices):
        """Use the same machine-side S4 anchor as unloading for a three-point layout."""
        if self.machine_center is None:
            return self._numbering_order(points)
        predicted = next(iter(predicted_indices))
        distance = np.linalg.norm(points-self.machine_center, axis=1)
        measured = [index for index in np.argsort(distance) if index != predicted]
        if distance[predicted]+2 < distance[measured[0]]:
            anchor = predicted
        elif distance[measured[1]]-distance[measured[0]] >= 2:
            # Prefer a measured point when the projected point is within the distance tolerance.
            anchor = measured[0]
        else:
            return None
        return self._numbering_from_s4(points, anchor)

    def _project_missing_load_point(self, observed, roi_contour):
        """Recover an in-ROI missing point from the same group's numbered unloading layout."""
        reference = self._load_numbering_reference
        alternatives = []
        for assignment in itertools.permutations(range(4), 3):
            source = np.column_stack((reference[list(assignment)], np.ones(3)))
            if np.linalg.matrix_rank(source) != 3:
                continue
            matrix = np.linalg.solve(source, observed)
            u, scales, vh = np.linalg.svd(matrix[:2])
            rotation = u @ vh
            angle = abs(np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0])))
            if np.linalg.det(matrix[:2]) <= 0 or scales.min() < .5 or scales.max() > 2 or angle > 45:
                continue
            missing = next(i for i in range(4) if i not in assignment)
            point = np.append(reference[missing], 1.) @ matrix
            if cv2.pointPolygonTest(roi_contour, tuple(point), False) < 0:
                continue
            layout = np.vstack((observed, point))
            order = self._load_three_point_order(layout, {3})
            labels = np.asarray((*assignment, missing))
            if (valid_slot_geometry(layout) and order is not None
                    and np.array_equal(labels[order], np.arange(4))):
                alternatives.append(point)
        return alternatives[0] if len(alternatives) == 1 else None

    def _infer_entry_s4(self, three_points, roi_contour, *, support_track_ids=None):
        """Project previous measured layouts onto the current three visible slots."""
        if self.phase == "load" and self._load_perspective_numbering is not None:
            subsequent = self.group_id > 1
            reference = self._g1_load_numbering_reference if subsequent else self._load_numbering_reference
            ids = tuple(support_track_ids) if support_track_ids is not None else ()
            usable_ids = len(ids) == 3 and None not in ids and len(set(ids)) == 3
            previous = self._load_corner_history
            preferred = None
            if usable_ids and previous is not None and self.last_time is not None:
                if (set(ids) == set(previous["roles"])
                        and 0 <= self.last_time-previous["at"] <= self.s4_contact_prediction_seconds):
                    preferred = [previous["roles"][tid] for tid in ids]
                else:
                    self._load_corner_history = None
            point, diagnostic = self._load_perspective_numbering.missing_load_point(
                reference, three_points, roi_contour, same_load_pose=subsequent,
                preferred_slot_ids=preferred)
            if (point is not None and usable_ids and self.last_time is not None
                    and diagnostic.get("selection") != "previous_verified_roles"):
                self._load_corner_history = {"at": float(self.last_time),
                    "roles": dict(zip(ids, diagnostic["slot_ids_in_observation_order"]))}
            self.initialization["three_point_template_prediction"] = diagnostic
            return point, [1 if subsequent else self.group_id] if point is not None else []
        fallback = infer_fourth_square_center(three_points)
        if (self._load_numbering_reference is not None and roi_contour is not None
                and (fallback is None or cv2.pointPolygonTest(roi_contour, tuple(fallback), False) < 0)):
            projected = self._project_missing_load_point(three_points, roi_contour)
            return projected, [self.group_id] if projected is not None else []
        if fallback is None or len(self._s4_reference_layouts) < 2:
            return fallback, []
        layout = np.vstack((three_points, fallback))
        order = self._numbering_order_with_predicted_s4(layout, {3})
        target = layout[order[:3]]
        projected, group_ids = [], []
        for group_id, reference in self._s4_reference_layouts:
            source = np.column_stack((reference[:3], np.ones(3)))
            matrix = np.linalg.solve(source, target)
            singular = np.linalg.svd(matrix[:2], compute_uv=False)
            if np.linalg.det(matrix[:2]) <= 0 or singular.min() < .5 or singular.max() > 2:
                continue
            point = reference[3] @ matrix[:2] + matrix[2]
            if (np.linalg.norm(point-fallback) > self._entry_gate*1.5
                    or roi_contour is not None and cv2.pointPolygonTest(
                        roi_contour, tuple(point), False) < 0):
                continue
            projected.append(point)
            group_ids.append(group_id)
        if len(projected) < 2:
            return fallback, []
        # 两组以上独立实测参考必须支持近似相同的位置，离群参考不参与推算。
        projected = np.asarray(projected)
        median = np.median(projected, axis=0)
        agree = np.linalg.norm(projected-median, axis=1) <= self._entry_gate*.2
        if np.count_nonzero(agree) < 2:
            return fallback, []
        layout[3] = np.median(projected[agree], axis=0)
        if (not valid_slot_geometry(layout)
                or not np.array_equal(self._numbering_order_with_predicted_s4(layout, {3}), order)):
            return fallback, []
        return layout[3], [group_id for group_id, accepted in zip(group_ids, agree) if accepted]

    def _preview_three_point_box(self, candidates, matches, hands, roi_contour, *, ignored_cols=(),
                                 load_support_resolver=None):
        """Build an unnumbered overlay from the three current reliable detections."""
        if not self.allow_three_point_initialization:
            return None
        support = [(col, obj) for col, obj in enumerate(candidates)
                   if col not in ignored_cols and not obj["conflict"] and obj.get("detection_source") != "local_crop"
                   and not self._occluded(obj, hands) and (roi_contour is None or
                       cv2.pointPolygonTest(roi_contour, tuple(obj["center"]), False) >= 0)]
        if not {col for col, _ in support}.issubset(matches.values()):
            return None
        local_support = None
        if len(support) == 2 and self.phase == "load" and load_support_resolver is not None:
            local_support = load_support_resolver(self, [obj for _, obj in support], hands, candidates)
            if local_support is not None:
                support.append((None, local_support))
        if len(support) != 3:
            return None
        tracks = [obj["track_id"] for _, obj in support if obj.get("track_id") is not None]
        if len(tracks) != len(set(tracks)):
            return None
        points = np.asarray([obj["center"] for _, obj in support], dtype=float)
        point = (infer_fourth_square_center(points) if self._entry_gate is None
                 else self._infer_entry_s4(points, roi_contour, support_track_ids=tracks)[0])
        if (point is None or slot_layout_error(np.vstack((points, point))) is None
                or (roi_contour is not None and cv2.pointPolygonTest(roi_contour, tuple(point), False) < 0)):
            return None
        boxes = np.asarray([obj["xyxy"] for _, obj in support], dtype=float)
        half_size = np.median(boxes[:, 2:]-boxes[:, :2], axis=0)/2
        preview = {"group_id": self.group_id, "phase": self.phase, "center_xy": point.round(2).tolist(),
                   "tracking_box_xyxy": np.r_[point-half_size, point+half_size].round(2).tolist(),
                   "support_byte_track_ids": sorted(tracks)}
        if local_support is not None:
            preview["local_support_box_xyxy"] = local_support["xyxy"]
        if self.phase == "load" and self.load_hand_contact_enabled and len(tracks) == 3:
            order = np.argsort(tracks)
            layout = np.vstack((points[order], point))
            sizes = (boxes[:, 2:]-boxes[:, :2])[order]
            preview["hand_contact"] = self._observe_load_preview_contact(
                tuple(sorted(tracks)), layout, np.vstack((sizes, half_size*2)), hands, roi_contour)
        return preview

    @staticmethod
    def _same_load_preview_pose(old, new):
        matrix, _ = cv2.estimateAffinePartial2D(old[:3], new[:3])
        if matrix is None:
            return False
        scale = np.linalg.norm(matrix[:, 0])
        spacing = np.min(np.linalg.norm(new[:, None]-new[None, :], axis=2)[np.triu_indices(4, 1)])
        return (.8 <= scale <= 1.2 and
                np.max(np.linalg.norm(old @ matrix[:, :2].T+matrix[:, 2]-new, axis=1)) <= .2*spacing)

    def _observe_load_preview_contact(self, anchor_ids, layout, sizes, hands, roi_contour):
        now = self.last_time
        hands = [hand for hand in hands if roi_contour is None or cv2.pointPolygonTest(
            roi_contour, tuple(np.asarray(hand["xyxy"]).reshape(2, 2).mean(axis=0)), False) >= 0]
        policy = HandContactPolicy(self.s4_prediction_contact_overlap_ratio, False, -1, 0., 0.)
        overlap = evaluate_s4_contact(layout, sizes, 3, hands, policy)
        hit = overlap.accepted and overlap.overlap_ratio >= policy.overlap_ratio
        record = self._entry_load_preview_contact
        if record is not None and (record.anchor_ids != anchor_ids
                or now-record.evidence.last_seen > self.s4_contact_prediction_seconds
                or not self._same_load_preview_pose(record.layout, layout)):
            self._entry_load_preview_contact = record = None
        if hit:
            if record is None or record.evidence.released_at is not None:
                evidence = PredictionContact(now, now, "empty", "empty", entry_preview_source="three_point_preview")
                self._entry_load_preview_contact = record = EntryLoadPreviewContact(anchor_ids, layout, evidence)
            evidence = record.evidence
            record.layout = layout
            evidence.last_seen = now
            evidence.frames += 1
            evidence.pending = (evidence.frames >= self.load_hand_contact_min_frames
                and now-evidence.since+1e-9 >= self.load_hand_contact_seconds)
        elif record is not None:
            if not record.evidence.pending:
                self._entry_load_preview_contact = record = None
            elif overlap.overlap_ratio < policy.overlap_ratio and record.evidence.released_at is None:
                record.evidence.released_at = now
        evidence = record.evidence if record is not None else None
        return {"accepted": bool(hit), "overlap_ratio": overlap.overlap_ratio,
                "frames": evidence.frames if evidence is not None else 0,
                "pending": bool(evidence is not None and evidence.pending),
                "released_at": evidence.released_at if evidence is not None else None}

    def adopt_entry_load_preview_contact(self, now):
        """Attach preview evidence to the uniquely corresponding inferred slot."""
        record, self._entry_load_preview_contact = self._entry_load_preview_contact, None
        target = self.inferred_load_contact_index()
        if record is None or not record.evidence.pending or target is None or self.identity_lost:
            return False
        slot = self.slots[target]
        evidence = record.evidence
        if (now-evidence.last_seen > self.s4_contact_prediction_seconds
                or target+1 not in self._load_verified_empty_slots or slot.confirmed != "empty"
                or slot.evidence_time is None or slot.evidence_time > evidence.since
                or slot.pending_visual_event is not None or any(e["slot_id"] == slot.slot_id for e in self.events)):
            return False
        anchors = [self.byte_track_ids.index(track_id) for track_id in record.anchor_ids
                   if track_id in self.byte_track_ids]
        if set(anchors) != set(range(4))-{target}:
            return False
        if not self._same_load_preview_pose(record.layout, self.reference_points[[*anchors, target]]):
            return False
        slot.prediction_contact = evidence
        return True

    def review_entry_two_point_load(self, slot, contact):
        """Keep a resolved preview contact pending until the same slot is visually filled."""
        slot.prediction_contact = PredictionContact(contact.since, contact.released_at, "empty", "empty",
            frames=contact.frames, pending=True, released_at=contact.released_at,
            entry_preview_source="two_point_preview")

    def _observe_machine_s4(self, observed, now, enabled):
        if self.machine_center is None:
            return
        if not enabled:
            self._machine_center_votes.clear()
            return
        if len(observed) != 4 or any(obj.get("detection_source") == "local_crop"
                                    or obj["conflict"] for obj in observed):
            return
        track_ids = [obj.get("track_id") for obj in observed]
        if len(set(track_ids) - {None}) != 4:
            return
        points = np.asarray([obj["center"] for obj in observed], dtype=float)
        if not valid_slot_geometry(points):
            return
        nearest, gap = self._machine_anchor(points, use_slot_spacing=self.phase == "unload")
        self._machine_center_votes.append((now, frozenset(track_ids),
            None if nearest is None else track_ids[nearest], gap))

    def _machine_anchor(self, points, *, use_slot_spacing):
        distance = np.linalg.norm(points-self.machine_center, axis=1)
        first, second = np.argsort(distance)[:2]
        gap = float(distance[second]-distance[first])
        minimum_gap = 2.
        # A small lead relative to slot spacing can reverse as a detection box drifts.
        if use_slot_spacing:
            spacing = np.min(np.linalg.norm(points[:, None]-points[None, :], axis=2)[np.triu_indices(4, 1)])
            minimum_gap = max(minimum_gap, .10*spacing)
        return (int(first) if gap >= minimum_gap else None), gap

    @staticmethod
    def _numbering_conflicts_with_s4_contact(s4, slots, events):
        return (any(slot.load_contact_pending and i != s4 for i, slot in enumerate(slots))
                or any(e.get("inferred") and e["slot_id"] != s4+1 for e in events))

    def _final_numbering_order(self, points, track_ids, legacy, now, *, inferred, slots, events):
        order = self._machine_numbering_order(points, track_ids, legacy, now,
                                               inferred=inferred, slots=slots, events=events)
        if self.phase != "unload" or self.group_id == 1 or inferred or len(set(track_ids)-{None}) != 4:
            return order
        reference = next((layout for group_id, layout in self._s4_reference_layouts if group_id == 1), None)
        if reference is None:
            return order
        diagnostic = self.initialization.get("machine_center_numbering", {})
        matched = self._match_layout_order(reference, points, minimum_score=.8, minimum_gap=.2)
        if matched is not None:
            proposed, score = matched
            spacing = np.min(np.linalg.norm(points[:, None]-points[None, :], axis=2)
                             [np.triu_indices(4, 1)])
            if not self.slot2_below_slot3 or points[proposed[1], 1]-points[proposed[2], 1] >= -.2*spacing:
                s4 = int(proposed[3])
                if not self._numbering_conflicts_with_s4_contact(s4, slots, events):
                    self.initialization["machine_center_numbering"] = {
                        "status": "matched_g1_stable_layout", "reference_group_id": 1,
                        "s4_byte_track_id": track_ids[s4], "layout_score": score,
                        "machine_s4_byte_track_id": diagnostic.get("s4_byte_track_id"),
                    }
                    return proposed
        if diagnostic.get("status") in ("confirmed", "contact_evidence_fallback") or self.machine_center is None:
            return order
        self.initialization["machine_center_numbering"] = {
            **diagnostic, "status": "waiting_for_g1_layout_match", "reference_group_id": 1,
        }
        return None

    def _machine_numbering_order(self, points, track_ids, legacy, now, *, inferred, slots, events):
        if self.machine_center is None:
            return legacy
        diagnostic = {"status": "insufficient_distance_evidence"}
        self.initialization["machine_center_numbering"] = diagnostic
        if inferred:
            diagnostic["status"] = "three_point_fallback"
            return legacy
        votes = [vote for vote in self._machine_center_votes
                 if now-vote[0] <= .25 and vote[1] == frozenset(track_ids)][-3:]
        if len(votes) < 3 or votes[-1][2] is None or any(v[2] != votes[-1][2] for v in votes):
            return legacy
        s4 = track_ids.index(votes[-1][2])
        if s4 != int(np.argmin(np.linalg.norm(points - self.machine_center, axis=1))):
            return legacy
        order = self._numbering_from_s4(points, s4)
        spacing = np.min(np.linalg.norm(points[:, None]-points[None, :], axis=2)[np.triu_indices(4, 1)])
        if self.slot2_below_slot3 and points[order[1], 1]-points[order[2], 1] < -.2*spacing:
            diagnostic["status"] = "position_constraint_fallback"
            return legacy
        if self._numbering_conflicts_with_s4_contact(s4, slots, events):
            diagnostic["status"] = "contact_evidence_fallback"
            return legacy
        diagnostic.update(status="confirmed", s4_byte_track_id=track_ids[s4],
                          minimum_gap_px=min(v[3] for v in votes))
        return order

    def _s4_index(self):
        """Return the current point-array index for formal S4, if known."""
        return self._entry_s4_index() if self._entry_points is not None else 3

    def inferred_load_contact_index(self):
        """Use the slot inferred at formal numbering, even after it binds a track."""
        if self.phase != "load" or self.points is None or not self.load_hand_contact_enabled:
            return None
        inferred = self.initialization["inferred_slot_ids"]
        return inferred[0]-1 if len(inferred) == 1 else None

    def inferred_load_contact_indices(self):
        if self.phase != "load" or self.points is None or not self.load_hand_contact_enabled:
            return ()
        return tuple(slot_id-1 for slot_id in self.initialization["inferred_slot_ids"])

    def _orbit_contact_enabled(self):
        contact = self.two_point_hand_contact
        return self.points is not None and contact is not None and contact.uses_orbit(self)

    def _is_hand_contact_slot(self, slot_index):
        if self._orbit_contact_enabled():
            return slot_index in self.inferred_load_contact_indices()
        return slot_index == self._hand_contact_index()

    def _predicted_load_is_locked(self, slot):
        """Use the formal inferred role even after the slot acquires a real track."""
        if (self.phase != "load" or self.points is None or not self._region_formal_numbering
                or slot.slot_id not in self.initialization["inferred_slot_ids"]
                or slot.confirmed != "filled"):
            return False
        pending = slot.pending_visual_event
        return ((pending is not None and pending["action"] == "load")
                or any(event["slot_id"] == slot.slot_id and event.get("action") == "load"
                       for event in self.events))

    def _hand_contact_index(self):
        if self.phase != "load":
            return self._s4_index()
        index = self.inferred_load_contact_index()
        if index is None or index == 3:
            return index
        slot = self.slots[index]
        if (slot.load_contact_pending or self._missing_load_contact_index == index
                or (slot.geometry_inferred and index+1 in self._load_verified_empty_slots
                    and slot.confirmed == "empty")):
            return index
        return None

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
        self._initial_seen = {mapping[i+1]-1 for i in self._initial_seen if i+1 in mapping}
        self.slots = [self.slots[i] for i in keep]
        self.slots.extend(SlotState(i+1) for i in range(len(keep), 4))
        for i, slot in enumerate(self.slots):
            slot.slot_id = i+1
        if (self._entry_locked_s4_slot is not None
                and not any(slot is self._entry_locked_s4_slot for slot in self.slots)):
            self._clear_entry_s4_lock("track_lost")
        self._entry_events = [e for e in self._entry_events if e["slot_id"] in mapping]
        for event in self._entry_events:
            event["slot_id"] = mapping[event["slot_id"]]
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        self._entry_reference_points = None
        self._s4_before_recalculation = None
        self._reset_three_point_stability()
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

    def _update_incremental_entry(self, objects, now, enabled, roi_contour, *, load_support_resolver=None,
                                  entry_position_candidates=()):
        numbering_seconds = self.unload_entry_numbering_seconds if self.phase == "unload" else None
        self._predicted_indices.clear()
        self._expire_entry_tracks(now)
        self._predicted_indices.update(self._entry_inferred_indices)
        candidates = self._candidates(objects)
        all_candidates = candidates
        hands = [o for o in objects if o["class_name"] == "hand"]
        quartet, waiting_for_quartet = self._review_load_entry_quartet(
            candidates, hands, now, enabled, entry_position_candidates)
        if quartet is not None:
            self._rebuild_entry_geometry(quartet, now)
            self._initial_seen.update(range(4))
            self._entry_load_preview_contact = None
        elif enabled and not waiting_for_quartet:
            candidates = self._reconcile_entry_geometry(candidates, hands, now)
        matches = {}
        # 只按上一帧真实中心作一对一距离匹配；新点按首次建立顺序编号，
        # 同帧多个新点按中心(x,y)排序；临时点号不代表最终S号。
        if self._entry_points and candidates:
            distances = np.linalg.norm(np.asarray(self._entry_points)[:, None]
                -np.asarray([c["center"] for c in candidates])[None, :], axis=2)
            matches = self._associate_entry_distances(distances, candidates)
            if waiting_for_quartet:
                matches = {i: j for i, j in matches.items() if i not in self._entry_inferred_indices}
        new_indices = [j for j in range(len(candidates)) if j not in matches.values()]
        # 一个已知点附近的重复/歧义框不能另开一个点号；多于四槽时也不选高分四框。
        new_indices = [j for j in new_indices if not self._entry_points or
            min(np.linalg.norm(np.asarray(candidates[j]["center"])-point)
                for point in self._entry_points) > self._entry_gate]
        # 三点推算只是S4暂时不可见时的中心估计。若后续出现落在该估计附近、
        # 且能形成有效四槽几何的真实框，用它逐帧校正推算点，而非把组判成超额候选。
        if (not waiting_for_quartet and self.allow_three_point_initialization
                and len(self._entry_points) == 4 and self._entry_inferred_indices):
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
                matches[i] = j
        reliable = [i for i, col in matches.items() if not candidates[col]["conflict"]
                    and not self._occluded(candidates[col], hands)]
        if enabled and not overflow:
            self.three_point_preview = self._preview_three_point_box(
                candidates, matches, hands, roi_contour, load_support_resolver=load_support_resolver)
        inferred_s4 = next(iter(self._entry_inferred_indices)) if len(self._entry_inferred_indices) == 1 else None
        can_refresh_s4 = (enabled and len(self._entry_points) == 4
                          and inferred_s4 is not None and inferred_s4 not in matches
                          and len(candidates) <= 4 and all(
                              candidates[matches[i]].get("detection_source") != "local_crop" for i in reliable))
        three_point_ready = False
        if (self.allow_three_point_initialization and len(reliable) == 3
                and (len(self._entry_points) == 3 or can_refresh_s4)):
            support = tuple((i, candidates[matches[i]].get("track_id")) for i in sorted(reliable))
            observed = np.asarray([candidates[matches[i]]["center"] for i, _ in support], dtype=float)
            if np.isfinite(observed).all():
                spacing = np.min(np.linalg.norm(observed[:, None]-observed[None, :], axis=2)
                    [np.triu_indices(3, 1)])
                stable_delta = (float("inf") if self._three_point_stable_points is None else
                    np.max(np.linalg.norm(observed-self._three_point_stable_points, axis=1)))
                if (self._three_point_support != support
                        or stable_delta > spacing*self.initialization_stability_ratio):
                    self._three_point_support = support
                    self._three_point_stable_since = float(now)
                    self._three_point_stable_count = 0
                    self._three_point_prediction_applied = False
                # 重算时用本轮起点判断停稳，避免连续小幅移动累计成明显位移。
                if not can_refresh_s4 or self._three_point_stable_count == 0:
                    self._three_point_stable_points = observed
                self._three_point_stable_count += 1
                three_point_ready = (not self._three_point_prediction_applied
                    and self._three_point_stable_count >= self.initialization_frames
                    and now-self._three_point_stable_since+1e-9 >= self.geometry_initialization_seconds)
            else:
                self._reset_three_point_stability()
        else:
            self._reset_three_point_stability()
        inferred_now = False
        refreshed_s4 = False
        if len(self._entry_points) == 3 and three_point_ready:
            inferred, reference_groups = self._infer_entry_s4(
                observed, roi_contour, support_track_ids=[tid for _, tid in support])
            if inferred is not None:
                index = len(self._entry_points)
                self.initialization.update(
                    s4_initial_position_source=("previous_measured_layouts" if reference_groups else "three_point_geometry"),
                    s4_reference_group_ids=reference_groups)
                self._entry_points.append(inferred)
                self._entry_track_ids.append(None)
                self._entry_seen_at.append(float(now))
                self._entry_support_since.append(None)
                self._entry_support_count.append(0)
                self._entry_inferred_indices.add(index)
                self._predicted_indices.add(index)
                self.slots[index].geometry_inferred = True
                self._three_point_prediction_applied = True
                inferred_now = True
        elif three_point_ready and not (self._entry_locked_s4_slot is not None
                                       and self._entry_locked_s4_slot.load_contact_pending):
            # 重新停稳后用当前三点重算一次，避免把遮挡前的定位偏差继续传递。
            refreshed, _ = self._infer_entry_s4(
                observed, roi_contour, support_track_ids=[tid for _, tid in support])
            if refreshed is not None and (roi_contour is None or cv2.pointPolygonTest(
                    roi_contour, tuple(refreshed), False) >= 0):
                if self._s4_before_recalculation is None:
                    self._s4_before_recalculation = self._entry_points[inferred_s4].copy()
                self._entry_points[inferred_s4] = refreshed
                self._three_point_prediction_applied = True
                self.initialization["s4_last_recalculated_at"] = float(now)
                refreshed_s4 = True
        if inferred_now:
            # 三个真实点已经完成停稳确认；加入虚拟S4后沿用这段稳定证据，
            # 避免因为数组从3点变4点而无意义地重新等待一轮稳定帧。
            self._initial_points = np.asarray(self._entry_points, dtype=float).copy()
            self._initial_since = self._three_point_stable_since
            self._initial_count = self._three_point_stable_count
        self._remember_track_ids(matches, candidates, self._entry_track_ids)
        if self.allow_three_point_initialization and self._entry_inferred_indices and len(candidates) <= 4:
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
                    s4_index = inferred_s4 if inferred_s4 is not None else self._entry_s4_index()
                    if s4_index is not None and s4_index not in reliable:
                        if self._s4_before_recalculation is not None:
                            self._s4_before_recalculation = (
                                self._s4_before_recalculation @ matrix[:, :2].T + matrix[:, 2])
                        if not refreshed_s4:
                            self._entry_points[s4_index] = predicted[s4_index]
                        self._predicted_indices.add(s4_index)
        if inferred_s4 in reliable:
            self._s4_before_recalculation = None
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
        cutoff = now-self.initialization_history_seconds
        for i, slot in enumerate(self.slots):
            if slot.evidence_time is not None and slot.evidence_time < cutoff:
                slot.confirmed, slot.evidence_time = "unknown", None
                slot.confirmed_since, slot.confirmed_at = None, None
                slot.confirmation_source = None
            obj = candidates[matches[i]] if i in matches else None
            event, history = self._observe_slot(slot, obj, hands, now, enabled and not overflow, roi_contour)
            self._entry_events.extend(self._review_visual_events(slot, history, event, now))
        self._observe_entry_s4_contact(
            candidates, matches, now, enabled and not overflow, all_candidates=all_candidates)
        # 已完成的状态变化随有效轨迹保留到正式编号；不能按旧状态证据的
        # 年龄清掉已确认事件。轨迹失效时由_expire_entry_tracks连同身份清理。
        self.unmatched = [c for j, c in enumerate(candidates) if j not in matches.values()]
        self._observe_machine_s4([candidates[col] for col in matches.values()], now, enabled and not overflow)
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
        stable = (quartet is not None or
                  (self._initial_count >= self.initialization_frames and self._initial_since is not None
                   and now-self._initial_since+1e-9 >= self.geometry_initialization_seconds))
        if numbering_seconds is not None and enabled and inferred_now:
            # 先用推算的 S4 记录接触，正式编号从当前姿态重新计时。
            self._initial_since, self._initial_count = float(now), 1
            stable = False
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
            reason=("too_many_slot_detections" if overflow else "collecting_slot_tracks"),
            eligible_slot_count=len(candidates),
            provisional_started_at=float(now) if started is None and self._entry_points else started,
            geometry_seen_slot_ids=[],
            geometry_seen_track_ids=[self._entry_track_ids[i] for i in sorted(self._initial_seen)
                                     if self._entry_track_ids[i] is not None],
            inferred_slot_ids=sorted(i+1 for i in self._entry_inferred_indices),
            required_slot_count=3 if self._entry_inferred_indices else 4,
            buffered_event_count=len(self._entry_events), stable_frames=self._initial_count,
            three_point_stable_frames=self._three_point_stable_count,
            three_point_stable_since=self._three_point_stable_since)
        recovered = []
        required_seen = 3 if self._entry_inferred_indices else 4
        if len(self._entry_points) == 4 and len(self._initial_seen) >= required_seen and not overflow:
            if waiting_for_quartet:
                self.initialization["reason"] = "reviewing_four_real_slots"
                return {**self.snapshot(), "new_events": []}
            # 临时点号可以按任意顺序出现；状态和事件随点一起映射，不丢掉早期满槽基线。
            try:
                points = validate_centers(self._entry_points)
            except ValueError:
                self.initialization.update(reason="invalid_initial_geometry")
                return {**self.snapshot(), "new_events": []}
            self.initialization.update(reason="confirming_group_geometry", stable_frames=self._initial_count)
            if numbering_seconds is not None and (
                    not stable or now-self._initial_since+1e-9 < numbering_seconds):
                return {**self.snapshot(), "new_events": []}
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
            reference_order = self._load_reference_order(pose, inferred_indices=self._entry_inferred_indices)
            order = (reference_order if self._load_perspective_numbering is not None
                     or reference_order is not None else self._entry_numbering_fallback(pose))
            if order is None:
                self.initialization.update(reason="ambiguous_s4_anchor")
                return {**self.snapshot(), "new_events": []}
            if reference_order is not None:
                self.initialization["machine_center_numbering"] = {"status": (
                    "matched_calibrated_load_layout" if self._load_perspective_numbering is not None
                    else "matched_unload_layout")}
            elif self._load_numbering_reference is not None and self._entry_inferred_indices:
                self.initialization["machine_center_numbering"] = {"status": "load_three_point_anchor"}
            elif stable:
                order = self._final_numbering_order(
                    points, self._entry_track_ids, order, now, inferred=bool(self._entry_inferred_indices),
                    slots=self.slots, events=self._entry_events)
                if order is None:
                    self.initialization.update(reason="waiting_for_numbering_pose")
                    return {**self.snapshot(), "new_events": []}
            elif self.machine_center is not None:
                self.initialization["machine_center_numbering"] = {"status": "current_pose_not_stable"}
            if self.slot2_below_slot3 and (not valid_slot_geometry(numbering_points)
                    or (reference_order is None and pose[order[1], 1]-pose[order[2], 1] < -spacing*.2)):
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
                inferred_canonical, _ = self._infer_entry_s4(canonical_entry, roi_contour)
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
            self._seed_formal_s4_detection_cache(now)
            self._temporary_s4_contact_box(self.slots[3], 3, now)
            self.last_geometry_time = now
            self.events = [event for event in self._entry_events
                           if not (self.transition_load_track_id is not None
                                   and event.get("byte_track_id") == self.transition_load_track_id
                                   and event["action"] == "load")]
            for event in self.events:
                event.update(slot_id=mapping[event["slot_id"]], reported_at=float(now),
                    recovered_before_initialization=event["confirmed_at"] < now)
            if self._entry_s4_contact is not None:
                self._entry_s4_contact.formal_track_id = self.byte_track_ids[3]
            self._recover_entry_s4_unload(now)
            recovered = list(self.events)
            self._entry_points = None
            self._entry_track_ids = []
            self._entry_events = []
            self._entry_inferred_indices = set()
            self._entry_canonical_points = None
            self._clear_entry_s4_lock("formal_numbering")
            reason = ("load_entry_four_real_slots_verified" if quartet is not None
                      else "four_slot_tracks_stable" if stable else "stable_reference_numbering")
            self.initialization.update(status="ready", reason=reason,
                initialized_at=float(now), initial_centers_xy=self.points.tolist(),
                numbering_reference_centers_xy=numbering_points[order].tolist(),
                geometry_seen_slot_ids=sorted(int(mapping[i+1]) for i in self._initial_seen),
                geometry_seen_track_ids=[self.byte_track_ids[mapping[i+1]-1]
                                         for i in sorted(self._initial_seen)
                                         if self.byte_track_ids[mapping[i+1]-1] is not None],
                inferred_slot_ids=inferred_slot_ids,
                required_slot_count=3 if inferred_slot_ids else 4)
        return {**self.snapshot(), "new_events": recovered}

    def _try_two_point_load_numbering(self, now):
        contact = self.two_point_hand_contact
        if (self.phase != "load" or self.points is not None or self._load_perspective_numbering is None
                or contact is None or contact.numbering_order is None or contact.prediction is None
                or contact.prediction.measured_at != now or self._entry_points is None
                or len(self._entry_points) != 2 or len(self._initial_seen) != 2
                or set(self._entry_track_ids) != set(contact.pair)):
            return None
        roles = contact.numbering_order
        order = np.argsort(roles)
        layout = contact.prediction.layout
        old_slots = dict(zip(self._entry_track_ids, self.slots[:2]))
        mapping = {i+1: roles[contact.pair.index(track_id)]+1
                   for i, track_id in enumerate(self._entry_track_ids)}
        slots, ids = [], []
        for slot_index, preview_index in enumerate(order):
            track_id = contact.pair[preview_index] if preview_index < 2 else None
            slot = old_slots[track_id] if track_id is not None else SlotState(slot_index+1)
            slot.slot_id = slot_index+1
            slot.geometry_inferred = track_id is None
            slots.append(slot)
            ids.append(track_id)
        self.slots, self.byte_track_ids = slots, ids
        self.points = layout[order].copy()
        self.reference_points, self.canonical_points = self.points.copy(), self.points.copy()
        self.group_transform = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.transform_valid_this_frame, self.transform_fit_error = False, None
        self.transform_support_slot_ids = [roles[i]+1 for i in range(2)]
        self.transform_support_byte_track_ids = list(contact.pair)
        self._predicted_indices = {roles[i] for i in (2, 3)}
        self.last_geometry_time = now
        self.events = self._entry_events
        for event in self.events:
            event.update(slot_id=mapping[event['slot_id']], reported_at=float(now),
                         recovered_before_initialization=event['confirmed_at'] < now)
        self._entry_points = None
        self._entry_track_ids, self._entry_events = [], []
        self._entry_inferred_indices = set()
        self._entry_canonical_points = None
        self._clear_entry_s4_lock("formal_two_point_numbering")
        self.initialization.update(status="ready", reason="stable_two_point_template_numbering",
            initialized_at=float(now), initial_centers_xy=self.points.tolist(),
            numbering_reference_centers_xy=self.points.tolist(),
            geometry_seen_slot_ids=sorted(roles[i]+1 for i in range(2)),
            geometry_seen_track_ids=list(contact.pair), inferred_slot_ids=sorted(i+1 for i in self._predicted_indices),
            required_slot_count=2, machine_center_numbering={"status": "matched_calibrated_two_point_layout"})
        self.initialization["perspective_load_numbering"] = dict(contact.numbering_diagnostic)
        contact.confirm_numbering()
        return list(self.events)

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
            error = slot_layout_error([obj['center'] for obj in subset])
            if error is not None:
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

    def _review_load_entry_quartet(self, candidates, hands, now, enabled, position_candidates):
        """Verify a late fourth track against the measured unload layout before numbering."""
        if (self.phase != "load" or self._load_numbering_reference is None
                or len(self._entry_points) != 4 or len(self._entry_inferred_indices) != 1):
            return None, False
        review = self.initialization.get("load_entry_quartet_review")
        if review is not None:
            if review["status"] != "collecting":
                return None, False
            if now-review["started_at"] > LOAD_ENTRY_QUARTET_REVIEW_SECONDS+1e-9:
                self.initialization["load_entry_quartet_review"] = dict(review, status="expired")
                return None, False
        waiting = review is not None
        if not enabled or len(candidates) not in (3, 4):
            return None, waiting
        measured_ids = {track_id for i, track_id in enumerate(self._entry_track_ids)
                        if i not in self._entry_inferred_indices and track_id is not None}
        candidate_ids = [obj.get("track_id") for obj in candidates]
        if (len(measured_ids) != 3 or None in candidate_ids or len(set(candidate_ids)) != len(candidates)
                or not measured_ids.issubset(candidate_ids)):
            return None, waiting
        geometry_only = len(candidates) == 3
        if geometry_only:
            # A weak ByteTrack continuation may finish an already supported
            # layout. It never starts a review or enters slot observation.
            if not waiting or review["frames"] != LOAD_ENTRY_QUARTET_REVIEW_FRAMES-1:
                return None, waiting
            fourth_id = review["fourth_track_id"]
            continuation = self._candidates(obj for obj in position_candidates if obj.get("track_id") == fourth_id)
            if (len(continuation) != 1 or np.linalg.norm(np.asarray(continuation[0]["center"])
                    -review["fourth_center_xy"]) > self._entry_gate*.5):
                return None, waiting
            candidates = [*candidates, continuation[0]]
            candidate_ids.append(fourth_id)
        if any(obj.get("detection_source") == "local_crop" or obj["conflict"] or self._occluded(obj, hands)
               for obj in candidates):
            return None, waiting
        points = np.asarray([obj["center"] for obj in candidates])
        order = self._load_reference_order(points) if slot_layout_error(points) is not None else None
        if order is None:
            return None, waiting
        ordered_ids = [candidate_ids[i] for i in order]
        same = waiting and review["track_ids_in_slot_order"] == ordered_ids
        if geometry_only and not same:
            return None, waiting
        count = review["frames"]+1 if same else 1
        confirmed = count >= LOAD_ENTRY_QUARTET_REVIEW_FRAMES
        fourth_id, = set(candidate_ids)-measured_ids
        self.initialization["load_entry_quartet_review"] = {
            "status": "confirmed" if confirmed else "collecting",
            "started_at": review["started_at"] if waiting else now, "last_matched_at": now,
            "frames": count, "track_ids_in_slot_order": ordered_ids, "fourth_track_id": fourth_id,
            "fourth_center_xy": points[candidate_ids.index(fourth_id)].tolist(),
            "geometry_only_last_frame": geometry_only}
        return (candidates, False) if confirmed else (None, True)

    def _reconcile_entry_geometry(self, candidates, hands, now):
        """A late clear quartet may replace an incorrect provisional candidate."""
        chosen = self._select_track_geometry(candidates, hands, now)
        if chosen is None:
            return candidates
        return self._rebuild_entry_geometry(chosen, now)

    def _rebuild_entry_geometry(self, chosen, now):
        old_ids = set(self._entry_track_ids)-{None}
        new_ids = {obj['track_id'] for obj in chosen}
        if (len(self._entry_points) == 4 and not self._entry_inferred_indices
                and old_ids == new_ids):
            return chosen
        locked_slot = self._entry_locked_s4_slot
        locked_old_index = next((index for index, slot in enumerate(self.slots)
                                 if slot is locked_slot), None)
        locked_track_id = (self._entry_track_ids[locked_old_index]
                           if locked_old_index is not None
                           and locked_old_index < len(self._entry_track_ids) else None)
        locked_center = (None if locked_old_index is None
                         else np.asarray(self._entry_points[locked_old_index], dtype=float))
        # Rebuild provisional geometry only. Retained tracks keep their state,
        # support and events; newly accepted IDs restore independent history.
        old = {track_id: i for i, track_id in enumerate(self._entry_track_ids)
               if track_id is not None}
        for track_id in old_ids-new_ids:
            self.track_store.release_candidate(track_id, (self.phase, self.group_id))
        order = sorted(chosen, key=lambda obj: tuple(obj['center']))
        mapping = {old[obj['track_id']]+1: i+1 for i, obj in enumerate(order)
                   if obj['track_id'] in old}
        fields = ('_entry_seen_at', '_entry_support_since', '_entry_support_count')
        saved = {name: list(getattr(self, name)) for name in fields}
        slots = self.slots
        seen = self._initial_seen.copy()
        self._entry_points = [np.asarray(obj['center'], dtype=float) for obj in order]
        self._entry_track_ids = [obj['track_id'] for obj in order]
        locked_replacement_index = None
        if (locked_slot is not None and locked_old_index is not None
                and locked_old_index < len(self._entry_points)
                and (locked_track_id is None or locked_track_id not in new_ids)):
            # If the locked ByteTrack ID was replaced, transfer the lock to the
            # nearest selected candidate only within the normal entry gate.
            # This preserves the physical temporary slot across an ID rollover
            # without allowing a distant candidate to inherit its evidence.
            distances = np.linalg.norm(np.asarray(self._entry_points)-locked_center, axis=1)
            nearest = int(np.argmin(distances))
            gate = max(float(self._entry_gate or 0.), 2.)
            if distances[nearest] <= gate:
                locked_replacement_index = nearest
        self.slots = []
        self._initial_seen = set()
        defaults = (now, None, 0)
        for name, default in zip(fields, defaults):
            setattr(self, name, [saved[name][old[obj['track_id']]]
                if obj['track_id'] in old else default for obj in order])
        for i, obj in enumerate(order):
            index = old.get(obj['track_id'])
            slot = (slots[index] if index is not None else
                    locked_slot if i == locked_replacement_index else SlotState(i+1))
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
        self._s4_before_recalculation = None
        self._reset_three_point_stability()
        self._entry_revision += 1
        if locked_slot is not None and any(slot is locked_slot for slot in self.slots):
            self._entry_locked_s4_slot = locked_slot
        else:
            self._clear_entry_s4_lock("geometry_rebuild_removed_slot")
        return chosen

    def reset_group(self, group_id, phase=None, centers=None):
        """供未来分组模块调用；切组必须显式执行，旧组摘要应由调用方先保存。"""
        contact = self.two_point_hand_contact
        # 新组不能沿用旧组中心。默认重新等待YOLO四槽初始化。
        self.__init__(centers, group_id=group_id,
                      phase=phase or self.phase, clockwise=self.clockwise,
                      load_trajectory_enabled=self.load_trajectory_enabled,
                      match_distance_ratio=self.match_distance_ratio, confirm_seconds=self.confirm_seconds,
                      minimum_frames=self.minimum_frames, lost_seconds=self.lost_seconds,
                      conflict_margin=self.conflict_margin, hand_overlap=self.hand_overlap,
                      ambiguity_ratio=self.ambiguity_ratio,
                      initialization_seconds=self.initialization_seconds,
                      geometry_initialization_seconds=self.geometry_initialization_seconds,
                      unload_entry_numbering_seconds=self.unload_entry_numbering_seconds,
                      machine_center=self.machine_center,
                      initialization_frames=self.initialization_frames,
                      initialization_stability_ratio=self.initialization_stability_ratio,
                      initialization_history_seconds=self.initialization_history_seconds,
                      load_hand_contact_enabled=self.load_hand_contact_enabled,
                      load_hand_contact_seconds=self.load_hand_contact_seconds,
                      load_hand_contact_min_frames=self.load_hand_contact_min_frames,
                      load_hand_contact_overlap_ratio=self.load_hand_contact_overlap_ratio,
                      load_hand_contact_competing_slot_enabled=self.load_hand_contact_competing_slot_enabled,
                      load_hand_contact_competing_slot_id=self.load_hand_contact_competing_slot_id,
                      load_hand_contact_competing_margin=self.load_hand_contact_competing_margin,
                      load_hand_contact_competing_ratio=self.load_hand_contact_competing_ratio,
                      s4_contact_prediction_seconds=self.s4_contact_prediction_seconds,
                      s4_prediction_contact_overlap_ratio=self.s4_prediction_contact_overlap_ratio,
                      s4_hand_region=self.s4_hand_region,
                      slot2_below_slot3=self.slot2_below_slot3,
                      allow_three_point_initialization=self.allow_three_point_initialization,
                      visual_reversal_review_seconds=self.visual_reversal_review_seconds,
                      require_unload_event_before_numbering=(
                          self.require_unload_event_before_numbering))
        if contact is not None:
            contact.reset()
            self.two_point_hand_contact = contact

    def _reset_initialization(self, reason):
        """候选混组或轨迹中断时丢弃未归属的历史，不能接到另一组上。"""
        if self._provisional is not None:
            if self.track_store is not None:
                for track_id in self._provisional.byte_track_ids:
                    self.track_store.release_candidate(track_id, (self.phase, self.group_id))
            if self.two_point_hand_contact is not None:
                self.two_point_hand_contact.reset()
        self._provisional = None
        self._entry_load_preview_contact = None
        self._machine_center_votes.clear()
        self._load_corner_history = None
        self._provisional_inferred_slot_ids = set()
        self._reset_three_point_stability()
        self._initial_seen.clear()
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        self._pending_numbering_points = self._pending_numbering_order = None
        self._provisional_missing_since.clear()
        self._provisional_last_real_centers.clear()
        self._provisional_quarantined_track_ids.clear()
        self._provisional_reidentification_events.clear()
        self._clear_entry_s4_lock("initialization_reset")
        self.initialization.update(status="waiting", reason=reason, stable_frames=0,
            provisional_started_at=None, initial_centers_xy=None,
            numbering_reference_centers_xy=None,
            geometry_seen_slot_ids=[], geometry_seen_track_ids=[], inferred_slot_ids=[], required_slot_count=4,
            buffered_event_count=0,
            provisional_quarantined_track_ids=[],
            provisional_reidentification_events=[])

    def _provisional_recovery_limits(self, provisional):
        """Return the bounded pre-numbering ID recovery window and distance gate."""
        spacing = provisional._spacing() if provisional.points is not None else 0.0
        # Keep this window short enough that a genuinely new carrier can still
        # replace a lost point, while covering the observed BT124 -> BT202 gap.
        window = min(max(float(self.lost_seconds), 1.0), 1.2)
        distance = max(40.0, float(spacing) * self.match_distance_ratio)
        return window, distance

    def _replace_initial_edge_slot(self, row, candidate, now):
        """Replace only an unnumbered edge hypothesis; retain the other slots' evidence."""
        provisional = self._provisional
        if (self.phase != 'unload' or self.group_id != 1 or self.points is not None
                or provisional is None or not 0 <= row < 4 or candidate.get('track_id') is None):
            return False
        old_id, new_id = provisional.byte_track_ids[row], candidate['track_id']
        if new_id in provisional.byte_track_ids:
            return False
        center = np.asarray(candidate['center'], dtype=float)
        previous_center = provisional.points[row].copy()
        discarded = [event for event in provisional.events if event['slot_id'] == row+1]
        provisional.events = [event for event in provisional.events if event['slot_id'] != row+1]
        if self.track_store is not None and old_id is not None:
            self.track_store.discard_candidate_events(old_id, (self.phase, self.group_id), discarded)
            self.track_store.release_candidate(old_id, (self.phase, self.group_id))
        provisional.slots[row] = SlotState(row+1)
        provisional.byte_track_ids[row] = new_id
        provisional.points[row] = provisional.reference_points[row] = center
        provisional.canonical_points = provisional.reference_points.copy()
        provisional.group_transform = np.array([[1., 0., 0.], [0., 1., 0.]])
        provisional.transform_valid_this_frame = False
        provisional.last_geometry_time = now
        provisional._predicted_indices.discard(row)
        self._predicted_indices.discard(row)
        self._provisional_inferred_slot_ids.discard(row)
        self._provisional_missing_since.pop(row, None)
        self._provisional_last_real_centers[row] = center.copy()
        self._provisional_quarantined_track_ids.discard(new_id)
        self._initial_seen.discard(row)
        self._pending_numbering_points = self._pending_numbering_order = None
        self._initial_points, self._initial_since, self._initial_count = None, None, 0
        self._machine_center_votes.clear()
        if row == 3:
            provisional._pending_s4_unload = None
            provisional._entry_s4_contact = self._entry_s4_contact = None
        self.initialization.update(initial_centers_xy=provisional.points.tolist(),
            initial_edge_replacement={'at': float(now), 'from_track_id': old_id, 'to_track_id': new_id,
                'from_center_xy': previous_center.tolist(), 'to_center_xy': center.tolist(),
                'support_track_ids': [track_id for i, track_id in enumerate(provisional.byte_track_ids) if i != row],
                'preserved_event_count': len(provisional.events),
                'reason': 'stable_interior_quartet'})
        return True

    def _reconcile_provisional_candidates(self, provisional, candidates, now, hands):
        """Keep a stable provisional slot alive across a short ByteTrack ID gap.

        The old implementation reset the whole provisional group whenever a
        new unmatched box appeared.  That allowed an unrelated previous-group
        box to replace a temporarily missing slot.  Here unmatched boxes are
        quarantined first; a new ID can claim a missing slot only when it is
        close to that slot's last real center, or after the recovery window has
        expired.
        """
        if not candidates:
            matches = {}
        else:
            matches = provisional._associate(
                candidates, provisional._spacing() * provisional.match_distance_ratio)
        if len(self._provisional_inferred_slot_ids) == 1 and len(candidates) == 4:
            missing, = self._provisional_inferred_slot_ids
            edge_cols = [col for col, obj in enumerate(candidates) if obj.get("initial_edge_slot")]
            measured = {row: col for row, col in matches.items() if row != missing}
            if len(edge_cols) == 1 and len(measured) == 3 and edge_cols[0] not in measured.values():
                # Three measured tracks already validated the edge box as the fourth slot.
                matches[missing] = edge_cols[0]
                self._pending_numbering_points = self._pending_numbering_order = None
                self._initial_points, self._initial_since, self._initial_count = None, None, 0
        row_count = len(provisional.points)
        window, recovery_gate = self._provisional_recovery_limits(provisional)
        last_real = self._provisional_last_real_centers
        missing_since = self._provisional_missing_since
        for row in range(row_count):
            if row not in last_real:
                last_real[row] = np.asarray(provisional.points[row], dtype=float).copy()

        missing_rows = {row for row in range(row_count) if row not in matches}
        for row in missing_rows:
            missing_since.setdefault(row, float(now))
        for row in list(missing_since):
            if row in matches:
                missing_since.pop(row, None)

        unmatched_cols = set(range(len(candidates))) - set(matches.values())

        # First try to recover a disappeared slot from its last *real*
        # detection.  This is deliberately independent of the predicted point
        # that the provisional tracker may use for display.
        recovery_pairs = []
        for row in sorted(missing_rows):
            center = last_real[row]
            for col in unmatched_cols:
                distance = float(np.linalg.norm(
                    np.asarray(candidates[col]["center"], dtype=float) - center))
                if distance <= recovery_gate:
                    recovery_pairs.append((distance, row, col))
        used_rows, used_cols = set(), set()
        for distance, row, col in sorted(recovery_pairs):
            if row in used_rows or col in used_cols:
                continue
            matches[row] = col
            used_rows.add(row)
            used_cols.add(col)
            old_id = provisional.byte_track_ids[row]
            new_id = candidates[col].get("track_id")
            if old_id is not None and new_id is not None and int(old_id) != int(new_id):
                self._provisional_reidentification_events.append({
                    "slot_index": int(row),
                    "from_track_id": int(old_id),
                    "to_track_id": int(new_id),
                    "gap_seconds": round(float(now - missing_since.get(row, now)), 6),
                    "distance_px": round(distance, 3),
                    "at": float(now),
                    "method": "provisional_last_real_center",
                })
            missing_since.pop(row, None)
            self._provisional_quarantined_track_ids.discard(
                None if new_id is None else int(new_id))

        unmatched_cols -= used_cols
        missing_rows = {row for row in range(row_count) if row not in matches}

        # A replacement is allowed only after the old slot's recovery window
        # has elapsed.  Before that, every unrelated ID is isolated for this
        # provisional group and cannot affect geometry or slot state.
        ignored_cols = set()
        for col in sorted(unmatched_cols):
            candidate = candidates[col]
            center = np.asarray(candidate["center"], dtype=float)
            replacement = []
            for row in missing_rows:
                if now - missing_since.get(row, now) < window:
                    continue
                revised = np.asarray(provisional.points, dtype=float).copy()
                revised[row] = center
                if valid_slot_geometry(revised):
                    replacement.append((float(np.linalg.norm(center-last_real[row])), row))
            if replacement:
                _, row = min(replacement)
                matches[row] = col
                missing_rows.remove(row)
                missing_since.pop(row, None)
                self._provisional_quarantined_track_ids.discard(
                    None if candidate.get("track_id") is None else int(candidate["track_id"]))
                continue
            ignored_cols.add(col)
            track_id = candidate.get("track_id")
            if track_id is not None:
                self._provisional_quarantined_track_ids.add(int(track_id))

        for col in matches.values():
            track_id = candidates[col].get("track_id")
            if track_id is not None:
                self._provisional_quarantined_track_ids.discard(int(track_id))

        # If three or more established slots remain matched, an ignored extra
        # box is not a geometry discontinuity.  The caller may keep the
        # provisional group alive and wait for the missing slot to recover.
        reliable_rows = [row for row, col in matches.items()
                         if not candidates[col].get("conflict")
                         and not self._occluded(candidates[col], hands)]
        protected = bool(missing_rows and reliable_rows and
                         len(reliable_rows) >= max(3, row_count-1))
        return matches, ignored_cols, missing_rows, protected, window

    def _review_initial_unload(self, provisional):
        """Review a visual reversal before releasing the first G1 unload."""
        events = provisional.events
        first = next((event for event in events if event['action'] == 'unload'
                      and event.get('expected_action')), None)
        if first is None or first['from'] != 'filled':
            return False
        track_id = first.get('byte_track_id')
        contact_without_track = track_id is None and first.get('reason') == 'hand_contact_inferred_unload'
        if contact_without_track:
            slot_index = first['slot_id']-1
            track_id = provisional.byte_track_ids[slot_index]
        if track_id is None:
            return False

        def is_visual_reversal(event):
            return (event is not None and event['action'] == 'load'
                    and event['from'] == 'empty' and event['to'] == 'filled'
                    and event.get('byte_track_id') == track_id
                    and event['slot_id'] == first['slot_id'] and not event.get('inferred')
                    and event['confirmed_at'] > first['confirmed_at'])

        reversal = next((event for event in events if is_visual_reversal(event)), None)
        if reversal is None:
            if contact_without_track and is_visual_reversal(provisional.slots[slot_index].pending_visual_event):
                self.initialization["reason"] = "waiting_for_initial_unload_visual_review"
                return True
            return False
        provisional.events = [event for event in events if event is not first and event is not reversal]
        if self.track_store is not None:
            self.track_store.discard_candidate_events(track_id, (self.phase, self.group_id), (first, reversal))
        if contact_without_track or any(event['action'] == 'unload' and event.get('expected_action')
                                        for event in provisional.events):
            return False
        self._reset_initialization('provisional_first_unload_reversed')
        return True

    def _initialize(self, candidates, hands, now, enabled, roi_contour, objects, *,
                    recheck_initial_unload=False, load_support_resolver=None):
        """先建立候选轨迹并逐槽记录状态；身份确认后一次性发布缓存事件。"""
        if enabled and len(candidates) > 4 and self._provisional is None:
            selected = self._select_track_geometry(candidates, hands, now)
            if selected is not None:
                candidates = selected
                selected_ids = {obj['track_id'] for obj in selected}
                objects = [o for o in objects if o['class_name'] not in SLOT_STATES
                           or o.get('track_id') in selected_ids]
        self.initialization["eligible_slot_count"] = len(candidates)
        if len(candidates) > 4 and self._provisional is None:
            self._reset_initialization("too_many_slot_detections")
            return []
        ignored_candidate_cols = set()
        provisional_missing_rows = set()
        provisional_recovery_window = None
        provisional_matches_ready = False
        if self._provisional is not None:
            p = self._provisional
            matches, ignored_candidate_cols, provisional_missing_rows, protected, \
                provisional_recovery_window = self._reconcile_provisional_candidates(
                    p, candidates, now, hands)
            stale = (p.last_geometry_time is not None
                     and now-p.last_geometry_time > self.lost_seconds)
            # A stable three-slot support protects the provisional group while
            # the fourth point is inside its short ID-recovery window.  Extra
            # unmatched IDs are isolated instead of resetting the group.
            if stale and not protected:
                self._reset_initialization("initialization_track_discontinuity")
                ignored_candidate_cols = set()
                provisional_missing_rows = set()
                provisional_recovery_window = None
            else:
                provisional_matches_ready = True
        if self._provisional is None:
            inferred_missing = None
            if len(candidates) == 3 and self.allow_three_point_initialization:
                inferred_missing = infer_fourth_square_center([c["center"] for c in candidates])
            if len(candidates) == 4:
                raw_points = np.asarray([c["center"] for c in candidates], dtype=float)
            elif inferred_missing is not None:
                raw_points = np.vstack(([c["center"] for c in candidates], inferred_missing))
            else:
                if enabled and len(candidates) == 2 and self.phase == "load" and load_support_resolver is not None:
                    self.three_point_preview = self._preview_three_point_box(
                        candidates, {0: 0, 1: 1}, hands, roi_contour, load_support_resolver=load_support_resolver)
                reason = ("waiting_for_valid_three_slot_geometry" if len(candidates) == 3
                          and self.allow_three_point_initialization else "waiting_for_four_detections")
                self.initialization.update(reason=reason, inferred_slot_ids=[])
                return []
            try:
                raw_points = validate_centers(raw_points)
            except ValueError:
                self._reset_initialization("invalid_initial_geometry")
                return []
            # 槽号在候选轨迹开始时只排序一次；之后停稳或旋转均不重排。
            order = (self._numbering_order_with_predicted_s4(raw_points, {3})
                     if inferred_missing is not None else self._numbering_order(raw_points))
            raw_track_ids = [candidate.get("track_id") for candidate in candidates]
            if inferred_missing is not None:
                raw_track_ids.append(None)
            missing_index = int(np.flatnonzero(order == 3)[0]) if inferred_missing is not None else None
            points = raw_points[order]
            self._provisional = SlotGroupTracker(points, group_id=self.group_id, phase=self.phase,
                clockwise=self.clockwise, match_distance_ratio=self.match_distance_ratio,
                confirm_seconds=self.confirm_seconds, minimum_frames=self.minimum_frames,
                lost_seconds=self.lost_seconds, conflict_margin=self.conflict_margin,
                hand_overlap=self.hand_overlap, ambiguity_ratio=self.ambiguity_ratio,
                load_hand_contact_enabled=self.load_hand_contact_enabled,
                load_hand_contact_seconds=self.load_hand_contact_seconds,
                load_hand_contact_min_frames=self.load_hand_contact_min_frames,
                load_hand_contact_overlap_ratio=self.load_hand_contact_overlap_ratio,
                 load_hand_contact_competing_slot_enabled=self.load_hand_contact_competing_slot_enabled,
                 load_hand_contact_competing_slot_id=self.load_hand_contact_competing_slot_id,
                 load_hand_contact_competing_margin=self.load_hand_contact_competing_margin,
                 load_hand_contact_competing_ratio=self.load_hand_contact_competing_ratio,
                s4_contact_prediction_seconds=self.s4_contact_prediction_seconds,
                s4_prediction_contact_overlap_ratio=self.s4_prediction_contact_overlap_ratio,
                s4_hand_region=self.s4_hand_region,
                allow_three_point_initialization=self.allow_three_point_initialization,
                visual_reversal_review_seconds=self.visual_reversal_review_seconds,
                require_unload_event_before_numbering=(
                    self.require_unload_event_before_numbering))
            self._provisional._region_formal_numbering = False
            self._provisional.byte_track_ids = [raw_track_ids[int(i)] for i in order]
            self._provisional.track_store = self.track_store
            self._provisional.two_point_hand_contact = self.two_point_hand_contact
            self._provisional_missing_since.clear()
            self._provisional_last_real_centers = {
                int(i): np.asarray(point, dtype=float).copy()
                for i, point in enumerate(points)}
            self._provisional_quarantined_track_ids.clear()
            self._provisional_reidentification_events.clear()
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
        if not provisional_matches_ready:
            matches = p._associate(candidates, p._spacing()*p.match_distance_ratio)
            ignored_candidate_cols = set()
            provisional_missing_rows = {
                row for row in range(len(p.points)) if row not in matches}
        else:
            # The reconciliation above already assigned recovered IDs and
            # marked unrelated visible boxes for this provisional frame.
            matches = dict(matches)
        p._remember_track_ids(matches, candidates, p.byte_track_ids)
        for row, col in matches.items():
            self._provisional_last_real_centers[row] = np.asarray(
                candidates[col]["center"], dtype=float).copy()
            self._provisional_missing_since.pop(row, None)
        self._observe_machine_s4([candidates[col] for col in matches.values()], now, enabled)
        reliable = {i: candidates[col] for i,col in matches.items()
                    if not self._occluded(candidates[col], hands)}
        self._provisional_inferred_slot_ids.difference_update(reliable)
        self._predicted_indices.update(self._provisional_inferred_slot_ids)
        accepted_ids = {candidates[col].get("track_id") for col in matches.values()}
        accepted_ids.discard(None)
        provisional_objects = [o for o in objects
                               if o["class_name"] not in SLOT_STATES
                               or o.get("track_id") in accepted_ids
                               or o.get("track_id") is None]
        p.update(provisional_objects,
                 now, observations_enabled=enabled, roi_contour=roi_contour)
        if recheck_initial_unload and self._review_initial_unload(p):
            return []
        if enabled:
            self.three_point_preview = self._preview_three_point_box(
                candidates, matches, hands, roi_contour, ignored_cols=ignored_candidate_cols,
                load_support_resolver=load_support_resolver)
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
            buffered_event_count=len(p.events),
            provisional_quarantined_track_ids=sorted(self._provisional_quarantined_track_ids),
            provisional_reidentification_events=list(self._provisional_reidentification_events))
        required_seen = 3 if self._provisional_inferred_slot_ids else 4
        self.initialization["required_slot_count"] = required_seen
        self.initialization["inferred_slot_ids"] = sorted(i+1 for i in self._provisional_inferred_slot_ids)
        provisional_missing_rows = {row for row in range(len(p.points)) if row not in matches}
        recovery_pending = bool(provisional_recovery_window is not None and any(
            now-self._provisional_missing_since.get(row, now) < provisional_recovery_window
            for row in provisional_missing_rows))
        if recovery_pending:
            self.initialization.update(reason="waiting_for_provisional_id_recovery",
                                       stable_frames=self._initial_count)
            return []
        if (len(self._initial_seen) < required_seen or self._initial_count < self.initialization_frames
                or self._initial_since is None
                or now-self._initial_since+1e-9 < self.geometry_initialization_seconds):
            return []
        if (self.require_unload_event_before_numbering and self.group_id == 1
                and self.phase == "unload"
                and not any(event.get("action") == "unload"
                            and event.get("expected_action", False)
                            and event.get("confirmed_at") is not None
                            for event in p.events)):
            # 几何已经稳定，但这是尚未确认首个下料动作的G1候选。
            # 继续调用候选跟踪器记录空满和事件，等动作证据出现后再一次性映射。
            if self._pending_numbering_order is None:
                stable_points = p.points.copy()
                self._pending_numbering_points = stable_points
                self._pending_numbering_order = np.asarray(
                    self._numbering_order(stable_points), dtype=int)
            self.initialization.update(reason="waiting_for_first_unload_event",
                                       numbering_reference_centers_xy=(
                                           self._pending_numbering_points.tolist()),
                                       buffered_event_count=len(p.events))
            return []
        numbering_points = (self._pending_numbering_points
                            if self._pending_numbering_points is not None else p.points)
        order = (self._pending_numbering_order.copy()
                 if self._pending_numbering_order is not None else
                 (self._numbering_order(numbering_points)
                  if self.slot2_below_slot3 else np.arange(4)))
        reference_order = self._load_reference_order(numbering_points,
            inferred_indices=self._provisional_inferred_slot_ids)
        if self._load_perspective_numbering is not None and reference_order is None:
            self.initialization.update(reason="waiting_for_calibrated_load_numbering")
            return []
        if reference_order is not None:
            order = reference_order
            self.initialization["machine_center_numbering"] = {"status": (
                "matched_calibrated_load_layout" if self._load_perspective_numbering is not None
                else "matched_unload_layout")}
        else:
            order = self._final_numbering_order(
                numbering_points, p.byte_track_ids, order, now, inferred=bool(self._provisional_inferred_slot_ids),
                slots=p.slots, events=p.events)
            if order is None:
                self.initialization.update(reason="waiting_for_numbering_pose")
                return []
        self.points = p.points[order].copy()
        # 正式编号时以当帧位置建立参考，避免继承候选阶段的旧参考点。
        self.reference_points = self.points.copy()
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
        self._seed_formal_s4_detection_cache(now)
        self._temporary_s4_contact_box(self.slots[3], 3, now)
        self.last_geometry_time = p.last_geometry_time
        for event in self.events:
            event.update(slot_id=mapping[event["slot_id"]], reported_at=float(now),
                         recovered_before_initialization=event["confirmed_at"] < now)
        self._provisional = None
        inferred_slot_ids = sorted(mapping[i+1] for i in self._provisional_inferred_slot_ids)
        self._provisional_inferred_slot_ids = set()
        self.initialization.update(status="ready", reason="initialized", initialized_at=float(now),
            inferred_slot_ids=inferred_slot_ids, required_slot_count=3 if inferred_slot_ids else 4,
            numbering_reference_centers_xy=(None if numbering_points is None
                                             else numbering_points.tolist()),
            geometry_seen_slot_ids=sorted(mapping[i+1] for i in self._initial_seen),
            geometry_seen_track_ids=[self.byte_track_ids[mapping[i+1]-1] for i in sorted(self._initial_seen)
                                     if self.byte_track_ids[mapping[i+1]-1] is not None],
            provisional_quarantined_track_ids=sorted(self._provisional_quarantined_track_ids),
            provisional_reidentification_events=list(self._provisional_reidentification_events))
        return list(self.events)

    def _spacing(self):
        distances = np.linalg.norm(self.reference_points[:, None] - self.reference_points[None, :], axis=2)
        return float(np.min(distances[np.triu_indices(4, 1)]))

    def _candidates(self, objects):
        candidates = []
        # 高度重合或包含候选先合并位置；分数接近的空满冲突仍保留为unknown，
        # 不按高分强判。包含关系用于处理一个槽框包住另一个重复框时 IoU 偏低的情况。
        for obj in sorted((o for o in objects if o["class_name"] in SLOT_STATES),
                          key=lambda o: o["confidence"], reverse=True):
            duplicate = next((c for c in candidates if (
                box_iou(c["xyxy"], obj["xyxy"]) > .85
                or box_containment(c["xyxy"], obj["xyxy"]) >= .9)), None)
            if duplicate is not None:
                if obj["class_name"] != duplicate["class_name"] and duplicate["confidence"]-obj["confidence"] < self.conflict_margin:
                    duplicate["conflict"] = True
                elif obj["class_name"] == duplicate["class_name"]:
                    # 已连续存在的轨迹优先于刚出现的重复框，避免小框抢走旧身份。
                    duplicate_record = (self.track_store.tracks.get(duplicate.get("track_id"))
                                         if self.track_store is not None else None)
                    current_record = (self.track_store.tracks.get(obj.get("track_id"))
                                      if self.track_store is not None else None)
                    duplicate_key = self._duplicate_track_priority(duplicate, duplicate_record)
                    current_key = self._duplicate_track_priority(obj, current_record)
                    if current_key > duplicate_key:
                        index = next(index for index, candidate in enumerate(candidates)
                                     if candidate is duplicate)
                        a = obj["xyxy"]
                        candidates[index] = {**obj,
                            "center": [(a[0]+a[2])/2, (a[1]+a[3])/2],
                            "conflict": False}
                continue
            a = obj["xyxy"]
            candidates.append({**obj, "center": [(a[0]+a[2])/2, (a[1]+a[3])/2], "conflict": False})
        return candidates

    def _duplicate_track_priority(self, obj, record):
        """Rank overlapping same-class boxes by identity continuity first."""
        if record is None:
            return (0, 0, 0., float(obj.get("confidence", 0.)))
        owner = record.get("owner")
        active_owner = int(owner == (self.phase, self.group_id))
        reliable_count = int(record.get("reliable_count") or 0)
        first_seen = float(record.get("first_seen") or 0.)
        return (active_owner, reliable_count, -first_seen, float(obj.get("confidence", 0.)))

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

    def _associate_transition_load(self, candidates, matches, hands, gate, now):
        """Reacquire new IDs using this frame's measured geometry after three-point numbering."""
        known = {i: col for i, col in matches.items()
                 if self.byte_track_ids[i] is not None
                 and self.byte_track_ids[i] == candidates[col].get("track_id")}
        if len(known) < 2 or len(known) == 4 or any(
                candidates[col]["conflict"] or self._occluded(candidates[col], hands) for col in known.values()):
            return known
        missing = sorted(set(range(4))-known.keys())
        missing_slot_ids = {i+1 for i in missing}
        identity = (self.phase, self.group_id)
        eligible = []
        for col, obj in enumerate(candidates):
            track_id = obj.get("track_id")
            record = self.track_store.tracks.get(track_id)
            if (record is not None and track_id not in self.byte_track_ids
                    and record["owner"] in (None, identity)
                    and (record["slot_id"] is None or record["slot_id"] in missing_slot_ids)
                    and obj.get("detection_source") != "local_crop" and not obj["conflict"]
                    and not self._occluded(obj, hands)
                    and record["reliable_since"] is not None
                    and record["reliable_count"] >= self.initialization_frames
                    and now-record["reliable_since"]+1e-9 >= self.initialization_seconds):
                eligible.append(col)
        if len(eligible) > 10:
            return known
        reference = self.reference_points.copy()
        winding = self._slot_ring(self.canonical_points)
        for i, col in known.items():
            reference[i] = candidates[col]["center"]
        layouts = []
        for columns in itertools.permutations(eligible, len(missing)):
            proposed = dict(zip(missing, columns))
            for i, col in proposed.items():
                reference[i] = candidates[col]["center"]
            if (slot_layout_error(reference) is not None and self._slot_ring(reference) == winding
                    and all(self.track_store.tracks[candidates[col]["track_id"]]["slot_id"] in (None, i+1)
                            for i, col in proposed.items())):
                layouts.append(proposed)
                if len(layouts) > 1:
                    return known
        if layouts:
            return known | layouts[0]
        # With S4 still hidden, two anchors can justify replacement of the reviewed
        # slot only while their relative positions still agree with the numbered pose.
        index = self.transition_load_slot_index
        if len(known) == 2 and index in missing and len(eligible) == 1:
            support = list(known)
            delta = reference[support]-self.canonical_points[support]
            if np.linalg.norm(delta[0]-delta[1]) <= gate*.2:
                projected = self.canonical_points + delta.mean(axis=0)
                col = eligible[0]
                point = candidates[col]["center"]
                record = self.track_store.tracks[candidates[col]["track_id"]]
                if (record["slot_id"] in (None, index+1)
                        and np.linalg.norm(point-projected[index]) <= min(gate*.5, self._spacing()*.25)):
                    projected[index] = point
                    projected[support] = reference[support]
                    if slot_layout_error(projected) is not None and self._slot_ring(projected) == winding:
                        known[index] = col
        return known

    @staticmethod
    def _slot_ring(points):
        offsets = points-points.mean(axis=0)
        ring = np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))
        return tuple(np.roll(ring, -int(np.flatnonzero(ring == 0)[0])))

    def _recover_slot_position(self, slot_index, candidates, matches, position_candidates, hands, now,
                               gate, roi_contour, *, allow_binding):
        """Recover a missing position and return a uniquely supported load track."""
        if self._position_recovery_index != slot_index:
            self._position_recovery_candidate = None
            self._load_binding_frames = 0
        self._position_recovery_index = slot_index
        previous = self._position_recovery_candidate
        previous_binding_frames = self._load_binding_frames
        self._position_recovery_candidate = None
        self._load_binding_frames = 0
        if not self._region_formal_numbering or self.identity_lost or slot_index in matches:
            return
        support = [i for i, col in matches.items()
                   if not candidates[col]["conflict"]
                   and candidates[col].get("detection_source") != "local_crop"
                   and not self._occluded(candidates[col], hands)]
        if len(support) < 2 or not valid_slot_geometry(self.reference_points):
            if previous is not None and previous[2] >= 3 and now-previous[3] <= self.s4_contact_prediction_seconds:
                self._position_recovery_candidate = previous
            return

        used = set(matches.values())
        unused = [(col, obj) for col, obj in enumerate(candidates) if col not in used]
        # Supplemental boxes have no ByteTrack ID and only contribute position.
        if slot_index == 3:
            unused.extend((None, obj) for obj in self._candidates(position_candidates))
        layout = self.reference_points.copy()
        layout[support] = [candidates[matches[i]]["center"] for i in support]
        hull = cv2.convexHull(layout.astype(np.float32), returnPoints=False).ravel()
        if len(hull) != 4:
            return
        order = np.roll(hull, -int(np.flatnonzero(hull == slot_index)[0]))
        neighbours = [i for i in range(4) if i != slot_index]
        other_ids = {self.byte_track_ids[i] for i in neighbours}
        eligible = []
        for col, obj in unused:
            track_id = obj.get("track_id")
            if (obj["conflict"] or obj.get("detection_source") == "local_crop"
                    or track_id is not None and track_id in other_ids
                    or self._occluded(obj, hands)):
                continue
            if self.track_store is not None and track_id is not None:
                record = self.track_store.tracks.get(track_id)
                if record is not None and (record["owner"] not in (None, (self.phase, self.group_id))
                                           or record["slot_id"] not in (None, slot_index+1)):
                    continue
            center = np.asarray(obj["center"], dtype=float)
            if (np.linalg.norm(center-layout[slot_index]) > gate*1.5
                    or np.min(np.linalg.norm(layout[neighbours]-center, axis=1)) < gate*.75
                    or roi_contour is not None and cv2.pointPolygonTest(roi_contour, tuple(center), False) < 0
                    or any(box_containment(obj["xyxy"], candidates[match]["xyxy"]) >= .9 for match in used)):
                continue
            revised = layout.copy()
            revised[slot_index] = center
            if not valid_slot_geometry(revised):
                continue
            hull = cv2.convexHull(revised.astype(np.float32), returnPoints=False).ravel()
            if np.array_equal(np.roll(hull, -int(np.flatnonzero(hull == slot_index)[0])), order):
                eligible.append((col, obj))
        if len(eligible) != 1:
            if not eligible and previous is not None and previous[2] >= 3 \
                    and now-previous[3] <= self.s4_contact_prediction_seconds:
                self._position_recovery_candidate = previous
            return

        col, obj = eligible[0]
        observed_id = obj.get("track_id")
        track_id, center = observed_id, np.asarray(obj["center"], dtype=float)
        count = 1
        same_track = False
        if previous is not None:
            old_id, anchor, old_count, seen_at = previous
            if (now-seen_at <= self.s4_contact_prediction_seconds
                    and (track_id is None or old_id is None or track_id == old_id)
                    and np.linalg.norm(center-anchor) <= max(2., gate*.3)):
                count = old_count+1
                same_track = observed_id is not None and observed_id == old_id
                track_id = old_id if track_id is None else track_id
                if count < 3:
                    center = anchor
        self._position_recovery_candidate = (track_id, center, count, float(now))
        slot = self.slots[slot_index]
        if slot_index == 3 and count >= 3:
            self._remember_formal_s4_detection(slot, 3, obj, now, source="unassigned_detection")
        if (allow_binding and self.phase == "load" and slot.geometry_inferred
                and self.byte_track_ids[slot_index] is None and col is not None and observed_id is not None
                and obj.get("detection_source", "full_frame") == "full_frame"
                and sum(c.get("track_id") == observed_id for c in candidates) == 1):
            self._load_binding_frames = previous_binding_frames+1 if same_track else 1
            if self._load_binding_frames >= 3 and len(matches) == 3:
                return col
        return None

    def clear_byte_track_ids(self):
        """Drop IDs when the ROI sequence starts a different process group."""
        self.byte_track_ids = [None] * 4
        self._entry_track_ids = [None] * len(self._entry_points or [])
        self._clear_entry_s4_lock("byte_track_ids_cleared")
        if self._provisional is not None:
            self._provisional.byte_track_ids = [None] * 4

    def _slot_point(self, slot_index):
        """返回当前槽的几何参考点；初始化逐点跟踪时也能使用。"""
        points = self.reference_points if self.reference_points is not None else self._entry_points
        if points is None or slot_index >= len(points):
            return None
        return np.asarray(points[slot_index], dtype=float)

    def _inferred_load_contact_box(self, slot_index, candidates, geometry_matches, hands, roi_contour):
        """Center a contact box on the missing slot using same-group box sizes."""
        point = self._slot_point(slot_index)
        if (point is None or not np.isfinite(point).all()
                or roi_contour is not None and cv2.pointPolygonTest(roi_contour, tuple(point), False) < 0):
            return None
        sizes = []
        for i, support in enumerate(self.slots):
            if i == slot_index:
                continue
            size = support.box_size_wh if not support.geometry_inferred else None
            if i in geometry_matches:
                obj = candidates[geometry_matches[i]]
                if not obj["conflict"] and not self._occluded(obj, hands):
                    x1, y1, x2, y2 = obj["xyxy"]
                    size = (x2-x1, y2-y1)
            if size is not None:
                sizes.append(size)
        return centered_box(point, np.median(sizes, axis=0)) if sizes else None

    def _update_load_contact_boxes(self, candidates, matches, geometry_matches, hands, now, enabled, roi_contour):
        self._missing_load_contact_index = None
        if self.phase != "load":
            return
        inverse = cv2.invertAffineTransform(self.group_transform) if enabled else None
        for i, slot in enumerate(self.slots[:3]):
            slot.temporary_contact_box_xyxy = slot.temporary_contact_box_reason = None
            if not enabled:
                self._invalidate_slot_prediction(slot, "observations_disabled")
                continue
            if i in matches:
                obj = candidates[matches[i]]
                if i in geometry_matches and not self._occluded(obj, hands) \
                        and self._remember_slot_contact_box(slot, obj, now):
                    center = inverse[:, :2] @ obj["center"] + inverse[:, 2]
                    slot.last_valid_box_group_center_xy = tuple(center)
                continue
            if slot.geometry_inferred:
                box = self._inferred_load_contact_box(i, candidates, geometry_matches, hands, roi_contour)
                reason = "three_point_geometry"
            else:
                box, reason = self._project_cached_load_box(slot, i, now, roi_contour)
            if box is None:
                continue
            slot.temporary_contact_box_xyxy = box
            slot.temporary_contact_box_reason = reason
        target = self.inferred_load_contact_index()
        if target is None or target == 3:
            return
        slot = self.slots[target]
        if (slot.geometry_inferred or slot.temporary_contact_box_xyxy is None
                or target+1 not in self._load_verified_empty_slots or slot.confirmed != "empty"):
            return
        points = self.reference_points.copy()
        for i, slot in enumerate(self.slots[:3]):
            if slot.temporary_contact_box_xyxy is not None:
                points[i] = np.asarray(slot.temporary_contact_box_xyxy).reshape(2, 2).mean(axis=0)
        if any(np.flatnonzero(self._points_inside_box(points, hand["xyxy"])).tolist() == [target]
               for hand in hands):
            self._missing_load_contact_index = target

    @staticmethod
    def _points_inside_box(points, box):
        return np.all((points >= box[:2]) & (points <= box[2:]), axis=1)

    def _project_cached_load_box(self, slot, slot_index, now, roi_contour):
        box = slot.last_valid_box_xyxy
        if box is None or slot.last_valid_box_group_center_xy is None:
            return None, None
        center = slot.last_valid_center_xy
        fresh = (self.transform_valid_this_frame
                 and set(self.transform_support_slot_ids) == {1, 2, 3, 4}-{slot_index+1})
        if fresh:
            center = self.group_transform[:, :2] @ slot.last_valid_box_group_center_xy + self.group_transform[:, 2]
        elif now-slot.last_valid_box_at > self.s4_contact_prediction_seconds:
            return None, None
        if roi_contour is not None and cv2.pointPolygonTest(roi_contour, tuple(center), False) < 0:
            return None, None
        if fresh:
            box = centered_box(center, (box[2]-box[0], box[3]-box[1]))
            slot.last_valid_center_xy = tuple(center)
            slot.last_valid_box_xyxy, slot.last_valid_box_at = box, float(now)
            slot.last_valid_box_source = "three_point_geometry"
        return box, "three_point_geometry" if fresh else "held_position"

    def _byte_track_id_at(self, slot_index):
        track_ids = self.byte_track_ids if self.points is not None else self._entry_track_ids
        return track_ids[slot_index] if slot_index < len(track_ids) else None

    def _s4_region_expected(self, events=None):
        """按时间上界判断下一槽是否为S4；同上界的歧义不能开启区域门控。"""
        if not self._region_formal_numbering or self.points is None or self._entry_points is not None:
            return False
        if events is None:
            events = self.events
        if (not 1 <= len(events) <= 3 or any(not e.get("expected_action", False) for e in events)
                or any(e.get("slot_id") not in (1, 2, 3) for e in events)
                or len({e["slot_id"] for e in events}) != len(events)):
            return False
        possible = list(_event_slot_orders_by_time_upper(events))
        cycles = [self.clockwise[i:]+self.clockwise[:i] for i in range(4)]
        return bool(possible) and all(any(
            cycle[:len(order)] == order and cycle[len(order)] == 4 for cycle in cycles)
            for order in possible)

    @staticmethod
    def _valid_box_and_center(obj):
        """Return finite detection geometry suitable for contact fallback."""
        try:
            box = np.asarray(obj["xyxy"], dtype=float)
            center = np.asarray(obj["center"], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None, None
        if box.shape != (4,) or center.shape != (2,) or not np.isfinite(box).all() \
                or not np.isfinite(center).all() or box[2] <= box[0] or box[3] <= box[1]:
            return None, None
        return box, center

    def _invalidate_slot_prediction(self, slot, reason):
        slot.temporary_contact_box_xyxy = None
        slot.temporary_contact_box_reason = reason
        slot.last_valid_box_xyxy = slot.last_valid_center_xy = None
        slot.last_valid_box_at = slot.last_valid_box_source = None
        slot.last_valid_box_group_center_xy = None
        slot.prediction_contact = None
        if slot.load_contact_prediction_seen_at is not None:
            self._clear_load_contact(slot)

    def _remember_formal_s4_detection(self, slot, slot_index, obj, now, *, source="detection"):
        """Cache the latest real S4 box while formal numbering is active."""
        if not self._region_formal_numbering or self.points is None or slot_index != 3:
            return False
        return self._remember_slot_contact_box(slot, obj, now, source=source)

    def _remember_slot_contact_box(self, slot, obj, now, *, source="detection"):
        if obj is None or obj.get("conflict", False):
            return False
        box, center = self._valid_box_and_center(obj)
        if box is None:
            return False
        slot.last_valid_box_xyxy = tuple(float(value) for value in box)
        slot.last_valid_center_xy = tuple(float(value) for value in center)
        slot.last_valid_box_at = float(now)
        slot.last_valid_box_source = source
        slot.temporary_contact_box_xyxy = None
        slot.temporary_contact_box_reason = None
        return True

    def _seed_formal_s4_detection_cache(self, now):
        """Cache the current supported S4 geometry after formal numbering."""
        if not self._region_formal_numbering or self.points is None or len(self.slots) != 4:
            return
        slot = self.slots[3]
        # Only a fresh three-point prediction may refresh the geometry cache.
        if 3 not in self._predicted_indices or self.identity_lost or slot.geometry_observed:
            return
        if (slot.last_valid_box_source in ("unassigned_detection", "local_crop")
                and now-slot.last_valid_box_at <= self.s4_contact_prediction_seconds):
            return
        sizes = [s.box_size_wh for s in self.slots[:3] if s.box_size_wh is not None]
        size = slot.box_size_wh
        if size is None:
            if not sizes:
                return
            size = np.median(sizes, axis=0)
        center = self._slot_point(3)
        width, height = size
        if center.shape != (2,) or not np.isfinite(center).all():
            return
        slot.last_valid_center_xy = tuple(float(value) for value in center)
        slot.last_valid_box_xyxy = (
            float(center[0] - width / 2), float(center[1] - height / 2),
            float(center[0] + width / 2), float(center[1] + height / 2))
        slot.last_valid_box_at = float(now)
        slot.last_valid_box_source = "three_point_geometry"
        slot.temporary_contact_box_reason = None

    def _temporary_s4_contact_box(self, slot, slot_index, now):
        """Keep the last reliable S4 box usable while the same geometry is valid."""
        if not self._region_formal_numbering or self.points is None or slot_index != 3:
            return None
        if slot.last_valid_box_xyxy is None or slot.last_valid_center_xy is None \
                or slot.last_valid_box_at is None:
            return None
        if self.identity_lost:
            self._invalidate_slot_prediction(slot, "identity_lost")
            return None
        points = self.reference_points
        if points is None or np.asarray(points).shape != (4, 2) \
                or not np.isfinite(points).all() or not valid_slot_geometry(points):
            self._invalidate_slot_prediction(slot, "geometry_unavailable")
            return None
        if now-slot.last_valid_box_at > self.s4_contact_prediction_seconds:
            slot.temporary_contact_box_reason = "held_position"
        center = np.asarray(slot.last_valid_center_xy, dtype=float)
        predicted = self._slot_point(slot_index)
        if (slot_index in self._predicted_indices and predicted is not None
                and np.isfinite(predicted).all()
                and slot.last_valid_box_source not in ("unassigned_detection", "local_crop")):
            center = predicted
        box = np.asarray(slot.last_valid_box_xyxy, dtype=float)
        width, height = max(1., box[2] - box[0]), max(1., box[3] - box[1])
        temporary = (float(center[0] - width / 2), float(center[1] - height / 2),
                     float(center[0] + width / 2), float(center[1] + height / 2))
        slot.temporary_contact_box_xyxy = temporary
        return temporary

    def _held_s4_competing_unloads(self, prediction):
        if (self.phase != "unload" or prediction.reason != "other_slot_center_covered"
                or prediction.overlap_ratio < self.s4_prediction_contact_overlap_ratio):
            return ()
        hand = max(prediction.candidates, key=lambda candidate: candidate["confidence"])
        covered = set(hand["covered_slot_ids"])-{4}
        pending = [s.pending_visual_event for s in self.slots[:3] if s.pending_visual_event is not None]
        events = self.events+pending
        if not self._s4_region_expected(events):
            return ()
        # A hand can still cover a neighbour whose unload awaits the reversal review.
        # Store the exact visual transition so a cancelled neighbour cannot authorize S4.
        predecessors = tuple((e["slot_id"], e["confirmed_at"]) for e in events
            if e["slot_id"] in covered and e.get("from") == "filled" and e.get("to") == "empty"
            and not e.get("inferred") and self.slots[e["slot_id"]-1].confirmed == "empty"
            and self.slots[e["slot_id"]-1].confirmation_source == "yolo")
        return predecessors if len(predecessors) == len(covered) else ()

    def _hand_contact(self, slot_index, hands):
        """策略只返回接触证据；计时、状态和事件仍由当前跟踪器统一管理。"""
        if not self._is_hand_contact_slot(slot_index):
            return False
        if self.phase == 'load' and self.load_trajectory.blocks_inference(slot_index+1):
            self._clear_load_contact(self.slots[slot_index])
            self.slots[slot_index].prediction_contact = None
            self.s4_contact_diagnostic = {'contact_detected': False, 'source': None,
                'decision_reason': 'trajectory_passed_toward_other_slot', 'target_slot_id': slot_index+1}
            return False
        self.s4_region_diagnostic = None
        self.s4_contact_diagnostic = None
        if not self.load_hand_contact_enabled:
            self.s4_contact_diagnostic = {"contact_detected": False, "decision_reason": "contact_disabled"}
            return False
        if self._orbit_contact_enabled():
            record = self.two_point_hand_contact.orbit_contact(slot_index+1, self.last_time)
            accepted = record is not None and slot_index+1 in self.two_point_hand_contact.accepted_slot_ids
            self.s4_contact_diagnostic = {"contact_detected": bool(accepted),
                "decision_reason": "orbit_prediction_contact" if accepted else "orbit_no_unique_contact",
                "source": ("predicted_slot_center" if accepted and record["center_covered"]
                           else "predicted_slot_box" if accepted else None),
                "overlap_ratio": record['coverage'] if record is not None else 0.,
                "orbit_clockwise_degrees": self.two_point_hand_contact.orbit_degrees,
                "slot_id": slot_index+1}
            return bool(accepted)
        points = self.reference_points if self.reference_points is not None else self._entry_points
        if points is None or len(points) != 4 or slot_index >= len(points):
            self.s4_contact_diagnostic = {"contact_detected": False, "decision_reason": "geometry_unavailable"}
            return False
        box_sizes = [slot.box_size_wh for slot in self.slots]
        policy = HandContactPolicy(self.load_hand_contact_overlap_ratio,
            self.load_hand_contact_competing_slot_enabled,
            self.load_hand_contact_competing_slot_id-1,
            self.load_hand_contact_competing_margin, self.load_hand_contact_competing_ratio)
        is_s4 = slot_index == self._s4_index()
        slot = self.slots[slot_index]
        contact_box = slot.temporary_contact_box_xyxy
        if not is_s4 and contact_box is not None:
            x1, y1, x2, y2 = contact_box
            box_sizes[slot_index] = (x2-x1, y2-y1)
            points = np.asarray(points).copy()
            points[slot_index] = ((x1+x2)/2, (y1+y2)/2)
            if slot_index == self._missing_load_contact_index:
                hands = [hand for hand in hands if self._points_inside_box(points, hand["xyxy"])[slot_index]]
        region_expected = is_s4 and self._s4_region_expected()
        evidence = evaluate_s4_contact(points, box_sizes, slot_index, hands, policy,
            region=self.s4_hand_region if is_s4 else None, region_expected=region_expected)
        self.s4_contact_diagnostic = evidence.diagnostic()
        if not is_s4 and self.s4_contact_diagnostic["source"] is not None:
            self.s4_contact_diagnostic["source"] = self.s4_contact_diagnostic["source"].replace(
                "s4_", "predicted_slot_")
        if not is_s4 and contact_box is not None:
            self.s4_contact_diagnostic.update(
                contact_box_source=slot.temporary_contact_box_reason,
                temporary_contact_box_xyxy=[round(value, 2) for value in contact_box],
                temporary_contact_overlap_ratio=evidence.overlap_ratio)
        if is_s4 and contact_box is not None:
            x1, y1, x2, y2 = contact_box
            predicted_points = np.asarray(points, dtype=float).copy()
            predicted_points[slot_index] = ((x1+x2)/2, (y1+y2)/2)
            box_sizes[slot_index] = (x2-x1, y2-y1)
            prediction = evaluate_s4_contact(predicted_points, box_sizes, slot_index, hands, policy)
            held_box = slot.temporary_contact_box_reason in ("held_position", "hand_left")
            competing = self._held_s4_competing_unloads(prediction) if held_box else ()
            self.s4_contact_diagnostic.update(
                contact_box_source=slot.last_valid_box_source,
                temporary_contact_box_xyxy=[round(value, 2) for value in contact_box],
                temporary_contact_box_held=held_box,
                temporary_contact_detected=prediction.accepted or bool(competing),
                temporary_contact_overlap_ratio=prediction.overlap_ratio,
                prediction_contact_required_overlap_ratio=self.s4_prediction_contact_overlap_ratio,
                temporary_contact_competing_unloads=competing,
                temporary_contact_candidates=prediction.candidates)
        if evidence.region_hits:
            # status保留旧字段含义：eligible仅代表顺序允许；实际接触结果看contact_accepted。
            self.s4_region_diagnostic = {
                "status": "eligible" if region_expected else "suspected_unexpected_or_uncertain",
                "hands": evidence.region_hits, "contact_accepted": evidence.accepted,
                "decision_reason": evidence.reason, "source": evidence.source}
        return evidence.accepted

    def _record_hand_contact(self, slot, now, contact, enabled, slot_index, *,
                             disabled_reason=None, hold_pending=False, can_infer=True):
        previous_count = slot.load_contact_count
        if not hold_pending:
            self._advance_hand_contact(slot, now, contact, enabled, slot_index, can_infer=can_infer)
        if not self._is_hand_contact_slot(slot_index):
            return
        if slot.load_contact_pending and self._entry_points is not None and slot_index == self._s4_index():
            # The evidence is now valid enough to infer an operation after the
            # hand leaves.  Pin this SlotState before the next geometry frame
            # can choose a different temporary S4 index.
            self._lock_entry_s4_slot(slot_index, now)
            evidence = self._entry_s4_contact
            if (evidence is not None and self._entry_locked_s4_slot is slot
                    and self._byte_track_id_at(slot_index) == evidence.track_id
                    and slot.load_contact_since == evidence.contact_since):
                evidence.last_contact_at = slot.load_contact_last_seen
                evidence.released_at = slot.load_contact_released_at
        if not self.load_hand_contact_enabled:
            status = "disabled"
        elif not enabled and not slot.load_contact_pending:
            status = disabled_reason or "observations_disabled"
        elif hold_pending:
            status = "pending_contact_during_prediction" if slot.temporary_contact_box_xyxy is not None \
                else "pending_contact_waiting_for_visual"
        elif slot.load_contact_pending:
            status = "pending_contact" if contact else "released_waiting_verification"
        elif slot.load_contact_count:
            status = "collecting_contact"
        elif previous_count and not contact:
            status = "contact_interrupted_before_confirmation"
        else:
            status = "no_accepted_contact"
        diagnostic = self.s4_contact_diagnostic or {}
        diagnostic.update(contact_accepted=bool(contact and enabled and self.load_hand_contact_enabled),
            target_slot_id=slot.slot_id if self._entry_points is None else None,
            recording_enabled=bool(enabled), accumulation_status=status,
            contact_count=slot.load_contact_count, required_frames=self.load_hand_contact_min_frames,
            contact_duration_seconds=(0. if slot.load_contact_since is None else
                float(now-slot.load_contact_since)), required_seconds=self.load_hand_contact_seconds,
            evidence_ready=slot.load_contact_pending,
            visual_confirmation_required=slot.load_contact_pending and not slot.load_contact_can_infer)
        self.s4_contact_diagnostic = diagnostic
        if self.s4_region_diagnostic is not None:
            self.s4_region_diagnostic.update(contact_accepted=diagnostic["contact_accepted"],
                accumulation_status=status, contact_count=slot.load_contact_count,
                evidence_ready=slot.load_contact_pending)

    def _advance_hand_contact(self, slot, now, contact, enabled, slot_index, *, can_infer):
        if not self._is_hand_contact_slot(slot_index):
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
        if slot.load_contact_pending and slot.load_contact_released_at is not None:
            # A new gesture must not inherit the timing of an earlier edge-only contact.
            if now-slot.load_contact_released_at > self.lost_seconds or (contact and not slot.load_contact_can_infer):
                self._clear_load_contact(slot)
        if contact:
            slot.load_contact_released_at = None
            slot.load_contact_last_seen = now
            if not enabled:
                return
            if slot.load_contact_since is None:
                slot.load_contact_since = now
            slot.load_contact_count += 1
            slot.load_contact_can_infer |= can_infer
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
        slot.load_contact_can_infer = False
        slot.load_contact_released_at = None
        slot.load_contact_unknown_since = None
        slot.load_contact_unknown_count = 0
        slot.load_contact_prediction_seen_at = None
        slot.entry_contact_center = None
        slot.entry_contact_filled_frames = 0
        slot.entry_contact_saw_filled = False
        if getattr(self, "_entry_locked_s4_slot", None) is slot:
            self._clear_entry_s4_lock("contact_cleared")

    def _record_prediction_contact(self, slot, now, contact, hand_present, *,
                                   held_box=False, competing_unloads=()):
        """Accumulate consecutive prediction hits without arming the loss fallback."""
        evidence = slot.prediction_contact
        if self.identity_lost or not self.load_hand_contact_enabled:
            slot.prediction_contact = None
            return
        if evidence is not None and evidence.pending:
            if evidence.entry_preview_source and evidence.released_at is not None:
                if hand_present:
                    slot.prediction_contact = None
                return
            if evidence.started_on_held_box and evidence.released_at is not None:
                return
            # A missing prediction box does not itself signal that the hand left.
            if hand_present is True:
                evidence.released_at = None
                evidence.last_seen = float(now)
                if contact:
                    evidence.frames += 1
            elif hand_present is False and evidence.released_at is None:
                evidence.released_at = float(now)
            return
        if not contact:
            slot.prediction_contact = None
            return
        if evidence is None:
            visual_baseline = slot.confirmation_source == "yolo" or (
                self.phase == "load" and slot.slot_id != 4 and slot.confirmation_source == "previous_unload_yolo")
            baseline = slot.confirmed if visual_baseline else None
            evidence = slot.prediction_contact = PredictionContact(
                float(now), float(now), baseline, slot.confirmed, started_on_held_box=held_box)
        if competing_unloads:
            evidence.competing_unloads = tuple(sorted(set(evidence.competing_unloads) | set(competing_unloads)))
        evidence.frames += 1
        evidence.last_seen = float(now)
        evidence.pending = (evidence.frames >= self.load_hand_contact_min_frames
            and now-evidence.since+1e-9 >= self.load_hand_contact_seconds)

    def _verify_prediction_contact(self, slot, state, prior_confirmed, prior_confirmed_since,
                                   event, now):
        evidence = slot.prediction_contact
        if evidence is None:
            return event, None
        if (evidence.entry_preview_source and now-max(evidence.last_seen, evidence.released_at or evidence.last_seen)
                > self.initialization_history_seconds):
            slot.prediction_contact = None
            return event, None
        if not evidence.pending:
            return event, evidence.diagnostic()
        if state == "empty" and evidence.released_at is not None:
            if evidence.empty_since is None:
                evidence.empty_since = float(now)
            evidence.empty_frames += 1
        else:
            evidence.empty_since = None
            evidence.empty_frames = 0
        diagnostic = evidence.diagnostic()
        if event is None and (state == "unknown" or slot.state != state):
            return event, diagnostic

        # Track history restored when S4 reappears can establish that it was
        # already empty at the time of contact, even if its state was unknown then.
        already_empty = prior_confirmed == "empty" and (prior_confirmed_since is None
            or prior_confirmed_since <= evidence.since)
        infer_unload = (event is None and self.phase == "unload" and state == "empty"
            and evidence.initial_state == "unknown" and not already_empty
            and not any(e["slot_id"] == slot.slot_id and e["action"] == "unload" for e in self.events))
        if infer_unload:
            if (evidence.empty_frames < self.minimum_frames
                    or now-evidence.empty_since+1e-9 < self.confirm_seconds):
                return None, diagnostic
            if evidence.competing_unloads:
                confirmed = {(e["slot_id"], e["confirmed_at"]) for e in self.events
                             if e["action"] == "unload" and not e.get("inferred")}
                missing = set(evidence.competing_unloads)-confirmed
                if missing:
                    under_review = {(e["slot_id"], e["confirmed_at"]) for s in self.slots[:3]
                                    if (e := s.pending_visual_event) is not None}
                    if missing <= under_review:
                        return None, dict(diagnostic, status="waiting_for_competing_unloads")
                    slot.prediction_contact = None
                    return None, dict(diagnostic, pending=False, status="competing_unload_cancelled")
            event = {"slot_id": slot.slot_id, "from": "unknown", "to": "empty", "action": "unload",
                     "time_lower": evidence.since, "time_upper": evidence.empty_since,
                     "confirmed_at": float(now), "reason": "empty_after_prediction_contact",
                     "inferred": True}
            if evidence.started_on_held_box:
                event.update(time_upper=min(evidence.released_at, evidence.empty_since),
                    visual_time_upper=evidence.empty_since)
            status = "visual_empty_after_prediction_contact"
        elif event is not None:
            status = ("visual_change_confirmed" if not event.get("inferred", False)
                      and event["from"] == evidence.visual_baseline else "event_already_confirmed")
        elif state == evidence.initial_state or (state == "empty" and already_empty):
            status = "visual_state_unchanged"
        else:
            status = "visual_baseline_only"
        diagnostic.update(pending=False, visual_state=state, status=status)
        if status in ("visual_change_confirmed", "visual_empty_after_prediction_contact"):
            event["prediction_contact"] = dict(diagnostic)
        if (status == "visual_change_confirmed" and self.phase == "load"
                and (slot.slot_id != 4 or evidence.entry_preview_source)
                and event["action"] == "load" and evidence.released_at is not None
                and (event["time_lower"] is None or event["time_lower"] <= evidence.since)
                and evidence.released_at <= event["time_upper"]):
            reason = (f"visual_after_{evidence.entry_preview_source}_contact" if evidence.entry_preview_source
                      else "visual_after_prediction_contact")
            event.update(reason=reason, visual_time_upper=event["time_upper"],
                         time_upper=evidence.released_at, hand_contact_since=evidence.since,
                         hand_contact_released_at=evidence.released_at, hand_contact_frames=evidence.frames)
        slot.prediction_contact = None
        return event, diagnostic

    def update(self, objects, now, *, observations_enabled=True, initialization_objects=None,
               roi_contour=None, before_observe=None, translation_direction=None,
               s4_position_candidates=(), recheck_initial_unload=False, load_support_resolver=None,
               entry_position_candidates=()):
        """先关联几何，再确认状态；回调可在转位开始的同一帧暂停取放事件。"""
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("源视频时间必须严格递增")
        self.last_time = float(now)
        self.s4_region_diagnostic = None
        self.s4_contact_diagnostic = None
        self.geometry_rejected = []
        preview_contact = self._entry_load_preview_contact
        if (preview_contact is not None and (not observations_enabled
                or (self.three_point_preview is None and not preview_contact.evidence.pending))):
            self._entry_load_preview_contact = None
        self.three_point_preview = None
        contact = self.two_point_hand_contact
        if contact is not None and self.points is None:
            contact.prepare(self, objects if initialization_objects is None else initialization_objects,
                            now, observations_enabled)
        if self._entry_points is not None:
            initial_objects = objects if initialization_objects is None else initialization_objects
            result = self._update_incremental_entry(initial_objects, now, observations_enabled, roi_contour,
                entry_position_candidates=entry_position_candidates, load_support_resolver=load_support_resolver)
            numbered = self._try_two_point_load_numbering(now) if observations_enabled else None
            return {**self.snapshot(), "new_events": numbered} if numbered is not None else result
        self._predicted_indices.clear()
        self.transform_valid_this_frame = False
        self.transform_support_slot_ids = []
        self.transform_support_byte_track_ids = []
        self.transform_fit_error = None
        candidate_objects = objects
        if self.points is not None and self._provisional_quarantined_track_ids:
            # IDs isolated while the provisional group was being established
            # remain rejected for this group after formal numbering too.
            candidate_objects = [o for o in objects if not (
                o.get("class_name") in SLOT_STATES
                and o.get("track_id") is not None
                and int(o["track_id"]) in self._provisional_quarantined_track_ids)]
        fallback_objects = [obj for obj in candidate_objects if obj.get("local_s4_fallback")]
        candidates = self._candidates([obj for obj in candidate_objects if not obj.get("local_s4_fallback")]
                                      if fallback_objects else candidate_objects)
        hands = [o for o in objects if o["class_name"] == "hand"]
        if self.points is None:
            initial_objects = objects if initialization_objects is None else initialization_objects
            initial = self._candidates(initial_objects)
            recovered = self._initialize(initial, hands, now, observations_enabled, roi_contour,
                initial_objects, recheck_initial_unload=recheck_initial_unload,
                load_support_resolver=load_support_resolver)
            if self.points is None:
                self.unmatched = candidates
                # 候选轨迹已经能给出暂定的槽位状态；身份尚未确认时不发布
                # 正式事件，但把这些状态保留在当前帧，避免初始化等待期间
                # 的满槽检测被统一覆盖成unknown。
                if self._provisional is not None:
                    provisional = self._provisional.snapshot()
                    provisional["three_point_preview"] = self.three_point_preview
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
        if not self.identity_lost and self.transition_load_track_id is not None:
            matches = self._associate_transition_load(candidates, matches, hands, gate, now)
        if observations_enabled and not self.identity_lost and 3 not in matches and len(fallback_objects) == 1:
            matches[3] = len(candidates)
            candidates.extend(self._candidates(fallback_objects))
        self._remember_track_ids(matches, candidates, self.byte_track_ids)
        # 局部补检只补空满观测；裁剪框的偏移不改变几何轨迹，也不作为越界证据。
        geometry_matches = {i: col for i,col in matches.items()
                            if candidates[col].get("detection_source") != "local_crop"}
        if geometry_matches:
            self.last_geometry_time = now
            # 至少三个槽共同支持时才向不可见槽传播运动。
            # 两个检测框中心的抖动会被相似变换完全拟合，逐帧外推会把其余槽推离真实位置。
            indices = [i for i in geometry_matches
                       if not candidates[matches[i]]["conflict"] and not self._occluded(candidates[matches[i]],hands)]
            if 3 in indices:
                self._s4_before_recalculation = None
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
                            if self._s4_before_recalculation is not None:
                                self._s4_before_recalculation = (
                                    self._s4_before_recalculation @ affine[:, :2].T + affine[:, 2])
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
                            if self._s4_before_recalculation is not None:
                                self._s4_before_recalculation = (
                                    self._s4_before_recalculation @ matrix[:, :2].T + matrix[:, 2])
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
                    if self._s4_before_recalculation is not None:
                        self._s4_before_recalculation += shift
        # 点仍显示YOLO框中心；漏检槽显示几何参考位置，关联门限不被单框抖动越缩越小。
        self.points = self.reference_points.copy()
        for i,col in geometry_matches.items():
            self.points[i] = candidates[col]["center"]
        source_enabled = observations_enabled
        moving = translation_direction is not None
        if before_observe is not None:
            moving = not before_observe(self, candidates, geometry_matches, now)
            observations_enabled = not moving and observations_enabled
        if self._region_formal_numbering and (self.identity_lost or moving):
            self._predicted_indices.discard(3)
            self._invalidate_slot_prediction(self.slots[3],
                "identity_lost" if self.identity_lost else "rotation_or_transition")
        if contact is not None:
            contact.prepare(self, candidate_objects, now, observations_enabled)
            prediction = contact.prediction
            if (self._load_perspective_numbering is not None
                    and len(self.initialization["inferred_slot_ids"]) == 2 and prediction is not None
                    and prediction.measured_at == now and prediction.slot_ids is not None):
                for preview_index, slot_id in enumerate(prediction.slot_ids, start=2):
                    index = slot_id-1
                    if index not in matches and self.slots[index].geometry_inferred:
                        self.points[index] = self.reference_points[index] = prediction.layout[preview_index]
                        self._predicted_indices.add(index)
        recovered_col = None
        recover_index = 3
        if self.phase == "load" and self._load_numbering_reference is not None:
            recover_index = next((i for i, slot in enumerate(self.slots)
                                  if slot.geometry_inferred and i not in matches), 3)
        if observations_enabled:
            recovered_col = self._recover_slot_position(
                recover_index, candidates, matches, s4_position_candidates, hands, now, gate, roi_contour,
                allow_binding=not moving)
            if recovered_col is not None:
                matches[recover_index] = recovered_col
                self._remember_track_ids({recover_index: recovered_col}, candidates, self.byte_track_ids)
                center = candidates[recovered_col]["center"]
                self.reference_points[recover_index] = self.points[recover_index] = center
        else:
            self._position_recovery_candidate = None
            self._load_binding_frames = 0
        self.unmatched = [c for i, c in enumerate(candidates) if i not in matches.values()]
        self._update_load_contact_boxes(candidates, matches, geometry_matches, hands, now,
                                        observations_enabled and not moving and not self.identity_lost, roi_contour)
        self._observe_load_trajectory(hands, now, observations_enabled and not moving)
        new_events = []
        for i,slot in enumerate(self.slots):
            obj = candidates[matches[i]] if i in matches else None
            if moving and self._is_pending_unload_reversal(slot):
                events = self.review_departing_unload_slot(slot, obj if source_enabled else None, hands, now)
                self.events.extend(events)
                new_events.extend(events)
                continue
            # 转位时继续核验已有基线的槽位；无需先检测到手才能确认空满变化。
            verify_only = self.phase == "load" and not observations_enabled
            enabled = source_enabled and slot.confirmed in ("empty", "filled") if verify_only else observations_enabled
            review_s4 = self.phase == "unload" and i == 3 and self._region_formal_numbering
            contact_supported = review_s4 and self._has_pending_s4_contact(slot)
            # 新绑定的槽从当前画面确认状态，避免其未归属历史重复生成上料事件。
            event, history = self._observe_slot(slot, obj, hands, now, enabled, roi_contour,
                                                record_load_contact=not verify_only,
                                                restore_track_events=i != recover_index or recovered_col is None)
            if review_s4:
                events = self._review_visual_events(slot, history, event, now, review_current=False)
                # The existing formal S4 review still owns new S4 unloads.
                current = events.pop() if events and events[-1] is event else None
                events.extend(self._review_s4_unload(current, now,
                    contact_supported or self._has_pending_s4_contact(slot)))
            else:
                events = self._review_visual_events(slot, history, event, now)
            events = [self._apply_load_trajectory(e, now) for e in events]
            self.events.extend(events)
            new_events.extend(events)
        if self._entry_s4_contact is not None:
            self._observe_entry_s4_contact(candidates, geometry_matches, now, observations_enabled)
            recovered_s4 = self._recover_entry_s4_unload(now)
            if recovered_s4 is not None:
                new_events.append(recovered_s4)
        result = self.snapshot()
        result["new_events"] = new_events
        return result

    @staticmethod
    def _has_pending_s4_contact(slot):
        return slot.load_contact_pending or bool(slot.prediction_contact and slot.prediction_contact.pending)

    def _release_visual_event(self, slot, now):
        event = slot.pending_visual_event
        slot.pending_visual_event = None
        initialized_at = self.initialization.get("initialized_at")
        released = dict(event, slot_id=slot.slot_id, reported_at=float(now),
                    recovered_before_initialization=event["recovered_before_initialization"] or
                    (initialized_at is not None and event["confirmed_at"] < initialized_at))
        return self._apply_load_trajectory(self._apply_s4_load_contact_lookback(slot, released, now), now)

    def _observe_load_trajectory(self, hands, now, enabled):
        if (not self.load_trajectory_enabled or not self.load_hand_contact_enabled
                or self.phase != 'load' or self.points is None):
            return
        if self.identity_lost:
            self.load_trajectory = LoadHandTrajectory()
            return
        targets = {}
        for i, slot in enumerate(self.slots):
            if slot.confirmed != 'empty' or any(e['slot_id'] == slot.slot_id and e['action'] == 'load'
                                               for e in self.events):
                continue
            point = self.points[i]
            box = slot.temporary_contact_box_xyxy or slot.last_valid_box_xyxy
            if box is None:
                size = slot.box_size_wh
                if size is None:
                    continue
                box = centered_box(point, size)
            else:
                box = centered_box(point, (box[2]-box[0], box[3]-box[1]))
            targets[slot.slot_id] = {'center': point, 'box': box,
                'baseline_at': float(slot.evidence_time if slot.evidence_time is not None else now),
                'bound': self.byte_track_ids[i] is not None}
        self.load_trajectory.observe(hands, targets, self.initialization['inferred_slot_ids'], now, enabled)

    def _apply_load_trajectory(self, event, now):
        if not self.load_trajectory_enabled or self.phase != 'load':
            return event
        return self.load_trajectory.confirm(event, now)

    def _apply_s4_load_contact_lookback(self, slot, event, now):
        contact = self.s4_load_contact_lookback
        if (contact is None or slot.slot_id != 4 or event["action"] != "load"
                or event["from"] != "empty" or event.get("inferred")):
            return event
        self.s4_load_contact_lookback = None
        check_at = event["confirmed_at"] if contact.from_reach_crop else now
        if check_at > contact.expires_at or event["time_upper"] < contact.last_at:
            return event
        if contact.require_bound_visual and (
                not self._region_formal_numbering or self.points is None or self.identity_lost
                or event.get("byte_track_id") is None or event["byte_track_id"] != self.byte_track_ids[3]
                or slot.confirmation_source != "yolo"
                or (event.get("time_lower") is not None and event["time_lower"] > contact.first_at)):
            return event
        revised = dict(event, time_lower=contact.first_at, time_upper=contact.last_at,
                       visual_time_lower=event["time_lower"], visual_time_upper=event["time_upper"],
                       hand_contact_since=contact.first_at, hand_contact_last_at=contact.last_at,
                       hand_contact_frames=contact.frames, reason="visual_after_s4_contact_lookback")
        if contact.from_reach_crop:
            revised["hand_contact_source"] = "entry_reach_crop"
        return revised

    def _review_visual_events(self, slot, history, current, now, *, review_current=True):
        """Hold visual changes until the same track has had time to return to its prior state."""
        seconds = self.visual_reversal_review_seconds
        events = history if current is None else [*history, current]
        contact = self.s4_load_contact_lookback
        if slot.slot_id == 4 and contact is not None:
            visual = current or slot.pending_visual_event
            reviewing_filled = (contact.from_reach_crop and visual is not None
                and visual["to"] == "filled" and visual["confirmed_at"] <= contact.expires_at)
            newly_empty = (slot.state == "empty" and slot.confirmation_source == "yolo"
                and (not contact.from_reach_crop or
                     (slot.evidence_time is not None and slot.evidence_time > contact.last_at)))
            if newly_empty or (now > contact.expires_at and not reviewing_filled):
                self.s4_load_contact_lookback = None
                contact = None
        if not seconds:
            return [self._apply_s4_load_contact_lookback(slot, event, now) for event in events] if contact else events
        ready = []
        for event in events:
            visual = event["from"] in ("filled", "empty") and not event.get("inferred")
            pending = slot.pending_visual_event
            if pending is not None:
                observed_since = (slot.confirmed_since if event is current
                    and slot.confirmed_at == event["confirmed_at"] else event["time_upper"])
                if (visual and event["confirmed_at"] > pending["confirmed_at"]
                        and observed_since <= pending["confirmed_at"]+seconds+1e-9
                        and event["from"] == pending["to"] and event["to"] == pending["from"]
                        and event.get("byte_track_id") == pending.get("byte_track_id")):
                    slot.pending_visual_event = None
                    if self.track_store is not None:
                        self.track_store.discard_visual_events(event.get("byte_track_id"),
                            (self.phase, self.group_id), (pending, event))
                    continue
                ready.append(self._release_visual_event(slot, now))
            if visual and (event is not current or review_current):
                slot.pending_visual_event = event
            else:
                ready.append(event)
        pending = slot.pending_visual_event
        if pending is not None:
            deadline = pending["confirmed_at"]+seconds+1e-9
            confirming_reversal = (slot.confirmed == pending["to"] and slot.candidate == pending["from"]
                and slot.candidate_visual_track_id == pending.get("byte_track_id")
                and slot.candidate_since is not None and slot.candidate_since <= deadline)
            if now > deadline and not confirming_reversal:
                ready.append(self._release_visual_event(slot, now))
        return ready

    def _review_s4_unload(self, event, now, contact_supported):
        """Hold visual-only S4 unloads long enough to reject a quick visual reversal."""
        pending = self._pending_s4_unload
        ready = []
        if pending is not None:
            within_window = now-pending["confirmed_at"] <= S4_UNLOAD_REVIEW_SECONDS+1e-9
            if (within_window and not contact_supported and event is not None
                    and event["action"] == "load" and not event.get("inferred")):
                self._pending_s4_unload = None
                if self.s4_contact_diagnostic is not None:
                    self.s4_contact_diagnostic["unload_review"] = "visual_reversal_cancelled"
                return []
            if contact_supported or event is not None or not within_window:
                ready = self._release_s4_unload(now)
        if event is None:
            return ready
        if (event["action"] == "unload" and event["from"] == "filled"
                and not contact_supported and not event.get("inferred")):
            self._pending_s4_unload = event
            if self.s4_contact_diagnostic is not None:
                self.s4_contact_diagnostic["unload_review"] = "pending"
        else:
            ready.append(event)
        return ready

    def _release_s4_unload(self, now):
        event = self._pending_s4_unload
        self._pending_s4_unload = None
        return [] if event is None else [dict(event, reported_at=float(now))]

    def finish_s4_unload_review(self, now):
        """Publish a remaining visual unload before archiving its group."""
        events = self._release_s4_unload(now)
        self.events.extend(events)
        return events

    def _is_pending_unload_reversal(self, slot):
        event = slot.pending_visual_event
        return self.phase == "unload" and event is not None and event["from"] == "empty" and event["to"] == "filled"

    def pending_unload_reversals(self):
        return [slot for slot in self.slots if self._is_pending_unload_reversal(slot)]

    def finish_pending_unload_reversals(self):
        for slot in self.pending_unload_reversals():
            self._mark_unload_reversal_unconfirmed(slot)

    def _mark_unload_reversal_unconfirmed(self, slot):
        event = slot.pending_visual_event
        if self.track_store is not None:
            self.track_store.discard_visual_events(event.get("byte_track_id"), (self.phase, self.group_id), (event,))
        slot.pending_visual_event = None
        slot.confirmed = slot.state = "unknown"
        slot.confirmation_source = slot.confirmed_visual_track_id = None
        slot.evidence_time = slot.confirmed_since = slot.confirmed_at = None
        slot.reason = "unload_reversal_unconfirmed_at_exit"

    def review_departing_unload_slot(self, slot, obj, hands, now):
        """Resolve only an existing reversal, using fresh visual evidence from the same track."""
        pending = slot.pending_visual_event
        track_id = pending.get("byte_track_id")
        deadline = pending["confirmed_at"]+self.visual_reversal_review_seconds
        reliable = (not self.identity_lost and track_id is not None and obj is not None
                    and obj.get("track_id") == track_id and not obj["conflict"]
                    and obj.get("detection_source") != "local_crop" and not self._occluded(obj, hands))
        state = SLOT_STATES[obj["class_name"]] if reliable else "unknown"
        if reliable and slot.candidate_visual_track_id != track_id:
            slot.candidate = "unknown"
        current = slot.observe(state, now, obj["confidence"] if reliable else None,
            "detected" if reliable else "unload_reversal_waiting_for_track",
            self.confirm_seconds, self.minimum_frames, visual_track_id=track_id if reliable else None)
        if current is not None and state == "empty" and current["time_upper"] <= deadline+1e-9:
            current.update(byte_track_id=track_id)
            return self._review_visual_events(slot, [], current, now)
        if now+1e-9 < deadline:
            return []
        if slot.state == "filled" and slot.candidate_visual_track_id == track_id:
            return [self._release_visual_event(slot, now)]
        # Let an empty run that began within the window finish its normal frame count.
        if (state == "empty" and slot.candidate_since <= deadline+1e-9
                and now <= deadline+max(.25, self.confirm_seconds)+1e-9):
            return []
        self._mark_unload_reversal_unconfirmed(slot)
        return []

    def finish_visual_event_reviews(self, now, *, preserve_unload_reversals=False):
        """Publish the remaining reviewed changes of a formally mapped group."""
        if self.points is None:
            return []
        events = [self._release_visual_event(slot, now) for slot in self.slots
                  if slot.pending_visual_event is not None
                  and not (preserve_unload_reversals and self._is_pending_unload_reversal(slot))]
        self.events.extend(events)
        return events

    def infer_entry_s4_load(self, contact, now):
        """Use a released provisional contact after the empty S4 baseline is restored."""
        slot = self.slots[3]
        if slot.candidate_since is None or slot.candidate_since <= contact.released_at:
            slot.candidate, slot.candidate_since, slot.candidate_count = "unknown", None, 0
            slot.candidate_visual_track_id = None
        slot.confirmed = slot.state = "filled"
        slot.confirmed_since, slot.confirmed_at = contact.since, float(now)
        slot.evidence_time = float(now)
        slot.confirmed_visual_track_id = None
        slot.confirmation_source = "hand_contact"
        slot.reason = "two_point_hand_contact_inferred_load"
        self._clear_load_contact(slot)
        event = {"slot_id": 4, "group_id": self.group_id, "phase": "load",
            "byte_track_id": self.byte_track_ids[3], "from": "empty", "to": "filled",
            "action": "load", "expected_action": True, "inferred": True, "reason": slot.reason,
            "time_lower": contact.since, "time_upper": contact.released_at, "confirmed_at": float(now),
            "reported_at": float(now), "recovered_before_initialization": True,
            "hand_contact_since": contact.since, "hand_contact_released_at": contact.released_at,
            "hand_contact_frames": contact.frames}
        self.events.append(event)
        self.s4_inferred_load = None if self._predicted_load_is_locked(slot) else event
        return event

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
            event, history = self._observe_slot(slot, obj, hands, now, enabled, None,
                                                record_load_contact=False)
            events.extend(self._review_visual_events(slot, history, event, now))
        self.events.extend(events)
        return events

    def _slot_observation(self, obj, hands, enabled, roi_contour, *, contact=False):
        if self.identity_lost:
            return "unknown", None, "identity_lost"
        if obj is None:
            return "unknown", None, "not_detected"
        confidence = obj["confidence"]
        if not enabled:
            return "unknown", confidence, "rotation_or_transition"
        if (roi_contour is not None and not obj.get("initial_edge_slot")
                and cv2.pointPolygonTest(roi_contour, tuple(obj["center"]), False) < 0):
            return "unknown", confidence, "outside_roi"
        if obj["conflict"]:
            return "unknown", confidence, "empty_filled_conflict"
        if self._occluded(obj, hands) or contact:
            return "unknown", confidence, "hand_occlusion"
        return SLOT_STATES[obj["class_name"]], confidence, "detected"

    def _observe_slot(self, slot, obj, hands, now, enabled, roi_contour, *,
                      record_load_contact=True, restore_track_events=True):
        """正式槽与进入中的临时点共用同一套空满确认规则。"""
        slot_index = slot.slot_id-1
        load_locked = self._predicted_load_is_locked(slot)
        self._unlock_lost_entry_s4(slot, slot_index, obj, now)
        recovered = []
        if not load_locked and enabled and obj is not None and self.track_store is not None:
            recovered = self.track_store.restore(slot, obj.get('track_id'),
                (self.phase, self.group_id), now, include_events=restore_track_events and (
                    self.transition_load_track_id is None or obj.get('track_id') != self.transition_load_track_id))
        local_crop = obj is not None and obj.get("detection_source") == "local_crop"
        formal_s4 = slot_index == 3 and self._region_formal_numbering and self.points is not None
        slot.geometry_observed = obj is not None and not local_crop
        if formal_s4 and slot.geometry_observed:
            slot.geometry_observed = not obj["conflict"] and not self._occluded(obj, hands)
        has_box = obj is not None and (not formal_s4 or slot.geometry_observed or local_crop)
        if has_box:
            if slot.geometry_observed or self.phase == "load":
                slot.geometry_inferred = False
                box = obj["xyxy"]
                slot.box_size_wh = (max(1., box[2]-box[0]), max(1., box[3]-box[1]))
            slot.temporary_contact_box_xyxy = None
            if enabled and record_load_contact:
                self._remember_formal_s4_detection(slot, slot_index, obj, now,
                    source="local_crop" if local_crop and self.phase == "unload" else "detection")
        elif formal_s4 and enabled and record_load_contact:
            self._seed_formal_s4_detection_cache(now)
        temporary_contact_box = (None if has_box else
                                 self._temporary_s4_contact_box(slot, slot_index, now))
        if load_locked:
            # Preserve raw observations and geometry for diagnostics and group tracking.
            slot.observation, slot.confidence, _ = self._slot_observation(obj, hands, enabled, roi_contour)
            if slot.observation != "unknown":
                slot.last_seen = float(now)
            if slot.load_contact_count or slot.load_contact_pending:
                self._clear_load_contact(slot)
            slot.prediction_contact = None
            slot.state, slot.reason = "filled", "predicted_load_state_locked"
            return None, []
        contact = self._hand_contact(slot_index, hands)
        resumed_box = (slot.temporary_contact_box_xyxy if self.phase == "load" and slot_index != 3
                       and not slot.geometry_inferred else None)
        resumed_contact = resumed_box is not None and contact
        if resumed_contact:
            contact = False
        point = self._slot_point(slot_index)
        if self._is_hand_contact_slot(slot_index) and slot.load_contact_pending and point is not None:
            # 多槽覆盖不能建立新证据，但已经有待核验操作时，手仍覆盖
            # 此槽就不算离手，避免宽手框被误当成接触结束。
            contact = contact or any(h["xyxy"][0] <= point[0] <= h["xyxy"][2]
                and h["xyxy"][1] <= point[1] <= h["xyxy"][3] for h in hands)
        inside = roi_contour is None or (point is not None and cv2.pointPolygonTest(
            roi_contour, tuple(point), False) >= 0)
        contact_allowed = self.phase != "load" or slot_index in self.inferred_load_contact_indices()
        contact_enabled = contact_allowed and enabled and record_load_contact and inside and not self.identity_lost
        contact_diagnostic = self.s4_contact_diagnostic or {}
        prediction_contact = temporary_contact_box is not None and contact_diagnostic.get(
            "temporary_contact_detected", False)
        entry_lookback = self.s4_load_contact_lookback
        if formal_s4 and entry_lookback is not None and entry_lookback.require_bound_visual:
            # A crop-verified reach waits for a newer empty observation to disprove it.
            # The original lookback also treats a fresh full-frame contact as a separate operation.
            if self.identity_lost or (not entry_lookback.from_reach_crop
                    and slot.pending_visual_event is None and (contact or prediction_contact)):
                self.s4_load_contact_lookback = None
        if self.identity_lost and slot.load_contact_prediction_seen_at is not None:
            self._clear_load_contact(slot)
        if contact or obj is not None or (contact_enabled and temporary_contact_box is not None
                                         and not prediction_contact):
            slot.load_contact_prediction_seen_at = None
        if (contact_enabled and prediction_contact and not contact and slot.load_contact_pending
                and slot.load_contact_released_at is None):
            slot.load_contact_prediction_seen_at = float(now)
        # 已在预测框上延续的接触，失去框后短时等待视觉核验。只有独立接触
        # 能累计确认帧，框失效本身也不能开始离手计时或产生空满翻转。
        prediction_seen_at = slot.load_contact_prediction_seen_at
        if prediction_seen_at is not None and now-prediction_seen_at > self.lost_seconds:
            self._clear_load_contact(slot)
        had_contact = bool(slot.load_contact_count or slot.load_contact_pending)
        source = contact_diagnostic.get("source")
        # A known-empty slot hidden in a three-point layout has independent geometry and
        # state evidence. Require an extra contact frame when only its projected box is hit.
        sustained_inferred_contact = (source == "predicted_slot_box" and slot.geometry_inferred
            and slot.slot_id in self._load_verified_empty_slots and slot.load_contact_released_at is None
            and slot.load_contact_count >= self.load_hand_contact_min_frames)
        can_infer_contact = (self.phase != "load" or sustained_inferred_contact
            or source in ("s4_center", "s4_region", "predicted_slot_center"))
        if ((slot.prediction_contact is not None and slot.prediction_contact.entry_preview_source)
                or (formal_s4 and entry_lookback is not None and entry_lookback.require_bound_visual)):
            can_infer_contact = False
        self._record_hand_contact(slot, now, contact, contact_enabled, slot_index,
                                  hold_pending=slot.load_contact_prediction_seen_at is not None,
                                  can_infer=can_infer_contact,
                                  disabled_reason=(
                                      "identity_lost" if self.identity_lost else
                                      "outside_roi" if not inside else
                                      "contact_recording_paused" if not record_load_contact else
                                      "observations_disabled"))
        if prediction_contact or resumed_contact or slot.prediction_contact is not None:
            hand_present = None
            if contact_enabled and slot.prediction_contact is not None and slot.prediction_contact.pending:
                visible_box = obj["xyxy"] if has_box and not obj["conflict"] else resumed_box or temporary_contact_box
                if visible_box is not None:
                    hand_present = contact or resumed_contact or any(box_overlap_ratio(visible_box, hand["xyxy"])
                        >= (self.s4_prediction_contact_overlap_ratio if resumed_box is not None
                            else self.load_hand_contact_overlap_ratio) for hand in hands)
            strong_prediction = (prediction_contact and contact_diagnostic["temporary_contact_overlap_ratio"]
                                 >= self.s4_prediction_contact_overlap_ratio)
            self._record_prediction_contact(slot, now,
                contact_enabled and (strong_prediction or resumed_contact) and not contact and not slot.load_contact_pending,
                hand_present, held_box=resumed_box is not None or contact_diagnostic.get("temporary_contact_box_held", False),
                competing_unloads=contact_diagnostic.get("temporary_contact_competing_unloads", ()))
        if temporary_contact_box is not None and not contact and not prediction_contact and had_contact:
            slot.temporary_contact_box_reason = "hand_left"
        contact_slot_for_frame = self._is_hand_contact_slot(slot_index)
        pending_contact = slot.load_contact_pending
        contact_evidence = None
        if (pending_contact and slot.confirmed == "unknown"
                and slot.load_contact_released_at is not None):
            contact_evidence = {"since": slot.load_contact_since,
                                "released_at": slot.load_contact_released_at,
                                "count": slot.load_contact_count}
        state, confidence, reason = self._slot_observation(obj, hands, enabled, roi_contour, contact=contact)
        if self.load_trajectory_enabled and self.phase == 'load':
            if state == 'empty':
                self.load_trajectory.observe_empty(slot.slot_id, now, self.minimum_frames)
            else:
                self.load_trajectory.break_empty_run(slot.slot_id)
        verify_entry_full = (self._entry_points is not None and pending_contact
                             and slot.confirmed == "filled" and slot.entry_contact_center is not None)
        if verify_entry_full:
            if contact:
                slot.entry_contact_saw_filled = False
            if state == "filled":
                layout = np.asarray(self._entry_points).copy()
                layout[slot_index] = obj["center"]
                order = self._entry_numbering_order(layout)
                if (obj.get("detection_source") == "local_crop"
                        or np.linalg.norm(layout[slot_index]-slot.entry_contact_center) > self._entry_gate*.5
                        or not valid_slot_geometry(layout)
                        or order is None or order[3] != slot_index):
                    state, reason = "unknown", "entry_s4_filled_position_mismatch"
                else:
                    slot.entry_contact_saw_filled = True
            slot.entry_contact_filled_frames = (slot.entry_contact_filled_frames+1
                                                if state == "filled" else 0)
        if (state == "empty" and slot.confirmed == "unknown"
                and (edge_filled_at := obj.get("edge_filled_at")) is not None):
            slot.confirmed = "filled"
            slot.confirmed_since = slot.confirmed_at = slot.evidence_time = edge_filled_at
            slot.confirmation_source = "yolo"
        prior_confirmed = slot.confirmed
        prior_confirmed_since = slot.confirmed_since
        missing_since_before = slot.missing_since
        # A missing detection is not itself an unload.  Keep the last
        # confirmed baseline and remember the first empty observation so a
        # later confirmed empty state can retain its earlier event time.
        if state == "unknown":
            if slot.confirmed in ("filled", "empty") and slot.missing_since is None:
                slot.missing_since = float(now)
            if (slot.empty_evidence_since is not None
                    and slot.empty_evidence_last_seen is not None
                    and now-slot.empty_evidence_last_seen > self.lost_seconds):
                slot.empty_evidence_since = None
                slot.empty_evidence_last_seen = None
        elif state == "filled":
            # The slot returned to its previous filled state; cancel any
            # pending empty interpretation accumulated before the gap.
            slot.missing_since = None
            slot.empty_evidence_since = None
            slot.empty_evidence_last_seen = None
        elif state == "empty":
            slot.missing_since = None
            if self.phase == "unload" and prior_confirmed == "filled":
                if slot.empty_evidence_since is None:
                    slot.empty_evidence_since = float(now)
                slot.empty_evidence_last_seen = float(now)
            else:
                slot.empty_evidence_since = None
                slot.empty_evidence_last_seen = None
        target = "empty" if self.phase == "unload" else "filled"
        if contact_evidence is not None:
            inferred_action = "load" if state == "filled" else "unload" if state == "empty" else None
            if inferred_action != self.phase:
                contact_evidence = None
        two_point = self.two_point_hand_contact
        two_point_evidence = (two_point.evidence(obj, slot.slot_id, now)
                              if contact_allowed and two_point is not None and state != "unknown" else None)
        if (two_point_evidence is not None and not two_point_evidence["requires_visual_load"]
                and state == target and prior_confirmed == "unknown"
                and (slot.pending_visual_event or {}).get("action") != self.phase
                and not any(e["action"] == self.phase for e in recovered)
                and not any(e["slot_id"] == slot.slot_id and e["action"] == self.phase
                            for e in (self._entry_events if self._entry_points is not None else self.events))):
            contact_evidence = two_point_evidence
        minimum_frames = self.minimum_frames
        if (self.phase == "load" and contact_enabled and not contact
                and prior_confirmed == "empty" and state == "filled"):
            prediction = slot.prediction_contact
            if ((pending_contact and slot.load_contact_released_at is not None)
                    or (prediction is not None and prediction.pending and prediction.released_at is not None)
                    or (two_point_evidence is not None and two_point_evidence["released_at"] is not None)):
                minimum_frames = min(minimum_frames, LOAD_POST_CONTACT_CONFIRM_FRAMES)
        event = slot.observe(state, now, confidence, reason, self.confirm_seconds,
                             minimum_frames, contact_evidence=contact_evidence,
                             visual_track_id=None if obj is None or local_crop else obj.get("track_id"))
        if (slot.slot_id == 4 and self.s4_inferred_load is not None
                and slot.state in ("filled", "empty")):
            if slot.state == "empty":
                self.events.remove(self.s4_inferred_load)
                self.cancelled_s4_inferred_load = self.s4_inferred_load
                event = None
            self.s4_inferred_load = None
        # 直接 YOLO 状态变化优先；目标状态仍可由视觉确认并使用正常事件路径。
        if pending_contact and state == "unknown" and slot.state == "unknown":
            slot.reason = "hand_contact_pending"
        if (pending_contact and not contact and state in ("empty", "filled") and state != target
                and (not verify_entry_full or slot.entry_contact_filled_frames >= 2)):
            # 临时 S4 必须连续两帧在同一槽位观测到满槽，才能取消下料待核验。
            if verify_entry_full:
                evidence = self._entry_s4_contact
                if evidence is not None and slot.load_contact_since == evidence.contact_since:
                    self._entry_s4_contact = None
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
        if (pending_contact and not verify_entry_full and not contact and prior_confirmed in ("empty", "filled")
                and slot.state == prior_confirmed and slot.confirmed == prior_confirmed):
            self._clear_load_contact(slot)
        elif event is not None:
            if pending_contact and event["action"] == self.phase and not event.get("inferred", False):
                event.update(reason="detected_after_hand_contact", inferred=False)
                if (minimum_frames < self.minimum_frames and formal_s4
                        and slot.load_contact_can_infer and slot.load_contact_released_at is not None
                        and slot.load_contact_released_at <= event["time_upper"]
                        and self._s4_region_expected()):
                    # Preserve the eligible S4 operation time when vision confirms before the contact fallback.
                    event.update(visual_time_upper=event["time_upper"],
                                 time_lower=slot.load_contact_since,
                                 time_upper=slot.load_contact_released_at,
                                 hand_contact_since=slot.load_contact_since,
                                 hand_contact_released_at=slot.load_contact_released_at,
                                 hand_contact_frames=slot.load_contact_count)
            if pending_contact or event.get("inferred"):
                self._clear_load_contact(slot)
        elif (pending_contact and slot.load_contact_can_infer and not contact
              and slot.load_contact_released_at is not None
              and not (slot.slot_id == 4 and self.s4_inferred_load is not None)
              and not slot.entry_contact_saw_filled
              and prior_confirmed in ("empty", "filled") and state == "unknown"
              and reason == "not_detected" and slot.load_contact_unknown_since is not None
              and slot.load_contact_unknown_count >= 1
              and now-slot.load_contact_released_at+1e-9 >= min(
                  self.lost_seconds * .75,
                  max(.5, self.confirm_seconds * 2, self.load_hand_contact_seconds))):
            # Repeated contact on an occluded S4 needs visual evidence to
            # reverse an inferred load (or an early inferred unload).
            if (self.phase == "load" and slot_index == 3 and prior_confirmed == "filled"
                    and slot.confirmation_source == "hand_contact"):
                self._clear_load_contact(slot)
            elif (self.phase == "unload" and self._entry_points is None and slot_index == 3
                    and prior_confirmed == "empty" and slot.confirmation_source == "hand_contact"
                    and any(old["slot_id"] == 4 and old.get("recovered_before_initialization")
                            and old.get("reason") == "hand_contact_inferred_unload"
                            for old in self.events)):
                self._clear_load_contact(slot)
            else:
                event = self._infer_hand_contact_toggle(slot, prior_confirmed, now)
        if event is not None and slot.load_contact_pending:
            if pending_contact and not event.get("inferred", False):
                event.update(reason="detected_after_hand_contact", inferred=False)
            self._clear_load_contact(slot)
        if (event is not None and not event.get("inferred", False)
                and event.get("action") == "unload"
                and event.get("from") == "filled"
                and slot.empty_evidence_since is not None):
            # The first empty frame is the earliest visual evidence.  The
            # later confirmation frame only validates it; using that later
            # time would incorrectly place this unload after another slot.
            evidence_since = float(slot.empty_evidence_since)
            event["time_upper"] = min(float(event["time_upper"]), evidence_since)
            event["empty_evidence_since"] = evidence_since
            if missing_since_before is not None:
                event["detection_gap_since"] = float(missing_since_before)
            event["reason"] = "empty_after_detection_gap"
            slot.empty_evidence_since = None
            slot.empty_evidence_last_seen = None
        if obj is not None and (edge_at := obj.get("edge_first_empty_at")) is not None:
            for visual in recovered if event is None else [*recovered, event]:
                if (visual["action"] == "unload" and visual["from"] == "filled"
                        and not visual.get("inferred") and obj["edge_filled_at"] <= edge_at < visual["time_upper"]):
                    visual.update(time_lower=obj["edge_filled_at"], visual_time_upper=visual["time_upper"],
                                  time_upper=edge_at, reason="roi_edge_visual_confirmation")
        prediction_diagnostic = None
        if contact_allowed:
            event, prediction_diagnostic = self._verify_prediction_contact(
                slot, state, prior_confirmed, prior_confirmed_since, event, now)
        if two_point_evidence is not None and slot.state == state:
            visual_match = (not two_point_evidence["requires_visual_load"] or (
                event is not None and event["from"] == "empty" and not event.get("inferred")
                and event["time_lower"] is not None and event["time_lower"] <= two_point_evidence["since"]
                and two_point_evidence["released_at"] <= event["time_upper"]))
            if (event is not None and state == target
                    and visual_match and event.get("reason") not in (
                        "visual_after_three_point_preview_contact", "visual_after_two_point_preview_contact")):
                # Keep the real observation time for audit and use the end of
                # the verified contact as the estimated operation time.
                visual_at = event["time_upper"]
                lower = event["time_lower"]
                estimate = min(visual_at, max(two_point_evidence["released_at"],
                                              lower if lower is not None else two_point_evidence["since"]))
                event.update(reason="visual_after_two_point_hand_contact", time_upper=estimate,
                             visual_time_upper=visual_at, hand_contact_since=two_point_evidence["since"],
                             hand_contact_released_at=two_point_evidence["released_at"],
                             hand_contact_frames=two_point_evidence["count"])
            two_point.resolve(obj, state, event)
        if event is not None:
            event.update(group_id=self.group_id, phase=self.phase,
                byte_track_id=self._byte_track_id_at(slot.slot_id-1),
                expected_action=event["action"] == self.phase,
                reported_at=float(now), recovered_before_initialization=False)
        if contact_slot_for_frame and self.s4_contact_diagnostic is not None:
            diagnostic = self.s4_contact_diagnostic
            diagnostic["prediction_contact"] = prediction_diagnostic
            if verify_entry_full:
                diagnostic["entry_s4_filled_confirmation"] = reason if reason == "entry_s4_filled_position_mismatch" \
                    else slot.entry_contact_filled_frames
            if temporary_contact_box is not None:
                diagnostic.update(temporary_contact_box_active=slot.temporary_contact_box_xyxy is not None,
                                 temporary_contact_box_reason=slot.temporary_contact_box_reason,
                                 temporary_contact_box_age_seconds=round(
                                     float(now-slot.last_valid_box_at), 6))
            elif self.points is not None and slot.last_valid_box_at is not None:
                diagnostic.update(
                    temporary_contact_box_active=False,
                    temporary_contact_box_reason=slot.temporary_contact_box_reason,
                    temporary_contact_box_age_seconds=round(
                        float(now-slot.last_valid_box_at), 6))
            if event is not None:
                diagnostic.update(accumulation_status="event_confirmed", event_reason=slot.reason,
                                  confirmation_source=slot.confirmation_source)
            elif diagnostic.get("evidence_ready") and not slot.load_contact_pending:
                diagnostic["accumulation_status"] = "cancelled_by_visual_state"
            diagnostic.update(contact_count=slot.load_contact_count, evidence_ready=slot.load_contact_pending)
            if self.s4_region_diagnostic is not None:
                self.s4_region_diagnostic.update(accumulation_status=diagnostic["accumulation_status"],
                    contact_count=slot.load_contact_count, evidence_ready=slot.load_contact_pending)
        return event, recovered

    def _infer_hand_contact_toggle(self, slot, prior_confirmed, now):
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
        return event

    def order_result(self):
        return slot_event_order_result(self.events, self.clockwise)

    def snapshot(self):
        target = "filled" if self.phase == "load" else "empty"
        slots = []
        sizes = [slot.box_size_wh for slot in self.slots if slot.box_size_wh is not None]
        typical_size = None if not sizes else np.median(sizes, axis=0)
        track_ids = (self.byte_track_ids if self.points is not None else self._entry_track_ids)
        for i,s in enumerate(self.slots):
            point = (self.points[i] if self.points is not None else
                self._entry_points[i] if self._entry_points is not None and i < len(self._entry_points) else None)
            byte_track_id = track_ids[i] if i < len(track_ids) else None
            track_description = (self.track_store.describe(byte_track_id)
                                 if self.track_store is not None and byte_track_id is not None else None)
            size = s.box_size_wh if s.box_size_wh is not None else typical_size
            tracking_box = None if point is None or size is None else [
                round(float(point[0]-size[0]/2), 2), round(float(point[1]-size[1]/2), 2),
                round(float(point[0]+size[0]/2), 2), round(float(point[1]+size[1]/2), 2)]
            is_prediction = point is not None and i in self._predicted_indices and not s.geometry_observed
            held_position = (s.temporary_contact_box_xyxy is not None
                             and s.temporary_contact_box_reason == "held_position") or (
                             i == 3 and self._region_formal_numbering and self.points is not None
                             and not self.identity_lost and not s.geometry_observed
                             and s.last_valid_box_xyxy is not None and (s.temporary_contact_box_xyxy is None
                                 or s.temporary_contact_box_reason in ("held_position", "hand_left")))
            if s.temporary_contact_box_xyxy is not None:
                is_prediction = True
                if s.last_valid_box_source != "local_crop":
                    tracking_box = [round(float(value), 2)
                                    for value in s.temporary_contact_box_xyxy]
                    point = np.asarray(s.temporary_contact_box_xyxy).reshape(2, 2).mean(axis=0)
            elif held_position:
                is_prediction = True
                if s.last_valid_box_source != "local_crop":
                    tracking_box = [round(float(value), 2) for value in s.last_valid_box_xyxy]
                    point = np.asarray(s.last_valid_center_xy)
            elif i == 3 and not s.geometry_observed and s.temporary_contact_box_reason in (
                    "identity_lost", "geometry_unavailable", "rotation_or_transition"):
                point = None
                is_prediction = False
            if not s.geometry_observed and not is_prediction:
                tracking_box = None
            formal_slot_id = s.slot_id if self.points is not None else None
            pending = s.pending_visual_event
            slots.append({"group_id": self.group_id, "slot_id": formal_slot_id,
                "formal_slot_id": formal_slot_id, "internal_slot_index": i+1,
                "byte_track_id": byte_track_id,
                "logical_track_id": (track_description.get("logical_track_id", byte_track_id)
                                      if track_description is not None else byte_track_id),
                "reidentified_from_track_id": (track_description.get("reidentified_from_track_id")
                                                if track_description is not None else None),
                "reid_gap_seconds": (track_description.get("reid_gap_seconds")
                                      if track_description is not None else None),
                "reid_method": (track_description.get("reid_method")
                                 if track_description is not None else None),
                "center_xy": None if point is None else [round(float(v),2) for v in point],
                "state": "unknown" if pending else s.state, "observation": s.observation,
                "last_confirmed_state": pending["from"] if pending else s.confirmed,
                "last_confirmed_at": pending["time_lower"] if pending else s.evidence_time,
                "last_seen_at": s.last_seen,
                "confirmed_state_since": None if pending else s.confirmed_since,
                "confirmed_state_first_confirmed_at": None if pending else s.confirmed_at,
                "confidence": s.confidence,
                "reason": "visual_reversal_review_pending" if pending else s.reason,
                "confirmation_source": None if pending else s.confirmation_source,
                "visual_reversal_review_pending": pending is not None,
                "load_state_locked": self._predicted_load_is_locked(s),
                "geometry_inferred": s.geometry_inferred,
                "position_is_prediction": is_prediction,
                "position_is_held": held_position,
                "tracking_box_xyxy": tracking_box,
                "last_valid_box_xyxy": (None if s.last_valid_box_xyxy is None else
                    [round(float(value), 2) for value in s.last_valid_box_xyxy]),
                "last_valid_center_xy": (None if s.last_valid_center_xy is None else
                    [round(float(value), 2) for value in s.last_valid_center_xy]),
                "last_valid_box_at": s.last_valid_box_at,
                "last_valid_box_source": s.last_valid_box_source,
                "temporary_contact_box_xyxy": (None if s.temporary_contact_box_xyxy is None else
                    [round(float(value), 2) for value in s.temporary_contact_box_xyxy]),
                "temporary_contact_box_reason": s.temporary_contact_box_reason,
                "load_contact_pending": s.load_contact_pending,
                "load_contact_count": s.load_contact_count,
                "load_contact_since": s.load_contact_since,
                "load_contact_last_seen": s.load_contact_last_seen,
                "load_contact_released_at": s.load_contact_released_at,
                "load_contact_unknown_since": s.load_contact_unknown_since,
                "load_contact_unknown_count": s.load_contact_unknown_count,
                "prediction_contact": None if s.prediction_contact is None else s.prediction_contact.diagnostic(),
                "missing_since": s.missing_since,
                "empty_evidence_since": s.empty_evidence_since,
                "empty_evidence_last_seen": s.empty_evidence_last_seen})
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
                "event_ordering": "time_upper",
                "s4_contact_diagnostic": self.s4_contact_diagnostic,
                "s4_unload_review_pending": self._pending_s4_unload is not None,
                "s4_inferred_load_pending": self.s4_inferred_load is not None,
                "hand_trajectory_review": self.load_trajectory.snapshot(),
                "hand_trajectory_review_pending": bool(self.load_trajectory.pending),
                "s4_contact_prediction_seconds": self.s4_contact_prediction_seconds,
                "s4_contact_lock": self._entry_s4_lock_snapshot(),
                "three_point_preview": self.three_point_preview if self.points is None else None,
                "two_point_hand_contacts": (self.two_point_hand_contact.diagnostics
                                            if self.two_point_hand_contact is not None else []),
                "s4_region_diagnostic": self.s4_region_diagnostic,
                "s4_region_expected": self._s4_region_expected(),
                "confirmed_event_sequence": [e["slot_id"] for e in sort_events_by_time_upper(self.events)
                    if e["expected_action"]],
                "current_target_count": sum(s["state"] == target for s in slots),
                "last_verified_target_count": sum(s["last_confirmed_state"] == target for s in slots),
                "all_four_currently_confirmed": all(s["state"] == target for s in slots),
                "all_four_target_previously_verified": all(s["last_confirmed_state"] == target for s in slots),
                "unassigned_slot_detections": self.unmatched}
