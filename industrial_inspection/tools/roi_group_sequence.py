"""同一ROI内按顺序处理五组，可自动衔接下料和上料；以真实越界切组。"""
from __future__ import annotations

from collections import deque
from copy import deepcopy

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from industrial_inspection.tools.slot_group_tracker import (
    SLOT_STATES, box_containment, box_iou, slot_layout_error, valid_slot_geometry)
from industrial_inspection.tools.slot_s4_load_lookback import S4LoadLookback
from industrial_inspection.tools.slot_track_state import InitialEdgeSlot, SlotTrackStateStore
from industrial_inspection.tools.slot_two_point_contact import TwoPointHandContact


class RoiGroupSequence:
    """包裹四槽跟踪器，管理初始化、操作、退出、等待下一组和工序结束。

    组号是从已知起始组号按转位顺序计数，并非靠外观识别物理组号。
    contour 使用原图像素坐标；不要求新增ROI或设备信号。
    """

    def __init__(self, tracker, contour, *, enabled=True, buffer_px=40,
                 boundary_margin_px=4, exit_seconds=.3, exit_min_slots=3,
                 partial_exit_slots=2, minimum_motion_ratio=.5,
                 motion_pause_ratio=.18, entry_seconds=.5,
                 entry_stability_ratio=.12, entry_match_distance_px=40, exit_direction=None,
                 auto_load_after_unload=False, entry_birth_fraction=.6,
                 reidentification_enabled=False, reidentification_max_gap_seconds=15.0,
                 reidentification_max_distance_px=40.0,
                 reject_transition_roi_detections=True, transition_quarantine_min_frames=4,
                 unload_to_load_quarantine_min_frames=None,
                 require_first_unload_event_for_g1=False,
                 outside_mapping_quarantine_seconds=3.0,
                 use_previous_s4_layouts=False, edge_evidence_px=0.0,
                 unload_entry_upper_right_band=(0., 0.), two_point_extension_multiplier=0.,
                 two_point_hand_hold_seconds=.8, load_numbering_calibration=None, two_point_upper_s3=False,
                 load_orbit_rotation_degrees=None):
        self.tracker = tracker
        self.contour = np.asarray(contour,dtype=np.float32)
        if self.contour.ndim != 2 or self.contour.shape[1] != 2 or not cv2.isContourConvex(self.contour):
            raise ValueError("换组ROI必须为凸多边形像素坐标")
        if not 0 < boundary_margin_px < buffer_px or exit_seconds <= 0 or entry_seconds <= 0:
            raise ValueError("换组缓冲带必须大于边界容差；退出/进入确认时间须大于0")
        if not np.isfinite(edge_evidence_px) or edge_evidence_px < 0:
            raise ValueError("ROI边缘取证距离必须为非负有限像素值")
        if not 1 <= partial_exit_slots <= exit_min_slots <= 4:
            raise ValueError("完整/部分可见时的越界槽数应满足1<=部分<=完整<=4")
        if not 0 < motion_pause_ratio < minimum_motion_ratio or not 0 < entry_stability_ratio < .5:
            raise ValueError("运动暂停比例须小于退出位移比例；停稳比例须在0..0.5之间")
        if not np.isfinite(entry_match_distance_px) or entry_match_distance_px <= 0:
            raise ValueError("新组逐点匹配距离须为正有限像素值")
        if not np.isfinite(entry_birth_fraction) or not 0 < entry_birth_fraction <= 1:
            raise ValueError("新轨迹入口深度比例须在0..1之间")
        if unload_to_load_quarantine_min_frames is None:
            unload_to_load_quarantine_min_frames = transition_quarantine_min_frames
        for frames in (transition_quarantine_min_frames, unload_to_load_quarantine_min_frames):
            if not isinstance(frames, (int, np.integer)) or isinstance(frames, bool) or frames < 1:
                raise ValueError("旧组未绑定轨迹的隔离出现帧数须为正整数")
        self.enabled, self.buffer_px, self.margin = enabled, buffer_px, boundary_margin_px
        self.auto_load_after_unload = bool(auto_load_after_unload)
        self.exit_seconds, self.exit_min_slots = exit_seconds, exit_min_slots
        self.partial_exit_slots, self.minimum_motion_ratio = partial_exit_slots, minimum_motion_ratio
        self.motion_pause_ratio, self.entry_seconds = motion_pause_ratio, entry_seconds
        self.entry_stability_ratio = entry_stability_ratio
        self.entry_match_distance_px = float(entry_match_distance_px)
        self.entry_birth_fraction = float(entry_birth_fraction)
        band_start, band_width = map(float, unload_entry_upper_right_band)
        if not np.isfinite([band_start, band_width]).all() or min(band_start, band_width) < 0:
            raise ValueError("下料入组右上窄带的起点和宽度必须为非负有限像素值")
        self.unload_entry_upper_right_band = band_start, band_width
        orientation = 1 if cv2.contourArea(self.contour, oriented=True) > 0 else -1
        self._upper_roi_edges = [(start, end) for start, end in
            zip(self.contour, np.roll(self.contour, -1, axis=0))
            if orientation*(end[0]-start[0]) > abs(end[1]-start[1])]
        if two_point_extension_multiplier:
            tracker.two_point_hand_contact = TwoPointHandContact(
                two_point_extension_multiplier, self._upper_roi_edges, self.contour,
                hold_seconds=two_point_hand_hold_seconds, upper_slot_three=two_point_upper_s3,
                orbit_degrees=load_orbit_rotation_degrees)
        self.reject_transition_roi_detections = bool(reject_transition_roi_detections)
        self.transition_quarantine_min_frames = int(transition_quarantine_min_frames)
        self.unload_to_load_quarantine_min_frames = int(unload_to_load_quarantine_min_frames)
        self.require_first_unload_event_for_g1 = bool(require_first_unload_event_for_g1)
        if (not np.isfinite(outside_mapping_quarantine_seconds)
                or outside_mapping_quarantine_seconds < 0):
            raise ValueError("映射后ROI外框隔离时间必须是非负有限秒数")
        self.outside_mapping_quarantine_seconds = float(outside_mapping_quarantine_seconds)
        self._entry_rejected = []
        self._entry_revision = 0
        self.configured_direction = None
        if exit_direction is not None:
            direction = np.asarray(exit_direction,dtype=float)
            if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) <= 0:
                raise ValueError("EXIT_DIRECTION须为非零二维向量或None")
            self.configured_direction = direction/np.linalg.norm(direction)
        self.state = "initializing" if tracker.points is None else "active"
        self.anchor = None
        self.anchor_distance = None
        self.anchor_points = None
        self.anchor_spacing = None
        self.inside_seen = {}
        self.crossed = {}
        self.visible_history = deque()
        self.center_history = deque()
        self.direction = None
        self.exit_since = None
        self.return_since = None
        self.closed_groups, self.all_events = [], []
        self.cancelled_s4_inferred_loads = 0
        self.revised_transition_loads = 0
        self.pending_tracker = None
        self.departing_points = None
        self.departing_seen = None
        # Indices that have no real old-group identity.  These points may be
        # be kept for exclusion, but must not consume an incoming detection
        # after the group has closed. In particular, an inferred S4 must not
        # be used to recover a new ByteTrack ID into the closed group.
        self.departing_untrackable_indices = set()
        self.departing_group_id = None
        self.departing_gate = None
        # Retain the last real exit geometry for diagnostics and compatibility;
        # it is no longer extrapolated or used to create exclusion predictions.
        self.closed_at, self.departed_points, self.departed_velocity = None, None, None
        self.last_time = None
        self._group_events = []
        self._required_crossings = exit_min_slots
        self._entry_count = 0
        self._motion_support = []
        self.departing_load_tracker = None
        self.departing_load_deadline = None
        self._deferred_new_events = []
        self._departing_unload_reviews = {}
        self._observations_enabled = True
        self._restored_load_groups = set()
        self.track_store = SlotTrackStateStore(
            tracker, self.contour,
            edge_evidence_px=edge_evidence_px,
            reidentification_enabled=reidentification_enabled,
            reidentification_max_gap_seconds=reidentification_max_gap_seconds,
            reidentification_max_distance_px=reidentification_max_distance_px)
        tracker.track_store = self.track_store
        self.departing_track_ids = []
        self.departing_phase = None
        self._track_rejections = {}
        # 按工序和组号保留所有曾进入ROI的槽框，包括未绑定槽号的ID。
        # 这里只记录历史；当前组内漏检后恢复时不因此拒绝该ID。
        self._group_roi_tracks = {}
        # 换组时隔离仍在ROI内或短暂漏检的稳定未绑定轨迹；空间限制短时生效。
        self._transition_quarantine_track_ids = set()
        self._transition_quarantine_centers = []
        self._transition_quarantine_until = None
        self._transition_recheck_ids = set()
        self._transition_recheck = None
        self._transition_entry_track = None
        self._transition_load_review = None
        self._s4_load_lookback = S4LoadLookback(self.contour, self.entry_match_distance_px)
        self._s4_load_hint = False
        self.mapping_time = None
        self._g1_waiting_event = bool(
            self.require_first_unload_event_for_g1
            and tracker.phase == "unload" and tracker.group_id == 1)
        self._g1_pending_events = []
        self._g1_edge_slot = None
        self._g1_interior_quartet_review = None
        self._g1_rejected_edge_ids = set()
        # IDs observed outside the ROI during the short handoff window are
        # permanently rejected for this group.  The set is group-scoped and
        # is cleared when the next group is armed.
        self._outside_mapping_quarantine_track_ids = set()
        self._outside_mapping_quarantine_group = (tracker.phase, tracker.group_id)
        self._reused_id_candidates = {}
        self._load_rebind_pose = None
        self._pending_load_rebind = None
        self.use_previous_s4_layouts = bool(use_previous_s4_layouts)
        self._unload_layouts = {}
        self._unload_orbit_layouts = {}
        self._layout_samples = deque(maxlen=3)
        self._layout_sample_key = None
        self.load_numbering_calibration = load_numbering_calibration
        self._g1_load_layout_samples = deque(maxlen=3)
        self._g1_load_layout_key = None
        self._g1_load_layout = None
        self._g1_load_layout_inferred_slot_ids = ()
        if tracker.phase == "load":
            tracker._load_perspective_numbering = load_numbering_calibration

    def _distance(self, point):
        return cv2.pointPolygonTest(self.contour,tuple(float(v) for v in point),True)

    def is_retired_track(self, track_id):
        """Return whether a ByteTrack ID belongs to a closed previous group."""
        if track_id is None:
            return False
        owner = self.track_store.owner(int(track_id))
        return owner in self.track_store.closed

    def is_transition_quarantined_track(self, track_id):
        """Return whether an unassigned track has stable old-group ROI evidence."""
        return (track_id is not None and int(track_id) in self._transition_quarantine_track_ids)

    def _entry_progress(self, point):
        """沿已核实转位方向计算ROI内进度：入口为0，出口为1。"""
        if self.configured_direction is None:
            return None
        polygon = self.contour.astype(float)
        edges = np.roll(polygon, -1, axis=0)-polygon
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        normals *= 1 if cv2.contourArea(self.contour, oriented=True) > 0 else -1
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        signed = np.sum((np.asarray(point)-polygon)*normals, axis=1)
        flow = normals @ self.configured_direction
        entering, leaving = flow > 1e-6, flow < -1e-6
        entry_distance = np.min(signed[entering]/flow[entering])
        exit_distance = np.min(signed[leaving]/-flow[leaving])
        return float(entry_distance/max(entry_distance+exit_distance, 1e-9))

    def _in_unload_entry_upper_right_band(self, center):
        start_x, width = self.unload_entry_upper_right_band
        if width == 0 or center[0] < start_x:
            return False
        for start, end in self._upper_roi_edges:
            left, right = sorted((start, end), key=lambda point: point[0])
            if left[0] <= center[0] <= right[0]:
                dx, dy = right-left
                distance = (dx*(center[1]-left[1])-dy*(center[0]-left[0])) / np.hypot(dx, dy)
                if -1e-6 <= distance <= width+1e-6:
                    return True
        return False

    def _in_initial_edge_strip(self, center):
        depth = self._distance(center)
        if not -self.track_store.edge_evidence_px <= depth < 0:
            return False
        return any(min(start[0], end[0]) <= center[0] <= max(start[0], end[0])
                   and (end[0]-start[0])*(center[1]-start[1]) <
                       (end[1]-start[1])*(center[0]-start[0])
                   for start, end in (sorted(edge, key=lambda point: point[0])
                                      for edge in self._upper_roi_edges))

    def _admit_initial_edge_slot(self, objects, untracked, now):
        tracker = self.tracker
        if (tracker.phase, tracker.group_id) == ('unload', 1) and self._g1_rejected_edge_ids:
            retained = []
            for obj in objects:
                if obj['class_name'] in SLOT_STATES and obj.get('track_id') in self._g1_rejected_edge_ids:
                    self._reject_track(obj, 'initial_edge_replaced_by_interior_quartet')
                else:
                    retained.append(obj)
            objects = retained
        if (not self._observations_enabled or not self.require_first_unload_event_for_g1
                or (tracker.phase, tracker.group_id) != ('unload', 1)
                or self.state not in ('initializing', 'active') or tracker.identity_lost
                or not self.track_store.edge_evidence_px
                or (tracker.points is not None and self._g1_edge_slot is None)):
            self._g1_interior_quartet_review = None
            return objects
        identity = (tracker.phase, tracker.group_id)
        eligible = [obj for obj in objects if obj['class_name'] not in SLOT_STATES
                    or (self.track_store.owner(obj.get('track_id')) in (None, identity)
                        and obj.get('track_id') not in self._transition_quarantine_track_ids)]
        if self._prefer_initial_interior_quartet(eligible, now):
            return self._admit_initial_edge_slot(objects, untracked, now)
        self._g1_edge_slot, selected = InitialEdgeSlot.review(
            self._g1_edge_slot, eligible, untracked, now, tracker, self.contour,
            self._in_initial_edge_strip, 2*self.track_store.edge_evidence_px)
        if selected is None:
            return objects
        return [selected if obj['class_name'] in SLOT_STATES and obj.get('track_id') == selected['track_id']
                else obj for obj in objects]

    def _prefer_initial_interior_quartet(self, objects, now):
        """A stable real quartet sharing three IDs outranks an extra edge candidate."""
        tracker, edge = self.tracker, self._g1_edge_slot
        provisional = tracker._provisional
        if tracker.points is not None or provisional is None or edge is None:
            self._g1_interior_quartet_review = None
            return False
        hands = [obj for obj in objects if obj['class_name'] == 'hand']
        inside = [obj for obj in tracker._candidates(objects)
                  if obj.get('track_id') is not None and not obj['conflict']
                  and obj.get('detection_source') != 'local_crop'
                  and not tracker._occluded(obj, hands) and self._distance(obj['center']) >= 0]
        ids = [obj['track_id'] for obj in inside]
        inside_ids = set(ids)
        bound_ids = set(provisional.byte_track_ids)-{None}
        rows = [i for i, track_id in enumerate(provisional.byte_track_ids)
                if track_id in edge.track_ids and track_id not in inside_ids]
        if (len(inside) != 4 or len(inside_ids) != 4 or len(rows) != 1
                or len(bound_ids & inside_ids) != 3
                or len(set(edge.peers) & inside_ids) < 3
                or slot_layout_error([obj['center'] for obj in inside]) is None):
            self._g1_interior_quartet_review = None
            return False
        new_id, = inside_ids-bound_ids
        row, = rows
        by_id = {obj['track_id']: obj for obj in inside}
        _, recovery_gate = tracker._provisional_recovery_limits(provisional)
        previous_center = tracker._provisional_last_real_centers.get(row, provisional.points[row])
        if np.linalg.norm(np.asarray(by_id[new_id]['center'])-previous_center) <= recovery_gate:
            # A nearby new ID belongs to normal recovery, which retains the edge slot's history.
            self._g1_interior_quartet_review = None
            return False
        key = (row, tuple(sorted(inside_ids)))
        points = np.asarray([by_id[track_id]['center'] for track_id in sorted(inside_ids)])
        spacing = float(np.min(np.linalg.norm(points[:, None]-points[None, :], axis=2)
                               [np.triu_indices(4, 1)]))
        review = self._g1_interior_quartet_review
        if (review is None or review['key'] != key or now-review['last_seen'] > .25
                or np.max(np.linalg.norm(points-review['points'], axis=1))
                   > spacing*tracker.initialization_stability_ratio):
            review = {'key': key, 'since': now, 'frames': 0, 'points': points.copy()}
        review.update(last_seen=now, frames=review['frames']+1)
        self._g1_interior_quartet_review = review
        if (review['frames'] < max(3, tracker.initialization_frames)
                or now-review['since']+1e-9 < tracker.geometry_initialization_seconds):
            return False
        if not tracker._replace_initial_edge_slot(row, by_id[new_id], now):
            return False
        self._g1_rejected_edge_ids.update(edge.track_ids-inside_ids)
        self._g1_edge_slot = None
        self._g1_interior_quartet_review = None
        return True

    def _is_supported_initial_edge(self, track_id, center):
        edge = self._g1_edge_slot
        return (edge is not None and edge.frames >= max(3, self.tracker.initialization_frames)
                and track_id in edge.track_ids and self._in_initial_edge_strip(center))

    def _admit_entry_objects(self, objects, tracker, now):
        """已建立轨迹可走完整个ROI；新身份只能在入口侧建立。"""
        # 多点模式本身等四槽整体停稳，保留原有完整组初始化路径。
        if self.state != "entering":
            return [o for o in objects if o["class_name"] not in SLOT_STATES or self._distance(
                ((o["xyxy"][0]+o["xyxy"][2])/2, (o["xyxy"][1]+o["xyxy"][3])/2)) >= 0]
        points = tracker._entry_points
        if points is None:
            points = tracker.reference_points
        elif points:
            points = [p for p, seen in zip(points, tracker._entry_seen_at)
                      if now-seen <= tracker.lost_seconds]
        result = []
        # 至少两条入口轨迹已建立后，可接收入口处曾被遮挡的其他槽。
        # 旧组的所有可见中心（含未编号框）由独立退出轨迹持续排除。
        established_group = points is not None and len(points) >= 2
        entry_ids = set(tracker._entry_track_ids) - {None}
        for obj in objects:
            if obj["class_name"] not in SLOT_STATES:
                result.append(obj)
                continue
            a = obj["xyxy"]
            center = np.array([(a[0]+a[2])/2, (a[1]+a[3])/2])
            established = (points is not None and len(points) and
                np.min(np.linalg.norm(np.asarray(points)-center, axis=1)) <= self.entry_match_distance_px)
            track_id = obj.get('track_id')
            owner = self.track_store.owner(track_id)
            identity = (tracker.phase, tracker.group_id)
            if owner is not None and owner != identity:
                self._reject_track(obj, 'previous_group_track')
                continue
            known_current = owner == identity
            distance = self._distance(center)
            if distance < 0:
                if (established or known_current) and distance >= -self.buffer_px:
                    result.append(obj)
                else:
                    self._reject_track(obj, 'outside_roi')
                continue
            progress = self._entry_progress(center)
            late_visible = established_group and distance >= self.margin
            if established or progress is None or progress <= self.entry_birth_fraction or late_visible:
                if (tracker.phase == 'unload' and owner is None and track_id not in entry_ids
                        and self._in_unload_entry_upper_right_band(center)):
                    self._reject_track(obj, 'unload_entry_upper_right_band')
                    self._entry_rejected.append({**obj, 'center': center.tolist(),
                        'reason': 'unload_entry_upper_right_band', 'entry_progress': progress})
                else:
                    result.append(obj)
            else:
                self._reject_track(obj, 'new_track_outside_entry_region')
                self._entry_rejected.append({**obj, "center": center.tolist(),
                    "reason": "new_track_outside_entry_region", "entry_progress": progress})
        return result

    def _new_tracker(self, group_id, *, entry=False, phase=None):
        result = deepcopy(self.tracker)
        result.reset_group(group_id,phase=phase)
        result.track_store = self.track_store
        if self.use_previous_s4_layouts and result.phase == "unload":
            references = [(old_group, np.median(samples, axis=0))
                          for old_group, samples in self._unload_layouts.items() if old_group < group_id]
            result._s4_reference_layouts = tuple((old_group, layout)
                for old_group, layout in references if valid_slot_geometry(layout))
        elif result.phase == "load" and (self.use_previous_s4_layouts
                                         or self.load_numbering_calibration is not None):
            samples = self._unload_layouts.get(group_id)
            if samples:
                layout = np.median(samples, axis=0)
                if valid_slot_geometry(layout):
                    result._load_numbering_reference = layout
        if result.phase == "load":
            result._load_perspective_numbering = self.load_numbering_calibration
            result._g1_load_numbering_reference = self._g1_load_layout
            result._g1_load_reference_inferred_slot_ids = self._g1_load_layout_inferred_slot_ids
            result._orbit_unload_reference = self._unload_orbit_layouts.get(group_id)
        if entry:
            result.initialization_seconds = self.entry_seconds
            result.geometry_initialization_seconds = self.entry_seconds
            result.initialization_stability_ratio = self.entry_stability_ratio
        return result

    def _record_unload_layout(self, result, now):
        """Keep a bounded set of stable measured quartets per unloading group."""
        tracker = self.tracker
        if (self.state != "active" or result["slot_numbering_status"] != "final"
                or tracker.phase != "unload" or tracker.last_time != now
                or not self._observations_enabled or tracker.identity_lost
                or any(not slot.geometry_observed or slot.geometry_inferred
                       or slot.observation not in ("empty", "filled") for slot in tracker.slots)
                or None in tracker.byte_track_ids or len(set(tracker.byte_track_ids)) != 4
                or not valid_slot_geometry(tracker.points)):
            self._layout_samples.clear()
            return
        points = tracker.points
        key = (tracker.group_id, tuple(tracker.byte_track_ids))
        if key != self._layout_sample_key or (self._layout_samples and np.max(
                np.linalg.norm(points-self._layout_samples[0], axis=1))
                > tracker._spacing()*self.entry_stability_ratio):
            self._layout_samples.clear()
        self._layout_sample_key = key
        self._layout_samples.append(points.copy())
        if len(self._layout_samples) == self._layout_samples.maxlen:
            self._unload_orbit_layouts.setdefault(tracker.group_id, self._layout_samples[1].copy())
            samples = self._unload_layouts.setdefault(tracker.group_id, deque(maxlen=128))
            samples.append(points.copy())

    def _reject_track(self, obj, reason):
        if obj.get('track_id') is not None:
            self._track_rejections[int(obj['track_id'])] = reason

    def _record_g1_load_layout(self, result, now):
        """Record a stable G1 template, including uniquely verified two-point numbering."""
        tracker = self.tracker
        # Formal numbering has already validated this pose. Cache it immediately;
        # a hidden measured slot must not prevent later groups obtaining a template.
        if (self._g1_load_layout is None and self.load_numbering_calibration is not None
                and tracker.phase == "load" and tracker.group_id == 1
                and result["slot_numbering_status"] == "final" and tracker.last_time == now
                and self._observations_enabled and not tracker.identity_lost
                and tracker.initialization.get("perspective_load_numbering", {}).get("status") == "matched"):
            pose = tracker.initialization.get("numbering_reference_centers_xy")
            if pose is not None and valid_slot_geometry(pose):
                self._g1_load_layout = np.asarray(pose, dtype=float).copy()
                self._g1_load_layout_inferred_slot_ids = tuple(tracker.initialization.get("inferred_slot_ids", ()))
                tracker.initialization["g1_load_template"] = {"recorded_at": float(now),
                    "source": "validated_formal_numbering", "measured_support":
                        4-len(self._g1_load_layout_inferred_slot_ids),
                    "inferred_slot_ids": list(self._g1_load_layout_inferred_slot_ids)}
        if (self.load_numbering_calibration is None or (self._g1_load_layout is not None
                and not self._g1_load_layout_inferred_slot_ids)):
            return
        if (self.state != "active" or tracker.phase != "load" or tracker.group_id != 1
                or result["slot_numbering_status"] != "final" or tracker.last_time != now
                or not self._observations_enabled or tracker.identity_lost
                or tracker.initialization.get("perspective_load_numbering", {}).get("status") != "matched"
                or not valid_slot_geometry(tracker.points)):
            self._g1_load_layout_samples.clear()
            return
        measured = [i for i, slot in enumerate(tracker.slots)
                    if slot.geometry_observed and not slot.geometry_inferred
                    and slot.observation in ("empty", "filled")]
        predicted = sorted(set(range(4))-set(measured))
        real_ids = [tracker.byte_track_ids[i] for i in measured]
        minimum = 2 if tracker.initialization.get("reason") == "stable_two_point_template_numbering" else 3
        if (len(measured) < minimum or None in real_ids or len(set(real_ids)) != len(real_ids)
                or any(not tracker.slots[i].geometry_inferred
                       or i+1 not in tracker.initialization.get("inferred_slot_ids", ()) for i in predicted)):
            self._g1_load_layout_samples.clear()
            return
        # Keep stable templates until a layout with more measured support can upgrade them.
        if self._g1_load_layout is not None and len(predicted) >= len(self._g1_load_layout_inferred_slot_ids):
            return
        points = tracker.points.copy()
        alignment = None
        if self._g1_load_layout is not None:
            anchors = [i for i in measured if i+1 not in self._g1_load_layout_inferred_slot_ids]
            aligned, alignment = self.load_numbering_calibration.align_load_layout(
                self._g1_load_layout, points, anchors)
            if aligned is None:
                self._g1_load_layout_samples.clear()
                return
            # Improve the measured shape without changing its reference pose.
            # Coordinates still inferred retain their previous values.
            points = self._g1_load_layout.copy()
            for i in measured:
                points[i] = aligned[i]
            if not valid_slot_geometry(points):
                self._g1_load_layout_samples.clear()
                return
        key = (tuple((i, tracker.byte_track_ids[i]) for i in measured), tuple(predicted))
        if key != self._g1_load_layout_key or (self._g1_load_layout_samples and np.max(
                np.linalg.norm(points-self._g1_load_layout_samples[0], axis=1))
                > tracker._spacing()*self.entry_stability_ratio):
            self._g1_load_layout_samples.clear()
        self._g1_load_layout_key = key
        self._g1_load_layout_samples.append(points)
        if len(self._g1_load_layout_samples) == self._g1_load_layout_samples.maxlen:
            self._g1_load_layout = np.median(self._g1_load_layout_samples, axis=0)
            self._g1_load_layout_inferred_slot_ids = tuple(i+1 for i in predicted)
            tracker.initialization["g1_load_template"] = {"recorded_at": float(now),
                "measured_support": len(measured), "inferred_slot_ids": list(self._g1_load_layout_inferred_slot_ids),
                "source": "aligned_measured_slots" if alignment is not None else "stable_measured_slots",
                "pose_alignment": alignment}

    def _record_outside_mapping_quarantine(self, objects, now):
        """Remember unbound ROI-outside IDs seen during this group's handoff."""
        if (self.mapping_time is None
                or self._outside_mapping_quarantine_group !=
                   (self.tracker.phase, self.tracker.group_id)
                or self.outside_mapping_quarantine_seconds <= 0):
            return
        elapsed = float(now) - self.mapping_time
        if elapsed < 0 or elapsed >= self.outside_mapping_quarantine_seconds:
            return
        formal_ids = {int(track_id) for track_id in self.tracker.byte_track_ids
                      if track_id is not None}
        for obj in objects:
            if obj.get('class_name') not in SLOT_STATES:
                continue
            track_id = obj.get('track_id')
            if track_id is None or int(track_id) in formal_ids or obj.get('initial_edge_slot'):
                continue
            box = obj.get('xyxy')
            if box is None:
                continue
            center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)
            if self._distance(center) < 0:
                self._outside_mapping_quarantine_track_ids.add(int(track_id))

    def _exclude_outside_mapping_quarantine(self, objects):
        """Reject IDs blacklisted for the current group at every later frame."""
        if not self._outside_mapping_quarantine_track_ids:
            return objects
        result = []
        for obj in objects:
            track_id = obj.get('track_id')
            if (obj.get('class_name') in SLOT_STATES and track_id is not None and not obj.get('initial_edge_slot')
                    and int(track_id) in self._outside_mapping_quarantine_track_ids):
                self._reject_track(obj, 'outside_mapping_quarantine')
                continue
            result.append(obj)
        return result

    def _record_group_roi_tracks(self, objects, now):
        """记录本组ROI真实出现帧数和最后位置；每个ID每帧最多计一次。"""
        if self.state == "sequence_complete":
            return
        tracker = (self.pending_tracker if self.state in ("waiting_next", "stabilizing_next")
                   and self.pending_tracker is not None else self.tracker)
        history = self._group_roi_tracks.setdefault((tracker.phase, tracker.group_id), {})
        seen = set()
        for obj in objects:
            if obj.get('class_name') not in SLOT_STATES or obj.get('track_id') is None:
                continue
            box = obj.get('xyxy')
            if box is None:
                continue
            track_id = int(obj['track_id'])
            if track_id in seen:
                continue
            seen.add(track_id)
            center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
            inside = self._distance(center) >= 0
            if not inside and track_id not in history:
                continue
            record = history.setdefault(track_id, {
                'first_inside': float(now),
                'inside_frames': 0,
                'inside_since': float(now) if inside else None,
                'outside_since': None,
                'previous_center': None,
                'previous_seen': None,
            })
            previous_center = record.get('center')
            previous_seen = record.get('last_seen')
            if previous_center is not None:
                record['previous_center'] = list(previous_center)
                record['previous_seen'] = previous_seen
            if inside:
                if not record.get('inside_roi', False):
                    record['inside_since'] = float(now)
                record.setdefault('inside_since', float(now))
                record['outside_since'] = None
            elif record.get('outside_since') is None:
                record['outside_since'] = float(now)
            record.update(last_seen=float(now), center=center, inside_roi=inside)
            if inside:
                record['inside_frames'] += 1
                record['last_inside'] = float(now)

    def _initial_g1_waiting_for_unload(self):
        """Return whether the initial G1 candidate is still unconfirmed.

        The slot tracker keeps a provisional geometry and buffers its state
        evidence.  Until an expected unload event exists, that geometry is
        not allowed to occupy G1 as a formal group.
        """
        tracker = self.tracker
        events = list(getattr(tracker, "events", []))
        if tracker._provisional is not None:
            events.extend(tracker._provisional.events)
        return bool(self._g1_waiting_event
                    and not any(event.get("action") == "unload"
                                and event.get("expected_action", False)
                                and event.get("confirmed_at") is not None
                                for event in events))

    def _release_initial_g1_event_gate(self, result):
        """Publish G1 evidence once the candidate has formal slot numbers."""
        if not self._g1_waiting_event or self.tracker.points is None:
            return result
        if not any(event.get("action") == "unload"
                   and event.get("expected_action", False)
                   and event.get("confirmed_at") is not None
                   for event in self.tracker.events):
            return result
        self._g1_waiting_event = False
        buffered = self._g1_pending_events
        self._g1_pending_events = []
        result["new_events"] = buffered + list(result.get("new_events", []))
        return result

    def _mask_initial_g1_result(self, result):
        """Hide formal S numbers while G1 is waiting for its first unload."""
        if not self._initial_g1_waiting_for_unload():
            return result
        result["slot_numbering_status"] = "provisional"
        result["identity_status"] = "waiting_first_unload_event"
        result["group_assignment"] = "configured"
        result["confirmed_event_sequence"] = []
        result["initialization"] = dict(result.get("initialization", {}),
                                          reason="waiting_for_first_unload_event")
        for slot in result.get("slots", []):
            slot["slot_id"] = None
            slot["formal_slot_id"] = None
        return result

    def _discard_unconfirmed_initial_g1(self, now, objects=None):
        """Drop an unconfirmed G1 candidate after it actually leaves the ROI."""
        if not self._initial_g1_waiting_for_unload():
            return False
        provisional = self.tracker._provisional
        candidate = provisional or self.tracker
        if any(
                event is not None and event["action"] == "unload" and event["expected_action"]
                for event in (slot.pending_visual_event for slot in candidate.slots)):
            # 本帧先判断是否丢弃候选，再更新复核。保留候选直到复核实际完成，
            # 包括超过时限后由第一帧确认或撤销事件的时刻。
            return False
        identity = (self.tracker.phase, self.tracker.group_id)
        history = self._group_roi_tracks.get(identity, {})
        provisional_ids = {int(track_id) for track_id in
                           getattr(provisional, "byte_track_ids", [])
                           if track_id is not None}
        # Geometry alone protects the candidate while all four real provisional tracks
        # have continuous in-ROI support and stable relative geometry.  Three
        # visible tracks are insufficient: the fourth may be the stale point
        # that later produces a delayed or incorrect S-slot mapping.
        current_inside_ids = set()
        current_centers = {}
        if objects is not None:
            for obj in objects:
                if (obj.get("class_name") not in SLOT_STATES
                        or obj.get("track_id") is None):
                    continue
                box = obj.get("xyxy")
                if box is None:
                    continue
                center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
                if self._distance(center) >= 0:
                    track_id = int(obj["track_id"])
                    if track_id in provisional_ids:
                        current_inside_ids.add(track_id)
                        current_centers[track_id] = np.asarray(center, dtype=float)

        stable_seconds = max(.2, float(getattr(
            provisional, 'geometry_initialization_seconds', .3))) if provisional is not None else .3
        stable_frames = max(3, int(getattr(provisional, 'initialization_frames', 3))) \
            if provisional is not None else 3
        outside_seconds = max(.25, min(.6, stable_seconds))
        all_four_ids = (provisional is not None and len(provisional_ids) == 4
                        and len(current_inside_ids) == 4)
        all_four_stable = all_four_ids
        if all_four_stable:
            for track_id in provisional_ids:
                record = history.get(track_id, {})
                inside_since = record.get('inside_since')
                if (record.get('inside_frames', 0) < stable_frames
                        or inside_since is None
                        or now - float(inside_since) + 1e-9 < stable_seconds):
                    all_four_stable = False
                    break

        # Compare one frame of motion after removing the common group motion.
        # A rigid translation of all four slots is not a reason to reset G1;
        # one slot deviating from the other three is.
        position_outlier = False
        if all_four_stable:
            deltas = []
            for track_id in provisional_ids:
                previous = history.get(track_id, {}).get('previous_center')
                if previous is None:
                    all_four_stable = False
                    break
                deltas.append(current_centers[track_id] - np.asarray(previous, dtype=float))
            if all_four_stable:
                deltas = np.asarray(deltas, dtype=float)
                common_delta = np.median(deltas, axis=0)
                residuals = np.linalg.norm(deltas - common_delta, axis=1)
                try:
                    spacing = float(provisional._spacing())
                except (AttributeError, ValueError, TypeError):
                    spacing = 0.0
                position_threshold = max(12.0, spacing * .25)
                stable_motion_count = int(np.count_nonzero(residuals <= position_threshold))
                position_outlier = (stable_motion_count >= 3
                                    and bool(np.any(residuals > position_threshold)))
                if position_outlier:
                    all_four_stable = False

        protect_stable_candidate = bool(all_four_stable and not position_outlier)
        # A bound track must remain outside continuously for a short interval
        # before the candidate is released.  A one-frame boundary excursion
        # remains recoverable by the provisional ID logic.
        outside = (False if protect_stable_candidate else any(
            int(track_id) in provisional_ids
            and not record.get("inside_roi", True)
            and record.get("center") is not None
            and self._distance(record["center"]) < -self.margin
            and not self._is_supported_initial_edge(track_id, record["center"])
            and record.get('outside_since') is not None
            and now - float(record['outside_since']) + 1e-9 >= outside_seconds
            for track_id, record in history.items()))
        if position_outlier:
            outside = True
        # If ByteTrack changed the ID on the exit frame, the history key alone
        # is insufficient.  Accept only a frame that is both outside and close
        # to a known provisional/formal point, so an unrelated outside object
        # cannot discard G1.
        if not outside and objects is not None and not protect_stable_candidate:
            provisional = getattr(self.tracker, "_provisional", None)
            known_ids = set(self.tracker.byte_track_ids)
            reference = self.tracker.reference_points
            if provisional is not None:
                known_ids.update(provisional.byte_track_ids)
                reference = provisional.reference_points
            reference = None if reference is None else np.asarray(reference, dtype=float)
            for obj in objects:
                if obj.get("class_name") not in SLOT_STATES:
                    continue
                box = obj.get("xyxy")
                if box is None:
                    continue
                center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)
                if (self._distance(center) >= -self.margin
                        or self._is_supported_initial_edge(obj.get("track_id"), center)):
                    continue
                near_known_point = (reference is not None and len(reference)
                                    and np.min(np.linalg.norm(reference-center, axis=1))
                                    <= self.entry_match_distance_px)
                if obj.get("track_id") in known_ids or near_known_point:
                    outside = True
                    break
        if not outside:
            return False

        # The provisional IDs have never been bound to a formal group, but
        # release defensively in case an external caller bound one meanwhile.
        provisional = getattr(self.tracker, "_provisional", None)
        for track_id in getattr(provisional, "byte_track_ids", []):
            if track_id is not None:
                self.track_store.release_candidate(track_id, identity)
        self._group_roi_tracks.pop(identity, None)
        self.tracker.reset_group(1, phase="unload")
        self.tracker.track_store = self.track_store
        self.state = "initializing"
        self.pending_tracker = None
        self.anchor = self.anchor_distance = self.anchor_points = self.anchor_spacing = None
        self.inside_seen.clear()
        self.crossed.clear()
        self.visible_history.clear()
        self.center_history.clear()
        self.direction = self.exit_since = self.return_since = None
        self._entry_count = 0
        self._g1_pending_events = []
        # 边缘证据独立于临时编号，后续须再次通过同组支撑点和有效期核验。
        self.mapping_time = None
        self._outside_mapping_quarantine_track_ids.clear()
        self._outside_mapping_quarantine_group = identity
        self._group_events.append({
            "type": "group_candidate_discarded",
            "group_id": 1,
            "phase": "unload",
            "at": float(now),
            "reason": "initial_g1_left_roi_without_confirmed_unload",
        })
        return True

    def _arm_transition_quarantine(self, objects, now):
        """按旧组稳定证据隔离未绑定ID，并保留换组瞬间的短时空间隔离。"""
        self._transition_quarantine_centers = []
        self._transition_quarantine_until = None
        self._transition_recheck_ids.clear()
        self._transition_recheck = None
        self._transition_entry_track = None
        self._s4_load_lookback.clear()
        self._s4_load_hint = False
        if not self.reject_transition_roi_detections:
            return
        phase, group_id = self.tracker.phase, self.tracker.group_id
        history = self._group_roi_tracks.get((phase, group_id), {})
        min_frames = self.transition_quarantine_min_frames
        if phase == "unload" and group_id == 5 and self.auto_load_after_unload:
            min_frames = self.unload_to_load_quarantine_min_frames
        for track_id, record in history.items():
            if (self.track_store.owner(track_id) is None
                    and record['inside_frames'] >= min_frames
                    and record['inside_roi']
                    and now-record['last_seen'] <= self.tracker.lost_seconds):
                if track_id not in self._transition_quarantine_track_ids:
                    self._transition_recheck_ids.add(track_id)
                self._transition_quarantine_track_ids.add(track_id)
        # 已绑定旧组的ID由_close登记为closed，再由旧组归属过滤排除；
        # 即使它们最后已经出ROI或漏检超过时限，也不会成为下一组的槽。
        # 历史漏检点只隔离ID，不把滞后坐标扩成空间屏蔽区域。
        for obj in objects:
            if obj.get('class_name') not in SLOT_STATES:
                continue
            # 上料换组时，刚进入ROI的未绑定轨迹不能用自身位置挡住下一组。
            track_id = obj.get('track_id')
            if (phase == "load" and track_id is not None and self.track_store.owner(track_id) is None
                    and history.get(track_id, {}).get('inside_frames', 0) < min_frames):
                continue
            box = obj.get('xyxy')
            if box is None:
                continue
            center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)
            if self._distance(center) < 0:
                continue
            self._transition_quarantine_centers.append(center)
        if self._transition_quarantine_centers:
            duration = max(float(self.tracker.lost_seconds), self.entry_seconds, .5)
            self._transition_quarantine_until = float(now)+duration

    def _exclude_transition_quarantine(self, objects, now):
        """持续排除有稳定旧组证据的ID，短时过滤换组瞬间ROI中心附近的框。"""
        if not self._transition_quarantine_track_ids and not self._transition_quarantine_centers:
            return objects
        if (self._transition_quarantine_until is not None
                and now >= self._transition_quarantine_until):
            self._transition_quarantine_centers = []
            self._transition_quarantine_until = None
        result = []
        centers = np.asarray(self._transition_quarantine_centers, dtype=float)
        for obj in objects:
            if obj.get('class_name') not in SLOT_STATES:
                result.append(obj)
                continue
            box = obj.get('xyxy')
            center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)
            track_id = obj.get('track_id')
            near_quarantined = (len(centers) and
                np.min(np.linalg.norm(centers-center, axis=1)) <= self.entry_match_distance_px)
            if (track_id is not None and int(track_id) in self._transition_quarantine_track_ids) or near_quarantined:
                self._reject_track(obj, 'transition_roi_quarantine')
                continue
            result.append(obj)
        return result

    def _sync_track_bindings(self, result, now=None):
        tracker = self.tracker
        # A pre-event G1 candidate must not occupy the shared store as a
        # formal group; otherwise a later candidate could be rejected as an
        # old G1 track even though this candidate is about to be discarded.
        if self._initial_g1_waiting_for_unload():
            return
        identity = (tracker.phase, tracker.group_id)
        slots = result.get('slots', [])
        for slot in slots:
            self.track_store.bind(slot.get('byte_track_id'), identity,
                                  slot.get('formal_slot_id'),
                                  center=slot.get('center_xy'), now=now)
            description = self.track_store.describe(slot.get('byte_track_id'))
            if description is not None:
                slot.update(logical_track_id=description.get('logical_track_id'),
                            reidentified_from_track_id=description.get('reidentified_from_track_id'),
                            reid_gap_seconds=description.get('reid_gap_seconds'),
                            reid_method=description.get('reid_method'))
        for obj in tracker.geometry_rejected:
            self._reject_track(obj, obj['reason'])

    def _recover_crossed_load_binding(self, objects, now):
        """Hold a jumping ID until a second track confirms both physical positions."""
        tracker = self.tracker
        pending, self._pending_load_rebind = self._pending_load_rebind, None
        pose = self._load_rebind_pose
        if (pose is None or now-pose[0] > .5 or self.state != "active"
                or not self.track_store.reidentification_enabled or tracker.identity_lost
                or (tracker.phase, tracker.group_id) != pose[1]):
            return objects
        _, identity, points, support = pose
        gate = min(self.entry_match_distance_px, self.track_store.reidentification_max_distance_px)
        bound_ids = tracker.byte_track_ids
        detected_ids = {obj["track_id"] for obj in objects
                        if obj["class_name"] in SLOT_STATES and obj.get("track_id") is not None}
        observations = {}
        for obj in tracker._candidates(objects):
            track_id = obj.get("track_id")
            if (track_id is not None and not obj["conflict"]
                    and obj.get("detection_source", "full_frame") == "full_frame"
                    and self._distance(obj["center"]) >= self.margin
                    and track_id not in self._outside_mapping_quarantine_track_ids
                    and track_id not in self._transition_quarantine_track_ids
                    and not tracker._occluded(obj, self._hands)):
                observations[track_id] = None if track_id in observations else obj
        visible = {i: observations[track_id] for i, track_id in enumerate(bound_ids)
                   if observations.get(track_id) is not None}
        jumps = []
        for old in support:
            track_id = bound_ids[old]
            if old not in visible or self.track_store.owner(track_id) != identity:
                continue
            obj = visible[old]
            distances = np.linalg.norm(points-obj["center"], axis=1)
            nearest = np.argsort(distances)
            target = int(nearest[0])
            if (target == old or target not in support or bound_ids[target] in detected_ids
                    or distances[old] <= gate or distances[target] > gate
                    or distances[nearest[1]]-distances[target] < gate*.15
                    or any(tracker.slots[i].pending_visual_event is not None for i in (old, target))):
                continue
            anchors = [np.linalg.norm(points[i]-other["center"])
                       for i, other in visible.items() if i != old]
            if anchors and max(anchors) <= gate*.3:
                jumps.append((old, target, obj))
        if len(jumps) != 1:
            return objects
        old, target, obj = jumps[0]
        track_id = obj["track_id"]
        replacements = []
        for new_id, candidate in observations.items():
            record = self.track_store.tracks.get(new_id)
            if (candidate is None or new_id in bound_ids or self.is_retired_track(new_id)
                    or (record is not None and (record["owner"] not in (None, identity)
                                                or record["slot_id"] is not None))):
                continue
            distances = np.linalg.norm(points-candidate["center"], axis=1)
            if (distances[old] <= gate*.5
                    and np.min(np.delete(distances, old))-distances[old] >= gate*.15):
                replacements.append(new_id)
        if len(replacements) == 1:
            new_id, = replacements
            pair = (old, target, track_id, new_id)
            if pending is not None and pair == pending[0] and now-pending[1] <= .25:
                self.track_store.split_reused_id(track_id, self.track_store.tracks[track_id],
                    target+1, now, pending[1], method="crossed_load_slot_binding")
                bound_ids[old], bound_ids[target] = new_id, track_id
                return objects
            self._pending_load_rebind = (pair, now)
        held_ids = {track_id, *replacements}
        for held_id in held_ids:
            self._reject_track(observations[held_id], "crossed_load_binding_pending")
        return [item for item in objects if item.get("track_id") not in held_ids]

    def _split_reused_track_ids(self, objects, now):
        """Split a reused BT ID after its old and new slot are unambiguous."""
        tracker = self.tracker
        if (not self.track_store.reidentification_enabled or self.state != 'active'
                or tracker.points is None or tracker.reference_points is None):
            self._reused_id_candidates.clear()
            return
        identity = (tracker.phase, tracker.group_id)
        gate = min(self.track_store.reidentification_max_distance_px,
                   tracker._spacing() * tracker.match_distance_ratio)
        observations = {}
        for obj in objects:
            track_id = obj.get('track_id')
            if obj.get('class_name') in SLOT_STATES and track_id is not None:
                track_id = int(track_id)
                observations[track_id] = None if track_id in observations else obj
        seen = set()
        for obj in objects:
            track_id = obj.get('track_id')
            if obj.get('class_name') not in SLOT_STATES or track_id is None:
                continue
            track_id = int(track_id)
            if track_id in seen:
                continue
            seen.add(track_id)
            box = obj.get('xyxy')
            record = self.track_store.tracks.get(track_id)
            if box is None or record is None:
                self._reused_id_candidates.pop(track_id, None)
                continue
            center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)
            candidate = self._reused_id_candidates.get(track_id)
            if candidate is None:
                old_slot = record.get('slot_id')
                old_center = record.get('center')
                gap = now-float(record['last_seen'])
                if (record.get('owner') != identity or old_slot is None
                        or old_center is None
                        or gap > self.track_store.reidentification_max_gap_seconds
                        or self._distance(center) < 0
                        or track_id in self._outside_mapping_quarantine_track_ids
                        or track_id in self._transition_quarantine_track_ids):
                    continue
                old_slot = int(old_slot)
                if not 1 <= old_slot <= len(tracker.slots):
                    continue
                replacement_id = tracker.byte_track_ids[old_slot-1]
                source_slot = tracker.slots[old_slot-1]
                source_seen = source_slot.last_seen
                expected_state = 'empty' if tracker.phase == 'unload' else 'filled'
                if (replacement_id is None or replacement_id == track_id
                        or source_seen is None or now-source_seen > .25
                        or source_slot.confirmed != expected_state):
                    continue
                distances = np.linalg.norm(tracker.reference_points-center, axis=1)
                order = np.argsort(distances)
                target = int(order[0])
                if (target == old_slot-1 or distances[target] > gate
                        or distances[int(order[1])]-distances[target] < max(5., gate*.15)):
                    continue
                target_slot = tracker.slots[target]
                target_seen = target_slot.last_seen
                if (target_seen is not None and now-target_seen < .2):
                    continue
                if target_slot.confirmed != ('filled' if tracker.phase == 'unload' else 'empty'):
                    continue
                after_gap = (gap > tracker.lost_seconds and SLOT_STATES[obj['class_name']] == expected_state
                             and np.linalg.norm(center-np.asarray(old_center)) > gate)
                if after_gap:
                    method = 'bytetrack_id_reused_after_gap'
                elif tracker.phase == 'load' and self._supports_reused_load_id(
                        track_id, old_slot-1, target, center, observations, identity, gate):
                    method = 'bytetrack_id_reused_at_other_slot'
                else:
                    continue
                candidate = {'old_record': deepcopy(record), 'old_slot': old_slot,
                             'target_slot': target+1, 'last_seen': float(now),
                             'first_seen': float(now), 'count': 0, 'method': method}
                self._reused_id_candidates[track_id] = candidate

            target = candidate['target_slot']-1
            old_slot = candidate['old_slot']-1
            source_seen = tracker.slots[old_slot].last_seen
            target_seen = tracker.slots[target].last_seen
            expected_state = 'empty' if tracker.phase == 'unload' else 'filled'
            if candidate['method'] == 'bytetrack_id_reused_at_other_slot':
                supported = self._supports_reused_load_id(track_id, old_slot, target, center,
                                                          observations, identity, gate)
            else:
                supported = SLOT_STATES[obj['class_name']] == expected_state
            if (now-candidate['last_seen'] > .25
                    or self._distance(center) < 0
                    or np.linalg.norm(tracker.reference_points[target]-center) > gate
                    or tracker.byte_track_ids[old_slot] in (None, track_id)
                    or source_seen is None or now-source_seen > .25
                    or tracker.slots[old_slot].confirmed != expected_state
                    or target_seen is not None and now-target_seen < .2
                        or not supported):
                self._reused_id_candidates.pop(track_id, None)
                continue
            candidate['count'] += 1
            candidate['last_seen'] = float(now)
            if candidate['count'] >= 3:
                self.track_store.split_reused_id(
                    track_id, candidate['old_record'], candidate['target_slot'],
                    now, candidate['first_seen'], method=candidate['method'])
                self._reused_id_candidates.pop(track_id, None)
        for track_id in list(self._reused_id_candidates):
            if track_id not in seen or now-self._reused_id_candidates[track_id]['last_seen'] > .25:
                self._reused_id_candidates.pop(track_id, None)

    def _supports_reused_load_id(self, track_id, old_slot, target, center, observations, identity, gate):
        tracker = self.tracker
        replacement_id = tracker.byte_track_ids[old_slot]
        target_id = tracker.byte_track_ids[target]
        detection, replacement = observations.get(track_id), observations.get(replacement_id)
        if (detection is None or replacement is None or replacement_id == track_id
                or target_id in observations or self.track_store.owner(replacement_id) != identity
                or replacement['class_name'] != 'wafer_slot_filled'
                or tracker.slots[target].confirmed != 'empty'
                or any(obj.get('detection_source', 'full_frame') != 'full_frame'
                       for obj in (detection, replacement))):
            return False
        old_point = tracker.reference_points[old_slot]
        distances = np.linalg.norm(tracker.reference_points-center, axis=1)
        other_distance = np.min(np.delete(distances, target))
        if (distances[target] > gate or other_distance-distances[target] < max(5., gate*.15)
                or distances[old_slot] <= gate):
            return False
        box = np.asarray(replacement['xyxy'])
        replacement_center = (box[:2]+box[2:])/2
        return (self._distance(replacement_center) >= 0 and np.linalg.norm(replacement_center-old_point) <= gate
                and not tracker._occluded(dict(replacement, center=replacement_center), self._hands)
                and not tracker._occluded(dict(detection, center=center), self._hands))

    def _track_metadata(self, objects, result):
        records = self.track_store.visible_records(objects)
        known = {record['byte_track_id'] for record in records}
        for track_id, reason in self.track_store.rejected.items():
            if track_id not in known:
                records.append(dict(byte_track_id=track_id, state='unknown', observation='unknown',
                    last_confirmed_state='unknown', reason=reason, group_id=None,
                    formal_slot_id=None, slot_id=None, phase=None))
        mapped = {s.get('byte_track_id'): s for s in result.get('slots', [])
                  if s.get('byte_track_id') is not None}
        for record in records:
            track_id = record['byte_track_id']
            slot = mapped.get(track_id)
            reason = self._track_rejections.get(track_id) or self.track_store.rejected.get(track_id)
            if reason is None and slot is None:
                reason = 'waiting_group_geometry' if self.tracker.points is None else 'association_ambiguous_or_too_far'
            record['mapping_reason'] = reason or ('mapped' if slot.get('formal_slot_id') else 'waiting_slot_numbering')
        return records

    def _restore_load_baseline(self, result, now):
        """正式编号后复用同组下料时真实检测的空槽，绝不复用旧坐标。"""
        tracker = self.tracker
        if tracker.phase != "load" or tracker.points is None or tracker.group_id in self._restored_load_groups:
            return result
        self._restored_load_groups.add(tracker.group_id)
        previous = next((g for g in self.closed_groups
                         if g["phase"] == "unload" and g["group_id"] == tracker.group_id), None)
        if previous is None or previous["snapshot"].get("slot_numbering_status") == "provisional":
            return result
        tracker._load_verified_empty_slots = {slot["slot_id"] for slot in previous["snapshot"]["slots"]
            if slot["last_confirmed_state"] == "empty" and slot["last_confirmed_at"] is not None
            and slot.get("confirmation_source") == "yolo"}
        new_events = list(result["new_events"])
        identity = (tracker.phase, tracker.group_id)
        transition_track_id = tracker.transition_load_track_id
        transition_record = self.track_store.tracks.get(transition_track_id)
        defer_transition_history = transition_record is not None and any(
            event['action'] == 'load' and event['confirmed_at'] < now
            for event in transition_record['events'])
        for slot, baseline, track_id in zip(tracker.slots, previous["snapshot"]["slots"],
                                            tracker.byte_track_ids):
            if ((self._transition_load_review is not None
                    and slot.slot_id == tracker.transition_load_slot_index+1)
                    or (defer_transition_history and track_id == transition_track_id)
                    or baseline["last_confirmed_state"] != "empty" or baseline["last_confirmed_at"] is None
                    or slot.pending_visual_event is not None
                    or any(e["slot_id"] == slot.slot_id for e in tracker.events)):
                continue
            evidence_id = slot.confirmed_visual_track_id
            if (slot.confirmed == "filled" and slot.confirmation_source == "yolo"
                    and evidence_id is not None and track_id is not None
                    and self.track_store.logical_id(evidence_id) != self.track_store.logical_id(track_id)):
                # The temporary spatial slot may have been reused by another track.
                # For backfilled load events only, use that track's own observations.
                record = self.track_store.tracks.get(track_id)
                history = record["state"] if (record is not None and record["owner"] in (None, identity)
                    and record["slot_id"] in (None, slot.slot_id)) else None
                if (history is None or history.evidence_time is None
                        or now-history.evidence_time > tracker.initialization_history_seconds):
                    slot.confirmed = slot.state = "unknown"
                    slot.confirmed_since = slot.confirmed_at = slot.evidence_time = None
                    slot.confirmed_visual_track_id = slot.confirmation_source = None
                    slot.candidate, slot.candidate_since, slot.candidate_count = "unknown", None, 0
                    slot.candidate_visual_track_id = None
                else:
                    slot.confirmed, slot.evidence_time = history.confirmed, history.evidence_time
                    slot.confirmed_since, slot.confirmed_at = history.confirmed_since, history.confirmed_at
                    slot.confirmed_visual_track_id = history.confirmed_visual_track_id
                    slot.confirmation_source = history.confirmation_source
                    slot.state = history.state
                    slot.candidate, slot.candidate_since, slot.candidate_count = (
                        history.candidate, history.candidate_since, history.candidate_count)
                    slot.candidate_visual_track_id = history.candidate_visual_track_id
            if slot.confirmed == "unknown":
                slot.confirmed, slot.evidence_time = "empty", baseline["last_confirmed_at"]
                slot.confirmed_since = slot.confirmed_at = baseline["last_confirmed_at"]
                slot.confirmed_visual_track_id = None
                slot.confirmation_source = "previous_unload_yolo"
            elif slot.confirmed == "filled":
                # 上料在正式编号前已完成时，空和满都必须有实际检测依据。
                # 区间使用上次空槽时间到首次本轮满槽时间，不用手接触猜时刻。
                event = {"group_id":tracker.group_id,"slot_id":slot.slot_id,"phase":"load",
                    "from":"empty","to":"filled","action":"load","expected_action":True,
                    "time_lower":baseline["last_confirmed_at"],
                    "time_upper":slot.confirmed_since if slot.confirmed_since is not None else slot.evidence_time,
                    "confirmed_at":slot.confirmed_at if slot.confirmed_at is not None else slot.evidence_time,
                    "reported_at":float(now),
                    "recovered_before_initialization":True,"reason":"detected_after_unload_baseline"}
                tracker.events.append(event)
                new_events.append(event)
        preview_contact = tracker.adopt_entry_load_preview_contact(now)
        contact_tracker = tracker.two_point_hand_contact
        for inferred_index in tracker.inferred_load_contact_indices():
            if preview_contact or inferred_index+1 not in tracker._load_verified_empty_slots:
                continue
            slot = tracker.slots[inferred_index]
            empty_at = previous["snapshot"]["slots"][inferred_index]["last_confirmed_at"]
            if slot.confirmed == "empty":
                if (inferred_index == 3 and (tracker.transition_load_track_id is not None
                                             or tracker._load_numbering_reference is not None)):
                    tracker.s4_load_contact_lookback = self._s4_load_lookback.resolve(
                        tracker, now, empty_at, require_bound_visual=tracker.transition_load_track_id is None)
                if (contact_tracker is not None and slot.pending_visual_event is None
                        and not any(e["slot_id"] == slot.slot_id for e in tracker.events)):
                    contact = contact_tracker.take_inferred_load(tracker, inferred_index, empty_at, now)
                    if contact is not None and slot.evidence_time is not None:
                        if contact.requires_visual_load:
                            if slot.evidence_time <= contact.since:
                                tracker.review_entry_two_point_load(slot, contact)
                                tracker.s4_load_contact_lookback = None
                        elif (slot.confirmation_source == "previous_unload_yolo"
                              or slot.evidence_time < contact.released_at):
                            new_events.append(tracker.infer_entry_s4_load(contact, now))
        s4 = tracker.slots[3]
        if (not preview_contact and tracker._region_formal_numbering and 4 in tracker._load_verified_empty_slots
                and s4.confirmed == "empty" and s4.evidence_time is not None
                and s4.prediction_contact is None and s4.pending_visual_event is None
                and tracker.s4_load_contact_lookback is None
                and not any(e["slot_id"] == 4 and e["action"] == "load" for e in tracker.events)):
            tracker.s4_load_contact_lookback = self._s4_load_lookback.resolve_reach(
                tracker, now, s4.evidence_time)
        return {**tracker.snapshot(),"new_events":new_events}

    def _arm(self, now):
        self.state = "active"
        identity = (self.tracker.phase, self.tracker.group_id)
        if self._outside_mapping_quarantine_group != identity:
            self._outside_mapping_quarantine_track_ids.clear()
            self._outside_mapping_quarantine_group = identity
        self.mapping_time = float(now)
        self.anchor = self.tracker.reference_points.mean(axis=0).copy()
        self.anchor_distance = self._distance(self.anchor)
        self.anchor_points = self.tracker.points.copy()
        self.anchor_spacing = self.tracker._spacing()
        self.inside_seen, self.crossed = {}, {}
        self.visible_history.clear()
        self.center_history.clear()
        self.center_history.append((now,self.anchor.copy()))
        self.direction = self.exit_since = self.return_since = None
        self._required_crossings = self.exit_min_slots
        self._entry_count = 0
        self._motion_support = []

    def _active_objects(self, objects, now):
        """ROI外只接受能关联到当前已编号槽的候选，其他外部槽不参与。"""
        objects = self._exclude_outside_mapping_quarantine(objects)
        result = []
        reference = self.tracker.reference_points
        gate = None if reference is None else (self.tracker._point_tracking_gate
            or self.tracker._spacing()*self.tracker.match_distance_ratio)
        formal_ids = {int(track_id) for track_id in self.tracker.byte_track_ids
                      if track_id is not None}
        initial_unload_group = self.tracker.phase == "unload" and self.tracker.group_id == 1
        quarantine_active = (self.mapping_time is not None
                             and self.outside_mapping_quarantine_seconds > 0
                             and now - self.mapping_time < self.outside_mapping_quarantine_seconds)
        for obj in objects:
            if obj["class_name"] not in SLOT_STATES or obj.get('initial_edge_slot'):
                result.append(obj)
                continue
            a = obj["xyxy"]
            center = np.array([(a[0]+a[2])/2,(a[1]+a[3])/2])
            distance = self._distance(center)
            if (quarantine_active and distance < 0 and obj.get('track_id') is not None
                    and int(obj['track_id']) not in formal_ids):
                self._reject_track(obj, 'outside_mapping_quarantine')
                continue
            if distance >= 0 or (reference is not None and distance >= -self.buffer_px
                    and ((obj.get('track_id') is not None and obj.get('track_id') in self.tracker.byte_track_ids)
                         or np.min(np.linalg.norm(reference-center,axis=1)) <= gate)):
                if (initial_unload_group
                        and (self._g1_waiting_event or obj.get('track_id') not in formal_ids)
                        and self._in_unload_entry_upper_right_band(center)):
                    self._reject_track(obj, 'unload_entry_upper_right_band')
                    continue
                result.append(obj)
        return result

    def _observe_motion(self, tracker, candidates, matches, now):
        """在状态确认前执行；单点配置按真实槽中心越界触发换组。"""
        if not self.enabled:
            return True
        if self.anchor is None:
            self._arm(now)
        # 单点规则一旦触发，后续漏检不能把退出状态改成身份不确定；
        # 当前组已经由真实越界点确认，等待退出确认时间后关闭即可。
        if self.exit_min_slots == 1 and self.state == "exiting":
            return False
        if tracker.identity_lost:
            self.state = "identity_uncertain"
            return False
        hands = self._hands
        reliable = {i: candidates[col] for i,col in matches.items()
                    if not candidates[col]["conflict"] and not tracker._occluded(candidates[col],hands)}
        self.visible_history.append((now,len(reliable)))
        while self.visible_history and now-self.visible_history[0][0] > .5:
            self.visible_history.popleft()
        center = tracker.reference_points.mean(axis=0)
        spacing = tracker._spacing()
        shift = center-self.anchor
        travel = float(np.linalg.norm(shift))
        if len(reliable) >= 2:
            self.center_history.append((now,center.copy()))
        while len(self.center_history) > 1 and now-self.center_history[0][0] > .5:
            self.center_history.popleft()
        # 每槽须真实看见过边界内、随后又真实看见边界外；预测点不能算越界。
        for i,obj in reliable.items():
            point = np.asarray(obj["center"])
            distance = self._distance(point)
            if distance >= self.margin:
                self.inside_seen[i] = (now,point.copy())
                self.crossed.pop(i,None)
            elif distance <= -self.margin and i in self.inside_seen:
                seen_at,old_point = self.inside_seen[i]
                if now-seen_at <= tracker.lost_seconds:
                    self.crossed[i] = {"at": now,"delta": point-old_point}
        # 缺失只能保留已有越界证据；从未实际越界的漏检不会增加计数。
        self.crossed = {i:c for i,c in self.crossed.items() if now-c["at"] <= 4.0}

        # 简化换组规则：配置为一个点时，任意一个已编号槽的真实中心越出
        # ROI 就立即进入退出状态，不再等待整组运动或多个槽同时越界。
        if self.exit_min_slots == 1:
            if self.state == "active" and self.crossed:
                self.state = "exiting"
                self.direction = (self.configured_direction.copy() if self.configured_direction is not None
                                  else np.zeros(2, dtype=float))
                self._required_crossings = 1
                self.exit_since = now
                self._group_events.append({"type": "group_exiting", "group_id": tracker.group_id,
                                           "phase": tracker.phase, "at": now,
                                           "moving_slot_ids": [i+1 for i in sorted(self.crossed)]})
                # 返回False会暂停这一帧的空满确认；退出确认时间到后由update关闭当前组。
                return False
            if self.state == "exiting":
                return False
            return True

        # 框在满槽变空槽时可能收缩、偏移，拟合出的组中心也会漂移。
        # 进入退出状态须有至少两个实际检测中心各自沿外移方向移动，不能只依赖参考模板均值。
        motion_direction = shift/max(travel,1e-9)
        moving_slots = [i for i,obj in reliable.items()
                        if np.dot(np.asarray(obj["center"])-self.anchor_points[i],motion_direction)
                        >= self.anchor_spacing*self.motion_pause_ratio]
        self._motion_support = [i+1 for i in moving_slots]
        # 两槽作共同外移证据时，其位移向量也须一致，避免一个空框大幅偏移带走整组。
        deltas = {i: np.asarray(reliable[i]["center"])-self.anchor_points[i] for i in moving_slots}
        coherent_pair = any(np.linalg.norm(deltas[i]-deltas[j]) <= self.anchor_spacing*self.motion_pause_ratio
                            for index,i in enumerate(moving_slots) for j in moving_slots[index+1:])
        if self.state == "active" and coherent_pair and travel >= self.anchor_spacing*self.motion_pause_ratio:
            if self.configured_direction is None or np.dot(shift,self.configured_direction) > 0:
                self.state = "exiting"
                self.direction = shift/max(travel,1e-9)
                peak_visible = max(n for _,n in self.visible_history)
                self._required_crossings = self.exit_min_slots if peak_visible >= 4 else self.partial_exit_slots
                self._group_events.append({"type": "group_exiting","group_id": tracker.group_id,
                                           "phase": tracker.phase,"at": now,
                                           "moving_slot_ids": list(self._motion_support)})
        if self.state != "exiting":
            return True
        # ROI内的小幅挪动停稳后仍属原组，避免退出候选让状态确认一直暂停。
        if not self.crossed and len(reliable) >= 3 and self._distance(center) >= self.margin:
            if self.center_history and now-self.center_history[0][0]+1e-9 >= .4:
                centers = np.asarray([c for _,c in self.center_history])
                if np.max(np.linalg.norm(centers-center,axis=1)) <= spacing*.05:
                    self._arm(now)
                    self._group_events.append({"type": "group_exit_cancelled","group_id": tracker.group_id,
                                               "phase": tracker.phase,"at": now})
                    return True
        # 退回原位并停稳可取消退出；组号和历史事件保持不变。
        if travel < spacing*self.motion_pause_ratio*.5 and len(reliable) >= 3:
            self.return_since = now if self.return_since is None else self.return_since
            if now-self.return_since+1e-9 >= self.exit_seconds:
                self.state = "active"
                self.crossed.clear()
                self.direction = self.exit_since = None
                self._group_events.append({"type": "group_exit_cancelled","group_id": tracker.group_id,
                                           "phase": tracker.phase,"at": now})
                return True
        else:
            self.return_since = None
        direction = shift/max(travel,1e-9)
        # 转盘沿圆弧运动时，当前位置的径向方向会变化；越界瞬间仍应
        # 与转位开始方向一致或与当前径向方向一致，不能把固定左上向量
        # 当作整段轨迹的切线，否则会漏掉后续槽位。
        onset_direction = self.direction if self.direction is not None else direction
        coherent = [i for i,c in self.crossed.items()
                    if max(np.dot(c["delta"],direction),
                           np.dot(c["delta"],onset_direction))
                    >= np.linalg.norm(c["delta"])*.3]
        outward = self.anchor_distance-self._distance(center) >= self.margin
        direction_ok = self.configured_direction is None or (
            self.direction is not None and np.dot(self.direction,self.configured_direction) >= .7)
        enough = len(coherent) >= self._required_crossings and travel >= spacing*self.minimum_motion_ratio
        if enough and outward and direction_ok:
            self.exit_since = now if self.exit_since is None else self.exit_since
        else:
            self.exit_since = None
        return False

    def _close(self, now):
        self._finish_departing_load(now)
        self._finish_transition_load_review(now)
        review_unload = self.tracker.points is not None and bool(self.tracker.pending_unload_reversals())
        verify_departing = (self.tracker.points is not None and self.tracker.phase == "load"
            and (self.tracker.s4_inferred_load is not None or any(
                s.confirmed == "empty" or s.pending_visual_event is not None for s in self.tracker.slots)))
        if not verify_departing:
            self._deferred_new_events.extend(self.tracker.finish_visual_event_reviews(
                now, preserve_unload_reversals=review_unload))
        snap = self.tracker.snapshot()
        if self.tracker.points is None:
            snap["provisional_events"] = deepcopy(self.tracker._entry_events)
        snap["group_assignment"] = "roi_exit_sequence"
        order = snap["order_result"]
        operation = "fail" if order == "fail" else ("pass" if order == "pass" and snap["all_four_target_previously_verified"] else "pending")
        record = {"type": "group_closed","group_id": self.tracker.group_id,"phase": self.tracker.phase,
                  "at": now,"close_reason": "observed_roi_exit","operation_result": operation,
                  "roi_observed_track_ids": sorted(self._group_roi_tracks.get(
                      (self.tracker.phase, self.tracker.group_id), {})),
                  "roi_observed_track_frame_counts": {track_id: track['inside_frames']
                      for track_id, track in self._group_roi_tracks.get(
                          (self.tracker.phase, self.tracker.group_id), {}).items()},
                  "crossed_slot_ids": [i+1 for i in sorted(self.crossed)],
                  "required_crossings": self._required_crossings,"snapshot": snap,
                  "events": deepcopy(self.tracker.events)}
        if review_unload:
            self._departing_unload_reviews[(self.tracker.phase, self.tracker.group_id)] = self.tracker
            record.update(unload_reversal_review_pending=True, operation_result="pending")
        self.closed_groups.append(deepcopy(record))
        self.track_store.closed.add((self.tracker.phase, self.tracker.group_id))
        self._group_events.append(record)
        # 组号立即切换，但旧组保留一个丢失时限供旋转后的填充框核验。
        # 旧框仍由原有退出点关联排除，不会参与下一组的编号。
        if verify_departing:
            self.departing_load_tracker = self.tracker
            self.departing_load_deadline = now+self.tracker.lost_seconds
            self.departing_points = self.tracker.points.copy()
            self.departing_seen = np.full(4, now)
            self.departing_gate = self.entry_match_distance_px
            self.departing_track_ids = list(self.tracker.byte_track_ids)
            self.departing_untrackable_indices = {
                index for index, slot in enumerate(self.tracker.slots)
                if index == 3 and (slot.geometry_inferred
                                   or self.departing_track_ids[index] is None)
            }
            self.departing_group_id, self.departing_phase = self.tracker.group_id, self.tracker.phase
        self.closed_at = now
        self.departed_points = np.asarray(self.tracker._entry_points if self.tracker.points is None
                                          else self.tracker.reference_points).copy()
        self.departed_velocity = np.zeros(2)
        if len(self.center_history) >= 2:
            t0, c0 = self.center_history[0]
            t1, c1 = self.center_history[-1]
            if t1 > t0:
                self.departed_velocity = (c1-c0)/(t1-t0)
        if self.tracker.group_id == 5 and self.tracker.phase == "unload" and self.auto_load_after_unload:
            self.state = "waiting_next"
            self.pending_tracker = self._new_tracker(1,entry=True,phase="load")
            self._group_events.append({"type": "phase_changed", "from": "unload", "to": "load",
                "group_id": 1, "phase": "load", "at": now,
                "reason": "fifth_unload_group_exited"})
        elif self.tracker.group_id == 5:
            self.state = "sequence_complete"
        else:
            self.state = "waiting_next"
            self.pending_tracker = self._new_tracker(self.tracker.group_id+1,entry=True)
        # 旧组已关闭，永久隔离集合只对该组有效；下一组重新建立自己的集合。
        self._outside_mapping_quarantine_track_ids.clear()

    def _refresh_closed_group(self, tracker, now, reason):
        record = next(g for g in self.closed_groups
                      if g["group_id"] == tracker.group_id and g["phase"] == tracker.phase)
        snapshot = tracker.snapshot()
        snapshot["group_assignment"] = "roi_exit_sequence"
        order = snapshot["order_result"]
        record.update(snapshot=snapshot, events=deepcopy(tracker.events),
            operation_result="fail" if order == "fail" else (
                "pass" if order == "pass" and snapshot["all_four_target_previously_verified"] else "pending"))
        record.pop("unload_reversal_review_pending", None)
        self._group_events.append({"type": "group_updated", "group_id": tracker.group_id, "phase": tracker.phase,
            "at": now, "reason": reason, "snapshot": deepcopy(snapshot)})

    def _review_departing_unloads(self, objects, now, enabled):
        for identity, tracker in list(self._departing_unload_reviews.items()):
            candidates = (tracker._candidates([obj for obj in objects
                if obj.get("detection_source") != "local_crop" and self.track_store.owner(obj.get("track_id")) == identity])
                if enabled else [])
            for slot in tracker.pending_unload_reversals():
                track_id = slot.pending_visual_event.get("byte_track_id")
                record = self.track_store.tracks.get(track_id)
                matches = [obj for obj in candidates if obj.get("track_id") == track_id
                    and record is not None and record["slot_id"] == slot.slot_id
                    and self._distance(obj["center"]) >= -self.buffer_px
                    and np.linalg.norm(np.asarray(obj["center"])-tracker.points[slot.slot_id-1])
                        <= self.entry_match_distance_px]
                obj = matches[0] if len(matches) == 1 else None
                events = tracker.review_departing_unload_slot(slot, obj, self._hands, now)
                if obj is not None and slot.observation != "unknown":
                    tracker.points[slot.slot_id-1] = obj["center"]
                tracker.events.extend(events)
                self._deferred_new_events.extend(events)
            if not tracker.pending_unload_reversals():
                del self._departing_unload_reviews[identity]
                self._refresh_closed_group(tracker, now, "departing_unload_reversal_review")

    def _finish_departing_load(self, now):
        tracker = self.departing_load_tracker
        if tracker is None:
            return
        self._deferred_new_events.extend(tracker.finish_visual_event_reviews(now))
        self._refresh_closed_group(tracker, now, "departing_load_verification")
        self.departing_load_tracker = self.departing_load_deadline = None

    def _verify_departing_load(self, matches, objects, now):
        tracker = self.departing_load_tracker
        if tracker is None:
            return
        hands = [o for o in objects if o["class_name"] == "hand"]
        events = tracker.verify_departing_load(matches, hands, now, enabled=self._observations_enabled)
        self._deferred_new_events.extend(events)
        cancelled = self._record_s4_inferred_load_cancellation(tracker, now)
        if events or cancelled:
            self._refresh_closed_group(tracker, now, "departing_load_verification")
        if now >= self.departing_load_deadline:
            self._finish_departing_load(now)

    def _record_s4_inferred_load_cancellation(self, tracker, now):
        event = tracker.cancelled_s4_inferred_load
        if event is None:
            return False
        tracker.cancelled_s4_inferred_load = None
        self.all_events.remove(event)
        self.cancelled_s4_inferred_loads += 1
        self._group_events.append({"type": "s4_inferred_load_cancelled", "phase": tracker.phase,
            "group_id": tracker.group_id, "at": float(now), "event": deepcopy(event)})
        return True

    def _recheck_quarantined_entry_track(self, objects, admitted, tracker, now):
        """Check whether an unbound handoff track fits the incoming slot geometry."""
        previous = self._transition_recheck
        self._transition_recheck = None
        if not self._transition_recheck_ids or self._transition_quarantine_until is not None:
            return None
        established_ids = {track_id for index, track_id in enumerate(tracker._entry_track_ids)
                           if track_id is not None and index not in tracker._entry_inferred_indices}
        support = tracker._candidates(admitted)
        allowed_support_counts = (2, 3) if tracker.phase == 'load' else (3,)
        if (len(established_ids) not in allowed_support_counts or len(support) != len(established_ids)
                or {obj.get('track_id') for obj in support} != established_ids):
            return None

        def reliable(obj):
            record = self.track_store.tracks.get(obj.get('track_id'))
            return (record is not None and record['reliable_since'] is not None
                    and record['reliable_count'] >= tracker.initialization_frames
                    and now-record['reliable_since']+1e-9 >= tracker.initialization_seconds
                    and self._distance(obj['center']) >= 0 and not obj['conflict']
                    and not tracker._occluded(obj, self._hands))

        if not all(reliable(obj) for obj in support):
            return None
        # The departing tracker has already checked ownership and changed IDs.
        # A candidate still close to a live old slot is ambiguous.
        old_points = [] if self.departing_points is None else [
            point for index, (point, seen) in enumerate(zip(self.departing_points, self.departing_seen))
            if index not in self.departing_untrackable_indices and now-seen <= tracker.lost_seconds]
        centers = [obj['center'] for obj in sorted(support, key=lambda item: item['track_id'])]
        eligible = []
        for obj in tracker._candidates(objects):
            if (obj.get('track_id') not in self._transition_recheck_ids
                    or self.track_store.owner(obj['track_id']) is not None or not reliable(obj)
                    or (tracker.phase == 'unload' and self._in_unload_entry_upper_right_band(obj['center']))
                    or any(np.linalg.norm(np.asarray(obj['center'])-point) <= self.departing_gate*.35
                           for point in old_points)):
                continue
            measured = np.asarray([*centers, obj['center']])
            layout = measured
            if len(measured) == 3:
                if min(np.linalg.norm(measured[-1]-measured[:2], axis=1)) <= self.entry_match_distance_px:
                    continue
                predicted, _ = tracker._infer_entry_s4(measured, self.contour)
                if predicted is None or self._distance(predicted) < 0:
                    continue
                layout = np.vstack((measured, predicted))
            if slot_layout_error(layout) is not None:
                numbering = (tracker._numbering_order_with_predicted_s4(layout, {3}) if len(measured) == 3
                             else tracker._numbering_order(layout))
                eligible.append((obj, measured, tuple(numbering)))
        if len(eligible) != 1:
            return None
        candidate, measured, ordering = eligible[0]
        key = (candidate['track_id'], tuple(sorted(established_ids)), ordering)
        count = 1
        anchor = measured
        spacing = min(np.linalg.norm(a-b) for i, a in enumerate(measured) for b in measured[i+1:])
        if (previous is not None and previous[0] == key and now-previous[2] <= .25
                and np.max(np.linalg.norm(measured-previous[3], axis=1)) <= spacing*self.entry_stability_ratio):
            count, anchor = previous[1]+1, previous[3]
        self._transition_recheck = (key, count, now, anchor)
        return candidate if count >= max(3, tracker.initialization_frames) else None

    def _entry_objects(self, objects, now, *, review_quarantined=False):
        # 旧组的局部补检不能在同帧换组后成为下一组的新轨迹。
        objects = [o for o in objects if o.get("detection_source") != "local_crop"]
        tracker = self.tracker if self.state == "entering" else self.pending_tracker
        # 即使空间隔离或退出几何已超时，旧组归属和被隔离的ID仍须过滤。
        objects = self._exclude_departing(objects, now)
        admitted = self._admit_entry_objects(self._exclude_transition_quarantine(objects, now), tracker, now)
        recovered = None
        if self._transition_entry_track is not None:
            track_id = self._transition_entry_track[0]
            if track_id in tracker._entry_track_ids:
                admitted.extend(obj for obj in objects if obj.get('track_id') == track_id
                                and obj['class_name'] in SLOT_STATES)
                self._track_rejections.pop(track_id, None)
        elif review_quarantined:
            recovered = self._recheck_quarantined_entry_track(objects, admitted, tracker, now)
        else:
            self._transition_recheck = None
        if recovered is not None:
            admitted.append(recovered)
        return admitted, recovered

    def _exclude_departing(self, objects, now):
        """旧组退出后继续沿用旧身份匹配，残留点不能成为下一组的初始槽。"""
        current_tracker = (self.pending_tracker if self.state in ("waiting_next", "stabilizing_next")
                           and self.pending_tracker is not None else self.tracker)
        identity = (current_tracker.phase, current_tracker.group_id)
        known_old = []
        remaining = []
        for obj in objects:
            owner = self.track_store.owner(obj.get('track_id'))
            if obj['class_name'] in SLOT_STATES and owner in self.track_store.closed and owner != identity:
                known_old.append(obj)
                self._reject_track(obj, 'previous_group_track')
            else:
                remaining.append(obj)
        if self.departing_points is None:
            return remaining
        live = np.flatnonzero(now-self.departing_seen <= self.tracker.lost_seconds)
        if not len(live):
            self._verify_departing_load({}, objects, now)
            self._finish_departing_load(now)
            self.departing_points = None
            self.departing_untrackable_indices = set()
            self.departing_group_id = None
            return remaining
        # 只跟随每个旧点自己的真实中心，不用退出时变形较大的整组模板外推。
        candidates = self.tracker._candidates([o for o in objects if o.get("detection_source") != "local_crop"])
        old_points = self.departing_points[live].copy()
        old_identity_indices = [int(index) for index in live
                                if int(index) not in self.departing_untrackable_indices]
        old_identity_points = (self.departing_points[old_identity_indices].copy()
                               if old_identity_indices else np.empty((0, 2), dtype=float))
        prior_points = self.departing_points.copy()
        current_points = (self.tracker._entry_points if self.tracker._entry_points is not None
                          else self.tracker.reference_points)

        def belongs_to_current(center):
            # 旧点短暂漏检时，不能抢走更靠近已建立的新组轨迹的框。
            if current_points is None or not len(current_points):
                return False
            distance = np.min(np.linalg.norm(np.asarray(current_points)-center, axis=1))
            if not len(old_identity_points):
                return False
            return (distance <= self.entry_match_distance_px
                    and distance < np.min(np.linalg.norm(old_identity_points-center, axis=1)))

        known_old_ids = {obj.get('track_id') for obj in known_old}
        candidates = [c for c in candidates if c.get('track_id') in known_old_ids
                      or not belongs_to_current(c["center"])]
        old_boxes, old_matches = [], {}
        if candidates:
            gate = self.departing_gate
            distances = np.linalg.norm(old_points[:, None]
                -np.asarray([c["center"] for c in candidates])[None, :], axis=2)
            # Persistent IDs take precedence over proximity. A clearly new ID
            # must not feed an old-tail track just because it passes nearby.
            for row, index in enumerate(live):
                # An old slot without a real identity cannot associate an
                # ID-less or new-ID box.
                if int(index) in self.departing_untrackable_indices:
                    distances[row, :] = 1e6
                    continue
                old_id = self.departing_track_ids[index] if index < len(self.departing_track_ids) else None
                for col, candidate in enumerate(candidates):
                    track_id = candidate.get('track_id')
                    if track_id is not None and old_id is not None:
                        distances[row, col] = 0. if track_id == old_id else 1e6
                    elif track_id is not None:
                        # A new ID cannot become an old slot merely because it
                        # is nearby; changed IDs are handled only below when
                        # a real old identity supports the same motion.
                        distances[row, col] = 1e6
            cost = np.concatenate((np.where(distances <= gate, distances, 1e6),
                np.full((len(live), len(live)), gate+1)), axis=1)
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if col < len(candidates) and distances[row, col] <= gate:
                    self.departing_points[live[row]] = candidates[col]["center"]
                    self.departing_seen[live[row]] = now
                    old_boxes.append(candidates[col])
                    if live[row] < 4:
                        old_matches[int(live[row])] = candidates[col]
        # Recover changed old IDs only when common real old tracks support the
        # same motion. This is a bounded association check, not a new prediction
        # box for S1-S3 and not a nearest-point claim on an incoming carrier.
        support = [i for i, obj in old_matches.items()
                   if not obj['conflict'] and not self.tracker._occluded(obj, self._hands)]
        if len(support) >= 2 and self.departing_group_id is not None:
            deltas = np.asarray([np.asarray(old_matches[i]['center'])-prior_points[i] for i in support])
            shift = deltas.mean(axis=0)
            if np.max(np.linalg.norm(deltas-shift, axis=1)) <= self.departing_gate*.25:
                used_ids = {c.get('track_id') for c in old_boxes}
                empty_old_slots = set()
                if self.departing_phase == 'unload':
                    old_group = next((group for group in reversed(self.closed_groups)
                                      if group['phase'] == 'unload'
                                      and group['group_id'] == self.departing_group_id), None)
                    if old_group is not None:
                        empty_old_slots = {slot['slot_id']-1 for slot in old_group['snapshot']['slots']
                                           if slot['slot_id'] is not None
                                           and slot['last_confirmed_state'] == 'empty'}
                for i in live:
                    if i in old_matches:
                        continue
                    if int(i) in self.departing_untrackable_indices:
                        continue
                    expected = prior_points[i]+shift
                    options = [c for c in candidates if c.get('track_id') is not None
                        and c['track_id'] not in used_ids
                        and self.track_store.owner(c['track_id']) is None
                        # A newly detected filled slot cannot recover a confirmed empty unload tail.
                        and not (i in empty_old_slots and SLOT_STATES[c['class_name']] == 'filled')
                        and not belongs_to_current(c['center'])
                        and np.linalg.norm(np.asarray(c['center'])-expected) <= self.departing_gate*.35
                        and (len(support) >= 3 or self._entry_progress(c['center']) is None
                             or self._entry_progress(c['center']) > self.entry_birth_fraction)]
                    if len(options) != 1:
                        continue
                    recovered = options[0]
                    other_old = [np.linalg.norm(np.asarray(recovered['center'])-(prior_points[j]+shift))
                                 for j in live if j != i]
                    if other_old and min(other_old) <= np.linalg.norm(np.asarray(recovered['center'])-expected)+self.departing_gate*.1:
                        continue
                    old_boxes.append(recovered)
                    old_matches[int(i)] = recovered
                    used_ids.add(recovered['track_id'])
                    self.departing_points[i], self.departing_seen[i] = recovered['center'], now
                    if i < len(self.departing_track_ids):
                        self.departing_track_ids[i] = recovered['track_id']
                    self.track_store.bind(recovered['track_id'],
                        (self.departing_phase, self.departing_group_id), int(i)+1 if i < 4 else None)
        self._verify_departing_load(old_matches, objects, now)
        result = []
        for obj in objects:
            if obj["class_name"] in SLOT_STATES:
                if obj.get('track_id') in known_old_ids:
                    continue
                a = obj["xyxy"]
                center = np.array([(a[0]+a[2])/2, (a[1]+a[3])/2])
                if belongs_to_current(center):
                    result.append(obj)
                    continue
                if any((obj.get('track_id') is None or obj.get('track_id') == old.get('track_id'))
                       and (np.linalg.norm(np.asarray(old["center"])-center) <= 2
                            or box_iou(old["xyxy"], a) > .85) for old in old_boxes):
                    self._reject_track(obj, 'previous_group_track')
                    continue
                live_identity_indices = [int(index) for index in live
                                         if int(index) not in self.departing_untrackable_indices]
                if (obj.get('track_id') is None and live_identity_indices and
                        np.min(np.linalg.norm(self.departing_points[live_identity_indices]-center, axis=1))
                        <= self.departing_gate*.5):
                    continue
            result.append(obj)
        return result

    def _start_immediate_next(self, now, objects):
        previous_tracker = self.tracker
        self.departing_group_id = previous_tracker.group_id
        self.departing_phase = previous_tracker.phase
        self.departing_points = np.asarray(self.tracker._entry_points if self.tracker.points is None
                                          else self.tracker.points).copy()
        self.departing_track_ids = list(previous_tracker.byte_track_ids if previous_tracker.points is not None
                                         else previous_tracker._entry_track_ids)
        # If the old S4 was inferred or never had a real ByteTrack ID, freeze
        # it at group handoff.  Keep the old real tracks for quarantine, but do
        # not let a new detection become the closed group's S4.
        self.departing_untrackable_indices = set()
        if (len(self.departing_points) >= 4 and len(self.departing_track_ids) >= 4
                and (self.departing_track_ids[3] is None
                     or (previous_tracker.points is not None
                         and previous_tracker.slots[3].geometry_inferred))):
            self.departing_untrackable_indices.add(3)
        # 已知旧组ID继续作为尾部排除；有新ID的出口侧框仍需重新判断归属，
        # 不能一律扣到旧组。无ID输入保留原几何兼容路径。
        for obj in self.tracker._candidates([o for o in objects if o.get("detection_source") != "local_crop"]):
            center = np.asarray(obj["center"])
            progress = self._entry_progress(center)
            if (self._distance(center) >= 0 and progress is not None and progress > self.entry_birth_fraction
                    and (obj.get('track_id') is None or self.track_store.owner(obj.get('track_id'))
                         == (previous_tracker.phase, previous_tracker.group_id))
                    and np.min(np.linalg.norm(self.departing_points-center, axis=1)) > 2):
                self.departing_points = np.vstack((self.departing_points, center))
                self.departing_track_ids.append(obj.get('track_id'))
        self.departing_seen = np.full(len(self.departing_points), now)
        # 进入时用明确的像素门限，避免旧组退出时透视压缩把新点匹配距离缩得过小。
        gate = self.entry_match_distance_px
        self.departing_gate = gate
        self.tracker = self.pending_tracker
        self.pending_tracker = None
        self.tracker.start_incremental_entry(gate)
        self._entry_revision = self.tracker._entry_revision
        self.state = "entering"
        self.inside_seen, self.crossed = {}, {}
        self.direction = self.exit_since = None
        self._motion_support = []
        self._group_events.append({"type": "group_started", "group_id": self.tracker.group_id,
            "phase": self.tracker.phase, "at": now, "reason": "single_slot_roi_exit"})

    def _start_transition_load_review(self, tracker, track_id, support_ids, numbering, now):
        roles = {point: slot for slot, point in enumerate(numbering)}
        anchors = tuple((roles[i], track) for i, track in enumerate(support_ids))
        index = roles[2]
        recorded = self.track_store.tracks[track_id]['events']
        has_history = any(event['action'] == 'load' and event['confirmed_at'] < now for event in recorded)
        if (any(tracker.byte_track_ids[i] != track for i, track in anchors)
                or (track_id in tracker.byte_track_ids and tracker.byte_track_ids[index] != track_id)):
            if has_history:
                self._group_events.append({"type": "transition_load_history_unconfirmed", "track_id": track_id,
                    "group_id": tracker.group_id, "phase": tracker.phase, "at": now,
                    "slot_id": None, "reason": "numbering_roles_disagree"})
            return
        tracker.transition_load_slot_index = index
        if has_history:
            self._transition_load_review = (track_id, anchors, now)

    def _finish_transition_load_review(self, now):
        if self._transition_load_review is None:
            return
        track_id, _, _ = self._transition_load_review
        self._group_events.append({"type": "transition_load_history_unconfirmed", "track_id": track_id,
            "group_id": self.tracker.group_id, "phase": self.tracker.phase, "at": now,
            "slot_id": self.tracker.transition_load_slot_index+1})
        self._transition_load_review = None

    def _review_transition_load(self, result, now):
        if self._transition_load_review is None or self.state != 'active':
            return result
        tracker = self.tracker
        track_id, anchors, started = self._transition_load_review
        index = tracker.transition_load_slot_index
        slot = tracker.slots[index]
        matched_id = tracker.byte_track_ids[index]
        record = self.track_store.tracks.get(matched_id)
        old = self.track_store.tracks.get(track_id)
        if (record is None or old is None or slot.state != 'filled' or not slot.geometry_observed
                or record['state'].state != 'filled' or record['state'].confirmation_source != 'yolo'
                or slot.pending_visual_event is not None
                or any(tracker.byte_track_ids[i] != anchor or not tracker.slots[i].geometry_observed
                       or self.track_store.tracks[anchor]['reliable_since'] is None for i, anchor in anchors)
                or record['last_seen'] != now or record['reliable_since'] is None
                or record['reliable_count'] < max(tracker.initialization_frames, tracker.minimum_frames)
                or now-max(started, record['reliable_since'])+1e-9
                    < max(tracker.initialization_seconds, tracker.confirm_seconds)):
            return result
        recorded = old['events']
        if (len(recorded) != 1 or recorded[0]['action'] != 'load' or recorded[0]['confirmed_at'] >= started
                or now-recorded[0]['confirmed_at']+1e-9 < tracker.visual_reversal_review_seconds
                or (matched_id != track_id and (old['last_seen'] >= record['first_seen']
                    or record['first_seen']-old['last_seen'] > tracker.initialization_history_seconds))):
            return result
        existing = [event for event in tracker.events if event['slot_id'] == slot.slot_id]
        if existing and (len(existing) != 1 or existing[0]['action'] != 'load'):
            return result
        event = {**recorded[0], "slot_id": slot.slot_id, "group_id": tracker.group_id, "phase": "load",
                 "expected_action": True, "reported_at": now, "recovered_before_initialization": True,
                 "reason": "track_history_before_slot_mapping"}
        if matched_id != track_id:
            event["reassociated_to_byte_track_id"] = matched_id
        self.track_store.bind(track_id, (tracker.phase, tracker.group_id), slot.slot_id)
        self._transition_load_review = None
        self._group_events.append({"type": "transition_load_history_confirmed", "track_id": track_id,
            "group_id": tracker.group_id, "phase": tracker.phase, "at": now,
            "slot_id": slot.slot_id, "matched_track_id": matched_id})
        new_events = result["new_events"]
        if existing:
            previous = existing[0]
            if previous["time_upper"] <= event["time_upper"]:
                return result
            tracker.events.remove(previous)
            new_events = [item for item in new_events if item != previous]
            if previous in self.all_events:
                self.all_events.remove(previous)
                self.revised_transition_loads += 1
                event["supersedes_reported_at"] = previous["reported_at"]
        tracker.events.append(event)
        return {**tracker.snapshot(), "new_events": [*new_events, event]}

    def _update_immediate_entry(self, objects, now, enabled, *, load_support_resolver=None,
                                entry_position_candidates=()):
        objects, recovered = self._entry_objects(
            objects, now, review_quarantined=enabled)
        candidate = self._transition_entry_track
        if recovered is not None:
            candidate = self._transition_recheck[0]
        self._entry_count = len(self.tracker._candidates(objects))
        two_anchor_track_id = candidate[0] if candidate is not None and len(candidate[1]) == 2 else None
        self.tracker.transition_load_track_id = two_anchor_track_id
        self._record_entry_s4_load_contact(self.tracker, objects, now, enabled)
        result = self.tracker.update(objects, now, observations_enabled=enabled, roi_contour=self.contour,
                                     entry_position_candidates=entry_position_candidates,
                                     load_support_resolver=load_support_resolver)
        if recovered is not None:
            track_id = recovered['track_id']
            if track_id in self.tracker._entry_track_ids or track_id in self.tracker.byte_track_ids:
                self._track_rejections.pop(track_id, None)
                self._transition_entry_track = candidate
                self._transition_recheck = None
        if self.tracker.points is not None and self._transition_entry_track is not None:
            track_id, support_ids, numbering = self._transition_entry_track
            if two_anchor_track_id is not None:
                self._start_transition_load_review(self.tracker, track_id, support_ids, numbering, now)
            if track_id in self.tracker.byte_track_ids or self._transition_load_review is not None:
                self._transition_quarantine_track_ids.discard(track_id)
                self._transition_recheck_ids.discard(track_id)
                self._group_events.append({'type': 'transition_track_recovered', 'track_id': track_id,
                    'group_id': self.tracker.group_id, 'phase': self.tracker.phase, 'at': now,
                    'support_track_ids': list(support_ids)})
            self._transition_entry_track = None
        if self._entry_revision != self.tracker._entry_revision:
            self.inside_seen, self.crossed = {}, {}
            self._entry_revision = self.tracker._entry_revision
        if self.tracker.points is not None:
            result = self._restore_load_baseline(result, now)
            self._s4_load_lookback.clear()
            self._s4_load_hint = False
            self._arm(now)
        else:
            # A single physical turn closes at most one established group.
            # Unnumbered/residual entry points cannot advance G again.
            self.inside_seen.clear()
            self.crossed.clear()
        return result

    def _record_entry_s4_load_contact(self, tracker, objects, now, enabled):
        if (not enabled or tracker.phase != "load" or not tracker.load_hand_contact_enabled
                or tracker.identity_lost or self._transition_quarantine_until is not None
                or (not self._transition_recheck_ids and tracker._load_numbering_reference is None)):
            self._s4_load_lookback.clear()
            self._s4_load_hint = False
            return
        identity = (tracker.phase, tracker.group_id)
        admitted = {obj["track_id"]: obj for obj in tracker._candidates(objects)
                    if obj.get("track_id") is not None and obj.get("detection_source") != "local_crop"}
        tracks = {}
        for track_id in admitted.keys() | set(tracker._entry_track_ids) | self._transition_recheck_ids:
            record = self.track_store.tracks.get(track_id)
            if record is not None and record["owner"] in (None, identity):
                obj = admitted.get(track_id)
                if track_id in self._transition_recheck_ids or (obj is not None and not obj["conflict"]
                        and not tracker._occluded(obj, self._hands)):
                    tracks[track_id] = record
                elif track_id in tracker._entry_track_ids:
                    tracks[track_id] = {**record, "reliable_since": None}
        previous = self.closed_groups[-1] if self.closed_groups else None
        reference = None
        if previous is not None and previous["phase"] == "load":
            centers = previous["snapshot"]["initialization"].get("initial_centers_xy")
            if centers is not None and slot_layout_error(centers) is not None:
                reference = np.asarray(centers)
        self._s4_load_hint = self._s4_load_lookback.record(
            tracker, now, self._hands, tracks, reference, self._transition_recheck_ids,
            reach_hand=self._entry_load_reach_hand)

    def _update_entry(self, objects, now, enabled, *, load_support_resolver=None, entry_position_candidates=()):
        objects, _ = self._entry_objects(objects,now)
        t = self.pending_tracker
        candidates = t._candidates(objects)
        self._entry_count = len(candidates)
        self._record_entry_s4_load_contact(t, objects, now, enabled)
        result = t.update(objects,now,observations_enabled=enabled,initialization_objects=objects,roi_contour=self.contour,
                          entry_position_candidates=entry_position_candidates,
                          load_support_resolver=load_support_resolver)
        # 停稳计时只确认几何身份；候选跟踪器独立缓存各槽空满和取放事件。
        self.state = "stabilizing_next" if t.initialization["stable_frames"] else "waiting_next"
        if t.points is not None:
            self.tracker = t
            result = self._restore_load_baseline(result, now)
            self._s4_load_lookback.clear()
            self._s4_load_hint = False
            self.pending_tracker = None
            self._arm(now)
            self._group_events.append({"type": "group_started","group_id": t.group_id,"phase": t.phase,"at": now})
            return result
        return None

    def update(self, objects, now, *, observations_enabled=True, s4_position_candidates=(),
               load_support_resolver=None, entry_load_reach_hand=None, entry_position_candidates=(),
               initial_edge_candidates=()):
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("分组时间必须严格递增")
        self.last_time = float(now)
        self._group_events = []
        self._deferred_new_events = []
        self._entry_rejected = []
        self._track_rejections = {}
        self._observations_enabled = observations_enabled
        self._hands = [o for o in objects if o["class_name"] == "hand"]
        self._entry_load_reach_hand = entry_load_reach_hand
        audit_objects = objects
        if observations_enabled:
            objects = self._recover_crossed_load_binding(objects, now)
        else:
            self._load_rebind_pose = self._pending_load_rebind = None
        objects = self._admit_initial_edge_slot(objects, initial_edge_candidates, now)
        self._record_group_roi_tracks(objects, now)
        self._record_outside_mapping_quarantine(objects, now)
        edge_tracker = self.pending_tracker or self.tracker
        self.track_store.observe_roi_edge(objects, now, (edge_tracker.phase, edge_tracker.group_id),
            observations_enabled and self.state not in ('exiting', 'identity_uncertain', 'sequence_complete'),
            self._transition_quarantine_track_ids | self._outside_mapping_quarantine_track_ids)
        result = None
        discarded_initial_candidate = self._discard_unconfirmed_initial_g1(now, objects)
        if discarded_initial_candidate:
            # Do not feed the just-discarded outside frame back into the new
            # G1 candidate.  The next frame starts a clean initialization.
            result = {**self.tracker.snapshot(), "new_events": []}
        else:
            if observations_enabled:
                self._split_reused_track_ids(objects, now)
            self.track_store.observe(objects, now, observations_enabled,
                reentry_allowed=lambda center: self._distance(center) >= self.margin and (
                    self._entry_progress(center) is None
                    or self._entry_progress(center) <= self.entry_birth_fraction))
        self._review_departing_unloads(objects, now, observations_enabled)
        if not discarded_initial_candidate and self.state == "entering":
            result = self._update_immediate_entry(objects, now, observations_enabled,
                                                  entry_position_candidates=entry_position_candidates,
                                                  load_support_resolver=load_support_resolver)
        elif not discarded_initial_candidate and self.state in ("waiting_next","stabilizing_next"):
            result = self._update_entry(objects, now, observations_enabled,
                                        entry_position_candidates=entry_position_candidates,
                                        load_support_resolver=load_support_resolver)
        elif not discarded_initial_candidate and self.state == "sequence_complete":
            self._exclude_departing(objects, now)
        elif not discarded_initial_candidate:
            # 已归属旧组或具有稳定旧组证据的ID不能回流到当前槽号跟踪；
            # 先让旧组尾部逻辑观察，再从当前组候选中移除。
            active = self._active_objects(self._exclude_transition_quarantine(
                self._exclude_departing(objects, now), now), now)
            initial = [o for o in active if o["class_name"] not in SLOT_STATES or o.get('initial_edge_slot') or self._distance(
                ((o["xyxy"][0]+o["xyxy"][2])/2,(o["xyxy"][1]+o["xyxy"][3])/2)) >= 0]
            position_candidates = ()
            if s4_position_candidates and observations_enabled and self.state == "active":
                # 无ID框与已有轨迹重复时采用已有轨迹。objects 也包含被排除的旧组轨迹。
                tracked_boxes = [obj["xyxy"] for obj in objects if obj["class_name"] in SLOT_STATES]
                position_candidates = [obj for obj in s4_position_candidates
                    if not any(box_containment(obj["xyxy"], box) >= .9 for box in tracked_boxes)]
            result = self.tracker.update(active,now,observations_enabled=observations_enabled,
                initialization_objects=initial,roi_contour=self.contour,
                before_observe=self._observe_motion,
                translation_direction=self.direction if self.state == "exiting" else None,
                s4_position_candidates=position_candidates,
                recheck_initial_unload=self._g1_waiting_event,
                load_support_resolver=load_support_resolver)
            if observations_enabled:
                result = self._review_transition_load(result, now)
            self._record_s4_inferred_load_cancellation(self.tracker, now)
            if self.tracker.points is not None and self.state == "initializing":
                self._arm(now)
            if self.state == "exiting" and (self.exit_min_slots == 1 or
                    self.exit_since is not None and now-self.exit_since+1e-9 >= self.exit_seconds):
                # 未绑定ID仅在旧组真实出现帧数足够且仍在ROI/短暂漏检时隔离；
                # 已绑定ID继续由旧组归属排除，空间隔离保持不变。
                self._arm_transition_quarantine(objects, now)
                reviewed = self.tracker.finish_s4_unload_review(now)
                if reviewed:
                    result = {**result, **self.tracker.snapshot(), "new_events": result["new_events"] + reviewed}
                self._close(now)
                if self.exit_min_slots == 1 and self.state != "sequence_complete":
                    closing_events = result["new_events"]
                    self._start_immediate_next(now, objects)
                    result = self._update_immediate_entry(objects, now, observations_enabled,
                                                          load_support_resolver=load_support_resolver)
                    result["new_events"] = closing_events + result["new_events"]
        if result is None:
            result = {**self.tracker.snapshot(),"new_events": []}
        g1_waiting_before_release = self._g1_waiting_event
        result = self._release_initial_g1_event_gate(result)
        if (g1_waiting_before_release and not self._g1_waiting_event
                and self.tracker.group_id == 1 and self.tracker.phase == 'unload'):
            # 首个下料事件可能早于正式编号；只有正式映射后才启动ROI外ID隔离。
            # 若几何已提前进入active，则从事件确认时重新起算窗口。
            self._outside_mapping_quarantine_track_ids.clear()
            self.mapping_time = float(now) if self.tracker.points is not None else None
        if self._initial_g1_waiting_for_unload():
            self._g1_pending_events.extend(deepcopy(result.get("new_events", [])))
            result["new_events"] = []
            result = self._mask_initial_g1_result(result)
        self._sync_track_bindings(result, now)
        tracker = self.tracker
        if (observations_enabled and self.state == "active" and tracker.phase == "load"
                and result["slot_numbering_status"] == "final" and tracker.transform_valid_this_frame):
            self._load_rebind_pose = (now, (tracker.phase, tracker.group_id), tracker.reference_points.copy(),
                                      tuple(i-1 for i in tracker.transform_support_slot_ids))
        if observations_enabled and self.state == 'active' and result['slot_numbering_status'] == 'final':
            edge_events = self.track_store.restore_roi_edge_events(self.tracker, now)
            if edge_events:
                result = {**result, **self.tracker.snapshot(), 'new_events': result['new_events'] + edge_events}
        if self.use_previous_s4_layouts or self.load_numbering_calibration is not None:
            self._record_unload_layout(result, now)
        self._record_g1_load_layout(result, now)
        result["new_events"] = list(result.get("new_events", [])) + deepcopy(self._deferred_new_events)
        self.all_events.extend(deepcopy(result["new_events"]))
        if self.state in ("waiting_next","stabilizing_next","sequence_complete"):
            # 旧组已关闭；等待阶段不重复输出旧组的当前空/满状态。
            for slot in result["slots"]:
                slot.update(state="unknown",observation="unknown",reason=self.state,
                            position_is_prediction=(slot["slot_id"] == 4
                                and slot["center_xy"] is not None
                                and slot.get("tracking_box_xyxy") is not None))
            result["current_target_count"] = 0
            result["all_four_currently_confirmed"] = False
        return {**result,**self.metadata(),"track_states": self._track_metadata(audit_objects, result),
                "new_group_events": deepcopy(self._group_events)}

    def finish(self):
        """结束退出核验；没有空到满的视觉证据就始终保留未完成。"""
        self._group_events, self._deferred_new_events = [], []
        if self.last_time is None:
            return {**self.snapshot(), "new_events": [], "new_group_events": []}
        now = self.last_time
        self._finish_transition_load_review(now)
        self._deferred_new_events.extend(self.tracker.finish_s4_unload_review(now))
        for tracker in self._departing_unload_reviews.values():
            tracker.finish_pending_unload_reversals()
            self._refresh_closed_group(tracker, now, "departing_unload_reversal_review")
        self._departing_unload_reviews.clear()
        self._deferred_new_events.extend(self.tracker.finish_visual_event_reviews(now))
        self._finish_departing_load(now)
        self.all_events.extend(deepcopy(self._deferred_new_events))
        return {**self.snapshot(), "new_events": deepcopy(self._deferred_new_events),
                "new_group_events": deepcopy(self._group_events)}

    def metadata(self):
        preview_tracker = self.pending_tracker or self.tracker
        return {"group_assignment": "roi_exit_sequence" if self.enabled else "configured",
                "group_status": self.state,"pending_group_id": self.pending_tracker.group_id
                    if self.pending_tracker is not None else None,
                "s4_load_contact_review": ("suspected" if self._s4_load_hint else
                    "waiting_for_visual" if self.tracker.s4_load_contact_lookback is not None else None),
                "three_point_preview": (preview_tracker.three_point_preview
                                        if preview_tracker.points is None else None),
                "pending_phase": self.pending_tracker.phase if self.pending_tracker is not None else None,
                "closed_group_count": len(self.closed_groups),
                "crossed_slot_ids": [i+1 for i in sorted(self.crossed)],
                "required_exit_slots": self._required_crossings,
                "motion_support_slot_ids": list(self._motion_support),
                "entry_candidate_count": self._entry_count,
                "entry_rejected_slot_detections": deepcopy(self._entry_rejected),
                "roi_observed_track_ids": sorted(self._group_roi_tracks.get(
                    (self.tracker.phase, self.tracker.group_id), {})),
                "roi_observed_track_frame_counts": {track_id: track['inside_frames']
                    for track_id, track in self._group_roi_tracks.get(
                        (self.tracker.phase, self.tracker.group_id), {}).items()},
                "transition_quarantine_track_ids": sorted(self._transition_quarantine_track_ids),
                "transition_quarantine_until": self._transition_quarantine_until,
                "mapping_time": self.mapping_time,
                "outside_mapping_quarantine_seconds": self.outside_mapping_quarantine_seconds,
                "outside_mapping_quarantine_until": (None if self.mapping_time is None else
                    self.mapping_time + self.outside_mapping_quarantine_seconds),
                "outside_mapping_quarantine_track_ids": sorted(
                    self._outside_mapping_quarantine_track_ids),
                "departing_untrackable_slot_ids": [index + 1 for index in
                    sorted(self.departing_untrackable_indices)],
                "pending_initialization": deepcopy(self.pending_tracker.initialization)
                    if self.pending_tracker is not None else None}

    def snapshot(self):
        return {**self.tracker.snapshot(),**self.metadata()}
