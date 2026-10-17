"""同一ROI内按顺序处理五组，可自动衔接下料和上料；以真实越界切组。"""
from __future__ import annotations

from collections import deque
from copy import deepcopy

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from industrial_inspection.tools.slot_group_tracker import SLOT_STATES, box_iou


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
                 auto_load_after_unload=False):
        self.tracker = tracker
        self.contour = np.asarray(contour,dtype=np.float32)
        if self.contour.ndim != 2 or self.contour.shape[1] != 2 or not cv2.isContourConvex(self.contour):
            raise ValueError("换组ROI必须为凸多边形像素坐标")
        if not 0 < boundary_margin_px < buffer_px or exit_seconds <= 0 or entry_seconds <= 0:
            raise ValueError("换组缓冲带必须大于边界容差；退出/进入确认时间须大于0")
        if not 1 <= partial_exit_slots <= exit_min_slots <= 4:
            raise ValueError("完整/部分可见时的越界槽数应满足1<=部分<=完整<=4")
        if not 0 < motion_pause_ratio < minimum_motion_ratio or not 0 < entry_stability_ratio < .5:
            raise ValueError("运动暂停比例须小于退出位移比例；停稳比例须在0..0.5之间")
        if not np.isfinite(entry_match_distance_px) or entry_match_distance_px <= 0:
            raise ValueError("新组逐点匹配距离须为正有限像素值")
        self.enabled, self.buffer_px, self.margin = enabled, buffer_px, boundary_margin_px
        self.auto_load_after_unload = bool(auto_load_after_unload)
        self.exit_seconds, self.exit_min_slots = exit_seconds, exit_min_slots
        self.partial_exit_slots, self.minimum_motion_ratio = partial_exit_slots, minimum_motion_ratio
        self.motion_pause_ratio, self.entry_seconds = motion_pause_ratio, entry_seconds
        self.entry_stability_ratio = entry_stability_ratio
        self.entry_match_distance_px = float(entry_match_distance_px)
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
        self.pending_tracker = None
        self.departing_points = None
        self.departing_seen = None
        self.departing_gate = None
        self.closed_at, self.departed_points, self.departed_velocity = None, None, None
        self.last_time = None
        self._group_events = []
        self._required_crossings = exit_min_slots
        self._entry_count = 0
        self._motion_support = []
        self.departing_load_tracker = None
        self.departing_load_deadline = None
        self._deferred_new_events = []
        self._observations_enabled = True
        self._restored_load_groups = set()

    def _distance(self, point):
        return cv2.pointPolygonTest(self.contour,tuple(float(v) for v in point),True)

    def _new_tracker(self, group_id, *, entry=False, phase=None):
        result = deepcopy(self.tracker)
        result.reset_group(group_id,phase=phase)
        if entry:
            result.initialization_seconds = self.entry_seconds
            result.initialization_stability_ratio = self.entry_stability_ratio
        return result

    def _restore_load_baseline(self, result, now):
        """正式编号后复用同组下料时真实检测的空槽，绝不复用旧坐标。"""
        tracker = self.tracker
        if tracker.phase != "load" or tracker.points is None or tracker.group_id in self._restored_load_groups:
            return result
        self._restored_load_groups.add(tracker.group_id)
        previous = next((g for g in self.closed_groups
                         if g["phase"] == "unload" and g["group_id"] == tracker.group_id), None)
        if previous is None:
            return result
        new_events = list(result["new_events"])
        for slot, baseline in zip(tracker.slots, previous["snapshot"]["slots"]):
            if (baseline["last_confirmed_state"] != "empty" or baseline["last_confirmed_at"] is None
                    or any(e["slot_id"] == slot.slot_id for e in tracker.events)):
                continue
            if slot.confirmed == "unknown":
                slot.confirmed, slot.evidence_time = "empty", baseline["last_confirmed_at"]
                slot.confirmation_source = "previous_unload_yolo"
            elif slot.confirmed == "filled":
                # 上料在正式编号前已完成时，空和满都必须有实际检测依据。
                # 区间使用上次空槽时间到首次本轮满槽时间，不用手接触猜时刻。
                event = {"group_id":tracker.group_id,"slot_id":slot.slot_id,"phase":"load",
                    "from":"empty","to":"filled","action":"load","expected_action":True,
                    "time_lower":baseline["last_confirmed_at"],
                    "time_upper":slot.candidate_since if slot.candidate == "filled" else slot.evidence_time,
                    "confirmed_at":slot.evidence_time,"reported_at":float(now),
                    "recovered_before_initialization":True,"reason":"detected_after_unload_baseline"}
                tracker.events.append(event)
                new_events.append(event)
        return {**tracker.snapshot(),"new_events":new_events}

    def _arm(self, now):
        self.state = "active"
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

    def _active_objects(self, objects):
        """ROI外只接受能关联到当前已编号槽的候选，其他外部槽不参与。"""
        result = []
        reference = self.tracker.reference_points
        gate = None if reference is None else (self.tracker._point_tracking_gate
            or self.tracker._spacing()*self.tracker.match_distance_ratio)
        for obj in objects:
            if obj["class_name"] not in SLOT_STATES:
                result.append(obj)
                continue
            a = obj["xyxy"]
            center = np.array([(a[0]+a[2])/2,(a[1]+a[3])/2])
            distance = self._distance(center)
            if distance >= 0 or (reference is not None and distance >= -self.buffer_px
                    and np.min(np.linalg.norm(reference-center,axis=1)) <= gate):
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
        snap = self.tracker.snapshot()
        snap["group_assignment"] = "roi_exit_sequence"
        order = snap["order_result"]
        operation = "fail" if order == "fail" else ("pass" if order == "pass" and snap["all_four_target_previously_verified"] else "pending")
        record = {"type": "group_closed","group_id": self.tracker.group_id,"phase": self.tracker.phase,
                  "at": now,"close_reason": "observed_roi_exit","operation_result": operation,
                  "crossed_slot_ids": [i+1 for i in sorted(self.crossed)],
                  "required_crossings": self._required_crossings,"snapshot": snap,
                  "events": deepcopy(self.tracker.events)}
        self.closed_groups.append(deepcopy(record))
        self._group_events.append(record)
        # 组号立即切换，但旧组保留一个丢失时限供旋转后的填充框核验。
        # 旧框仍由原有退出点关联排除，不会参与下一组的编号。
        if self.tracker.phase == "load" and any(s.confirmed == "empty" for s in self.tracker.slots):
            self.departing_load_tracker = self.tracker
            self.departing_load_deadline = now+self.tracker.lost_seconds
            self.departing_points = self.tracker.points.copy()
            self.departing_seen = np.full(4, now)
            self.departing_gate = self.entry_match_distance_px
        self.closed_at = now
        self.departed_points = self.tracker.reference_points.copy()
        self.departed_velocity = np.zeros(2)
        if len(self.center_history) >= 2:
            t0,c0 = self.center_history[0]
            t1,c1 = self.center_history[-1]
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

    def _refresh_closed_load(self, tracker, now):
        record = next(g for g in self.closed_groups
                      if g["group_id"] == tracker.group_id and g["phase"] == tracker.phase)
        snapshot = tracker.snapshot()
        snapshot["group_assignment"] = "roi_exit_sequence"
        order = snapshot["order_result"]
        record.update(snapshot=snapshot, events=deepcopy(tracker.events),
            operation_result="fail" if order == "fail" else (
                "pass" if order == "pass" and snapshot["all_four_target_previously_verified"] else "pending"))
        self._group_events.append({"type": "group_updated", "group_id": tracker.group_id, "phase": tracker.phase,
            "at": now, "reason": "departing_load_verification", "snapshot": deepcopy(snapshot)})

    def _finish_departing_load(self, now):
        tracker = self.departing_load_tracker
        if tracker is None:
            return
        self._refresh_closed_load(tracker, now)
        self.departing_load_tracker = self.departing_load_deadline = None

    def _verify_departing_load(self, matches, objects, now):
        tracker = self.departing_load_tracker
        if tracker is None:
            return
        hands = [o for o in objects if o["class_name"] == "hand"]
        events = tracker.verify_departing_load(matches, hands, now, enabled=self._observations_enabled)
        self._deferred_new_events.extend(events)
        if events:
            self._refresh_closed_load(tracker, now)
        if now >= self.departing_load_deadline:
            self._finish_departing_load(now)

    def _entry_objects(self, objects, now):
        # 旧组的局部补检不能在同帧换组后成为下一组的新轨迹。
        objects = [o for o in objects if o.get("detection_source") != "local_crop"]
        if self.state == "entering" or self.departing_points is not None:
            return [o for o in self._exclude_departing(objects, now) if o["class_name"] not in SLOT_STATES
                or self._distance(((o["xyxy"][0]+o["xyxy"][2])/2,
                                   (o["xyxy"][1]+o["xyxy"][3])/2)) >= 0]
        inside,others = [],[]
        # 沿已确认退出方向短时预测旧组尾部，只作排除，不把预测当成检测证据。
        age = now-self.closed_at
        old = self.departed_points+self.departed_velocity*age
        gate = self.tracker._spacing()*self.tracker.match_distance_ratio
        for obj in objects:
            if obj["class_name"] not in SLOT_STATES:
                others.append(obj)
                continue
            a = obj["xyxy"]
            point = np.array([(a[0]+a[2])/2,(a[1]+a[3])/2])
            if self._distance(point) < self.margin:
                continue
            if age <= self.tracker.lost_seconds and np.min(np.linalg.norm(old-point,axis=1)) <= gate:
                continue
            inside.append(obj)
        return inside+others

    def _exclude_departing(self, objects, now):
        """旧组退出后继续沿用旧身份匹配，残留点不能成为下一组的初始槽。"""
        if self.departing_points is None:
            return objects
        live = np.flatnonzero(now-self.departing_seen <= self.tracker.lost_seconds)
        if not len(live):
            self._verify_departing_load({}, objects, now)
            self._finish_departing_load(now)
            self.departing_points = None
            return objects
        # 只跟随每个旧点自己的真实中心，不用退出时变形较大的整组模板外推。
        candidates = self.tracker._candidates([o for o in objects if o.get("detection_source") != "local_crop"])
        old_points = self.departing_points[live].copy()
        current_points = (self.tracker._entry_points if self.tracker._entry_points is not None
                          else self.tracker.reference_points)

        def belongs_to_current(center):
            # 旧点短暂漏检时，不能抢走更靠近已建立的新组轨迹的框。
            if current_points is None or not len(current_points):
                return False
            distance = np.min(np.linalg.norm(np.asarray(current_points)-center, axis=1))
            return (distance <= self.entry_match_distance_px
                    and distance < np.min(np.linalg.norm(old_points-center, axis=1)))

        candidates = [c for c in candidates if not belongs_to_current(c["center"])]
        old_boxes, old_matches = [], {}
        if candidates:
            gate = self.departing_gate
            distances = np.linalg.norm(old_points[:, None]
                -np.asarray([c["center"] for c in candidates])[None, :], axis=2)
            cost = np.concatenate((np.where(distances <= gate, distances, 1e6),
                np.full((len(live), len(live)), gate+1)), axis=1)
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if col < len(candidates) and distances[row, col] <= gate:
                    self.departing_points[live[row]] = candidates[col]["center"]
                    self.departing_seen[live[row]] = now
                    old_boxes.append(candidates[col])
                    old_matches[int(live[row])] = candidates[col]
        self._verify_departing_load(old_matches, objects, now)
        result = []
        for obj in objects:
            if obj["class_name"] in SLOT_STATES:
                a = obj["xyxy"]
                center = np.array([(a[0]+a[2])/2, (a[1]+a[3])/2])
                if belongs_to_current(center):
                    result.append(obj)
                    continue
                if any(np.linalg.norm(np.asarray(old["center"])-center) <= 2
                        or box_iou(old["xyxy"], a) > .85 for old in old_boxes):
                    continue
                if np.min(np.linalg.norm(self.departing_points[live]-center, axis=1)) <= self.departing_gate*.5:
                    continue
            result.append(obj)
        return result

    def _start_immediate_next(self, now):
        self.departing_points = self.tracker.points.copy()
        self.departing_seen = np.full(4, now)
        # 进入时用明确的像素门限，避免旧组退出时透视压缩把新点匹配距离缩得过小。
        gate = self.entry_match_distance_px
        self.departing_gate = gate
        self.tracker = self.pending_tracker
        self.pending_tracker = None
        self.tracker.start_incremental_entry(gate)
        self.state = "entering"
        self.inside_seen, self.crossed = {}, {}
        self.direction = self.exit_since = None
        self._motion_support = []
        self._group_events.append({"type": "group_started", "group_id": self.tracker.group_id,
            "phase": self.tracker.phase, "at": now, "reason": "single_slot_roi_exit"})

    def _update_immediate_entry(self, objects, now, enabled):
        objects = self._entry_objects(objects, now)
        self._entry_count = len(self.tracker._candidates(objects))
        result = self.tracker.update(objects, now, observations_enabled=enabled, roi_contour=self.contour)
        if self.tracker.points is not None:
            result = self._restore_load_baseline(result, now)
            self._arm(now)
        return result

    def _update_entry(self, objects, now, enabled):
        objects = self._entry_objects(objects,now)
        t = self.pending_tracker
        candidates = t._candidates(objects)
        self._entry_count = len(candidates)
        result = t.update(objects,now,observations_enabled=enabled,initialization_objects=objects,roi_contour=self.contour)
        # 停稳计时只确认几何身份；候选跟踪器独立缓存各槽空满和取放事件。
        self.state = "stabilizing_next" if t.initialization["stable_frames"] else "waiting_next"
        if t.points is not None:
            self.tracker = t
            result = self._restore_load_baseline(result, now)
            self.pending_tracker = None
            self._arm(now)
            self._group_events.append({"type": "group_started","group_id": t.group_id,"phase": t.phase,"at": now})
            return result
        return None

    def update(self, objects, now, *, observations_enabled=True):
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("分组时间必须严格递增")
        self.last_time = float(now)
        self._group_events = []
        self._deferred_new_events = []
        self._observations_enabled = observations_enabled
        self._hands = [o for o in objects if o["class_name"] == "hand"]
        result = None
        if self.state == "entering":
            result = self._update_immediate_entry(objects, now, observations_enabled)
        elif self.state in ("waiting_next","stabilizing_next"):
            result = self._update_entry(objects,now,observations_enabled)
        elif self.state == "sequence_complete":
            self._exclude_departing(objects, now)
        else:
            active = self._active_objects(self._exclude_departing(objects, now))
            initial = [o for o in active if o["class_name"] not in SLOT_STATES or self._distance(
                ((o["xyxy"][0]+o["xyxy"][2])/2,(o["xyxy"][1]+o["xyxy"][3])/2)) >= 0]
            result = self.tracker.update(active,now,observations_enabled=observations_enabled,
                initialization_objects=initial,roi_contour=self.contour,
                before_observe=self._observe_motion,
                translation_direction=self.direction if self.state == "exiting" else None)
            if self.tracker.points is not None and self.state == "initializing":
                self._arm(now)
            if self.state == "exiting" and (self.exit_min_slots == 1 or
                    self.exit_since is not None and now-self.exit_since+1e-9 >= self.exit_seconds):
                self._close(now)
                if self.exit_min_slots == 1 and self.state != "sequence_complete":
                    closing_events = result["new_events"]
                    self._start_immediate_next(now)
                    result = self._update_immediate_entry(objects, now, observations_enabled)
                    result["new_events"] = closing_events + result["new_events"]
        if result is None:
            result = {**self.tracker.snapshot(),"new_events": []}
        result["new_events"] = list(result.get("new_events", [])) + deepcopy(self._deferred_new_events)
        self.all_events.extend(deepcopy(result["new_events"]))
        if self.state in ("waiting_next","stabilizing_next","sequence_complete"):
            # 旧组已关闭；等待阶段不重复输出旧组的当前空/满状态。
            for slot in result["slots"]:
                slot.update(state="unknown",observation="unknown",reason=self.state,
                            position_is_prediction=slot["center_xy"] is not None)
            result["current_target_count"] = 0
            result["all_four_currently_confirmed"] = False
        return {**result,**self.metadata(),"new_group_events": deepcopy(self._group_events)}

    def finish(self):
        """结束退出核验；没有空到满的视觉证据就始终保留未完成。"""
        self._group_events, self._deferred_new_events = [], []
        if self.last_time is None:
            return {**self.snapshot(), "new_events": [], "new_group_events": []}
        now = self.last_time
        self._finish_departing_load(now)
        self.all_events.extend(deepcopy(self._deferred_new_events))
        return {**self.snapshot(), "new_events": deepcopy(self._deferred_new_events),
                "new_group_events": deepcopy(self._group_events)}

    def metadata(self):
        return {"group_assignment": "roi_exit_sequence" if self.enabled else "configured",
                "group_status": self.state,"pending_group_id": self.pending_tracker.group_id
                    if self.pending_tracker is not None else None,
                "pending_phase": self.pending_tracker.phase if self.pending_tracker is not None else None,
                "closed_group_count": len(self.closed_groups),
                "crossed_slot_ids": [i+1 for i in sorted(self.crossed)],
                "required_exit_slots": self._required_crossings,
                "motion_support_slot_ids": list(self._motion_support),
                "entry_candidate_count": self._entry_count,
                "pending_initialization": deepcopy(self.pending_tracker.initialization)
                    if self.pending_tracker is not None else None}

    def snapshot(self):
        return {**self.tracker.snapshot(),**self.metadata()}
