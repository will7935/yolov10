"""Record hand contact at two predicted slots for visual or formal slot verification."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from industrial_inspection.tools.s4_hand_contact import box_overlap_ratio
from industrial_inspection.tools.slot_group_tracker import slot_layout_error

LOAD_CONTACT_DOMINANCE_MARGIN = .10


def two_point_slot_layout(support, direction, multiplier, roi):
    """Assume two left-hand adjacent slots and project them along the upper ROI edge."""
    points = np.asarray([obj["center"] for obj in support], dtype=float)
    vector = points[1] - points[0]
    # Adjacent left-hand slots must be transverse to the extension.
    if abs(vector @ direction) > .65 * np.linalg.norm(vector):
        return None
    boxes = np.asarray([obj["xyxy"] for obj in support], dtype=float)
    half_sizes = (boxes[:, 2:] - boxes[:, :2]) / 2
    components = np.abs(direction) > 1e-9
    distances = np.min(half_sizes[:, components] / np.abs(direction[components]), axis=1)
    if distances.min() <= 0 or distances.max() > distances.min() * 1.5:
        return None
    extension = float(distances.mean() * multiplier)
    predicted = points + extension * direction
    layout = np.vstack((points, predicted))
    if (slot_layout_error(layout) is None
            or any(cv2.pointPolygonTest(roi, tuple(point), False) < 0 for point in predicted)
            or any(np.any(np.all((boxes[:, :2] <= point) & (point <= boxes[:, 2:]), axis=1))
                   for point in predicted)):
        return None
    return layout, extension


def _contact_areas(boxes, centers, hands):
    """Measure hand coverage after assigning the shared area to the closer center."""
    size = boxes[0, 2:]-boxes[0, :2]
    box_area = float(np.prod(size))
    low = np.maximum(boxes[0, :2], boxes[1, :2])
    high = np.minimum(boxes[0, 2:], boxes[1, 2:])
    if np.any(low >= high):
        return [max((box_overlap_ratio(box, hand["xyxy"]) for hand in hands), default=0.)
                for box in boxes]
    if np.prod(high-low)*2 >= box_area:
        return None
    shared = np.float32([low, [high[0], low[1]], high, [low[0], high[1]]])
    middle = centers.mean(axis=0)
    normal = centers[1]-centers[0]
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]]) * (np.linalg.norm(size)*2)
    scores = []
    for box, direction in zip(boxes, (normal, -normal)):
        offset = direction * np.linalg.norm(tangent)
        far_side = np.float32([middle-tangent, middle+tangent,
                               middle+tangent+offset, middle-tangent+offset])
        removed_area, removed = cv2.intersectConvexConvex(shared, far_side)
        score = 0.
        for hand in hands:
            x1, y1, x2, y2 = hand["xyxy"]
            covered = box_overlap_ratio(box, hand["xyxy"])*box_area
            if removed_area > 0:
                hand_polygon = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                shared_covered, _ = cv2.intersectConvexConvex(removed, hand_polygon)
                covered -= shared_covered
            score = max(score, covered/(box_area-removed_area))
        scores.append(score)
    return scores


@dataclass
class _Contact:
    since: float
    frames: int = 0
    ready: bool = False
    released_at: float | None = None
    track_id: int | None = None
    load_pose: _ContactPose | None = None


@dataclass(frozen=True)
class _Prediction:
    layout: np.ndarray
    boxes: np.ndarray
    half_size: np.ndarray
    spacing: float
    slot_ids: list | None
    measured_at: float


@dataclass(frozen=True)
class _ContactPose:
    layout: np.ndarray
    anchor_ids: tuple
    anchor_slots: tuple


class TwoPointHandContact:
    """Own temporary predictions separately from real slot identity and state."""

    def __init__(self, multiplier, upper_roi_edges, roi, *, hold_seconds=.8):
        if not np.isfinite(multiplier) or multiplier < 0:
            raise ValueError("TWO_POINT_EXTENSION_MULTIPLIER 必须是非负有限数")
        if not np.isfinite(hold_seconds) or hold_seconds < 0:
            raise ValueError("TWO_POINT_HAND_HOLD_SECONDS 必须是非负有限数")
        self.multiplier = float(multiplier)
        self.hold_seconds = float(hold_seconds)
        self.roi = roi
        self.direction = None
        if upper_roi_edges:
            start, end = max(upper_roi_edges, key=lambda edge: abs(edge[1][0]-edge[0][0]))
            edge = np.asarray(end-start, dtype=float)
            self.direction = edge * (np.sign(edge[0])/np.linalg.norm(edge))
        self.reset()

    def reset(self, now=None, *, keep_released_loads=False):
        self.last_update = now
        self.last_support = self.pair = self.anchor = self.since = None
        self.frames = 0
        self.sizes = deque(maxlen=5)
        self.pending = {}
        self.visual = {}
        self.diagnostics = []
        self.formal_slot_ids = None
        if not keep_released_loads:
            self.released_loads = []
        self._clear_prediction()

    def _clear_prediction(self):
        self.prediction = self.prediction_supported_at = self.coherent_shift = None

    def prepare(self, tracker, objects, now, enabled):
        if self.last_update == now:
            return
        self.visual.clear()
        self.diagnostics = []
        self.formal_slot_ids = None
        if self.direction is None or tracker.identity_lost:
            self.reset(now)
            return
        self.last_update = now
        self.released_loads = [(index, contact) for index, contact in self.released_loads
                               if now-contact.released_at <= tracker.initialization_history_seconds]
        self.pending = {index: item for index, item in self.pending.items()
                        if item.released_at is None
                        or now-item.released_at <= tracker.initialization_history_seconds}
        if not enabled:
            self._clear_prediction()
            self.pending = {i: item for i, item in self.pending.items() if item.ready}
            return
        hands = []
        for obj in objects:
            if obj["class_name"] == "hand":
                x1, y1, x2, y2 = obj["xyxy"]
                center = ((x1+x2)/2, (y1+y2)/2)
                if cv2.pointPolygonTest(self.roi, center, False) >= 0:
                    hands.append(obj)
        candidates = tracker._candidates([obj for obj in objects
                                          if obj.get("detection_source") != "local_crop"])
        candidates = [obj for obj in candidates
                      if cv2.pointPolygonTest(self.roi, tuple(obj["center"]), False) >= 0]
        usable = {obj["track_id"]: obj for obj in candidates if obj.get("track_id") is not None
                  and not obj["conflict"] and not tracker._occluded(obj, hands)}
        if len(candidates) > 2:
            self.pending = {i: item for i, item in self.pending.items() if item.ready}
            if not self.pending:
                self.reset(now, keep_released_loads=True)
                return
        if len(candidates) == len(usable) == 2 and (self.pair is None or not self.pending):
            pair = tuple(sorted(usable))
        else:
            pair = self.pair
        if pair != self.pair:
            self.reset(now, keep_released_loads=True)
            self.pair = pair
        self._check_previous_pose(tracker, usable, now)
        if (self.prediction is None and self.last_support is not None
                and now-self.last_support > tracker.lost_seconds):
            self.reset(now, keep_released_loads=True)
            self.pair = pair
        can_start = len(candidates) <= 2 and all(track_id in (pair or ()) for track_id in usable)
        if pair is None or len(candidates) > 4 or not all(track_id in usable for track_id in pair):
            self._use_previous_pose(tracker, hands, now, can_start=can_start)
            return
        support = [usable[track_id] for track_id in pair]
        projected = two_point_slot_layout(support, self.direction, self.multiplier, self.roi)
        if projected is None:
            self.anchor = self.since = None
            self.frames = 0
            self.sizes.clear()
            self._use_previous_pose(tracker, hands, now, can_start=can_start)
            return
        layout, extension = projected
        spacing = min(extension, float(np.linalg.norm(layout[0]-layout[1])))
        if (self.anchor is None or self.last_support is None
                or now-self.last_support > max(.25, tracker.geometry_initialization_seconds)
                or np.max(np.linalg.norm(layout-self.anchor, axis=1)) >
                   spacing*tracker.initialization_stability_ratio):
            self.anchor, self.since = layout, now
            self.frames = 0
            self.sizes.clear()
        self.last_support = now
        self.frames += 1
        sizes = np.asarray([obj["xyxy"] for obj in support])
        self.sizes.append(sizes[:, 2:]-sizes[:, :2])
        if (self.frames < tracker.initialization_frames
                or now-self.since+1e-9 < tracker.geometry_initialization_seconds):
            self._use_previous_pose(tracker, hands, now, can_start=can_start)
            return
        if tracker.points is not None:
            self.formal_slot_ids = self._formal_roles(tracker, support, layout)
            if self.formal_slot_ids is None:
                self.reset(now, keep_released_loads=True)
                return
            if (self.prediction is not None and self.prediction.slot_ids is not None
                    and self.prediction.slot_ids != self.formal_slot_ids):
                self.pending.clear()
        half_size = np.median(np.vstack(self.sizes), axis=0)/2
        boxes = np.hstack((layout[2:]-half_size, layout[2:]+half_size))
        prediction = _Prediction(layout, boxes, half_size, spacing, self.formal_slot_ids, now)
        other_slots = [obj for track_id, obj in usable.items() if track_id not in pair]
        self._evaluate(tracker, prediction, hands, now, can_start=len(candidates) == len(usable) == 2,
                       other_slots=other_slots)

    def _check_previous_pose(self, tracker, usable, now):
        prediction = self.prediction
        if prediction is None:
            return
        if not self.hold_seconds:
            self._clear_prediction()
            return
        if now-self.prediction_supported_at > self.hold_seconds+1e-9:
            self._clear_prediction()
            self.pending.clear()
            return

        # One reliable anchor at the old pose can keep that pose supported.
        # Reject it when both anchors agree on a shift for consecutive frames.
        tolerance = prediction.spacing*tracker.initialization_stability_ratio
        offsets = [np.subtract(usable[track_id]["center"], prediction.layout[index])
                   for index, track_id in enumerate(self.pair) if track_id in usable]
        if any(np.linalg.norm(offset) <= tolerance for offset in offsets):
            self.prediction_supported_at = now
        shift = None
        if len(offsets) == 2:
            mean = (offsets[0]+offsets[1])/2
            if (np.linalg.norm(mean) > tolerance
                    and np.linalg.norm(offsets[0]-offsets[1]) <= tolerance):
                shift = mean
        if shift is not None and self.coherent_shift is not None and shift @ self.coherent_shift > 0:
            self._clear_prediction()
            self.pending.clear()
        else:
            self.coherent_shift = shift

    def _use_previous_pose(self, tracker, hands, now, *, can_start):
        if self.prediction is None:
            self.pending = {i: item for i, item in self.pending.items() if item.ready}
            return
        self.formal_slot_ids = self.prediction.slot_ids
        self._evaluate(tracker, self.prediction, hands, now, can_start=can_start, held=True)

    def _evaluate(self, tracker, prediction, hands, now, *, can_start, other_slots=(), held=False):
        layout, boxes = prediction.layout, prediction.boxes
        scores = _contact_areas(boxes, layout[2:], hands)
        if scores is None:
            if not held:
                self._use_previous_pose(tracker, hands, now, can_start=can_start)
            return
        if not held:
            self.prediction = prediction
            self.prediction_supported_at = now
            self.coherent_shift = None
        threshold = tracker.s4_prediction_contact_overlap_ratio
        touched = [i for i, score in enumerate(scores) if score >= threshold]
        provisional_load = (tracker.phase == "load" and tracker.load_hand_contact_enabled
                            and (tracker.points is None or not tracker._region_formal_numbering))
        winner = touched[0] if len(touched) == 1 else None
        if provisional_load and len(touched) == 2 and abs(scores[0]-scores[1]) >= LOAD_CONTACT_DOMINANCE_MARGIN:
            winner = 0 if scores[0] > scores[1] else 1
        for index, (box, score) in enumerate(zip(boxes, scores)):
            contact = self.pending.get(index)
            if contact is None and winner == index and can_start:
                contact = self.pending[index] = _Contact(now)
            if contact is not None:
                if index in touched:
                    if winner == index and not contact.ready and can_start:
                        contact.frames += 1
                        contact.ready = (contact.frames >= tracker.load_hand_contact_min_frames
                            and now-contact.since+1e-9 >= tracker.load_hand_contact_seconds)
                        if contact.ready and provisional_load:
                            contact.load_pose = self._load_contact_pose(tracker, prediction)
                elif contact.ready:
                    if contact.released_at is None:
                        contact.released_at = now
                        if provisional_load and contact.load_pose is not None:
                            self.released_loads.append((index, contact))
                    candidate = self._unique_visual(layout, prediction.half_size, index, other_slots)
                    if candidate is not None and contact.track_id in (None, candidate):
                        contact.track_id = candidate
                        self.visual[candidate] = index
                if not contact.ready and (winner != index or not can_start):
                    self.pending.pop(index)
                    contact = None
            self.diagnostics.append({"center_xy": layout[index+2].tolist(),
                "box_xyxy": box.tolist(), "coverage": score, "anchor_track_ids": list(self.pair),
                "held": held, "prediction_age_seconds": now-prediction.measured_at,
                "prediction_support_age_seconds": now-self.prediction_supported_at,
                "slot_id": None if self.formal_slot_ids is None else self.formal_slot_ids[index],
                "status": ("ambiguous" if index in touched and winner is None else
                           "prediction" if contact is None else
                           "waiting_visual" if contact.released_at is not None else
                           "contact" if contact.ready else "collecting"),
                "contact_since": None if contact is None else contact.since,
                "released_at": None if contact is None else contact.released_at,
                "track_id": None if contact is None else contact.track_id})

    def _load_contact_pose(self, tracker, prediction):
        provisional = tracker._provisional or tracker
        track_ids = (provisional._entry_track_ids if provisional._entry_points is not None
                     else provisional.byte_track_ids)
        anchors = dict(zip(track_ids, provisional.slots))
        return _ContactPose(prediction.layout.copy(), self.pair, tuple(anchors.get(i) for i in self.pair))

    def take_s4_load(self, tracker, empty_at, now):
        """Consume the released contact only when the final layout assigns it uniquely to S4."""
        candidates, self.released_loads = self.released_loads, []
        matched = [contact for index, contact in candidates if empty_at <= contact.since
                   and now-contact.released_at <= tracker.initialization_history_seconds
                   and self._matches_formal_s4(tracker, index, contact.load_pose)]
        return matched[0] if len(matched) == 1 else None

    def _matches_formal_s4(self, tracker, index, pose):
        anchors = []
        for old_id, old_slot in zip(pose.anchor_ids, pose.anchor_slots):
            matches = [i for i, (track_id, slot) in enumerate(zip(tracker.byte_track_ids, tracker.slots))
                       if track_id is not None and (track_id == old_id or slot is old_slot)]
            if len(matches) != 1:
                return False
            anchors.append(matches[0])
        if (len(set(anchors)) != 2 or 3 in anchors
                or not any(tracker.byte_track_ids[i] == old_id for i, old_id in zip(anchors, pose.anchor_ids))):
            return False
        points = tracker.reference_points
        matrix, _ = cv2.estimateAffinePartial2D(pose.layout[:2], points[anchors])
        if matrix is None or not .8 <= np.linalg.norm(matrix[:, 0]) <= 1.2:
            return False
        predicted = pose.layout[2:] @ matrix[:, :2].T + matrix[:, 2]
        distances = np.linalg.norm(predicted[:, None]-points, axis=2)
        nearest = np.argsort(distances, axis=1)
        slots = nearest[:, 0]
        if set(slots) != set(range(4))-set(anchors) or slots[index] != 3:
            return False
        separation = np.linalg.norm(points[slots[0]]-points[slots[1]])
        # Allow a biased extension length when both predictions preserve their distinct roles.
        for row, (best, second) in enumerate(nearest[:, :2]):
            if (distances[row, best] > .85*separation
                    or distances[row, second]-distances[row, best] < .15*separation):
                return False
        return True

    def _formal_roles(self, tracker, support, layout):
        gate = tracker._point_tracking_gate or tracker._spacing()*tracker.match_distance_ratio
        matches = tracker._associate(support, gate)
        if len(matches) != 2:
            return None
        missing = sorted(set(range(4))-matches.keys())
        points = tracker.reference_points[missing]
        normal = np.array([layout[0, 1]-layout[1, 1], layout[1, 0]-layout[0, 0]])
        normal *= np.sign(normal @ self.direction)
        if np.any((points-layout[0]) @ normal <= 0):
            return None
        right_side = np.linalg.norm(points[:, None]-layout[None, 2:], axis=2)
        indices = np.argmin(right_side, axis=0)
        if indices[0] == indices[1]:
            return None
        return [missing[i]+1 for i in indices]

    def _unique_visual(self, layout, half_size, index, candidates):
        # The candidate must fit a single predicted role, the measured pair,
        # and the same plausible four-slot geometry.
        gate = min(np.linalg.norm(layout[2]-layout[3])*.5, np.linalg.norm(half_size))
        matches = []
        for obj in candidates:
            center = np.asarray(obj["center"], dtype=float)
            distances = np.linalg.norm(layout-center, axis=1)
            distance = distances[index+2]
            if distance > gate or np.any(np.delete(distances, index+2) <= distance+1e-6):
                continue
            trial = layout.copy()
            trial[index+2] = center
            if slot_layout_error(trial) is not None:
                matches.append(obj["track_id"])
        return matches[0] if len(matches) == 1 else None

    def evidence(self, obj, slot_id, now):
        if now != self.last_update or obj is None or obj.get("detection_source") == "local_crop":
            return None
        index = self.visual.get(obj.get("track_id"))
        contact = self.pending.get(index)
        if contact is None or (self.formal_slot_ids is not None and self.formal_slot_ids[index] != slot_id):
            return None
        return {"since": contact.since, "released_at": contact.released_at, "count": contact.frames}

    def resolve(self, obj, state, event):
        index = self.visual.pop(obj["track_id"])
        contact = self.pending.pop(index)
        self.released_loads = [(index, released) for index, released in self.released_loads
                               if released is not contact]
        self.diagnostics[index]["status"] = "visual_confirmed" if event else "visual_observed"
        self.diagnostics[index]["visual_state"] = state
