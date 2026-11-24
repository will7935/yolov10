"""Record hand contact at two predicted slots for visual or formal slot verification."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import cv2
import numpy as np

from industrial_inspection.tools.s4_hand_contact import box_overlap_ratio
from industrial_inspection.tools.slot_group_tracker import slot_layout_error

LOAD_CONTACT_DOMINANCE_MARGIN = .10
LOAD_CLOCKWISE_ANCHORS = frozenset((0, 2))  # S1 and S3, in either detection order.


def two_point_slot_layout(support, direction, multiplier, roi, *, require_transverse=True):
    """Assume two left-hand adjacent slots and project them along the upper ROI edge."""
    points = np.asarray([obj["center"] for obj in support], dtype=float)
    vector = points[1] - points[0]
    # Adjacent left-hand slots must be transverse to the extension.
    if require_transverse and abs(vector @ direction) > .65 * np.linalg.norm(vector):
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


def _clockwise_roles(tracker, layout, first_slot):
    """Map the projected quadrilateral in image order, starting at its first observed slot."""
    ring = tracker._slot_ring(layout)
    start = tracker.clockwise.index(first_slot+1)
    order = tracker.clockwise[start:] + tracker.clockwise[:start]
    return tuple(order[ring.index(index)]-1 for index in range(4))


def _load_preview_roles(tracker, layout, direction):
    """Identify the support pair from the unload layout without matching projected points."""
    reference = np.asarray(tracker._load_numbering_reference, dtype=float)
    if slot_layout_error(reference) is None:
        return None
    measured = layout[:2].astype(np.float32)
    edge = measured[1]-measured[0]
    normal = np.array([-edge[1], edge[0]])
    normal *= np.sign(normal @ direction)
    result = None
    for first_slot in range(4):
        roles = _clockwise_roles(tracker, layout, first_slot)
        matrix, _ = cv2.estimateAffinePartial2D(reference[list(roles[:2])].astype(np.float32), measured)
        if matrix is None:
            continue
        scale = np.linalg.norm(matrix[:, 0])
        angle = abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        if not .5 <= scale <= 2 or angle > 45:
            continue
        expected = reference[list(roles[2:])] @ matrix[:, :2].T + matrix[:, 2]
        if np.any((expected-measured[0]) @ normal <= 0):
            continue
        if result is not None:
            return None
        result = roles
    return result if result is not None and set(result[:2]) == LOAD_CLOCKWISE_ANCHORS else None


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
    requires_visual_load: bool = False


@dataclass(frozen=True)
class _Prediction:
    layout: np.ndarray
    boxes: np.ndarray
    half_size: np.ndarray
    spacing: float
    slot_ids: list | None
    measured_at: float
    reference_roles: tuple | None = None


@dataclass(frozen=True)
class _ContactPose:
    layout: np.ndarray
    anchor_ids: tuple
    anchor_slots: tuple
    reference_roles: tuple | None = None


class TwoPointHandContact:
    """Own temporary predictions separately from real slot identity and state."""

    def __init__(self, multiplier, upper_roi_edges, roi, *, hold_seconds=.8, upper_slot_three=False,
                 orbit_degrees=None):
        if not np.isfinite(multiplier) or multiplier < 0:
            raise ValueError("TWO_POINT_EXTENSION_MULTIPLIER 必须是非负有限数")
        if not np.isfinite(hold_seconds) or hold_seconds < 0:
            raise ValueError("TWO_POINT_HAND_HOLD_SECONDS 必须是非负有限数")
        self.multiplier = float(multiplier)
        self.hold_seconds = float(hold_seconds)
        self.roi = roi
        self.upper_slot_three = bool(upper_slot_three)
        if orbit_degrees is not None and (not np.isfinite(orbit_degrees) or not 0 < abs(orbit_degrees) <= 360):
            raise ValueError("Orbit angle must be finite and within (0, 360] degrees")
        self.orbit_degrees = orbit_degrees
        self.direction = None
        if upper_roi_edges:
            start, end = max(upper_roi_edges, key=lambda edge: abs(edge[1][0]-edge[0][0]))
            edge = np.asarray(end-start, dtype=float)
            self.direction = edge * (np.sign(edge[0])/np.linalg.norm(edge))
        self.reset()

    def reset(self, now=None, *, keep_released_loads=False):
        self.last_update = now
        self.last_support = self.pair = self.anchor = self.since = None
        self.reference_roles = None
        self.frames = 0
        self.sizes = deque(maxlen=5)
        self.pending = {}
        self.visual = {}
        self.diagnostics = []
        self.formal_slot_ids = None
        self.numbering_order = None
        self.numbering_diagnostic = None
        self._numbering_key = self._numbering_anchor = self._numbering_since = None
        self._numbering_frames = 0
        self.orbit_roles = None
        self.accepted_slot_ids = set()
        if not keep_released_loads:
            self.released_loads = []
        self._clear_prediction()

    def _clear_prediction(self):
        self.prediction = self.prediction_supported_at = self.coherent_shift = None

    def uses_orbit(self, tracker):
        # Later groups use the established G1 loading template: their unloading
        # samples need not have the same stopped pose as the first group.
        if self.orbit_degrees is None or tracker.phase != "load" or tracker.group_id != 1:
            return False
        if tracker.points is not None and tracker._region_formal_numbering:
            return tracker.initialization.get("two_point_numbering", {}).get("reference") == "unload_orbit_rotation"
        return True

    def prepare(self, tracker, objects, now, enabled):
        if self.last_update == now:
            return
        self.visual.clear()
        self.diagnostics = []
        self.formal_slot_ids = None
        self.numbering_order = None
        self.accepted_slot_ids.clear()
        orbit = self.uses_orbit(tracker)
        formal = tracker.points is not None and tracker._region_formal_numbering
        use_reference = tracker.phase == "load" and (tracker._load_numbering_reference is not None
                                                     or tracker._load_perspective_numbering is not None)
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
            self._numbering_key = self._numbering_anchor = self._numbering_since = None
            self._numbering_frames = 0
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
        if len(candidates) > 2 and not (orbit and formal):
            self.pending = {i: item for i, item in self.pending.items() if item.ready}
            if not self.pending:
                self.reset(now, keep_released_loads=True)
                return
        bound_pair = (tracker.byte_track_ids[0], tracker.byte_track_ids[2]) if orbit and formal else ()
        if bound_pair and all(track_id is not None and track_id in usable for track_id in bound_pair):
            pair = tuple(sorted(bound_pair))
        elif len(candidates) == len(usable) == 2 and (self.pair is None or not self.pending):
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
        can_start = (len(candidates) <= 2 and all(track_id in (pair or ()) for track_id in usable)) \
                    or (orbit and formal and len(candidates) <= 4 and all(tid in usable for tid in pair or ()))
        if pair is None or len(candidates) > 4 or not all(track_id in usable for track_id in pair):
            self._numbering_key = self._numbering_anchor = self._numbering_since = None
            self._numbering_frames = 0
            self._use_previous_pose(tracker, hands, now, can_start=can_start)
            return
        support = [usable[track_id] for track_id in pair]
        projected = (self._orbit_projection(tracker, support, formal=formal) if orbit else
                     two_point_slot_layout(support, self.direction, self.multiplier, self.roi,
                                           require_transverse=not use_reference))
        if projected is None:
            self._numbering_key = self._numbering_anchor = self._numbering_since = None
            self._numbering_frames = 0
            self.anchor = self.since = None
            self.frames = 0
            self.sizes.clear()
            self._use_previous_pose(tracker, hands, now, can_start=can_start)
            return
        layout, extension = projected
        spacing = min(extension, float(np.linalg.norm(layout[0]-layout[1])))
        if use_reference:
            roles = self.orbit_roles if orbit else self._load_reference_roles(tracker, support, layout, formal=formal)
            if formal and roles is None:
                self.reset(now, keep_released_loads=True)
                self.pair = pair
                return
            if self.reference_roles is not None and roles is not None and self.reference_roles != roles:
                diagnostic = self.numbering_diagnostic
                self.reset(now, keep_released_loads=True)
                self.pair = pair
                self.numbering_diagnostic = diagnostic
            self.reference_roles = roles
        if tracker._load_perspective_numbering is not None and not formal:
            self._update_numbering_stability(tracker, layout, now, can_start)
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
            if self.prediction is None and tracker._load_perspective_numbering is not None:
                self._show_live_preview(layout, support, hands, now)
            return
        if tracker.points is not None and (not use_reference or formal):
            self.formal_slot_ids = ([slot_index+1 for slot_index in self.reference_roles[2:]]
                                    if use_reference else self._formal_roles(tracker, support, layout))
            if self.formal_slot_ids is None:
                self.reset(now, keep_released_loads=True)
                return
            if (self.prediction is not None and self.prediction.slot_ids is not None
                    and self.prediction.slot_ids != self.formal_slot_ids):
                self.pending.clear()
        half_size = np.median(np.vstack(self.sizes), axis=0)/2
        boxes = np.hstack((layout[2:]-half_size, layout[2:]+half_size))
        prediction = _Prediction(layout, boxes, half_size, spacing, self.formal_slot_ids, now, self.reference_roles)
        other_slots = [obj for track_id, obj in usable.items() if track_id not in pair]
        self._evaluate(tracker, prediction, hands, now, can_start=can_start,
                       other_slots=other_slots)

    def _orbit_projection(self, tracker, support, *, formal):
        self.orbit_roles = None
        info = {"reference": "unload_orbit_rotation", "clockwise_degrees": self.orbit_degrees,
                "measured_support": 2, "status": "waiting_for_unload_orbit_reference"}
        self.numbering_diagnostic = info
        if (tracker._load_perspective_numbering is None or tracker._orbit_unload_reference is None
                or tracker.machine_center is None):
            return None
        predicted = tracker._load_perspective_numbering.rotate_about_machine(
            tracker._orbit_unload_reference, tracker.machine_center, self.orbit_degrees)
        observed = np.asarray([obj['center'] for obj in support])
        if formal:
            if not all(obj['track_id'] in tracker.byte_track_ids for obj in support):
                return None
            anchors = tuple(tracker.byte_track_ids.index(obj['track_id']) for obj in support)
        else:
            upper = int(np.argmin(observed[:, 1]))
            anchors = (2, 0) if upper == 0 else (0, 2)
        spacing = np.min(np.linalg.norm(predicted[:, None]-predicted[None, :], axis=2)[np.triu_indices(4, 1)])
        gate = min(40., spacing*tracker.match_distance_ratio)
        errors = np.linalg.norm(observed-predicted[list(anchors)], axis=1)
        # The explicit S3/S1 rule supplies identity even when the calibrated
        # projection is biased. Once numbered, stop contact inference if the
        # real anchors move away from that initial loading pose.
        explicit = self.upper_slot_three and tracker.group_id == 1
        pose_errors = (np.linalg.norm(observed-tracker.reference_points[list(anchors)], axis=1)
                       if formal else errors)
        if set(anchors) != {0, 2} or (pose_errors.max() > gate and (formal or not explicit)):
            self.numbering_diagnostic = dict(info, status="orbit_anchor_mismatch",
                anchor_errors_px=errors.tolist(), pose_errors_px=pose_errors.tolist(),
                maximum_anchor_error_px=float(gate))
            return None
        missing = sorted(set(range(4))-set(anchors), key=lambda i: predicted[i, 1])
        roles = (*anchors, *missing)
        layout = np.vstack((observed, predicted[missing]))
        if (slot_layout_error(layout) is None
                or any(cv2.pointPolygonTest(self.roi, tuple(point), False) < 0 for point in layout[2:])
                or _clockwise_roles(tracker, layout, anchors[0]) != roles):
            self.numbering_diagnostic = dict(info, status="orbit_geometry_mismatch")
            return None
        self.orbit_roles = roles
        self.numbering_diagnostic = dict(info, status="matched", candidate_count=1,
            anchor_errors_px=errors.tolist(), pose_errors_px=pose_errors.tolist(),
            maximum_pose_error_px=float(gate), pose_error_gate_applied=formal or not explicit,
            anchor_identity="configured_upper_s3_lower_s1" if explicit else "orbit_reference",
            reference_centers_xy=np.asarray(tracker._orbit_unload_reference).tolist(),
            machine_center_xy=np.asarray(tracker.machine_center).tolist(),
            slot_ids_in_preview_order=[i+1 for i in roles], predicted_centers_xy=predicted.tolist())
        return layout, float(spacing)

    def orbit_contact(self, slot_id, now):
        if self.orbit_degrees is None or now != self.last_update:
            return None
        return next((item for item in self.diagnostics if item.get('slot_id') == slot_id), None)

    def _load_reference_roles(self, tracker, support, layout, *, formal):
        if not formal:
            if self.upper_slot_three and tracker.phase == "load" and tracker.group_id == 1:
                upper = int(np.argmin(layout[:2, 1]))
                lower = 1-upper
                roles = _clockwise_roles(tracker, layout, 2 if upper == 0 else 0)
                valid = (abs(layout[0, 1]-layout[1, 1]) >= 2.
                         and roles[upper] == 2 and roles[lower] == 0
                         and set(roles[2:]) == {1, 3})
                self.numbering_diagnostic = {
                    "status": "matched" if valid else "explicit_anchor_geometry_mismatch",
                    "reference": "configured_upper_s3_lower_s1", "measured_support": 2,
                    "candidate_count": int(valid), "upper_anchor_track_id": support[upper]["track_id"],
                    "lower_anchor_track_id": support[lower]["track_id"],
                    "slot_ids_in_preview_order": [i+1 for i in roles] if valid else None}
                return roles if valid else None
            if tracker._load_perspective_numbering is not None:
                subsequent = tracker.group_id != 1
                reference = (tracker._g1_load_numbering_reference if subsequent
                             else tracker._load_numbering_reference)
                roles, diagnostic = tracker._load_perspective_numbering.match_pair(
                    reference, layout, self.direction, self.roi, same_load_pose=subsequent)
                self.numbering_diagnostic = diagnostic
                return roles
            return _load_preview_roles(tracker, layout, self.direction)
        track_ids = [obj["track_id"] for obj in support]
        if not all(track_id in tracker.byte_track_ids for track_id in track_ids):
            return None
        anchors = tuple(tracker.byte_track_ids.index(track_id) for track_id in track_ids)
        if tracker._load_perspective_numbering is None and set(anchors) != LOAD_CLOCKWISE_ANCHORS:
            return None
        roles = _clockwise_roles(tracker, layout, anchors[0])
        return roles if roles[:2] == anchors else None

    def _update_numbering_stability(self, tracker, layout, now, enabled):
        roles = self.reference_roles
        info = dict(self.numbering_diagnostic or {"status": "waiting_for_reference"})
        if roles is None or not enabled:
            self._numbering_key = self._numbering_anchor = self._numbering_since = None
            self._numbering_frames = 0
        else:
            key = (self.pair, roles)
            spacing = np.min(np.linalg.norm(layout[:, None]-layout[None, :], axis=2)[np.triu_indices(4, 1)])
            if (key != self._numbering_key or self._numbering_anchor is None
                    or self.last_support is None
                    or now-self.last_support > max(.25, tracker.geometry_initialization_seconds)
                    or np.max(np.linalg.norm(layout-self._numbering_anchor, axis=1))
                       > spacing*tracker.initialization_stability_ratio):
                self._numbering_key, self._numbering_anchor = key, layout.copy()
                self._numbering_since, self._numbering_frames = now, 0
            self._numbering_frames += 1
            if (self._numbering_frames >= max(3, tracker.minimum_frames, tracker.initialization_frames)
                    and now-self._numbering_since+1e-9 >= tracker.geometry_initialization_seconds):
                self.numbering_order = roles
        info.update(stable_frames=self._numbering_frames, ready=self.numbering_order is not None)
        self.numbering_diagnostic = info
        tracker.initialization["two_point_numbering"] = info

    def _show_live_preview(self, layout, support, hands, now):
        """Display the current two-point geometry while contact evidence warms up."""
        size = np.median([np.subtract(obj['xyxy'][2:], obj['xyxy'][:2]) for obj in support], axis=0)/2
        boxes = np.hstack((layout[2:]-size, layout[2:]+size))
        scores = _contact_areas(boxes, layout[2:], hands) or [0., 0.]
        self.diagnostics = [{"center_xy": center.tolist(), "box_xyxy": box.tolist(),
            "coverage": score, "anchor_track_ids": list(self.pair), "slot_id": None,
            "reference_slot_id": None if self.reference_roles is None else self.reference_roles[i+2]+1,
            "reference_anchor_slot_ids": (None if self.reference_roles is None
                                          else [j+1 for j in self.reference_roles[:2]]),
            "held": False, "prediction_age_seconds": 0., "prediction_support_age_seconds": 0.,
            "status": "prediction", "contact_enabled": False, "contact_since": None,
            "released_at": None, "track_id": None}
            for i, (center, box, score) in enumerate(zip(layout[2:], boxes, scores))]

    def confirm_numbering(self):
        """Publish the verified predicted roles together with the group's formal numbering."""
        if self.numbering_order is None or self.prediction is None:
            return
        self.formal_slot_ids = [i+1 for i in self.numbering_order[2:]]
        self.prediction = replace(self.prediction, slot_ids=self.formal_slot_ids)
        for entry, slot_id in zip(self.diagnostics, self.formal_slot_ids):
            entry['slot_id'] = slot_id

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
        prediction = self.prediction
        if prediction.slot_ids is None and prediction.reference_roles != self.reference_roles:
            prediction = replace(prediction, reference_roles=self.reference_roles)
        self.formal_slot_ids = prediction.slot_ids
        self._evaluate(tracker, prediction, hands, now, can_start=can_start, held=True)

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
        ring_load = tracker.phase == "load" and (tracker._load_numbering_reference is not None
                                                 or tracker._load_perspective_numbering is not None)
        winner = touched[0] if len(touched) == 1 else None
        if provisional_load and len(touched) == 2 and abs(scores[0]-scores[1]) >= LOAD_CONTACT_DOMINANCE_MARGIN:
            winner = 0 if scores[0] > scores[1] else 1
        coverage_winner = winner
        centers_covered = [False, False]
        if tracker.phase == "load" and tracker.load_hand_contact_enabled:
            centers_covered = [any(np.all((center >= hand["xyxy"][:2]) & (center <= hand["xyxy"][2:]))
                                   for hand in hands) for center in layout[2:]]
            if all(centers_covered):
                winner = None
                touched = [0, 1]
            elif len(touched) == 2 and any(centers_covered):
                winner = centers_covered.index(True)
        ambiguous_load = tracker.phase == "load" and len(touched) == 2 and winner is None
        center_breaks_tie = winner is not None and winner != coverage_winner
        if tracker.phase == "load" and not provisional_load:
            targets = tracker.inferred_load_contact_indices()
            eligible = [i for i, slot_id in enumerate(prediction.slot_ids or ()) if slot_id-1 in targets]
            winner = winner if winner in eligible else None
            touched = [i for i in touched if i in eligible]
            self.pending = {i: item for i, item in self.pending.items() if i in eligible}
        if self.uses_orbit(tracker) and can_start and winner is not None and prediction.slot_ids is not None:
            self.accepted_slot_ids.add(prediction.slot_ids[winner])
        for index, (box, score) in enumerate(zip(boxes, scores)):
            contact = self.pending.get(index)
            if contact is None and winner == index and can_start:
                contact = self.pending[index] = _Contact(now, requires_visual_load=ring_load)
            if contact is not None:
                if index in touched:
                    if contact.released_at is None:
                        contact.requires_visual_load |= (ambiguous_load or (winner == index and center_breaks_tie))
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
                    candidate = (self._unique_visual(layout, prediction.half_size, index, other_slots)
                                 if not (ring_load and provisional_load) else None)
                    if candidate is not None and contact.track_id in (None, candidate):
                        contact.track_id = candidate
                        self.visual[candidate] = index
                if not contact.ready and (winner != index or not can_start):
                    self.pending.pop(index)
                    contact = None
            self.diagnostics.append({"center_xy": layout[index+2].tolist(),
                "box_xyxy": box.tolist(), "coverage": score, "anchor_track_ids": list(self.pair),
                "center_covered": centers_covered[index],
                "reference_anchor_slot_ids": (None if prediction.reference_roles is None
                                              else [i+1 for i in prediction.reference_roles[:2]]),
                "reference_slot_id": (None if prediction.reference_roles is None
                                      else prediction.reference_roles[index+2]+1),
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
        return _ContactPose(prediction.layout.copy(), self.pair, tuple(anchors.get(i) for i in self.pair),
                            prediction.reference_roles)

    def take_inferred_load(self, tracker, target, empty_at, now):
        """Consume a released contact only when it matches the final inferred slot."""
        candidates = self.released_loads
        matched = [contact for index, contact in candidates if empty_at <= contact.since
                   and now-contact.released_at <= tracker.initialization_history_seconds
                   and (target == 3 or contact.requires_visual_load)
                   and self._matches_formal_load(tracker, index, contact.load_pose, target)]
        if len(matched) != 1:
            return None
        self.released_loads = [(index, contact) for index, contact in candidates if contact is not matched[0]]
        return matched[0]

    def _matches_formal_load(self, tracker, index, pose, target):
        anchors = []
        for old_id, old_slot in zip(pose.anchor_ids, pose.anchor_slots):
            matches = [i for i, (track_id, slot) in enumerate(zip(tracker.byte_track_ids, tracker.slots))
                       if track_id is not None and (track_id == old_id or slot is old_slot)]
            if len(matches) != 1:
                return False
            anchors.append(matches[0])
        if (len(set(anchors)) != 2 or target in anchors
                or not any(tracker.byte_track_ids[i] == old_id for i, old_id in zip(anchors, pose.anchor_ids))):
            return False
        if (pose.reference_roles is not None and (tuple(anchors) != pose.reference_roles[:2]
                                                  or pose.reference_roles[index+2] != target)):
            return False
        points = tracker.reference_points
        matrix, _ = cv2.estimateAffinePartial2D(pose.layout[:2], points[anchors])
        if matrix is None or not .8 <= np.linalg.norm(matrix[:, 0]) <= 1.2:
            return False
        predicted = pose.layout[2:] @ matrix[:, :2].T + matrix[:, 2]
        if tracker._load_numbering_reference is not None:
            roles = _clockwise_roles(tracker, pose.layout, anchors[0])
            if ((tracker._load_perspective_numbering is None and set(anchors) != LOAD_CLOCKWISE_ANCHORS)
                    or roles[:2] != tuple(anchors) or roles[index+2] != target
                    or _clockwise_roles(tracker, points, 0) != tuple(range(4))):
                return False
            separation = np.linalg.norm(points[roles[2]]-points[roles[3]])
            return np.linalg.norm(predicted[index]-points[target]) <= .85*separation
        distances = np.linalg.norm(predicted[:, None]-points, axis=2)
        nearest = np.argsort(distances, axis=1)
        slots = nearest[:, 0]
        if set(slots) != set(range(4))-set(anchors) or slots[index] != target:
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
        return {"since": contact.since, "released_at": contact.released_at, "count": contact.frames,
                "requires_visual_load": contact.requires_visual_load}

    def resolve(self, obj, state, event):
        index = self.visual.pop(obj["track_id"])
        contact = self.pending.pop(index)
        self.released_loads = [(index, released) for index, released in self.released_loads
                               if released is not contact]
        self.diagnostics[index]["status"] = "visual_confirmed" if event else "visual_observed"
        self.diagnostics[index]["visual_state"] = state
