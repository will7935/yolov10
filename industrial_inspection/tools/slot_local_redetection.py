"""局部补检策略：只返回补充观测，身份关联、状态确认和工艺判断由调用方管理。"""
from dataclasses import dataclass
from itertools import permutations

import cv2
import numpy as np

from industrial_inspection.tools.slot_group_tracker import (
    SLOT_STATES, box_containment, box_iou, slot_layout_error, valid_slot_geometry)


@dataclass(frozen=True)
class LocalRedetectionPolicy:
    enabled: bool
    crop_size: tuple
    image_size: int


def recover_missing_slots(model, frame, objects, sequence, args, thresholds, policy):
    """仅为下料操作中已正式编号的漏检 S4 补充观测。"""
    tracker = sequence.tracker
    if (not policy.enabled or tracker.phase != "unload" or sequence.state != "active" or tracker.points is None
            or tracker.identity_lost):
        return objects, []
    gate = tracker._point_tracking_gate or tracker._spacing()*tracker.match_distance_ratio
    candidates = tracker._candidates(objects)
    matches = tracker._associate(candidates, gate)
    points = tracker.reference_points
    s4_index = 3
    center = points[s4_index]
    if s4_index in matches or sequence._distance(center) < sequence.margin:
        return objects, []

    bounds = _crop_bounds(center, policy.crop_size, frame.shape[:2])
    accepted, hands = _detect_in_crop(model, frame, bounds, points, gate, sequence, args, thresholds, policy)
    attempts = [{"group_id": tracker.group_id, "slot_id": 4,
                 "crop_xyxy": list(bounds), "accepted_detections": accepted}]
    support = {i: candidates[col]["center"] for i, col in matches.items()
               if not candidates[col]["conflict"]}
    if not accepted and len(support) >= 2:
        # Use a supported alternate center without widening the association gate.
        fallback_points = points.copy()
        for i, point in support.items():
            fallback_points[i] = point
        alternate = tracker._s4_before_recalculation
        if (alternate is not None and np.linalg.norm(alternate-center) <= gate
                and sequence._distance(alternate) >= 0):
            fallback_points[3] = alternate
        fallback_bounds = _crop_bounds(fallback_points[3], policy.crop_size, frame.shape[:2])
        if fallback_bounds == bounds:
            fallback_bounds = _crop_bounds(center, policy.crop_size, frame.shape[:2], keep_size=True)
        if fallback_bounds != bounds:
            recovered, extra_hands = _detect_in_crop(model, frame, fallback_bounds, fallback_points,
                gate, sequence, args, thresholds, policy)
            recovered = tracker._candidates([obj for obj in recovered if not any(
                box_containment(obj["xyxy"], candidate["xyxy"]) >= .9 for candidate in candidates)])
            if len(recovered) == 1 and not recovered[0]["conflict"]:
                revised = fallback_points.copy()
                revised[3] = recovered[0]["center"]
                if (valid_slot_geometry(fallback_points) and valid_slot_geometry(revised)
                        and _slot_order(revised) == _slot_order(fallback_points)):
                    accepted = [dict(recovered[0], local_s4_fallback=True)]
                    hands.extend(extra_hands)
            attempts.append({"group_id": tracker.group_id, "slot_id": 4, "fallback": True,
                             "crop_xyxy": list(fallback_bounds), "accepted_detections": accepted})

    unique_hands = {tuple(round(float(v), 1) for v in hand["xyxy"]): hand
                    for hand in hands} if accepted else {}
    return objects+accepted+list(unique_hands.values()), attempts


def recover_load_reach_hand(model, frame, detections, sequence, args, thresholds, policy):
    """Recheck a far-side hand candidate without admitting it to the main detector."""
    tracker = sequence.pending_tracker or sequence.tracker
    if (tracker.phase != "load" or tracker.points is not None or tracker.identity_lost
            or tracker._load_numbering_reference is None
            or sequence.state not in ("entering", "waiting_next", "stabilizing_next")):
        return None
    left, right = sequence.contour[:, 0].min(), sequence.contour[:, 0].max()
    boundary = right-.125*(right-left)
    candidates = []
    for obj in detections:
        if obj["class_name"] != "hand" or obj["confidence"] < .2*thresholds.hand:
            continue
        x1, y1, x2, y2 = obj["xyxy"]
        if x2 >= boundary and sequence._distance((x2, (y1+y2)/2)) >= 0:
            candidates.append(obj)
    if not candidates:
        return None
    hand = max(candidates, key=lambda obj: obj["confidence"])
    _, y1, x2, y2 = hand["xyxy"]
    width, height = policy.crop_size
    bounds = _crop_bounds((x2, (y1+y2)/2), (round(width*1.2), round(height*8/7)),
                         frame.shape[:2], keep_size=True)
    matches = []
    for obj in _crop_detections(model, frame, bounds, args, thresholds, policy):
        if obj["class_name"] != "hand" or obj["xyxy"][2] < boundary:
            continue
        overlap = box_iou(obj["xyxy"], hand["xyxy"])
        if overlap >= .5:
            matches.append((overlap, obj))
    return max(matches, key=lambda match: match[0])[1] if matches else None


def recover_load_preview_support(model, frame, tracker, anchors, hands, candidates,
                                 sequence, args, thresholds, policy):
    """Find one additional measured center for a pre-numbering load preview."""
    if not policy.enabled or tracker.phase != "load" or tracker.points is not None or tracker.identity_lost:
        return None, None
    ids = [anchor.get("track_id") for anchor in anchors]
    if None in ids or len(set(ids)) != 2:
        return None, None
    # A pair from the entry filter must have already been seen in this group.
    previous_ids = tracker._entry_track_ids
    if tracker._provisional is not None:
        previous_ids = tracker._provisional.byte_track_ids
    if not set(ids).issubset(previous_ids):
        if tracker._entry_points is not None or tracker._provisional is not None or tracker.track_store is None:
            return None, None
        records = [tracker.track_store.tracks.get(track_id) for track_id in ids]
        if any(record is None or record["reliable_count"] < 2
               or record["owner"] not in (None, (tracker.phase, tracker.group_id)) for record in records):
            return None, None
    measured = np.float32([obj["center"] for obj in anchors])
    recent = _recent_load_support_target(tracker, ids, measured, candidates)
    if recent is None:
        reference = tracker._load_numbering_reference
        if reference is None or slot_layout_error(reference) is None:
            return None, None
        targets = _load_reference_targets(reference, measured, sequence.entry_match_distance_px)
    else:
        point, gate, _ = recent
        targets = [(point, gate, None, None)]
    targets = [target for target in targets if sequence._distance(target[0]) >= 0 and not any(
        np.all((target[0] >= hand["xyxy"][:2]) & (target[0] <= hand["xyxy"][2:])) for hand in hands)]
    if not targets:
        return None, None
    locations = np.array([point for point, _, _, _ in targets])
    low, high = locations.min(axis=0), locations.max(axis=0)
    if np.any(high-low > np.asarray(policy.crop_size)*.8):
        return None, None
    bounds = _crop_bounds((low+high)/2, policy.crop_size, frame.shape[:2], keep_size=True)
    detections = _crop_detections(model, frame, bounds, args, thresholds, policy)
    visible_hands = hands+[obj for obj in detections if obj["class_name"] == "hand"]
    accepted = []
    for obj in tracker._candidates(detections):
        point = np.asarray(obj["center"])
        if (obj["conflict"] or sequence._distance(point) < 0 or tracker._occluded(obj, visible_hands)
                or any(box_containment(obj["xyxy"], other["xyxy"]) >= .9 for other in candidates)):
            continue
        for expected, gate, layout, index in targets:
            if np.linalg.norm(point-expected) > gate:
                continue
            if layout is None:
                accepted.append(obj)
                break
            revised = layout.copy()
            revised[index] = point
            if np.argmin(np.linalg.norm(layout-point, axis=1)) == index and slot_layout_error(revised) is not None:
                accepted.append(obj)
                break
    attempt = {"group_id": tracker.group_id, "phase": "load", "purpose": "third_preview_support",
               "anchor_track_ids": ids, "crop_xyxy": list(bounds), "accepted_detections": []}
    if len(accepted) != 1:
        return None, attempt
    support = accepted[0]
    if recent is not None:
        support = dict(support, track_id=recent[2])
    attempt["accepted_detections"] = [support]
    return support, attempt


def _recent_load_support_target(tracker, anchor_ids, measured, candidates):
    """Prefer a position backed by a real track already observed in the current group."""
    points, ids = tracker._entry_points, tracker._entry_track_ids
    if points is None or not set(anchor_ids).issubset(ids):
        return None
    visible_ids = {obj.get("track_id") for obj in candidates}
    recent = [i for i, track_id in enumerate(ids) if track_id is not None and track_id not in visible_ids
              and tracker.last_time-tracker._entry_seen_at[i] <= tracker.s4_contact_prediction_seconds]
    if len(recent) != 1:
        return None
    source = np.float32([points[ids.index(track_id)] for track_id in anchor_ids])
    transform, _ = cv2.estimateAffinePartial2D(source, measured)
    if transform is None or not .8 <= np.linalg.norm(transform[:, 0]) <= 1.2:
        return None
    index = recent[0]
    point = points[index] @ transform[:, :2].T + transform[:, 2]
    distance = np.min(np.linalg.norm(measured-point, axis=1))
    return point, min(tracker._entry_gate*.5, .25*distance), ids[index]


def _load_reference_targets(reference, measured, match_distance):
    targets = []
    for pair in permutations(range(4), 2):
        transform, _ = cv2.estimateAffinePartial2D(reference[list(pair)].astype(np.float32), measured)
        if transform is None:
            continue
        scale = np.linalg.norm(transform[:, 0])
        angle = abs(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
        if not .5 <= scale <= 2 or angle > 45:
            continue
        projected = reference @ transform[:, :2].T + transform[:, 2]
        spacing = np.min(np.linalg.norm(projected[:, None]-projected[None, :], axis=2)[np.triu_indices(4, 1)])
        gate = min(match_distance*.5, .25*spacing)
        targets.extend((point, gate, projected, index) for index, point in enumerate(projected) if index not in pair)
    return targets


def _slot_order(points):
    hull = cv2.convexHull(points.astype(np.float32), returnPoints=False).ravel()
    return tuple(np.roll(hull, -int(np.flatnonzero(hull == 3)[0])))


def _crop_bounds(center, crop_size, frame_size, *, keep_size=False):
    height, width = frame_size
    crop_width, crop_height = crop_size
    left, top = round(center[0])-crop_width//2, round(center[1])-crop_height//2
    if keep_size:
        left = min(max(0, left), max(0, width-crop_width))
        top = min(max(0, top), max(0, height-crop_height))
    return max(0, left), max(0, top), min(width, left+crop_width), min(height, top+crop_height)


def _crop_detections(model, frame, bounds, args, thresholds, policy):
    x1, y1, x2, y2 = bounds
    if x2-x1 < 32 or y2-y1 < 32:
        return []

    result = model.predict(frame[y1:y2,x1:x2], imgsz=policy.image_size,
        conf=min(thresholds.slot_state, thresholds.hand), device=args.device, iou=args.nms_iou,
        agnostic_nms=False, verbose=False, save=False)[0]
    boxes = result.boxes.cpu()
    detections = []
    for cls, score, xy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
        name = model.names[int(cls)]
        if name not in SLOT_STATES and name != "hand":
            continue
        if float(score) < thresholds.business_threshold(name):
            continue
        box = [xy[0]+x1, xy[1]+y1, xy[2]+x1, xy[3]+y1]
        detections.append({"class_id": int(cls), "class_name": name, "confidence": float(score),
                           "xyxy": box, "detection_source": "local_crop"})
    return detections


def _detect_in_crop(model, frame, bounds, points, gate, sequence, args, thresholds, policy):
    center = points[3]
    accepted, hands = [], []
    for obj in _crop_detections(model, frame, bounds, args, thresholds, policy):
        box = obj["xyxy"]
        point = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
        if obj["class_name"] == "hand":
            if np.linalg.norm(point-center) <= gate or (
                    box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]):
                hands.append(obj)
            continue
        distances = np.linalg.norm(points-point, axis=1)
        if (distances[3] <= gate and np.argmin(distances) == 3
                and sequence._distance(point) >= 0):
            accepted.append(obj)

    return accepted, hands

