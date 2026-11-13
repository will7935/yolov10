"""局部补检策略：只返回补充观测，身份关联、状态确认和工艺判断由调用方管理。"""
from dataclasses import dataclass

import cv2
import numpy as np

from industrial_inspection.tools.slot_group_tracker import SLOT_STATES, box_containment, valid_slot_geometry


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


def _detect_in_crop(model, frame, bounds, points, gate, sequence, args, thresholds, policy):
    x1, y1, x2, y2 = bounds
    if x2-x1 < 32 or y2-y1 < 32:
        return [], []

    center = points[3]
    result = model.predict(frame[y1:y2,x1:x2], imgsz=policy.image_size,
        conf=min(thresholds.slot_state, thresholds.hand), device=args.device, iou=args.nms_iou,
        agnostic_nms=False, verbose=False, save=False)[0]
    boxes = result.boxes.cpu()
    accepted, hands = [], []
    for cls, score, xy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
        name = model.names[int(cls)]
        if name not in SLOT_STATES and name != "hand":
            continue
        if float(score) < thresholds.business_threshold(name):
            continue
        box = [xy[0]+x1, xy[1]+y1, xy[2]+x1, xy[3]+y1]
        obj = {"class_id": int(cls), "class_name": name, "confidence": float(score),
               "xyxy": box, "detection_source": "local_crop"}
        point = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
        if name == "hand":
            if np.linalg.norm(point-center) <= gate or (
                    box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]):
                hands.append(obj)
            continue
        distances = np.linalg.norm(points-point, axis=1)
        if (distances[3] <= gate and np.argmin(distances) == 3
                and sequence._distance(point) >= 0):
            accepted.append(obj)

    return accepted, hands

