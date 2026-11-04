"""局部补检策略：只返回补充观测，身份关联、状态确认和工艺判断由调用方管理。"""
from dataclasses import dataclass

import cv2
import numpy as np

from industrial_inspection.tools.slot_group_tracker import SLOT_STATES


@dataclass(frozen=True)
class LocalRedetectionPolicy:
    enabled: bool
    crop_size: tuple
    image_size: int


def recover_missing_slots(model, frame, objects, sequence, args, thresholds, policy):
    """仅为操作中已编号的漏检槽补框；坐标还原后仍走原来的关联、遮挡和连续确认。"""
    tracker = sequence.tracker
    if (not policy.enabled or sequence.state != "active" or tracker.points is None
            or tracker.identity_lost):
        return objects, []
    gate = tracker._point_tracking_gate or tracker._spacing()*tracker.match_distance_ratio
    matches = tracker._associate(tracker._candidates(objects), gate)
    points = tracker.reference_points.copy()
    height,width = frame.shape[:2]
    crop_width,crop_height = policy.crop_size
    additions,attempts = [],[]
    added_hand_boxes = set()
    for i,center in enumerate(points):
        if i in matches or sequence._distance(center) < sequence.margin:
            continue
        left,top = round(center[0])-crop_width//2,round(center[1])-crop_height//2
        x1,y1 = max(0,left),max(0,top)
        x2,y2 = min(width,left+crop_width),min(height,top+crop_height)
        if x2-x1 < 32 or y2-y1 < 32:
            continue
        result = model.predict(frame[y1:y2,x1:x2],imgsz=policy.image_size,
            conf=min(thresholds.slot_state, thresholds.hand),device=args.device,iou=args.nms_iou,agnostic_nms=False,
            verbose=False,save=False)[0]
        boxes = result.boxes.cpu()
        accepted,hands = [],[]
        for cls,score,xy in zip(boxes.cls.tolist(),boxes.conf.tolist(),boxes.xyxy.tolist()):
            name = model.names[int(cls)]
            if name not in SLOT_STATES and name != "hand":
                continue
            if float(score) < thresholds.business_threshold(name):
                continue
            box = [xy[0]+x1,xy[1]+y1,xy[2]+x1,xy[3]+y1]
            obj = {"class_id": int(cls),"class_name": name,"confidence": float(score),
                   "xyxy": box,"detection_source": "local_crop"}
            if name == "hand":
                hands.append(obj)
                continue
            point = np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
            distances = np.linalg.norm(points-point,axis=1)
            # 只给本次漏检槽补框，邻槽和ROI外的框不能借补检参与编号或换组。
            if (distances[i] <= gate and np.argmin(distances) == i
                    and sequence._distance(point) >= 0):
                accepted.append(obj)
        additions.extend(accepted)
        # 槽框和手框可能同时被局部模型检出，也可能只有手框：后一种
        # 情况只提供 S4 的遮挡/接触信息，不能证明已上料。只保留靠近当前补检槽
        # 的手框，并跨裁剪去重，避免一个手框重复参与接触计数。
        relevant_hands = hands if accepted or tracker.phase == "load" else []
        for hand in relevant_hands:
            a = hand["xyxy"]
            hand_center = np.array([(a[0]+a[2])/2,(a[1]+a[3])/2])
            if np.linalg.norm(hand_center-center) <= gate or (
                    a[0] <= center[0] <= a[2] and a[1] <= center[1] <= a[3]):
                key = tuple(round(float(v), 1) for v in a)
                if key not in added_hand_boxes:
                    added_hand_boxes.add(key)
                    additions.append(hand)
        attempts.append({"group_id": tracker.group_id,"slot_id": i+1,"crop_xyxy": [x1,y1,x2,y2],
                         "accepted_detections": accepted})
    return objects+additions, attempts

