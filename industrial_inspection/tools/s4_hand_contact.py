"""S4接触证据策略。只判断当前帧，不累计时间、不写槽位状态或工艺事件。"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HandContactPolicy:
    overlap_ratio: float
    competing_enabled: bool
    competing_index: int
    competing_margin: float
    competing_ratio: float


@dataclass
class HandContactEvidence:
    accepted: bool
    reason: str
    source: str | None
    region_hits: list
    candidates: list
    overlap_ratio: float = 0.0

    def diagnostic(self):
        return {"contact_detected": self.accepted, "decision_reason": self.reason,
                "source": self.source, "candidates": self.candidates}


def _centered_box(point, size):
    if size is None or not np.isfinite(size).all():
        return None
    width, height = max(float(size[0]), 1.), max(float(size[1]), 1.)
    return [point[0]-width/2, point[1]-height/2, point[0]+width/2, point[1]+height/2]


def box_overlap_ratio(box, hand_box):
    if box is None:
        return 0.0
    overlap = max(0., min(box[2], hand_box[2])-max(box[0], hand_box[0])) * \
              max(0., min(box[3], hand_box[3])-max(box[1], hand_box[1]))
    return overlap / max((box[2]-box[0])*(box[3]-box[1]), 1.)


def evaluate_s4_contact(points, box_sizes, slot_index, hands, policy, *, region=None, region_expected=False):
    """保留原有最高置信度选择、中心覆盖和竞争判定，额外返回拒绝原因。

    先尝试原有中心/框接触，再尝试区域补偿。开启区域不能覆盖原本有效的接触。
    其他槽中心覆盖的歧义检查独立于competing_enabled，关闭竞争面积比较仍保留此检查。
    """
    target_box = _centered_box(points[slot_index], box_sizes[slot_index])
    competing_index = policy.competing_index
    competing_box = (_centered_box(points[competing_index], box_sizes[competing_index])
                     if 0 <= competing_index < len(points) else None)
    direct, regional, hits, diagnostics = [], [], [], []

    def decision(candidate, allow_region):
        covered = candidate["covered_slot_ids"]
        if covered == [slot_index+1]:
            return True, "s4_center_covered", "s4_center"
        box_contact = target_box is not None and candidate["s4_overlap_ratio"] >= policy.overlap_ratio
        if not box_contact and not (allow_region and candidate["region_hit"]):
            return False, "no_s4_box_or_region_contact", None
        if covered and covered != [slot_index+1]:
            return False, "other_slot_center_covered", None
        if policy.competing_enabled:
            if competing_index+1 in covered and slot_index+1 not in covered:
                return False, "competing_slot_center_covered", None
            dominant = (candidate["competing_overlap_ratio"] >= policy.competing_margin
                        and candidate["competing_overlap_ratio"] >= policy.competing_ratio *
                            max(candidate["s4_overlap_ratio"], 1e-9))
            if dominant and slot_index+1 not in covered:
                return False, "competing_slot_dominant", None
        return True, "s4_box_contact" if box_contact else "s4_region_contact", \
            "s4_box" if box_contact else "s4_region"

    for hand in hands:
        hand_box = hand["xyxy"]
        region_hit = False
        if region is not None:
            x1, y1, x2, y2 = hand_box
            if x2 > x1 and y2 > y1:
                contour = np.asarray([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
                overlap, _ = cv2.intersectConvexConvex(region, contour)
                region_hit = overlap > 0
        covered = [i+1 for i, point in enumerate(points)
                   if hand_box[0] <= point[0] <= hand_box[2] and hand_box[1] <= point[1] <= hand_box[3]]
        s4_ratio = box_overlap_ratio(target_box, hand_box)
        competing_ratio = box_overlap_ratio(competing_box, hand_box)
        relevant = (slot_index+1 in covered or competing_index+1 in covered
                    or s4_ratio >= policy.overlap_ratio or competing_ratio >= policy.competing_margin)
        try:
            confidence = float(hand.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if region_hit:
            hits.append({"xyxy": [float(v) for v in hand_box], "confidence": confidence})
        candidate = {"confidence": confidence, "xyxy": [float(v) for v in hand_box],
                     "covered_slot_ids": covered, "s4_overlap_ratio": s4_ratio,
                     "competing_overlap_ratio": competing_ratio, "region_hit": region_hit}
        if relevant or region_hit:
            candidate["direct_reason"] = decision(candidate, False)[1]
            candidate["region_reason"] = (decision(candidate, True)[1] if region_expected
                                           else "region_order_not_expected")
            diagnostics.append(candidate)
        if relevant:
            direct.append(candidate)
        if relevant or (region_hit and region_expected):
            regional.append(candidate)

    outcome = (False, "no_contact_evidence", None)
    selected = None
    if direct:
        selected = max(direct, key=lambda item: item["confidence"])
        outcome = decision(selected, False)
        if outcome[0]:
            return HandContactEvidence(*outcome, hits, diagnostics, selected["s4_overlap_ratio"])
    if region_expected and regional:
        selected = max(regional, key=lambda item: item["confidence"])
        outcome = decision(selected, True)
    elif hits and not region_expected:
        outcome = (False, "region_order_not_expected", None)
    return HandContactEvidence(*outcome, hits, diagnostics,
                               0.0 if selected is None else selected["s4_overlap_ratio"])
