"""独立解析各类证据的置信度门槛，不负责槽位关联或状态变化。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DetectionThresholds:
    display: float
    audit: float
    slot_state: float
    hand: float
    handle: float
    tracking_high: float
    tracking_low: float

    @classmethod
    def resolve(cls, default, audit, *, slot_state=None, hand=None, handle=None,
                tracking_high=None, tracking_low=None, legacy_low_cap=.10):
        """None显式沿用旧门槛；跟踪低分下限默认保留旧公式。"""
        def value(override):
            return float(default if override is None else override)

        high = value(tracking_high)
        low = min(float(legacy_low_cap), high * .5) if tracking_low is None else float(tracking_low)
        result = cls(float(default), float(audit), value(slot_state), value(hand), value(handle), high, low)
        if any(not math.isfinite(v) or not 0 < v <= 1 for v in asdict(result).values()):
            raise ValueError("各置信度门槛必须为(0,1]范围内的有限数")
        if low >= high:
            raise ValueError("ByteTrack低分门槛必须小于高分门槛")
        if result.audit > min(result.display, result.slot_state, result.hand, result.handle, low):
            raise ValueError("audit-conf必须不高于显示、槽位、手部、把手及跟踪低分门槛，否则证据会在推理时被过滤")
        return result

    def business_threshold(self, class_name):
        if class_name in ("wafer_slot_empty", "wafer_slot_filled"):
            return self.slot_state
        if class_name == "hand":
            return self.hand
        if class_name == "hatch_handle":
            return self.handle
        return self.display

    def admission(self, item):
        """返回证据准入及拒绝原因；不修改检测行，也不创建轨迹。"""
        score = float(item.get("tracking_confidence", item["confidence"]))
        threshold = self.business_threshold(item.get("class_name"))
        if score < threshold:
            return False, "below_business_confidence_threshold", threshold
        if item.get("class_name") in ("wafer_slot_empty", "wafer_slot_filled") and item.get("track_id") is None:
            return False, "missing_bytetrack_id", threshold
        return True, "admitted", threshold

    def snapshot(self):
        return asdict(self)


def prepare_business_detections(tracked_raw, confidence_threshold):
    """旧数值接口仍可用；分职责门槛通过DetectionThresholds传入。"""
    policy = (confidence_threshold if isinstance(confidence_threshold, DetectionThresholds)
              else DetectionThresholds.resolve(confidence_threshold, min(.001, confidence_threshold * .1)))
    business = []
    for item in tracked_raw:
        if not policy.admission(item)[0]:
            continue
        working = dict(item)
        working["confidence"] = float(item.get("tracking_confidence", item["confidence"]))
        business.append(working)
    return business
