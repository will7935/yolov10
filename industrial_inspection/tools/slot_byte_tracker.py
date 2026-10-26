"""Ultralytics ByteTrack adapter for wafer-slot detections.

``wafer_slot_empty`` and ``wafer_slot_filled`` are two observations of the
same physical slot. They share one ByteTrack pool; the original detector
class remains available to the process state machine.

The old class name is retained as an import-compatible alias for the legacy
entry point. It no longer tracks ``slot_group`` detections.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from industrial_inspection.tools.slot_group_tracker import SLOT_STATES


class SlotByteTracker:
    """Assign ByteTrack IDs to slot detections while preserving all objects."""

    def __init__(self, fps, *, high_thresh=0.35, low_thresh=0.10, buffer_seconds=3.0):
        from ultralytics.trackers.byte_tracker import BYTETracker

        if fps <= 0 or buffer_seconds < 0 or not 0 <= low_thresh < high_thresh <= 1:
            raise ValueError("Invalid ByteTrack frame rate, buffer, or confidence thresholds")
        self._tracker_args = SimpleNamespace(
            track_high_thresh=float(high_thresh),
            track_low_thresh=float(low_thresh),
            new_track_thresh=float(high_thresh),
            match_thresh=0.80,
            # Ultralytics scales this 30-fps reference value internally.
            track_buffer=max(1, round(buffer_seconds * 30)),
        )
        self._tracker_class = BYTETracker
        self._frame_rate = float(fps)
        self.tracker = BYTETracker(self._tracker_args, frame_rate=fps)

    def reset(self):
        """Explicitly restart tracking (e.g. for another video), never at a G switch."""
        self.tracker = self._tracker_class(self._tracker_args, frame_rate=self._frame_rate)

    def update(self, objects):
        """Return copies of all detector rows, adding ``track_id`` to slots.

        Low-confidence rows can be used by ByteTrack's second association. A
        caller should still filter them before feeding the process state
        machine; they are intentionally returned with their original score.
        """
        rows = [dict(obj) for obj in objects]
        slots = [obj for obj in rows if obj.get("class_name") in SLOT_STATES]
        for obj in slots:
            obj["track_id"] = None
            obj["tracking_backend"] = "ultralytics.ByteTrack"
        if not slots:
            # Advance ByteTrack on empty frames so lost-track ageing is valid.
            empty = np.empty((0, 4), dtype=np.float32)
            inputs = SimpleNamespace(xywh=empty, conf=np.empty((0,), dtype=np.float32),
                                     cls=np.empty((0,), dtype=np.float32))
            self.tracker.update(inputs)
            return rows

        boxes = np.asarray([obj["xyxy"] for obj in slots], dtype=np.float32).reshape(-1, 4)
        xywh = boxes.copy()
        xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        inputs = SimpleNamespace(
            xywh=xywh,
            conf=np.asarray([obj["confidence"] for obj in slots], dtype=np.float32),
            # Empty and filled are the same physical slot, so use one class
            # for association and retain the detector class on each row.
            cls=np.zeros((len(slots),), dtype=np.float32),
        )
        tracks = self.tracker.update(inputs)
        for row in tracks:
            source_index = int(row[-1])
            if 0 <= source_index < len(slots):
                slots[source_index]["track_id"] = int(row[4])
        return rows


# Compatibility for scripts importing the former name. The implementation
# above tracks slots, never ``slot_group``.
CarrierByteTracker = SlotByteTracker

