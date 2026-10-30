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
from industrial_inspection.tools.track_confidence_smoother import TrackConfidenceSmoother


class SlotByteTracker:
    """Assign ByteTrack IDs to slot detections while preserving all objects."""

    def __init__(self, fps, *, high_thresh=0.35, low_thresh=0.10, buffer_seconds=3.0,
                 confidence_smoothing_enabled=True, confidence_smoothing_window=5,
                 confidence_preassociation_iou=.30, confidence_preassociation_roi=None):
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
        self._confidence_smoother = TrackConfidenceSmoother(
            confidence_smoothing_window, confidence_preassociation_iou,
            enabled=confidence_smoothing_enabled,
            preassociation_roi_contour=confidence_preassociation_roi)
        self._confidence_history_max_age = max(1, round(buffer_seconds * 30))

    def reset(self):
        """Explicitly restart tracking (e.g. for another video), never at a G switch."""
        self.tracker = self._tracker_class(self._tracker_args, frame_rate=self._frame_rate)
        self._confidence_smoother.reset()

    def _previous_track_boxes(self):
        """Read the last available boxes without advancing ByteTrack."""
        tracks = []
        seen = set()
        for track in list(self.tracker.tracked_stracks) + list(self.tracker.lost_stracks):
            track_id = getattr(track, "track_id", None)
            if track_id is None or int(track_id) in seen or not getattr(track, "is_activated", False):
                continue
            box = getattr(track, "xyxy", None)
            if box is None:
                continue
            seen.add(int(track_id))
            tracks.append({"track_id": int(track_id), "xyxy": np.asarray(box, dtype=np.float32)})
        return tracks

    def update(self, objects, frame_index=None):
        """Return copies of all detector rows, adding ``track_id`` to slots.

        Low-confidence rows can be used by ByteTrack's second association.
        ``confidence`` remains the raw model score; ``tracking_confidence`` is
        the pre-associated moving-average score used inside ByteTrack.
        """
        rows = [dict(obj) for obj in objects]
        slots = [obj for obj in rows if obj.get("class_name") in SLOT_STATES]
        for obj in slots:
            obj["track_id"] = None
            obj["tracking_backend"] = "ultralytics.ByteTrack"
        for obj in rows:
            obj.setdefault("raw_confidence", float(obj["confidence"]))
            obj.setdefault("tracking_confidence", float(obj["confidence"]))
            obj.setdefault("smoothed_confidence", float(obj["confidence"]))
            obj.setdefault("smoothing_window_count", 0)
            obj.setdefault("smoothing_ready", False)
            obj.setdefault("smoothing_status", "not_slot")
            obj.setdefault("smoothing_source_track_id", None)
            obj.setdefault("smoothing_preassociation_iou", None)
            obj.setdefault("smoothing_roi_eligible", None)
        if frame_index is None:
            frame_index = self.tracker.frame_id + 1
        self._confidence_smoother.prune(frame_index, self._confidence_history_max_age)
        if not slots:
            # Advance ByteTrack on empty frames so lost-track ageing is valid.
            empty = np.empty((0, 4), dtype=np.float32)
            inputs = SimpleNamespace(xywh=empty, conf=np.empty((0,), dtype=np.float32),
                                     cls=np.empty((0,), dtype=np.float32))
            self.tracker.update(inputs)
            return rows

        # Scheme 2: preview each current score from a conservative geometric
        # match to the previous frame's track boxes before ByteTrack splits
        # detections into high/low/new-track groups.
        self._confidence_smoother.prepare(slots, self._previous_track_boxes())

        boxes = np.asarray([obj["xyxy"] for obj in slots], dtype=np.float32).reshape(-1, 4)
        xywh = boxes.copy()
        xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        inputs = SimpleNamespace(
            xywh=xywh,
            conf=np.asarray([obj["tracking_confidence"] for obj in slots], dtype=np.float32),
            # Empty and filled are the same physical slot, so use one class
            # for association and retain the detector class on each row.
            cls=np.zeros((len(slots),), dtype=np.float32),
        )
        tracks = self.tracker.update(inputs)
        for row in tracks:
            source_index = int(row[-1])
            if 0 <= source_index < len(slots):
                slots[source_index]["track_id"] = int(row[4])
        for obj in slots:
            actual_id = obj.get("track_id")
            hinted_id = obj.get("smoothing_source_track_id")
            if not self._confidence_smoother.enabled:
                obj["smoothing_status"] = "disabled"
            elif actual_id is not None and hinted_id is not None and int(actual_id) != int(hinted_id):
                obj["smoothing_status"] = "preassociation_mismatch"
            elif actual_id is not None and hinted_id is None:
                obj["smoothing_status"] = "track_bootstrap"
        self._confidence_smoother.commit(slots, frame_index)
        return rows


# Compatibility for scripts importing the former name. The implementation
# above tracks slots, never ``slot_group``.
CarrierByteTracker = SlotByteTracker

