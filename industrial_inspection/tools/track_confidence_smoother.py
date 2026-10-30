"""Pre-association confidence smoothing for slot ByteTrack detections.

The final ByteTrack ID for the current frame is not known until the tracker
has run. This module therefore uses the previous frame's track boxes only as
a conservative identity hint, computes a causal moving average, and lets the
tracker use that score for its high/low/new-track split. Histories are
committed only under the ID returned by ByteTrack, so a wrong pre-association
does not permanently move a confidence history to another ID.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import cv2

from ultralytics.trackers.utils import matching


def _box_iou(left, right):
    """Return IoU for two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = (float(value) for value in left)
    bx1, by1, bx2, by2 = (float(value) for value in right)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1))
    area_left = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_right = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_left + area_right - intersection
    return intersection / union if union > 0.0 else 0.0


class TrackConfidenceSmoother:
    """Compute a causal mean using previous-track geometry as a hint."""

    def __init__(self, window_size: int = 5, preassociation_iou: float = .30,
                 enabled: bool = True, preassociation_roi_contour=None):
        window_size = int(window_size)
        preassociation_iou = float(preassociation_iou)
        if window_size < 1:
            raise ValueError("confidence smoothing window_size must be at least 1")
        if not 0.0 < preassociation_iou <= 1.0:
            raise ValueError("preassociation_iou must be in (0, 1]")
        self.window_size = window_size
        self.preassociation_iou = preassociation_iou
        self.enabled = bool(enabled)
        self.preassociation_roi_contour = preassociation_roi_contour
        self._histories: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size))
        self._last_seen: dict[int, int] = {}

    def _match_previous_tracks(self, slots, previous_tracks):
        """Return ``current_slot_index -> previous_track_id`` matches."""
        if not self.enabled or not slots or not previous_tracks:
            return {}, {}
        distances = np.ones((len(previous_tracks), len(slots)), dtype=np.float32)
        ious = np.zeros_like(distances)
        for row, track in enumerate(previous_tracks):
            for column, slot in enumerate(slots):
                iou = _box_iou(track["xyxy"], slot["xyxy"])
                ious[row, column] = iou
                distances[row, column] = 1.0 - iou
        matches, _, _ = matching.linear_assignment(
            distances, thresh=1.0 - self.preassociation_iou)
        slot_to_track = {int(column): int(previous_tracks[row]["track_id"])
                         for row, column in matches}
        slot_to_iou = {int(column): float(ious[row, column])
                       for row, column in matches}
        return slot_to_track, slot_to_iou

    def _inside_preassociation_roi(self, slot):
        if self.preassociation_roi_contour is None:
            return True
        x1, y1, x2, y2 = (float(value) for value in slot["xyxy"])
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return cv2.pointPolygonTest(self.preassociation_roi_contour, center, False) >= 0

    def prepare(self, slots, previous_tracks):
        """Attach the score that will be passed to ByteTrack.

        ``slots`` are mutated in place. The current raw sample is included
        in the preview mean but is committed to history only after ByteTrack
        returns the actual current-frame ID.
        """
        eligible_indices = [index for index, slot in enumerate(slots)
                            if self._inside_preassociation_roi(slot)]
        eligible_slots = [slots[index] for index in eligible_indices]
        matched_indices, matched_ious = self._match_previous_tracks(eligible_slots, previous_tracks)
        slot_to_track = {eligible_indices[index]: track_id
                         for index, track_id in matched_indices.items()}
        slot_to_iou = {eligible_indices[index]: iou
                       for index, iou in matched_ious.items()}
        for index, slot in enumerate(slots):
            raw = float(slot["confidence"])
            slot["raw_confidence"] = raw
            roi_eligible = index in eligible_indices
            slot["smoothing_roi_eligible"] = roi_eligible
            previous_id = slot_to_track.get(index)
            if not roi_eligible:
                status = "outside_roi" if self.enabled else "disabled"
                slot.update(
                    tracking_confidence=raw,
                    smoothed_confidence=raw,
                    smoothing_window_count=0,
                    smoothing_ready=False,
                    smoothing_status=status,
                    smoothing_source_track_id=None,
                    smoothing_preassociation_iou=None)
                continue
            if previous_id is None or not self.enabled:
                slot.update(
                    tracking_confidence=raw,
                    smoothed_confidence=raw,
                    smoothing_window_count=0,
                    smoothing_ready=False,
                    smoothing_status="disabled" if not self.enabled else "new_candidate",
                    smoothing_source_track_id=None,
                    smoothing_preassociation_iou=None)
                continue
            history = self._histories.get(previous_id)
            values = list(history) if history else []
            values.append(raw)
            smoothed = sum(values) / len(values)
            slot.update(
                tracking_confidence=smoothed,
                smoothed_confidence=smoothed,
                smoothing_window_count=len(values),
                smoothing_ready=len(values) >= self.window_size,
                smoothing_status=("preassociated_ready" if len(values) >= self.window_size
                                  else "preassociated_warming_up"),
                smoothing_source_track_id=previous_id,
                smoothing_preassociation_iou=slot_to_iou.get(index))

    def commit(self, slots, frame_index: int):
        """Commit raw samples under ByteTrack's returned IDs."""
        if not self.enabled:
            return
        for slot in slots:
            track_id = slot.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            history = self._histories[track_id]
            history.append(float(slot.get("raw_confidence", slot["confidence"])))
            self._last_seen[track_id] = int(frame_index)

    def prune(self, frame_index: int, max_age_frames: int):
        """Drop histories after the same age used to retain ByteTrack tracks."""
        if not self.enabled:
            return
        max_age_frames = max(1, int(max_age_frames))
        cutoff = int(frame_index) - max_age_frames
        stale = [track_id for track_id, last_seen in self._last_seen.items()
                 if last_seen < cutoff]
        for track_id in stale:
            self._last_seen.pop(track_id, None)
            self._histories.pop(track_id, None)

    def reset(self):
        self._histories.clear()
        self._last_seen.clear()
