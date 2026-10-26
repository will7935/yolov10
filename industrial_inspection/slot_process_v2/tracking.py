"""YOLO's bundled ByteTrack, restricted to measured slot_group boxes.

Kalman output is used only to obtain an ID. The process always receives the
original detector box; predicted/smoothed coordinates are never switch evidence.
"""

from types import SimpleNamespace

import numpy as np


class CarrierTracker:
    def __init__(self, fps=18.565, buffer_seconds=3.0, high_thresh=0.35, low_thresh=0.10):
        from ultralytics.trackers.byte_tracker import BYTETracker

        if fps <= 0 or buffer_seconds < 0 or not 0 <= low_thresh < high_thresh <= 1:
            raise ValueError("Invalid ByteTrack frame rate, buffer, or confidence thresholds")
        # ByteTrack scales track_buffer by frame_rate / 30 internally. Keep
        # the configured duration in 30-fps reference frames (like bytetrack.yaml).
        args = SimpleNamespace(
            track_high_thresh=float(high_thresh),
            track_low_thresh=float(low_thresh),
            new_track_thresh=float(high_thresh),
            match_thresh=0.80,
            track_buffer=max(1, round(buffer_seconds * 30)),
        )
        self.tracker = BYTETracker(args, frame_rate=fps)

    def update(self, objects):
        groups = [dict(d) for d in objects if d.get("class_name") == "slot_group"]
        boxes = np.asarray([d["xyxy"] for d in groups], dtype=np.float32).reshape(-1, 4)
        xywh = boxes.copy()
        xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        inputs = SimpleNamespace(
            xywh=xywh,
            conf=np.asarray([d["confidence"] for d in groups]),
            cls=np.asarray([d.get("class_id", 5) for d in groups]),
        )
        tracks = self.tracker.update(inputs)
        for row in tracks:
            source_index = int(row[-1])
            if 0 <= source_index < len(groups):
                groups[source_index]["track_id"] = int(row[4])
        for d in groups:
            d["center_source"] = "detector"
            d["tracking_backend"] = "ultralytics.ByteTrack"
        return groups
