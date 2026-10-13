"""Polygon ROI selection and center-based filtering for wafer slots only."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SLOT_NAMES = {"wafer_slot_empty", "wafer_slot_filled", "wafer_slot"}


def validate_polygon(points):
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("ROI requires at least three (x, y) vertices")
    if not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any():
        raise ValueError("Normalized ROI coordinates must be finite and within [0, 1]")
    if len(np.unique(polygon, axis=0)) != len(polygon):
        raise ValueError("ROI vertices must be distinct")
    # Convex polygons cover the intended rectangle/quadrilateral without
    # accepting self-crossed or ambiguous selections.
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < 1e-5:
        raise ValueError("ROI must be a convex polygon; click vertices around its perimeter")
    return polygon


def load_roi(path, frame_size):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("coordinate_system") != "normalized_xy":
        raise ValueError("Unsupported ROI coordinate system")
    polygon = validate_polygon(data["polygon"])
    old_w, old_h = data["reference_frame_size"]
    width, height = frame_size
    if old_w <= 0 or old_h <= 0 or abs((old_w / old_h) / (width / height) - 1) > .01:
        raise ValueError("Video aspect ratio changed; redraw ROI with ROI_RESELECT=True")
    return polygon


def save_roi(path, polygon, frame_size, source, reference_frame):
    polygon = validate_polygon(polygon)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"coordinate_system": "normalized_xy", "polygon": polygon.tolist(),
            "reference_frame_size": list(frame_size), "source": str(Path(source).resolve()),
            "reference_frame": int(reference_frame), "filter_rule": "slot_box_center_inside_polygon"}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def pixel_polygon(polygon, frame_size):
    return np.asarray(polygon, dtype=np.float32) * np.asarray(frame_size, dtype=np.float32)


def filter_slot_indices(boxes, names, polygon, frame_size):
    """Include boundary centers; leave all non-slot classes unchanged."""
    contour = pixel_polygon(polygon, frame_size)
    keep, excluded = [], []
    for i, (xyxy, cls) in enumerate(zip(boxes.xyxy.tolist(), boxes.cls.tolist())):
        center = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
        if names[int(cls)] in SLOT_NAMES and cv2.pointPolygonTest(contour, center, False) < 0:
            excluded.append(i)
        else:
            keep.append(i)
    return keep, excluded


def draw_roi(frame, polygon):
    height, width = frame.shape[:2]
    contour = np.rint(pixel_polygon(polygon, (width, height))).astype(np.int32)
    cv2.polylines(frame, [contour], True, (255, 255, 0), 2)
    x, y = contour[0]
    cv2.putText(frame, "Slot ROI", (max(0, x), max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 0), 2)


def select_roi(source, path, initial_frame=0):
    """Select on a seekable video preview; no cropping of model input."""
    capture = cv2.VideoCapture(str(source))
    window = "Select lower slots ROI - Enter save / Esc cancel"
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video for ROI selection: {source}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or not 0 <= initial_frame < total:
            raise ValueError("ROI_SELECT_FRAME is outside the video frame range")
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        requested = [initial_frame]
        points = []
        display_size = [0, 0]

        def seek(value):
            requested[0] = value

        def click(event, x, y, flags, param):
            w, h = display_size
            if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < w and 0 <= y < h:
                points.append((x / w, y / h))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

        cv2.createTrackbar("Frame", window, initial_frame, max(1, total - 1), seek)
        cv2.setMouseCallback(window, click)
        displayed = None
        image = None
        error = ""
        while True:
            if requested[0] != displayed:
                capture.set(cv2.CAP_PROP_POS_FRAMES, requested[0])
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot read ROI selection frame {requested[0]}")
                height, width = frame.shape[:2]
                scale = min(1.0, 1200 / width, 750 / height)
                image = cv2.resize(frame, (round(width * scale), round(height * scale)))
                display_size[:] = [image.shape[1], image.shape[0]]
                # Never carry vertices from another door position unnoticed.
                points.clear()
                displayed = requested[0]
                error = ""
            preview = image.copy()
            contour = np.rint(pixel_polygon(points, display_size)).astype(np.int32) if points else None
            if contour is not None:
                cv2.polylines(preview, [contour], len(points) >= 3, (255, 255, 0), 2)
                for x, y in contour:
                    cv2.circle(preview, (x, y), 4, (0, 255, 255), -1)
            for i, text in enumerate((f"Frame {displayed}: slider to choose a clear frame",
                                      "Left click vertices / Right click undo / R clear",
                                      "Enter save / Esc cancel; select LOWER four slot centers", error)):
                if text:
                    cv2.putText(preview, text, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                                .48, (0, 0, 0), 3)
                    cv2.putText(preview, text, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                                .48, (255, 255, 255), 1)
            cv2.imshow(window, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("ROI selection cancelled; no video was processed")
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13):
                try:
                    polygon = validate_polygon(points)
                except ValueError as exc:
                    error = str(exc)
                    continue
                save_roi(path, polygon, (width, height), source, displayed)
                print(f"Saved slot ROI: {Path(path).resolve()}", flush=True)
                return polygon
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
