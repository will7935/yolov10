"""Create top-down previews and full-video exports."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from contextlib import ExitStack, closing, contextmanager, suppress
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import cv2
import numpy as np


# 在 PyCharm 修改下面的配置，然后直接运行本文件。
SOURCE = r"I:\youyan\test_dataset\测试视频10\camera_B_20260924_062208.avi"
EXPORT_FULL_VIDEO = True  # True 导出整段俯视和左右对照 MP4，False 导出选中的帧。
SELECT_FRAMES_IN_WINDOW = False  # 运行视频时打开选帧窗口。设为 False 时直接使用下方帧号。
FRAME_NUMBERS = (509, 1685)  # 从 0 开始。打开窗口时只用第一项作为起始帧，图片自动使用第 0 帧。
OUTPUT_FOLDER = None  # 也可填写 r"D:\slot_perspective"。留空时保存在输入文件旁。

# 已有 calibration.json 时填写其路径，复用所保存的四角。整段导出会直接开始。
CALIBRATION_FILE = r"I:\youyan\test_dataset\测试视频11\slot_perspective\calibration.json"

# 新标定可在视频窗口依次点左上、右上、右下、左下。保留 None 即可点选。
# 也可填写四组原图像素坐标 (x, y)，用于复用点位或处理单张图片。
CALIBRATION_POINTS = None
OUTPUT_SIZE = (300, 600)  # (宽像素, 高像素)。设为 None 时沿用标定文件中的尺寸。
VIDEO_CRF = 19  # H.264 视频质量，数值越低画质越高。默认 19。
FFMPEG_PATH = None  # 可选，填写 ffmpeg.exe 的路径。留空时自动查找。

SLOT_STATES_FILE = None  # 可选，填写同一输入视频的 slot_states.jsonl 路径。

# 独立旋转测试。None 恢复原来的俯视图片/整段视频导出；不调用正式检测流程。
ROTATION_TEST_DEGREES = 40.0
ROTATION_TEST_DIRECTION = "counterclockwise"  # 下料布局绕机心逆时针50°得到上料预测；both可对照两方向。
MACHINE_CENTER_XY = (637.0, 266.0)  # 原相机画面中的机心像素坐标。
ROTATION_REFERENCE_FRAME = None  # None 自动选择 G1 下料的稳定四实槽帧。
ROTATION_TARGET_FRAMES = None    # None 自动选择 G1 上料首次可靠看见两槽的帧。


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Rectify a planar rectangle without running slot detection.",
        epilog="Select the physical rectangle in TL, TR, BR, BL order. "
               "Set the output width and height in the rectangle's real aspect ratio.",
    )
    parser.add_argument("--source", type=Path, required=True, help="Image or video file")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the previews")
    parser.add_argument("--frames", type=int, nargs="+", default=[0],
                        help="Zero-based frame numbers. The first is the calibration frame")
    parser.add_argument("--full-video", action="store_true",
                        help="Export the entire video and an original/top-down comparison as MP4")
    input_mode = parser.add_mutually_exclusive_group(required=True)
    input_mode.add_argument("--calibration", type=Path, help="Reuse a saved calibration.json")
    input_mode.add_argument("--points", type=float, nargs=8,
                            metavar=("TL_X", "TL_Y", "TR_X", "TR_Y", "BR_X", "BR_Y", "BL_X", "BL_Y"),
                            help="Pixel coordinates of the four corners")
    parser.add_argument("--size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Output dimensions, including when reusing calibration points")
    parser.add_argument("--crf", type=int, default=VIDEO_CRF, help="H.264 quality for full-video exports")
    parser.add_argument("--ffmpeg", type=Path, default=FFMPEG_PATH, help="Optional path to FFmpeg")
    parser.add_argument("--slot-states", type=Path,
                        help="Optional slot_states.jsonl for selected-frame image overlays")
    parser.add_argument("--rotation-degrees", type=float,
                        help="Run the independent fixed-angle rotation test instead of video export")
    parser.add_argument("--rotation-direction", choices=("clockwise", "counterclockwise", "both"),
                        default=ROTATION_TEST_DIRECTION)
    parser.add_argument("--machine-center", type=float, nargs=2, default=MACHINE_CENTER_XY,
                        metavar=("X", "Y"), help="Machine center in original image pixels")
    parser.add_argument("--rotation-reference-frame", type=int,
                        help="G1 unload frame with four measured and numbered slot centers")
    parser.add_argument("--rotation-target-frames", type=int, nargs="+",
                        help="Load frames for the rotation comparison; defaults to the first two-slot frame")
    return parser.parse_args(argv), parser


def validate_frames(indices):
    if not indices or min(indices) < 0:
        raise ValueError("Specify at least one nonnegative frame number")
    if len(set(indices)) != len(indices):
        raise ValueError("Frame numbers must be distinct")


def validate_options(args):
    validate_frames(args.frames)
    if args.calibration is None and args.size is None:
        raise ValueError("A new calibration requires an output width and height")
    if args.size is not None and (len(args.size) != 2
                                 or any(type(dimension) is not int or dimension < 2 for dimension in args.size)):
        raise ValueError("Output width and height must be integers of at least 2 pixels")


def read_video_frame(capture, index):
    if index != int(capture.get(cv2.CAP_PROP_POS_FRAMES)):
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError(f"Cannot seek to video frame {index}")
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Cannot read video frame {index}")
    return frame


def select_video_frames(source, initial_frame, *, mark_corners=False, full_video=False):
    if not source.is_file():
        raise ValueError(f"Input does not exist: {source}")
    window = "Select frames - Enter confirm / Esc cancel"
    trackbar = "Frame"
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        last_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if last_frame < 0:
            raise ValueError("Cannot determine the video frame count. Set SELECT_FRAMES_IN_WINDOW = False.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(trackbar, window, 0, max(1, last_frame), lambda _: None)
        cv2.setTrackbarPos(trackbar, window, min(max(initial_frame, 0), last_frame))
        marked = []
        displayed = None
        image = None
        frame_size = None
        corners = []
        corner_labels = ("TL", "TR", "BR", "BL")
        calibration_frame = None
        calibration_size = None
        error = ""
        header_height = 139 if mark_corners else 74

        def click(event, x, y, flags, userdata):
            nonlocal calibration_frame, calibration_size, error
            if image is None or not (0 <= x < image.shape[1] and header_height <= y < header_height + image.shape[0]):
                return
            if event == cv2.EVENT_RBUTTONDOWN and displayed == calibration_frame and corners:
                corners.pop()
                if not corners:
                    calibration_frame = None
                    calibration_size = None
                error = ""
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if corners and displayed != calibration_frame:
                error = f"Return to frame {calibration_frame} to edit points, or press R to clear."
                return
            if len(corners) == len(corner_labels):
                return
            calibration_frame = displayed
            calibration_size = frame_size
            image_height, image_width = image.shape[:2]
            width, height = frame_size
            corners.append((
                round(x * (width - 1) / max(1, image_width - 1)),
                round((y - header_height) * (height - 1) / max(1, image_height - 1)),
            ))
            error = ""

        if mark_corners:
            cv2.setMouseCallback(window, click)

        while True:
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return [], None, None
                index = min(cv2.getTrackbarPos(trackbar, window), last_frame)
            except cv2.error:
                return [], None, None
            if index < 0:
                return [], None, None
            if index != displayed:
                frame = read_video_frame(capture, index)
                height, width = frame.shape[:2]
                frame_size = (width, height)
                scale = min(1.0, 1200 / width, (754 - header_height) / height)
                image = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
                displayed = index

            preview = cv2.copyMakeBorder(image, header_height, 0, 0, max(0, 600 - image.shape[1]),
                                         cv2.BORDER_CONSTANT, value=(32, 32, 32))
            time_text = f" | {displayed / fps:.2f} s" if fps > 0 else ""
            marked_text = ", ".join(map(str, marked[-6:])) or "-"
            if len(marked) > 6:
                marked_text = f"..., {marked_text}"
            lines = [f"Frame {displayed} / {last_frame}{time_text}"]
            if full_video:
                lines += ["Full video export | Choose a frame for calibration",
                          "A/D: step | Enter: confirm four corners | Esc: cancel"]
            else:
                lines += [f"Marked ({len(marked)}): {marked_text}",
                          "A/D: step | Space: mark/unmark frame | Enter: confirm | Esc: cancel"]
            if mark_corners:
                point_status = f"Corners: {len(corners)}/4"
                if corners:
                    point_status += f" on frame {calibration_frame}"
                lines += [
                    "Click corners: TL, TR, BR, BL | Right click: undo | R: clear",
                    point_status,
                    error,
                ]
            if corners and displayed == calibration_frame:
                image_height, image_width = image.shape[:2]
                width, height = frame_size
                vertices = [
                    (round(x * (image_width - 1) / max(1, width - 1)),
                     header_height + round(y * (image_height - 1) / max(1, height - 1)))
                    for x, y in corners
                ]
                yellow = (0, 255, 255)
                cv2.polylines(preview, [np.int32(vertices)], len(vertices) == 4, yellow, 2, cv2.LINE_AA)
                for label, (x, y) in zip(corner_labels, vertices):
                    cv2.circle(preview, (x, y), 4, yellow, -1, cv2.LINE_AA)
                    cv2.putText(preview, label, (x + 6, max(header_height + 15, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, yellow, 1, cv2.LINE_AA)
            for number, line in enumerate(lines):
                cv2.putText(preview, line, (8, 19 + number * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return [], None, None
            if key in (10, 13):
                if corners or full_video:
                    if len(corners) != len(corner_labels):
                        error = ("Click all four corners to export the full video." if full_video else
                                 "Click all four corners, or press R to continue without points.")
                        continue
                    try:
                        rectangle_points(corners, calibration_size)
                    except ValueError as exc:
                        error = str(exc)
                        continue
                return marked or [displayed], calibration_frame, tuple(corners) if corners else None
            if key in (ord("r"), ord("R")) and mark_corners:
                corners.clear()
                calibration_frame = None
                calibration_size = None
                error = ""
            if key == ord(" ") and not full_video:
                if displayed in marked:
                    marked.remove(displayed)
                else:
                    marked.append(displayed)
            if key in (ord("a"), ord("A"), ord("d"), ord("D")):
                step = -1 if key in (ord("a"), ord("A")) else 1
                cv2.setTrackbarPos(trackbar, window, min(max(displayed + step, 0), last_frame))
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def read_frames(source, indices):
    if not source.is_file():
        raise ValueError(f"Input does not exist: {source}")
    if source.suffix.lower() in IMAGE_SUFFIXES:
        if indices != [0]:
            raise ValueError("A still image only has frame 0")
        frame = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode image: {source}")
        yield 0, frame
        return
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0 and max(indices) >= frame_count:
            raise ValueError(f"Video only contains {frame_count} frames")
        for index in indices:
            yield index, read_video_frame(capture, index)
    finally:
        capture.release()


def rectangle_points(values, frame_size):
    points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    if (points.shape != (4, 2) or not np.isfinite(points).all()
            or (points < 0).any() or (points > np.asarray(frame_size)-1).any()):
        raise ValueError("Select four points within the image")
    if not cv2.isContourConvex(points) or cv2.contourArea(points, oriented=True) <= 1:
        raise ValueError("Select the four rectangle corners clockwise in TL, TR, BR, BL order")
    return points


def load_calibration(args, frame, index):
    height, width = frame.shape[:2]
    source_size = [width, height]
    if args.calibration is None:
        points = args.points
        calibration = {"image_size": source_size, "output_size": args.size,
                       "reference_frame": index, "reference_source": str(args.source.resolve())}
    else:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8-sig"))
        if not isinstance(calibration, dict) or calibration.get("image_size") != source_size:
            raise ValueError("Calibration image size does not match the source")
        points = calibration.get("source_points_xy", [])
    size = args.size if args.size is not None else calibration.get("output_size")
    if (not isinstance(size, list) or len(size) != 2
            or any(type(dimension) is not int or dimension < 2 for dimension in size)):
        raise ValueError("Output width and height must be integers of at least 2 pixels")
    points = rectangle_points(points, source_size)
    out_width, out_height = size
    target = np.float32([[0, 0], [out_width-1, 0], [out_width-1, out_height-1], [0, out_height-1]])
    transform = cv2.getPerspectiveTransform(points, target)
    calibration.update(output_size=size, source_points_xy=points.tolist(), homography=transform.tolist())
    return calibration, transform


def read_slot_centers(path, indices):
    frames = {}
    requested = set(indices)
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            index = row["source_frame"]
            if index in requested:
                frames[index] = row
                if len(frames) == len(requested):
                    break
    if missing := requested-frames.keys():
        raise ValueError(f"Slot log has no data for video frames: {sorted(missing)}")
    return frames


def overlay_slots(image, row, transform):
    marks = []
    for slot in row["slots"]:
        center = slot.get("center_xy")
        if center is None:
            continue
        point = transform @ np.array([*center, 1.])
        if not np.isfinite(point).all() or abs(point[2]) < 1e-9:
            continue
        x, y = (point[:2]/point[2]).tolist()
        predicted = slot.get("position_is_prediction", False)
        observed = (not slot.get("position_is_held", False)
                    and slot.get("observation") in ("empty", "filled"))
        kind = "predicted" if predicted else (
            "observed" if observed else "unverified")
        mark = {"slot_id": slot.get("slot_id"), "byte_track_id": slot.get("byte_track_id"),
                "source_xy": center, "top_down_xy": [round(x, 2), round(y, 2)], "kind": kind}
        marks.append(mark)
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue
        x, y = round(x), round(y)
        label = f"S{mark['slot_id']}" if mark["slot_id"] is not None else f"BT{mark['byte_track_id']}"
        if kind != "observed":
            label += " PRED" if predicted else " ?"
        color = {"observed": (60, 240, 60), "predicted": (0, 230, 255), "unverified": (180, 180, 180)}[kind]
        cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (x+5, max(14, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return {"source_frame": row["source_frame"], "time_seconds": row.get("time_seconds"),
            "group_id": row.get("group_id"), "phase": row.get("phase"), "slots": marks}


def write_png(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    data.tofile(path)


def map_plane_points(points, transform):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Point coordinates must be a finite N x 2 array")
    mapped = np.column_stack((points, np.ones(len(points)))) @ np.asarray(transform).T
    if not np.isfinite(mapped).all() or np.any(np.abs(mapped[:, 2]) < 1e-9):
        raise ValueError("Perspective projection crosses the plane horizon")
    return mapped[:, :2]/mapped[:, 2:]


def rotate_slot_centers(centers, machine_center, transform, degrees):
    """Rotate in the calibrated plane; positive angles are clockwise on the image."""
    top = map_plane_points(centers, transform)
    pivot, = map_plane_points([machine_center], transform)
    radians = np.radians(degrees)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    rotated = (top-pivot) @ rotation.T+pivot
    raw = map_plane_points(rotated, np.linalg.inv(transform))
    return top, pivot, rotated, raw


def find_rotation_slot_states(source, configured=None):
    if configured is not None:
        path = Path(configured)
        if not path.is_file():
            raise ValueError(f"Slot log does not exist: {path}")
        candidates = [path]
    else:
        source = Path(source).resolve()
        roots = {source.parent, source.parent.parent, source.parent.parent / "regression_runs"}
        candidates = [path for root in roots
                      for path in root.glob(f"{source.stem}_slots_*/slot_states.jsonl")]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        summary_path = path.with_name("summary.json")
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            recorded = summary.get("source")
            if recorded and Path(recorded).resolve() != Path(source).resolve():
                if configured is not None:
                    raise ValueError("Slot log belongs to a different source video")
                continue
        elif configured is None:
            continue
        return path
    raise ValueError("No matching slot_states.jsonl found. Set SLOT_STATES_FILE or --slot-states.")


def measured_slots(row):
    return [slot for slot in row.get("slots", []) if slot.get("center_xy") is not None
            and slot.get("observation") in ("empty", "filled")
            and not slot.get("geometry_inferred", False)
            and not slot.get("position_is_prediction", False) and not slot.get("position_is_held", False)
            and slot.get("byte_track_id") is not None]


def rotation_test_rows(path, reference_index=None, target_indices=None):
    if reference_index is not None and reference_index < 0:
        raise ValueError("Rotation reference frame must be nonnegative")
    if target_indices is not None:
        validate_frames(target_indices)
    reference, targets, stable = None, {}, []
    requested = set(target_indices or ())
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            slots = measured_slots(row)
            quartet = (row.get("group_id") == 1 and row.get("phase") == "unload"
                and row.get("slot_numbering_status") == "final" and len(slots) == 4
                and {slot.get("slot_id") for slot in slots} == {1, 2, 3, 4}
                and len({slot["byte_track_id"] for slot in slots}) == 4)
            if reference_index is not None and row["source_frame"] == reference_index:
                if not quartet:
                    raise ValueError("Rotation reference needs four measured G1 unload slots with formal S1-S4")
                reference = row
            elif reference_index is None and reference is None:
                if quartet and row.get("group_status") == "active":
                    slots.sort(key=lambda slot: slot["slot_id"])
                    points = np.asarray([slot["center_xy"] for slot in slots])
                    key = tuple(slot["byte_track_id"] for slot in slots)
                    spacing = np.min(np.linalg.norm(points[:, None]-points[None, :], axis=2)
                                     [np.triu_indices(4, 1)])
                    if (stable and (key != stable[0][0] or row["source_frame"]-stable[-1][2]["source_frame"] > 1
                            or np.max(np.linalg.norm(points-stable[0][1], axis=1)) > .1*spacing)):
                        stable = []
                    stable.append((key, points, row))
                    if len(stable) == 3:
                        reference = stable[1][2]
                else:
                    stable = []
            if row["source_frame"] in requested:
                targets[row["source_frame"]] = row
            elif target_indices is None and not targets and row.get("phase") == "load" \
                    and row.get("group_id") == 1 and len(slots) >= 2:
                targets[row["source_frame"]] = row
            if reference is not None and targets and (target_indices is None or targets.keys() == requested):
                break
    if reference is None:
        raise ValueError("No stable measured G1 unloading quartet found; specify --rotation-reference-frame")
    if not targets or requested-targets.keys():
        raise ValueError(f"Load comparison frames unavailable: {sorted(requested-targets.keys())}")
    if any(row.get("phase") != "load" or row.get("group_id") != 1 for row in targets.values()):
        raise ValueError("Rotation target frames must belong to G1 loading")
    return reference, [targets[index] for index in sorted(targets)]


def draw_plane_marks(image, points, labels, color, *, outline=False):
    for point, label in zip(points, labels):
        x, y = np.rint(point).astype(int)
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 7 if outline else 4, color, 2 if outline else -1, cv2.LINE_AA)
            width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)[0][0]
            text_x = max(1, min(x+8, image.shape[1]-width-2))
            cv2.putText(image, label, (text_x, max(14, y-7)), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)


def expanded_plane_view(image, *point_groups):
    """Include the machine center and out-of-crop predictions in the diagnostic view."""
    points = np.vstack(point_groups)
    low = np.minimum(np.floor(points.min(axis=0)-26), [0, 0]).astype(int)
    high = np.maximum(np.ceil(points.max(axis=0)+70), image.shape[1::-1]).astype(int)
    if np.max(high-low) > 10000:
        raise ValueError("Rotated points are too far from the calibrated area to display")
    canvas = np.full((high[1]-low[1], high[0]-low[0], 3), 40, np.uint8)
    canvas[-low[1]:image.shape[0]-low[1], -low[0]:image.shape[1]-low[0]] = image
    return canvas, -low


def rotation_comparison(raw, top, title):
    header, gap = 46, 12
    height = max(raw.shape[0], top.shape[0])
    canvas = np.full((height+header, raw.shape[1]+gap+top.shape[1], 3), 25, np.uint8)
    for image, left in ((raw, 0), (top, raw.shape[1]+gap)):
        y = header+(height-image.shape[0])//2
        canvas[y:y+image.shape[0], left:left+image.shape[1]] = image
    cv2.putText(canvas, title, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, .49, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Green: observed  Yellow: pure orbit prediction  Blue: unload reference",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def export_rotation_test(args):
    """Read saved detections and export fixed-angle predictions without business-pipeline calls."""
    validate_options(args)
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    angle = float(args.rotation_degrees)
    pivot = np.asarray(args.machine_center, dtype=float)
    if not np.isfinite(angle) or not 0 < abs(angle) <= 360 or pivot.shape != (2,) or not np.isfinite(pivot).all():
        raise ValueError("Rotation needs a finite angle in (0, 360] and a finite machine center")
    states = find_rotation_slot_states(args.source, getattr(args, "slot_states", None))
    reference, targets = rotation_test_rows(states, getattr(args, "rotation_reference_frame", None),
                                           getattr(args, "rotation_target_frames", None))
    indices = list(dict.fromkeys([reference["source_frame"], *(row["source_frame"] for row in targets)]))
    with closing(read_frames(args.source, indices)) as reader:
        frames = dict(reader)
    calibration, transform = load_calibration(args, frames[reference["source_frame"]], reference["source_frame"])
    slots = sorted(reference["slots"], key=lambda slot: slot["slot_id"])
    centers = np.asarray([slot["center_xy"] for slot in slots])
    labels = [f"S{slot['slot_id']}" for slot in slots]
    direction = args.rotation_direction
    angles = ([abs(angle), -abs(angle)] if direction == "both" else
              [abs(angle) if direction == "clockwise" else -abs(angle)])
    report = {"source": str(args.source.resolve()), "slot_states": str(states.resolve()),
        "reference_frame": reference["source_frame"], "reference_time_seconds": reference.get("time_seconds"),
        "machine_center_source_xy": pivot.tolist(), "model": "pure_rotation_about_machine_center",
        "angle_convention": "positive clockwise in calibrated image coordinates",
        "translation_correction": False, "slot_labels": labels, "reference_source_xy": centers.tolist(),
        "calibrated_output_size": calibration["output_size"], "predictions": []}
    args.output_dir.mkdir(parents=True)
    ref = frames[reference["source_frame"]].copy()
    top_ref = cv2.warpPerspective(ref, transform, tuple(calibration["output_size"]))
    ref_top_points = map_plane_points(centers, transform)
    machine_top, = map_plane_points([pivot], transform)
    top_ref, reference_offset = expanded_plane_view(top_ref, ref_top_points, [machine_top])
    draw_plane_marks(ref, centers, labels, (255, 160, 40))
    draw_plane_marks(top_ref, ref_top_points+reference_offset, labels, (255, 160, 40))
    draw_plane_marks(ref, [pivot], ["MACHINE"], (220, 60, 220), outline=True)
    draw_plane_marks(top_ref, [machine_top+reference_offset], ["MACHINE"], (220, 60, 220), outline=True)
    write_png(args.output_dir/"rotation_unload_reference.png", rotation_comparison(ref, top_ref,
        f"G1 unload reference frame {reference['source_frame']}"))
    for degrees in angles:
        _, machine_top, predicted_top, predicted_raw = rotate_slot_centers(centers, pivot, transform, degrees)
        entry = {"clockwise_degrees": degrees, "reference_top_down_xy": ref_top_points.tolist(),
            "machine_center_top_down_xy": machine_top.tolist(), "predicted_top_down_xy": predicted_top.tolist(),
            "predicted_source_xy": predicted_raw.tolist(), "comparisons": []}
        for row in targets:
            index = row["source_frame"]
            raw = frames[index].copy()
            top = cv2.warpPerspective(raw, transform, tuple(calibration["output_size"]))
            observed = measured_slots(row)
            observations = np.asarray([slot["center_xy"] for slot in observed]).reshape(-1, 2)
            observed_top = map_plane_points(observations, transform)
            top, top_offset = expanded_plane_view(top, predicted_top, observed_top, [machine_top])
            observation_labels = [f"BT{slot['byte_track_id']}" for slot in observed]
            draw_plane_marks(raw, observations, observation_labels, (60, 240, 60))
            draw_plane_marks(top, observed_top+top_offset, observation_labels, (60, 240, 60))
            draw_plane_marks(raw, predicted_raw, labels, (0, 230, 255), outline=True)
            draw_plane_marks(top, predicted_top+top_offset, labels, (0, 230, 255), outline=True)
            draw_plane_marks(raw, [pivot], ["MACHINE"], (220, 60, 220), outline=True)
            draw_plane_marks(top, [machine_top+top_offset], ["MACHINE"], (220, 60, 220), outline=True)
            name = f"rotation_{'cw' if degrees > 0 else 'ccw'}_{abs(degrees):g}_frame_{index:06d}.png"
            write_png(args.output_dir/name, rotation_comparison(raw, top,
                f"G1 load frame {index} | {'CW' if degrees > 0 else 'CCW'} {abs(degrees):g} deg about machine"))
            distances = np.linalg.norm(observations[:, None]-predicted_raw[None, :], axis=2)
            nearest = distances.argmin(axis=1) if len(observed) else []
            entry["comparisons"].append({"source_frame": index, "time_seconds": row.get("time_seconds"),
                "image": name, "observed_slots": [{"byte_track_id": slot["byte_track_id"],
                    "source_xy": slot["center_xy"], "nearest_predicted_slot_id": int(best)+1,
                    "nearest_distance_px": float(distances[i, best])}
                    for i, (slot, best) in enumerate(zip(observed, nearest))],
                "predictions_inside_image": [bool(0 <= x < raw.shape[1] and 0 <= y < raw.shape[0])
                                             for x, y in predicted_raw]})
        report["predictions"].append(entry)
    (args.output_dir/"calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (args.output_dir/"rotation_test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                               allow_nan=False), encoding="utf-8")
    print(f"旋转测试参考帧 {reference['source_frame']}，对照帧 {[row['source_frame'] for row in targets]}")
    return args.output_dir.resolve()


def export_previews(args, *, reference_frame=None):
    validate_options(args)
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    slot_frames = read_slot_centers(args.slot_states, args.frames) if args.slot_states else {}
    frames = read_frames(args.source, args.frames)
    try:
        first_index, first_frame = next(frames)
        if reference_frame is None or reference_frame == first_index:
            calibration, transform = load_calibration(args, first_frame, first_index)
        else:
            with closing(read_frames(args.source, [reference_frame])) as reference:
                reference_index, reference_image = next(reference)
                calibration, transform = load_calibration(args, reference_image, reference_index)
        args.output_dir.mkdir(parents=True)
        marks = []
        for index, frame in itertools.chain(((first_index, first_frame),), frames):
            if list(frame.shape[1::-1]) != calibration["image_size"]:
                raise ValueError(f"Frame {index} changed size")
            warped = cv2.warpPerspective(frame, transform, tuple(calibration["output_size"]),
                                         flags=cv2.INTER_CUBIC)
            suffix = f"frame_{index:06d}"
            write_png(args.output_dir/f"top_down_{suffix}.png", warped)
            if index in slot_frames:
                marks.append(overlay_slots(warped, slot_frames[index], transform))
                write_png(args.output_dir/f"top_down_{suffix}_slots.png", warped)
        (args.output_dir/"calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        if marks:
            (args.output_dir/"rectified_slot_centers.json").write_text(
                json.dumps(marks, indent=2, allow_nan=False), encoding="utf-8")
    finally:
        frames.close()
    return args.output_dir.resolve()


def find_ffmpeg(configured):
    if configured:
        path = shutil.which(str(configured))
        if path is None:
            raise ValueError(f"Cannot find FFmpeg: {configured}")
        return path
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


@contextmanager
def open_video_output(path, size, fps, *, executable, crf):
    if executable is None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        try:
            if not writer.isOpened():
                raise ValueError(f"Cannot create video: {path.name}")
            yield writer.write
        finally:
            writer.release()
        return

    width, height = size
    rate = str(Fraction(str(fps)).limit_denominator(1_000_000))
    command = [
        executable, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", rate, "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    with TemporaryFile() as errors, subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as process:
        def encoder_error():
            with suppress(OSError):
                process.stdin.close()
            process.wait()
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace").strip()
            return ValueError(f"FFmpeg could not encode {path.name}: {detail or 'closed the input pipe'}")

        def write_frame(image):
            try:
                process.stdin.write(memoryview(image).cast("B"))
            except OSError as exc:
                raise encoder_error() from exc

        try:
            yield write_frame
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            with suppress(OSError):
                process.stdin.close()
            raise
        try:
            process.stdin.close()
        except OSError as exc:
            raise encoder_error() from exc
        if process.wait() != 0:
            raise encoder_error()


def export_full_video(args, *, reference_frame=None, ffmpeg=None, crf=VIDEO_CRF):
    validate_options(args)
    if args.source.suffix.lower() in IMAGE_SUFFIXES:
        raise ValueError("整段导出需要视频文件。")
    if not args.source.is_file():
        raise ValueError(f"Input does not exist: {args.source}")
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    if type(crf) is not int or not 0 <= crf <= 51:
        raise ValueError("H.264 CRF must be an integer between 0 and 51")
    executable = find_ffmpeg(ffmpeg)

    capture = cv2.VideoCapture(str(args.source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {args.source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot determine the source video frame rate")
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = args.frames[0] if reference_frame is None else reference_frame
        if args.calibration is not None:
            index = 0
        reference = read_video_frame(capture, index)
        calibration, transform = load_calibration(args, reference, index)
        frame = reference if index == 0 else read_video_frame(capture, 0)

        raw_width, raw_height = calibration["image_size"]
        top_width, top_height = calibration["output_size"]
        encoder_name = f"H.264，CRF {crf}" if executable else "OpenCV MPEG-4（未找到 FFmpeg）"
        print(f"俯视尺寸 {top_width}×{top_height}，视频编码 {encoder_name}", flush=True)
        top_size = (top_width + top_width % 2, top_height + top_height % 2)
        header, gap = 36, 12
        view_height = max(raw_height, top_height)
        compare_width = raw_width + gap + top_width
        compare_height = header + view_height
        compare_size = (compare_width + compare_width % 2, compare_height + compare_height % 2)
        canvas = np.full((compare_size[1], compare_size[0], 3), 25, dtype=np.uint8)
        raw_y = header + (view_height - raw_height) // 2
        top_y = header + (view_height - top_height) // 2
        top_x = raw_width + gap
        for x, label in ((10, "Original"), (top_x + 10, "Top-down")):
            cv2.putText(canvas, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        .60, (240, 240, 240), 1, cv2.LINE_AA)

        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".slot-video-", dir=args.output_dir.parent) as temp:
            result = Path(temp) / "export"
            result.mkdir()
            outputs = [(result / "top_down.mp4", top_size),
                       (result / "original_vs_top_down.mp4", compare_size)]
            with ExitStack() as writers_to_close:
                write_top, write_comparison = (
                    writers_to_close.enter_context(open_video_output(path, size, fps, executable=executable, crf=crf))
                    for path, size in outputs
                )
                count = 0
                while frame is not None:
                    if frame.shape[:2] != (raw_height, raw_width):
                        raise ValueError(f"Frame {count} changed size")
                    top = cv2.warpPerspective(frame, transform, (top_width, top_height), flags=cv2.INTER_CUBIC)
                    encoded_top = top
                    if top_size != (top_width, top_height):
                        encoded_top = cv2.copyMakeBorder(top, 0, top_height % 2, 0, top_width % 2,
                                                         cv2.BORDER_CONSTANT)
                    canvas[raw_y:raw_y + raw_height, :raw_width] = frame
                    canvas[top_y:top_y + top_height, top_x:top_x + top_width] = top
                    write_top(encoded_top)
                    write_comparison(canvas)
                    count += 1
                    if count % 300 == 0:
                        print(f"已处理 {count}/{expected_frames or '?'} 帧", flush=True)
                    ok, frame = capture.read()
                    if not ok:
                        break
            if expected_frames > 0 and count < expected_frames:
                raise ValueError(f"Video ended at frame {count}, expected {expected_frames} frames")
            for path, _ in outputs:
                check = cv2.VideoCapture(str(path))
                try:
                    if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
                        raise ValueError(f"Exported frame count does not match the source: {path.name}")
                finally:
                    check.release()
            (result / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            result.rename(args.output_dir)
    finally:
        capture.release()
    print(f"整段视频导出完成，共 {count} 帧", flush=True)
    return args.output_dir.resolve()


def run_configured():
    if not SOURCE:
        raise ValueError("请在脚本顶部的 SOURCE 填写图片或视频路径。")
    source = Path(SOURCE).resolve()
    rotation_test = ROTATION_TEST_DEGREES is not None
    full_video = EXPORT_FULL_VIDEO and not rotation_test
    points = CALIBRATION_POINTS
    reference_frame = None
    needs_corners = not CALIBRATION_FILE and points is None
    if full_video:
        if source.suffix.lower() in IMAGE_SUFFIXES:
            raise ValueError("整段导出需要视频文件。")
        if not CALIBRATION_FILE and OUTPUT_SIZE is None:
            raise ValueError("整段导出前，请填写 CALIBRATION_FILE 或按实物宽高比设置 OUTPUT_SIZE。")
        if needs_corners and not SELECT_FRAMES_IN_WINDOW:
            raise ValueError("整段导出需要标定，请填写标定文件或开启选帧窗口点选四角。")
    if CALIBRATION_POINTS is not None and not CALIBRATION_FILE and OUTPUT_SIZE is None:
        raise ValueError("请在脚本顶部填写 OUTPUT_SIZE，宽高比应与实体矩形一致。")
    if source.suffix.lower() in IMAGE_SUFFIXES:
        indices = [0]
    elif SELECT_FRAMES_IN_WINDOW and (not full_video or needs_corners):
        initial = FRAME_NUMBERS[0] if FRAME_NUMBERS else 0
        indices, reference_frame, clicked_points = select_video_frames(
            source, initial, mark_corners=needs_corners, full_video=full_video,
        )
        if not indices:
            print("已取消选帧。")
            return
        if clicked_points is not None:
            points = clicked_points
            print(f"标定参考帧：{reference_frame}")
            print(f"CALIBRATION_POINTS = {points!r}")
    else:
        indices = [0] if full_video or rotation_test else list(FRAME_NUMBERS)
    validate_frames(indices)
    if not full_video and not rotation_test:
        print(f"选定帧号：{', '.join(map(str, indices))}")
    parent = Path(OUTPUT_FOLDER) if OUTPUT_FOLDER else source.parent / "slot_perspective"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = parent / f"{source.stem}_{stamp}"
    if not CALIBRATION_FILE and (points is None or OUTPUT_SIZE is None):
        frames = read_frames(source, indices)
        try:
            for index, frame in frames:
                output_dir.mkdir(parents=True, exist_ok=True)
                reference = output_dir / f"calibration_reference_frame_{index:06d}.png"
                write_png(reference, frame)
                print(f"已导出原始参考帧：{reference.resolve()}")
        finally:
            frames.close()
        if points is None:
            print("请在窗口点选四角或填写 CALIBRATION_POINTS，再按实物宽高比设置 OUTPUT_SIZE。")
        else:
            print("请把上方坐标填入 CALIBRATION_POINTS，再按实物宽高比填写 OUTPUT_SIZE，即可生成俯视图。")
        return

    args = argparse.Namespace(
        source=source, frames=indices, points=points,
        size=list(OUTPUT_SIZE) if OUTPUT_SIZE is not None else None,
        output_dir=output_dir,
        calibration=Path(CALIBRATION_FILE) if CALIBRATION_FILE else None,
        slot_states=Path(SLOT_STATES_FILE) if SLOT_STATES_FILE else None,
        rotation_degrees=ROTATION_TEST_DEGREES, rotation_direction=ROTATION_TEST_DIRECTION,
        machine_center=MACHINE_CENTER_XY, rotation_reference_frame=ROTATION_REFERENCE_FRAME,
        rotation_target_frames=list(ROTATION_TARGET_FRAMES) if ROTATION_TARGET_FRAMES is not None else None,
    )
    if rotation_test:
        result = export_rotation_test(args)
        print(f"绕机心旋转测试已保存到：{result}")
    elif full_video:
        result = export_full_video(args, reference_frame=reference_frame, ffmpeg=FFMPEG_PATH, crf=VIDEO_CRF)
        print(f"俯视视频和左右对照视频已保存到：{result}")
    else:
        result = export_previews(args, reference_frame=reference_frame)
        print(f"已导出 {len(indices)} 帧俯视图：{result}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        try:
            run_configured()
        except (ValueError, TypeError, KeyError, OSError, cv2.error) as exc:
            raise SystemExit(f"运行失败：{exc}") from None
        return

    args, parser = parse_args(argv)
    try:
        if args.rotation_degrees is not None:
            output_dir = export_rotation_test(args)
        elif args.full_video:
            output_dir = export_full_video(args, ffmpeg=args.ffmpeg, crf=args.crf)
        else:
            output_dir = export_previews(args)
    except (ValueError, TypeError, KeyError, OSError, cv2.error) as exc:
        parser.error(str(exc))
    print(f"Saved top-down previews to {output_dir}")


if __name__ == "__main__":
    main()
