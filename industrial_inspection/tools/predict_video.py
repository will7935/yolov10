"""Run YOLOv10 on one local video and stream detections to disk."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import torch

from industrial_inspection.tools.slot_roi import draw_roi, filter_slot_indices, load_roi, select_roi
from industrial_inspection.tools.detection_display import draw_detection_points


# =============================================================================
# 用户配置：在这里修改，然后在 PyCharm / VS Code 中直接运行此文件。
# Windows 路径请保留 r 前缀，避免反斜杠被解释为转义字符。
# =============================================================================
SOURCE_VIDEO = Path(r"I:\youyan\test_dataset\camera_B_20260924_021033.avi")
MODEL_WEIGHTS = Path(r"C:\Users\win10\xanylabeling_data\trainer\ultralytics\runs\detect\exp3.3\weights\best.pt")

# 每次在此目录内新建“视频名_时间”文件夹，可重复运行而不覆盖结果。
OUTPUT_ROOT = Path(r"I:\youyan\test_dataset")
# 例如改为：OUTPUT_ROOT = Path(r"I:\youyan\video_results")

CONFIDENCE = 0.5
IMAGE_SIZE = 1280
DEVICE = "0"          # "auto" 自动选择 GPU/CPU；"0" 使用 GPU 0；"cpu" 使用 CPU
FRAME_STRIDE = 1         # 1 = 逐帧；2 = 每两帧处理一帧
MAX_FRAMES = 0           # 0 = 整段视频；300 = 只处理 300 帧试跑
SHOW_PREVIEW = False     # True = 显示检测窗口，按 Q / Escape 停止
DISPLAY_STYLE = "point" # "point"：中心点＋标注；"box"：原来的矩形框＋标注
SHOW_LABELS = True       # True：显示类别名；False：只画点/框，同时隐藏置信度文字
SHOW_CONFIDENCE = True   # 显示标签时，True：附带置信度；False：只显示类别名
POINT_RADIUS = 4         # 点半径，单位像素
POINT_FONT_SCALE = 0.4   # 点标注文字大小
# 点模式使用简短名称；修改右侧显示名即可，JSON/CSV 仍记录模型原类别名。
POINT_LABEL_NAMES = {"hand": "hand", "hatch_handle": "handle", "water_gun": "gun",
                     "wafer_slot_empty": "empty", "wafer_slot_filled": "filled"}
NMS_IOU = 0.70          # YOLOv8 同类别 NMS；YOLOv10 的原生后处理不使用此阈值
SLOT_DEDUPLICATE = True # 对空槽/满槽一起去重，仅保留高度重叠框中分数较高者
SLOT_DUPLICATE_IOU = 0.85 # 越低越容易误删相邻槽；建议先保持保守的 0.85
SLOT_ROI_ENABLED = True   # True：只输出所选区域内的槽位；其他类别仍检测全画面
# 同一视频自动复用标定文件；换视频默认重新圈选，避免套用错误门体位置。
SLOT_ROI_FILE = None      # None：使用视频旁的 <视频名>_slot_roi.json
ROI_RESELECT = False     # True：本次重新圈选，覆盖该视频保存的 ROI
ROI_SELECT_FRAME = 0     # 初始预览帧号；窗口滑块可选槽位清楚、舱门打开的一帧
DRAW_SLOT_ROI = True      # 在输出视频中画出区域边界
# =============================================================================


def load_detection_model(weights):
    """Select the correct predictor from the actual checkpoint architecture."""
    from ultralytics import YOLO, YOLOv10
    from ultralytics.nn.tasks import YOLOv10DetectionModel

    model = YOLO(str(weights.resolve()))
    if isinstance(model.model, YOLOv10DetectionModel) and not isinstance(model, YOLOv10):
        model = YOLOv10(str(weights.resolve()))
    architecture = "YOLOv10" if isinstance(model, YOLOv10) else "YOLO"
    print(f"Loaded {architecture} with {type(model.model).__name__}", flush=True)
    return model, architecture


def slot_keep_indices(boxes, names, threshold):
    """Suppress only almost-identical slot boxes across empty/filled classes.

    Hands/guns/handles can overlap valid slots and must not suppress them.
    This is geometric deduplication, not an occupancy correctness guarantee.
    """
    slot_ids = {int(i) for i, name in names.items()
                if name in {"wafer_slot_empty", "wafer_slot_filled"}}
    classes = boxes.cls.tolist()
    coords = boxes.xyxy.tolist()
    scores = boxes.conf.tolist()
    candidates = sorted((i for i, cls in enumerate(classes) if int(cls) in slot_ids),
                        key=lambda i: scores[i], reverse=True)
    kept_slots = []
    suppressed = []
    for i in candidates:
        a = coords[i]
        for j in kept_slots:
            b = coords[j]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
            if union > 0 and intersection / union > threshold:
                suppressed.append(i)
                break
        else:
            kept_slots.append(i)
    removed = set(suppressed)
    return [i for i in range(len(boxes)) if i not in removed], suppressed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_VIDEO, help="Input video file")
    parser.add_argument("--weights", type=Path, default=MODEL_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, help="New directory; existing directories are never overwritten")
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE, help="Process every Nth frame; output FPS is reduced accordingly")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Maximum processed frames, 0 means entire video")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW_PREVIEW,
                        help="Show preview; press Q or Escape to stop")
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=SHOW_LABELS,
                        help="Draw class labels; when disabled, draw markers without text")
    parser.add_argument("--show-confidence", action=argparse.BooleanOptionalAction, default=SHOW_CONFIDENCE,
                        help="Include confidence in visible class labels")
    parser.add_argument("--display-style", choices=("box", "point"), default=DISPLAY_STYLE)
    parser.add_argument("--point-radius", type=int, default=POINT_RADIUS)
    parser.add_argument("--point-font-scale", type=float, default=POINT_FONT_SCALE)
    parser.add_argument("--slot-deduplicate", action=argparse.BooleanOptionalAction, default=SLOT_DEDUPLICATE)
    parser.add_argument("--slot-duplicate-iou", type=float, default=SLOT_DUPLICATE_IOU)
    parser.add_argument("--slot-roi", action=argparse.BooleanOptionalAction, default=SLOT_ROI_ENABLED)
    parser.add_argument("--roi-file", type=Path, default=SLOT_ROI_FILE)
    parser.add_argument("--reselect-roi", action=argparse.BooleanOptionalAction, default=ROI_RESELECT)
    parser.add_argument("--roi-select-frame", type=int, default=ROI_SELECT_FRAME)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    if not 0 < args.conf < 1 or args.stride < 1 or args.max_frames < 0 or args.imgsz < 32:
        parser.error("Require 0 < conf < 1, stride >= 1, max-frames >= 0, imgsz >= 32")
    if not 0 < args.nms_iou < 1 or not 0 < args.slot_duplicate_iou < 1:
        parser.error("NMS and duplicate IoU thresholds must lie between 0 and 1")
    if args.display_style not in {"box", "point"} or not 1 <= args.point_radius <= 20 or not .1 <= args.point_font_scale <= 2:
        parser.error("Require display-style box/point, point-radius 1..20, point-font-scale 0.1..2")
    if not args.source.is_file() or not args.weights.is_file():
        parser.error("Input video and weights must exist")

    # The trusted official/self-trained checkpoints in this older repository
    # contain a serialized model, not only tensors (PyTorch 2.6+ compatibility).
    original_load = torch.load

    def trusted_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = trusted_load
    model, architecture = load_detection_model(args.weights)
    cap = cv2.VideoCapture(str(args.source.resolve()))
    writer = None
    processed = 0
    read_frames = 0
    status = "failed"
    suppressed_total = 0
    roi_excluded_total = 0
    roi_polygon = None
    roi_path = args.roi_file or args.source.with_name(args.source.stem + "_slot_roi.json")
    started = time.perf_counter()
    output = args.output_dir or OUTPUT_ROOT / (
        args.source.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    summary = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Video has invalid FPS; repair the file before inference")
        source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.slot_roi:
            ok, reference = cap.read()
            if not ok:
                raise RuntimeError("Cannot read video to initialize slot ROI")
            ref_height, ref_width = reference.shape[:2]
            cap.release()
            if args.reselect_roi or not roi_path.is_file():
                select_roi(args.source, roi_path, args.roi_select_frame)
            roi_polygon = load_roi(roi_path, (ref_width, ref_height))
            cap = cv2.VideoCapture(str(args.source.resolve()))
            if not cap.isOpened():
                raise RuntimeError("Cannot reopen video after ROI setup")
            print(f"Using slot ROI: {roi_path.resolve()}", flush=True)
        output.mkdir(parents=True, exist_ok=False)
        names = dict(model.names)
        summary = {"source": str(args.source.resolve()), "weights": str(args.weights.resolve()),
                   "source_fps": fps, "source_frames_reported": source_frames,
                   "output_fps": fps / args.stride, "stride": args.stride,
                   "conf": args.conf, "imgsz": args.imgsz, "device": args.device,
                   "show_labels": args.show_labels, "show_confidence": args.show_labels and args.show_confidence,
                   "display_style": args.display_style,
                   "point_radius": args.point_radius, "point_font_scale": args.point_font_scale,
                   "point_label_names": POINT_LABEL_NAMES, "point_position": "detection_box_center",
                   "architecture": architecture, "nms_iou": args.nms_iou,
                   "native_nms": architecture != "YOLOv10",
                   "slot_deduplicate": args.slot_deduplicate, "slot_duplicate_iou": args.slot_duplicate_iou,
                   "slot_roi_enabled": args.slot_roi,
                   "slot_roi_file": str(roi_path.resolve()) if args.slot_roi else None,
                   "slot_roi_polygon_normalized": roi_polygon.tolist() if roi_polygon is not None else None,
                   "slot_roi_filter_rule": "slot_box_center_inside_polygon",
                   "classes": names, "output_video": str((output / "annotated.mp4").resolve()),
                   "timestamp_method": "zero-based source_frame / source_fps; constant-frame-rate video",
                   "audio_included": False}
        with (output / "detections.jsonl").open("w", encoding="utf-8") as json_file, (
            output / "frame_counts.csv"
        ).open("w", encoding="utf-8-sig", newline="") as csv_file:
            counts_writer = csv.DictWriter(csv_file, fieldnames=["source_frame", "time_seconds", *names.values()])
            counts_writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    status = "end_of_stream"
                    break
                index = read_frames
                read_frames += 1
                if index % args.stride:
                    continue
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / args.stride, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create MP4 video writer")
                    summary["frame_size"] = [width, height]
                result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device,
                                       iou=args.nms_iou, agnostic_nms=False, verbose=False, save=False)[0]
                raw_box_count = len(result.boxes)
                roi_excluded = []
                if roi_polygon is not None:
                    raw_boxes = result.boxes.cpu()
                    keep, excluded = filter_slot_indices(raw_boxes, names, roi_polygon, (frame.shape[1], frame.shape[0]))
                    roi_excluded = [{"class_name": names[int(raw_boxes.cls[i].item())],
                                     "confidence": float(raw_boxes.conf[i].item()),
                                     "xyxy": raw_boxes.xyxy[i].tolist()} for i in excluded]
                    roi_excluded_total += len(excluded)
                    result = result[keep]
                suppressed_objects = []
                if args.slot_deduplicate:
                    raw_boxes = result.boxes.cpu()
                    keep, suppressed = slot_keep_indices(raw_boxes, names, args.slot_duplicate_iou)
                    suppressed_objects = [{"class_name": names[int(raw_boxes.cls[i].item())],
                                           "confidence": float(raw_boxes.conf[i].item()),
                                           "xyxy": raw_boxes.xyxy[i].tolist()} for i in suppressed]
                    suppressed_total += len(suppressed)
                    result = result[keep]
                boxes = result.boxes.cpu()
                objects = [{"class_id": int(cls), "class_name": names[int(cls)],
                            "confidence": round(float(conf), 6), "xyxy": [round(float(v), 2) for v in xyxy]}
                           for xyxy, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())]
                timestamp = round(index / fps, 6)
                counts = Counter(obj["class_name"] for obj in objects)
                json_file.write(json.dumps({"source_frame": index, "time_seconds": timestamp,
                                            "raw_box_count": raw_box_count,
                                            "suppressed_slots": suppressed_objects,
                                            "roi_excluded_slots": roi_excluded,
                                            "detections": objects}, ensure_ascii=False) + "\n")
                counts_writer.writerow({"source_frame": index, "time_seconds": timestamp,
                                        **{name: counts[name] for name in names.values()}})
                if args.display_style == "point":
                    plotted = draw_detection_points(frame, objects, show_labels=args.show_labels,
                        show_confidence=args.show_confidence, radius=args.point_radius,
                        font_scale=args.point_font_scale, label_names=POINT_LABEL_NAMES)
                else:
                    plotted = result.plot(line_width=2, font_size=12,
                                          labels=args.show_labels, conf=args.show_confidence)
                if roi_polygon is not None and DRAW_SLOT_ROI:
                    draw_roi(plotted, roi_polygon)
                writer.write(plotted)
                processed += 1
                if processed == 1 or processed % 100 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"Processed {processed} frames, source time {timestamp:.2f}s, "
                          f"average {processed / elapsed:.1f} frames/s", flush=True)
                if args.show:
                    cv2.imshow("YOLO detection (Q/Escape to stop)", plotted)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        status = "preview_stopped"
                        break
                if args.max_frames and processed >= args.max_frames:
                    status = "frame_limit"
                    break
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        if summary is not None:
            summary.update(status=status, processed_frames=processed, source_frames_read=read_frames,
                           suppressed_slot_boxes=suppressed_total,
                           roi_excluded_slot_boxes=roi_excluded_total,
                           elapsed_seconds=round(time.perf_counter() - started, 3))
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if processed == 0:
        raise RuntimeError("No video frames were processed")
    print(f"Finished ({status}). Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
