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


# =============================================================================
# 用户配置：在这里修改，然后在 PyCharm / VS Code 中直接运行此文件。
# Windows 路径请保留 r 前缀，避免反斜杠被解释为转义字符。
# =============================================================================
SOURCE_VIDEO = Path(r"I:\youyan\test_dataset\camera_B_20260918_161258.avi")
MODEL_WEIGHTS = Path(r"C:\Users\win10\xanylabeling_data\trainer\ultralytics\runs\detect\exp3.2\weights\best.pt")

# 每次在此目录内新建“视频名_时间”文件夹，可重复运行而不覆盖结果。
OUTPUT_ROOT = Path(r"I:\youyan\test_dataset")
# 例如改为：OUTPUT_ROOT = Path(r"I:\youyan\video_results")

CONFIDENCE = 0.6
IMAGE_SIZE = 1280
DEVICE = "0"          # "auto" 自动选择 GPU/CPU；"0" 使用 GPU 0；"cpu" 使用 CPU
FRAME_STRIDE = 1         # 1 = 逐帧；2 = 每两帧处理一帧
MAX_FRAMES = 0           # 0 = 整段视频；300 = 只处理 300 帧试跑
SHOW_PREVIEW = False     # True = 显示检测窗口，按 Q / Escape 停止
NMS_IOU = 0.70          # YOLOv8 同类别 NMS；YOLOv10 的原生后处理不使用此阈值
SLOT_DEDUPLICATE = True # 对空槽/满槽一起去重，仅保留高度重叠框中分数较高者
SLOT_DUPLICATE_IOU = 0.85 # 越低越容易误删相邻槽；建议先保持保守的 0.85
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
    parser.add_argument("--slot-deduplicate", action=argparse.BooleanOptionalAction, default=SLOT_DEDUPLICATE)
    parser.add_argument("--slot-duplicate-iou", type=float, default=SLOT_DUPLICATE_IOU)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    if not 0 < args.conf < 1 or args.stride < 1 or args.max_frames < 0 or args.imgsz < 32:
        parser.error("Require 0 < conf < 1, stride >= 1, max-frames >= 0, imgsz >= 32")
    if not 0 < args.nms_iou < 1 or not 0 < args.slot_duplicate_iou < 1:
        parser.error("NMS and duplicate IoU thresholds must lie between 0 and 1")
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
        output.mkdir(parents=True, exist_ok=False)
        names = dict(model.names)
        summary = {"source": str(args.source.resolve()), "weights": str(args.weights.resolve()),
                   "source_fps": fps, "source_frames_reported": source_frames,
                   "output_fps": fps / args.stride, "stride": args.stride,
                   "conf": args.conf, "imgsz": args.imgsz, "device": args.device,
                   "architecture": architecture, "nms_iou": args.nms_iou,
                   "native_nms": architecture != "YOLOv10",
                   "slot_deduplicate": args.slot_deduplicate, "slot_duplicate_iou": args.slot_duplicate_iou,
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
                                            "detections": objects}, ensure_ascii=False) + "\n")
                counts_writer.writerow({"source_frame": index, "time_seconds": timestamp,
                                        **{name: counts[name] for name in names.values()}})
                plotted = result.plot(line_width=2, font_size=12)
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
                           elapsed_seconds=round(time.perf_counter() - started, 3))
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if processed == 0:
        raise RuntimeError("No video frames were processed")
    print(f"Finished ({status}). Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
