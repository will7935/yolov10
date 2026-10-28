"""Detect a local video with the UI7V interface and stream results to disk."""

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

# 将项目根目录加入搜索路径，便于在编辑器中直接运行脚本并导入本地模块。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import torch

from industrial_inspection.tools.slot_roi import (
    DEFAULT_SLOT_ROI_FILE, draw_roi, filter_slot_indices, load_roi, select_roi)
from industrial_inspection.tools.detection_display import draw_detection_legend, draw_detection_points


# =============================================================================
# 用户配置：在这里修改，然后在 PyCharm / VS Code 中直接运行此文件。
# Windows 路径请保留 r 前缀，避免反斜杠被解释为转义字符。
# =============================================================================
SOURCE_VIDEO = Path(r"I:\youyan\test_dataset\测试视频1\camera_B_20260924_021033.avi")
MODEL_WEIGHTS = Path(r"C:\Users\win10\xanylabeling_data\trainer\ultralytics\runs\detect\exp4.2\weights\best.pt")

# 每次在此目录内新建“视频名_时间”文件夹，可重复运行而不覆盖结果。
OUTPUT_ROOT = Path(r"I:\youyan\test_dataset")
# 例如改为：OUTPUT_ROOT = Path(r"I:\youyan\video_results")

CONFIDENCE = 0.7        # 检测分数门槛；降低可减少部分漏检，也可能增加误报，不等于准确率
AUDIT_CONFIDENCE = 0.001 # 置信度审计下限；低于显示门槛的模型候选框也写入 CSV
IMAGE_SIZE = 1280        # 推理输入尺寸；输出视频和记录坐标仍使用原图分辨率
DEVICE = "0"          # "auto" 自动选择 GPU/CPU；"0" 使用 GPU 0；"cpu" 使用 CPU
FRAME_STRIDE = 1         # 1 = 逐帧；2 = 每两帧处理一帧
MAX_FRAMES = 0           # 0 = 整段视频；300 = 只处理 300 帧试跑
UI_ENABLED = True       # True = UI7V 界面，点击开始/停止；False = 命令行批处理
SHOW_PREVIEW = False     # 仅命令行模式：True 显示 OpenCV 窗口，Q / Escape 停止
DISPLAY_STYLE = "point" # "point"：中心点＋左上角颜色图例；"box"：矩形框＋标注
SHOW_LABELS = True       # 仅框模式：True 显示类别名；False 只画框
SHOW_CONFIDENCE = True   # 仅框模式：显示标签时附带置信度
POINT_RADIUS = 4         # 点半径，单位像素
POINT_FONT_SCALE = 0.4   # 左上角图例文字大小
# 图例使用简短名称；修改右侧显示名即可，JSON/CSV 仍记录模型原类别名。
POINT_LABEL_NAMES = {"hand": "hand", "hatch_handle": "handle", "water_gun": "water_gun",
                     "wafer_slot_empty": "empty", "wafer_slot_filled": "filled"}
NMS_IOU = 0.70          # YOLOv8 同类别 NMS；YOLOv10 的原生后处理不使用此阈值
SLOT_DEDUPLICATE = True # 对空槽/满槽一起去重，仅保留高度重叠框中分数较高者
SLOT_DUPLICATE_IOU = 0.85 # 越低越容易误删相邻槽；建议先保持保守的 0.85
SLOT_ROI_ENABLED = True   # True：只输出所选区域内的槽位；其他类别仍检测全画面
# 默认使用第一次验证通过的ROI；界面可选手动划分或读取其他JSON配置。
SLOT_ROI_FILE = DEFAULT_SLOT_ROI_FILE  # 项目内 configs/slot_roi.json；None也使用此默认文件
ROI_RESELECT = False     # 仅显式设为True时重新圈选并更新所选ROI配置
ROI_SELECT_FRAME = 0     # 初始预览帧号；窗口滑块可选槽位清楚、舱门打开的一帧
DRAW_SLOT_ROI = True      # 在输出视频中画出区域边界
# =============================================================================


def load_detection_model(weights):
    """Select the correct predictor from the actual checkpoint architecture."""
    from ultralytics import YOLO, YOLOv10
    from ultralytics.nn.tasks import YOLOv10DetectionModel

    # 根据权重内部架构选择预测器，不能靠目录名或 best.pt 文件名判断版本。
    # YOLOv8 与 YOLOv10 后处理不同，加载类错误可能导致重复框。
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
    # 只对空槽/满槽做跨类别去重；手、水枪、把手与槽位重叠时仍需保留。
    slot_ids = {int(i) for i, name in names.items()
                if name in {"wafer_slot_empty", "wafer_slot_filled"}}
    classes = boxes.cls.tolist()
    coords = boxes.xyxy.tolist()
    scores = boxes.conf.tolist()
    # 高分框优先保留，后续框只与已经保留的槽位比较。
    candidates = sorted((i for i, cls in enumerate(classes) if int(cls) in slot_ids),
                        key=lambda i: scores[i], reverse=True)
    kept_slots = []
    suppressed = []
    for i in candidates:
        a = coords[i]
        for j in kept_slots:
            b = coords[j]
            # IoU = 交集面积 / 并集面积；阈值过低可能把相邻真实槽位误当成重复。
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
            if union > 0 and intersection / union > threshold:
                suppressed.append(i)
                break
        else:
            # for-else：与所有保留框比较后均未触发 break，才接纳该槽位框。
            kept_slots.append(i)
    removed = set(suppressed)
    return [i for i in range(len(boxes)) if i not in removed], suppressed


def detection_confidence_row(*, source_frame, time_seconds, index, detection, confidence_threshold,
                             displayed, display_reason, display_style):
    """Return one audit row for a model detection before display filtering."""
    x1, y1, x2, y2 = detection["xyxy"]
    return {
        "source_frame": source_frame,
        "time_seconds": time_seconds,
        "detection_index": index,
        "class_id": detection["class_id"],
        "class_name": detection["class_name"],
        "confidence": round(float(detection["confidence"]), 6),
        "x1": round(float(x1), 2),
        "y1": round(float(y1), 2),
        "x2": round(float(x2), 2),
        "y2": round(float(y2), 2),
        "meets_display_confidence": detection["confidence"] >= confidence_threshold,
        "displayed": displayed,
        "display_reason": display_reason,
        "display_style": display_style,
    }


def parse_args(argv=None):
    # 1. 参数与输入检查：直接运行时使用顶部配置，也支持命令行临时覆盖。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_VIDEO, help="Input video file")
    parser.add_argument("--weights", type=Path, default=MODEL_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, help="New directory; existing directories are never overwritten")
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--audit-conf", type=float, default=AUDIT_CONFIDENCE,
                        help="Lowest model confidence written to detection_confidences.csv")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE, help="Process every Nth frame; output FPS is reduced accordingly")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Maximum processed frames, 0 means entire video")
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=UI_ENABLED,
                        help="Use the UI7V interface; --no-ui runs the command-line batch mode")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW_PREVIEW,
                        help="Show preview; press Q or Escape to stop")
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=SHOW_LABELS,
                        help="Draw class labels in box mode; point mode always uses a color legend")
    parser.add_argument("--show-confidence", action=argparse.BooleanOptionalAction, default=SHOW_CONFIDENCE,
                        help="Include confidence in box-mode class labels")
    parser.add_argument("--display-style", choices=("box", "point"), default=DISPLAY_STYLE)
    parser.add_argument("--point-radius", type=int, default=POINT_RADIUS)
    parser.add_argument("--point-font-scale", type=float, default=POINT_FONT_SCALE)
    parser.add_argument("--slot-deduplicate", action=argparse.BooleanOptionalAction, default=SLOT_DEDUPLICATE)
    parser.add_argument("--slot-duplicate-iou", type=float, default=SLOT_DUPLICATE_IOU)
    parser.add_argument("--slot-roi", action=argparse.BooleanOptionalAction, default=SLOT_ROI_ENABLED)
    parser.add_argument("--roi-file", type=Path, default=SLOT_ROI_FILE,
                        help="Shared ROI calibration; defaults to the first validated slot ROI")
    parser.add_argument("--reselect-roi", action=argparse.BooleanOptionalAction, default=ROI_RESELECT)
    parser.add_argument("--roi-select-frame", type=int, default=ROI_SELECT_FRAME)
    args = parser.parse_args(argv)
    # 在加载模型和创建输出目录之前检查参数，便于及早定位路径或配置错误。
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    if (not 0 < args.conf < 1 or not 0 < args.audit_conf <= args.conf
            or args.stride < 1 or args.max_frames < 0 or args.imgsz < 32):
        parser.error("Require 0 < audit-conf <= conf < 1, stride >= 1, max-frames >= 0, imgsz >= 32")
    if not 0 < args.nms_iou < 1 or not 0 < args.slot_duplicate_iou < 1:
        parser.error("NMS and duplicate IoU thresholds must lie between 0 and 1")
    if args.display_style not in {"box", "point"} or not 1 <= args.point_radius <= 20 or not .1 <= args.point_font_scale <= 2:
        parser.error("Require display-style box/point, point-radius 1..20, point-font-scale 0.1..2")
    if not args.source.is_file() or not args.weights.is_file():
        parser.error("Input video and weights must exist")
    return args


def prepare_slot_roi(args):
    """Load the shared calibration; only an explicit reselection opens the GUI."""
    if not args.slot_roi:
        return None
    roi_path = Path(args.roi_file or DEFAULT_SLOT_ROI_FILE)
    cap = cv2.VideoCapture(str(args.source.resolve()))
    try:
        ok, reference = cap.read()
        if not ok:
            raise RuntimeError("Cannot read video to initialize slot ROI")
        height, width = reference.shape[:2]
    finally:
        cap.release()
    if args.reselect_roi:
        select_roi(args.source, roi_path, args.roi_select_frame)
    elif not roi_path.is_file():
        raise FileNotFoundError(f"找不到ROI配置：{roi_path}。请选择已有配置，或使用手动划分 / --reselect-roi 重新标定。")
    return load_roi(roi_path, (width, height))


def run_prediction(args, *, on_frame=None, on_log=None, should_stop=None, roi_polygon=None):
    """Process every selected frame; callbacks connect the same pipeline to Qt."""
    def log(message):
        print(message, flush=True)
        if on_log is not None:
            on_log(message)

    def stopped():
        return should_stop is not None and should_stop()

    if args.slot_roi and roi_polygon is None:
        roi_polygon = prepare_slot_roi(args)

    # 2. 加载模型：兼容 PyTorch 2.6+ 默认 weights_only=True 的变化。
    # 本项目权重包含序列化模型对象，因此仅对可信的本地权重使用完整加载。
    original_load = torch.load

    def trusted_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    log(f"Loading model: {args.weights.resolve()}")
    torch.load = trusted_load
    try:
        model, architecture = load_detection_model(args.weights)
    finally:
        torch.load = original_load
    log(f"{architecture} detection engine ready")
    # 3. 初始化视频与运行统计；read_frames 含跳过帧，processed 只计实际推理帧。
    cap = cv2.VideoCapture(str(args.source.resolve()))
    writer = None
    processed = 0
    read_frames = 0
    status = "failed"
    suppressed_total = 0
    roi_excluded_total = 0
    roi_path = Path(args.roi_file or DEFAULT_SLOT_ROI_FILE)
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
        # ROI 已在推理前准备好；区域只是位置筛选，不能自动跟随门体位置。
        if roi_polygon is not None:
            log(f"Using slot ROI: {roi_path.resolve()}")
        # 5. 创建独立结果目录，记录本次配置；已有目录不会被覆盖。
        output.mkdir(parents=True, exist_ok=False)
        names = dict(model.names)
        summary = {"source": str(args.source.resolve()), "weights": str(args.weights.resolve()),
                   "source_fps": fps, "source_frames_reported": source_frames,
                   "output_fps": fps / args.stride, "stride": args.stride,
                   "conf": args.conf, "imgsz": args.imgsz, "device": args.device,
                   "show_labels": args.display_style == "box" and args.show_labels,
                   "show_confidence": args.display_style == "box" and args.show_labels and args.show_confidence,
                   "point_legend": args.display_style == "point", "point_legend_position": "top_left",
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
                   "classes": names, "output_dir": str(output.resolve()),
                   "output_video": str((output / "annotated.mp4").resolve()),
                   "timestamp_method": "zero-based source_frame / source_fps; constant-frame-rate video",
                   "audio_included": False}
        # JSONL 每个已处理帧一行，含最终检测和被过滤的框；CSV 保存每帧类别框数。
        # 边推理边写入，避免整段视频结果驻留内存；utf-8-sig 方便 Excel 读取 CSV。
        with (output / "detections.jsonl").open("w", encoding="utf-8") as json_file, (
            output / "frame_counts.csv"
        ).open("w", encoding="utf-8-sig", newline="") as csv_file:
            counts_writer = csv.DictWriter(csv_file, fieldnames=["source_frame", "time_seconds", *names.values()])
            counts_writer.writeheader()
            # 6. 逐帧流程：读帧 → 全图检测 → ROI 筛选 → 槽位去重 → 保存与绘制。
            while True:
                if stopped():
                    status = "user_stopped"
                    break
                ok, frame = cap.read()
                if not ok:
                    status = "end_of_stream"
                    break
                index = read_frames
                read_frames += 1
                # 源帧号从 0 开始；跳帧后仍使用原帧号计算源视频时间。
                if index % args.stride:
                    continue
                # 首个处理帧确定输出尺寸；帧率除以 stride，保持采样视频的播放时长。
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / args.stride, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create MP4 video writer")
                    summary["frame_size"] = [width, height]
                # 全图检测保留手、把手、水枪；YOLOv8 在此执行同类别 NMS。
                result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device,
                                       iou=args.nms_iou, agnostic_nms=False, verbose=False, save=False)[0]
                raw_box_count = len(result.boxes)
                roi_excluded = []
                if roi_polygon is not None:
                    # 只排除中心点位于 ROI 外的槽位，其他类别不受限制；保留排除记录供复核。
                    raw_boxes = result.boxes.cpu()
                    keep, excluded = filter_slot_indices(raw_boxes, names, roi_polygon, (frame.shape[1], frame.shape[0]))
                    roi_excluded = [{"class_name": names[int(raw_boxes.cls[i].item())],
                                     "confidence": float(raw_boxes.conf[i].item()),
                                     "xyxy": raw_boxes.xyxy[i].tolist()} for i in excluded]
                    roi_excluded_total += len(excluded)
                    result = result[keep]
                suppressed_objects = []
                if args.slot_deduplicate:
                    # 对 ROI 内空/满槽再去重；高分优先仅是几何处理，不保证空满判断正确。
                    raw_boxes = result.boxes.cpu()
                    keep, suppressed = slot_keep_indices(raw_boxes, names, args.slot_duplicate_iou)
                    suppressed_objects = [{"class_name": names[int(raw_boxes.cls[i].item())],
                                           "confidence": float(raw_boxes.conf[i].item()),
                                           "xyxy": raw_boxes.xyxy[i].tolist()} for i in suppressed]
                    suppressed_total += len(suppressed)
                    result = result[keep]
                # 统一最终结果，供 JSON、CSV 和显示共用；xyxy 是原图像素坐标。
                boxes = result.boxes.cpu()
                objects = [{"class_id": int(cls), "class_name": names[int(cls)],
                            "confidence": round(float(conf), 6), "xyxy": [round(float(v), 2) for v in xyxy]}
                           for xyxy, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())]
                timestamp = round(index / fps, 6)
                # 这里统计的是单帧检测框数，尚不是逐槽稳定状态、晶圆总量或放入次数。
                counts = Counter(obj["class_name"] for obj in objects)
                json_file.write(json.dumps({"source_frame": index, "time_seconds": timestamp,
                                            "raw_box_count": raw_box_count,
                                            "suppressed_slots": suppressed_objects,
                                            "roi_excluded_slots": roi_excluded,
                                            "detections": objects}, ensure_ascii=False) + "\n")
                counts_writer.writerow({"source_frame": index, "time_seconds": timestamp,
                                        **{name: counts[name] for name in names.values()}})
                # 7. 点模式仅画中心点，类别通过左上角图例说明；不改变检测与计数。
                if args.display_style == "point":
                    plotted = draw_detection_points(frame, objects, radius=args.point_radius)
                else:
                    plotted = result.plot(line_width=2, font_size=12,
                                          labels=args.show_labels, conf=args.show_confidence)
                if roi_polygon is not None and DRAW_SLOT_ROI:
                    draw_roi(plotted, roi_polygon)
                if args.display_style == "point":
                    draw_detection_legend(plotted, names, radius=args.point_radius,
                                          font_scale=args.point_font_scale, label_names=POINT_LABEL_NAMES)
                writer.write(plotted)
                processed += 1
                if on_frame is not None:
                    on_frame(plotted, {"status": f"已处理 {processed} 帧 · 视频 {timestamp:.2f}s",
                                       "result": ", ".join(f"{name}: {count}" for name, count in counts.items()) or "未检测到目标",
                                       "frame_id": processed, "source_frame": index,
                                       "time_seconds": timestamp, "counts": dict(counts)})
                # 每 100 个处理帧报告进度，避免逐帧打印拖慢推理。
                if processed == 1 or processed % 100 == 0:
                    elapsed = time.perf_counter() - started
                    log(f"Processed {processed} frames, source time {timestamp:.2f}s, "
                        f"average {processed / elapsed:.1f} frames/s")
                # 预览按键或处理帧数上限可提前停止，结束原因写入 summary.json。
                if args.show:
                    cv2.imshow("YOLO detection (Q/Escape to stop)", plotted)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        status = "preview_stopped"
                        break
                if args.max_frames and processed >= args.max_frames:
                    status = "frame_limit"
                    break
    except KeyboardInterrupt:
        # Ctrl+C 按已处理部分收尾，视频写入器仍需释放才能完成文件封装。
        status = "interrupted"
    finally:
        # 8. 释放资源并保存摘要；异常退出时也保留已有进度，其他异常继续向外抛出。
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
    if processed == 0 and status != "user_stopped":
        raise RuntimeError("No video frames were processed")
    log(f"Finished ({status}). Results: {output.resolve()}")
    return summary


def main() -> None:
    args = parse_args()
    if args.ui:
        try:
            from industrial_inspection.tools.predict_video_ui import launch_ui
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("PyQt5"):
                raise RuntimeError("UI mode requires PyQt5: install it with 'pip install PyQt5', or use --no-ui") from exc
            raise
        launch_ui(args, run_prediction, prepare_slot_roi)
    else:
        run_prediction(args)


if __name__ == "__main__":
    # 直接运行打开 UI7V 界面；--no-ui 保留批处理，被导入时不会自动执行。
    main()
