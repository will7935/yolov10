"""v2 运行入口：实时 YOLO、检测缓存回放、视频标注和汇总输出。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2

from .audit import write_audit
from .core import ProcessConfig, ProcessPipeline
from .tracking import CarrierTracker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = Path(r"C:\Users\win10\xanylabeling_data\trainer\ultralytics\runs\detect\exp4.2\weights\best.pt")
DEFAULT_VIDEO = Path(r"I:\youyan\test_dataset\测试视频4\camera_B_20260924_161009.avi")


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _objects(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row.get("objects", row.get("detections", []))


def _draw(frame, objects: list[dict[str, Any]], result: dict[str, Any], pipeline: ProcessPipeline) -> None:
    colors = {
        "slot_group": (255, 180, 0),
        "wafer_slot_empty": (0, 220, 0),
        "wafer_slot_filled": (0, 0, 230),
        "hand": (220, 100, 220),
    }
    for obj in objects:
        if not obj.get("xyxy"):
            continue
        x1, y1, x2, y2 = [round(x) for x in obj["xyxy"]]
        name = obj.get("class_name", "")
        color = colors.get(name, (180, 180, 180))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        if name in ("slot_group", "wafer_slot_empty", "wafer_slot_filled"):
            cv2.putText(
                frame,
                f"{name}:{obj.get('confidence', 0):.2f}",
                (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )
    for carrier in result.get("carriers", []):
        x1, y1, x2, y2 = [round(x) for x in carrier["xyxy"]]
        color = (0, 170, 255) if carrier["observed"] else (100, 100, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        yolo_ids = ",".join(str(i) for i in carrier.get("yolo_track_ids", []))
        label = f"C{carrier['carrier_track_id']}" + (f" BT:{yolo_ids}" if yolo_ids else " BT:?")
        cv2.putText(frame, label + ("" if carrier["observed"] else " ?"),
                    (x1, min(frame.shape[0] - 3, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    p1, p2 = pipeline.exit_p1.astype(int), pipeline.exit_p2.astype(int)
    cv2.line(frame, tuple(p1), tuple(p2), (255, 0, 255), 2)
    group = result.get("current_group") or {}
    text = f"C{result.get('carrier_track_id', '?')}  G{result.get('process_group_id', '?')} {result.get('phase', '?')}"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"box={'Y' if result.get('carrier_box_observed') else 'N'}  {group.get('status', '')}",
        (10, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    for slot in group.get("slots", []):
        if slot.get("center_xy"):
            x, y = [round(v) for v in slot["center_xy"]]
            if slot.get("baseline_center_xy"):
                bx, by = [round(v) for v in slot["baseline_center_xy"]]
                cv2.circle(frame, (bx, by), 3, (255, 255, 0), 1)
                cv2.line(frame, (bx, by), (x, y), (255, 255, 0), 1)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
            cv2.putText(
                frame,
                f"{('S' + str(slot['slot_id'])) if slot['slot_id'] is not None else ('T' + str(slot['temporary_slot_id']))}:{slot['state'] or 'unknown'}",
                (x + 3, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )


def _pipeline(args, size: tuple[int, int]) -> ProcessPipeline:
    config = ProcessConfig(confidence=args.conf, image_size=args.imgsz, circle_check_enabled=args.circle_check)
    return ProcessPipeline(size, config)


def _open_video(source):
    value = str(source)
    return cv2.VideoCapture(int(value) if value.isdecimal() else value)


def _save_summary(out, pipeline, metadata):
    summary = pipeline.finalize()
    summary.update(metadata)
    with (out / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in summary["events"]:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    _json_write(out / "group_summary.json", summary)
    write_audit(out, summary, metadata, pipeline.config)
    fields = [
        "phase",
        "group_id",
        "carrier_track_id",
        "status",
        "numbering_status",
        "started_at",
        "closed_at",
        "close_reason",
        "target_verified_count",
        "all_targets_verified",
        "slot_states",
        "unknown_slots",
    ]
    with (out / "group_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in summary["groups"]:
            row = {k: group.get(k) for k in fields}
            row["slot_states"] = [s["state"] for s in group.get("slots", [])]
            row["unknown_slots"] = [
                {"slot_id": s["slot_id"], "reason": s["reason"]}
                for s in group.get("slots", [])
                if s["state"] == "unknown"
            ]
            writer.writerow(
                {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            )
    return summary


def replay(args: argparse.Namespace) -> dict[str, Any]:
    cache = Path(args.cache)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = _open_video(args.video) if args.video else None
    if cap and not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap else args.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else args.height
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap and cap.get(cv2.CAP_PROP_FPS) else args.fps
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else None
    pipeline = _pipeline(args, (width, height))
    tracker = CarrierTracker(fps, high_thresh=args.conf, low_thresh=args.track_low_conf)
    writer = None
    frame_rows = []
    with cache.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source):
            row = json.loads(line)
            frame = None
            frame_index = int(row.get("source_frame", line_no))
            if cap and not args.no_render:
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video ended before cache frame {frame_index}")
            now = float(row.get("time_seconds", row.get("at", frame_index / fps)))
            objects = _objects(row)
            carrier_boxes = row.get("carrier_boxes")
            if carrier_boxes is None:
                carrier_boxes = tracker.update(objects)
            result = pipeline.process(objects, now, carrier_boxes)
            frame_rows.append(
                {
                    "source_frame": frame_index,
                    "time_seconds": now,
                    "objects": objects,
                    "carrier_boxes": carrier_boxes,
                    **result,
                }
            )
            if frame is not None:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                _draw(frame, objects, result, pipeline)
                writer.write(frame)
            if args.max_frames and line_no + 1 >= args.max_frames:
                break
    if cap:
        cap.release()
    if writer:
        writer.release()
    with (out / "frame_records.jsonl").open("w", encoding="utf-8") as stream:
        for row in frame_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return _save_summary(
        out,
        pipeline,
        {
            "mode": "replay",
            "cache": str(cache),
            "video": str(args.video),
            "processed_frames": len(frame_rows),
            "video_total_frames": video_total_frames,
            "video_coverage_ratio": (len(frame_rows) / video_total_frames
                                      if video_total_frames else None),
            "annotated_video_written": writer is not None,
            "fps": fps,
            "weights": str(args.weights),
            "conf": args.conf,
            "tracker_low_conf": args.track_low_conf,
            "imgsz": args.imgsz,
            "tracker": "ultralytics.ByteTrack",
            "tracked_class": "slot_group",
            "detection_provenance": "existing_cache; model settings not embedded in source rows",
        },
    )


def _load_model(weights: Path):
    import torch

    from ultralytics import YOLO

    original = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = trusted_load
    try:
        return YOLO(str(weights))
    finally:
        torch.load = original


def _parse_result(result, names: dict[int, str], only_group: bool = False) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    conf = boxes.conf.cpu().numpy().tolist()
    cls = boxes.cls.cpu().numpy().astype(int).tolist()
    ids = None if boxes.id is None else boxes.id.cpu().numpy().astype(int).tolist()
    result_rows = []
    for i, (box, score, class_id) in enumerate(zip(xyxy, conf, cls)):
        name = names.get(class_id, str(class_id))
        if only_group and name != "slot_group":
            continue
        item = {
            "class_id": int(class_id),
            "class_name": name,
            "confidence": float(score),
            "xyxy": [float(v) for v in box],
        }
        if ids is not None:
            item["track_id"] = int(ids[i])
        result_rows.append(item)
    return result_rows


def live(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = _open_video(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 18.565)
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = _load_model(Path(args.weights))
    names = (
        {int(k): str(v) for k, v in model.names.items()}
        if isinstance(model.names, dict)
        else dict(enumerate(model.names))
    )
    group_ids = [k for k, v in names.items() if v == "slot_group"]
    if not group_ids:
        raise RuntimeError(f"权重中没有 slot_group 类别，实际类别为: {names}")
    pipeline = _pipeline(args, (width, height))
    tracker = CarrierTracker(fps, high_thresh=args.conf, low_thresh=args.track_low_conf)
    writer = None
    rows = []
    with (out / "detections.jsonl").open("w", encoding="utf-8") as cache:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and index >= args.max_frames):
                break
            # Retain low-score group boxes for ByteTrack's second-stage association.
            # The process core independently filters all evidence at args.conf.
            pred = model.predict(frame, conf=args.track_low_conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            objects = _parse_result(pred, names)
            # Same built-in tracker as model.track(), with class filtering before
            # association and without running the neural network a second time.
            carrier_boxes = tracker.update(objects)
            row = {
                "source_frame": index,
                "time_seconds": index / fps,
                "objects": objects,
                "carrier_boxes": carrier_boxes,
            }
            cache.write(json.dumps(row, ensure_ascii=False) + "\n")
            process_objects = [o for o in objects if o["confidence"] >= args.conf]
            result = pipeline.process(process_objects, row["time_seconds"], carrier_boxes)
            rows.append({**row, **result})
            if not args.no_render:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                _draw(frame, process_objects, result, pipeline)
                writer.write(frame)
            index += 1
            if index % 300 == 0:
                print(f"processed {index} frames", flush=True)
    cap.release()
    if writer:
        writer.release()
    with (out / "frame_records.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return _save_summary(
        out,
        pipeline,
        {
            "mode": "live",
            "video": str(args.video),
            "weights": str(args.weights),
            "conf": args.conf,
            "tracker_low_conf": args.track_low_conf,
            "imgsz": args.imgsz,
            "processed_frames": len(rows),
            "fps": fps,
            "video_total_frames": video_total_frames,
            "video_coverage_ratio": len(rows) / video_total_frames if video_total_frames else None,
            "annotated_video_written": writer is not None,
            "tracker": "ultralytics.ByteTrack",
            "tracked_class": "slot_group",
        },
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="晶圆槽工艺检测 v2")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(s):
        s.add_argument("--video", type=str, default=str(DEFAULT_VIDEO), help="视频路径，live 模式也可传摄像头编号如 0")
        s.add_argument("--output-dir", type=Path, required=True)
        s.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
        s.add_argument("--conf", type=float, default=0.35, help="工艺检测证据及ByteTrack新轨迹阈值")
        s.add_argument("--track-low-conf", type=float, default=0.10,
                       help="ByteTrack低分框关联阈值；live推理会读取该阈值以上的检测")
        s.add_argument("--imgsz", type=int, default=1280)
        s.add_argument("--max-frames", type=int, default=0)
        s.add_argument("--circle-check", action="store_true")
        s.add_argument("--no-render", action="store_true", help="Skip video encoding during diagnostic replay")

    s = sub.add_parser("replay", help="回放 JSONL 检测缓存")
    common(s)
    s.add_argument("--cache", type=Path, required=True)
    s.add_argument("--fps", type=float, default=18.565)
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=480)
    s = sub.add_parser("live", help="YOLO 实时/视频检测")
    common(s)
    s.add_argument("--device", default="0")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary = replay(args) if args.mode == "replay" else live(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "processed_frames": summary.get("processed_frames"),
                "closed_group_count": summary["closed_group_count"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
