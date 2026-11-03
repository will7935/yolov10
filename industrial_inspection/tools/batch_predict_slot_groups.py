"""批量运行 :mod:`predict_slot_groups`，处理事件目录下的全部视频。

默认输入和输出目录对应 2026-09-24 数据集：

    python batch_predict_slot_groups.py

每个视频的结果保存为 ``<输出根目录>/<事件目录>/<视频名>/``，其中包含
``predict_slot_groups.py`` 原有的工艺报告、标注视频、JSONL 和 CSV。批次根目录
另外生成 ``batch_results.csv``、``batch_group_process_report.csv`` 及
``batch_process_report.txt``，便于一次查看所有视频的结果。

脚本默认只处理 ``camera_B`` 视频，使用已有的共享 ROI 配置并关闭交互式 ROI 选择。
如需临时切换相机，可使用 ``--camera A`` 或 ``--camera C``。若不同相机有不同
的 ROI，可通过 ``--roi-map`` 指定按相机名称映射的 JSON 文件，例如：

    {"camera_A": "...\\camera_A_slot_roi.json",
     "camera_B": "...\\slot_roi.json",
     "camera_C": "...\\camera_C_slot_roi.json"}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE_DIR = Path(r"I:\德州有研视频数据集\第二次数据\2026-09-24")
DEFAULT_OUTPUT_DIR = Path(r"I:\youyan\test_dataset\9-24测试视频")
DEFAULT_WEIGHTS = Path(r"C:\Users\win10\xanylabeling_data\trainer\ultralytics\runs\detect\exp3.3\weights\best.pt")
DEFAULT_ROI_FILE = PROJECT_ROOT / "industrial_inspection" / "configs" / "slot_roi.json"
DEFAULT_S4_HAND_REGION_FILE = PROJECT_ROOT / "industrial_inspection" / "configs" / "s4_hand_region.json"
DEFAULT_DEVICE = "0"
DEFAULT_CONFIDENCE = 0.1
DEFAULT_AUDIT_CONFIDENCE = 0.001
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_FRAME_STRIDE = 1
DEFAULT_MAX_FRAMES = 0
DEFAULT_CAMERA = "B"
VIDEO_EXTENSIONS = frozenset({".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"})
CAMERA_RE = re.compile(r"(?:^|[_-])camera[_-]?([A-Za-z0-9]+)(?:[_-]|$)", re.IGNORECASE)

slot_groups = None


def _load_slot_groups():
    """延迟导入重型检测依赖，使 ``--dry-run`` 可在无 PyTorch 环境中使用。"""
    global slot_groups
    if slot_groups is None:
        from industrial_inspection.tools import predict_slot_groups
        slot_groups = predict_slot_groups
    return slot_groups


def _natural_key(path: Path) -> tuple[Any, ...]:
    """让事件目录和视频名按数字部分自然排序。"""
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", path.name))


def camera_name(video: Path) -> str:
    """从 ``camera_A_...`` 文件名取得规范化相机名。"""
    match = CAMERA_RE.search(video.stem)
    return f"camera_{match.group(1).upper()}" if match else "unknown_camera"


def discover_videos(
    source_dir: Path,
    *,
    camera: str | None = None,
    exclude_dirs: Iterable[str] = (),
) -> list[tuple[Path, list[Path]]]:
    """发现一级事件目录中的视频，保留目录分组以便输出时对应原数据集。"""
    excluded = {name.casefold() for name in exclude_dirs}
    camera_filter = camera.upper() if camera else None
    groups: list[tuple[Path, list[Path]]] = []
    for group_dir in sorted((p for p in source_dir.iterdir() if p.is_dir()), key=_natural_key):
        if group_dir.name.casefold() in excluded:
            continue
        videos = [
            p for p in group_dir.iterdir()
            if p.is_file() and p.suffix.casefold() in VIDEO_EXTENSIONS
            and (camera_filter is None or camera_name(p).endswith(f"_{camera_filter}"))
        ]
        videos.sort(key=_natural_key)
        if videos:
            groups.append((group_dir, videos))
    return groups


def _read_roi_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"ROI 映射文件不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--roi-map 必须是 JSON 对象，键应为 camera_A/camera_B/camera_C")
    result: dict[str, Path] = {}
    for key, value in data.items():
        if not isinstance(value, (str, Path)):
            raise ValueError(f"ROI 映射值必须是路径：{key}")
        normalized = str(key).lower()
        if not normalized.startswith("camera_"):
            normalized = f"camera_{normalized}"
        roi_path = Path(value).expanduser()
        if not roi_path.is_absolute():
            roi_path = path.parent / roi_path
        result[normalized] = roi_path.resolve()
    return result


def resolve_roi_file(video: Path, default_roi: Path, roi_map: dict[str, Path]) -> Path:
    """选择该视频的 ROI；未映射的相机回退到共享配置。"""
    return roi_map.get(camera_name(video).lower(), default_roi)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                        help="包含多个事件子目录的视频根目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="批量结果根目录")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--roi-file", type=Path, default=DEFAULT_ROI_FILE,
                        help="未在 --roi-map 中匹配时使用的共享 ROI")
    parser.add_argument("--roi-map", type=Path,
                        help="按相机映射 ROI 的 JSON 文件")
    parser.add_argument("--s4-hand-region-file", type=Path, default=DEFAULT_S4_HAND_REGION_FILE,
                        help="S4 手部区域配置；批处理时不重新手动选择")
    parser.add_argument("--disable-s4-hand-region", action="store_true",
                        help="不使用 S4 手部区域配置")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--audit-conf", type=float, default=DEFAULT_AUDIT_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help="每个视频最多处理的帧数，0 表示完整视频")
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="只处理指定相机，例如 A、B 或 C；默认只处理 camera_B")
    parser.add_argument("--exclude-dir", action="append", default=[],
                        help="跳过事件目录名，可重复指定；默认不跳过带‘无用’的目录")
    parser.add_argument("--limit", type=int, default=0,
                        help="最多处理多少个视频，0 表示全部；适合先试跑")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="已有 summary.json 的视频跳过，默认开启")
    parser.add_argument("--fail-fast", action="store_true", help="首个视频失败后立即停止")
    parser.add_argument("--dry-run", action="store_true", help="只列出待处理视频，不加载模型")
    return parser


def _configure_module(args: argparse.Namespace) -> None:
    """把单视频入口中依赖顶部配置的交互项切换为批处理配置。"""
    slot_groups = _load_slot_groups()
    if not args.weights.is_file():
        raise FileNotFoundError(f"模型权重不存在：{args.weights}")
    if not args.disable_s4_hand_region and not args.s4_hand_region_file.is_file():
        raise FileNotFoundError(f"S4 手部区域配置不存在：{args.s4_hand_region_file}；可使用 --disable-s4-hand-region")
    slot_groups.S4_HAND_REGION_RESELECT = False
    slot_groups.S4_HAND_REGION_FILE = args.s4_hand_region_file.resolve()
    slot_groups.S4_HAND_REGION_ENABLED = not args.disable_s4_hand_region


def _single_video_args(
    args: argparse.Namespace,
    video: Path,
    output: Path,
    roi_file: Path,
) -> argparse.Namespace:
    slot_groups = _load_slot_groups()
    argv = [
        "--source", str(video),
        "--weights", str(args.weights),
        "--output-dir", str(output),
        "--roi-file", str(roi_file),
        "--no-reselect-roi",
        "--conf", str(args.conf),
        "--audit-conf", str(args.audit_conf),
        "--imgsz", str(args.imgsz),
        "--device", str(args.device),
        "--stride", str(args.stride),
        "--max-frames", str(args.max_frames),
        "--no-ui",
        "--no-show",
    ]
    return slot_groups.build_args(argv)


def _record_base(group_dir: Path, video: Path, output: Path, roi_file: Path) -> dict[str, Any]:
    return {
        "source_folder": group_dir.name,
        "video_name": video.name,
        "camera": camera_name(video),
        "source": str(video.resolve()),
        "output_dir": str(output.resolve()),
        "roi_file": str(roi_file.resolve()),
    }


def _summary_record(base: dict[str, Any], summary: dict[str, Any], status: str = "completed") -> dict[str, Any]:
    reports = summary.get("group_process_reports", {})
    completed = {
        phase: sum(bool(group.get("completed")) for group in report.get("groups", []))
        for phase, report in reports.items()
    }
    return {
        **base,
        "status": status,
        "processed_frames": summary.get("processed_frames"),
        "inspection_processed_frames": summary.get("inspection_processed_frames"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "end_status": summary.get("status"),
        "unload_completed_groups": completed.get("unload", 0),
        "load_completed_groups": completed.get("load", 0),
        "error": "",
    }


def run_one(
    args: argparse.Namespace,
    group_dir: Path,
    video: Path,
    output: Path,
    roi_file: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    slot_groups = _load_slot_groups()
    base = _record_base(group_dir, video, output, roi_file)
    summary_path = output / "summary.json"
    if output.exists():
        if args.resume and summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"无法读取已有结果 {summary_path}: {exc}") from exc
            return _summary_record(base, summary, status="skipped_existing"), summary
        raise FileExistsError(
            f"输出目录已存在且未发现可恢复的 summary.json：{output}。"
            "请更换 --output-dir、清理该目录，或使用 --resume 跳过已完成结果。"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        video_args = _single_video_args(args, video, output, roi_file)
        summary = slot_groups.run_prediction(video_args)
        return _summary_record(base, summary), summary
    except Exception as exc:  # 单个视频失败不影响其余视频，除非 --fail-fast
        record = {**base, "status": "failed", "processed_frames": None,
                  "inspection_processed_frames": None, "elapsed_seconds": None,
                  "end_status": None, "unload_completed_groups": None,
                  "load_completed_groups": None,
                  "error": f"{type(exc).__name__}: {exc}"}
        if output.exists():
            error_path = output / "batch_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
        return record, None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_batch_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    summaries: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    source_dir: Path,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_fields = [
        "source_folder", "video_name", "camera", "source", "output_dir", "roi_file",
        "status", "processed_frames", "inspection_processed_frames", "elapsed_seconds",
        "end_status", "unload_completed_groups", "load_completed_groups", "error",
    ]
    _write_csv(output_dir / "batch_results.csv", records, result_fields)

    group_rows: list[dict[str, Any]] = []
    report_sections: list[str] = []
    for record, summary in summaries:
        reports = summary.get("group_process_reports", {})
        report_sections.append(f"===== {record['source_folder']} / {record['video_name']} =====")
        report_file = Path(record["output_dir"]) / "group_process_report.txt"
        if report_file.is_file():
            report_sections.append(report_file.read_text(encoding="utf-8").rstrip())
        for phase, report in reports.items():
            for group in report.get("groups", []):
                group_rows.append({
                    "source_folder": record["source_folder"],
                    "video_name": record["video_name"],
                    "camera": record["camera"],
                    "phase": phase,
                    "group_id": group.get("group_id"),
                    "observed": group.get("observed"),
                    "completed": group.get("completed"),
                    "slot_completed": json.dumps(group.get("slot_completed"), ensure_ascii=False),
                    "sequence": json.dumps(group.get("sequence"), ensure_ascii=False),
                    "order_result": group.get("order_result"),
                    "end_status": report.get("end_status"),
                })
    group_fields = [
        "source_folder", "video_name", "camera", "phase", "group_id", "observed",
        "completed", "slot_completed", "sequence", "order_result", "end_status",
    ]
    _write_csv(output_dir / "batch_group_process_report.csv", group_rows, group_fields)
    (output_dir / "batch_process_report.txt").write_text(
        "\n\n".join(report_sections) + ("\n" if report_sections else ""), encoding="utf-8"
    )

    manifest = {
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "video_count": len(records),
        "completed_count": sum(r["status"] == "completed" for r in records),
        "skipped_count": sum(r["status"] == "skipped_existing" for r in records),
        "failed_count": sum(r["status"] == "failed" for r in records),
        "config": config,
    }
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{source_dir}")
    if args.limit < 0 or args.max_frames < 0 or args.stride < 1:
        raise ValueError("limit/max-frames 必须非负，stride 必须至少为 1")
    if args.camera:
        args.camera = args.camera.removeprefix("camera_").upper()
    roi_map = _read_roi_map(args.roi_map.expanduser().resolve() if args.roi_map else None)
    groups = discover_videos(source_dir, camera=args.camera, exclude_dirs=args.exclude_dir)
    jobs = [(group_dir, video) for group_dir, videos in groups for video in videos]
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        raise RuntimeError(f"输入目录中没有找到视频：{source_dir}")

    print(f"发现 {len(jobs)} 个视频，输出目录：{output_dir}", flush=True)
    for group_dir, video in jobs:
        print(f"  {group_dir.name}/{video.name}", flush=True)
    if args.dry_run:
        return 0

    _configure_module(args)
    records: list[dict[str, Any]] = []
    summaries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, (group_dir, video) in enumerate(jobs, start=1):
        roi_file = resolve_roi_file(video, args.roi_file.resolve(), roi_map)
        if not roi_file.is_file():
            record = _record_base(group_dir, video, output_dir / group_dir.name / video.stem, roi_file)
            record.update(status="failed", processed_frames=None, inspection_processed_frames=None,
                          elapsed_seconds=None, end_status=None, unload_completed_groups=None,
                          load_completed_groups=None, error=f"ROI 配置不存在：{roi_file}")
            records.append(record)
            print(f"[{index}/{len(jobs)}] 失败：{video.name}（ROI 不存在）", flush=True)
            if args.fail_fast:
                break
            continue

        result_dir = output_dir / group_dir.name / video.stem
        print(f"[{index}/{len(jobs)}] 开始：{group_dir.name}/{video.name}", flush=True)
        record, summary = run_one(args, group_dir, video, result_dir, roi_file)
        records.append(record)
        if summary is not None:
            summaries.append((record, summary))
        print(f"[{index}/{len(jobs)}] {record['status']}：{video.name}", flush=True)
        if record["status"] == "failed" and args.fail_fast:
            break

    write_batch_outputs(
        output_dir,
        records,
        summaries,
        source_dir=source_dir,
        config={
            "weights": str(args.weights.resolve()),
            "shared_roi": str(args.roi_file.resolve()),
            "roi_map": str(args.roi_map.resolve()) if args.roi_map else None,
            "s4_hand_region_file": (None if args.disable_s4_hand_region
                                    else str(args.s4_hand_region_file.resolve())),
            "device": args.device,
            "conf": args.conf,
            "audit_conf": args.audit_conf,
            "imgsz": args.imgsz,
            "stride": args.stride,
            "max_frames": args.max_frames,
            "camera": args.camera,
            "exclude_dirs": args.exclude_dir,
        },
    )
    failed = sum(record["status"] == "failed" for record in records)
    print(f"批处理完成：成功 {len(records) - failed}，失败 {failed}。", flush=True)
    print(f"汇总：{output_dir / 'batch_results.csv'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
