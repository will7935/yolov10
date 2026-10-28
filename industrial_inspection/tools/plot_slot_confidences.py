"""Plot per-slot detection confidence from ``detection_confidences.csv``.

The script is configured to run directly from PyCharm.  Set ``INPUT_CSV`` to a
specific prediction output, or leave it as ``None`` to use the newest CSV under
``SEARCH_ROOT``.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


# ---------------------------------------------------------------------------
# PyCharm configuration
# ---------------------------------------------------------------------------
INPUT_CSV = None  # None -> use the newest detection_confidences.csv under SEARCH_ROOT
SEARCH_ROOT = Path(r"I:\youyan\test_dataset")
OUTPUT_IMAGE = None  # None -> save beside INPUT_CSV as slot_confidence_plot.png
OUTPUT_SUMMARY_CSV = None  # None -> save beside INPUT_CSV as slot_confidence_summary.csv
SHOW_WINDOW = False  # PyCharm direct run: save images only; enable --show to open windows.

SLOT_POINTS = ("S1", "S2", "S3", "S4")
GROUP_IDS = (1, 2, 3, 4, 5)
DISPLAY_CONFIDENCE = .7
BYTE_TRACK_LOW_CONFIDENCE = .10
PLOT_PENDING = False  # Set True to include the temporary "待编号" group as a fifth panel.
SHOW_PER_FRAME_SUMMARY = False  # Hide max/median curves; summary CSV is still written.
POINT_SIZE = 9
POINT_ALPHA = .78
CONNECT_POINTS = True
LINE_ALPHA = .72
LINE_WIDTH = .9
SHOW_TRACK_ID = True  # 在置信度图的点旁标注 ByteTrack 轨迹编号，例如 BT12。
CLASS_LABELS = {"wafer_slot_empty": "空槽", "wafer_slot_filled": "已填充"}


def _latest_csv(root: Path) -> Path:
    candidates = [path for path in root.rglob("detection_confidences.csv") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"未找到 detection_confidences.csv：{root}。请设置 INPUT_CSV。")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _as_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: str | None, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _read_rows(path: Path):
    records = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "slot_point" not in reader.fieldnames or "group_id" not in reader.fieldnames:
            raise ValueError(
                "CSV缺少 slot_point 或 group_id 列。请使用更新后的 predict_slot_groups.py 重新运行视频。")
        for row_number, row in enumerate(reader):
            slot_point = (row.get("slot_point") or "").strip()
            if slot_point not in SLOT_POINTS and not (PLOT_PENDING and slot_point == "待编号"):
                continue
            try:
                group_id = int(float(row.get("group_id") or ""))
            except (TypeError, ValueError):
                continue
            confidence = _as_float(row.get("confidence"))
            if confidence is None:
                continue
            source_frame = _as_float(row.get("source_frame"), float(row_number))
            detection_index = _as_float(row.get("detection_index"), float(row_number))
            time_seconds = _as_float(row.get("time_seconds"))
            if time_seconds is None:
                time_seconds = source_frame
            phase = (row.get("phase") or "").strip()
            records[(group_id, phase, slot_point)].append({
                "source_frame": int(source_frame),
                "detection_index": int(detection_index),
                "time_seconds": time_seconds,
                "confidence": confidence,
                "class_name": (row.get("class_name") or "").strip(),
                "displayed": _as_bool(row.get("displayed")),
                "track_id": (row.get("track_id") or "").strip(),
            })
    for values in records.values():
        values.sort(key=lambda item: (item["source_frame"], item["time_seconds"]))
    return records


def _aggregate(values):
    grouped = defaultdict(list)
    metadata = {}
    for item in values:
        key = item["source_frame"]
        grouped[key].append(item["confidence"])
        metadata[key] = item["time_seconds"]
    result = []
    for frame in sorted(grouped):
        scores = grouped[frame]
        result.append({
            "source_frame": frame,
            "time_seconds": metadata[frame],
            "detection_count": len(scores),
            "max_confidence": max(scores),
            "median_confidence": median(scores),
        })
    return result


def _write_summary(path: Path, records) -> None:
    fields = ["group_id", "phase", "slot_point", "source_frame", "time_seconds", "detection_count",
              "max_confidence", "median_confidence"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (group_id, phase, slot_point), values in sorted(records.items()):
            for row in _aggregate(values):
                writer.writerow({"group_id": group_id, "phase": phase,
                                 "slot_point": slot_point, **row})


def _group_track_records(records):
    """Flatten slot-keyed rows into per-group, per-phase, per-track series."""
    tracks = defaultdict(list)
    for (group_id, phase, slot_point), values in records.items():
        for item in values:
            track_id = item.get("track_id", "")
            if not track_id:
                continue
            tracks[(group_id, phase, track_id)].append({**item, "slot_point": slot_point})
    for values in tracks.values():
        values.sort(key=lambda item: (item["time_seconds"], item["source_frame"]))
    return tracks


def _configure_chinese_font(plt) -> str | None:
    """Select an installed Chinese font so plot titles and headers render correctly."""
    from matplotlib import font_manager

    windows_fonts = Path(r"C:\Windows\Fonts")
    candidates = [
        windows_fonts / "msyh.ttc",   # Microsoft YaHei
        windows_fonts / "simhei.ttf",  # SimHei
        windows_fonts / "Deng.ttf",  # DengXian
        windows_fonts / "simsun.ttc",  # SimSun
    ]
    for font_path in candidates:
        if not font_path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
        except (OSError, RuntimeError):
            continue
    # The fallback list is useful on Linux/macOS or when Windows fonts are unavailable.
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None


def plot_confidences(input_csv: Path, output_image: Path, output_summary: Path, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    _configure_chinese_font(plt)
    records = _read_rows(input_csv)
    if not records:
        raise ValueError("CSV中没有可用于绘图的正式槽位检测框记录。")

    _write_summary(output_summary, records)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_root = output_image.parent / "slot_confidence_groups"
    output_root.mkdir(parents=True, exist_ok=True)
    colors = {"S1": "#1f77b4", "S2": "#2ca02c", "S3": "#d62728", "S4": "#9467bd",
              "待编号": "#7f7f7f"}
    figures = []

    for group_id in GROUP_IDS:
        group_root = output_root / f"G{group_id}"
        group_root.mkdir(parents=True, exist_ok=True)
        phases = sorted({phase for gid, phase, _ in records if gid == group_id}) or [""]
        for phase in phases:
            phase_label = phase or "未分阶段"
            phase_suffix = f"_{phase}" if phase else ""
            for slot_point in SLOT_POINTS:
                fig, ax = plt.subplots(figsize=(12, 6))
                if show:
                    figures.append(fig)
                values = records.get((group_id, phase, slot_point), [])
                color = colors[slot_point]
                if values:
                    shown = [item for item in values if item["displayed"]]
                    hidden = [item for item in values if not item["displayed"]]
                    if CONNECT_POINTS and len(values) >= 2:
                        ordered = sorted(values, key=lambda item: (item["time_seconds"],
                                                                    item["source_frame"]))
                        ax.plot([item["time_seconds"] for item in ordered],
                                [item["confidence"] for item in ordered],
                                color=color, alpha=LINE_ALPHA, linewidth=LINE_WIDTH,
                                label="按时间连接", zorder=1)
                    for subset, label, marker, marker_color in ((shown, "实际显示", "o", color),
                                                                 (hidden, "未显示", "o", "#ff7f0e")):
                        if subset:
                            ax.scatter([item["time_seconds"] for item in subset],
                                       [item["confidence"] for item in subset],
                                       s=POINT_SIZE, alpha=POINT_ALPHA, color=marker_color,
                                       marker=marker,
                                       edgecolors="#222222" if marker == "o" else "none",
                                       linewidths=.25,
                                       label=label, rasterized=True, zorder=2)
                    if SHOW_TRACK_ID:
                        ordered = sorted(values, key=lambda item: (item["time_seconds"],
                                                                    item["source_frame"]))
                        previous_detection = object()
                        for item in ordered:
                            track_id = item["track_id"]
                            detection_state = (track_id, item["class_name"])
                            if detection_state == previous_detection:
                                continue
                            ax.annotate(
                                f"BT{track_id or '-'} {CLASS_LABELS.get(item['class_name'], item['class_name'] or '未知')} "
                                f"{item['confidence']:.2f}",
                                (item["time_seconds"], item["confidence"]),
                                xytext=(3, 3), textcoords="offset points",
                                fontsize=6, color=colors[slot_point], alpha=.9,
                                zorder=3)
                            previous_detection = detection_state
                    if SHOW_PER_FRAME_SUMMARY:
                        summary = _aggregate(values)
                        times = [item["time_seconds"] for item in summary]
                        ax.plot(times, [item["max_confidence"] for item in summary],
                                color=color, linewidth=1.5, label="每帧最高")
                        ax.plot(times, [item["median_confidence"] for item in summary],
                                color="#333333", linewidth=1.1, linestyle="--", label="每帧中位数")
                    ax.set_title(f"G{group_id} - {phase_label} - {slot_point}  检测框 {len(values)} 个")
                else:
                    ax.set_title(f"G{group_id} - {phase_label} - {slot_point}  无记录")
                    ax.text(.5, .5, "无该槽位记录", ha="center", va="center", transform=ax.transAxes)
                ax.axhline(DISPLAY_CONFIDENCE, color="#c62828", linestyle="--", linewidth=1,
                           label=f"显示阈值 {DISPLAY_CONFIDENCE:.2f}")
                ax.axhline(BYTE_TRACK_LOW_CONFIDENCE, color="#1565c0", linestyle=":", linewidth=1,
                           label=f"关联阈值 {BYTE_TRACK_LOW_CONFIDENCE:.2f}")
                ax.set_ylim(0, 1.02)
                ax.grid(alpha=.22)
                ax.set_xlabel("视频时间（秒）")
                ax.set_ylabel("检测框置信度")
                ax.legend(fontsize=9, loc="best")
                fig.suptitle(f"G{group_id} - {phase_label} - {slot_point}\n{input_csv.name}", fontsize=14)
                fig.tight_layout(rect=(0, 0, 1, .93))
                output_path = group_root / f"{slot_point}{phase_suffix}.png"
                fig.savefig(output_path, dpi=180)
                print(f"G{group_id} {phase_label} {slot_point} 图已保存：{output_path}")
                if not show:
                    plt.close(fig)
    # Add one confidence plot for each ByteTrack trajectory under its group.
    track_records = _group_track_records(records)
    for (group_id, phase, track_id), values in sorted(track_records.items()):
        group_root = output_root / f"G{group_id}" / "track_confidence"
        group_root.mkdir(parents=True, exist_ok=True)
        phase_label = phase or "未分阶段"
        phase_suffix = f"_{phase}" if phase else ""
        slot_points_for_track = sorted({item["slot_point"] for item in values})
        base_slot = slot_points_for_track[0] if slot_points_for_track else "待编号"
        color = colors.get(base_slot, "#1f77b4")
        fig, ax = plt.subplots(figsize=(12, 6))
        if show:
            figures.append(fig)
        shown = [item for item in values if item["displayed"]]
        hidden = [item for item in values if not item["displayed"]]
        ordered = sorted(values, key=lambda item: (item["time_seconds"], item["source_frame"]))
        if CONNECT_POINTS and len(ordered) >= 2:
            ax.plot([item["time_seconds"] for item in ordered],
                    [item["confidence"] for item in ordered],
                    color=color, alpha=LINE_ALPHA, linewidth=LINE_WIDTH,
                    label="按时间连接", zorder=1)
        for subset, label, marker, marker_color in ((shown, "实际显示", "o", color),
                                                     (hidden, "未显示", "o", "#ff7f0e")):
            if subset:
                ax.scatter([item["time_seconds"] for item in subset],
                           [item["confidence"] for item in subset],
                           s=POINT_SIZE, alpha=POINT_ALPHA, color=marker_color,
                           marker=marker, edgecolors="#222222", linewidths=.25,
                           label=label, rasterized=True, zorder=2)
        if SHOW_TRACK_ID:
            previous_detection = object()
            for item in ordered:
                detection_state = (item["track_id"], item["class_name"])
                if detection_state == previous_detection:
                    continue
                label = CLASS_LABELS.get(item["class_name"], item["class_name"] or "未知")
                ax.annotate(f"BT{track_id} {label} {item['confidence']:.2f}",
                            (item["time_seconds"], item["confidence"]),
                            xytext=(3, 3), textcoords="offset points",
                            fontsize=6, color=color, alpha=.9, zorder=3)
                previous_detection = detection_state
        ax.axhline(DISPLAY_CONFIDENCE, color="#c62828", linestyle="--", linewidth=1,
                   label=f"显示阈值 {DISPLAY_CONFIDENCE:.2f}")
        ax.axhline(BYTE_TRACK_LOW_CONFIDENCE, color="#1565c0", linestyle=":", linewidth=1,
                   label=f"关联阈值 {BYTE_TRACK_LOW_CONFIDENCE:.2f}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.22)
        ax.set_xlabel("视频时间（秒）")
        ax.set_ylabel("检测框置信度")
        ax.set_title(f"G{group_id} - {phase_label} - BT{track_id} - "
                     f"{','.join(slot_points_for_track)}  检测框 {len(values)} 个")
        ax.legend(fontsize=9, loc="best")
        fig.suptitle(f"G{group_id} - {phase_label} - BT{track_id}\n{input_csv.name}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .93))
        output_path = group_root / f"BT{track_id}{phase_suffix}.png"
        fig.savefig(output_path, dpi=180)
        print(f"G{group_id} {phase_label} BT{track_id} 图已保存：{output_path}")
        if not show:
            plt.close(fig)
    if show:
        plt.show()
    for fig in figures:
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=INPUT_CSV,
                        help="detection_confidences.csv; omitted means newest file under SEARCH_ROOT")
    parser.add_argument("--search-root", type=Path, default=SEARCH_ROOT)
    parser.add_argument("--output-image", type=Path, default=OUTPUT_IMAGE)
    parser.add_argument("--output-summary", type=Path, default=OUTPUT_SUMMARY_CSV)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW_WINDOW,
                        help="Open plot windows after saving; default is disabled for PyCharm direct run")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    input_csv = args.csv or _latest_csv(args.search_root)
    input_csv = input_csv.resolve()
    output_image = (args.output_image or input_csv.with_name("slot_confidence_plot.png")).resolve()
    output_summary = (args.output_summary or input_csv.with_name("slot_confidence_summary.csv")).resolve()
    plot_confidences(input_csv, output_image, output_summary, show=args.show)
    print(f"G1~G5 槽位图已保存到：{output_image.parent / 'slot_confidence_groups'}")
    print(f"聚合数据已保存：{output_summary}")


if __name__ == "__main__":
    main()
