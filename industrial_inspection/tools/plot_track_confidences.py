"""Plot confidence curves for selected ByteTrack IDs.

Configure ``INPUT_CSV`` and ``TRACK_IDS`` for direct PyCharm execution, or use
the command line. One image is saved for each selected track ID, combining all
groups and phases where that ID appears.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# PyCharm configuration
# ---------------------------------------------------------------------------
INPUT_CSV = None  # Example: Path(r"I:\youyan\run\detection_confidences.csv")
# INPUT_CSV =Path(r"I:\youyan\test_dataset\测试视频2\置信度0.7camera_B_20260923_143845_slots_20261027_162650_098934"
#                 r"\detection_confidences.csv")
SEARCH_ROOT = Path(r"I:\youyan\test_dataset")
TRACK_IDS = [578]  # 逻辑/ByteTrack IDs to plot; logical mode also joins reidentified IDs
OUTPUT_ROOT = None  # None -> slot_confidence_tracks beside INPUT_CSV
SHOW_WINDOW = False
USE_LOGICAL_TRACK_ID = True  # 合并长遮挡后通过槽位几何重连的ByteTrack轨迹

DISPLAY_CONFIDENCE = .1
BYTE_TRACK_LOW_CONFIDENCE = .10
CONNECT_POINTS = True
POINT_SIZE = 14
POINT_ALPHA = .82
LINE_ALPHA = .75
LINE_WIDTH = 1.0
SHOW_TRACK_LABELS = True

SLOT_COLORS = {"S1": "#1f77b4", "S2": "#2ca02c", "S3": "#d62728", "S4": "#9467bd",
               "待编号": "#7f7f7f", "": "#1f77b4"}
CLASS_LABELS = {"wafer_slot_empty": "空槽", "wafer_slot_filled": "已填充"}


def _configure_chinese_font(plt) -> None:
    from matplotlib import font_manager

    windows_fonts = Path(r"C:\Windows\Fonts")
    for font_path in (windows_fonts / "msyh.ttc", windows_fonts / "simhei.ttf",
                      windows_fonts / "Deng.ttf", windows_fonts / "simsun.ttc"):
        if not font_path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
        except (OSError, RuntimeError):
            continue
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _latest_csv(root: Path) -> Path:
    candidates = [path for path in root.rglob("detection_confidences.csv") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"未找到 detection_confidences.csv：{root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _as_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _track_id(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    except ValueError:
        return text


def _as_bool(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _read_rows(path: Path, selected_ids: set[str]):
    records = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "track_id" not in reader.fieldnames:
            raise ValueError("CSV缺少 track_id 列，请使用新版 predict_slot_groups.py 重新运行视频。")
        for row_number, row in enumerate(reader):
            track_id = _track_id(row.get("track_id"))
            logical_track_id = _track_id(row.get("logical_track_id")) or track_id
            plot_id = logical_track_id if USE_LOGICAL_TRACK_ID else track_id
            if plot_id not in selected_ids:
                continue
            group_value = _as_float(row.get("group_id"), 0)
            group_id = int(group_value) if group_value is not None else 0
            phase = (row.get("phase") or "").strip()
            source_frame = _as_float(row.get("source_frame"), row_number)
            time_seconds = _as_float(row.get("time_seconds"), source_frame)
            confidence = _as_float(row.get("confidence"))
            if confidence is None:
                continue
            records[plot_id].append({
                "source_frame": int(source_frame),
                "time_seconds": time_seconds,
                "confidence": confidence,
                "group_id": group_id,
                "phase": phase,
                "track_id": track_id,
                "logical_track_id": logical_track_id,
                "slot_point": (row.get("slot_point") or "").strip(),
                "class_name": (row.get("class_name") or "").strip(),
                "displayed": _as_bool(row.get("displayed")),
            })
    for values in records.values():
        values.sort(key=lambda item: (item["time_seconds"], item["source_frame"]))
    return records


def _plot_track(track_id, values, input_csv: Path, output_root: Path, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    _configure_chinese_font(plt)
    groups = sorted({item["group_id"] for item in values})
    phases = sorted({item["phase"] or "未分阶段" for item in values})
    slots = sorted({item["slot_point"] for item in values if item["slot_point"]})
    slot_label = ",".join(slots) or "未映射槽位"
    color = SLOT_COLORS.get(slots[0] if slots else "", "#1f77b4")
    fig, ax = plt.subplots(figsize=(12, 6))
    ordered = sorted(values, key=lambda item: (item["time_seconds"], item["source_frame"]))
    if CONNECT_POINTS and len(ordered) >= 2:
        ax.plot([item["time_seconds"] for item in ordered],
                [item["confidence"] for item in ordered],
                color=color, alpha=LINE_ALPHA, linewidth=LINE_WIDTH,
                label="按时间连接", zorder=1)
    shown = [item for item in ordered if item["displayed"]]
    hidden = [item for item in ordered if not item["displayed"]]
    for subset, label, marker, marker_color in ((shown, "实际显示", "o", color),
                                                 (hidden, "未显示", "o", "#ff7f0e")):
        if subset:
            ax.scatter([item["time_seconds"] for item in subset],
                       [item["confidence"] for item in subset],
                       s=POINT_SIZE, alpha=POINT_ALPHA, color=marker_color,
                       marker=marker, edgecolors="#222222", linewidths=.25,
                       label=label, rasterized=True, zorder=2)
    if SHOW_TRACK_LABELS:
        previous_detection = object()
        for item in ordered:
            state = (item["track_id"] if "track_id" in item else track_id, item["class_name"])
            if state == previous_detection:
                continue
            class_label = CLASS_LABELS.get(item["class_name"], item["class_name"] or "未知")
            id_label = (f"L{track_id}" if USE_LOGICAL_TRACK_ID else f"BT{track_id}")
            actual_id = item.get("track_id", track_id)
            if actual_id and str(actual_id) != str(track_id):
                id_label += f"/BT{actual_id}"
            ax.annotate(f"{id_label} {class_label} {item['confidence']:.2f}",
                        (item["time_seconds"], item["confidence"]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color=color, alpha=.9, zorder=3)
            previous_detection = state
    ax.axhline(DISPLAY_CONFIDENCE, color="#c62828", linestyle="--", linewidth=1,
               label=f"显示阈值 {DISPLAY_CONFIDENCE:.2f}")
    ax.axhline(BYTE_TRACK_LOW_CONFIDENCE, color="#1565c0", linestyle=":", linewidth=1,
               label=f"关联阈值 {BYTE_TRACK_LOW_CONFIDENCE:.2f}")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.22)
    ax.set_xlabel("视频时间（秒）")
    ax.set_ylabel("检测框置信度")
    group_label = ",".join(f"G{group_id}" for group_id in groups) or "未知组"
    phase_label = ",".join(phases)
    id_label = f"L{track_id}" if USE_LOGICAL_TRACK_ID else f"BT{track_id}"
    ax.set_title(f"{group_label} - {phase_label} - {id_label} - {slot_label}\n{input_csv.name}")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{'L' if USE_LOGICAL_TRACK_ID else 'BT'}{track_id}.png"
    fig.savefig(output_path, dpi=180)
    print(f"已保存：{output_path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--search-root", type=Path, default=SEARCH_ROOT)
    parser.add_argument("--track-id", nargs="+", default=[str(value) for value in TRACK_IDS],
                        help="逻辑/ByteTrack ID，例如 --track-id 34 18 27")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW_WINDOW)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    input_csv = (args.csv or _latest_csv(args.search_root)).resolve()
    selected_ids = {_track_id(value) for value in args.track_id}
    selected_ids.discard("")
    if not selected_ids:
        raise ValueError("至少指定一个有效的 ByteTrack ID")
    output_root = (args.output_root or input_csv.parent / "slot_confidence_tracks").resolve()
    records = _read_rows(input_csv, selected_ids)
    if not records:
        raise ValueError(f"CSV中没有找到指定的 track_id：{', '.join(sorted(selected_ids))}")
    for track_id, values in sorted(records.items()):
        _plot_track(track_id, values, input_csv, output_root, show=args.show)


if __name__ == "__main__":
    main()
