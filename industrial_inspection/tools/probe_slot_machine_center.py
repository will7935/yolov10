"""Pick the machine center and compare slot distances in a saved prediction run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_group_poses(summary):
    for group in summary["closed_groups"]:
        snapshot = group["snapshot"]
        init = snapshot["initialization"]
        points = np.asarray(init.get("initial_centers_xy"), dtype=float)
        inferred = init.get("inferred_slot_ids", [])
        if (points.shape != (4, 2) or not np.isfinite(points).all()
                or inferred or len(set(init.get("geometry_seen_track_ids", [])) - {None}) != 4):
            continue
        yield {
            "phase": group["phase"],
            "group_id": group["group_id"],
            "at": init["initialized_at"],
            "points": points,
            "observed_sequence": snapshot["confirmed_event_sequence"],
        }


def select_machine_center(source, initial_time):
    capture = cv2.VideoCapture(str(source))
    window = "Mark grinder center"
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or frame_count <= 0:
            raise ValueError(f"无法读取视频 {source}")
        selected = None

        def click(event, x, y, flags, userdata):
            nonlocal selected
            if event == cv2.EVENT_LBUTTONDOWN:
                selected = (x, y)

        frame_index = min(max(0, round(initial_time * fps)), frame_count - 1)
        current_frame = None
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, click)
        while True:
            if frame_index != current_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"无法读取第 {frame_index} 帧")
                current_frame = frame_index
            preview = frame.copy()
            label = f"{frame_index/fps:.2f}s | Click center | Enter save | A/D seek | Esc cancel"
            cv2.rectangle(preview, (0, 30), (preview.shape[1], 55), (0, 0, 0), -1)
            cv2.putText(preview, label, (6, 48), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
            if selected is not None:
                cv2.drawMarker(preview, selected, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(preview, str(selected), (8, 76), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(40) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None, None
            if key in (10, 13) and selected is not None:
                return selected, frame_index / fps
            if key in (ord("a"), ord("d")):
                step = round(fps) if key == ord("d") else -round(fps)
                frame_index = min(max(0, frame_index + step), frame_count - 1)
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def compare_distances(poses, center, anchor_group):
    anchor = next((pose for pose in poses
                   if pose["phase"] == "unload" and pose["group_id"] == anchor_group), None)
    if anchor is None:
        raise ValueError(f"下料 G{anchor_group} 缺少四个实测槽，无法用作编号基准")
    anchor_slot = int(np.argmin(np.linalg.norm(anchor["points"] - center, axis=1))) + 1
    clockwise = (1, 3, 4, 2)
    records = []
    for pose in poses:
        distances = np.linalg.norm(pose["points"] - center, axis=1)
        ranked = np.argsort(distances, kind="stable")
        nearest = int(ranked[0]) + 1
        gap = float(distances[ranked[1]] - distances[ranked[0]])
        offset = clockwise.index(anchor_slot) - clockwise.index(nearest)
        mapping = {old: clockwise[(i + offset) % 4] for i, old in enumerate(clockwise)}
        records.append({**pose, "points": pose["points"].tolist(),
                        "distances_px": distances.round(3).tolist(), "nearest_old_slot": nearest,
                        "distance_gap_px": round(gap, 3), "trial_slot_mapping": mapping,
                        "trial_sequence": [mapping[slot] for slot in pose["observed_sequence"]]})
    return anchor_slot, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="预测输出目录，需包含 summary.json")
    parser.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"),
                        help="已知机心的原始画面像素坐标；省略则打开点选窗口")
    parser.add_argument("--time", type=float, help="点选窗口起始的源视频秒数")
    parser.add_argument("--anchor-group", type=int, choices=range(1, 6), default=3,
                        help="已核实编号的下料组；保留该组中距离最近槽的原编号")
    args = parser.parse_args()
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8-sig"))
    poses = list(read_group_poses(summary))
    if not poses:
        parser.error("报告没有可比较的四槽实测姿态")
    if args.center is not None and not all(math.isfinite(x) for x in args.center):
        parser.error("--center 需要有限坐标")
    if args.time is not None and (not math.isfinite(args.time) or args.time < 0):
        parser.error("--time 需要非负的有限时间")
    if not any(p["phase"] == "unload" and p["group_id"] == args.anchor_group for p in poses):
        parser.error(f"下料 G{args.anchor_group} 缺少四槽实测姿态，请另选 --anchor-group")
    center, selected_at = args.center, None
    if center is None:
        initial_time = args.time if args.time is not None else poses[0]["at"]
        print("请在点选窗口单击机心，按 Enter 确认。A/D 换画面，Esc 取消。", flush=True)
        center, selected_at = select_machine_center(Path(summary["source"]), initial_time)
        if center is None:
            return
    anchor_slot, records = compare_distances(poses, np.asarray(center), args.anchor_group)
    scored = {(record["phase"], record["group_id"]) for record in records}
    unscored = [{"phase": group["phase"], "group_id": group["group_id"]}
                for group in summary["closed_groups"] if (group["phase"], group["group_id"]) not in scored]
    output = args.result_dir / "machine_center_distance_trial.json"
    output.write_text(json.dumps({
        "source": summary["source"], "center_xy": list(center), "center_selected_at": selected_at,
        "anchor_group": args.anchor_group, "nearest_slot_from_anchor": anchor_slot,
        "groups": records, "groups_without_four_observed_slots": unscored,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"机心 {tuple(center)}，下料 G{args.anchor_group} 校准得到最近槽为 S{anchor_slot}。")
    for record in records:
        phase = "下料" if record["phase"] == "unload" else "上料"
        distances = ", ".join(f"S{i}={value:.2f}" for i, value in enumerate(record["distances_px"], 1))
        print(f"{phase} G{record['group_id']}  {distances}  最近槽原号 S{record['nearest_old_slot']}  "
              f"前两名相差 {record['distance_gap_px']:.2f}px  "
              f"顺序 {record['observed_sequence']} → {record['trial_sequence']}")
    if unscored:
        skipped = ", ".join(f"{'下料' if row['phase'] == 'unload' else '上料'} G{row['group_id']}"
                            for row in unscored)
        print(f"缺少四个实测槽，未参与距离试算：{skipped}")
    print(f"试验结果已保存到 {output}")


if __name__ == "__main__":
    main()
