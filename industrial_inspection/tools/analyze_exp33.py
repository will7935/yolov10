"""Audit exp3.3 and compare the exact 300-frame video segment across models."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import torch
import yaml

from analyze_exp31_exp32 import (ROOT, RUNS, load_samples, dataset_stats,
                                sequence_distribution, evaluate_fixed_thresholds)
from predict_video import slot_keep_indices
from slot_roi import draw_roi, filter_slot_indices

VIDEO_RUN = Path(r"I:\youyan\test_dataset\camera_B_20260925_221958_20261012_180811_946913")
OUTPUT = ROOT / "industrial_inspection/video_results/exp33_analysis"


def frame_statistics(rows):
    distribution = Counter()
    filled = Counter()
    changes = 0
    previous = None
    for row in rows:
        counts = Counter(o["class_name"] for o in row["detections"])
        pair = (counts["wafer_slot_empty"], counts["wafer_slot_filled"])
        distribution[sum(pair)] += 1
        filled[pair[1]] += 1
        if previous is not None and previous != pair:
            changes += 1
        previous = pair
    return {"frames": len(rows), "slot_count_distribution": dict(sorted(distribution.items())),
            "filled_count_distribution": dict(sorted(filled.items())),
            "adjacent_empty_filled_count_changes": changes,
            "frames_with_suppression": sum(bool(r["suppressed_slots"]) for r in rows),
            "suppressed_boxes": sum(len(r["suppressed_slots"]) for r in rows),
            "roi_excluded_boxes": sum(len(r["roi_excluded_slots"]) for r in rows)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((VIDEO_RUN / "summary.json").read_text(encoding="utf-8"))
    existing = [json.loads(row) for row in (VIDEO_RUN / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
    report = {"video_run": summary, "saved_output_statistics": frame_statistics(existing),
              "training": {}, "data_changes": {}, "video_comparison": {},
              "independent_video_ground_truth": False}
    snapshots = {}
    for name in ("exp3.1", "exp3.2", "exp3.3"):
        config = yaml.safe_load((RUNS / name / "args.yaml").read_text())
        dataset = yaml.safe_load(Path(config["data"]).read_text())
        records = load_samples(Path(dataset["path"]))
        snapshots[name] = records
        names = {int(k): v for k, v in dataset["names"].items()} if isinstance(dataset["names"], dict) else dict(enumerate(dataset["names"]))
        with (RUNS / name / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            results = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
        best = max(results, key=lambda row: float(row["metrics/mAP50-95(B)"]))
        report["training"][name] = {"config": config, "best_csv_row": best,
             "epochs_completed": len(results), "dataset": dataset,
             "train_images": sum(s["split"] == "train" for s in records.values()),
             "val_images": sum(s["split"] == "val" for s in records.values()),
             "statistics": dataset_stats(records, names), "sequence_distribution": sequence_distribution(records)}
    for old_name in ("exp3.1", "exp3.2"):
        old, new = snapshots[old_name], snapshots["exp3.3"]
        common = old.keys() & new.keys()
        report["data_changes"][old_name] = {"common_images": len(common), "added": len(new.keys()-old.keys()),
            "removed": len(old.keys()-new.keys()),
            "changed_labels": [new[k]["name"] for k in common if sorted(old[k]["labels"]) != sorted(new[k]["labels"])],
            "split_transitions": dict(Counter(old[k]["split"] + "->" + new[k]["split"] for k in common))}

    def save():
        (OUTPUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    print(json.dumps({"saved_video": report["saved_output_statistics"], "training": {
        n: {"train": t["train_images"], "val": t["val_images"], "best": t["best_csv_row"],
            "conflicts": t["statistics"]["high_overlap_slot_label_pairs"]} for n,t in report["training"].items()},
        "changes": report["data_changes"]}, ensure_ascii=False, indent=2), flush=True)

    # Display output samples before doing new inference.
    cap = cv2.VideoCapture(str(VIDEO_RUN / "annotated.mp4"))
    selected = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 270, 299]
    tiles = []
    for index in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode saved video frame {index}")
        cv2.rectangle(frame, (0,0), (640,27), (0,0,0), -1)
        cv2.putText(frame, f"exp3.3 frame {index}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
        tiles.append(cv2.resize(frame, (480,360)))
    cap.release()
    for start in range(0, len(tiles), 6):
        group = tiles[start:start+6]
        while len(group) % 3:
            group.append(group[-1]*0)
        cv2.imwrite(str(OUTPUT / f"saved_video_samples_{start//6+1}.jpg"),
                    cv2.vconcat([cv2.hconcat(group[i:i+3]) for i in range(0,len(group),3)]))
    print("Saved video contact sheets", flush=True)

    original_load = torch.load
    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)
    torch.load = trusted_load
    from ultralytics import YOLO
    polygon = summary["slot_roi_polygon_normalized"]
    # Reuse exact saved exp3.3 outputs. Process preceding models under identical settings.
    compared = {"exp3.3": existing}
    for name in ("exp3.1", "exp3.2"):
        model = YOLO(str(RUNS / name / "weights/best.pt"))
        rows = []
        cap = cv2.VideoCapture(summary["source"])
        try:
            for index in range(summary["processed_frames"]):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot read source frame {index}")
                result = model.predict(frame, imgsz=summary["imgsz"], conf=summary["conf"],
                    iou=summary["nms_iou"], device=0, verbose=False)[0]
                boxes = result.boxes.cpu()
                keep, excluded = filter_slot_indices(boxes, model.names, polygon, (frame.shape[1],frame.shape[0]))
                result = result[keep]
                boxes = result.boxes.cpu()
                keep, suppressed = slot_keep_indices(boxes, model.names, summary["slot_duplicate_iou"])
                result = result[keep]
                boxes = result.boxes.cpu()
                detections = [{"class_name": model.names[int(c)], "confidence": score, "xyxy": xy}
                    for c, score, xy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]
                rows.append({"source_frame": index, "detections": detections,
                             "suppressed_slots": suppressed, "roi_excluded_slots": excluded})
                if index in selected:
                    annotated = result.plot(line_width=2, font_size=12)
                    draw_roi(annotated, polygon)
                    cv2.imwrite(str(OUTPUT / f"frame_{index}_{name}.jpg"), annotated)
        finally:
            cap.release()
        compared[name] = rows
        report["video_comparison"][name] = frame_statistics(rows)
        print(f"{name}: {report['video_comparison'][name]}", flush=True)
        save()
    report["video_comparison"]["exp3.3"] = frame_statistics(existing)
    (OUTPUT / "comparison_detections.json").write_text(json.dumps(compared, ensure_ascii=False), encoding="utf-8")
    shared = snapshots["exp3.1"].keys() & snapshots["exp3.2"].keys() & snapshots["exp3.3"].keys()
    shared_val = [snapshots["exp3.3"][k] for k in shared if all(snapshots[n][k]["split"] == "val" for n in snapshots)]
    report["three_way_shared_val_images"] = len(shared_val)
    if shared_val:
        models = {n: YOLO(str(RUNS/n/"weights/best.pt")) for n in snapshots}
        report["shared_val_fixed_thresholds"] = evaluate_fixed_thresholds(models, shared_val)
    common_pair = snapshots["exp3.2"].keys() & snapshots["exp3.3"].keys()
    pair_val = [snapshots["exp3.3"][k] for k in common_pair
                if snapshots["exp3.2"][k]["split"] == snapshots["exp3.3"][k]["split"] == "val"
                and sorted(snapshots["exp3.2"][k]["labels"]) == sorted(snapshots["exp3.3"][k]["labels"])]
    report["exp32_exp33_shared_val_fixed_thresholds"] = evaluate_fixed_thresholds(
        {n: YOLO(str(RUNS/n/"weights/best.pt")) for n in ("exp3.2", "exp3.3")}, pair_val, (.2,.35,.6))
    save()
    print(f"Finished: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
