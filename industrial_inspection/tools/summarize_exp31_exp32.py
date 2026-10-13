"""Summarize saved audit results without running inference again."""
from __future__ import annotations

import json
import itertools
from collections import Counter
from pathlib import Path

import cv2

from analyze_exp31_exp32 import (OUTPUT, RUNS, best_csv_epoch, iou,
                                load_samples, sequence_distribution, xyxy)


def main():
    source = OUTPUT / "analysis.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    for name, audit in report["dataset_audit"].items():
        audit["best_epoch"] = best_csv_epoch(RUNS / name)
        audit["sequence_distribution"] = sequence_distribution(load_samples(Path(audit["dataset"]).parent))

    original = Path("I:/youyan/yolo_dataset/images")
    old = load_samples(Path(report["dataset_audit"]["exp3.1"]["dataset"]).parent)
    new = load_samples(Path(report["dataset_audit"]["exp3.2"]["dataset"]).parent)
    metadata = {"scope": "Current source JSON metadata, not necessarily metadata at training time",
                "found": 0, "missing": 0, "checked_true": 0, "checked_false": 0,
                "images_with_nonnull_shape_score": 0}
    for digest in new.keys() - old.keys():
        path = original / (Path(new[digest]["name"]).stem + ".json")
        if not path.is_file():
            metadata["missing"] += 1
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        metadata["found"] += 1
        metadata["checked_true"] += raw.get("checked") is True
        metadata["checked_false"] += raw.get("checked") is False
        metadata["images_with_nonnull_shape_score"] += any(s.get("score") is not None for s in raw.get("shapes", []))
    report["added_source_json_metadata"] = metadata

    summary = {}
    for name in ("exp3.1", "exp3.2"):
        frames = [s for s in report["video_samples"] if s["model"] == name]
        entry = {"sample_frames": len(frames), "thresholds": {}}
        for threshold in (.25, .6):
            counts = Counter()
            overlap_frames = 0
            overlap_pairs = 0
            for sample in frames:
                objects = [o for o in sample["detections"] if o["confidence"] >= threshold]
                counts.update(o["class"] for o in objects)
                pairs = [(a, b) for a, b in itertools.combinations(objects, 2)
                         if a["class"].startswith("wafer_slot") and b["class"].startswith("wafer_slot")
                         and iou(a["xyxy"], b["xyxy"]) > .85]
                overlap_frames += bool(pairs)
                overlap_pairs += len(pairs)
            entry["thresholds"][str(threshold)] = {"detections": dict(counts),
                "high_overlap_slot_frames": overlap_frames, "high_overlap_slot_pairs": overlap_pairs}
        summary[name] = entry

    indexed = {(s["source"], s["frame"], s["model"]): s for s in report["video_samples"]}
    differences = []
    for sample in report["video_samples"]:
        if sample["model"] != "exp3.1":
            continue
        other = indexed[(sample["source"], sample["frame"], "exp3.2")]
        for a in sample["detections"]:
            if a["confidence"] < .6 or not a["class"].startswith("wafer_slot"):
                continue
            matches = [(iou(a["xyxy"], b["xyxy"]), b) for b in other["detections"]
                       if b["class"].startswith("wafer_slot")]
            if not matches:
                continue
            overlap, b = max(matches, key=lambda pair: pair[0])
            if overlap >= .5 and b["confidence"] < .6:
                differences.append({"source": sample["source"], "frame": sample["frame"],
                    "iou": overlap, "exp31": a, "exp32": b})
    report["video_sample_summary"] = summary
    report["exp31_above_060_exp32_matched_below_060"] = differences

    panels = []
    newer = Path(report["dataset_audit"]["exp3.2"]["dataset"]).parent
    for pair in report["inclusion"]["added_statistics"]["high_overlap_slot_label_pairs"]:
        image_path = newer / "images" / pair["split"] / pair["image"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Cannot read {image_path}")
        h, w = frame.shape[:2]
        annotated = frame.copy()
        bounds = []
        for row, color, text in ((pair["a"], (0, 140, 255), "empty"), (pair["b"], (255, 255, 0), "filled")):
            box = [round(v * (w if idx % 2 == 0 else h)) for idx, v in enumerate(xyxy(row))]
            bounds.append(box)
            cv2.rectangle(annotated, tuple(box[:2]), tuple(box[2:]), color, 2)
        x1 = max(0, min(b[0] for b in bounds)-40)
        y1 = max(0, min(b[1] for b in bounds)-50)
        x2 = min(w, max(b[2] for b in bounds)+40)
        y2 = min(h, max(b[3] for b in bounds)+50)
        crop = cv2.resize(annotated[y1:y2, x1:x2], (480, 240))
        context = cv2.resize(annotated, (480, 360))
        panel = cv2.copyMakeBorder(cv2.vconcat([context, crop]), 50, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, f"{pair['image']} {pair['split']} IoU={pair['iou']:.3f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.putText(panel, "orange=empty / cyan=filled", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
        panels.append(panel)
    cv2.imwrite(str(OUTPUT / "conflicting_annotations.jpg"), cv2.hconcat(panels))
    source.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"epochs": {n: a["best_epoch"] for n, a in report["dataset_audit"].items()},
        "distribution": {n: a["sequence_distribution"] for n, a in report["dataset_audit"].items()},
        "source_json_metadata": metadata,
        "video_summary": summary, "confidence_crossings": differences}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
