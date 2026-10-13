"""Audit dataset inclusion and compare exp3.1/exp3.2 on shared conditions."""
from __future__ import annotations

import hashlib
import csv
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import torch
import yaml

RUNS = Path("C:/Users/win10/xanylabeling_data/trainer/ultralytics/runs/detect")
OUTPUT = ROOT / "industrial_inspection/video_results/exp31_exp32_analysis"
VIDEOS = [Path("I:/youyan/test_dataset/camera_B_20260918_144258.avi"),
          Path("I:/youyan/test_dataset/camera_B_20260918_161258.avi")]


def load_samples(data_root):
    records = {}
    for split in ("train", "val"):
        for image in sorted((data_root / "images" / split).iterdir()):
            if not image.is_file():
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            label = data_root / "labels" / split / (image.stem + ".txt")
            rows = [list(map(float, row.split())) for row in label.read_text(encoding="utf-8").splitlines() if row.strip()]
            if digest in records:
                raise ValueError(f"Duplicate identical image within dataset: {image}")
            records[digest] = {"name": image.name, "split": split, "path": str(image), "labels": rows}
    return records


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(union, 1e-12)


def xyxy(row):
    _, x, y, w, h = row
    return [x-w/2, y-h/2, x+w/2, y+h/2]


def dataset_stats(samples, names):
    counts = Counter()
    image_counts = Counter()
    shapes = {}
    suspicious = []
    for sample in samples.values():
        labels = sample["labels"]
        counts.update(names[int(row[0])] for row in labels)
        image_counts.update(set(names[int(row[0])] for row in labels))
        for a, b in itertools.combinations(labels, 2):
            if int(a[0]) not in (3, 4) or int(b[0]) not in (3, 4):
                continue
            overlap = iou(xyxy(a), xyxy(b))
            if overlap > .85:
                suspicious.append({"image": sample["name"], "split": sample["split"], "iou": overlap,
                                   "class_a": names[int(a[0])], "class_b": names[int(b[0])], "a": a, "b": b})
        for row in labels:
            name = names[int(row[0])]
            shapes.setdefault(name, []).append(row[3] * row[4])
    return {"images": len(samples), "images_per_class": dict(image_counts), "objects": dict(counts),
            "empty_label_images": sum(not s["labels"] for s in samples.values()),
            "high_overlap_slot_label_pairs": suspicious,
            "mean_normalized_box_area": {name: sum(areas)/len(areas) for name, areas in shapes.items()}}


def best_csv_epoch(run):
    """Finalized checkpoints may contain the full history, including later epochs."""
    with (run / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return int(float(best["epoch"]))


def sequence_distribution(samples):
    counts = {}
    for split in ("all", "train", "val"):
        selected = [s for s in samples.values() if split == "all" or s["split"] == split]
        groups = Counter(int(re.search(r"frame_(\d+)", s["name"])[1]) // 100000 for s in selected)
        counts[split] = {str(group): count for group, count in sorted(groups.items())}
    return counts


def evaluate_fixed_thresholds(models, samples, thresholds=(.25, .6)):
    """Count class-correct matches at IoU>=0.5 with fixed confidence settings."""
    output = {}
    for name, model in models.items():
        totals = {threshold: {class_name: Counter() for class_name in model.names.values()}
                  for threshold in thresholds}
        for sample in samples:
            frame = cv2.imread(sample["path"])
            if frame is None:
                raise RuntimeError(f"Cannot read {sample['path']}")
            h, w = frame.shape[:2]
            labels = [(model.names[int(row[0])], [value * (w if index % 2 == 0 else h)
                       for index, value in enumerate(xyxy(row))]) for row in sample["labels"]]
            result = model.predict(frame, imgsz=1280, device=0, conf=min(thresholds), iou=.7,
                                   max_det=300, verbose=False)[0]
            predictions = [(model.names[int(c)], score, box) for c, score, box in
                           zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]
            for threshold, counts in totals.items():
                selected = [p for p in predictions if p[1] >= threshold]
                candidates = sorted([(iou(p[2], target[1]), pi, ti)
                    for pi, p in enumerate(selected) for ti, target in enumerate(labels)
                    if p[0] == target[0] and iou(p[2], target[1]) >= .5], reverse=True)
                used_predictions, used_targets = set(), set()
                for overlap, pi, ti in candidates:
                    if pi in used_predictions or ti in used_targets:
                        continue
                    used_predictions.add(pi)
                    used_targets.add(ti)
                    counts[selected[pi][0]]["tp"] += 1
                for pi, p in enumerate(selected):
                    if pi not in used_predictions:
                        counts[p[0]]["fp"] += 1
                for ti, target in enumerate(labels):
                    if ti not in used_targets:
                        counts[target[0]]["fn"] += 1
        output[name] = {}
        for threshold, counts in totals.items():
            output[name][str(threshold)] = {class_name: {"tp": count["tp"], "fp": count["fp"],
                "fn": count["fn"], "precision": count["tp"] / max(count["tp"]+count["fp"], 1),
                "recall": count["tp"] / max(count["tp"]+count["fn"], 1)}
                for class_name, count in counts.items()}
    return {"images": len(samples), "match_iou": .5, "nms_iou": .7,
            "imgsz": 1280, "class_aware_greedy_iou_matching": True, "results": output}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator

    configs, samples, models = {}, {}, {}
    report = {"dataset_audit": {}, "evaluations": {}, "video_samples": [],
              "video_ground_truth_verified": False, "runtime": "repository ultralytics 8.1.34; both checkpoints saved by 8.4.163"}
    for name in ("exp3.1", "exp3.2"):
        configs[name] = yaml.safe_load((RUNS / name / "args.yaml").read_text())
        data_yaml = Path(configs[name]["data"])
        data = yaml.safe_load(data_yaml.read_text())
        samples[name] = load_samples(Path(data["path"]))
        model = YOLO(str(RUNS / name / "weights/best.pt"))
        models[name] = model
        report["dataset_audit"][name] = {
            "dataset": str(data_yaml), "names": model.names,
            "statistics": dataset_stats(samples[name], model.names),
            "train_images": sum(s["split"] == "train" for s in samples[name].values()),
            "val_images": sum(s["split"] == "val" for s in samples[name].values()),
            "best_saved_metrics": model.ckpt.get("train_metrics"),
            "best_epoch": best_csv_epoch(RUNS / name),
            "sequence_distribution": sequence_distribution(samples[name])}
    report["config_differences"] = {key: {name: configs[name].get(key) for name in configs}
                                    for key in set(configs["exp3.1"]) | set(configs["exp3.2"])
                                    if configs["exp3.1"].get(key) != configs["exp3.2"].get(key)}
    old, new = samples["exp3.1"], samples["exp3.2"]
    common = set(old) & set(new)
    added = {key: value for key, value in new.items() if key not in old}
    transitions = Counter(old[key]["split"] + "->" + new[key]["split"] for key in common)
    changed = [{"name": old[key]["name"], "old_rows": old[key]["labels"], "new_rows": new[key]["labels"]}
               for key in sorted(common) if sorted(old[key]["labels"]) != sorted(new[key]["labels"])]
    report["inclusion"] = {"old_images": len(old), "new_images": len(new), "common_images": len(common),
                           "missing_old_images": [old[key]["name"] for key in old.keys()-new.keys()],
                           "added_images": len(added), "split_transitions": dict(transitions),
                           "changed_annotations": changed, "added_statistics": dataset_stats(added, models["exp3.2"].names)}
    shared_val = [old[key] for key in common if old[key]["split"] == new[key]["split"] == "val"]
    report["shared_val_images"] = len(shared_val)
    evaluation_sets = {name + "_val": configs[name]["data"] for name in configs}
    if shared_val:
        listing = OUTPUT / "shared_val_images.txt"
        listing.write_text("\n".join(s["path"] for s in sorted(shared_val, key=lambda s: s["name"])) + "\n", encoding="utf-8")
        common_yaml = OUTPUT / "shared_val.yaml"
        common_yaml.write_text(yaml.safe_dump({"path": str(OUTPUT), "train": str(listing), "val": str(listing),
                                               "names": list(models["exp3.1"].names.values())}), encoding="utf-8")
        evaluation_sets["shared_val"] = str(common_yaml)
    report["evaluation_limitations"] = "Shared val was used for checkpoint selection by both models and consists of neighboring video frames. Cross-val sets also contain training images for the other model. Neither is an independent new-video test."

    def save_report():
        (OUTPUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save_report()
    report["shared_val_fixed_thresholds"] = evaluate_fixed_thresholds(models, shared_val)
    save_report()
    print(json.dumps({"inclusion": {k: v for k, v in report["inclusion"].items() if k not in ("changed_annotations", "added_statistics")},
                      "changed_annotations": len(changed), "shared_val_images": len(shared_val)}, ensure_ascii=False), flush=True)
    for name, model in models.items():
        for set_name, data_yaml in evaluation_sets.items():
            captured = {}

            class AuditValidator(DetectionValidator):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    captured["validator"] = self

            metrics = model.val(data=data_yaml, split="val", imgsz=1280, device=0, batch=8,
                                workers=0, conf=.001, iou=.7, max_det=300, plots=True,
                                validator=AuditValidator,
                                project=str(OUTPUT), name=name.replace(".", "_") + "_" + set_name, exist_ok=True)
            matrix = captured["validator"].confusion_matrix
            report["evaluations"][name + "/" + set_name] = {
                "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
                "per_class": {model.names[int(c)]: {"precision": float(metrics.box.p[i]),
                    "recall": float(metrics.box.r[i]), "ap50": float(metrics.box.ap50[i]), "ap50_95": float(metrics.box.ap[i])}
                    for i, c in enumerate(metrics.box.ap_class_index)},
                "confusion_conf": matrix.conf, "confusion_iou": matrix.iou_thres,
                "confusion_matrix_predicted_rows_true_cols": matrix.matrix.tolist()}
            save_report()
    for video in VIDEOS:
        if not video.is_file():
            continue
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        montage = []
        try:
            # Twelve evenly spaced frames plus selected operation frames.
            indices = sorted(set([round(i * (total - 1) / 11) for i in range(12)] + [1500, 2500, 3500, 6500]))
            for index in indices:
                if index >= total:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    continue
                pair = []
                for name, model in models.items():
                    result = model.predict(frame, imgsz=1280, device=0, conf=.25, iou=.7, max_det=300, verbose=False)[0]
                    objects = [{"class": model.names[int(c)], "confidence": float(score), "xyxy": box}
                               for c, score, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]
                    pairs = [{"iou": iou(a["xyxy"], b["xyxy"]), "class_a": a["class"], "class_b": b["class"]}
                             for a, b in itertools.combinations(objects, 2)
                             if a["class"].startswith("wafer_slot") and b["class"].startswith("wafer_slot")
                             and iou(a["xyxy"], b["xyxy"]) > .85]
                    report["video_samples"].append({"source": str(video), "frame": index, "time_seconds": index/fps,
                        "model": name, "detections": objects, "high_overlap_slot_pairs": pairs,
                        "counts_conf_025": dict(Counter(o["class"] for o in objects)),
                        "counts_conf_060": dict(Counter(o["class"] for o in objects if o["confidence"] >= .6))})
                    counts = Counter(o["class"] for o in objects)
                    plotted = result.plot(line_width=1, labels=False)
                    cv2.rectangle(plotted, (0,0), (640,44), (0,0,0), -1)
                    cv2.putText(plotted, f'{name} frame {index} conf=0.25', (5,18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                    cv2.putText(plotted, f"hand:{counts['hand']} handle:{counts['hatch_handle']} gun:{counts['water_gun']} empty:{counts['wafer_slot_empty']} filled:{counts['wafer_slot_filled']}",
                                (5,38), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                    cv2.imwrite(str(OUTPUT / f"{video.stem}_{index}_{name}.jpg"), plotted)
                    pair.append(cv2.resize(plotted, (480,360)))
                montage.append(cv2.hconcat(pair))
            for start in range(0, len(montage), 4):
                cv2.imwrite(str(OUTPUT / f"{video.stem}_comparison_{start//4+1}.jpg"), cv2.vconcat(montage[start:start+4]))
        finally:
            cap.release()
        save_report()
    print(f"Saved analysis to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
