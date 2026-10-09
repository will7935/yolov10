"""Train the YOLOv10 detection model with this repository's data layout.

The repository predates PyTorch 2.6. The small torch.load compatibility shim
is needed when loading a trusted local YOLOv10 checkpoint with PyTorch 2.6+.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="I:/youyan/yolo_dataset/yolo_detect_grouped/dataset.yaml")
    parser.add_argument("--weights", default=str(ROOT / "yolov10s.pt"))
    parser.add_argument("--project", default="I:/youyan/yolo_dataset/runs")
    parser.add_argument("--name", default="yolov10s_detect_grouped")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", type=Path, help="Resume from this run's trusted local last.pt")
    parser.add_argument("--evaluate-only", type=Path, help="Evaluate this completed run's best.pt and save examples")

    args = parser.parse_args()
    original_load = torch.load

    def trusted_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = trusted_load

    from ultralytics import YOLOv10

    model = YOLOv10(str(args.resume) if args.resume else args.weights) if not args.evaluate_only else None
    if model is not None:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            pretrained=True,
            patience=30,
            cos_lr=True,
            close_mosaic=10,
            exist_ok=bool(args.resume),
            resume=bool(args.resume),
            seed=20261009,
            cache="ram",
            optimizer="AdamW",
            lr0=0.001,
            hsv_s=0.2,
            hsv_v=0.2,
            fliplr=0.0,
            translate=0.05,
            scale=0.25,
            mosaic=0.5,
        )

    # Select the checkpoint on val only; inspect the held-out test once.
    best_path = args.evaluate_only.resolve() if args.evaluate_only else model.trainer.best
    best = YOLOv10(str(best_path))

    def save_confusion_counts(validator):
        matrix = validator.confusion_matrix
        report = {"names": [best.names[i] for i in sorted(best.names)] + ["background"],
                  "rows": "predicted", "columns": "true", "confidence": matrix.conf,
                  "iou_threshold": matrix.iou_thres, "matrix": matrix.matrix.tolist()}
        (validator.save_dir / "confusion_counts.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    best.add_callback("on_val_end", save_confusion_counts)
    test_metrics = best.val(data=args.data, split="test", imgsz=args.imgsz,
                            batch=args.batch, device=args.device, workers=args.workers,
                            project=args.project, name=f"{best_path.parents[1].name}_test", plots=True)
    import yaml
    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    test_dir = Path(data["path"]) / data["test"]
    all_images = sorted(test_dir.iterdir())
    n_examples = min(12, len(all_images))
    sample_images = [all_images[round(i * (len(all_images) - 1) / max(1, n_examples - 1))] for i in range(n_examples)]
    best.predict(source=[str(p) for p in sample_images], imgsz=args.imgsz,
                 device=args.device, conf=0.25, save=True,
                 project=args.project, name=f"{best_path.parents[1].name}_predictions")
    summary = {
        "best_weights": str(best_path), "data": args.data,
        "test_results_dir": str(test_metrics.save_dir),
        "test_metrics": {k: float(v) for k, v in test_metrics.results_dict.items()},
        "per_class": {best.names[int(class_id)]: {
            "precision": float(test_metrics.box.p[i]), "recall": float(test_metrics.box.r[i]),
            "mAP50": float(test_metrics.box.ap50[i]), "mAP50_95": float(test_metrics.box.ap[i])}
            for i, class_id in enumerate(test_metrics.box.ap_class_index)},
    }
    history = (best.ckpt or {}).get("train_results", {})
    if history.get("epoch"):
        summary["selected_epoch"] = int(max(history["epoch"]))
    history_path = best_path.parents[1] / "results.csv"
    if history_path.is_file():
        with history_path.open(encoding="utf-8") as stream:
            rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
        summary["trained_epochs"] = int(rows[-1]["epoch"])
    summary["validation_metrics"] = (best.ckpt or {}).get("train_metrics", {})
    summary["prediction_results_dir"] = str(best.predictor.save_dir)
    summary_path = best_path.parents[1] / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
