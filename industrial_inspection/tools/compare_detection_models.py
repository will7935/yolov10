"""Diagnostic comparison; common images are NOT held out for exp3.2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import torch


def main():
    original = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = trusted_load
    from ultralytics import YOLO, YOLOv10

    output = ROOT / "industrial_inspection/video_results/model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    models = {
        "exp3_2_yolov8n": YOLO("C:/Users/win10/xanylabeling_data/trainer/ultralytics/runs/detect/exp3.2/weights/best.pt"),
        "ours_yolov10s": YOLOv10("I:/youyan/yolo_dataset/runs/yolov10s_detect_grouped/weights/best.pt"),
    }
    data = "I:/youyan/yolo_dataset/yolo_detect_grouped/dataset.yaml"
    report = {"warning": "All 142 common images were used by exp3.2: 113 train, 29 val. This is NOT an independent generalization comparison.",
              "diagnostic_metrics": {}, "video": "I:/youyan/test_dataset/camera_B_20260918_144258.avi",
              "video_has_verified_ground_truth": False, "sample_frames": [233, 1000, 1500, 2500, 3500, 4500, 6500, 7500]}
    for name, model in models.items():
        for size in (640, 1280):
            key = f"{name}_{size}"
            metrics = model.val(data=data, split="test", imgsz=size, batch=8, device=0, workers=0,
                                conf=0.001, iou=0.7, max_det=300, plots=True,
                                project=str(output), name=key, exist_ok=False)
            report["diagnostic_metrics"][key] = {
                "overall": {k: float(v) for k, v in metrics.results_dict.items()},
                "speed_ms_per_image": metrics.speed,
                "per_class": {model.names[int(c)]: {"precision": float(metrics.box.p[i]),
                    "recall": float(metrics.box.r[i]), "ap50": float(metrics.box.ap50[i]),
                    "ap50_95": float(metrics.box.ap[i])} for i, c in enumerate(metrics.box.ap_class_index)}}
            (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cap = cv2.VideoCapture(report["video"])
    panels = {640: [], 1280: []}
    video_results = []
    try:
        for index in report["sample_frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read video frame {index}")
            for size in (640, 1280):
                pair = []
                for name, model in models.items():
                    result = model.predict(frame, imgsz=size, device=0, conf=0.25, iou=0.7, max_det=300,
                                           verbose=False, save=False)[0]
                    row = {"frame": index, "model": name, "imgsz": size,
                           "detections": [{"class": model.names[int(c)], "confidence": float(s), "xyxy": box}
                               for c, s, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]}
                    video_results.append(row)
                    counts = {label: sum(x["class"] == label for x in row["detections"]) for label in model.names.values()}
                    image = result.plot(line_width=1, labels=False)
                    cv2.rectangle(image, (0, 0), (640, 44), (0, 0, 0), -1)
                    cv2.putText(image, f"{name} {size} frame {index}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                    cv2.putText(image, f"hand:{counts['hand']} handle:{counts['hatch_handle']} gun:{counts['water_gun']} empty:{counts['wafer_slot_empty']} filled:{counts['wafer_slot_filled']}",
                                (5, 38), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                    cv2.imwrite(str(output / f"{name}_{size}_frame_{index}.jpg"), image)
                    pair.append(cv2.resize(image, (480, 360)))
                panels[size].append(cv2.hconcat(pair))
    finally:
        cap.release()
    for size, images in panels.items():
        # Two four-row sheets preserve legibility.
        for start in (0, 4):
            cv2.imwrite(str(output / f"video_comparison_{size}_{start//4+1}.jpg"), cv2.vconcat(images[start:start+4]))
    report["video_samples"] = video_results
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Comparison saved to {output}")


if __name__ == "__main__":
    main()
