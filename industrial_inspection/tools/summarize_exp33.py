"""Inspect saved exp3.3 comparison, without modifying user data or settings."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from analyze_exp31_exp32 import iou
from analyze_exp33 import OUTPUT, VIDEO_RUN
from slot_roi import draw_roi


def slot_objects(row):
    return [o for o in row["detections"] if o["class_name"].startswith("wafer_slot")]


def main():
    path = OUTPUT / "analysis.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    compared = json.loads((OUTPUT / "comparison_detections.json").read_text(encoding="utf-8"))
    reversals = []
    rows = compared["exp3.3"]
    for index in range(1, len(rows)-1):
        for middle in slot_objects(rows[index]):
            before = [(iou(middle["xyxy"], o["xyxy"]), o) for o in slot_objects(rows[index-1])]
            after = [(iou(middle["xyxy"], o["xyxy"]), o) for o in slot_objects(rows[index+1])]
            if not before or not after:
                continue
            overlap_a, a = max(before, key=lambda pair: pair[0])
            overlap_b, b = max(after, key=lambda pair: pair[0])
            if min(overlap_a, overlap_b) >= .7 and a["class_name"] == b["class_name"] != middle["class_name"]:
                reversals.append({"middle_frame": index, "before": a, "middle": middle, "after": b,
                                  "min_iou": min(overlap_a, overlap_b)})
    foreground = {}
    for name, samples in compared.items():
        candidates = [{"frame": s["source_frame"], "detection": o} for s in samples for o in s["detections"]
            if o["class_name"] == "hand" and o["xyxy"][1] >= 350 and o["xyxy"][2]-o["xyxy"][0] >= 200]
        foreground[name] = {"rule": "hand box y1>=350px and width>=200px; candidates require visual verification",
                            "count": len(candidates),
                            "count_conf_060": sum(c["detection"]["confidence"] >= .6 for c in candidates),
                            "max_confidence": max((c["detection"]["confidence"] for c in candidates),default=None),
                            "examples": candidates[:5]}
    report["exp33_single_frame_slot_class_reversals"] = reversals
    report["foreground_large_hand_candidates"] = foreground

    source_cap = cv2.VideoCapture(report["video_run"]["source"])
    saved_cap = cv2.VideoCapture(str(VIDEO_RUN / "annotated.mp4"))
    tiles = []
    selected = [0, 100, 160, 200, 299]
    for index in selected:
        row = []
        for name in ("exp3.1", "exp3.2", "exp3.3"):
            if name == "exp3.3":
                saved_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = saved_cap.read()
                if not ok:
                    raise RuntimeError("Cannot decode saved output")
            else:
                frame = cv2.imread(str(OUTPUT / f"frame_{index}_{name}.jpg"))
            cv2.rectangle(frame, (0,0), (640,45), (0,0,0), -1)
            counts = {cls: sum(o["class_name"] == cls for o in compared[name][index]["detections"])
                      for cls in ("wafer_slot_empty", "wafer_slot_filled")}
            cv2.putText(frame, f"{name} frame {index} empty:{counts['wafer_slot_empty']} filled:{counts['wafer_slot_filled']}",
                        (5,22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
            row.append(cv2.resize(frame,(480,360)))
        tiles.append(cv2.hconcat(row))
    cv2.imwrite(str(OUTPUT / "three_model_comparison.jpg"), cv2.vconcat(tiles))

    if reversals:
        candidate = max(reversals, key=lambda r: min(r[k]["confidence"] for k in ("before","middle","after")))
        report["strongest_single_frame_reversal"] = candidate
        panels = []
        for index, key in zip(range(candidate["middle_frame"]-1,candidate["middle_frame"]+2), ("before","middle","after")):
            source_cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok, frame = source_cap.read()
            if not ok:
                raise RuntimeError("Cannot read reversal source frame")
            obj = candidate[key]
            coords = [round(v) for v in obj["xyxy"]]
            cv2.rectangle(frame, tuple(coords[:2]), tuple(coords[2:]), (0,255,255), 2)
            draw_roi(frame, report["video_run"]["slot_roi_polygon_normalized"])
            panel = cv2.copyMakeBorder(frame, 45,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(panel, f"frame {index}: {obj['class_name']} {obj['confidence']:.3f}",
                        (5,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            panels.append(panel)
        cv2.imwrite(str(OUTPUT / "single_frame_state_reversal.jpg"),cv2.hconcat(panels))
    source_cap.release()
    saved_cap.release()
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"single_frame_reversals": len(reversals), "strongest": report.get("strongest_single_frame_reversal"),
                     "foreground_candidates":foreground},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
