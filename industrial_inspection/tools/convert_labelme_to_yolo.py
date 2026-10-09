"""Convert LabelMe 4.x JSON files to a YOLO detection dataset.

The source directory may contain images and their JSON annotations together.
Only rectangle shapes whose labels occur in the supplied classes YAML are
converted. Polygon labels (for example ``hatch``) are reported and skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


def read_class_names(path: Path) -> list[str]:
    """Keep exactly the class order from the supplied YAML."""
    names = yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"No class names found in {path}")
    return names


def find_image(source: Path, annotation: dict, stem: str) -> Path | None:
    image_path = annotation.get("imagePath")
    candidates = []
    if image_path:
        candidates.append(source / Path(image_path).name)
    candidates.extend(source / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def yolo_box(points: list[list[float]], width: float, height: float) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    if not all(math.isfinite(v) for v in xs + ys):
        return None
    x1, x2 = max(0.0, min(xs)), min(width, max(xs))
    y1, y2 = max(0.0, min(ys)), min(height, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--ignored-classes", type=Path, required=True,
                        help="Separate segmentation classes intentionally excluded from detection.")
    parser.add_argument("--frame-gap", type=int, default=1000,
                        help="A frame-number gap greater than this starts a separate sequence.")
    parser.add_argument("--group-regex", help="Optional regex with one capture group for known source videos.")
    args = parser.parse_args()

    if args.val < 0 or args.test < 0 or args.val + args.test >= 1:
        raise ValueError("val + test must be less than 1")
    class_names = read_class_names(args.classes)
    ignored_names = set(read_class_names(args.ignored_classes))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be a new or empty directory to prevent stale split files")
    class_ids = {name: i for i, name in enumerate(class_names)}
    json_files = sorted(args.source.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {args.source}")

    records = []
    stats = Counter()
    seen_stems = set()
    for annotation_path in json_files:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        image = find_image(args.source, annotation, annotation_path.stem)
        if image is None:
            raise FileNotFoundError(f"Missing image for {annotation_path}")
        if image.stem in seen_stems:
            raise ValueError(f"Multiple annotations refer to {image}")
        seen_stems.add(image.stem)
        width = float(annotation.get("imageWidth") or 0)
        height = float(annotation.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid imageWidth/imageHeight in {annotation_path}")
        with Image.open(image) as actual:
            if actual.size != (width, height):
                raise ValueError(f"Image dimensions disagree with JSON: {image}")
            actual.verify()
        labels = []
        for shape in annotation.get("shapes", []):
            label = shape.get("label")
            shape_type = shape.get("shape_type", "rectangle")
            if label in ignored_names:
                stats[f"skipped_label:{label}"] += 1
                continue
            if label not in class_ids:
                raise ValueError(f"Unknown label {label!r} in {annotation_path}")
            if shape_type != "rectangle":
                raise ValueError(f"Detection label {label!r} is {shape_type!r}: {annotation_path}")
            box = yolo_box(shape.get("points", []), width, height)
            if box is None:
                raise ValueError(f"Invalid box for {label!r}: {annotation_path}")
            labels.append((class_ids[label], box))
            stats[label] += 1
        records.append((image, labels))

    groups = {}
    if args.group_regex:
        pattern = re.compile(args.group_regex)
        for record in records:
            match = pattern.search(record[0].stem)
            if match is None or pattern.groups != 1:
                raise ValueError(f"Group regex must match {record[0].name} with one capture group")
            groups.setdefault(match[1], []).append(record)
    else:
        # This dataset uses frame_00003 / frame_100046 / frame_200000 names.
        # Infer disjoint sequences at large numbering gaps; never shuffle
        # individual frames. Actual video IDs are preferable when available.
        numbered = []
        for record in records:
            match = re.fullmatch(r"(.*?)(\d+)", record[0].stem)
            if match is None:
                raise ValueError("Use --group-regex for names without trailing frame numbers")
            numbered.append((match[1], int(match[2]), record))
        last_prefix, last_number, group_id = None, None, -1
        for prefix, number, record in sorted(numbered, key=lambda row: (row[0], row[1])):
            if prefix != last_prefix or number - last_number > args.frame_gap:
                group_id += 1
            groups.setdefault(f"sequence_{group_id}", []).append(record)
            last_prefix, last_number = prefix, number
    if len(groups) < 3:
        raise ValueError("At least three source groups are needed for train/val/test")

    rng = random.Random(args.seed)
    group_order = list(groups)
    rng.shuffle(group_order)
    group_order.sort(key=lambda key: len(groups[key]), reverse=True)
    targets = {"train": len(records) * (1 - args.val - args.test),
               "val": len(records) * args.val, "test": len(records) * args.test}
    splits = {name: [] for name in targets}
    group_assignments = {}
    for key in group_order:
        split = max(targets, key=lambda name: targets[name] - len(splits[name]))
        splits[split].extend(groups[key])
        group_assignments[key] = split
    if any(not items for items in splits.values()):
        raise ValueError("Group sizes do not permit three nonempty splits")

    hashes = {}
    for split, items in splits.items():
        counts = Counter(class_names[class_id] for _, labels in items for class_id, _ in labels)
        if any(counts[name] == 0 for name in class_names):
            raise ValueError(f"Missing class in {split}: {dict(counts)}; change source groups")
        for image, _ in items:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f"Identical image occurs across splits: {image}")
            hashes[digest] = split
    for split, items in splits.items():
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image, labels in items:
            target_image = args.output / "images" / split / image.name
            shutil.copy2(image, target_image)
            target_label = args.output / "labels" / split / f"{image.stem}.txt"
            target_label.write_text(
                "".join(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n" for class_id, (xc, yc, bw, bh) in labels),
                encoding="utf-8",
            )

    dataset_yaml = args.output / "dataset.yaml"
    root = args.output.resolve().as_posix()
    dataset_yaml.write_text(yaml.safe_dump({"path": root, "train": "images/train", "val": "images/val",
                                          "test": "images/test", "names": class_names}, sort_keys=False), encoding="utf-8")
    report = {"source": str(args.source.resolve()), "classes": class_names,
              "split_method": "source_regex" if args.group_regex else "inferred_frame_gap",
              "frame_gap": args.frame_gap, "seed": args.seed, "objects": dict(stats),
              "groups": {key: {"split": group_assignments[key], "images": len(items),
                                "first": items[0][0].name, "last": items[-1][0].name}
                         for key, items in groups.items()},
              "splits": {split: {"images": len(items),
                                  "objects": dict(Counter(class_names[c] for _, rows in items for c, _ in rows)),
                                  "files": [image.name for image, _ in items]}
                         for split, items in splits.items()}}
    (args.output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"classes={class_names}")
    print(f"images={len(records)} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print("objects=" + ", ".join(f"{k}:{v}" for k, v in sorted(stats.items())))
    print("groups=" + json.dumps(report["groups"]))
    print(f"dataset_yaml={dataset_yaml}")


if __name__ == "__main__":
    main()
