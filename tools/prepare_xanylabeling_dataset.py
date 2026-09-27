#!/usr/bin/env python3
r"""Prepare X-AnyLabeling annotations for Ultralytics training.

The script supports two workflows:

1. Workspace mode: validate X-AnyLabeling image/JSON pairs and generate a
   names-only YAML file accepted by X-AnyLabeling's built-in trainer.
2. Prepared-dataset mode: additionally convert rectangle annotations to YOLO
   text labels and split train/val by source-video group, so adjacent frames
   from one video never appear in both splits.

Examples:

    python tools/prepare_xanylabeling_dataset.py ^
        --images-dir D:\inspection\frames ^
        --classes-yaml industrial_inspection\configs\classes.yaml

    python tools/prepare_xanylabeling_dataset.py ^
        --images-dir D:\inspection\frames ^
        --classes-yaml industrial_inspection\configs\classes.yaml ^
        --output-dataset D:\inspection\yolo_dataset

Frame names should retain their video prefix, for example
``batch01_000001.jpg``. Use ``--group-regex`` when a different naming scheme
is used.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_CLASSES = [
    "hand",
    "hatch_handle",
    "water_gun",
    "gun_nozzle",
    "marble_wall",
    "planetary_plate",
    "planetary_ring",
    "wafer",
    "wafer_slot",
]

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
AUTO_GROUP_PATTERN = re.compile(
    r"^(.*?)(?:[_-](?:frame[_-]?)?)\d{3,}$", re.IGNORECASE
)


class PreparationError(RuntimeError):
    """Raised when input data cannot be prepared safely."""


@dataclass(frozen=True)
class Sample:
    image: Path
    relative_image: Path
    annotation: Path | None
    group: str
    width: int
    height: int
    yolo_rows: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate X-AnyLabeling JSON files, generate classes.yaml, and "
            "optionally create a video-grouped Ultralytics dataset."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing extracted images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help=(
            "Directory containing X-AnyLabeling JSON files. Defaults to "
            "--images-dir. Relative subdirectories are preserved."
        ),
    )
    parser.add_argument(
        "--classes-yaml",
        type=Path,
        default=Path("classes.yaml"),
        help="Names-only YAML for X-AnyLabeling workspace training.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Ordered class names. Their positions become YOLO class IDs.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        help=(
            "Optional new directory for a prepared YOLO train/val dataset. "
            "The directory must be absent or empty."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of video groups assigned to train (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to split video groups (default: 42).",
    )
    parser.add_argument(
        "--group-regex",
        help=(
            "Regex used on each image stem to derive the source-video group. "
            "It must contain one capture group, e.g. ^(.*)_\\d{6}$."
        ),
    )
    parser.add_argument(
        "--allow-missing-json",
        action="store_true",
        help=(
            "Treat an image without JSON as an intentional negative sample. "
            "Without this flag, missing JSON is an error."
        ),
    )
    parser.add_argument(
        "--force-yaml",
        action="store_true",
        help="Allow replacement of an existing different classes YAML.",
    )
    return parser.parse_args()


def validate_classes(classes: Sequence[str]) -> list[str]:
    cleaned = [name.strip() for name in classes]
    if not cleaned or any(not name for name in cleaned):
        raise PreparationError("Class names must be non-empty.")
    duplicates = sorted(name for name, count in Counter(cleaned).items() if count > 1)
    if duplicates:
        raise PreparationError(f"Duplicate class names: {', '.join(duplicates)}")
    invalid = [name for name in cleaned if not re.fullmatch(r"[a-z][a-z0-9_]*", name)]
    if invalid:
        raise PreparationError(
            "Use lowercase snake_case class names. Invalid: " + ", ".join(invalid)
        )
    return cleaned


def find_images(images_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise PreparationError(f"No supported images found under: {images_dir}")
    return images


def find_annotation(image: Path, images_dir: Path, labels_dir: Path) -> Path | None:
    relative_json = image.relative_to(images_dir).with_suffix(".json")
    candidates = [labels_dir / relative_json]
    flat_candidate = labels_dir / f"{image.stem}.json"
    if flat_candidate not in candidates:
        candidates.append(flat_candidate)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def derive_group(relative_image: Path, group_pattern: re.Pattern[str] | None) -> str:
    stem = relative_image.stem
    parent = relative_image.parent.as_posix()
    if group_pattern is not None:
        match = group_pattern.search(stem)
        if not match or match.lastindex is None:
            raise PreparationError(
                f"--group-regex did not capture a group for: {relative_image}"
            )
        prefix = match.group(1).strip()
        if not prefix:
            raise PreparationError(f"Empty video group for: {relative_image}")
    else:
        match = AUTO_GROUP_PATTERN.match(stem)
        prefix = match.group(1).strip("_-") if match else stem
        if not prefix and parent != ".":
            return parent
        if not prefix:
            prefix = stem

    return prefix if parent == "." else f"{parent}/{prefix}"


def read_image_size(annotation_data: dict, image: Path) -> tuple[int, int]:
    width = annotation_data.get("imageWidth")
    height = annotation_data.get("imageHeight")
    if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0:
        return width, height

    try:
        from PIL import Image
    except ImportError as exc:
        raise PreparationError(
            f"Missing imageWidth/imageHeight in annotation for {image}; "
            "install Pillow or resave the annotation in X-AnyLabeling."
        ) from exc

    with Image.open(image) as opened:
        return opened.size


def rectangle_to_yolo(
    points: object,
    width: int,
    height: int,
    class_id: int,
    source: Path,
) -> str:
    if not isinstance(points, list) or len(points) < 2:
        raise PreparationError(f"Rectangle has fewer than two points: {source}")
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (IndexError, TypeError, ValueError) as exc:
        raise PreparationError(f"Invalid rectangle points in: {source}") from exc

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))
    if x2 <= x1 or y2 <= y1:
        raise PreparationError(f"Zero-area rectangle in: {source}")

    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return (
        f"{class_id} {center_x:.6f} {center_y:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def parse_annotation(
    annotation: Path | None,
    image: Path,
    class_to_id: dict[str, int],
) -> tuple[int, int, tuple[str, ...], Counter[str]]:
    if annotation is None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise PreparationError(
                "Pillow is required for negative samples without JSON."
            ) from exc
        with Image.open(image) as opened:
            width, height = opened.size
        return width, height, (), Counter()

    try:
        with annotation.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Cannot parse annotation JSON: {annotation}") from exc
    if not isinstance(data, dict):
        raise PreparationError(f"Annotation root must be an object: {annotation}")

    width, height = read_image_size(data, image)
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        raise PreparationError(f"'shapes' must be a list: {annotation}")

    rows: list[str] = []
    counts: Counter[str] = Counter()
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise PreparationError(f"Invalid shape #{index}: {annotation}")
        label = shape.get("label")
        if label not in class_to_id:
            raise PreparationError(
                f"Unknown label {label!r} in {annotation}. "
                "Add it to --classes or correct the annotation."
            )
        shape_type = shape.get("shape_type", "polygon")
        if shape_type != "rectangle":
            raise PreparationError(
                f"Detect training accepts rectangles only; found {shape_type!r} "
                f"for label {label!r} in {annotation}."
            )
        rows.append(
            rectangle_to_yolo(
                shape.get("points"), width, height, class_to_id[label], annotation
            )
        )
        counts[label] += 1
    return width, height, tuple(rows), counts


def collect_samples(
    images_dir: Path,
    labels_dir: Path,
    classes: Sequence[str],
    group_pattern: re.Pattern[str] | None,
    allow_missing_json: bool,
) -> tuple[list[Sample], Counter[str], list[Path]]:
    class_to_id = {name: index for index, name in enumerate(classes)}
    samples: list[Sample] = []
    total_counts: Counter[str] = Counter()
    missing_annotations: list[Path] = []

    for image in find_images(images_dir):
        relative = image.relative_to(images_dir)
        annotation = find_annotation(image, images_dir, labels_dir)
        if annotation is None:
            missing_annotations.append(image)
            if not allow_missing_json:
                continue
        width, height, rows, counts = parse_annotation(
            annotation, image, class_to_id
        )
        total_counts.update(counts)
        samples.append(
            Sample(
                image=image,
                relative_image=relative,
                annotation=annotation,
                group=derive_group(relative, group_pattern),
                width=width,
                height=height,
                yolo_rows=rows,
            )
        )

    if missing_annotations and not allow_missing_json:
        preview = "\n".join(f"  - {path}" for path in missing_annotations[:10])
        extra = max(0, len(missing_annotations) - 10)
        suffix = f"\n  ... and {extra} more" if extra else ""
        raise PreparationError(
            "Images without X-AnyLabeling JSON were found. They may be "
            "unlabeled rather than true negatives. Label/save them first, or "
            "rerun with --allow-missing-json if they are intentional negatives:\n"
            f"{preview}{suffix}"
        )
    return samples, total_counts, missing_annotations


def render_names_yaml(classes: Sequence[str]) -> str:
    lines = ["names:"]
    lines.extend(f"  - {json.dumps(name, ensure_ascii=False)}" for name in classes)
    return "\n".join(lines) + "\n"


def write_classes_yaml(path: Path, classes: Sequence[str], force: bool) -> None:
    content = render_names_yaml(classes)
    if path.exists():
        old_content = path.read_text(encoding="utf-8-sig")
        if old_content != content and not force:
            raise PreparationError(
                f"Refusing to replace a different YAML: {path}. "
                "Use --force-yaml after checking the class order."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise PreparationError(
            f"Output dataset is not empty: {path}. Choose a new directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def split_video_groups(
    samples: Sequence[Sample], train_ratio: float, seed: int
) -> tuple[set[str], set[str]]:
    if not 0.05 <= train_ratio <= 0.95:
        raise PreparationError("--train-ratio must be between 0.05 and 0.95.")
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 2:
        only_group = groups[0] if groups else "<none>"
        raise PreparationError(
            "A video-safe train/val split requires at least two source-video "
            f"groups; found only {only_group!r}. Keep the classes YAML for "
            "workspace training, or add frames from another video."
        )
    random.Random(seed).shuffle(groups)
    train_count = round(len(groups) * train_ratio)
    train_count = max(1, min(len(groups) - 1, train_count))
    return set(groups[:train_count]), set(groups[train_count:])


def safe_output_stem(relative_image: Path) -> str:
    parts = list(relative_image.with_suffix("").parts)
    return "__".join(part.replace(" ", "_") for part in parts)


def prepare_dataset(
    output: Path,
    samples: Sequence[Sample],
    classes: Sequence[str],
    train_ratio: float,
    seed: int,
) -> tuple[Counter[str], dict[str, set[str]]]:
    train_groups, val_groups = split_video_groups(samples, train_ratio, seed)
    ensure_empty_output(output)
    split_groups = {"train": train_groups, "val": val_groups}
    split_counts: Counter[str] = Counter()
    used_names: set[str] = set()

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for sample in samples:
        split = "train" if sample.group in train_groups else "val"
        output_stem = safe_output_stem(sample.relative_image)
        if output_stem in used_names:
            raise PreparationError(
                f"Flattened output filename collision for: {sample.relative_image}"
            )
        used_names.add(output_stem)
        destination_image = (
            output / "images" / split / f"{output_stem}{sample.image.suffix.lower()}"
        )
        destination_label = output / "labels" / split / f"{output_stem}.txt"
        shutil.copy2(sample.image, destination_image)
        destination_label.write_text(
            "\n".join(sample.yolo_rows) + ("\n" if sample.yolo_rows else ""),
            encoding="utf-8",
        )
        split_counts[split] += 1

    data_yaml = [
        f"path: {json.dumps(output.resolve().as_posix(), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "",
        render_names_yaml(classes).rstrip(),
        "",
    ]
    (output / "data.yaml").write_text("\n".join(data_yaml), encoding="utf-8")
    return split_counts, split_groups


def print_summary(
    samples: Sequence[Sample],
    classes: Sequence[str],
    counts: Counter[str],
    missing_annotations: Sequence[Path],
) -> None:
    groups = defaultdict(int)
    for sample in samples:
        groups[sample.group] += 1

    print(f"Images validated: {len(samples)}")
    print(f"Video groups: {len(groups)}")
    for group, count in sorted(groups.items()):
        print(f"  {group}: {count} images")
    print("Class instances:")
    for class_id, name in enumerate(classes):
        print(f"  {class_id:>2}  {name:<24} {counts[name]}")
    if missing_annotations:
        print(f"Intentional empty-label images: {len(missing_annotations)}")


def main() -> int:
    args = parse_args()
    try:
        images_dir = args.images_dir.expanduser().resolve()
        labels_dir = (args.labels_dir or args.images_dir).expanduser().resolve()
        if not images_dir.is_dir():
            raise PreparationError(f"Images directory does not exist: {images_dir}")
        if not labels_dir.is_dir():
            raise PreparationError(f"Labels directory does not exist: {labels_dir}")

        classes = validate_classes(args.classes)
        group_pattern = re.compile(args.group_regex) if args.group_regex else None
        if group_pattern is not None and group_pattern.groups < 1:
            raise PreparationError("--group-regex must contain a capture group.")

        samples, counts, missing = collect_samples(
            images_dir=images_dir,
            labels_dir=labels_dir,
            classes=classes,
            group_pattern=group_pattern,
            allow_missing_json=args.allow_missing_json,
        )
        print_summary(samples, classes, counts, missing)

        classes_yaml = args.classes_yaml.expanduser().resolve()
        write_classes_yaml(classes_yaml, classes, args.force_yaml)
        print(f"Workspace classes YAML: {classes_yaml}")

        if args.output_dataset:
            output_dataset = args.output_dataset.expanduser().resolve()
            split_counts, split_groups = prepare_dataset(
                output=output_dataset,
                samples=samples,
                classes=classes,
                train_ratio=args.train_ratio,
                seed=args.seed,
            )
            print(f"Prepared dataset YAML: {output_dataset / 'data.yaml'}")
            print(
                "Split summary: "
                f"train={split_counts['train']} images/"
                f"{len(split_groups['train'])} groups, "
                f"val={split_counts['val']} images/"
                f"{len(split_groups['val'])} groups"
            )
        return 0
    except (PreparationError, re.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
