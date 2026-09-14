import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO dataset contract")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--expected-train-images", type=int, default=0)
    parser.add_argument("--expected-val-images", type=int, default=0)
    return parser.parse_args()


def resolve_split(data_yaml: Path, config: dict, split_name: str) -> Path:
    split_value = config.get(split_name)
    if not isinstance(split_value, str):
        raise ValueError(f"{split_name} must be one relative folder path")
    configured_root = Path(config.get("path", "."))
    split_path = Path(split_value)
    if configured_root.is_absolute() or split_path.is_absolute():
        raise ValueError("Dataset root and split paths must be relative")
    yaml_root = data_yaml.parent.resolve()
    resolved_path = (yaml_root / configured_root / split_path).resolve()
    if resolved_path != yaml_root and yaml_root not in resolved_path.parents:
        raise ValueError(f"{split_name} resolves outside the dataset root")
    return resolved_path


def image_digest(image_path: Path) -> str:
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Invalid image file: {image_path}") from error
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def validate_label(label_path: Path, class_count: int) -> int:
    object_count = 0
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain 5 values")
        class_value, *coordinates = values
        try:
            class_id = int(class_value)
            box = [float(value) for value in coordinates]
        except ValueError as error:
            raise ValueError(
                f"{label_path}:{line_number} contains a non-numeric value"
            ) from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number} has invalid class {class_id}")
        if not all(math.isfinite(value) for value in box):
            raise ValueError(
                f"{label_path}:{line_number} contains a non-finite coordinate"
            )
        if any(value < 0 or value > 1 for value in box):
            raise ValueError(
                f"{label_path}:{line_number} has coordinates outside 0..1"
            )
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"{label_path}:{line_number} has an empty bounding box")
        object_count += 1
    return object_count


def validate_dataset(data_yaml: Path) -> dict[str, int]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping")
    names = config.get("names")
    if not isinstance(names, (dict, list)) or not names:
        raise ValueError("Dataset YAML must define non-empty class names")
    if isinstance(names, dict):
        try:
            class_ids = {int(class_id) for class_id in names}
        except (TypeError, ValueError) as error:
            raise ValueError("Dataset class IDs must be integers") from error
        if class_ids != set(range(len(names))):
            raise ValueError("Dataset names must use contiguous class IDs from 0")
    class_count = len(names)

    summary = {"classes": class_count, "images": 0, "objects": 0}
    split_digests: dict[str, set[str]] = {}
    for split_name in ("train", "val"):
        image_dir = resolve_split(data_yaml, config, split_name)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing {split_name} image folder: {image_dir}")
        image_paths = sorted(
            path
            for path in image_dir.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise ValueError(f"No images found for {split_name}")
        split_digests[split_name] = {image_digest(path) for path in image_paths}

        try:
            images_index = image_dir.parts.index("images")
        except ValueError as error:
            raise ValueError(
                f"Split path must contain an images folder: {image_dir}"
            ) from error
        label_dir = Path(
            *image_dir.parts[:images_index],
            "labels",
            *image_dir.parts[images_index + 1 :],
        )
        for image_path in image_paths:
            relative_label = image_path.relative_to(image_dir).with_suffix(".txt")
            label_path = label_dir / relative_label
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            summary["objects"] += validate_label(label_path, class_count)
        summary["images"] += len(image_paths)
        summary[f"{split_name}_images"] = len(image_paths)

    overlap = split_digests["train"] & split_digests["val"]
    if overlap:
        raise ValueError("Train and validation splits contain duplicate images")
    return summary


def validate_expected_counts(
    summary: dict[str, int], expected_train_images: int, expected_val_images: int
) -> None:
    expected_counts = {
        "train_images": expected_train_images,
        "val_images": expected_val_images,
    }
    for field_name, expected_count in expected_counts.items():
        if expected_count > 0 and summary[field_name] != expected_count:
            raise ValueError(
                f"{field_name} is {summary[field_name]}, expected {expected_count}"
            )


def main() -> None:
    args = parse_args()
    summary = validate_dataset(args.data)
    validate_expected_counts(
        summary, args.expected_train_images, args.expected_val_images
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()