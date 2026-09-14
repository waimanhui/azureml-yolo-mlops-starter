import argparse
import json
import os
import random
import re
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

COCO128_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
)
COCO128_YAML_URL = (
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/"
    "ultralytics/cfg/datasets/coco128.yaml"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SAFE_STORAGE_PREFIX = re.compile(r"^[A-Za-z0-9._/-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cumulative YOLO dataset snapshots from COCO128."
    )
    parser.add_argument("--source", type=Path, help="Existing extracted YOLO dataset")
    parser.add_argument(
        "--output", type=Path, default=Path("data/continuous-simulation")
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--batch-count", type=int, default=4)
    parser.add_argument("--validation-size", type=int, default=26)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--storage-prefix",
        default=os.getenv("YOLO_STORAGE_PREFIX", "continuous-training"),
        help="Blob path prefix written into each readiness marker",
    )
    return parser.parse_args()


def extract_zip_safely(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Archive contains unsafe path: {member.filename}")
        archive.extractall(destination)


def normalize_storage_prefix(storage_prefix: str) -> str:
    normalized_prefix = storage_prefix.strip("/")
    if (
        not normalized_prefix
        or not SAFE_STORAGE_PREFIX.fullmatch(normalized_prefix)
        or ".." in normalized_prefix
    ):
        raise ValueError(f"Unsafe storage prefix: {storage_prefix!r}")
    return normalized_prefix


def download_coco128(cache_dir: Path) -> Path:
    archive_path = cache_dir / "coco128.zip"
    dataset_path = cache_dir / "coco128"
    if not dataset_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not archive_path.exists():
            urllib.request.urlretrieve(COCO128_URL, archive_path)
        extract_zip_safely(archive_path, cache_dir)
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        urllib.request.urlretrieve(COCO128_YAML_URL, data_yaml)
    return dataset_path


def discover_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(dataset_root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_path = image_path.relative_to(dataset_root)
        parts = list(relative_path.parts)
        try:
            images_index = parts.index("images")
        except ValueError:
            continue
        parts[images_index] = "labels"
        label_path = dataset_root.joinpath(*parts).with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))
    if not pairs:
        raise ValueError(
            f"No matching YOLO image and label pairs found in {dataset_root}"
        )
    return pairs


def load_names(dataset_root: Path) -> dict | list:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        candidates = sorted(dataset_root.glob("*.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No dataset YAML found in {dataset_root}")
        data_yaml = candidates[0]
    dataset_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = dataset_config.get("names")
    if not names:
        raise ValueError(f"Dataset YAML {data_yaml} does not define class names")
    return names


def copy_pairs(
    pairs: list[tuple[Path, Path]], snapshot_dir: Path, split: str
) -> None:
    image_dir = snapshot_dir / "images" / split
    label_dir = snapshot_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)


def create_snapshots(
    dataset_root: Path,
    output_dir: Path,
    batch_size: int,
    batch_count: int,
    validation_size: int,
    seed: int,
    storage_prefix: str = "continuous-training",
) -> list[Path]:
    if min(batch_size, batch_count, validation_size) < 1:
        raise ValueError(
            "Batch size, batch count, and validation size must be positive"
        )

    storage_prefix = normalize_storage_prefix(storage_prefix)
    pairs = discover_pairs(dataset_root)
    required_pairs = batch_size * batch_count + validation_size
    if len(pairs) < required_pairs:
        raise ValueError(
            f"Need {required_pairs} pairs but only found {len(pairs)} in {dataset_root}"
        )

    shuffled_pairs = pairs.copy()
    random.Random(seed).shuffle(shuffled_pairs)
    validation_pairs = shuffled_pairs[:validation_size]
    training_pairs = shuffled_pairs[validation_size:required_pairs]
    names = load_names(dataset_root)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    snapshots = []
    for version in range(1, batch_count + 1):
        snapshot_dir = output_dir / f"version-{version:03d}"
        cumulative_pairs = training_pairs[: version * batch_size]
        copy_pairs(cumulative_pairs, snapshot_dir, "train")
        copy_pairs(validation_pairs, snapshot_dir, "val")
        (snapshot_dir / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "names": names,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "dataset_version": version,
            "dataset_path": f"{storage_prefix}/version-{version:03d}",
            "training_image_count": len(cumulative_pairs),
            "validation_image_count": len(validation_pairs),
            "seed": seed,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (snapshot_dir / "_READY.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        snapshots.append(snapshot_dir)
    return snapshots


def main() -> None:
    args = parse_args()
    dataset_root = args.source or download_coco128(args.output / ".cache")
    snapshots = create_snapshots(
        dataset_root,
        args.output / "snapshots",
        args.batch_size,
        args.batch_count,
        args.validation_size,
        args.seed,
        args.storage_prefix,
    )
    for snapshot in snapshots:
        manifest = json.loads((snapshot / "_READY.json").read_text(encoding="utf-8"))
        print(
            f"Created {snapshot}: {manifest['training_image_count']} training and "
            f"{manifest['validation_image_count']} validation images"
        )


if __name__ == "__main__":
    main()