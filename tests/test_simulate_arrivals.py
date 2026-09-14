import json
import zipfile
from pathlib import Path

import pytest
import yaml

from scripts.simulate_arrivals import create_snapshots, extract_zip_safely


def create_dataset(root: Path, count: int) -> None:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(count):
        (image_dir / f"image-{index:03d}.jpg").write_bytes(b"image")
        (label_dir / f"image-{index:03d}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text("names:\n  0: object\n", encoding="utf-8")


def test_snapshots_are_cumulative_with_fixed_validation(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    create_dataset(source_dir, count=8)

    snapshots = create_snapshots(
        source_dir,
        tmp_path / "output",
        batch_size=2,
        batch_count=3,
        validation_size=2,
        seed=7,
    )

    training_counts = [
        len(list((snapshot / "images" / "train").glob("*.jpg")))
        for snapshot in snapshots
    ]
    assert training_counts == [2, 4, 6]
    validation_names = [
        {path.name for path in (snapshot / "images" / "val").glob("*.jpg")}
        for snapshot in snapshots
    ]
    assert validation_names[0] == validation_names[1] == validation_names[2]

    manifest = json.loads(
        (snapshots[-1] / "_READY.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_version"] == 3
    assert manifest["training_image_count"] == 6
    assert manifest["validation_image_count"] == 2
    assert manifest["dataset_path"] == "continuous-training/version-003"

    data_yaml = yaml.safe_load(
        (snapshots[-1] / "data.yaml").read_text(encoding="utf-8")
    )
    assert "path" not in data_yaml
    assert data_yaml["train"] == "images/train"
    assert data_yaml["val"] == "images/val"
    assert data_yaml["names"] == {0: "object"}


def test_snapshots_use_configured_storage_prefix(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    create_dataset(source_dir, count=2)

    snapshots = create_snapshots(
        source_dir,
        tmp_path / "output",
        batch_size=1,
        batch_count=1,
        validation_size=1,
        seed=7,
        storage_prefix="customer/project-a",
    )

    manifest = json.loads(
        (snapshots[0] / "_READY.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_path"] == "customer/project-a/version-001"


def test_extract_zip_safely_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="unsafe path"):
        extract_zip_safely(archive_path, tmp_path / "extract")

    assert not (tmp_path / "outside.txt").exists()