from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.validate_dataset import validate_dataset, validate_expected_counts


def create_dataset(root: Path) -> Path:
    for index, (split, image_name) in enumerate(
        (("train", "train.jpg"), ("val", "val.jpg"))
    ):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        color = (255, 0, 0) if index == 0 else (0, 255, 0)
        Image.new("RGB", (8, 8), color=color).save(image_dir / image_name)
        (label_dir / Path(image_name).with_suffix(".txt")).write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "names": {0: "object"},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_validate_dataset_returns_counts(tmp_path: Path) -> None:
    summary = validate_dataset(create_dataset(tmp_path))

    assert summary == {
        "classes": 1,
        "images": 2,
        "objects": 2,
        "train_images": 1,
        "val_images": 1,
    }


def test_validate_dataset_rejects_missing_label(tmp_path: Path) -> None:
    data_yaml = create_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "train.txt").unlink()

    with pytest.raises(FileNotFoundError, match="Missing label"):
        validate_dataset(data_yaml)


@pytest.mark.parametrize("invalid_value", ["nan", "inf", "-inf"])
def test_validate_dataset_rejects_non_finite_coordinates(
    tmp_path: Path, invalid_value: str
) -> None:
    data_yaml = create_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "train.txt").write_text(
        f"0 0.5 0.5 {invalid_value} 0.2\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="non-finite coordinate"):
        validate_dataset(data_yaml)


def test_validate_dataset_rejects_non_contiguous_class_ids(tmp_path: Path) -> None:
    data_yaml = create_dataset(tmp_path)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    config["names"] = {1: "object"}
    data_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="contiguous class IDs"):
        validate_dataset(data_yaml)


def test_validate_dataset_rejects_corrupt_image(tmp_path: Path) -> None:
    data_yaml = create_dataset(tmp_path)
    (tmp_path / "images" / "train" / "train.jpg").write_bytes(b"not-an-image")

    with pytest.raises(ValueError, match="Invalid image file"):
        validate_dataset(data_yaml)


def test_validate_dataset_rejects_path_escape(tmp_path: Path) -> None:
    data_yaml = create_dataset(tmp_path / "dataset")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    config["path"] = "../"
    data_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the dataset root"):
        validate_dataset(data_yaml)


def test_validate_dataset_rejects_duplicate_image_content(tmp_path: Path) -> None:
    data_yaml = create_dataset(tmp_path)
    train_image = tmp_path / "images" / "train" / "train.jpg"
    val_image = tmp_path / "images" / "val" / "val.jpg"
    val_image.write_bytes(train_image.read_bytes())

    with pytest.raises(ValueError, match="duplicate images"):
        validate_dataset(data_yaml)


@pytest.mark.parametrize(
    ("expected_train", "expected_val", "message"),
    [(2, 1, "train_images"), (1, 2, "val_images")],
)
def test_validate_expected_counts_rejects_marker_mismatch(
    expected_train: int, expected_val: int, message: str
) -> None:
    summary = {"train_images": 1, "val_images": 1}

    with pytest.raises(ValueError, match=message):
        validate_expected_counts(summary, expected_train, expected_val)