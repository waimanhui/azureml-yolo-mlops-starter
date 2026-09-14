import argparse
import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath

SAFE_DATASET_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")


def validate_marker(marker_file: Path, marker_name: str) -> tuple[int, str]:
    manifest = json.loads(marker_file.read_text(encoding="utf-8"))
    required_fields = {
        "schema_version",
        "dataset_version",
        "dataset_path",
        "training_image_count",
        "validation_image_count",
        "created_at",
    }
    missing_fields = required_fields - manifest.keys()
    if missing_fields:
        raise ValueError(f"Marker is missing fields: {sorted(missing_fields)}")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported marker schema_version")

    dataset_version = manifest["dataset_version"]
    if type(dataset_version) is not int or dataset_version < 1:
        raise ValueError("dataset_version must be a positive integer")

    for field_name in ("training_image_count", "validation_image_count"):
        value = manifest[field_name]
        if type(value) is not int or value < 1:
            raise ValueError(f"{field_name} must be a positive integer")

    dataset_path_value = manifest["dataset_path"]
    if (
        not isinstance(dataset_path_value, str)
        or not SAFE_DATASET_PATH.fullmatch(dataset_path_value)
    ):
        raise ValueError("dataset_path contains unsupported characters")
    dataset_path = PurePosixPath(dataset_path_value)
    if dataset_path.is_absolute() or ".." in dataset_path.parts:
        raise ValueError("dataset_path must be a safe relative path")
    if dataset_path.name != f"version-{dataset_version:03d}":
        raise ValueError("dataset_path version must match dataset_version")
    if str(dataset_path) != str(PurePosixPath(marker_name).parent):
        raise ValueError("dataset_path must match the marker parent folder")

    created_at_value = manifest["created_at"]
    if not isinstance(created_at_value, str):
        raise ValueError("created_at must be an ISO 8601 timestamp")
    try:
        created_at = datetime.fromisoformat(created_at_value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be an ISO 8601 timestamp") from error
    if created_at.tzinfo is None:
        raise ValueError("created_at must include a timezone")

    return dataset_version, str(dataset_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset readiness marker")
    parser.add_argument("marker_file", type=Path)
    parser.add_argument("marker_name")
    args = parser.parse_args()
    dataset_version, dataset_path = validate_marker(
        args.marker_file, args.marker_name
    )
    print(dataset_version)
    print(dataset_path)


if __name__ == "__main__":
    main()
