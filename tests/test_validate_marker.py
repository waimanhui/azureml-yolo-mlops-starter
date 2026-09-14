import json
from pathlib import Path

import pytest

from src.validate_marker import validate_marker


def write_marker(tmp_path: Path, **overrides: object) -> Path:
    manifest = {
        "schema_version": 1,
        "dataset_version": 3,
        "dataset_path": "customer/project/version-003",
        "training_image_count": 75,
        "validation_image_count": 26,
        "created_at": "2026-09-14T00:00:00+00:00",
    }
    manifest.update(overrides)
    marker_file = tmp_path / "_READY.json"
    marker_file.write_text(json.dumps(manifest), encoding="utf-8")
    return marker_file


def test_validate_marker_returns_version_and_path(tmp_path: Path) -> None:
    marker_file = write_marker(tmp_path)

    result = validate_marker(
        marker_file, "customer/project/version-003/_READY.json"
    )

    assert result == (3, "customer/project/version-003", 75, 26)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training_image_count": "75"}, "positive integer"),
        ({"dataset_path": "customer/project/version-002"}, "must match"),
        ({"created_at": "2026-09-14T00:00:00"}, "timezone"),
    ],
)
def test_validate_marker_rejects_invalid_contract(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    marker_file = write_marker(tmp_path, **overrides)

    with pytest.raises(ValueError, match=message):
        validate_marker(marker_file, "customer/project/version-003/_READY.json")
