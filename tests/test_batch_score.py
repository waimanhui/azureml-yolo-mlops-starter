import importlib
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=object))

find_model_path = importlib.import_module("src.batch_score").find_model_path


def test_find_model_path_prefers_stable_artifact(tmp_path: Path) -> None:
    stable_model = tmp_path / "registered" / "model.pt"
    stable_model.parent.mkdir()
    stable_model.touch()
    fallback_model = tmp_path / "training" / "best.pt"
    fallback_model.parent.mkdir()
    fallback_model.touch()

    assert find_model_path(tmp_path) == stable_model


def test_find_model_path_rejects_ambiguous_artifacts(tmp_path: Path) -> None:
    for folder in ("one", "two"):
        model_path = tmp_path / folder / "model.pt"
        model_path.parent.mkdir()
        model_path.touch()

    with pytest.raises(RuntimeError, match="Expected one model.pt"):
        find_model_path(tmp_path)


def test_find_model_path_requires_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No model.pt or best.pt"):
        find_model_path(tmp_path)