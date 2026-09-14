import sys
import types

import pytest

from src.train import validate_device


def test_validate_device_allows_cpu_without_importing_torch(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    validate_device("cpu")

    assert "torch" not in sys.modules


def test_validate_device_rejects_unavailable_cuda(monkeypatch) -> None:
    cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 1)
    torch = types.SimpleNamespace(
        __version__="test-version",
        version=types.SimpleNamespace(cuda="test-runtime"),
        cuda=cuda,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(RuntimeError, match="container CUDA version"):
        validate_device("0")


def test_validate_device_allows_available_cuda(monkeypatch) -> None:
    cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    torch = types.SimpleNamespace(
        __version__="test-version",
        version=types.SimpleNamespace(cuda="test-runtime"),
        cuda=cuda,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    validate_device("0")