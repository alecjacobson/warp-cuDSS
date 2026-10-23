"""Discover supported cuDSS wheels without assuming an importable package layout."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from warp_cudss import _bindings as binding


@pytest.fixture
def discovery(monkeypatch):
    parent, child = ModuleType("nvidia"), ModuleType("nvidia.cudss")
    parent.__path__ = []
    child.__file__ = None
    parent.cudss = child
    monkeypatch.setitem(sys.modules, "nvidia", parent)
    monkeypatch.setitem(sys.modules, "nvidia.cudss", child)
    monkeypatch.setattr(binding.ctypes.util, "find_library", lambda name: None)
    monkeypatch.delenv("CUDSS_LIBRARY_PATH", raising=False)
    return child


@pytest.mark.parametrize("layout", ["nvidia/cu12/lib", "nvidia/cudss/lib"])
def test_wheel_metadata_finds_supported_library_without_package_file(
    monkeypatch, discovery, layout
):
    library = Path(layout) / "libcudss.so.0"
    package = SimpleNamespace(
        files=[library, Path(layout) / "libcudss.so.1", Path("nvidia/header.h")],
        locate_file=lambda path: Path("/site-packages") / path,
    )
    names = []

    def installed(name):
        names.append(name)
        return package

    monkeypatch.setattr(binding, "distribution", installed)
    paths = list(binding._candidate_paths())
    assert str(package.locate_file(library)) in paths
    assert not any(path.endswith((".so.1", ".h")) for path in paths)
    assert names == ["nvidia-cudss-cu12"]


@pytest.mark.parametrize("files", [None, []])
def test_missing_wheel_files_preserve_system_and_legacy_candidates(
    monkeypatch, discovery, files
):
    discovery.__file__ = "/legacy/nvidia/cudss/__init__.py"
    monkeypatch.setattr(
        binding.ctypes.util, "find_library", lambda name: "/system/libcudss.so.0"
    )
    monkeypatch.setattr(
        binding, "distribution", lambda name: SimpleNamespace(files=files)
    )
    paths = list(binding._candidate_paths())
    assert "/system/libcudss.so.0" in paths
    assert "/legacy/nvidia/cudss/lib/libcudss.so.0" in paths
    assert "/legacy/nvidia/cudss/lib/libcudss.so" in paths


def test_absent_distribution_preserves_normal_loader_candidates(monkeypatch, discovery):
    def absent(name):
        raise binding.PackageNotFoundError(name)

    monkeypatch.setattr(binding, "distribution", absent)
    assert list(binding._candidate_paths()) == ["libcudss.so.0", "libcudss.so"]


def test_loader_reaches_wheel_metadata_after_system_paths_fail(monkeypatch, discovery):
    path = Path("nvidia/cu12/lib/libcudss.so.0")
    target = str(Path("/site-packages") / path)
    package = SimpleNamespace(
        files=[path], locate_file=lambda file: Path("/site-packages") / file
    )
    monkeypatch.setattr(binding, "distribution", lambda name: package)
    monkeypatch.setattr(binding, "_lib", None)
    configured, attempts = [], []
    library = object()

    def load(candidate):
        attempts.append(candidate)
        if candidate == target:
            return library
        raise OSError("not on loader path")

    monkeypatch.setattr(binding.ctypes, "CDLL", load)
    monkeypatch.setattr(binding, "_configure_argtypes", configured.append)
    assert binding.get_library() is library
    assert attempts[-1] == target and configured == [library]
