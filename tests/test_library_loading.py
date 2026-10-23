"""Library selection is host-only and an explicit backend must not silently change."""

from types import SimpleNamespace

import pytest

from warp_cudss import _bindings as binding


@pytest.fixture
def loader(monkeypatch):
    calls, configured = [], []
    monkeypatch.setattr(binding, "_lib", None)
    monkeypatch.delenv("CUDSS_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(binding, "_configure_argtypes", configured.append)
    return calls, configured


@pytest.mark.parametrize(
    "reason", ["No such file", "libnvrtc.so.12: cannot open shared object"]
)
def test_failed_explicit_library_never_tries_discovery(monkeypatch, loader, reason):
    calls, configured = loader
    cause = OSError(reason)
    monkeypatch.setenv("CUDSS_LIBRARY_PATH", "/candidate/libcudss.so.1")
    monkeypatch.setattr(
        binding,
        "_candidate_paths",
        lambda: iter(["/candidate/libcudss.so.1", "fallback"]),
    )

    def load(path):
        calls.append(path)
        if path == "/candidate/libcudss.so.1":
            raise cause
        return object()

    monkeypatch.setattr(binding.ctypes, "CDLL", load)
    with pytest.raises(OSError, match="explicitly requested cuDSS library") as caught:
        binding.get_library()
    assert reason in str(caught.value) and caught.value.__cause__ is cause
    assert calls == ["/candidate/libcudss.so.1"] and configured == []
    assert binding._lib is None


def test_successful_explicit_library_is_configured_once_and_cached(monkeypatch, loader):
    calls, configured = loader
    chosen = SimpleNamespace(_name="/candidate/libcudss.so.0")
    monkeypatch.setenv("CUDSS_LIBRARY_PATH", chosen._name)
    monkeypatch.setattr(
        binding, "_candidate_paths", lambda: iter([chosen._name, "unused"])
    )

    def load(path):
        calls.append(path)
        return chosen

    monkeypatch.setattr(binding.ctypes, "CDLL", load)
    assert binding.get_library() is chosen
    monkeypatch.setenv("CUDSS_LIBRARY_PATH", "/different/library.so")
    assert binding.get_library() is chosen
    assert calls == [chosen._name] and configured == [chosen]


@pytest.mark.parametrize("override", [None, ""])
def test_discovery_without_explicit_override_keeps_fallback(
    monkeypatch, loader, override
):
    calls, configured = loader
    chosen = object()
    if override is not None:
        monkeypatch.setenv("CUDSS_LIBRARY_PATH", override)
    monkeypatch.setattr(
        binding, "_candidate_paths", lambda: iter(["missing", "present", "unused"])
    )

    def load(path):
        calls.append(path)
        if path == "missing":
            raise OSError("not found")
        return chosen

    monkeypatch.setattr(binding.ctypes, "CDLL", load)
    assert binding.get_library() is chosen
    assert calls == ["missing", "present"] and configured == [chosen]


def test_failed_discovery_keeps_load_error_and_no_cache(monkeypatch, loader):
    calls, configured = loader
    monkeypatch.setattr(binding, "_candidate_paths", lambda: iter(["missing"]))

    def load(path):
        calls.append(path)
        raise OSError("missing dependency")

    monkeypatch.setattr(binding.ctypes, "CDLL", load)
    with pytest.raises(OSError, match="missing dependency"):
        binding.get_library()
    assert calls == ["missing"] and configured == [] and binding._lib is None


def test_failed_configuration_does_not_cache_library(monkeypatch, loader):
    chosen = object()
    monkeypatch.setenv("CUDSS_LIBRARY_PATH", "/candidate/library.so")
    monkeypatch.setattr(binding.ctypes, "CDLL", lambda path: chosen)

    def configure(lib):
        assert lib is chosen
        raise AttributeError("required symbol absent")

    monkeypatch.setattr(binding, "_configure_argtypes", configure)
    with pytest.raises(AttributeError, match="required symbol absent"):
        binding.get_library()
    assert binding._lib is None
