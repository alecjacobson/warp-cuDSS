"""Explicit pivot requests reach cuDSS before analysis; AUTO stays unchanged."""

import ctypes as ct

import pytest

import warp_cudss.solver as implementation
from warp_cudss._bindings import CUDSS_CONFIG_PIVOT_TYPE, CudssError

from .test_construction_failure import NativeResources, fake_matrix, install


class RecordedConfiguration(NativeResources):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configurations = []

    def cudssConfigSet(self, config, parameter, pointer, size):
        assert size == ct.sizeof(ct.c_int)
        self.configurations.append(
            (parameter, ct.cast(pointer, ct.POINTER(ct.c_int))[0])
        )
        return self.__getattr__("cudssConfigSet")(config, parameter, pointer, size)


@pytest.mark.parametrize(
    "policy,expected",
    [("none", 1), ("global_col", 2), ("global_row", 3), ("diagonal", 4),
     ("local_block", 5), ("bunch_kaufman", 6)],
)
def test_explicit_policy_precedes_analysis(monkeypatch, policy, expected):
    resources = RecordedConfiguration()
    install(monkeypatch, resources)
    with implementation.CudssSolver(pivoting=policy) as solver:
        matrix, array = fake_matrix(monkeypatch, solver)
        solver.setup(matrix, array, array)
        assert resources.configurations == [(CUDSS_CONFIG_PIVOT_TYPE, expected)]
        assert resources.calls.index("cudssConfigSet") < resources.calls.index("cudssExecute")
        assert solver.pivoting == policy
    assert not resources.live


@pytest.mark.parametrize("options", [{}, {"pivoting": "auto"}])
def test_auto_preserves_vendor_default(monkeypatch, options):
    resources = RecordedConfiguration()
    install(monkeypatch, resources)
    with implementation.CudssSolver(**options) as solver:
        assert solver.pivoting == "auto"
        assert resources.configurations == []
    assert not resources.live


def test_invalid_policy_fails_before_device_or_native_allocation(monkeypatch):
    resources = RecordedConfiguration()
    install(monkeypatch, resources)

    def unexpected_device(*args):
        raise AssertionError("Invalid configuration should be rejected first")

    monkeypatch.setattr(implementation.wp, "get_device", unexpected_device)
    with pytest.raises(ValueError, match="Unknown pivoting"):
        implementation.CudssSolver(pivoting="unsupported")
    assert resources.calls == []


def test_rejected_native_policy_unwinds_resources(monkeypatch):
    resources = RecordedConfiguration(fail="cudssConfigSet")
    install(monkeypatch, resources)
    with pytest.raises(CudssError, match=r"PIVOT_TYPE"):
        implementation.CudssSolver(pivoting="diagonal")
    assert not resources.live
    assert resources.calls[-3:] == ["cudssDataDestroy", "cudssConfigDestroy", "cudssDestroy"]
