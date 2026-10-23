"""Native ownership must unwind even when construction returns only partial objects."""

import ctypes as ct
import gc
from types import SimpleNamespace

import pytest

import warp_cudss.solver as implementation
from warp_cudss._bindings import CudssError


class NativeResources:
    def __init__(self, fail=None, write_failed_output=False, fail_occurrence=1):
        self.fail = fail
        self.write_failed_output = write_failed_output
        self.live = {}
        self.calls = []
        self.next_id = 1
        self.fail_occurrence = fail_occurrence

    def __getattr__(self, name):
        def call(*args):
            self.calls.append(name)
            failed = (
                name == self.fail and self.calls.count(name) == self.fail_occurrence
            )
            if name in (
                "cudssCreate",
                "cudssConfigCreate",
                "cudssDataCreate",
                "cudssMatrixCreateCsr",
                "cudssMatrixCreateDn",
            ) and (not failed or self.write_failed_output):
                pointer = args[1] if name == "cudssDataCreate" else args[0]
                ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = self.next_id
                self.live[self.next_id] = name
                self.next_id += 1
            if name.endswith("Destroy"):
                pointer = args[-1].value
                assert pointer in self.live, "duplicate or unowned native destruction"
                del self.live[pointer]
            return 4 if failed else 0

        return call


def install(monkeypatch, resources):
    monkeypatch.setattr(implementation, "get_library", lambda: resources)
    monkeypatch.setattr(
        implementation.wp, "get_device", lambda *_: SimpleNamespace(is_cuda=True)
    )
    monkeypatch.setattr(
        implementation.wp, "get_stream", lambda *_: SimpleNamespace(cuda_stream=1)
    )


@pytest.mark.parametrize(
    "failed",
    [
        "cudssCreate",
        "cudssConfigCreate",
        "cudssDataCreate",
        "cudssSetStream",
        "cudssConfigSet",
    ],
)
@pytest.mark.parametrize("write_failed_output", [False, True])
def test_constructor_failure_unwinds_returned_resources(
    monkeypatch, failed, write_failed_output
):
    resources = NativeResources(failed, write_failed_output)
    install(monkeypatch, resources)
    with pytest.raises(CudssError):
        implementation.CudssSolver(reordering="amd")
    gc.collect()
    assert resources.live == {}


@pytest.mark.parametrize(
    "kwargs", [{"mtype": "invalid"}, {"view": "invalid"}, {"reordering": "invalid"}]
)
def test_invalid_options_are_rejected_before_any_native_allocation(monkeypatch, kwargs):
    resources = NativeResources()
    install(monkeypatch, resources)
    with pytest.raises(ValueError):
        implementation.CudssSolver(**kwargs)
    assert resources.calls == []


def test_successful_release_is_idempotent(monkeypatch):
    resources = NativeResources()
    install(monkeypatch, resources)
    solver = implementation.CudssSolver(reordering="amd", ir_steps=1)
    assert len(resources.live) == 3
    solver.release()
    calls = resources.calls.copy()
    solver.release()
    assert not resources.live and resources.calls == calls


@pytest.mark.parametrize(
    "failed", ["cudssDataDestroy", "cudssConfigDestroy", "cudssDestroy"]
)
def test_release_attempts_all_resources_after_native_cleanup_error(monkeypatch, failed):
    resources = NativeResources(failed)
    install(monkeypatch, resources)
    solver = implementation.CudssSolver()
    with pytest.raises(CudssError):
        solver.release()
    assert not resources.live
    calls = resources.calls.copy()
    solver.release()
    assert resources.calls == calls


def fake_matrix(monkeypatch, solver):
    array = SimpleNamespace(size=4, dtype=implementation.wp.float64, ptr=8)
    csr = SimpleNamespace(
        nnz=4,
        scalar_type=implementation.wp.float64,
        row_offsets=array,
        columns=array,
        values=array,
    )
    monkeypatch.setattr(implementation, "bsr_to_scalar_csr", lambda _: csr)
    return SimpleNamespace(shape=(4, 4), device=solver.device), array


@pytest.mark.parametrize(
    "failed,occurrence",
    [
        ("cudssMatrixCreateCsr", 1),
        ("cudssMatrixCreateDn", 1),
        ("cudssMatrixCreateDn", 2),
        ("cudssExecute", 1),
        ("cudssExecute", 2),
    ],
)
def test_failed_setup_retains_partial_descriptor_ownership(
    monkeypatch, failed, occurrence
):
    resources = NativeResources(
        failed, write_failed_output=True, fail_occurrence=occurrence
    )
    install(monkeypatch, resources)
    solver = implementation.CudssSolver()
    matrix, array = fake_matrix(monkeypatch, solver)
    with pytest.raises(CudssError), solver:
        solver.setup(matrix, array, array)
    assert not resources.live


def test_descriptor_cleanup_error_invalidates_setup_and_does_not_double_destroy(
    monkeypatch,
):
    resources = NativeResources("cudssMatrixDestroy")
    install(monkeypatch, resources)
    solver = implementation.CudssSolver()
    matrix, array = fake_matrix(monkeypatch, solver)
    solver.setup(matrix, array, array)
    assert solver._analyzed
    with pytest.raises(CudssError):
        solver.setup(matrix, array, array)
    assert not solver._analyzed
    solver.release()
    assert not resources.live
