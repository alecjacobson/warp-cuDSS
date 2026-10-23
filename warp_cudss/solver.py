"""cuDSS-backed sparse direct solver, usable as a drop-in for the iterative solvers in
``warp.optim.linear`` (:func:`warp.optim.linear.cg`, :func:`cr`, :func:`bicgstab`, :func:`gmres`).

Typical single-shot usage, mirroring ``warp.optim.linear.cg(A, b, x)``::

    import warp_cudss

    warp_cudss.solve(A, b, x, mtype="spd")

Typical reused / graph-capturable usage, when ``A``'s sparsity pattern is fixed across
timesteps and only its values change (e.g. a Newton solve or a simulation loop)::

    solver = warp_cudss.CudssSolver(mtype="spd", device=A.device)
    solver.setup(A, x, b)          # analysis + first factorization; NOT capturable

    with wp.ScopedCapture(device) as capture:
        assemble_matrix_values(A)  # your kernels, write into A.values in place
        solver.refactor(A)         # refactorization + solve; capturable
        solver.solve()
    for _ in range(num_steps):
        wp.capture_launch(capture.graph)

All arrays referenced by the solver (``A``'s CSR storage, ``x``, ``b``) must keep a
stable device address for the lifetime of the solver: do not reallocate them between
:meth:`CudssSolver.setup` and subsequent :meth:`CudssSolver.refactor`/:meth:`solve` calls.
"""

from __future__ import annotations

import ctypes
from contextlib import suppress

import warp as wp

from ._bindings import (
    CUDA_R_32I,
    CUDSS_BASE_ZERO,
    CUDSS_CONFIG_IR_N_STEPS,
    CUDSS_CONFIG_REORDERING_ALG,
    CUDSS_LAYOUT_COL_MAJOR,
    CUDSS_PHASE_ANALYSIS,
    CUDSS_PHASE_FACTORIZATION,
    CUDSS_PHASE_REFACTORIZATION,
    CUDSS_PHASE_SOLVE,
    MTYPE,
    MVIEW,
    REORDERING_ALG,
    _check,
    cudssConfig_t,
    cudssData_t,
    cudssHandle_t,
    cudssMatrix_t,
    get_library,
)
from ._csr import bsr_to_scalar_csr

__all__ = ["CudssSolver", "solve"]

_SCALAR_TYPE_TO_CUDSS = {
    wp.float32: 0,  # CUDA_R_32F
    wp.float64: 1,  # CUDA_R_64F
}


def _cudss_value_type(scalar_type):
    try:
        return _SCALAR_TYPE_TO_CUDSS[scalar_type]
    except KeyError as e:
        raise TypeError(f"cuDSS solver supports float32/float64 values, got {scalar_type}") from e


def _scalar_length(arr: wp.array) -> int:
    return arr.size * wp.types.type_size(arr.dtype)


class CudssSolver:
    """Owns a cuDSS handle/config/data context and the matrix/vector descriptors for one
    sparse linear system. Not thread-safe; one instance per system being solved.
    """

    def __init__(
        self,
        mtype: str = "general",
        view: str = "full",
        reordering: str = "default",
        ir_steps: int = 0,
        device=None,
    ):
        self._handle = self._config = self._data = None
        self._csr = self._A_desc = self._x_desc = self._b_desc = None
        self._x = self._b = None
        self._analyzed = False
        self._closed = False

        if mtype not in MTYPE:
            raise ValueError(f"Unknown mtype {mtype!r}, expected one of {sorted(MTYPE)}")
        if view not in MVIEW:
            raise ValueError(f"Unknown view {view!r}, expected one of {sorted(MVIEW)}")

        if reordering not in REORDERING_ALG:
            raise ValueError(f"Unknown reordering {reordering!r}, expected one of {sorted(REORDERING_ALG)}")

        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("warp_cudss.CudssSolver requires a CUDA device")

        self.mtype = mtype
        self.view = view

        try:
            self._lib = get_library()

            self._handle = handle = cudssHandle_t()
            _check(self._lib.cudssCreate(ctypes.byref(handle)), "cudssCreate")

            self._config = config = cudssConfig_t()
            _check(self._lib.cudssConfigCreate(ctypes.byref(config)), "cudssConfigCreate")

            self._data = data = cudssData_t()
            _check(self._lib.cudssDataCreate(handle, ctypes.byref(data)), "cudssDataCreate")

            stream = wp.get_stream(self.device)
            _check(self._lib.cudssSetStream(handle, ctypes.c_void_p(stream.cuda_stream)), "cudssSetStream")

            if reordering != "default":
                alg = ctypes.c_int(REORDERING_ALG[reordering])
                _check(
                    self._lib.cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, ctypes.byref(alg), ctypes.sizeof(alg)),
                    "cudssConfigSet(REORDERING_ALG)",
                )
            if ir_steps:
                n_steps = ctypes.c_int(ir_steps)
                _check(
                    self._lib.cudssConfigSet(
                        config, CUDSS_CONFIG_IR_N_STEPS, ctypes.byref(n_steps), ctypes.sizeof(n_steps)
                    ),
                    "cudssConfigSet(IR_N_STEPS)",
                )
        except BaseException:
            # Create calls may return a non-null object even when reporting an
            # error. Preserve the original failure after attempting all cleanup.
            with suppress(Exception):
                self.release()
            raise

    # -- lifecycle -----------------------------------------------------

    def setup(self, A, x: wp.array, b: wp.array, nrhs: int = 1) -> CudssSolver:
        """(Re)build the CSR/dense descriptors from ``A``'s sparsity pattern, then run the
        analysis and first factorization phases.

        Call this once, and again whenever ``A``'s sparsity pattern changes (not merely its
        values). Not graph-capturable: it may synchronize with the host and allocate memory.

        ``nrhs`` sets the number of right-hand sides solved together. With ``nrhs > 1``,
        ``x`` and ``b`` must each hold ``n * nrhs`` scalars laid out column-major (right-hand
        side ``j`` occupying ``[j*n, (j+1)*n)``); cuDSS then back-substitutes all columns in a
        single solve, which is far cheaper than ``nrhs`` separate solves.
        """
        if A.device != self.device:
            raise ValueError(f"Matrix device {A.device} does not match solver device {self.device}")

        n = A.shape[0]
        self._nrhs = nrhs
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"cuDSS solver requires a square matrix, got shape {A.shape}")
        if _scalar_length(x) != n * nrhs or _scalar_length(b) != n * nrhs:
            raise ValueError(
                f"x and b must have {n * nrhs} scalar entries to match A's shape {A.shape} with "
                f"nrhs={nrhs}, got {_scalar_length(x)} and {_scalar_length(b)}"
            )

        self._destroy_descriptors()

        self._csr = bsr_to_scalar_csr(A)
        value_type = _cudss_value_type(self._csr.scalar_type)

        self._A_desc = A_desc = cudssMatrix_t()
        _check(
            self._lib.cudssMatrixCreateCsr(
                ctypes.byref(A_desc),
                n,
                n,
                self._csr.nnz,
                ctypes.c_void_p(self._csr.row_offsets.ptr),
                None,
                ctypes.c_void_p(self._csr.columns.ptr),
                ctypes.c_void_p(self._csr.values.ptr),
                CUDA_R_32I,
                CUDA_R_32I,
                value_type,
                MTYPE[self.mtype],
                MVIEW[self.view],
                CUDSS_BASE_ZERO,
            ),
            "cudssMatrixCreateCsr",
        )

        self._x_desc = x_desc = cudssMatrix_t()
        _check(
            self._lib.cudssMatrixCreateDn(
                ctypes.byref(x_desc), n, nrhs, n, ctypes.c_void_p(x.ptr), value_type, CUDSS_LAYOUT_COL_MAJOR
            ),
            "cudssMatrixCreateDn(x)",
        )

        self._b_desc = b_desc = cudssMatrix_t()
        _check(
            self._lib.cudssMatrixCreateDn(
                ctypes.byref(b_desc), n, nrhs, n, ctypes.c_void_p(b.ptr), value_type, CUDSS_LAYOUT_COL_MAJOR
            ),
            "cudssMatrixCreateDn(b)",
        )

        self._x = x
        self._b = b

        _check(
            self._lib.cudssExecute(
                self._handle, CUDSS_PHASE_ANALYSIS, self._config, self._data, self._A_desc, self._x_desc, self._b_desc
            ),
            "cudssExecute(ANALYSIS)",
        )
        _check(
            self._lib.cudssExecute(
                self._handle,
                CUDSS_PHASE_FACTORIZATION,
                self._config,
                self._data,
                self._A_desc,
                self._x_desc,
                self._b_desc,
            ),
            "cudssExecute(FACTORIZATION)",
        )
        self._analyzed = True
        return self

    def refactor(self, A=None) -> CudssSolver:
        """Refresh matrix values (if ``A`` given) and re-run factorization.

        Sparsity pattern must be unchanged since :meth:`setup`. Graph-capturable once
        :meth:`setup` has completed at least once outside of any capture.
        """
        self._require_setup()
        if A is not None:
            self._csr.refresh_values(A)
            _check(
                self._lib.cudssMatrixSetValues(self._A_desc, ctypes.c_void_p(self._csr.values.ptr)),
                "cudssMatrixSetValues(A)",
            )

        _check(
            self._lib.cudssExecute(
                self._handle,
                CUDSS_PHASE_REFACTORIZATION,
                self._config,
                self._data,
                self._A_desc,
                self._x_desc,
                self._b_desc,
            ),
            "cudssExecute(REFACTORIZATION)",
        )
        return self

    def solve(self, x: wp.array = None, b: wp.array = None) -> CudssSolver:
        """Run the solve phase, writing the result into ``x``.

        If ``x``/``b`` are omitted, reuses the arrays bound in :meth:`setup`
        (or the last call to this method with explicit arrays). Passing new arrays
        rebinds the dense descriptor pointers via ``cudssMatrixSetValues`` -- this is a
        lightweight host-side call safe to make before a capture, but the arrays'
        addresses must then remain stable across any subsequent replays.
        """
        self._require_setup()
        if x is not None and x is not self._x:
            _check(self._lib.cudssMatrixSetValues(self._x_desc, ctypes.c_void_p(x.ptr)), "cudssMatrixSetValues(x)")
            self._x = x
        if b is not None and b is not self._b:
            _check(self._lib.cudssMatrixSetValues(self._b_desc, ctypes.c_void_p(b.ptr)), "cudssMatrixSetValues(b)")
            self._b = b

        _check(
            self._lib.cudssExecute(
                self._handle, CUDSS_PHASE_SOLVE, self._config, self._data, self._A_desc, self._x_desc, self._b_desc
            ),
            "cudssExecute(SOLVE)",
        )
        return self

    def release(self):
        """Release every owned native object once; safe after partial construction."""
        if self._closed:
            return
        first_error = None
        for attribute, function in (
            ("_A_desc", "cudssMatrixDestroy"),
            ("_x_desc", "cudssMatrixDestroy"),
            ("_b_desc", "cudssMatrixDestroy"),
            ("_data", "cudssDataDestroy"),
            ("_config", "cudssConfigDestroy"),
            ("_handle", "cudssDestroy"),
        ):
            value = getattr(self, attribute)
            if not value:
                continue
            arguments = (self._handle, value) if attribute == "_data" else (value,)
            try:
                _check(getattr(self._lib, function)(*arguments), function)
            except Exception as error:  # noqa: BLE001 - finish cleanup, then re-raise
                if first_error is None:
                    first_error = error
            finally:
                # Destroy calls attempt cleanup even when reporting a pending
                # CUDA error. Do not destroy the same native pointer twice.
                setattr(self, attribute, None)
        self._analyzed = False
        self._closed = True
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        with suppress(Exception):
            self.release()

    # -- internals -------------------------------------------------------

    def _require_setup(self):
        if not self._analyzed:
            raise RuntimeError("CudssSolver.setup(A, x, b) must be called before solve()/refactor()")

    def _destroy_descriptors(self):
        self._analyzed = False
        first_error = None
        for attr in ("_A_desc", "_x_desc", "_b_desc"):
            desc = getattr(self, attr, None)
            if desc:
                try:
                    _check(self._lib.cudssMatrixDestroy(desc), f"cudssMatrixDestroy({attr})")
                except Exception as error:  # noqa: BLE001 - finish cleanup, then re-raise
                    if first_error is None:
                        first_error = error
                finally:
                    setattr(self, attr, None)
        if first_error is not None:
            raise first_error


def solve(
    A,
    b: wp.array,
    x: wp.array,
    mtype: str = "general",
    view: str = "full",
    reordering: str = "default",
    solver: CudssSolver = None,
) -> CudssSolver:
    """Solve ``A @ x = b`` with cuDSS, writing the result into ``x``.

    Drop-in replacement for :func:`warp.optim.linear.cg`/``cr``/``bicgstab``/``gmres`` for
    callers that want a direct solve instead of an iterative one::

        warp_cudss.solve(A, b, x, mtype="spd")

    Pass the returned :class:`CudssSolver` back in via ``solver=`` to reuse the symbolic
    factorization on a later call after updating ``A``'s values in place (same sparsity
    pattern only)::

        solver = warp_cudss.solve(A, b, x, mtype="spd")
        ...  # mutate A's values
        warp_cudss.solve(A, b, x, mtype="spd", solver=solver)
    """
    if solver is None:
        solver = CudssSolver(mtype=mtype, view=view, reordering=reordering, device=A.device)
        solver.setup(A, x, b)
    else:
        solver.refactor(A)
    solver.solve(x, b)
    return solver
