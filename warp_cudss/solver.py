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

import warp as wp

from ._bindings import (
    CUDSS_BASE_ZERO,
    CUDSS_CONFIG_IR_N_STEPS,
    CUDSS_CONFIG_REORDERING_ALG,
    CUDSS_LAYOUT_COL_MAJOR,
    CUDSS_PHASE_ANALYSIS,
    CUDSS_PHASE_FACTORIZATION,
    CUDSS_PHASE_REFACTORIZATION,
    CUDSS_PHASE_SOLVE,
    CUDA_R_32I,
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
        if mtype not in MTYPE:
            raise ValueError(f"Unknown mtype {mtype!r}, expected one of {sorted(MTYPE)}")
        if view not in MVIEW:
            raise ValueError(f"Unknown view {view!r}, expected one of {sorted(MVIEW)}")

        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("warp_cudss.CudssSolver requires a CUDA device")

        self.mtype = mtype
        self.view = view

        self._lib = get_library()

        handle = cudssHandle_t()
        _check(self._lib.cudssCreate(ctypes.byref(handle)), "cudssCreate")
        self._handle = handle

        config = cudssConfig_t()
        _check(self._lib.cudssConfigCreate(ctypes.byref(config)), "cudssConfigCreate")
        self._config = config

        data = cudssData_t()
        _check(self._lib.cudssDataCreate(handle, ctypes.byref(data)), "cudssDataCreate")
        self._data = data

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

        self._csr = None
        self._A_desc = None
        self._x_desc = None
        self._b_desc = None
        self._x = None
        self._b = None
        self._analyzed = False
        self._closed = False

    # -- lifecycle -----------------------------------------------------

    def setup(self, A, x: wp.array, b: wp.array) -> "CudssSolver":
        """(Re)build the CSR/dense descriptors from ``A``'s sparsity pattern, then run the
        analysis and first factorization phases.

        Call this once, and again whenever ``A``'s sparsity pattern changes (not merely its
        values). Not graph-capturable: it may synchronize with the host and allocate memory.
        """
        if A.device != self.device:
            raise ValueError(f"Matrix device {A.device} does not match solver device {self.device}")

        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"cuDSS solver requires a square matrix, got shape {A.shape}")
        if _scalar_length(x) != n or _scalar_length(b) != n:
            raise ValueError(
                f"x and b must have {n} scalar entries to match A's shape {A.shape}, "
                f"got {_scalar_length(x)} and {_scalar_length(b)}"
            )

        self._destroy_descriptors()

        self._csr = bsr_to_scalar_csr(A)
        value_type = _cudss_value_type(self._csr.scalar_type)

        A_desc = cudssMatrix_t()
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
        self._A_desc = A_desc

        x_desc = cudssMatrix_t()
        _check(
            self._lib.cudssMatrixCreateDn(
                ctypes.byref(x_desc), n, 1, n, ctypes.c_void_p(x.ptr), value_type, CUDSS_LAYOUT_COL_MAJOR
            ),
            "cudssMatrixCreateDn(x)",
        )
        self._x_desc = x_desc

        b_desc = cudssMatrix_t()
        _check(
            self._lib.cudssMatrixCreateDn(
                ctypes.byref(b_desc), n, 1, n, ctypes.c_void_p(b.ptr), value_type, CUDSS_LAYOUT_COL_MAJOR
            ),
            "cudssMatrixCreateDn(b)",
        )
        self._b_desc = b_desc

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

    def refactor(self, A=None) -> "CudssSolver":
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

    def solve(self, x: wp.array = None, b: wp.array = None) -> "CudssSolver":
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
        """Destroy all cuDSS objects owned by this solver. The solver is unusable afterwards."""
        if self._closed:
            return
        self._destroy_descriptors()
        _check(self._lib.cudssDataDestroy(self._handle, self._data), "cudssDataDestroy")
        _check(self._lib.cudssConfigDestroy(self._config), "cudssConfigDestroy")
        _check(self._lib.cudssDestroy(self._handle), "cudssDestroy")
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    # -- internals -------------------------------------------------------

    def _require_setup(self):
        if not self._analyzed:
            raise RuntimeError("CudssSolver.setup(A, x, b) must be called before solve()/refactor()")

    def _destroy_descriptors(self):
        for attr in ("_A_desc", "_x_desc", "_b_desc"):
            desc = getattr(self, attr, None)
            if desc:
                _check(self._lib.cudssMatrixDestroy(desc), f"cudssMatrixDestroy({attr})")
                setattr(self, attr, None)
        self._analyzed = False


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
