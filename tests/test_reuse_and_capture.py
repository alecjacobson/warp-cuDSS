import numpy as np
import warp as wp

import warp_cudss

from ._helpers import random_sparse_spd


def test_solver_reuse_after_value_update(device):
    """Same sparsity pattern, values change in place between solves (Newton-like loop)."""
    rng = np.random.default_rng(4)
    n = 150
    A, dense = random_sparse_spd(rng, n, device=device)

    b_np = rng.standard_normal(n)
    b = wp.array(b_np, dtype=wp.float64, device=device)
    x = wp.zeros(n, dtype=wp.float64, device=device)

    solver = warp_cudss.solve(A, b, x, mtype="spd")

    x_ref = np.linalg.solve(dense, b_np)
    np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-8, rtol=1e-8)

    # Scale the diagonal in place (same sparsity pattern) and re-solve reusing the analysis.
    scale = 3.0
    diag_scale_kernel(A, scale, device)
    dense2 = dense.copy()
    np.fill_diagonal(dense2, np.diag(dense2) * scale)

    warp_cudss.solve(A, b, x, mtype="spd", solver=solver)
    wp.synchronize()

    x_ref2 = np.linalg.solve(dense2, b_np)
    np.testing.assert_allclose(x.numpy(), x_ref2, atol=1e-8, rtol=1e-7)

    solver.release()


def diag_scale_kernel(A, scale, device):
    @wp.kernel
    def _scale_diag(
        offsets: wp.array(dtype=wp.int32),
        columns: wp.array(dtype=wp.int32),
        values: wp.array(dtype=wp.float64),
        scale: wp.float64,
    ):
        row = wp.tid()
        for k in range(offsets[row], offsets[row + 1]):
            if columns[k] == row:
                values[k] = values[k] * scale

    wp.launch(_scale_diag, dim=A.nrow, inputs=[A.offsets, A.columns, A.values, wp.float64(scale)], device=device)


def test_graph_capture_refactor_and_solve(device):
    rng = np.random.default_rng(5)
    n = 300
    A, dense = random_sparse_spd(rng, n, device=device)

    b_np = rng.standard_normal(n)
    b = wp.array(b_np, dtype=wp.float64, device=device)
    x = wp.zeros(n, dtype=wp.float64, device=device)

    solver = warp_cudss.CudssSolver(mtype="spd", device=device)
    solver.setup(A, x, b)

    with wp.ScopedCapture(device) as capture:
        solver.refactor(A)
        solver.solve()

    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize()

    x_ref = np.linalg.solve(dense, b_np)
    np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-7, rtol=1e-7)

    solver.release()
