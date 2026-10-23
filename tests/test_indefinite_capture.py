"""Capture-compatible explicit pivots preserve indefinite solves after updates."""

import numpy as np
import pytest
import warp as wp
from warp import sparse

from warp_cudss import CudssSolver


@pytest.mark.parametrize("reordering", ["default", "amd", "nested_dissection"])
def test_diagonal_pivot_changed_matrix_and_rhs_capture(device, reordering):
    n = 96
    diagonal = np.where(np.arange(n) % 2, -4.0, 4.0)
    dense = np.diag(diagonal) + np.diag(np.full(n - 1, 0.35), 1)
    dense += np.diag(np.full(n - 1, 0.35), -1)
    # This is indefinite and well conditioned; no SPD declaration or diagonal
    # perturbation is needed to pass. Compare every replay to its own system.
    spectrum = np.linalg.eigvalsh(dense)
    assert spectrum[0] < 0 < spectrum[-1]
    rows, columns = np.nonzero(dense)
    matrix = sparse.bsr_from_triplets(
        n, n,
        wp.array(rows.astype(np.int32), device=device),
        wp.array(columns.astype(np.int32), device=device),
        wp.array(dense[rows, columns], dtype=wp.float64, device=device),
    )
    offsets = matrix.offsets.numpy()
    actual_rows = np.repeat(np.arange(n), np.diff(offsets))
    actual_columns = matrix.columns.numpy()[:offsets[-1]]
    rhs = wp.zeros(n, dtype=wp.float64, device=device)
    solution = wp.zeros_like(rhs)
    rng = np.random.default_rng(83)
    with CudssSolver(
        mtype="symmetric", reordering=reordering, pivoting="diagonal", device=device
    ) as solver:
        solver.setup(matrix, solution, rhs)
        solver.solve()
        wp.synchronize_device(device)
        with wp.ScopedCapture(device) as captured:
            solver.refactor(matrix)
            solver.solve()
        for step in range(3):
            current = dense.copy()
            np.fill_diagonal(current, diagonal * (1.0 + 0.25 * step))
            expected_rhs = rng.standard_normal(n)
            values = wp.array(current[actual_rows, actual_columns], dtype=wp.float64, device=device)
            wp.copy(matrix.values, values, count=len(actual_rows))
            rhs.assign(expected_rhs)
            solution.fill_(float("nan"))
            wp.capture_launch(captured.graph)
            wp.synchronize_device(device)
            result = solution.numpy()
            np.testing.assert_allclose(result, np.linalg.solve(current, expected_rhs), rtol=1e-11, atol=1e-12)
            assert np.linalg.norm(current @ result - expected_rhs) / np.linalg.norm(expected_rhs) < 1e-12
            np.testing.assert_array_equal(rhs.numpy(), expected_rhs)
        # The graph must be destroyed while its cuDSS owner is still alive.
        del captured
