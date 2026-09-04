import numpy as np
import pytest
import warp as wp

import warp_cudss

from ._helpers import random_sparse_general, random_sparse_spd


@pytest.mark.parametrize("mtype", ["spd", "symmetric", "general"])
def test_scalar_spd_solve(device, mtype):
    rng = np.random.default_rng(0)
    n = 200
    A, dense = random_sparse_spd(rng, n, device=device)

    b_np = rng.standard_normal(n)
    b = wp.array(b_np, dtype=wp.float64, device=device)
    x = wp.zeros(n, dtype=wp.float64, device=device)

    warp_cudss.solve(A, b, x, mtype=mtype)
    wp.synchronize()

    x_ref = np.linalg.solve(dense, b_np)
    np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-8, rtol=1e-8)


def test_scalar_general_solve(device):
    rng = np.random.default_rng(0)
    n = 200
    A, dense = random_sparse_general(rng, n, device=device)

    b_np = rng.standard_normal(n)
    b = wp.array(b_np, dtype=wp.float64, device=device)
    x = wp.zeros(n, dtype=wp.float64, device=device)

    warp_cudss.solve(A, b, x, mtype="general")
    wp.synchronize()

    x_ref = np.linalg.solve(dense, b_np)
    np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-8, rtol=1e-8)


def test_scalar_float32(device):
    rng = np.random.default_rng(0)
    n = 100
    A, dense = random_sparse_spd(rng, n, dtype=wp.float32, device=device)

    b_np = rng.standard_normal(n).astype(np.float32)
    b = wp.array(b_np, dtype=wp.float32, device=device)
    x = wp.zeros(n, dtype=wp.float32, device=device)

    warp_cudss.solve(A, b, x, mtype="spd")
    wp.synchronize()

    x_ref = np.linalg.solve(dense.astype(np.float32), b_np)
    np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-3, rtol=1e-3)
