import numpy as np
import warp as wp
import warp.optim.linear as linear

import warp_cudss

from ._helpers import random_sparse_spd


def test_matches_warp_cg_result(device):
    """warp_cudss.solve should agree with warp.optim.linear.cg on the same SPD system,
    demonstrating it can be swapped in directly."""
    rng = np.random.default_rng(6)
    n = 250
    A, dense = random_sparse_spd(rng, n, device=device)

    b_np = rng.standard_normal(n)
    b = wp.array(b_np, dtype=wp.float64, device=device)

    x_cg = wp.zeros(n, dtype=wp.float64, device=device)
    linear.cg(A, b, x_cg, tol=1e-12, maxiter=10000)

    x_direct = wp.zeros(n, dtype=wp.float64, device=device)
    warp_cudss.solve(A, b, x_direct, mtype="spd")

    wp.synchronize()
    np.testing.assert_allclose(x_direct.numpy(), x_cg.numpy(), atol=1e-6, rtol=1e-6)
