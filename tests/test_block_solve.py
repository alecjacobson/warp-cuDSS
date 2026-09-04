import numpy as np
import pytest
import warp as wp

import warp_cudss

from ._helpers import random_block_spd


@pytest.mark.parametrize("br", [1, 3, 4])
def test_block_spd_solve(device, br):
    rng = np.random.default_rng(2)
    nb = 20
    dtype = {1: wp.float64, 3: wp.mat33d, 4: wp.mat44d}[br]
    A, dense = random_block_spd(rng, nb, br, dtype=dtype, device=device)

    n = nb * br
    b_np = rng.standard_normal(n)
    b = wp.array(b_np, dtype=wp.float64, device=device)
    x = wp.zeros(n, dtype=wp.float64, device=device)

    warp_cudss.solve(A, b, x, mtype="spd")
    wp.synchronize()

    x_ref = np.linalg.solve(dense, b_np)
    np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-7, rtol=1e-7)


def test_block_rhs_as_vector_array(device):
    """x/b may be typed as vector arrays (e.g. wp.vec3d) matching the block layout."""
    rng = np.random.default_rng(3)
    nb, br = 12, 3
    A, dense = random_block_spd(rng, nb, br, dtype=wp.mat33d, device=device)

    n = nb * br
    b_np = rng.standard_normal(n)
    b_vec = wp.array(b_np.reshape(nb, br), dtype=wp.vec3d, device=device)
    x_vec = wp.zeros(nb, dtype=wp.vec3d, device=device)

    warp_cudss.solve(A, b_vec, x_vec, mtype="spd")
    wp.synchronize()

    x_ref = np.linalg.solve(dense, b_np).reshape(nb, br)
    np.testing.assert_allclose(x_vec.numpy(), x_ref, atol=1e-7, rtol=1e-7)
