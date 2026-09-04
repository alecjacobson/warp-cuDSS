"""Drop-in replacement of warp.optim.linear.cg with warp_cudss for a 2D Poisson problem.

Builds the standard 5-point-stencil discrete Laplacian on an ``N x N`` grid with
Dirichlet boundary conditions, as a scalar ``warp.sparse.BsrMatrix``, and solves it two
ways: with Warp's built-in conjugate-gradient solver, and with cuDSS via warp_cudss.
Both should produce the same answer; cuDSS does it directly instead of iteratively.

Run with:
    python examples/example_poisson_2d.py
"""

import time

import numpy as np
import warp as wp
import warp.optim.linear as linear
import warp.sparse as sparse

import warp_cudss


@wp.kernel
def build_laplacian_triplets(
    n: int,
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=wp.float64),
    b: wp.array(dtype=wp.float64),
):
    i, j = wp.tid()
    row = i * n + j
    base = row * 5

    rows[base + 0] = row
    cols[base + 0] = row
    vals[base + 0] = wp.float64(4.0)

    # Neighbors; out-of-range entries collapse onto the diagonal (a harmless
    # duplicate that bsr_from_triplets sums), keeping every row exactly 5 wide.
    neighbor_i = wp.vec4i(i - 1, i + 1, i, i)
    neighbor_j = wp.vec4i(j, j, j - 1, j + 1)
    for k in range(4):
        ni = neighbor_i[k]
        nj = neighbor_j[k]
        idx = base + 1 + k
        if 0 <= ni and ni < n and 0 <= nj and nj < n:
            rows[idx] = row
            cols[idx] = ni * n + nj
            vals[idx] = wp.float64(-1.0)
        else:
            rows[idx] = row
            cols[idx] = row
            vals[idx] = wp.float64(0.0)

    x = (wp.float64(i) + wp.float64(0.5)) / wp.float64(n)
    y = (wp.float64(j) + wp.float64(0.5)) / wp.float64(n)
    b[row] = wp.sin(x * wp.float64(3.14159265)) * wp.sin(y * wp.float64(3.14159265))


def main():
    device = "cuda:0"
    n = 128
    ndof = n * n

    rows = wp.empty(ndof * 5, dtype=wp.int32, device=device)
    cols = wp.empty(ndof * 5, dtype=wp.int32, device=device)
    vals = wp.empty(ndof * 5, dtype=wp.float64, device=device)
    b = wp.empty(ndof, dtype=wp.float64, device=device)
    wp.launch(build_laplacian_triplets, dim=(n, n), inputs=[n, rows, cols, vals, b], device=device)

    A = sparse.bsr_from_triplets(ndof, ndof, rows, cols, vals)

    # --- Iterative reference: warp.optim.linear.cg -------------------------------
    x_cg = wp.zeros(ndof, dtype=wp.float64, device=device)
    wp.synchronize()
    t0 = time.perf_counter()
    iters, residual, _ = linear.cg(A, b, x_cg, tol=1e-10, maxiter=10000)
    wp.synchronize()
    t_cg = time.perf_counter() - t0
    print(f"warp.optim.linear.cg: {iters} iterations, residual {residual:.3e}, {t_cg * 1e3:.2f} ms")

    # --- Direct solve: warp_cudss.solve --------------------------------------------
    x_direct = wp.zeros(ndof, dtype=wp.float64, device=device)
    wp.synchronize()
    t0 = time.perf_counter()
    warp_cudss.solve(A, b, x_direct, mtype="spd")
    wp.synchronize()
    t_direct = time.perf_counter() - t0
    print(f"warp_cudss.solve (cuDSS, includes symbolic analysis): {t_direct * 1e3:.2f} ms")

    err = np.max(np.abs(x_cg.numpy() - x_direct.numpy()))
    print(f"max |x_cg - x_cudss| = {err:.3e}")


if __name__ == "__main__":
    main()
