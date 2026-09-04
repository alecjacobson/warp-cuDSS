"""Graph-capturable cuDSS solve inside a simulation-style time-stepping loop.

Models a common pattern in implicit time integration: a block-sparse system matrix
(e.g. per-node 3x3 blocks, as produced by an elasticity or cloth solver) whose
*sparsity pattern* is fixed across timesteps but whose *values* change every step
(e.g. because of a stiffness ramp or changing material state). The whole per-step
work -- updating matrix values, refactorizing, and solving -- is captured once into a
CUDA graph and then replayed, so steady-state timesteps pay no Python/launch overhead.

Run with:
    python examples/example_graph_capture_timestepping.py
"""

import numpy as np
import warp as wp
import warp.sparse as sparse

import warp_cudss


@wp.kernel
def build_block_diag_triplets(
    nb: int,
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=wp.mat33d),
):
    """Block tri-diagonal system: node i is coupled to i-1 and i+1 (a 1D chain of 3-dof nodes)."""
    i = wp.tid()
    base = i * 3
    ident = wp.identity(n=3, dtype=wp.float64)

    rows[base + 0] = i
    cols[base + 0] = i
    vals[base + 0] = ident * wp.float64(6.0)

    if i > 0:
        rows[base + 1] = i
        cols[base + 1] = i - 1
        vals[base + 1] = ident * wp.float64(-1.0)
    else:
        rows[base + 1] = i
        cols[base + 1] = i
        vals[base + 1] = ident * wp.float64(0.0)

    if i + 1 < nb:
        rows[base + 2] = i
        cols[base + 2] = i + 1
        vals[base + 2] = ident * wp.float64(-1.0)
    else:
        rows[base + 2] = i
        cols[base + 2] = i
        vals[base + 2] = ident * wp.float64(0.0)


@wp.kernel
def ramp_stiffness(
    step: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat33d),
):
    """Scale the diagonal blocks by a factor that increases every replayed step, entirely on-device."""
    row = wp.tid()
    scale = wp.float64(1.0) + wp.float64(0.1) * wp.float64(step[0])
    for k in range(offsets[row], offsets[row + 1]):
        if columns[k] == row:
            values[k] = wp.identity(n=3, dtype=wp.float64) * (wp.float64(6.0) * scale)


@wp.kernel
def increment_step(step: wp.array(dtype=wp.int32)):
    step[0] = step[0] + 1


def main():
    device = "cuda:0"
    nb = 64  # number of 3-dof nodes
    n = nb * 3

    rows = wp.empty(nb * 3, dtype=wp.int32, device=device)
    cols = wp.empty(nb * 3, dtype=wp.int32, device=device)
    vals = wp.empty(nb * 3, dtype=wp.mat33d, device=device)
    wp.launch(build_block_diag_triplets, dim=nb, inputs=[nb, rows, cols, vals], device=device)
    A = sparse.bsr_from_triplets(nb, nb, rows, cols, vals)

    b = wp.array(np.ones(n), dtype=wp.float64, device=device)
    x = wp.zeros(n, dtype=wp.float64, device=device)
    step = wp.zeros(1, dtype=wp.int32, device=device)

    solver = warp_cudss.CudssSolver(mtype="spd", device=device)
    solver.setup(A, x, b)  # analysis + first factorization: run once, outside capture

    with wp.ScopedCapture(device) as capture:
        wp.launch(ramp_stiffness, dim=nb * 3, inputs=[step, A.offsets, A.columns, A.values], device=device)
        solver.refactor(A)
        solver.solve()
        wp.launch(increment_step, dim=1, inputs=[step], device=device)

    num_steps = 10
    for i in range(num_steps):
        wp.capture_launch(capture.graph)
        wp.synchronize()
        print(f"step {i}: max|x| = {np.max(np.abs(x.numpy())):.6f}")

    solver.release()


if __name__ == "__main__":
    main()
