# warp-cuDSS

cuDSS-backed sparse **direct** solver helpers for [NVIDIA Warp](https://github.com/NVIDIA/warp),
usable as a drop-in replacement for the iterative solvers in `warp.optim.linear`
(`cg`, `cr`, `bicgstab`, `gmres`) when a direct factorization is preferable — e.g. for
ill-conditioned systems, systems solved many times with the same sparsity pattern, or
when iterative convergence is unreliable.

Everything stays on-device: matrix and vector data are passed to
[cuDSS](https://docs.nvidia.com/cuda/cudss/) by raw device pointer, and repeated
refactor+solve calls are safe to record into a CUDA graph via `wp.ScopedCapture`.

## Install

Requires an NVIDIA GPU, a Warp build with CUDA support, and the cuDSS shared library
(`libcudss.so.0`, cuDSS >= 0.8), available either via your system package manager, the
[`nvidia-cudss-cu12`](https://pypi.org/project/nvidia-cudss-cu12/) pip wheel, or the
[cuDSS download page](https://developer.nvidia.com/cudss-downloads).

```bash
pip install nvidia-cudss-cu12  # or install cuDSS via your system package manager
pip install -e .
```

If the library isn't found automatically (via `ctypes.util.find_library`, the
`nvidia-cudss-cu12` wheel layout, or the platform loader path), point at it explicitly:

```bash
export CUDSS_LIBRARY_PATH=/path/to/libcudss.so.0
```

## Quick start

Same call shape as `warp.optim.linear.cg(A, b, x, ...)`:

```python
import warp_cudss

warp_cudss.solve(A, b, x, mtype="spd")  # A: warp.sparse.BsrMatrix, b/x: wp.array
```

Self-contained runnable example (5-point Laplacian on a 1D chain):

```python
import numpy as np
import warp as wp
import warp.sparse as sparse
import warp_cudss

wp.init()
device = "cuda:0"

n = 5
rows = wp.array(np.arange(n), dtype=wp.int32, device=device)
cols = wp.array(np.arange(n), dtype=wp.int32, device=device)
vals = wp.array(np.full(n, 4.0), dtype=wp.float64, device=device)
A = sparse.bsr_from_triplets(n, n, rows, cols, vals)  # diag(4, 4, 4, 4, 4)

b = wp.array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), dtype=wp.float64, device=device)
x = wp.zeros(n, dtype=wp.float64, device=device)

warp_cudss.solve(A, b, x, mtype="spd")
print(x.numpy())  # [0.25 0.5  0.75 1.   1.25]
```

`mtype` is one of `"general"` (default, always correct), `"symmetric"`, `"spd"`,
`"hermitian"`, `"hpd"` — pick the tightest one that actually describes your matrix for
the best performance.

`A` may be a scalar CSR matrix (`block_shape == (1, 1)`, the common case for e.g.
`warp.fem` Poisson-type problems) or a block matrix with small dense blocks (e.g.
`wp.mat33d`, as produced by vector-valued elasticity assembly) — block matrices are
expanded to scalar CSR automatically. `x`/`b` may be plain scalar arrays or arrays of
the matching vector dtype (e.g. `wp.vec3d`); either way they just need
`A.shape[0]` total scalar entries.

## Reusing a factorization

When solving the same system repeatedly with only its values changing (e.g. a Newton
iteration, or a simulation loop with a fixed sparsity pattern), pass the returned
solver back in to skip the symbolic analysis and reuse the factorization workspace:

```python
solver = warp_cudss.solve(A, b, x, mtype="spd")
for _ in range(num_newton_iterations):
    update_matrix_values(A)  # your kernels, same sparsity pattern
    warp_cudss.solve(A, b, x, mtype="spd", solver=solver)
solver.release()
```

## CUDA graph capture

For maximum throughput, use the lower-level `CudssSolver` API directly and capture the
per-step work into a graph. `setup()` (which runs symbolic analysis and the first
factorization) must run once, outside any capture; subsequent `refactor()`/`solve()`
calls only touch device memory and are safe to capture and replay:

```python
solver = warp_cudss.CudssSolver(mtype="spd", device=A.device)
solver.setup(A, x, b)  # not capturable

with wp.ScopedCapture(device) as capture:
    assemble_matrix_values(A)  # your kernels, write into A.values in place
    solver.refactor(A)         # capturable
    solver.solve()              # capturable

for _ in range(num_steps):
    wp.capture_launch(capture.graph)
```

**Capture is only valid as long as `A`'s sparsity pattern (its row/column structure) is
unchanged from what `setup()` last saw.** `setup()`'s analysis + first factorization
size and allocate cuDSS's internal workspace/factor buffers for that specific pattern;
`refactor()`/`solve()` reuse those buffers, which is exactly what's safe to record into
a graph. Changing values in place (`assemble_matrix_values(A)` above) and re-running
`refactor()` is fine and expected. If the *pattern* changes, the buffer sizes may no
longer be valid — you need to call `setup()` again outside capture and re-capture the
graph. The same stability requirement applies to array addresses: `A`'s CSR storage,
`x`, and `b` must keep a stable device address for the solver's lifetime — don't
reallocate them between `setup()` and later `refactor()`/`solve()` calls.

A full runnable version of this pattern, including on-device value updates via a Warp
kernel and multiple graph replays, is in
[`examples/example_graph_capture_timestepping.py`](examples/example_graph_capture_timestepping.py).

## Examples

- [`examples/example_poisson_2d.py`](examples/example_poisson_2d.py) — scalar 2D Poisson
  problem, side-by-side with `warp.optim.linear.cg`.
- [`examples/example_graph_capture_timestepping.py`](examples/example_graph_capture_timestepping.py) —
  block (3x3) system inside a CUDA-graph-captured time-stepping loop.

## Tests

```bash
pip install -e ".[test]"
pytest tests/
```

## How it works

See [`CUDSS_WARP_INTEGRATION_NOTES.md`](CUDSS_WARP_INTEGRATION_NOTES.md) for the
integration notes this package is based on. In short:

- `warp_cudss/_bindings.py` is a minimal `ctypes` wrapper around the cuDSS 0.8 C API
  (handle/config/data lifecycle, CSR/dense matrix descriptors, `cudssExecute` phases).
- `warp_cudss/_csr.py` expands a block-sparse `warp.sparse.BsrMatrix` into scalar CSR,
  which is what cuDSS consumes. The scalar sparsity is built once per pattern; only the
  scalar values array is rebuilt (with a Warp kernel, on-device) when block values
  change.
- `warp_cudss/solver.py` owns the cuDSS lifecycle: analysis + first factorization run
  once in `setup()`, and `refactor()`/`solve()` reuse the symbolic factorization and
  are graph-capturable.

## License

Apache-2.0, matching Warp itself.
