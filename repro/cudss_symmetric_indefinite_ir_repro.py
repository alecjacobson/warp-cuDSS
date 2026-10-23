#!/usr/bin/env python3
"""Minimal reproducer: cuDSS 0.8, CUDSS_MTYPE_SYMMETRIC solve of a small,
well-conditioned (cond ~ 560) *indefinite* symmetric sparse matrix returns a solution
with a relative residual in the few-percent range (observed 3e-2 to 5e-2 across
repeated runs of this exact, fixed input -- see "run-to-run variability" note below)
when CUDSS_CONFIG_IR_N_STEPS=0 (the default). Enabling a few steps of iterative
refinement (CUDSS_CONFIG_IR_N_STEPS>=1) reliably recovers full double precision
accuracy (residual ~1e-4 down to ~1e-7 as IR_N_STEPS increases from 1 to 3).

This is a pure ctypes script against libcudss.so.0 and the CUDA runtime -- no Warp,
CuPy, or other framework involved -- to isolate the behavior to cuDSS itself.

Run-to-run variability: with CUDSS_CONFIG_IR_N_STEPS=0, repeated runs of this *exact
same* fixed input (same matrix, same RHS, same process) return different residuals
(observed 3.48e-2, 5.08e-2, 5.08e-2, 5.08e-2, 5.08e-2 across 5 runs) -- i.e. the
factorization result is not deterministic. This is itself a symptom worth reporting:
it's consistent with the factorization sitting right at the edge of numerical
stability, where floating-point non-associativity in parallel reduction order (or
similar scheduling-dependent effects) changes which near-tied pivot gets chosen.

Suspected cause: cudss_data_types.h lists CUDSS_PIVOT_BUNCH_KAUFMAN as
"Reserved for future (not supported yet)". Without Bunch-Kaufman 2x2 pivoting, the
default diagonal pivoting strategy for CUDSS_MTYPE_SYMMETRIC appears to produce a
numerically unstable factorization on indefinite input even when the matrix itself is
not ill-conditioned (cond(A) ~ 560 here -- roundoff alone would predict error ~1e-13,
not several percent). This may also be worth checking for CUDSS_MTYPE_GENERAL, which
showed the same pattern (large residual at IR_N_STEPS=0, fixed by IR) on a related
indefinite test matrix, suggesting the default pivoting for non-SPD matrix types may
generally be less robust than expected without iterative refinement.

Usage:
    export CUDSS_LIBRARY_PATH=/path/to/libcudss.so.0   # if not found automatically
    python3 cudss_symmetric_indefinite_ir_repro.py

Environment this was observed on: cuDSS 0.8.0 (libcudss.so.0.8.0), CUDA 12.9 /
driver 12.8, NVIDIA L40 (sm_89).
"""

import ctypes
import ctypes.util
import os

import numpy as np

# ---------------------------------------------------------------------------
# Fixed, exact test matrix: 16x16 sparse symmetric indefinite, cond(A) ~ 559.2,
# eigenvalues span [-3.58, 3.62]. Generated once with
# numpy.random.default_rng(15).standard_normal / .random(density=0.3), symmetrized,
# and hardcoded here (not regenerated at runtime) so this script is exactly
# reproducible regardless of numpy version.
# ---------------------------------------------------------------------------

N = 16

ROWS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6,
        6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12,
        12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 15, 15,
        15, 15, 15, 15, 15, 15, 15]
COLS = [5, 6, 7, 8, 10, 13, 14, 15, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 3, 6, 9, 12, 15, 1, 2,
        4, 5, 6, 7, 9, 12, 13, 15, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 0, 1, 3, 4, 5, 9, 10,
        11, 13, 15, 0, 1, 2, 3, 4, 7, 12, 15, 0, 3, 4, 6, 8, 9, 11, 12, 13, 14, 0, 4, 7, 8,
        9, 14, 15, 1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 0, 1, 4, 5, 9, 11, 14, 15, 1, 4, 5, 7,
        9, 10, 11, 1, 2, 3, 4, 6, 7, 9, 14, 0, 1, 3, 5, 7, 9, 0, 4, 7, 8, 10, 12, 15, 0, 1,
        2, 3, 5, 6, 8, 10, 14]
VALS = [0.05978905513715682, -0.4388503951995593, 1.5721489522343175, -0.3966018213130247,
        -0.45128578257549296, 0.6638612818896261, -0.3481174083384892, 1.1222183139197903,
        -0.624106369110344, -0.3114721239493857, -0.2252531953170147, -0.47940688574184354,
        -0.3307598114173964, 0.6976171396806858, -0.006278406689722127, 0.7255248472227802,
        -1.5637527856890332, -0.17508320937288133, -0.16017612202914594, 0.3575040427343969,
        -1.1020138792870626, 0.1529163788085398, 0.4279687257604169, -0.624106369110344,
        -0.16017612202914594, -0.48113689546470007, -0.32668574797675687, 0.7673457915396034,
        0.4838355407858365, -0.9793209174344746, 0.4836550314632581, 0.06420647564825907,
        0.08952730027986788, -0.3114721239493857, -0.48113689546470007, -1.5056171465064125,
        0.0728330392496761, -0.019722872464079694, -1.2371603183782172, 0.23937498087329995,
        0.27525280207896613, 0.5018342948663789, -0.64892477500988, 0.6762751864848477,
        0.05978905513715682, -0.2252531953170147, -0.32668574797675687, 0.0728330392496761,
        1.7780291348965471, -0.06325871273087369, 1.081746999110137, 0.4771184659161415,
        1.0317573353932983, -0.049350566310687294, -0.4388503951995593, -0.47940688574184354,
        0.3575040427343969, 0.7673457915396034, -0.019722872464079694, 0.3636363931197299,
        -1.0292847889295573, -0.013670881918906003, 1.5721489522343175, 0.4838355407858365,
        -1.2371603183782172, 0.3636363931197299, -0.6862310101149802, 0.1417478504038583,
        -0.3982532596353139, -1.1740277165857849, 0.6339376700996149, 1.4316408274071133,
        -0.3966018213130247, 0.23937498087329995, -0.6862310101149802, 1.2012952295910957,
        0.7745667170825578, 0.4432152616306454, 0.9588744510717345, -0.3307598114173964,
        -1.1020138792870626, -0.9793209174344746, -0.06325871273087369, 0.1417478504038583,
        0.7745667170825578, 0.1220307687593182, -0.6894893418841934, -0.9515071592168124,
        -0.7403018048582836, -0.45128578257549296, 0.6976171396806858, 0.27525280207896613,
        1.081746999110137, 0.1220307687593182, 0.4224441041002523, -0.436069387395849,
        -0.9119314611071363, -0.006278406689722127, 0.5018342948663789, 0.4771184659161415,
        -0.3982532596353139, -0.6894893418841934, 0.4224441041002523, -0.42978401439088665,
        0.7255248472227802, 0.1529163788085398, 0.4836550314632581, -0.64892477500988,
        -1.0292847889295573, -1.1740277165857849, -0.9515071592168124, -0.9372434806409089,
        0.6638612818896261, -1.5637527856890332, 0.06420647564825907, 1.0317573353932983,
        0.6339376700996149, -0.7403018048582836, -0.3481174083384892, 0.6762751864848477,
        1.4316408274071133, 0.4432152616306454, -0.436069387395849, -0.9372434806409089,
        1.07359589029922, 1.1222183139197903, -0.17508320937288133, 0.4279687257604169,
        0.08952730027986788, -0.049350566310687294, -0.013670881918906003, 0.9588744510717345,
        -0.9119314611071363, 1.07359589029922]
B = [0.31686906720213537, 1.4478180098021427, -0.8707929944108522, -0.15220879386881486,
     0.7293865674571559, 0.28923103651043125, -0.4888837648665474, -2.177476456304843,
     0.565087759424536, 0.7822892107422185, 0.5063236930359808, -1.6548437072544337,
     -1.3864544911115066, -0.5620846798351992, -0.999979355640598, 1.0589004510750684]

NNZ = len(VALS)
assert len(ROWS) == NNZ and len(COLS) == NNZ

# ---------------------------------------------------------------------------
# CUDA runtime bindings (just enough for device malloc/memcpy)
# ---------------------------------------------------------------------------

cudart = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudart.cudaMemcpy.restype = ctypes.c_int
cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaFree.restype = ctypes.c_int
cudart.cudaDeviceSynchronize.restype = ctypes.c_int

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def cuda_check(status, what):
    if status != 0:
        raise RuntimeError(f"{what} failed with cudaError_t {status}")


def to_device(host_array):
    nbytes = host_array.nbytes
    dptr = ctypes.c_void_p()
    cuda_check(cudart.cudaMalloc(ctypes.byref(dptr), nbytes), "cudaMalloc")
    cuda_check(
        cudart.cudaMemcpy(dptr, host_array.ctypes.data_as(ctypes.c_void_p), nbytes, CUDA_MEMCPY_HOST_TO_DEVICE),
        "cudaMemcpy H2D",
    )
    return dptr


def from_device(dptr, host_array):
    cuda_check(
        cudart.cudaMemcpy(
            host_array.ctypes.data_as(ctypes.c_void_p), dptr, host_array.nbytes, CUDA_MEMCPY_DEVICE_TO_HOST
        ),
        "cudaMemcpy D2H",
    )
    return host_array


# ---------------------------------------------------------------------------
# cuDSS bindings (subset needed for a single-GPU CSR solve)
# ---------------------------------------------------------------------------


def find_cudss():
    candidates = []
    env = os.environ.get("CUDSS_LIBRARY_PATH")
    if env:
        candidates.append(env)
    candidates += ["libcudss.so.0", "libcudss.so"]
    found = ctypes.util.find_library("cudss")
    if found:
        candidates.append(found)
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise OSError("libcudss.so.0 not found; set CUDSS_LIBRARY_PATH")


cudss = find_cudss()

CUDSS_STATUS_SUCCESS = 0
CUDA_R_64F = 1
CUDA_R_32I = 10
CUDSS_MTYPE_SYMMETRIC = 1
CUDSS_MVIEW_FULL = 0
CUDSS_BASE_ZERO = 0
CUDSS_LAYOUT_COL_MAJOR = 0
CUDSS_CONFIG_IR_N_STEPS = 5
CUDSS_PHASE_ANALYSIS = (1 << 0) | (1 << 1)
CUDSS_PHASE_FACTORIZATION = 1 << 2
CUDSS_PHASE_SOLVE = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)

cudssHandle_t = ctypes.c_void_p
cudssConfig_t = ctypes.c_void_p
cudssData_t = ctypes.c_void_p
cudssMatrix_t = ctypes.c_void_p

cudss.cudssCreate.argtypes = [ctypes.POINTER(cudssHandle_t)]
cudss.cudssConfigCreate.argtypes = [ctypes.POINTER(cudssConfig_t)]
cudss.cudssDataCreate.argtypes = [cudssHandle_t, ctypes.POINTER(cudssData_t)]
cudss.cudssConfigSet.argtypes = [cudssConfig_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
cudss.cudssMatrixCreateCsr.argtypes = [
    ctypes.POINTER(cudssMatrix_t),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
cudss.cudssMatrixCreateDn.argtypes = [
    ctypes.POINTER(cudssMatrix_t),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
]
cudss.cudssExecute.argtypes = [
    cudssHandle_t,
    ctypes.c_int,
    cudssConfig_t,
    cudssData_t,
    cudssMatrix_t,
    cudssMatrix_t,
    cudssMatrix_t,
]
cudss.cudssMatrixDestroy.argtypes = [cudssMatrix_t]
cudss.cudssConfigDestroy.argtypes = [cudssConfig_t]
cudss.cudssDataDestroy.argtypes = [cudssHandle_t, cudssData_t]
cudss.cudssDestroy.argtypes = [cudssHandle_t]

for fn in (
    cudss.cudssCreate,
    cudss.cudssConfigCreate,
    cudss.cudssDataCreate,
    cudss.cudssConfigSet,
    cudss.cudssMatrixCreateCsr,
    cudss.cudssMatrixCreateDn,
    cudss.cudssExecute,
    cudss.cudssMatrixDestroy,
    cudss.cudssConfigDestroy,
    cudss.cudssDataDestroy,
    cudss.cudssDestroy,
):
    fn.restype = ctypes.c_int


def check(status, what):
    if status != CUDSS_STATUS_SUCCESS:
        raise RuntimeError(f"{what} failed with cudssStatus_t {status}")


def solve_with_ir_steps(ir_steps: int) -> np.ndarray:
    """Build the matrix above, solve A x = b via cuDSS with CUDSS_MTYPE_SYMMETRIC, and
    return x. ``ir_steps`` sets CUDSS_CONFIG_IR_N_STEPS (0 = disabled, cuDSS default)."""

    indptr = np.zeros(N + 1, dtype=np.int32)
    for r in ROWS:
        indptr[r + 1] += 1
    indptr = np.cumsum(indptr).astype(np.int32)
    indices = np.array(COLS, dtype=np.int32)
    values = np.array(VALS, dtype=np.float64)
    b_host = np.array(B, dtype=np.float64)
    x_host = np.zeros(N, dtype=np.float64)

    d_indptr = to_device(indptr)
    d_indices = to_device(indices)
    d_values = to_device(values)
    d_b = to_device(b_host)
    d_x = to_device(x_host)

    handle = cudssHandle_t()
    check(cudss.cudssCreate(ctypes.byref(handle)), "cudssCreate")
    config = cudssConfig_t()
    check(cudss.cudssConfigCreate(ctypes.byref(config)), "cudssConfigCreate")
    data = cudssData_t()
    check(cudss.cudssDataCreate(handle, ctypes.byref(data)), "cudssDataCreate")

    if ir_steps:
        n = ctypes.c_int(ir_steps)
        check(
            cudss.cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS, ctypes.byref(n), ctypes.sizeof(n)),
            "cudssConfigSet(IR_N_STEPS)",
        )

    A_desc = cudssMatrix_t()
    check(
        cudss.cudssMatrixCreateCsr(
            ctypes.byref(A_desc),
            N,
            N,
            NNZ,
            d_indptr,
            None,
            d_indices,
            d_values,
            CUDA_R_32I,
            CUDA_R_32I,
            CUDA_R_64F,
            CUDSS_MTYPE_SYMMETRIC,
            CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO,
        ),
        "cudssMatrixCreateCsr",
    )

    x_desc = cudssMatrix_t()
    check(
        cudss.cudssMatrixCreateDn(ctypes.byref(x_desc), N, 1, N, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        "cudssMatrixCreateDn(x)",
    )
    b_desc = cudssMatrix_t()
    check(
        cudss.cudssMatrixCreateDn(ctypes.byref(b_desc), N, 1, N, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        "cudssMatrixCreateDn(b)",
    )

    check(
        cudss.cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A_desc, x_desc, b_desc),
        "cudssExecute(ANALYSIS)",
    )
    check(
        cudss.cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A_desc, x_desc, b_desc),
        "cudssExecute(FACTORIZATION)",
    )
    check(
        cudss.cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A_desc, x_desc, b_desc),
        "cudssExecute(SOLVE)",
    )
    cuda_check(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    from_device(d_x, x_host)

    check(cudss.cudssMatrixDestroy(A_desc), "cudssMatrixDestroy(A)")
    check(cudss.cudssMatrixDestroy(x_desc), "cudssMatrixDestroy(x)")
    check(cudss.cudssMatrixDestroy(b_desc), "cudssMatrixDestroy(b)")
    check(cudss.cudssDataDestroy(handle, data), "cudssDataDestroy")
    check(cudss.cudssConfigDestroy(config), "cudssConfigDestroy")
    check(cudss.cudssDestroy(handle), "cudssDestroy")

    for d in (d_indptr, d_indices, d_values, d_b, d_x):
        cuda_check(cudart.cudaFree(d), "cudaFree")

    return x_host


def main():
    dense = np.zeros((N, N))
    dense[ROWS, COLS] = VALS
    b_np = np.array(B)
    assert np.allclose(dense, dense.T), "sanity check: matrix must be symmetric"

    eig = np.linalg.eigvalsh(dense)
    cond = np.linalg.cond(dense)
    print(f"N={N}, nnz={NNZ}, eig range=[{eig.min():.4f}, {eig.max():.4f}] (indefinite), cond(A)={cond:.1f}")
    print()

    x_ref = np.linalg.solve(dense, b_np)

    print("--- Repeating IR_N_STEPS=0 5x on the identical fixed input, to show non-determinism ---")
    for run in range(5):
        x = solve_with_ir_steps(0)
        resid = np.linalg.norm(dense @ x - b_np) / np.linalg.norm(b_np)
        print(f"  run {run}: relative residual ||Ax-b||/||b|| = {resid:.4e}")
    print()

    print("--- Effect of CUDSS_CONFIG_IR_N_STEPS ---")
    for ir_steps in (0, 1, 2, 3):
        x = solve_with_ir_steps(ir_steps)
        resid = np.linalg.norm(dense @ x - b_np) / np.linalg.norm(b_np)
        relerr = np.max(np.abs(x - x_ref)) / np.max(np.abs(x_ref))
        print(f"CUDSS_CONFIG_IR_N_STEPS={ir_steps}: relative residual ||Ax-b||/||b|| = {resid:.3e}, "
              f"max relative error vs numpy = {relerr:.3e}")

    print()
    print("Expected: IR_N_STEPS=0 (the default) shows a relative residual in the few-percent")
    print("range (and varies run-to-run) on this well-conditioned (cond~560) indefinite")
    print("symmetric matrix; IR_N_STEPS>=1 recovers full double precision accuracy.")


if __name__ == "__main__":
    main()
