# cuDSS integration with Warp

Hi Alec,

The current cuDSS prototype is here:

- [Experimental branch: `jonathanw/quadcopter-amr-topology-optimization`](https://gitlab-master.nvidia.com/Devtech-Compute/warp/-/tree/jonathanw/quadcopter-amr-topology-optimization)
- [Implementation file: `adaptive_topology_optimization_utils.py`](https://gitlab-master.nvidia.com/Devtech-Compute/warp/-/blob/jonathanw/quadcopter-amr-topology-optimization/warp/examples/fem/adaptive_topology_optimization_utils.py)

The linked file contains the current optional cuDSS adapter prototype. The integration pattern applies directly to a standalone Warp sparse-direct solver.

## Recommended hook

Keep cuDSS as an optional runtime dependency rather than linking it into the standard Warp build:

1. Dynamically load `libcudss.so.0`.
2. Let Warp own the CSR arrays and pass their CUDA device pointers to cuDSS.
3. Set cuDSS to Warp's current CUDA stream.
4. Retain the cuDSS handle, configuration, data, and matrix descriptors between solves.
5. Run symbolic analysis only when the sparsity pattern changes.
6. When only values change, update the values pointer or values in place and run refactorization.
7. Reuse the factorization for additional right-hand sides.

The core lifecycle is:

```python
class CudssFactorization:
    def __init__(self, device, library_path):
        self.lib = ctypes.CDLL(library_path)
        self.handle = cudssCreate()
        self.config = cudssConfigCreate()
        self.data = cudssDataCreate(self.handle)

        stream = wp.get_stream(device)
        cudssSetStream(self.handle, stream.cuda_stream)

        self.structure = None
        self.A = None

    def update(self, csr):
        structure = (
            csr.nrow,
            csr.ncol,
            csr.nnz,
            csr.offsets.ptr,
            csr.columns.ptr,
        )

        if structure != self.structure:
            self.A = cudssMatrixCreateCsr(
                csr.nrow,
                csr.ncol,
                csr.nnz,
                csr.offsets.ptr,
                None,
                csr.columns.ptr,
                csr.values.ptr,
                CUDA_R_32I,       # row-offset type in cuDSS 0.8+
                CUDA_R_32I,       # column-index type
                CUDA_R_64F,
                CUDSS_MTYPE_SPD,
                CUDSS_MVIEW_FULL,
                CUDSS_BASE_ZERO,
            )
            cudssExecute(
                self.handle,
                CUDSS_PHASE_ANALYSIS,
                self.config,
                self.data,
                self.A,
                self.x_desc,
                self.b_desc,
            )
            self.structure = structure
        else:
            cudssMatrixSetValues(self.A, csr.values.ptr)

        cudssExecute(
            self.handle,
            CUDSS_PHASE_REFACTORIZATION,
            self.config,
            self.data,
            self.A,
            self.x_desc,
            self.b_desc,
        )

    def solve(self, x, b):
        cudssMatrixSetValues(self.x_desc, x.ptr)
        cudssMatrixSetValues(self.b_desc, b.ptr)
        cudssExecute(
            self.handle,
            CUDSS_PHASE_SOLVE,
            self.config,
            self.data,
            self.A,
            self.x_desc,
            self.b_desc,
        )
```

This is illustrative pseudocode. A production `ctypes` wrapper needs exact `argtypes`, status checks, version validation, and deterministic destruction of every cuDSS object.

## Warp matrix storage

cuDSS consumes scalar CSR. Warp FEM elasticity matrices are normally block CSR/BSR, such as `wp.mat33d`, so they need to be expanded to scalar CSR.

The useful approach is to create the BSR-to-CSR sparsity mapping once. When FEM assembly changes the stiffness values, update only the scalar CSR values on the GPU. Do not rebuild row offsets and column indices unless the sparsity pattern changes.

The Warp arrays must remain alive and at stable addresses while their pointers are referenced by cuDSS descriptors.

For the projected FP64 elasticity matrices in this example, use:

```c
CUDSS_MTYPE_SPD
CUDSS_MVIEW_FULL
CUDSS_BASE_ZERO
CUDA_R_64F
```

## Runtime options

Make the cuDSS shared-library path available to the adapter explicitly or through the platform loader:

```bash
export LD_LIBRARY_PATH="$CUDSS_DIR/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

With dynamic loading, Warp has no additional compile-time option. Runtime requirements are a compatible driver/CUDA toolkit, cuDSS, and cuBLAS. Library discovery should preferably check:

1. An explicit Python API or command-line library path
2. `CUDSS_LIBRARY_PATH`
3. `ctypes.util.find_library("cudss")`
4. The normal platform loader path

## Native-link alternative

If the adapter is eventually implemented as native C++ code:

```bash
nvcc adapter.cpp \
  -I"$CUDSS_DIR/include" \
  -L"$CUDSS_DIR/lib" \
  -l:libcudss.so.0 \
  -lcublas \
  -o warp_cudss_adapter
```

An installation providing the unversioned symlink can use `-lcudss`. The explicit soname is needed for some pip-installed cuDSS packages.

## Two important cautions

- cuDSS 0.8 added separate `offsetType` and `indexType` arguments to its CSR APIs. The `ctypes` bindings must be versioned or require a known cuDSS ABI.
- Set the cuDSS stream from `wp.get_stream(device)`. Keep analysis outside CUDA graph capture. Factorization and solve can be considered for capture only after all matrix, vector, and workspace pointers are persistent.

## NVIDIA references

- [Getting Started and build options](https://docs.nvidia.com/cuda/cudss/getting_started.html)
- [API functions](https://docs.nvidia.com/cuda/cudss/functions.html)
- [Matrix and data types](https://docs.nvidia.com/cuda/cudss/types.html)
- [cuDSS 0.8 migration guide](https://docs.nvidia.com/cuda/cudss/migration_guide.html)
- [Reuse and performance guidance](https://docs.nvidia.com/cuda/cudss/tips_and_tricks.html)
