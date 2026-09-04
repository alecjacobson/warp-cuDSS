"""Minimal ctypes bindings for the subset of the cuDSS 0.8 C API used by this package.

Only the entry points needed for a single-GPU, single right-hand-side sparse direct
solve are wrapped: handle/config/data lifecycle, CSR/dense matrix wrappers, and the
analysis / factorization / refactorization / solve phases of ``cudssExecute``.

See:
- https://docs.nvidia.com/cuda/cudss/functions.html
- https://docs.nvidia.com/cuda/cudss/types.html
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

__all__ = [
    "CudssError",
    "get_library",
    # enums
    "cudssStatus_t",
    "cudssDataType_t",
    "cudssMatrixType_t",
    "cudssMatrixViewType_t",
    "cudssIndexBase_t",
    "cudssLayout_t",
    "cudssPhase_t",
    "cudssConfigParam_t",
    "cudssReorderingAlg_t",
    "MTYPE",
    "DATA_TYPE",
]

# ---------------------------------------------------------------------------
# Enums (values copied from cudss_data_types.h / cudss.h, cuDSS 0.8)
# ---------------------------------------------------------------------------

cudssStatus_t = ctypes.c_int
CUDSS_STATUS_SUCCESS = 0
CUDSS_STATUS_NOT_INITIALIZED = 1
CUDSS_STATUS_ALLOC_FAILED = 2
CUDSS_STATUS_INVALID_VALUE = 3
CUDSS_STATUS_NOT_SUPPORTED = 4
CUDSS_STATUS_EXECUTION_FAILED = 5
CUDSS_STATUS_INTERNAL_ERROR = 6
CUDSS_STATUS_IR_FAILED = 7

_STATUS_NAMES = {
    CUDSS_STATUS_SUCCESS: "CUDSS_STATUS_SUCCESS",
    CUDSS_STATUS_NOT_INITIALIZED: "CUDSS_STATUS_NOT_INITIALIZED",
    CUDSS_STATUS_ALLOC_FAILED: "CUDSS_STATUS_ALLOC_FAILED",
    CUDSS_STATUS_INVALID_VALUE: "CUDSS_STATUS_INVALID_VALUE",
    CUDSS_STATUS_NOT_SUPPORTED: "CUDSS_STATUS_NOT_SUPPORTED",
    CUDSS_STATUS_EXECUTION_FAILED: "CUDSS_STATUS_EXECUTION_FAILED",
    CUDSS_STATUS_INTERNAL_ERROR: "CUDSS_STATUS_INTERNAL_ERROR",
    CUDSS_STATUS_IR_FAILED: "CUDSS_STATUS_IR_FAILED",
}

cudssDataType_t = ctypes.c_int
# cudssDataType_t reuses cudaDataType_t values for the floating point types.
CUDA_R_32F = 0
CUDA_R_64F = 1
CUDA_C_32F = 4
CUDA_C_64F = 5
CUDA_R_32I = 10
CUDA_R_64I = 24  # matches library_types.h CUDA_R_64I

DATA_TYPE = {
    "float32": CUDA_R_32F,
    "float64": CUDA_R_64F,
    "complex64": CUDA_C_32F,
    "complex128": CUDA_C_64F,
    "int32": CUDA_R_32I,
    "int64": CUDA_R_64I,
}

cudssMatrixType_t = ctypes.c_int
CUDSS_MTYPE_GENERAL = 0
CUDSS_MTYPE_SYMMETRIC = 1
CUDSS_MTYPE_HERMITIAN = 2
CUDSS_MTYPE_SPD = 3
CUDSS_MTYPE_HPD = 4

MTYPE = {
    "general": CUDSS_MTYPE_GENERAL,
    "symmetric": CUDSS_MTYPE_SYMMETRIC,
    "hermitian": CUDSS_MTYPE_HERMITIAN,
    "spd": CUDSS_MTYPE_SPD,
    "hpd": CUDSS_MTYPE_HPD,
}

cudssMatrixViewType_t = ctypes.c_int
CUDSS_MVIEW_FULL = 0
CUDSS_MVIEW_LOWER = 1
CUDSS_MVIEW_UPPER = 2

MVIEW = {
    "full": CUDSS_MVIEW_FULL,
    "lower": CUDSS_MVIEW_LOWER,
    "upper": CUDSS_MVIEW_UPPER,
}

cudssIndexBase_t = ctypes.c_int
CUDSS_BASE_ZERO = 0
CUDSS_BASE_ONE = 1

cudssLayout_t = ctypes.c_int
CUDSS_LAYOUT_COL_MAJOR = 0
CUDSS_LAYOUT_ROW_MAJOR = 1

cudssPhase_t = ctypes.c_int
CUDSS_PHASE_REORDERING = 1 << 0
CUDSS_PHASE_SYMBOLIC_FACTORIZATION = 1 << 1
CUDSS_PHASE_ANALYSIS = CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION
CUDSS_PHASE_FACTORIZATION = 1 << 2
CUDSS_PHASE_REFACTORIZATION = 1 << 3
CUDSS_PHASE_SOLVE_FWD_PERM = 1 << 4
CUDSS_PHASE_SOLVE_FWD = 1 << 5
CUDSS_PHASE_SOLVE_DIAG = 1 << 6
CUDSS_PHASE_SOLVE_BWD = 1 << 7
CUDSS_PHASE_SOLVE_BWD_PERM = 1 << 8
CUDSS_PHASE_SOLVE_REFINEMENT = 1 << 9
CUDSS_PHASE_SOLVE = (
    CUDSS_PHASE_SOLVE_FWD_PERM
    | CUDSS_PHASE_SOLVE_FWD
    | CUDSS_PHASE_SOLVE_DIAG
    | CUDSS_PHASE_SOLVE_BWD
    | CUDSS_PHASE_SOLVE_BWD_PERM
    | CUDSS_PHASE_SOLVE_REFINEMENT
)

cudssConfigParam_t = ctypes.c_int
CUDSS_CONFIG_REORDERING_ALG = 0
CUDSS_CONFIG_FACTORIZATION_ALG = 1
CUDSS_CONFIG_SOLVE_ALG = 2
CUDSS_CONFIG_MATCHING_ALG = 3
CUDSS_CONFIG_SOLVE_MODE = 4
CUDSS_CONFIG_IR_N_STEPS = 5
CUDSS_CONFIG_IR_TOL = 6
CUDSS_CONFIG_PIVOT_TYPE = 7
CUDSS_CONFIG_PIVOT_THRESHOLD = 8
CUDSS_CONFIG_PIVOT_EPSILON = 9
CUDSS_CONFIG_MAX_LU_NNZ = 10
CUDSS_CONFIG_HYBRID_MEMORY_MODE = 11
CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT = 12
CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY = 13
CUDSS_CONFIG_HOST_NTHREADS = 14
CUDSS_CONFIG_HYBRID_EXECUTE_MODE = 15
CUDSS_CONFIG_PIVOT_EPSILON_ALG = 16
CUDSS_CONFIG_ND_NLEVELS = 17
CUDSS_CONFIG_UBATCH_SIZE = 18
CUDSS_CONFIG_UBATCH_INDEX = 19
CUDSS_CONFIG_USE_SUPERPANELS = 20
CUDSS_CONFIG_DEVICE_COUNT = 21
CUDSS_CONFIG_DEVICE_INDICES = 22
CUDSS_CONFIG_SCHUR_MODE = 23
CUDSS_CONFIG_DETERMINISTIC_MODE = 24
CUDSS_CONFIG_ND_UBFACTOR = 25

cudssReorderingAlg_t = ctypes.c_int
CUDSS_REORDERING_ALG_DEFAULT = 0
CUDSS_REORDERING_ALG_BTF_COLAMD = 1
CUDSS_REORDERING_ALG_COLAMD = 2
CUDSS_REORDERING_ALG_AMD = 3
CUDSS_REORDERING_ALG_NESTED_DISSECTION = 4
CUDSS_REORDERING_ALG_NONE = 5

REORDERING_ALG = {
    "default": CUDSS_REORDERING_ALG_DEFAULT,
    "btf_colamd": CUDSS_REORDERING_ALG_BTF_COLAMD,
    "colamd": CUDSS_REORDERING_ALG_COLAMD,
    "amd": CUDSS_REORDERING_ALG_AMD,
    "nested_dissection": CUDSS_REORDERING_ALG_NESTED_DISSECTION,
    "none": CUDSS_REORDERING_ALG_NONE,
}

# Opaque handles are just void* to us.
cudssHandle_t = ctypes.c_void_p
cudssMatrix_t = ctypes.c_void_p
cudssData_t = ctypes.c_void_p
cudssConfig_t = ctypes.c_void_p


class CudssError(RuntimeError):
    """Raised when a cuDSS API call returns a non-success status."""

    def __init__(self, status: int, call: str):
        self.status = status
        name = _STATUS_NAMES.get(status, f"UNKNOWN({status})")
        super().__init__(f"{call} failed with status {name} ({status})")


def _check(status, call_name):
    if status != CUDSS_STATUS_SUCCESS:
        raise CudssError(status, call_name)


_LIBRARY_CANDIDATES = (
    "libcudss.so.0",
    "libcudss.so",
)


def _candidate_paths():
    env_path = os.environ.get("CUDSS_LIBRARY_PATH")
    if env_path:
        yield env_path

    for name in _LIBRARY_CANDIDATES:
        yield name

    found = ctypes.util.find_library("cudss")
    if found:
        yield found

    # nvidia-cudss-cu12 pip wheel layout: nvidia/cudss/lib/libcudss.so.0
    try:
        import nvidia.cudss  # noqa: PLC0415

        pkg_dir = os.path.dirname(nvidia.cudss.__file__)
        for name in _LIBRARY_CANDIDATES:
            yield os.path.join(pkg_dir, "lib", name)
    except ImportError:
        pass


_lib = None


def get_library():
    """Load (once) and return the cuDSS shared library with configured argtypes."""
    global _lib
    if _lib is not None:
        return _lib

    last_error = None
    lib = None
    for path in _candidate_paths():
        try:
            lib = ctypes.CDLL(path)
            break
        except OSError as e:
            last_error = e
            continue

    if lib is None:
        raise OSError(
            "Could not locate libcudss.so.0. Set the CUDSS_LIBRARY_PATH environment "
            "variable to the full path of the shared library, install the "
            "'nvidia-cudss-cu12' pip package, or add the cuDSS lib directory to "
            f"LD_LIBRARY_PATH. Last error: {last_error}"
        )

    _configure_argtypes(lib)
    _lib = lib
    return lib


def _configure_argtypes(lib):
    c_int = ctypes.c_int
    c_int64 = ctypes.c_int64
    c_size_t = ctypes.c_size_t
    c_void_p = ctypes.c_void_p
    c_char_p = ctypes.c_char_p
    POINTER = ctypes.POINTER

    lib.cudssCreate.argtypes = [POINTER(cudssHandle_t)]
    lib.cudssCreate.restype = cudssStatus_t

    lib.cudssDestroy.argtypes = [cudssHandle_t]
    lib.cudssDestroy.restype = cudssStatus_t

    lib.cudssConfigCreate.argtypes = [POINTER(cudssConfig_t)]
    lib.cudssConfigCreate.restype = cudssStatus_t

    lib.cudssConfigDestroy.argtypes = [cudssConfig_t]
    lib.cudssConfigDestroy.restype = cudssStatus_t

    lib.cudssDataCreate.argtypes = [cudssHandle_t, POINTER(cudssData_t)]
    lib.cudssDataCreate.restype = cudssStatus_t

    lib.cudssDataDestroy.argtypes = [cudssHandle_t, cudssData_t]
    lib.cudssDataDestroy.restype = cudssStatus_t

    lib.cudssConfigSet.argtypes = [cudssConfig_t, cudssConfigParam_t, c_void_p, c_size_t]
    lib.cudssConfigSet.restype = cudssStatus_t

    lib.cudssConfigGet.argtypes = [cudssConfig_t, cudssConfigParam_t, c_void_p, c_size_t, POINTER(c_size_t)]
    lib.cudssConfigGet.restype = cudssStatus_t

    lib.cudssSetStream.argtypes = [cudssHandle_t, c_void_p]
    lib.cudssSetStream.restype = cudssStatus_t

    lib.cudssExecute.argtypes = [
        cudssHandle_t,
        c_int,
        cudssConfig_t,
        cudssData_t,
        cudssMatrix_t,
        cudssMatrix_t,
        cudssMatrix_t,
    ]
    lib.cudssExecute.restype = cudssStatus_t

    lib.cudssMatrixCreateCsr.argtypes = [
        POINTER(cudssMatrix_t),
        c_int64,  # nrows
        c_int64,  # ncols
        c_int64,  # nnz
        c_void_p,  # rowStart
        c_void_p,  # rowEnd
        c_void_p,  # colIndices
        c_void_p,  # values
        cudssDataType_t,  # offsetType
        cudssDataType_t,  # indexType
        cudssDataType_t,  # valueType
        cudssMatrixType_t,
        cudssMatrixViewType_t,
        cudssIndexBase_t,
    ]
    lib.cudssMatrixCreateCsr.restype = cudssStatus_t

    lib.cudssMatrixCreateDn.argtypes = [
        POINTER(cudssMatrix_t),
        c_int64,
        c_int64,
        c_int64,
        c_void_p,
        cudssDataType_t,
        cudssLayout_t,
    ]
    lib.cudssMatrixCreateDn.restype = cudssStatus_t

    lib.cudssMatrixDestroy.argtypes = [cudssMatrix_t]
    lib.cudssMatrixDestroy.restype = cudssStatus_t

    lib.cudssMatrixSetValues.argtypes = [cudssMatrix_t, c_void_p]
    lib.cudssMatrixSetValues.restype = cudssStatus_t

    lib.cudssMatrixSetCsrPointers.argtypes = [cudssMatrix_t, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.cudssMatrixSetCsrPointers.restype = cudssStatus_t

    if hasattr(lib, "cudssGetProperty"):
        lib.cudssGetProperty.argtypes = [c_int, POINTER(c_int)]
        lib.cudssGetProperty.restype = cudssStatus_t
