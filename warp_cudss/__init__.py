"""cuDSS-backed sparse direct solver helpers for NVIDIA Warp.

Drop-in replacement for the iterative solvers in :mod:`warp.optim.linear`
(``cg``, ``cr``, ``bicgstab``, ``gmres``) when a direct factorization is preferable,
while keeping the whole solve on-device and CUDA-graph capturable.
"""

from ._bindings import CudssError
from ._csr import ScalarCsrView, bsr_to_scalar_csr
from .solver import CudssSolver, solve

__all__ = [
    "CudssSolver",
    "CudssError",
    "ScalarCsrView",
    "bsr_to_scalar_csr",
    "solve",
]

__version__ = "0.1.0"
