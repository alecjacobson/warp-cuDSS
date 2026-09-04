"""Expand a Warp BSR matrix with block size > 1 into a scalar CSR matrix.

cuDSS consumes scalar CSR. Warp assembles many operators (e.g. vector-valued FEM
elasticity systems) as block CSR/BSR with small dense blocks such as ``wp.mat33d``.
This module builds the scalar CSR sparsity once (when the block sparsity pattern
changes) and refreshes only the values array on subsequent updates.

The expansion relies on a documented Warp invariant: within each block row, block
columns are stored in ascending sorted order (this is what :func:`warp.sparse.bsr_from_triplets`
and the FEM assembly routines produce). Only compact matrices (``row_counts is None``)
are supported; call :func:`warp.sparse.bsr_compress` on padded matrices first.
"""

import warp as wp

__all__ = ["ScalarCsrView", "block_shape_is_scalar", "bsr_to_scalar_csr"]


def block_shape_is_scalar(A) -> bool:
    return A.block_shape == (1, 1)


class ScalarCsrView:
    """Scalar CSR arrays (int32 offsets/columns, matching-dtype values) for a BSR matrix.

    For scalar (block-shape ``(1, 1)``) matrices this is a zero-copy view directly onto
    the source matrix's own arrays. For block matrices, ``row_offsets`` and ``columns``
    are built once per sparsity pattern and ``values`` are rebuilt (cheaply, on-device)
    whenever :meth:`refresh_values` is called.
    """

    def __init__(self, A):
        self.nrow = A.shape[0]
        self.ncol = A.shape[1]
        self.device = A.device
        self.scalar_type = A.scalar_type
        self._is_scalar = block_shape_is_scalar(A)
        self._A = A

        if self._is_scalar:
            if A.row_counts is not None:
                raise NotImplementedError(
                    "cuDSS scalar CSR views require a compact BsrMatrix (row_counts=None). "
                    "Call warp.sparse.bsr_compress(A) first."
                )
            self.row_offsets = A.offsets
            self.columns = A.columns
            self.values = A.values
            self.nnz = A.nnz
        else:
            if A.row_counts is not None:
                raise NotImplementedError(
                    "cuDSS BSR->CSR expansion requires a compact BsrMatrix (row_counts=None). "
                    "Call warp.sparse.bsr_compress(A) first."
                )
            self._br, self._bc = A.block_shape
            self._build_sparsity(A)
            self.refresh_values(A)

    def _build_sparsity(self, A):
        br, bc = self._br, self._bc
        nrow_blocks = A.nrow
        device = self.device

        block_row_nnz = wp.empty(nrow_blocks, dtype=wp.int32, device=device)
        wp.launch(_block_row_nnz_kernel, dim=nrow_blocks, inputs=[A.offsets], outputs=[block_row_nnz], device=device)

        n_scalar_rows = nrow_blocks * br
        scalar_row_nnz = wp.empty(n_scalar_rows, dtype=wp.int32, device=device)
        wp.launch(
            _expand_row_nnz_kernel,
            dim=n_scalar_rows,
            inputs=[block_row_nnz, br, bc],
            outputs=[scalar_row_nnz],
            device=device,
        )

        scalar_offsets = wp.zeros(n_scalar_rows + 1, dtype=wp.int32, device=device)
        offsets_tail = wp.empty(n_scalar_rows, dtype=wp.int32, device=device)
        wp.utils.array_scan(scalar_row_nnz, offsets_tail, inclusive=True)
        wp.copy(dest=scalar_offsets, src=offsets_tail, dest_offset=1, count=n_scalar_rows)

        nnz_scalar = int(offsets_tail.numpy()[-1]) if n_scalar_rows > 0 else 0

        self.row_offsets = scalar_offsets
        self.columns = wp.empty(max(nnz_scalar, 1), dtype=wp.int32, device=device)
        self.values = wp.empty(max(nnz_scalar, 1), dtype=self.scalar_type, device=device)
        self.nnz = nnz_scalar

        self._block_rows = A.uncompress_rows()
        self._scalar_offsets = scalar_offsets

        expand_kernel = _get_expand_kernel(br, bc, self.scalar_type)
        wp.launch(
            expand_kernel,
            dim=A.nnz,
            inputs=[self._block_rows, A.columns, A.offsets, scalar_offsets, A.scalar_values],
            outputs=[self.columns, self.values],
            device=device,
        )

    def refresh_values(self, A):
        """Recompute the scalar values array from the current block values. Sparsity pattern is unchanged."""
        if self._is_scalar:
            self.values = A.values
            return

        br, bc = self._br, self._bc
        expand_kernel = _get_expand_kernel(br, bc, self.scalar_type)
        wp.launch(
            expand_kernel,
            dim=A.nnz,
            inputs=[self._block_rows, A.columns, A.offsets, self._scalar_offsets, A.scalar_values],
            outputs=[self.columns, self.values],
            device=self.device,
        )


@wp.kernel
def _block_row_nnz_kernel(offsets: wp.array(dtype=wp.int32), out_row_nnz: wp.array(dtype=wp.int32)):
    row = wp.tid()
    out_row_nnz[row] = offsets[row + 1] - offsets[row]


@wp.kernel
def _expand_row_nnz_kernel(
    block_row_nnz: wp.array(dtype=wp.int32),
    br: wp.int32,
    bc: wp.int32,
    out_scalar_row_nnz: wp.array(dtype=wp.int32),
):
    srow = wp.tid()
    r_b = srow // br
    out_scalar_row_nnz[srow] = block_row_nnz[r_b] * bc


_expand_kernel_cache = {}


def _get_expand_kernel(br: int, bc: int, scalar_type):
    key = (br, bc, scalar_type)
    kernel = _expand_kernel_cache.get(key)
    if kernel is not None:
        return kernel

    def _expand_fn(
        block_rows: wp.array(dtype=wp.int32),
        block_columns: wp.array(dtype=wp.int32),
        block_offsets: wp.array(dtype=wp.int32),
        scalar_row_offsets: wp.array(dtype=wp.int32),
        scalar_block_values: wp.array(dtype=scalar_type, ndim=3),
        out_columns: wp.array(dtype=wp.int32),
        out_values: wp.array(dtype=scalar_type),
    ):
        block = wp.tid()
        row = block_rows[block]
        col = block_columns[block]
        k = block - block_offsets[row]
        for li in range(br):
            srow = row * br + li
            base = scalar_row_offsets[srow] + k * bc
            for lj in range(bc):
                dest = base + lj
                out_columns[dest] = col * bc + lj
                out_values[dest] = scalar_block_values[block, li, lj]

    kernel = wp.Kernel(func=_expand_fn, key=f"warp_cudss_expand_{br}x{bc}_{scalar_type.__name__}")
    _expand_kernel_cache[key] = kernel
    return kernel


def bsr_to_scalar_csr(A) -> ScalarCsrView:
    """Build (or, for scalar matrices, alias) a :class:`ScalarCsrView` for BSR matrix ``A``."""
    return ScalarCsrView(A)
