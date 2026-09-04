import numpy as np
import warp as wp
import warp.sparse as sparse


def random_sparse_spd(rng, n, density=0.1, dtype=wp.float64, device="cuda:0"):
    """Random sparse SPD matrix as a scalar BsrMatrix, plus its dense numpy reference."""
    dense = rng.standard_normal((n, n))
    mask = rng.random((n, n)) < density
    mask = mask | mask.T
    dense = dense * mask
    dense = 0.5 * (dense + dense.T) + n * np.eye(n)

    rows, cols = np.nonzero(dense)
    vals = dense[rows, cols]

    rows_w = wp.array(rows.astype(np.int32), device=device)
    cols_w = wp.array(cols.astype(np.int32), device=device)
    vals_w = wp.array(vals, dtype=dtype, device=device)
    A = sparse.bsr_from_triplets(n, n, rows_w, cols_w, vals_w)
    return A, dense


def random_sparse_general(rng, n, density=0.1, dtype=wp.float64, device="cuda:0"):
    """Random sparse (non-symmetric, diagonally dominant so it's nonsingular) scalar BsrMatrix."""
    dense = rng.standard_normal((n, n))
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, True)
    dense = dense * mask
    row_sums = np.sum(np.abs(dense), axis=1)
    dense[np.arange(n), np.arange(n)] = row_sums + n

    rows, cols = np.nonzero(dense)
    vals = dense[rows, cols]

    rows_w = wp.array(rows.astype(np.int32), device=device)
    cols_w = wp.array(cols.astype(np.int32), device=device)
    vals_w = wp.array(vals, dtype=dtype, device=device)
    A = sparse.bsr_from_triplets(n, n, rows_w, cols_w, vals_w)
    return A, dense


def random_block_spd(rng, nb, br, density=0.2, dtype=wp.mat33d, device="cuda:0"):
    """Random block-sparse SPD matrix (dense blocks of shape (br, br)) as a BsrMatrix."""
    n = nb * br
    dense = rng.standard_normal((n, n))

    pattern = rng.random((nb, nb)) < density
    np.fill_diagonal(pattern, True)
    pattern = pattern | pattern.T
    mask = np.kron(pattern, np.ones((br, br), dtype=bool))
    dense = dense * mask
    dense = 0.5 * (dense + dense.T) + n * np.eye(n)

    triplet_rows, triplet_cols, triplet_vals = [], [], []
    block_dtype = dtype
    for i in range(nb):
        for j in range(nb):
            if pattern[i, j]:
                triplet_rows.append(i)
                triplet_cols.append(j)
                block = dense[i * br : (i + 1) * br, j * br : (j + 1) * br]
                triplet_vals.append(block_dtype(*block.flatten().tolist()))

    rows_w = wp.array(np.array(triplet_rows, dtype=np.int32), device=device)
    cols_w = wp.array(np.array(triplet_cols, dtype=np.int32), device=device)
    vals_w = wp.array(triplet_vals, dtype=block_dtype, device=device)
    A = sparse.bsr_from_triplets(nb, nb, rows_w, cols_w, vals_w)
    return A, dense
