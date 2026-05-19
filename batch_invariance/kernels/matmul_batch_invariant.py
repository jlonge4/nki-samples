"""
Batch-Invariant MatMul Kernel

This kernel demonstrates batch invariance in matrix multiplication by controlling
the K-dimension tiling strategy.

NKI version: 0.3.0 (Beta 3)
"""

import nki
import nki.isa as nisa
import nki.language as nl


def _matmul_m_slab(a, b, result, m_start, rows, n_start, K, K_TILE, N_TILE):
    """Accumulate one M slab (rows live partitions) for one N tile."""
    c_psum = nl.zeros((rows, N_TILE), dtype=nl.float32, buffer=nl.psum)

    for k in nl.affine_range(K // K_TILE):
        a_start = k * K_TILE
        a_end = min(K, a_start + K_TILE)
        m_end = m_start + rows

        a_tile = nl.ndarray((K_TILE, rows), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[a_start:a_end, m_start:m_end])

        b_start = k * K_TILE
        b_end = min(K, b_start + K_TILE)
        b_tile = nl.ndarray((K_TILE, N_TILE), dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[b_start:b_end, n_start : n_start + N_TILE])

        nisa.nc_matmul(dst=c_psum, stationary=a_tile, moving=b_tile)

    c_sbuf = nl.ndarray((rows, N_TILE), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=c_sbuf, src=c_psum)

    nisa.dma_copy(
        dst=result[m_start : m_start + rows, n_start : n_start + N_TILE],
        src=c_sbuf,
    )


@nki.jit
def nki_matmul_kernel_isa(a, b, deterministic=True):
    """
    Matrix multiplication with batch invariance parameter.

    Args:
        a: Input matrix of shape [K, M]
        b: Input matrix of shape [K, N]
        deterministic: If True, uses fixed K_TILE=128 regardless of K size,
                       producing identical results across different batch sizes.
                       If False, uses K_TILE=64 (more accumulations, different rounding).

    Returns:
        result: Output matrix of shape [M, N], same dtype as inputs

    Notes:
        PSUM always accumulates in float32 regardless of input dtype.
        The ONLY difference between modes is K_TILE size. Different K_TILE sizes
        change the number and order of float32 accumulations in PSUM, which can
        produce slightly different results due to non-associativity of FP arithmetic.
        With bfloat16 inputs this difference vanishes (invariant); with float32 it does not.

        M is processed in slabs of up to 128 rows; a tail slab uses the same
        inner math with fewer live partitions (no zero-padding of batch rows).
    """
    K, M = a.shape
    N = b.shape[1]
    M_TILE = 128
    # N_TILE <= 512: nc_matmul moving free-dimension limit on NeuronCore v3 (trn2)
    N_TILE = 128

    if deterministic:
        K_TILE = min(128, K)
    else:
        K_TILE = min(64, K)

    assert K % K_TILE == 0, f"K={K} must be divisible by K_TILE={K_TILE}"
    assert N % N_TILE == 0, f"N={N} must be divisible by N_TILE={N_TILE}"

    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    num_full = M // M_TILE
    tail = M - num_full * M_TILE

    for n in nl.affine_range(N // N_TILE):
        n_start = n * N_TILE

        for m in nl.affine_range(num_full):
            m_start = m * M_TILE
            _matmul_m_slab(
                a,
                b,
                result,
                m_start=m_start,
                rows=M_TILE,
                n_start=n_start,
                K=K,
                K_TILE=K_TILE,
                N_TILE=N_TILE,
            )

        if tail > 0:
            m_start = num_full * M_TILE
            _matmul_m_slab(
                a,
                b,
                result,
                m_start=m_start,
                rows=tail,
                n_start=n_start,
                K=K,
                K_TILE=K_TILE,
                N_TILE=N_TILE,
            )

    return result
