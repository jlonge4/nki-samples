"""
Batch-Invariance Test Kit
=========================

Test kit for row-independent ops F : (M, K) -> (M, N).

Predicates (see docs/part_a/METHOD.md):

  (1) Shape / schedule invariance:
        F(X)[i] doesn't depend on M.
  (2) Row-position invariance:
        F(X)[i] doesn't depend on i.
  (3) Neighbor-value isolation:
        F(X)[i] doesn't depend on X[j] for j != i.

The predicates are independent. The default battery covers all three.

Primary tests
-------------

  test_whole_block(op, K, ...)
      Primary test for (1). For each (M_small, M_big) pair, builds X_big
      whose first M_small rows equal X_small, runs op on both, and asserts
      every one of the M_small overlapping output rows is bitwise equal.
      For partial (3) coverage, runs multiple *filler* patterns on the tail
      rows — zeros, same distribution, adversarial — and asserts the prefix
      output is identical under every filler choice.

  test_position(op, K, ...)
      Primary test for (2). Builds X of shape (M, K) with a probe row at 0,
      swaps it into each p in `positions`, runs op, asserts out[p] bitwise
      identical across positions. Runs across multiple M values (not one
      fixed M) because the same p can hit different epilogue paths under
      different M.

  test_neighbor_mutation(op, K, ...)
      Primary test for (3). At fixed (M, p), holds row p constant and
      mutates all *other* rows across multiple distributions/seeds.
      Asserts out[p] is bitwise identical across mutations.

Input specs
-----------
Default battery:
  randn@1 / @10 / @100           — cover common activation scales
  sparse@5,10%                   — heavy-tailed (post-activation-like)

Adversarial specs (use `DEFAULT_INPUTS + ADVERSARIAL_INPUTS` for shipping):
  canceling                      — alternating +large / -large + residuals
  two_scale                      — one large element + many small
  tile_boundary                  — nonzeros only near H tile boundaries

Each (shape, distribution) runs across multiple seeds by default (>=3) to
guard against lucky-seed false passes.

Op contract
-----------
  op(X: torch.Tensor bf16 on CPU) -> torch.Tensor bf16 on CPU

The op closure must wrap the *production* dispatch path (same bucketing,
compilation, kernel selection, dtype conversions). For async / XLA-style
devices, synchronize before returning.

Bitwise comparison
------------------
Pass/fail is bitwise equality on the underlying dtype representation:
  torch.equal(
      out.detach().cpu().contiguous().view(torch.int16),   # bf16/fp16
      ref.detach().cpu().contiguous().view(torch.int16),
  )
fp32 uses int32. ULP is computed from an ordered integer representation
and reports adjacent-representable distance, not bit-pattern xor count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

# ============================================================
# metrics
# ============================================================


def _uint_view(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().contiguous()
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.view(torch.int16)
    if x.dtype == torch.float32:
        return x.view(torch.int32)
    if x.dtype == torch.float64:
        return x.view(torch.int64)
    raise TypeError(f"unsupported dtype for bitwise compare: {x.dtype}")


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(_uint_view(a), _uint_view(b))


def bitwise_equal_ndarray(a, b) -> bool:
    """Bitwise bf16/fp compare for numpy arrays (same contract as ``bitwise_equal``)."""
    import numpy as np

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a.view(np.uint16), b.view(np.uint16)))


def _ordered_int(x: torch.Tensor) -> torch.Tensor:
    """Map IEEE-754 sign-magnitude bits to ordered two's-complement-like ints,
    so that adjacent representable floats differ by exactly 1 and the ordering
    matches float ordering. Required for correct ULP across sign changes.
    """
    nbits = {
        torch.bfloat16: 16,
        torch.float16: 16,
        torch.float32: 32,
        torch.float64: 64,
    }[x.dtype]
    sign_bit = 1 << (nbits - 1)
    umask = (1 << nbits) - 1
    u = _uint_view(x).to(torch.int64) & umask
    # If negative (sign bit set): ordered = sign_bit - u  (u here includes sign bit,
    # so sign_bit - u flips the magnitude ordering and gives a negative int).
    # If non-negative: ordered = u (already ordered above -sign_bit).
    # Both branches fall on an axis where sign=0, u=0 maps to 0, and -0 (u=sign_bit)
    # also maps to 0.
    neg = (u & sign_bit) != 0
    ordered = torch.where(neg, sign_bit - u, u)
    return ordered


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Adjacent-representable ULP distance, correct across sign changes."""
    return (_ordered_int(a) - _ordered_int(b)).abs()


# ============================================================
# input specs
# ============================================================


@dataclass
class InputSpec:
    name: str
    generate: Callable[[tuple, int], torch.Tensor]


def _randn(scale: float) -> Callable:
    def gen(shape, seed):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(torch.bfloat16)

    return gen


def _zeros() -> Callable:
    def gen(shape, seed):
        return torch.zeros(*shape, dtype=torch.bfloat16)

    return gen


def _sparse_heavy(scale: float, frac: float) -> Callable:
    def gen(shape, seed):
        g_val = torch.Generator().manual_seed(seed)
        g_mask = torch.Generator().manual_seed(seed + 777)
        n = 1
        for s in shape:
            n *= s
        vals = torch.randn(n, generator=g_val, dtype=torch.float32) * scale
        mask = (torch.rand(n, generator=g_mask) < frac).to(torch.float32)
        return (vals * mask).reshape(shape).to(torch.bfloat16)

    return gen


def _alternating(mag: float, residual: float) -> Callable:
    """Reduction-order adversarial: row = [+mag, -mag, +mag, -mag, ..., +resid, +resid, ...]"""

    def gen(shape, seed):
        *_, K = shape
        # Build one K-vec then broadcast with per-row random residual offsets.
        g = torch.Generator().manual_seed(seed)
        base = torch.empty(K, dtype=torch.float32)
        sign = torch.ones(K)
        sign[1::2] = -1.0
        base = sign * mag
        base[::2] += residual * (torch.rand(K // 2 + K % 2, generator=g) - 0.5)
        base[1::2] += residual * (torch.rand(K // 2, generator=g) - 0.5)
        # broadcast over rows
        out = base.unsqueeze(0).expand(*shape).contiguous().clone()
        return out.to(torch.bfloat16)

    return gen


def _two_scale(big: float, small: float) -> Callable:
    """One large element + many small elements, position randomized per row."""

    def gen(shape, seed):
        g = torch.Generator().manual_seed(seed)
        *prefix, K = shape
        M = 1
        for s in prefix:
            M *= s
        x = torch.randn(M, K, generator=g, dtype=torch.float32) * small
        # place `big` at a random column per row
        cols = torch.randint(0, K, (M,), generator=g)
        rows = torch.arange(M)
        x[rows, cols] = big
        return x.reshape(*shape).to(torch.bfloat16)

    return gen


def _tile_boundary(boundaries: tuple[int, ...], mag: float) -> Callable:
    """Non-zeros only at specific H columns (±1 around each boundary)."""

    def gen(shape, seed):
        g = torch.Generator().manual_seed(seed)
        *prefix, K = shape
        M = 1
        for s in prefix:
            M *= s
        x = torch.zeros(M, K, dtype=torch.float32)
        cols = []
        for b in boundaries:
            for dc in (-1, 0, 1):
                c = b + dc
                if 0 <= c < K:
                    cols.append(c)
        if cols:
            cols_t = torch.tensor(cols, dtype=torch.long)
            x[:, cols_t] = torch.randn((M, len(cols)), generator=g, dtype=torch.float32) * mag
        return x.reshape(*shape).to(torch.bfloat16)

    return gen


DEFAULT_INPUTS: list[InputSpec] = [
    InputSpec("randn@1", _randn(1.0)),
    InputSpec("randn@10", _randn(10.0)),
    InputSpec("randn@100", _randn(100.0)),
    InputSpec("sparse@5,10%", _sparse_heavy(scale=5.0, frac=0.1)),
]

ADVERSARIAL_INPUTS: list[InputSpec] = [
    InputSpec("alternating", _alternating(mag=1e3, residual=1.0)),
    InputSpec("two_scale", _two_scale(big=1e4, small=1.0)),
    InputSpec(
        "tile_boundary",
        _tile_boundary(boundaries=(63, 64, 127, 128, 255, 256), mag=10.0),
    ),
]

# Filler patterns for whole-block tail (neighbor-value isolation coverage).
DEFAULT_FILLERS: list[InputSpec] = [
    InputSpec("zeros", _zeros()),
    InputSpec("randn@1", _randn(1.0)),
    InputSpec("sparse@5,10%", _sparse_heavy(scale=5.0, frac=0.1)),
]

DEFAULT_WB_PAIRS: list[tuple[int, int]] = [
    (1, 2),
    (1, 128),
    (1, 256),
    (2, 128),
    (127, 128),
    (127, 256),
    (128, 256),
    (128, 2048),
    (255, 256),
    (255, 2048),
    (1024, 2048),
    (2048, 8192),
]

# Multi-M position sweep. Includes aligned, one-below, one-above boundary cases.
DEFAULT_POSITION_M_VALUES: tuple[int, ...] = (
    64,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    2047,
    2048,
    2049,
)

DEFAULT_POSITIONS_M256 = [0, 1, 63, 64, 127, 128, 129, 191, 255]
DEFAULT_POSITIONS_M2048 = [0, 1, 127, 128, 255, 256, 511, 512, 1023, 1024, 2047]


def _positions_for(M: int) -> list[int]:
    """Positions straddling boundaries for a given M. Filtered to [0, M)."""
    candidates = {0, 1}
    for b in (63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024):
        candidates.add(b)
    candidates.add(M // 2)
    candidates.add(M - 1)
    if M >= 2:
        candidates.add(M - 2)
    return sorted(p for p in candidates if 0 <= p < M)


DEFAULT_NEIGHBOR_CONFIGS: tuple[tuple[int, int], ...] = (
    # (M, p)
    (128, 0),
    (128, 64),
    (128, 127),
    (256, 0),
    (256, 128),
    (256, 255),
    (255, 0),
    (255, 128),
    (255, 254),  # tail-slab M
    (2048, 1024),
)

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)


# ============================================================
# result type
# ============================================================


@dataclass
class TestResult:
    test: str  # "whole_block" / "position" / "neighbor"
    config: str
    passed: bool
    n_rows: int
    n_bad_rows: int
    max_ulp: int
    max_abs: float

    def oneline(self) -> str:
        if self.passed:
            return f"  {self.test:<14} PASS  ({self.n_rows:>5} rows)  {self.config}"
        return (
            f"  {self.test:<14} FAIL  bad={self.n_bad_rows}/{self.n_rows} "
            f"max_ulp={self.max_ulp} max_abs={self.max_abs:.2e}  {self.config}"
        )


def _compare_rows(test: str, config: str, A: torch.Tensor, B: torch.Tensor) -> TestResult:
    assert A.shape == B.shape, f"shape mismatch {A.shape} vs {B.shape}"
    ulp = ulp_diff(A, B)
    per_row = ulp.max(dim=-1).values if ulp.dim() >= 2 else ulp.clone()
    n_bad = int((per_row > 0).sum().item())
    passed = n_bad == 0
    max_abs = (A.float() - B.float()).abs().max().item()
    return TestResult(
        test=test,
        config=config,
        passed=passed,
        n_rows=int(A.shape[0]),
        n_bad_rows=n_bad,
        max_ulp=int(ulp.max().item()),
        max_abs=max_abs,
    )


# ============================================================
# Test 1: whole-block (schedule invariance + partial neighbor isolation)
# ============================================================


def test_whole_block(
    op: Callable[[torch.Tensor], torch.Tensor],
    K: int,
    pairs: Optional[list[tuple[int, int]]] = None,
    inputs: Optional[list[InputSpec]] = None,
    fillers: Optional[list[InputSpec]] = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[TestResult]:
    """Primary test for predicate (1) + partial predicate (3).

    For each (M_small, M_big) in `pairs`, each prefix distribution in `inputs`,
    each filler distribution in `fillers`, and each seed:
        X_small = inputs_spec.generate((M_small, K), seed)
        tail    = filler_spec.generate((M_big - M_small, K), seed + 1000)
        X_big   = cat([X_small, tail])
        Assert op(X_small) == op(X_big)[:M_small] bitwise, every row.

    The assertion must hold for *every* filler, making this simultaneously a
    schedule-invariance test (varying M) and a partial neighbor-isolation test
    (varying the values surrounding the prefix).
    """
    pairs = pairs or DEFAULT_WB_PAIRS
    inputs = inputs or DEFAULT_INPUTS
    fillers = fillers or DEFAULT_FILLERS

    results = []
    for m_s, m_b in pairs:
        for in_spec in inputs:
            for fill_spec in fillers:
                for seed in seeds:
                    X_small = in_spec.generate((m_s, K), seed=seed)
                    tail = fill_spec.generate((m_b - m_s, K), seed=seed + 1000)
                    X_big = torch.cat([X_small, tail], dim=0)

                    Y_small = op(X_small).detach().cpu().contiguous()
                    Y_big = op(X_big).detach().cpu().contiguous()

                    config = (
                        f"({m_s:>5},{m_b:<6}) in={in_spec.name:<14} "
                        f"fill={fill_spec.name:<14} seed={seed}"
                    )
                    results.append(_compare_rows("whole_block", config, Y_small, Y_big[:m_s]))
    return results


# ============================================================
# Test 2: position (row-index independence)
# ============================================================


def test_position(
    op: Callable[[torch.Tensor], torch.Tensor],
    K: int,
    M_values: tuple[int, ...] = DEFAULT_POSITION_M_VALUES,
    positions_fn: Callable[[int], list[int]] = _positions_for,
    inputs: Optional[list[InputSpec]] = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[TestResult]:
    """Primary test for predicate (2).

    For each M in `M_values`, each distribution, each seed:
        Build X of shape (M, K). probe = X[0].
        For each p in positions: swap row 0 and row p, run op, collect out[p].
        Assert all collected rows are bitwise identical.

    Runs across multiple M values so that the same position can be exercised
    under different tail/epilogue contexts.
    """
    inputs = inputs or DEFAULT_INPUTS

    results = []
    for M in M_values:
        positions = [p for p in positions_fn(M) if 0 <= p < M]
        if len(positions) < 2:
            continue
        for spec in inputs:
            for seed in seeds:
                X_base = spec.generate((M, K), seed=seed)
                rows = []
                for p in positions:
                    X_p = X_base.clone()
                    if p != 0:
                        tmp = X_p[0].clone()
                        X_p[0] = X_p[p]
                        X_p[p] = tmp
                    Y = op(X_p).detach().cpu().contiguous()
                    rows.append(Y[p].clone())
                stacked = torch.stack(rows)
                ref = stacked[0:1].expand_as(stacked).contiguous()

                config = f"M={M:<5} n_pos={len(positions):>2} in={spec.name:<14} seed={seed}"
                results.append(_compare_rows("position", config, stacked, ref))
    return results


# ============================================================
# Test 3: neighbor mutation (neighbor-value isolation)
# ============================================================


def test_neighbor_mutation(
    op: Callable[[torch.Tensor], torch.Tensor],
    K: int,
    configs: tuple[tuple[int, int], ...] = DEFAULT_NEIGHBOR_CONFIGS,
    neighbor_specs: Optional[list[InputSpec]] = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
) -> list[TestResult]:
    """Primary test for predicate (3).

    For each (M, p) in `configs`:
        Fix a probe row x_probe.
        For each neighbor_spec in `neighbor_specs`, for each seed:
            Build X of shape (M, K) from neighbor_spec, with row p = x_probe.
            Run op, collect out[p].
        Assert all collected out[p] are bitwise identical.
    """
    neighbor_specs = neighbor_specs or DEFAULT_FILLERS

    # Fixed probe row across the test (derived from randn@1, seed=-1).
    probe = _randn(1.0)((1, K), seed=-1).squeeze(0)

    results = []
    for M, p in configs:
        if p >= M:
            continue
        rows = []
        labels = []
        for spec in neighbor_specs:
            for seed in seeds:
                X = spec.generate((M, K), seed=seed)
                X[p] = probe
                Y = op(X).detach().cpu().contiguous()
                rows.append(Y[p].clone())
                labels.append(f"{spec.name}:{seed}")
        if len(rows) < 2:
            continue
        stacked = torch.stack(rows)
        ref = stacked[0:1].expand_as(stacked).contiguous()

        config = f"M={M:<5} p={p:<5} n_mut={len(rows):>2}"
        results.append(_compare_rows("neighbor_mut", config, stacked, ref))
    return results


# ============================================================
# top-level battery
# ============================================================


def run_battery(
    op: Callable[[torch.Tensor], torch.Tensor],
    K: int,
    # whole-block
    whole_block_pairs: Optional[list[tuple[int, int]]] = None,
    # position
    position_M_values: tuple[int, ...] = DEFAULT_POSITION_M_VALUES,
    # neighbor mutation
    neighbor_configs: tuple[tuple[int, int], ...] = DEFAULT_NEIGHBOR_CONFIGS,
    # common
    inputs: Optional[list[InputSpec]] = None,
    fillers: Optional[list[InputSpec]] = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    include_adversarial: bool = False,
) -> dict[str, list[TestResult]]:
    """Run the three primary tests. Returns {test_name: [TestResult, ...]}.

    Set include_adversarial=True to append ADVERSARIAL_INPUTS to the input
    distributions (recommended for shipping validation, off by default to
    keep the smoke battery fast).
    """
    inputs = list(inputs or DEFAULT_INPUTS)
    if include_adversarial:
        inputs = inputs + ADVERSARIAL_INPUTS

    results: dict[str, list[TestResult]] = {}
    results["whole_block"] = test_whole_block(
        op,
        K,
        pairs=whole_block_pairs,
        inputs=inputs,
        fillers=fillers,
        seeds=seeds,
    )
    results["position"] = test_position(
        op,
        K,
        M_values=position_M_values,
        inputs=inputs,
        seeds=seeds,
    )
    results["neighbor_mut"] = test_neighbor_mutation(
        op,
        K,
        configs=neighbor_configs,
        neighbor_specs=fillers,
        seeds=seeds,
    )
    return results


def print_report(
    results: dict[str, list[TestResult]] | list[TestResult],
    show_passes: bool = False,
) -> None:
    """Summary + all failures; passes only if show_passes=True."""
    if isinstance(results, dict):
        flat = []
        for rs in results.values():
            flat.extend(rs)
        by_test = results
    else:
        flat = results
        by_test = {}
        for r in flat:
            by_test.setdefault(r.test, []).append(r)

    print("=" * 80)
    for test, rs in by_test.items():
        n_pass = sum(1 for r in rs if r.passed)
        n = len(rs)
        status = "PASS" if n_pass == n else "FAIL"
        print(f"  [{status}] {test:<14} {n_pass}/{n} cases bitwise invariant")
    print("=" * 80)

    for r in flat:
        if r.passed and not show_passes:
            continue
        print(r.oneline())


# ============================================================
# bridges for non-torch kernels
# ============================================================


def numpy_bf16_op(fn: Callable) -> Callable[[torch.Tensor], torch.Tensor]:
    """Adapter: wrap a numpy-bfloat16 op into the torch-CPU contract the kit expects.

    fn signature: (X: np.ndarray of ml_dtypes.bfloat16) -> np.ndarray of bf16.
    """
    import numpy as np

    try:
        import ml_dtypes
    except ImportError as e:
        raise ImportError("numpy_bf16_op requires ml_dtypes for bf16") from e

    def wrapped(X: torch.Tensor) -> torch.Tensor:
        assert X.dtype == torch.bfloat16
        x_np = X.contiguous().view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        y_np = fn(x_np)
        assert y_np.dtype == ml_dtypes.bfloat16
        y_t = torch.from_numpy(y_np.view(np.int16).copy()).view(torch.bfloat16)
        return y_t

    return wrapped
