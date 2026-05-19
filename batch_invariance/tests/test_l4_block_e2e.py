"""
L4 — Transformer block: tile invariance, prefix packing, run-to-run (Trainium).

Demo dims (d_model=256); nanochat per-op at H=1280 is L1.
"""

from __future__ import annotations

import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.nanochat_shapes import DEMO_D_FFN, DEMO_D_MODEL, DEMO_SEQ
from harness.run_utils import require_neuron


def main() -> int:
    if not require_neuron():
        return 0

    import torch

    from harness.neuron_device import (
        get_device,
        init_nki_runtime,
        sync_device,
        to_neuron,
    )
    from transformer_block import make_block_weights, nki_transformer_block

    init_nki_runtime()

    def linspace_2d(rows: int, cols: int, dtype=torch.bfloat16):
        return torch.linspace(-0.5, 0.5, rows * cols, dtype=dtype).reshape(rows, cols)

    def test_block_tile_invariance() -> bool:
        print(f"\n--- L4a: block det vs nondet (bf16) device={get_device()} ---")
        seq, d_model, d_ffn = DEMO_SEQ, DEMO_D_MODEL, DEMO_D_FFN
        torch.manual_seed(42)
        weights = {k: to_neuron(v) for k, v in make_block_weights(d_model, 128, d_ffn).items()}
        x = to_neuron(linspace_2d(seq, d_model))
        out_det = nki_transformer_block(x, weights, deterministic=True)
        sync_device()
        out_nondet = nki_transformer_block(x, weights, deterministic=False)
        sync_device()
        diff = (out_det.cpu().float() - out_nondet.cpu().float()).abs().max().item()
        ok = diff == 0.0 and not out_det.isnan().any()
        print(f"  max_abs={diff:.2e}: {'PASS' if ok else 'FAIL'}")
        return bool(ok)

    def test_block_prefix_packing() -> bool:
        print("\n--- L4b: block prefix invariance (co-packed seq) ---")
        seq, d_model, d_ffn = DEMO_SEQ, DEMO_D_MODEL, DEMO_D_FFN
        torch.manual_seed(43)
        weights = {k: to_neuron(v) for k, v in make_block_weights(d_model, 128, d_ffn).items()}
        x = to_neuron(linspace_2d(seq, d_model))
        filler = to_neuron(linspace_2d(seq, d_model) * 0.1)
        x_big = torch.cat([x, filler], dim=0)
        y_small = nki_transformer_block(x, weights, deterministic=True)
        sync_device()
        y_big = nki_transformer_block(x_big, weights, deterministic=True)
        sync_device()
        ok = torch.equal(y_small.cpu().view(torch.int16), y_big[:seq].cpu().view(torch.int16))
        diff = (y_small.cpu().float() - y_big[:seq].cpu().float()).abs().max().item()
        print(f"  seq={seq}: {'PASS' if ok else 'FAIL'} max_abs={diff:.2e}")
        return bool(ok)

    def test_block_run_to_run(n_runs: int = 3) -> bool:
        print("\n--- L4c: block run-to-run ---")
        torch.manual_seed(44)
        weights = {
            k: to_neuron(v) for k, v in make_block_weights(DEMO_D_MODEL, 128, DEMO_D_FFN).items()
        }
        x = to_neuron(linspace_2d(DEMO_SEQ, DEMO_D_MODEL))
        ref = None
        for i in range(n_runs):
            y = nki_transformer_block(x, weights, deterministic=True)
            sync_device()
            y_cpu = y.cpu().view(torch.int16)
            if ref is None:
                ref = y_cpu
            elif not torch.equal(ref, y_cpu):
                print(f"  run {i}: FAIL")
                return False
        print(f"  {n_runs} runs: PASS")
        return True

    ok = test_block_tile_invariance() and test_block_prefix_packing() and test_block_run_to_run()
    print(f"\nL4 overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
