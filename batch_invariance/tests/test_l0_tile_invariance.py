"""
L0 — Tile invariance (det vs nondet) via test_tile_invariance.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    script = ROOT / "test_tile_invariance.py"
    if not script.exists():
        print(f"SKIP: {script} not found")
        return 0

    sys.path.insert(0, str(ROOT))
    try:
        import torch
        from harness.neuron_device import detect_backend, init_nki_runtime

        init_nki_runtime()
        if detect_backend() is None:
            print("SKIP: TorchNeuron (torch_neuronx) or torch_xla required for L0 tile tests")
            return 0
    except ImportError:
        print("SKIP: torch not available")
        return 0

    print("=== L0: tile invariance (test_tile_invariance.py) ===\n")
    r = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
