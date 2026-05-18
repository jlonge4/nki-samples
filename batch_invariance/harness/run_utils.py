"""Bootstrap paths and Neuron detection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def require_neuron() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: torch not available")
        return False
    from harness.neuron_device import detect_backend, init_nki_runtime
    from harness.serving_adapters import has_neuron

    if not has_neuron():
        print("SKIP: AWS Neuron SDK (nki) not available")
        return False
    init_nki_runtime()
    backend = detect_backend()
    if backend is None:
        print("SKIP: TorchNeuron (torch_neuronx) or torch_xla required for device tests")
        return False
    return True


def count_failures(results) -> int:
    if isinstance(results, dict):
        flat = []
        for rs in results.values():
            flat.extend(rs)
    else:
        flat = list(results)
    return sum(1 for r in flat if not r.passed)
