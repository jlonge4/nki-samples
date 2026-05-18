"""
Trainium device helpers — TorchNeuron (PyTorch Native) first, torch_xla fallback.

See: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html

- TorchNeuron: ``device = torch.device("neuron")``, ``import torch_neuronx`` when calling NKI.
- Legacy: ``torch_xla.core.xla_model.xla_device()`` (still supported as fallback).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

Backend = Literal["torch_neuronx", "torch_xla"]
_BACKEND: Optional[Backend] = None


def detect_backend() -> Optional[Backend]:
    """Detect available Trainium PyTorch backend."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import torch_neuronx  # noqa: F401

        _BACKEND = "torch_neuronx"
        return _BACKEND
    except ImportError:
        pass
    try:
        import torch_xla.core.xla_model  # noqa: F401

        _BACKEND = "torch_xla"
        return _BACKEND
    except ImportError:
        _BACKEND = None
        return None


def init_nki_runtime() -> Optional[Backend]:
    """Call before @nki.jit kernel use. Imports torch_neuronx per TorchNeuron docs."""
    return detect_backend()


def get_device() -> torch.device:
    """Primary device for NKI kernels invoked from torch tensors."""
    backend = init_nki_runtime()
    if backend == "torch_neuronx":
        return torch.device("neuron")
    if backend == "torch_xla":
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    raise RuntimeError(
        "No Trainium PyTorch backend found. Install TorchNeuron (torch_neuronx) "
        "or legacy torch-neuronx with torch_xla."
    )


def sync_device() -> None:
    """Flush pending device work before host reads (.cpu() also synchronizes on TorchNeuron)."""
    backend = detect_backend()
    if backend == "torch_xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    elif backend == "torch_neuronx":
        if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
            torch.neuron.synchronize()


def to_neuron(t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Move tensor to Trainium device."""
    dev = get_device()
    if dtype is not None:
        return t.to(dtype=dtype, device=dev)
    return t.to(device=dev)
