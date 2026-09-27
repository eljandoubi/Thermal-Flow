"""Device selection: prefers CUDA (training GPU machine), then Apple MPS, then CPU."""

from __future__ import annotations

import torch

_VALID_DEVICE_TYPES = {"cuda", "mps", "cpu"}


def get_device(preferred: str | None = None) -> torch.device:
    """Return the best available torch device.

    If `preferred` ("cuda", "mps" or "cpu") is given, it is used after checking
    it is actually available on this machine. Otherwise the best available
    accelerator is selected automatically: CUDA > MPS > CPU.
    """
    if preferred is not None:
        if preferred not in _VALID_DEVICE_TYPES:
            raise ValueError(
                f"Unknown device type: {preferred!r} (expected one of {_VALID_DEVICE_TYPES})"
            )
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available on this machine.")
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
