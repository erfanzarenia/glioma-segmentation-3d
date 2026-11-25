"""Device selection."""

import torch


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Turn a device spec into a ``torch.device``.

    ``None`` or ``"auto"`` picks CUDA when it is available and falls back to
    the CPU otherwise. Anything else (``"cpu"``, ``"cuda"``, ``"cuda:1"``, a
    ``torch.device``) is passed through unchanged.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
