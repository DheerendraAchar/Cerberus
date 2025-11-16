"""Model ingestion utilities.

Functions import torch lazily to avoid import-time failures in environments without torch.
"""
from typing import Any, Optional


def load_pytorch_model(path: str, device: str = "cpu") -> Any:
    """Load a PyTorch model from `path` and put it on `device`.

    This function tries common loading patterns (torch.jit, torch.load returning a model
    or a state_dict). For complex custom models, users should provide a model builder
    and call `load_state_dict` themselves.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required to load PyTorch models") from exc

    # Try torch.jit.load first
    try:
        model = torch.jit.load(path, map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    # Try torch.load
    obj = torch.load(path, map_location=device)
    # If it looks like a model instance, return it
    if hasattr(obj, "eval"):
        obj.eval()
        return obj

    # If it's a state_dict, return it (caller must construct model)
    if isinstance(obj, dict):
        return obj

    # Unknown type
    raise RuntimeError("Loaded object from path is not a recognizable PyTorch model or state_dict")
