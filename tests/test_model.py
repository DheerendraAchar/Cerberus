"""Tests for model loader (using mocks to avoid torch dependency)."""
import tempfile
import os
import sys
from unittest.mock import MagicMock, patch


def test_load_pytorch_model_jit():
    """Test loading a TorchScript model (mocked)."""
    mock_torch = MagicMock()
    mock_model = MagicMock()
    mock_model.eval = MagicMock(return_value=None)
    mock_torch.jit.load.return_value = mock_model

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        with patch.dict(sys.modules, {"torch": mock_torch}):
            from cerberus.model import load_pytorch_model

            result = load_pytorch_model(path, device="cpu")
            assert result == mock_model
            mock_model.eval.assert_called_once()
    finally:
        os.unlink(path)


def test_load_pytorch_model_raises_if_torch_missing():
    """Test that load_pytorch_model raises a clear error if torch is not available."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        # Block torch import by removing it and making import fail
        original_torch = sys.modules.pop("torch", None)
        try:
            from cerberus.model import load_pytorch_model

            try:
                load_pytorch_model(path)
                assert False, "Expected RuntimeError"
            except RuntimeError as e:
                assert "PyTorch is required" in str(e)
        finally:
            if original_torch:
                sys.modules["torch"] = original_torch
    finally:
        os.unlink(path)
