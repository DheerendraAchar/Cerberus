"""Tests for dataset loader (using mocks)."""
import sys
from unittest.mock import MagicMock, patch


def test_get_cifar10_loaders():
    """Test CIFAR-10 loader with mock torch and torchvision."""
    mock_torch = MagicMock()
    mock_torchvision = MagicMock()

    mock_train_ds = MagicMock()
    mock_test_ds = MagicMock()
    mock_loader_train = MagicMock()
    mock_loader_test = MagicMock()

    mock_torchvision.datasets.CIFAR10.side_effect = [mock_train_ds, mock_test_ds]
    mock_torch.utils.data.DataLoader.side_effect = [mock_loader_train, mock_loader_test]

    with patch.dict(sys.modules, {"torch": mock_torch, "torchvision": mock_torchvision, "torchvision.datasets": mock_torchvision.datasets, "torchvision.transforms": mock_torchvision.transforms}):
        from cerberus.dataset import get_cifar10_loaders

        train_loader, test_loader = get_cifar10_loaders(root="./data", batch_size=32, num_workers=0)

        assert train_loader == mock_loader_train
        assert test_loader == mock_loader_test


def test_get_cifar10_loaders_raises_if_torch_missing():
    """Test CIFAR-10 loader raises clear error if torch/torchvision missing."""
    # Remove torch from modules to simulate missing
    original_torch = sys.modules.pop("torch", None)
    original_tv = sys.modules.pop("torchvision", None)
    try:
        from cerberus.dataset import get_cifar10_loaders

        try:
            get_cifar10_loaders()
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "torch and torchvision are required" in str(e)
    finally:
        if original_torch:
            sys.modules["torch"] = original_torch
        if original_tv:
            sys.modules["torchvision"] = original_tv
