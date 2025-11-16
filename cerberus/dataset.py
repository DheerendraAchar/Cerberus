"""Dataset loaders (lightweight, CPU-friendly).

This module avoids importing torchvision at import-time. Callers should handle
installing torchvision if they want to use the provided CIFAR-10 loader.
"""
from typing import Tuple, Any


def get_cifar10_loaders(root: str = "./data", batch_size: int = 64, num_workers: int = 0) -> Tuple[Any, Any]:
    """Return train and test dataloaders for CIFAR-10.

    Requires `torch` and `torchvision` to be installed. This function imports them lazily.
    """
    try:
        import torch
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("torch and torchvision are required to load CIFAR-10") from exc

    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
