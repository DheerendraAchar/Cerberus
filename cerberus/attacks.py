"""Attack wrappers. Integrates with IBM ART when available.

If ART is not installed, functions raise a clear error indicating the dependency.
"""
from typing import Any, Dict, Optional, Tuple


def run_fgsm_attack(pytorch_model: Any, test_loader: Any, eps: float = 0.03, device: str = "cpu") -> Dict[str, float]:
    """Run FGSM attack using ART if available.

    Returns a dictionary of metrics, e.g. baseline_accuracy and adversarial_accuracy.
    """
    try:
        # ART imports
        from art.attacks.evasion import FastGradientMethod
        from art.estimators.classification import PyTorchClassifier
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("ART is required to run attacks. Install 'adversarial-robustness-toolbox' to enable attacks.") from exc

    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        raise RuntimeError("PyTorch is required to run attacks")

    # Build an ART classifier wrapper around the provided model. We assume the model
    # accepts inputs shaped like (N, C, H, W) and outputs logits.
    # This wrapper requires a loss and optimizer just for the ART API; we provide dummy ones.
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=pytorch_model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        device_type="cpu",
    )

    # Compute baseline accuracy
    pytorch_model.eval()
    correct = 0
    total = 0
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.no_grad():
            out = pytorch_model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    baseline_acc = correct / total if total else 0.0

    # Generate adversarial examples with FGSM
    attack = FastGradientMethod(estimator=classifier, eps=eps)

    # Collect all examples into numpy for ART
    import numpy as np

    xs = []
    ys = []
    for xb, yb in test_loader:
        xs.append(xb.numpy())
        ys.append(yb.numpy())
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)

    x_adv = attack.generate(x=xs)

    # Evaluate adversarial accuracy in batches
    batch = 256
    adv_correct = 0
    adv_total = 0
    import math

    for i in range(0, x_adv.shape[0], batch):
        chunk = x_adv[i : i + batch]
        chunk_t = torch.from_numpy(chunk).to(device)
        with torch.no_grad():
            out = pytorch_model(chunk_t)
            preds = out.argmax(dim=1).cpu().numpy()
            adv_correct += (preds == ys[i : i + batch]).sum()
            adv_total += preds.shape[0]

    adv_acc = adv_correct / adv_total if adv_total else 0.0
    return {"baseline_accuracy": float(baseline_acc), "adversarial_accuracy": float(adv_acc)}
