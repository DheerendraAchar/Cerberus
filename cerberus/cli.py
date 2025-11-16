"""Command-line entrypoints for Project Cerberus scaffold."""
from typing import Optional


def run_from_config(config_path: str) -> None:
    """Run a minimal pipeline based on the YAML config.

    Behaviour:
    - Loads config
    - If `model.path` is provided, attempts to load it, otherwise creates a tiny dummy model
    - Loads CIFAR-10 test loader (if requested)
    - Computes baseline accuracy (best-effort)
    - Attempts to run FGSM if specified and ART is available
    - Writes a small HTML report to outputs/report.html
    """
    from .config import load_config
    from .report import generate_html_report

    cfg = load_config(config_path)

    # Lazy imports
    try:
        import torch
        from .dataset import get_cifar10_loaders
        from .model import load_pytorch_model
    except Exception:
        torch = None

    metrics = {}

    # Prepare test loader
    test_loader = None
    if cfg.get("dataset", {}).get("name") == "cifar10":
        if torch is None:
            raise RuntimeError("torch is required to load CIFAR-10 dataset")
        _, test_loader = get_cifar10_loaders(root=cfg.get("dataset", {}).get("root", "./data"), batch_size=cfg.get("dataset", {}).get("batch_size", 64), num_workers=cfg.get("dataset", {}).get("num_workers", 0))

    # Load or build model
    model = None
    device = "cpu"
    model_path = cfg.get("model", {}).get("path") if cfg.get("model") else None
    if model_path:
        model = load_pytorch_model(model_path, device=device)
    else:
        # Build a tiny dummy model (very small conv net)
        if torch is None:
            raise RuntimeError("torch is required to build a demo model")
        import torch.nn as nn

        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 10)

            def forward(self, x):
                x = self.conv(x)
                x = torch.relu(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = TinyCNN()

    # Baseline evaluation (best-effort, only if we have a test loader)
    if test_loader is not None:
        model.eval()
        correct = 0
        total = 0
        for xb, yb in test_loader:
            if torch is not None:
                xb = xb.to(device)
                yb = yb.to(device)
            with torch.no_grad():
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        metrics["baseline_accuracy"] = float(correct / total) if total else 0.0

    # Optional attack
    attack_cfg = cfg.get("attack") or {}
    if attack_cfg.get("name") == "fgsm":
        try:
            from .attacks import run_fgsm_attack

            res = run_fgsm_attack(model, test_loader, eps=attack_cfg.get("eps", 0.03), device=device)
            metrics.update(res)
        except Exception as exc:
            metrics["attack_error"] = str(exc)

    # Generate report
    generate_html_report(metrics, out_path=cfg.get("output", {}).get("report_path", "outputs/report.html"))
