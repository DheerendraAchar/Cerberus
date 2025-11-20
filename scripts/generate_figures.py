"""Figure generation utilities for Project Cerberus.

Generates core Phase 1 visual artifacts:
  1. Clean vs adversarial (FGSM) sample grid
  2. Accuracy vs epsilon curve
  3. Confusion matrices (clean vs FGSM)
  4. Perturbation heatmap for a single sample

Usage (after installing torch, torchvision, matplotlib, scikit-learn):

    python scripts/generate_figures.py \
        --device cpu \
        --eps-list 0.0 0.01 0.02 0.03 0.05 0.07 0.10 \
        --samples 12 \
        --fgsm-eps 0.03

Outputs saved to ./figures/ by default.

If torch or torchvision are not installed, the script will skip figure generation
with a clear message.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Iterable, List

# Lazy imports guarded so the script can fail gracefully if environment incomplete
try:
    import torch
    import torchvision
    from torch import nn
except Exception as e:  # noqa: BLE001
    torch = None  # type: ignore
    torchvision = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt
except Exception as e:  # noqa: BLE001
    plt = None  # type: ignore
    _MPL_IMPORT_ERROR = e
else:
    _MPL_IMPORT_ERROR = None

try:
    from sklearn.metrics import confusion_matrix
except Exception as e:  # noqa: BLE001
    confusion_matrix = None  # type: ignore
    _SKLEARN_IMPORT_ERROR = e
else:
    _SKLEARN_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Utility: Tiny demo CNN (mirrors concept used in cerberus.cli)
# ---------------------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Core figure generation functions
# ---------------------------------------------------------------------------

def ensure_figures_dir(path: str = "figures") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _require(condition: bool, name: str, err) -> None:
    if not condition:
        raise RuntimeError(f"Missing dependency for {name}: {err}")


def generate_fgsm_examples(model: nn.Module, device: object, eps: float, n: int, save_path: str) -> None:
    """Generate a grid of clean vs FGSM adversarial examples."""
    _require(torch is not None and torchvision is not None, "torch/torchvision", _TORCH_IMPORT_ERROR)
    _require(plt is not None, "matplotlib", _MPL_IMPORT_ERROR)
    model.eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    import numpy as np

    idx = np.random.choice(len(test_set), n, replace=False)
    images = torch.stack([test_set[i][0] for i in idx]).to(device)
    labels = torch.tensor([test_set[i][1] for i in idx]).to(device)

    images.requires_grad = True
    outputs = model(images)
    _, pred_clean = outputs.max(1)
    loss = nn.functional.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad_sign = images.grad.detach().sign()
    adv_images = torch.clamp(images + eps * grad_sign, 0, 1)
    adv_outputs = model(adv_images)
    _, pred_adv = adv_outputs.max(1)

    classes = test_set.classes
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 3))
    for i in range(n):
        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[0, i].set_title(classes[pred_clean[i].item()], fontsize=8)
        axes[1, i].imshow(adv_images[i].detach().cpu().permute(1, 2, 0))
        axes[1, i].axis("off")
        color = "red" if pred_adv[i] != pred_clean[i] else "green"
        axes[1, i].set_title(classes[pred_adv[i].item()], fontsize=8, color=color)

    fig.suptitle(f"Clean (top) vs FGSM (bottom), eps={eps}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def fgsm_accuracy_curve(model: nn.Module, device: object, eps_values: Iterable[float], save_path: str) -> None:
    """Plot accuracy vs epsilon for FGSM."""
    _require(torch is not None and torchvision is not None, "torch/torchvision", _TORCH_IMPORT_ERROR)
    _require(plt is not None, "matplotlib", _MPL_IMPORT_ERROR)
    transform = torchvision.transforms.ToTensor()
    test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    def accuracy(eps: float) -> float:
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if eps > 0:
                x.requires_grad = True
                out = model(x)
                loss = nn.functional.cross_entropy(out, y)
                model.zero_grad()
                loss.backward()
                grad = x.grad.detach().sign()
                x_adv = torch.clamp(x + eps * grad, 0, 1)
                x = x_adv.detach()
            out = model(x)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        return correct / total

    eps_values_list = list(eps_values)
    base_acc = accuracy(0.0)
    adv_accs = [accuracy(e) for e in eps_values_list]

    plt.figure(figsize=(6, 4))
    plt.plot([0.0] + eps_values_list, [base_acc] + adv_accs, marker="o")
    plt.xlabel("FGSM epsilon")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs FGSM epsilon")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=160)
    plt.close()


def confusion_mats(model: nn.Module, device: object, eps: float, prefix: str) -> None:
    """Generate clean and adversarial confusion matrices."""
    _require(torch is not None and torchvision is not None, "torch/torchvision", _TORCH_IMPORT_ERROR)
    _require(plt is not None, "matplotlib", _MPL_IMPORT_ERROR)
    _require(confusion_matrix is not None, "scikit-learn", _SKLEARN_IMPORT_ERROR)
    transform = torchvision.transforms.ToTensor()
    test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    y_true, y_clean, y_adv = [], [], []
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        out = model(x)
        _, pc = out.max(1)
        loss = nn.functional.cross_entropy(out, y)
        model.zero_grad()
        loss.backward()
        adv_x = torch.clamp(x + eps * x.grad.detach().sign(), 0, 1)
        out_adv = model(adv_x)
        _, pa = out_adv.max(1)
        y_true.extend(y.cpu().tolist())
        y_clean.extend(pc.cpu().tolist())
        y_adv.extend(pa.cpu().tolist())

    cm_clean = confusion_matrix(y_true, y_clean)
    cm_adv = confusion_matrix(y_true, y_adv)
    classes = test_ds.classes

    def plot_cm(cm, title, fname):
        import numpy as np  # local
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title(title)
        plt.xticks(range(len(classes)), classes, rotation=90, fontsize=7)
        plt.yticks(range(len(classes)), classes, fontsize=7)
        plt.colorbar(fraction=0.046, pad=0.04)
        # annotate (optional)
        thresh = cm.max() * 0.6
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = cm[i, j]
                if val > 0:
                    plt.text(j, i, str(val), ha="center", va="center", color="white" if val > thresh else "black", fontsize=6)
        plt.tight_layout()
        plt.savefig(fname, dpi=160)
        plt.close()

    plot_cm(cm_clean, "Confusion Matrix (Clean)", f"{prefix}_clean.png")
    plot_cm(cm_adv, f"Confusion Matrix (FGSM eps={eps})", f"{prefix}_fgsm_eps{eps}.png")


def perturbation_heatmap(model: nn.Module, device: object, eps: float, save_path: str) -> None:
    """Generate a heatmap of |adv - clean| averaged over channels for a single sample."""
    _require(torch is not None and torchvision is not None, "torch/torchvision", _TORCH_IMPORT_ERROR)
    _require(plt is not None, "matplotlib", _MPL_IMPORT_ERROR)
    transform = torchvision.transforms.ToTensor()
    test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    x, _ = test_ds[0]
    x = x.unsqueeze(0).to(device)
    x.requires_grad = True
    out = model(x)
    target = out.argmax(1)
    loss = nn.functional.cross_entropy(out, target)
    model.zero_grad()
    loss.backward()
    adv = torch.clamp(x + eps * x.grad.detach().sign(), 0, 1)
    diff = (adv - x).abs().squeeze(0).mean(0).cpu()  # average over channels
    plt.figure(figsize=(4, 4))
    plt.imshow(diff, cmap="inferno")
    plt.title(f"FGSM Perturbation Heatmap (eps={eps})")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_model(device: object) -> nn.Module:
    model = TinyCNN().to(device)
    # Random initialization is fine for qualitative demonstration; optionally load trained weights if available.
    return model


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate adversarial evaluation figures for Cerberus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", help="Computation device (cpu or cuda if available)")
    parser.add_argument("--eps-list", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.03, 0.05], help="List of epsilons for accuracy curve")
    parser.add_argument("--fgsm-eps", type=float, default=0.03, help="Epsilon for sample grid & confusion matrices")
    parser.add_argument("--samples", type=int, default=12, help="Number of sample images in grid")
    parser.add_argument("--skip-grid", action="store_true", help="Skip clean vs adversarial example grid")
    parser.add_argument("--skip-curve", action="store_true", help="Skip accuracy vs epsilon curve")
    parser.add_argument("--skip-confusion", action="store_true", help="Skip confusion matrices")
    parser.add_argument("--skip-heatmap", action="store_true", help="Skip perturbation heatmap")
    parser.add_argument("--output-dir", default="figures", help="Directory to store generated figures")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Dependency check summary
    missing = []
    if torch is None:
        missing.append(f"torch/torchvision ({_TORCH_IMPORT_ERROR})")
    if plt is None:
        missing.append(f"matplotlib ({_MPL_IMPORT_ERROR})")
    if confusion_matrix is None:
        missing.append(f"scikit-learn ({_SKLEARN_IMPORT_ERROR})")
    if missing:
        print("WARNING: Missing required libraries; figures will be skipped:\n" + "\n".join(" - " + m for m in missing))
        return 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(device)

    out_dir = ensure_figures_dir(args.output_dir)
    print(f"Generating figures into {out_dir} ...")

    try:
        if not args.skip_grid:
            path = os.path.join(out_dir, f"fgsm_examples_eps{args.fgsm_eps}.png")
            generate_fgsm_examples(model, device, args.fgsm_eps, args.samples, path)
            print(f"[OK] Example grid: {path}")
        if not args.skip_curve:
            path = os.path.join(out_dir, "fgsm_accuracy_vs_epsilon.png")
            fgsm_accuracy_curve(model, device, args.eps_list, path)
            print(f"[OK] Accuracy curve: {path}")
        if not args.skip_confusion:
            prefix = os.path.join(out_dir, "confusion")
            confusion_mats(model, device, args.fgsm_eps, prefix)
            print(f"[OK] Confusion matrices: {prefix}_clean.png / {prefix}_fgsm_eps{args.fgsm_eps}.png")
        if not args.skip_heatmap:
            path = os.path.join(out_dir, f"fgsm_perturbation_heatmap_eps{args.fgsm_eps}.png")
            perturbation_heatmap(model, device, args.fgsm_eps, path)
            print(f"[OK] Perturbation heatmap: {path}")
    except Exception as e:  # noqa: BLE001
        print("ERROR during figure generation:", e)
        return 2

        print("Figure generation complete.")
        summary = f"""
        Add these figures to your report:
            - fgsm_examples_eps{args.fgsm_eps}.png (Clean vs Adv samples)
            - fgsm_accuracy_vs_epsilon.png (Accuracy curve)
            - confusion_clean.png / confusion_fgsm_eps{args.fgsm_eps}.png (Confusion matrices)
            - fgsm_perturbation_heatmap_eps{args.fgsm_eps}.png (Perturbation heatmap)
        """
        print(textwrap.dedent(summary).strip())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
