"""Command-line entrypoints for Project Cerberus scaffold."""
from typing import Optional


def run_training(
    config_path: str,
    training_type: str,
    num_epochs: Optional[int] = None,
    output_path: Optional[str] = None
) -> None:
    """Run model training pipeline (Phase 2).
    
    Args:
        config_path: Path to training config YAML file
        training_type: Either 'baseline' or 'adversarial'
        num_epochs: Override config epochs (optional)
        output_path: Override model save path (optional)
    """
    from .config import load_config
    
    print(f"\n{'='*70}")
    print(f"CERBERUS PHASE 2 - MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Training Type: {training_type}")
    print(f"Config:        {config_path}")
    print(f"{'='*70}\n")
    
    # Load config
    cfg = load_config(config_path)
    
    # Import dependencies
    try:
        import torch
        import torch.nn as nn
        from .dataset import get_cifar10_loaders
    except ImportError as e:
        print(f"❌ Error: Required dependencies not installed: {e}")
        print("   Install: pip install torch torchvision")
        return
    
    # Get device
    device = "cuda" if torch.cuda.is_available() and cfg.get("device", {}).get("use_gpu", False) else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load dataset
    dataset_cfg = cfg.get("dataset", {})
    print(f"📦 Loading {dataset_cfg.get('name', 'CIFAR10')} dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        root=dataset_cfg.get("data_dir", "./data"),
        batch_size=dataset_cfg.get("batch_size", 128),
        num_workers=dataset_cfg.get("num_workers", 2)
    )
    print(f"   ✅ Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Build model
    model_cfg = cfg.get("model", {})
    print(f"🏗️  Building {model_cfg.get('architecture', 'resnet18')} model...")
    
    # Simple ResNet-18 for CIFAR-10
    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.relu(out)
            return out
    
    class SimpleResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(64, 64, 2, stride=1)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, num_classes)
        
        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            layers.append(BasicBlock(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(BasicBlock(out_channels, out_channels, 1))
            return nn.Sequential(*layers)
        
        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
    
    model = SimpleResNet(num_classes=model_cfg.get("num_classes", 10))
    print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Get training parameters
    train_cfg = cfg.get("training", {})
    epochs = num_epochs if num_epochs is not None else train_cfg.get("num_epochs", 100)
    lr = train_cfg.get("learning_rate", 0.01)
    
    # Determine save path
    if output_path:
        save_path = output_path
    else:
        model_dir = train_cfg.get("model_save_dir", "outputs/models/")
        save_path = f"{model_dir}/{training_type}_model.pt"
    
    # Train based on type
    if training_type == "baseline":
        from .baseline_training import BaselineTrainer
        
        print(f"🚀 Starting Baseline Training...")
        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=lr,
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 5e-4)
        )
        
        history = trainer.train(
            num_epochs=epochs,
            save_best=train_cfg.get("save_best", True),
            save_path=save_path
        )
        
    elif training_type == "adversarial":
        from .adversarial_training import AdversarialTrainer
        
        adv_cfg = cfg.get("adversarial", {})
        print(f"🛡️  Starting Adversarial Training...")
        print(f"   Epsilon: {adv_cfg.get('epsilon', 0.03)}")
        print(f"   Alpha (mix ratio): {adv_cfg.get('alpha', 0.5)}")
        
        trainer = AdversarialTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epsilon=adv_cfg.get("epsilon", 0.03),
            alpha=adv_cfg.get("alpha", 0.5),
            learning_rate=lr,
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 5e-4)
        )
        
        history = trainer.train(
            num_epochs=epochs,
            save_best=train_cfg.get("save_best", True),
            save_path=save_path
        )
    
    else:
        print(f"❌ Unknown training type: {training_type}")
        return
    
    print(f"\n✅ Training complete! Model saved to: {save_path}")
    print(f"📊 To visualize training curves, run:")
    print(f"   python scripts/plot_training_curves.py --{training_type}-checkpoint {save_path}")
    print()


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
    attack_type = attack_cfg.get("name", "fgsm")
    
    if attack_type == "fgsm":
        try:
            from .attacks import run_fgsm_attack
            res = run_fgsm_attack(model, test_loader, eps=attack_cfg.get("eps", 0.03), device=device)
            metrics.update(res)
        except Exception as exc:
            metrics["attack_error"] = str(exc)
    
    elif attack_type == "pgd":
        try:
            from .attacks.pgd_attack import PGDAttack
            pgd = PGDAttack(
                model=model,
                eps=attack_cfg.get("epsilon", 0.03),
                eps_step=attack_cfg.get("eps_step", 0.01),
                max_iter=attack_cfg.get("max_iter", 20),
                device=device
            )
            res = pgd.evaluate(test_loader)
            metrics.update(res)
        except Exception as exc:
            metrics["attack_error"] = str(exc)
    
    elif attack_type == "cw":
        try:
            from .attacks.cw_attack import CWAttack
            cw = CWAttack(
                model=model,
                c=attack_cfg.get("c", 1.0),
                learning_rate=attack_cfg.get("learning_rate", 0.01),
                max_iterations=attack_cfg.get("max_iterations", 100),
                device=device
            )
            res = cw.evaluate(test_loader)
            metrics.update(res)
        except Exception as exc:
            metrics["attack_error"] = str(exc)
    
    elif attack_type == "deepfool":
        try:
            from .attacks.deepfool_attack import DeepFoolAttack
            deepfool = DeepFoolAttack(
                model=model,
                max_iterations=attack_cfg.get("max_iterations", 100),
                overshoot=attack_cfg.get("overshoot", 0.02),
                device=device
            )
            res = deepfool.evaluate(test_loader)
            metrics.update(res)
        except Exception as exc:
            metrics["attack_error"] = str(exc)
    
    elif attack_type == "jsma":
        try:
            from .attacks.jsma_attack import JSMAAttack
            jsma = JSMAAttack(
                model=model,
                theta=attack_cfg.get("theta", 1.0),
                gamma=attack_cfg.get("gamma", 0.1),
                max_pixels=attack_cfg.get("max_pixels", 100),
                device=device
            )
            res = jsma.evaluate(test_loader)
            metrics.update(res)
        except Exception as exc:
            metrics["attack_error"] = str(exc)

    # Generate report
    generate_html_report(metrics, out_path=cfg.get("output", {}).get("report_path", "outputs/report.html"))
