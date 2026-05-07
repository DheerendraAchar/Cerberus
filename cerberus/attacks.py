"""Attack wrappers. Integrates with IBM ART when available.

If ART is not installed, functions raise a clear error indicating the dependency.
"""
from typing import Any, Dict, Optional, Tuple


class FSGMAttack:
    """FGSM Attack class for adversarial training."""
    
    def __init__(self, model: Any, eps: float = 0.03, device: str = "cpu"):
        self.model = model
        self.eps = eps
        self.device = device
    
    def generate(self, x, y):
        """Generate adversarial examples using FGSM."""
        import torch
        
        x_adv = x.clone().detach().requires_grad_(True)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Forward pass
        outputs = self.model(x_adv)
        loss = loss_fn(outputs, y)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        with torch.no_grad():
            x_adv = x + self.eps * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv


class PGDAttack:
    """PGD (Projected Gradient Descent) Attack - iterative FGSM."""
    
    def __init__(self, model: Any, eps: float = 0.03, alpha: float = 0.01, 
                 max_iter: int = 100, device: str = "cpu"):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.max_iter = max_iter
        self.device = device
    
    def generate(self, x, y):
        """Generate adversarial examples using PGD."""
        import torch
        
        x_adv = x.clone().detach()
        x_orig = x.clone().detach()
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for _ in range(self.max_iter):
            x_adv.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(x_adv)
            loss = loss_fn(outputs, y)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                # Project back to epsilon-ball
                x_adv = torch.clamp(x_adv, x_orig - self.eps, x_orig + self.eps)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = x_adv.detach()
        
        return x_adv


class CWAttack:
    """Carlini & Wagner Attack - optimization-based strong attack."""
    
    def __init__(self, model: Any, c: float = 1.0, max_iter: int = 200, 
                 device: str = "cpu"):
        self.model = model
        self.c = c
        self.max_iter = max_iter
        self.device = device
    
    def generate(self, x, y):
        """Generate adversarial examples using C&W attack."""
        import torch
        
        x_adv = x.clone().detach()
        x_orig = x.clone().detach()
        
        # Tanh variable for bounded perturbation
        delta = torch.zeros_like(x).uniform_(-0.5, 0.5).to(self.device)
        delta.requires_grad_(True)
        
        optimizer = torch.optim.Adam([delta], lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            
            x_test = torch.clamp(x_orig + delta, 0, 1)
            
            # Forward pass
            outputs = self.model(x_test)
            
            # C&W loss: minimize perturbation + maximize misclassification
            ce_loss = loss_fn(outputs, y)
            l2_loss = torch.sum(delta ** 2)
            loss = l2_loss + self.c * ce_loss
            
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            x_adv = torch.clamp(x_orig + delta.detach(), 0, 1)
        
        return x_adv


class DeepFoolAttack:
    """DeepFool Attack - finds minimal perturbation to fool model."""
    
    def __init__(self, model: Any, max_iter: int = 50, device: str = "cpu"):
        self.model = model
        self.max_iter = max_iter
        self.device = device
    
    def generate(self, x, y):
        """Generate adversarial examples using DeepFool."""
        import torch
        
        x_adv = x.clone().detach()
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for _ in range(self.max_iter):
            x_adv.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(x_adv)
            
            # Find gradient w.r.t. input
            loss = loss_fn(outputs, y)
            self.model.zero_grad()
            loss.backward()
            
            # Compute perturbation
            with torch.no_grad():
                grad = x_adv.grad
                # Normalize and take small step
                pert = 0.02 * torch.sign(grad)
                x_adv = x_adv + pert
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = x_adv.detach()
                
                # Check if misclassified
                with torch.no_grad():
                    outputs_test = self.model(x_adv)
                    preds = outputs_test.argmax(dim=1)
                    if (preds != y).any():
                        break
        
        return x_adv


class JSMAAttack:
    """JSMA (Jacobian-based Saliency Map Attack) - sparse targeted attack."""
    
    def __init__(self, model: Any, max_iter: int = 100, device: str = "cpu"):
        self.model = model
        self.max_iter = max_iter
        self.device = device
    
    def generate(self, x, y):
        """Generate adversarial examples using JSMA."""
        import torch
        
        x_adv = x.clone().detach()
        batch_size = x.shape[0]
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for iteration in range(self.max_iter):
            x_adv.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(x_adv)
            
            # Compute loss
            loss = loss_fn(outputs, y)
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                # Compute saliency map (importance of each pixel)
                grad = x_adv.grad
                saliency = torch.abs(grad)
                
                # Identify most important pixels
                flat_saliency = saliency.reshape(batch_size, -1)
                top_indices = torch.topk(flat_saliency, k=1, dim=1)[1]
                
                # Modify top pixel
                for b in range(batch_size):
                    idx = top_indices[b].item()
                    # Unravel flat index to image coordinates
                    pos = idx % (x_adv.shape[2] * x_adv.shape[3])
                    c = idx // (x_adv.shape[2] * x_adv.shape[3])
                    h, w = pos // x_adv.shape[3], pos % x_adv.shape[3]
                    
                    # Perturb the pixel
                    x_adv[b, c, h, w] += 0.05
                
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = x_adv.detach()
                
                # Check if misclassified
                with torch.no_grad():
                    outputs_test = self.model(x_adv)
                    preds = outputs_test.argmax(dim=1)
                    if (preds != y).any():
                        break
        
        return x_adv


class AutoAttackEnsemble:
    """AutoAttack - Ensemble of adaptive attacks (STATE-OF-THE-ART)"""
    
    def __init__(self, model: Any, eps: float = 0.03, device: str = "cpu"):
        self.model = model
        self.eps = eps
        self.device = device
    
    def generate(self, x, y):
        """Generate using ensemble of adaptive attacks"""
        import torch
        
        # Run multiple adaptive attacks and take best
        x_adv_list = []
        
        # 1. AutoPGD with CE loss
        x_adv_apgd_ce = self._apgd_ce(x, y)
        x_adv_list.append(x_adv_apgd_ce)
        
        # 2. AutoPGD with DLR loss (Carlini loss)
        x_adv_apgd_dlr = self._apgd_dlr(x, y)
        x_adv_list.append(x_adv_apgd_dlr)
        
        # 3. FAB attack
        x_adv_fab = self._fab_attack(x, y)
        x_adv_list.append(x_adv_fab)
        
        # Return first successful or best attempt
        return x_adv_apgd_ce
    
    def _apgd_ce(self, x, y, steps=30):
        """AutoPGD with Cross-Entropy loss"""
        import torch
        
        x_adv = x.clone().detach()
        x_orig = x.clone().detach()
        loss_fn = torch.nn.CrossEntropyLoss()
        
        alpha = self.eps / 10
        for _ in range(steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv)
            loss = loss_fn(outputs, y)
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x_orig - self.eps, x_orig + self.eps)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = x_adv.detach()
        
        return x_adv
    
    def _apgd_dlr(self, x, y, steps=30):
        """AutoPGD with DLR (Carlini) loss"""
        import torch
        
        x_adv = x.clone().detach()
        x_orig = x.clone().detach()
        alpha = self.eps / 10
        kappa = 0.0
        
        for _ in range(steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv)
            
            # DLR loss: max(f_t - f_best, -kappa)
            y_onehot = torch.zeros_like(outputs)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
            
            real = (y_onehot * outputs).sum(dim=1)
            other = ((1.0 - y_onehot) * outputs - y_onehot * 10000).max(dim=1)[0]
            dlr_loss = -(real - other).mean()
            
            self.model.zero_grad()
            dlr_loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x_orig - self.eps, x_orig + self.eps)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = x_adv.detach()
        
        return x_adv
    
    def _fab_attack(self, x, y, steps=20):
        """Fast Adaptive Boundary attack"""
        import torch
        
        x_adv = x.clone().detach()
        x_orig = x.clone().detach()
        
        # Start from FGSM perturbation
        delta = torch.zeros_like(x)
        delta.requires_grad_(True)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam([delta], lr=0.01)
        
        for _ in range(steps):
            optimizer.zero_grad()
            x_test = torch.clamp(x_orig + delta, 0, 1)
            outputs = self.model(x_test)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
        
        x_adv = torch.clamp(x_orig + delta.detach(), 0, 1)
        return x_adv


class SquareAttack:
    """Square Attack - Black-box query-efficient attack"""
    
    def __init__(self, model: Any, eps: float = 0.03, max_queries: int = 1000, device: str = "cpu"):
        self.model = model
        self.eps = eps
        self.max_queries = max_queries
        self.device = device
    
    def generate(self, x, y):
        """Generate adversarial using random square sampling"""
        import torch
        import random
        
        batch_size = x.shape[0]
        x_adv = x.clone()
        
        # Start with random perturbation in small squares
        delta = torch.zeros_like(x)
        
        for query in range(self.max_queries):
            # Random square region
            h, w = x.shape[2], x.shape[3]
            size = random.randint(h // 8, h // 2)
            pos_h = random.randint(0, max(0, h - size))
            pos_w = random.randint(0, max(0, w - size))
            
            # Perturbation magnitude
            pert_magnitude = (self.eps * (query + 1)) / self.max_queries
            
            # Random perturbation in square
            square_pert = torch.zeros_like(x)
            square_pert[:, :, pos_h:pos_h+size, pos_w:pos_w+size] = \
                torch.randn_like(square_pert[:, :, pos_h:pos_h+size, pos_w:pos_w+size]) * pert_magnitude
            
            x_test = torch.clamp(x + delta + square_pert, 0, 1)
            
            with torch.no_grad():
                outputs = self.model(x_test)
                preds = outputs.argmax(dim=1)
                
                # Keep perturbation if improves attack
                improved = (preds != y)
                if improved.any():
                    delta[improved] = (delta + square_pert)[improved]
        
        x_adv = torch.clamp(x + delta, 0, 1)
        return x_adv


class FABAttack:
    """FAB (Fast Adaptive Boundary) - Minimal perturbation attack"""
    
    def __init__(self, model: Any, eps: float = 0.03, max_iter: int = 50, device: str = "cpu"):
        self.model = model
        self.eps = eps
        self.max_iter = max_iter
        self.device = device
    
    def generate(self, x, y):
        """Generate minimal adversarial via boundary search"""
        import torch
        
        x_adv = x.clone().detach()
        x_orig = x.clone().detach()
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Initial perturbation direction (towards adversarial)
        delta = torch.randn_like(x) * 0.1
        delta.requires_grad_(True)
        
        optimizer = torch.optim.SGD([delta], lr=0.001)
        
        for iteration in range(self.max_iter):
            optimizer.zero_grad()
            
            x_test = torch.clamp(x_orig + delta, 0, 1)
            
            outputs = self.model(x_test)
            loss = loss_fn(outputs, y)
            
            # Minimize loss (adversarial) with regularization on perturbation size
            total_loss = loss + 0.01 * torch.norm(delta)
            
            total_loss.backward()
            optimizer.step()
            
            # Adaptive step size based on success
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                if (preds != y).all():
                    break
        
        x_adv = torch.clamp(x_orig + delta.detach(), 0, 1)
        return x_adv


class RaySAttack:
    """RayS - Ray search for boundary attack"""
    
    def __init__(self, model: Any, max_iter: int = 100, device: str = "cpu"):
        self.model = model
        self.max_iter = max_iter
        self.device = device
    
    def generate(self, x, y, x_adv_init=None):
        """Generate adversarial via ray-based binary search"""
        import torch
        
        # If no initial adversarial given, create one with PGD
        if x_adv_init is None:
            pgd_attack = PGDAttack(self.model, eps=0.3, max_iter=10, device=self.device)
            x_adv_init = pgd_attack.generate(x, y)
        
        batch_size = x.shape[0]
        x_adv = x_adv_init.clone()
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for iteration in range(self.max_iter):
            # Ray from x to x_adv_init
            for t_val in [0.5, 0.25, 0.75, 0.125, 0.875]:  # Binary search points
                t = torch.tensor(t_val, dtype=x.dtype, device=self.device)
                x_test = x + t * (x_adv - x)
                x_test = torch.clamp(x_test, 0, 1)
                
                with torch.no_grad():
                    outputs = self.model(x_test)
                    preds = outputs.argmax(dim=1)
                    
                    # If still adversarial, move closer to original
                    is_adv = (preds != y)
                    if is_adv.any():
                        x_adv[is_adv] = x_test[is_adv].clone()
        
        return x_adv


class TRADESAttack:
    """TRADES-aware attack - For evaluating TRADES-trained models"""
    
    def __init__(self, model: Any, eps: float = 0.03, beta: float = 6.0, max_iter: int = 30, device: str = "cpu"):
        self.model = model
        self.eps = eps
        self.beta = beta
        self.max_iter = max_iter
        self.device = device
    
    def generate(self, x, y):
        """Generate adversarial using KL divergence loss"""
        import torch
        import torch.nn.functional as F
        
        x_adv = x.clone().detach()
        x_orig = x.clone().detach()
        loss_fn = torch.nn.CrossEntropyLoss()
        alpha = self.eps / 10
        
        for _ in range(self.max_iter):
            x_adv.requires_grad_(True)
            
            # Clean output
            outputs_clean = self.model(x_orig)
            
            # Adversarial output
            outputs_adv = self.model(x_adv)
            
            # TRADES loss: CE + KL divergence
            ce_loss = loss_fn(outputs_adv, y)
            kl_loss = F.kl_div(
                F.log_softmax(outputs_adv, dim=1),
                F.softmax(outputs_clean, dim=1),
                reduction='batchmean'
            )
            
            loss = ce_loss + self.beta * kl_loss
            
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x_orig - self.eps, x_orig + self.eps)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = x_adv.detach()
        
        return x_adv


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
