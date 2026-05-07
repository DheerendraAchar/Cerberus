"""DeepFool Attack Implementation.

DeepFool finds the minimum perturbation to cross the decision boundary
by iteratively computing the distance to the decision boundary and
moving towards it.

Reference: Moosavi-Dezfooli et al. (2016) - "DeepFool: a simple and accurate
method to fool deep neural networks" (https://arxiv.org/abs/1511.04508)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import time


class DeepFoolAttack:
    """
    DeepFool Attack.
    
    Iteratively computes distance to decision boundary and moves
    towards it to find minimal adversarial perturbation.
    
    Useful for understanding the minimum perturbation needed to fool the model.
    
    Parameters:
        model: PyTorch neural network model
        max_iterations: Maximum number of iterations (default 100)
        overshoot: How much to overshoot the boundary (default 0.02)
        device: 'cpu' or 'cuda'
    
    Example:
        >>> deepfool = DeepFoolAttack(model, max_iterations=100, overshoot=0.02)
        >>> adv_examples = deepfool.generate_adversarial(images)
        >>> results = deepfool.evaluate(test_loader)
    """
    
    def __init__(self,
                 model: nn.Module,
                 max_iterations: int = 100,
                 overshoot: float = 0.02,
                 device: str = "cpu"):
        """Initialize DeepFool Attack."""
        self.model = model.to(device)
        self.max_iter = max_iterations
        self.overshoot = overshoot
        self.device = device
        
    def generate_adversarial(self,
                           images: torch.Tensor) -> torch.Tensor:
        """
        Generate DeepFool adversarial examples.
        
        Args:
            images: Clean input images [batch_size, channels, height, width]
            
        Returns:
            Adversarial examples with same shape as input
        """
        images = images.to(self.device)
        batch_size = images.size(0)
        x_adv = images.clone().detach()
        
        self.model.eval()
        
        for iteration in range(self.max_iter):
            x_adv.requires_grad = True
            
            # Forward pass
            outputs = self.model(x_adv)
            _, target_class = outputs.max(1)
            
            # Compute gradients w.r.t. input
            loss = torch.nn.functional.cross_entropy(outputs, target_class)
            
            self.model.zero_grad()
            loss.backward()
            
            # Compute perturbation direction (towards decision boundary)
            with torch.no_grad():
                if x_adv.grad is not None:
                    grad = x_adv.grad
                    grad_norm = torch.norm(
                        grad.reshape(batch_size, -1),
                        p=2,
                        dim=1,
                        keepdim=True
                    ).clamp(min=1e-8)
                    
                    # Reshape grad_norm to match grad dimensions for broadcasting
                    grad_norm_expanded = grad_norm.reshape(batch_size, 1, 1, 1)
                    
                    # Normalized gradient direction
                    delta = grad / grad_norm_expanded
                    
                    # Move towards boundary with overshoot
                    x_adv = x_adv + self.overshoot * delta
                    x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()
    
    def evaluate(self,
                test_loader: Any) -> Dict[str, Any]:
        """
        Evaluate model robustness against DeepFool attack.
        
        Args:
            test_loader: DataLoader for test set
            
        Returns:
            Dictionary with results including perturbation statistics
        """
        self.model.eval()
        correct = 0
        total = 0
        perturbations = []
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Generate adversarial examples
                x_adv = self.generate_adversarial(images)
                
                # Compute perturbation magnitude
                pert = torch.norm(
                    (x_adv - images).reshape(images.size(0), -1),
                    p=2,
                    dim=1
                )
                perturbations.extend(pert.cpu().numpy().tolist())
                
                # Evaluate
                outputs = self.model(x_adv)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  [{batch_idx + 1}/{len(test_loader)}] "
                          f"Accuracy: {100. * correct / total:.2f}%")
        
        elapsed_time = time.time() - start_time
        accuracy = 100.0 * correct / total
        
        import numpy as np
        perturbations = np.array(perturbations)
        
        return {
            'attack_type': 'DeepFool',
            'overshoot': self.overshoot,
            'accuracy': accuracy,
            'attack_success_rate': 100.0 - accuracy,
            'avg_perturbation': float(perturbations.mean()),
            'median_perturbation': float(np.median(perturbations)),
            'max_perturbation': float(perturbations.max()),
            'min_perturbation': float(perturbations.min()),
            'total_samples': total,
            'time_seconds': elapsed_time
        }
    
    def __repr__(self) -> str:
        """String representation of DeepFool attack."""
        return (f"DeepFoolAttack(max_iter={self.max_iter}, "
                f"overshoot={self.overshoot}, device='{self.device}')")
