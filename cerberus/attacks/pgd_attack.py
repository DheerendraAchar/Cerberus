"""PGD (Projected Gradient Descent) Attack Implementation.

PGD is a stronger iterative attack compared to FGSM. It performs multiple
gradient steps with perturbation projections at each step to find adversarial
examples within the epsilon ball.

Reference: Madry et al. (2018) - "Towards Deep Learning Models Resistant to 
Adversarial Attacks" (https://arxiv.org/abs/1706.06083)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import time


class PGDAttack:
    """
    PGD (Projected Gradient Descent) Attack.
    
    Iterative attack that:
    1. Takes multiple gradient steps in the direction of increasing loss
    2. Projects perturbations back into the epsilon ball after each step
    3. Clips pixel values to valid range [0, 1]
    
    Parameters:
        model: PyTorch neural network model
        eps: Maximum perturbation magnitude (default 0.03)
        eps_step: Step size per iteration (default 0.01)
        max_iter: Number of iterations (default 20)
        device: 'cpu' or 'cuda'
    
    Example:
        >>> pgd = PGDAttack(model, eps=0.03, eps_step=0.01, max_iter=20)
        >>> adv_examples = pgd.generate_adversarial(images, labels)
        >>> results = pgd.evaluate(test_loader)
    """
    
    def __init__(self,
                 model: nn.Module,
                 eps: float = 0.03,
                 eps_step: float = 0.01,
                 max_iter: int = 20,
                 device: str = "cpu"):
        """Initialize PGD Attack."""
        self.model = model.to(device)
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def generate_adversarial(self,
                           images: torch.Tensor,
                           labels: torch.Tensor,
                           random_start: bool = True) -> torch.Tensor:
        """
        Generate PGD adversarial examples.
        
        Args:
            images: Clean input images [batch_size, channels, height, width]
            labels: True labels [batch_size]
            random_start: If True, start from random point in epsilon ball
            
        Returns:
            Adversarial examples with same shape as input
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Initialize perturbation
        if random_start:
            # Random start improves attack effectiveness
            delta = torch.rand_like(images) * (2 * self.eps) - self.eps
        else:
            delta = torch.zeros_like(images)
        
        delta = delta.to(self.device)
        delta.requires_grad = True
        
        # Ensure model doesn't require gradients (we only need input gradients)
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Iterative attack
        for iteration in range(self.max_iter):
            # Create a fresh variable for x_adv each iteration to track gradients
            delta_copy = delta.clone().detach().requires_grad_(True)
            x_adv = torch.clamp(images + delta_copy, 0, 1)
            
            self.model.eval()
            with torch.enable_grad():
                outputs = self.model(x_adv)
                loss = self.criterion(outputs, labels)
                loss.backward()
            
            # Get gradients from delta
            if delta_copy.grad is None:
                print(f"WARNING: delta_copy.grad is None at iteration {iteration}")
                break
            
            # PGD step: move in gradient direction
            with torch.no_grad():
                # Use gradient from delta_copy
                grad_sign = delta_copy.grad.sign()
                
                # Update delta
                delta = delta + self.eps_step * grad_sign
                
                # Project back to epsilon ball
                delta = torch.clamp(delta, -self.eps, self.eps)
                
                # Ensure valid pixel values
                delta = torch.clamp(images + delta, 0, 1) - images
                delta = torch.clamp(delta, -self.eps, self.eps)
        
        # Final adversarial examples
        x_adv = torch.clamp(images + delta, 0, 1)
        return x_adv.detach()
    
    def evaluate(self,
                test_loader: Any,
                batch_size: int = 100) -> Dict[str, Any]:
        """
        Evaluate model robustness against PGD attack.
        
        Args:
            test_loader: DataLoader for test set
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with results:
                - attack_type: 'PGD'
                - epsilon: Perturbation magnitude
                - iterations: Number of iterations
                - accuracy: Accuracy under attack
                - attack_success_rate: Percentage of successful attacks
                - time_seconds: Total evaluation time
        """
        self.model.eval()
        correct = 0
        total = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Generate adversarial examples
                x_adv = self.generate_adversarial(images, labels)
                
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
        
        return {
            'attack_type': 'PGD',
            'epsilon': self.eps,
            'eps_step': self.eps_step,
            'iterations': self.max_iter,
            'accuracy': accuracy,
            'attack_success_rate': 100.0 - accuracy,
            'total_samples': total,
            'time_seconds': elapsed_time
        }
    
    def __repr__(self) -> str:
        """String representation of PGD attack."""
        return (f"PGDAttack(eps={self.eps}, eps_step={self.eps_step}, "
                f"max_iter={self.max_iter}, device='{self.device}')")
