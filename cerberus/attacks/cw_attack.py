"""Carlini & Wagner (C&W) Attack Implementation.

C&W is an optimization-based attack that finds the minimum perturbation
needed to cause misclassification by solving an optimization problem.

Reference: Carlini & Wagner (2016) - "Towards Evaluating the Robustness of 
Neural Networks" (https://arxiv.org/abs/1608.04644)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import time


class CWAttack:
    """
    Carlini & Wagner (C&W) Attack.
    
    Finds minimal perturbation by solving:
    minimize ||x' - x||_2 + c * f(x')
    
    Where f is a loss function encouraging misclassification.
    Much stronger than FGSM/PGD but also slower.
    
    Parameters:
        model: PyTorch neural network model
        c: Weight parameter for classification loss (default 1.0)
        learning_rate: Learning rate for optimization (default 0.01)
        max_iterations: Number of optimization iterations (default 100)
        device: 'cpu' or 'cuda'
    
    Example:
        >>> cw = CWAttack(model, c=1.0, learning_rate=0.01, max_iterations=100)
        >>> adv_examples = cw.generate_adversarial(images, labels)
        >>> results = cw.evaluate(test_loader)
    """
    
    def __init__(self,
                 model: nn.Module,
                 c: float = 1.0,
                 learning_rate: float = 0.01,
                 max_iterations: int = 100,
                 device: str = "cpu"):
        """Initialize C&W Attack."""
        self.model = model.to(device)
        self.c = c
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def generate_adversarial(self,
                           images: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
        """
        Generate C&W adversarial examples.
        
        Args:
            images: Clean input images [batch_size, channels, height, width]
            labels: True labels [batch_size]
            
        Returns:
            Adversarial examples with same shape as input
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size = images.size(0)
        
        # Initialize perturbation as learnable parameter
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        
        best_delta = delta.clone().detach()
        best_loss = float('inf')
        
        self.model.eval()
        
        for iteration in range(self.max_iter):
            optimizer.zero_grad()
            
            # Perturbed images
            x_adv = images + delta
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # Forward pass
            outputs = self.model(x_adv)
            
            # Classification loss (minimize correct class logit, maximize target)
            ce_loss = self.criterion(outputs, labels)
            
            # L2 distance loss
            l2_loss = torch.norm(
                delta.reshape(batch_size, -1),
                p=2,
                dim=1
            ).mean()
            
            # Combined loss
            loss = l2_loss + self.c * ce_loss
            
            # Track best perturbation
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.clone().detach()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Project to valid range
            with torch.no_grad():
                delta.data = torch.clamp(delta, -0.5, 0.5)  # Reasonable bounds
        
        return torch.clamp(images + best_delta, 0, 1).detach()
    
    def evaluate(self,
                test_loader: Any) -> Dict[str, Any]:
        """
        Evaluate model robustness against C&W attack.
        
        Args:
            test_loader: DataLoader for test set
            
        Returns:
            Dictionary with results:
                - attack_type: 'C&W'
                - c_parameter: C weight value
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
            'attack_type': 'C&W',
            'c_parameter': self.c,
            'learning_rate': self.lr,
            'iterations': self.max_iter,
            'accuracy': accuracy,
            'attack_success_rate': 100.0 - accuracy,
            'total_samples': total,
            'time_seconds': elapsed_time
        }
    
    def __repr__(self) -> str:
        """String representation of C&W attack."""
        return (f"CWAttack(c={self.c}, lr={self.lr}, "
                f"max_iter={self.max_iter}, device='{self.device}')")
