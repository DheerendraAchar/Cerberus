#!/usr/bin/env python3
"""Fast Gradient Sign Method (FGSM) attack implementation.

Paper: "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2015)
Link: https://arxiv.org/abs/1412.6572

FGSM is the simplest and fastest adversarial attack, requiring only a single
gradient step. It's commonly used as a baseline for adversarial training.

Mathematical Formulation:
    x_adv = x + ε * sign(∇_x L(θ, x, y))

Where:
    - x: input image
    - ε: perturbation magnitude (maximum allowed)
    - L: loss function (e.g., cross-entropy)
    - ∇_x: gradient with respect to input
    - sign(): element-wise sign function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import time


class FSGMAttack:
    """Fast Gradient Sign Method (FGSM) Attack.
    
    Attributes:
        model: Target neural network
        eps: Perturbation magnitude (maximum L∞ norm)
        device: Computation device ('cpu' or 'cuda')
    """
    
    def __init__(self, model: nn.Module, eps: float = 0.03, device: str = "cpu"):
        """Initialize FGSM attack.
        
        Args:
            model: Target neural network
            eps: Perturbation magnitude (default 0.03 for CIFAR-10)
            device: Computation device
        """
        self.model = model
        self.eps = eps
        self.device = device
        self.model.eval()
    
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using FGSM.
        
        Args:
            x: Input images (shape: [batch_size, 3, 32, 32])
            y: True labels (shape: [batch_size])
        
        Returns:
            Adversarial images (same shape as input)
        """
        x.requires_grad = True
        
        # Forward pass
        outputs = self.model(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        
        # Backward pass to compute gradients
        self.model.zero_grad()
        loss.backward()
        
        # FGSM perturbation: x_adv = x + eps * sign(grad)
        data_grad = x.grad.data
        perturbed_x = x + self.eps * data_grad.sign()
        
        # Clip to valid range [0, 1]
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        
        return perturbed_x.detach()
    
    def evaluate(self, test_loader) -> Dict:
        """Evaluate FGSM attack on test dataset.
        
        Args:
            test_loader: DataLoader with test images and labels
        
        Returns:
            Dictionary with metrics:
                - accuracy: Clean accuracy on original images
                - attack_success_rate: % of misclassified adversarial examples
                - avg_perturbation: Mean L∞ perturbation
                - time_seconds: Total attack time
        """
        self.model.eval()
        
        correct_clean = 0
        correct_adv = 0
        total = 0
        total_perturbation = 0.0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (x_test, y_test) in enumerate(test_loader):
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                
                batch_size = x_test.size(0)
                
                # Clean accuracy
                outputs_clean = self.model(x_test)
                _, predicted_clean = outputs_clean.max(1)
                correct_clean += predicted_clean.eq(y_test).sum().item()
                
                # Adversarial accuracy
                x_adv = self.generate(x_test, y_test)
                outputs_adv = self.model(x_adv)
                _, predicted_adv = outputs_adv.max(1)
                correct_adv += predicted_adv.eq(y_test).sum().item()
                
                # Perturbation magnitude (L∞)
                perturbation = torch.abs(x_adv - x_test).max().item()
                total_perturbation += perturbation
                
                total += batch_size
        
        elapsed = time.time() - start_time
        
        clean_accuracy = 100. * correct_clean / total
        adv_accuracy = 100. * correct_adv / total
        attack_success_rate = 100. - adv_accuracy
        avg_perturbation = total_perturbation / len(test_loader)
        
        return {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': attack_success_rate,
            'avg_perturbation': avg_perturbation,
            'epsilon': self.eps,
            'time_seconds': elapsed,
            'attack_type': 'FGSM'
        }


def main():
    """Example usage of FGSM attack."""
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=None)
    model = model.to(device)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    
    # Run attack
    attack = FSGMAttack(model=model, eps=0.03, device=device)
    results = attack.evaluate(test_loader)
    
    # Print results
    print("FGSM Attack Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
