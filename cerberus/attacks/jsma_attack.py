"""JSMA (Jacobian Saliency Map Attack) Implementation.

JSMA is a targeted attack that modifies only a few pixels by analyzing
the model's Jacobian to identify the most influential pixels for
misclassification.

Reference: Papernot et al. (2015) - "The Limitations of Deep Learning in 
Adversarial Settings" (https://arxiv.org/abs/1511.04508)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import time


class JSMAAttack:
    """
    JSMA (Jacobian Saliency Map Attack).
    
    Analyzes model Jacobian to find most influential pixels.
    Modifies only few pixels but with larger magnitude.
    
    Different from FGSM: modifies few pixels vs all pixels.
    
    Parameters:
        model: PyTorch neural network model
        theta: Perturbation magnitude per step (default 1.0)
        gamma: Constraint tightness (default 0.1)
        max_pixels: Maximum number of pixels to modify (default 100)
        device: 'cpu' or 'cuda'
    
    Example:
        >>> jsma = JSMAAttack(model, theta=1.0, max_pixels=100)
        >>> adv_examples = jsma.generate_adversarial(images, labels)
        >>> results = jsma.evaluate(test_loader)
    """
    
    def __init__(self,
                 model: nn.Module,
                 theta: float = 1.0,
                 gamma: float = 0.1,
                 max_pixels: int = 100,
                 device: str = "cpu"):
        """Initialize JSMA Attack."""
        self.model = model.to(device)
        self.theta = theta
        self.gamma = gamma
        self.max_pixels = max_pixels
        self.device = device
        
    def generate_adversarial(self,
                           images: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
        """
        Generate JSMA adversarial examples.
        
        Args:
            images: Clean input images [batch_size, channels, height, width]
            labels: Target labels for misclassification [batch_size]
            
        Returns:
            Adversarial examples with same shape as input
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size = images.size(0)
        
        x_adv = images.clone().detach()
        self.model.eval()
        
        for i in range(batch_size):
            img = images[i:i+1].clone().detach()
            label = labels[i:i+1]
            
            # Compute Jacobian for this sample
            img.requires_grad = True
            outputs = self.model(img)
            
            # Get saliency scores for each output
            jacobian = torch.zeros(
                outputs.size(1),
                img.numel(),
                device=self.device
            )
            
            for class_idx in range(outputs.size(1)):
                self.model.zero_grad()
                if img.grad is not None:
                    img.grad.zero_()
                
                outputs = self.model(img)
                outputs[0, class_idx].backward(retain_graph=True)
                
                if img.grad is not None:
                    jacobian[class_idx] = img.grad.flatten().clone()
            
            # Compute saliency map
            target_idx = (label.item() + 1) % outputs.size(1)
            saliency = torch.abs(
                jacobian[target_idx] - jacobian[label.item()]
            )
            
            # Find most influential pixels
            if saliency.numel() > self.max_pixels:
                _, indices = torch.topk(saliency, self.max_pixels)
            else:
                indices = torch.arange(saliency.numel(), device=self.device)
            
            # Modify top pixels
            img_adv = img.clone().detach()
            img_flat = img_adv.reshape(-1)
            img_flat[indices] = torch.clamp(
                img_flat[indices] + self.theta / 255.0,
                0, 1
            )
            img_adv = img_flat.reshape(img.shape)
            
            x_adv[i] = torch.clamp(img_adv, 0, 1)
        
        return x_adv.detach()
    
    def evaluate(self,
                test_loader: Any) -> Dict[str, Any]:
        """
        Evaluate model robustness against JSMA attack.
        
        Args:
            test_loader: DataLoader for test set
            
        Returns:
            Dictionary with results
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
            'attack_type': 'JSMA',
            'max_pixels_modified': self.max_pixels,
            'theta': self.theta,
            'accuracy': accuracy,
            'attack_success_rate': 100.0 - accuracy,
            'total_samples': total,
            'time_seconds': elapsed_time
        }
    
    def __repr__(self) -> str:
        """String representation of JSMA attack."""
        return (f"JSMAAttack(theta={self.theta}, "
                f"max_pixels={self.max_pixels}, device='{self.device}')")
