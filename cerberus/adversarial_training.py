"""Adversarial training for improving model robustness.

This module provides training pipelines that incorporate adversarial examples
during training to improve model robustness against adversarial attacks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional
import time


class AdversarialTrainer:
    """Train models with adversarial examples for improved robustness.
    
    This trainer implements adversarial training by mixing clean and adversarial
    examples during training. Adversarial examples are generated on-the-fly using
    FGSM attack.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to train on ('cpu' or 'cuda')
        epsilon: Perturbation magnitude for FGSM attack
        alpha: Mix ratio for clean vs adversarial examples (0.5 = 50% each)
        learning_rate: Learning rate for optimizer
        momentum: Momentum for SGD optimizer
        weight_decay: Weight decay for regularization
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        test_loader: Any,
        device: str = "cpu",
        epsilon: float = 0.03,
        alpha: float = 0.5,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Cosine annealing learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )
        
        # Track training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_clean_acc': [],
            'test_adv_acc': [],
            'learning_rate': []
        }
    
    def _generate_adversarial_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using FGSM.
        
        Args:
            inputs: Clean input images
            labels: True labels
            
        Returns:
            Adversarial examples
        """
        self.model.eval()  # Set to eval mode for generating adversarial examples
        
        inputs.requires_grad = True
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        
        # Compute gradients
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial perturbation
        data_grad = inputs.grad.data
        perturbed_data = inputs + self.epsilon * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        self.model.train()  # Set back to train mode
        return perturbed_data.detach()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with mixed clean and adversarial examples.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training loss and accuracy
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Generate adversarial examples for this batch
            adv_inputs = self._generate_adversarial_batch(inputs, labels)
            
            # Mix clean and adversarial examples
            batch_size = inputs.size(0)
            num_clean = int(batch_size * self.alpha)
            num_adv = batch_size - num_clean
            
            # Create mixed batch
            if num_clean > 0 and num_adv > 0:
                mixed_inputs = torch.cat([inputs[:num_clean], adv_inputs[num_clean:]], dim=0)
                mixed_labels = labels  # Labels stay the same
            elif num_clean == batch_size:
                mixed_inputs = inputs
                mixed_labels = labels
            else:
                mixed_inputs = adv_inputs
                mixed_labels = labels
            
            # Shuffle to avoid model learning clean/adv pattern
            perm = torch.randperm(batch_size)
            mixed_inputs = mixed_inputs[perm]
            mixed_labels = mixed_labels[perm]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(mixed_inputs)
            loss = self.criterion(outputs, mixed_labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += mixed_labels.size(0)
            correct += predicted.eq(mixed_labels).sum().item()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Batch [{batch_idx:3d}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:6.2f}%")
        
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"  Epoch time: {epoch_time:.1f}s")
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def evaluate(self, adversarial: bool = False, subset_size: Optional[int] = None) -> float:
        """Evaluate model on clean or adversarial test set.
        
        Args:
            adversarial: If True, evaluate on adversarial examples
            subset_size: If provided, only evaluate on first N examples
            
        Returns:
            Test accuracy as percentage
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                if subset_size and total >= subset_size:
                    break
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                if adversarial:
                    # Generate adversarial examples
                    inputs.requires_grad = True
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    self.model.zero_grad()
                    loss.backward()
                    data_grad = inputs.grad.data
                    inputs = inputs + self.epsilon * data_grad.sign()
                    inputs = torch.clamp(inputs, 0, 1)
                    inputs = inputs.detach()
                
                # Evaluate
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, num_epochs: int = 10, save_best: bool = True, save_path: str = "outputs/adversarially_trained_model.pt") -> Dict[str, List[float]]:
        """Train model for multiple epochs and track all metrics.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: If True, save model with best adversarial accuracy
            save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*70}")
        print(f"Starting Adversarial Training")
        print(f"{'='*70}")
        print(f"Epochs:       {num_epochs}")
        print(f"Epsilon:      {self.epsilon}")
        print(f"Alpha:        {self.alpha} (mix ratio)")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Device:       {self.device}")
        print(f"{'='*70}\n")
        
        best_adv_acc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Evaluate on clean test set
            print("  Evaluating on clean test set...")
            clean_acc = self.evaluate(adversarial=False)
            self.history['test_clean_acc'].append(clean_acc)
            
            # Evaluate on adversarial test set (use subset for speed)
            print("  Evaluating on adversarial test set...")
            adv_acc = self.evaluate(adversarial=True, subset_size=1000)
            self.history['test_adv_acc'].append(adv_acc)
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\n  Epoch {epoch} Summary:")
            print(f"    Train Loss:     {train_metrics['loss']:.4f}")
            print(f"    Train Acc:      {train_metrics['accuracy']:.2f}%")
            print(f"    Test Clean:     {clean_acc:.2f}%")
            print(f"    Test Adversarial: {adv_acc:.2f}%")
            print(f"    Robustness:     {adv_acc/clean_acc*100:.1f}% (adv/clean ratio)")
            print(f"    Learning Rate:  {current_lr:.6f}")
            
            # Save best model based on adversarial accuracy
            if save_best and adv_acc > best_adv_acc:
                best_adv_acc = adv_acc
                self.save_model(save_path)
                print(f"    ✅ New best adversarial accuracy: {best_adv_acc:.2f}%")
            
            print("=" * 70)
            print()
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Final Clean Accuracy:       {self.history['test_clean_acc'][-1]:.2f}%")
        print(f"Final Adversarial Accuracy: {self.history['test_adv_acc'][-1]:.2f}%")
        print(f"Best Adversarial Accuracy:  {best_adv_acc:.2f}%")
        print(f"Final Robustness Ratio:     {self.history['test_adv_acc'][-1]/self.history['test_clean_acc'][-1]*100:.1f}%")
        print(f"{'='*70}\n")
        
        return self.history
    
    def save_model(self, path: str):
        """Save model checkpoint with training state.
        
        Args:
            path: Path to save the checkpoint
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'epsilon': self.epsilon,
            'alpha': self.alpha
        }, path)
        print(f"    💾 Model saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint.
        
        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.alpha = checkpoint.get('alpha', self.alpha)
        print(f"✅ Model loaded from {path}")
        return checkpoint
