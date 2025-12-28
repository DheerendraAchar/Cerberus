"""Baseline model training on clean data.

This module provides standard training pipelines for training models from
scratch on clean (non-adversarial) data. Used as a baseline for comparison
with adversarial training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List
import time


class BaselineTrainer:
    """Train models from scratch on clean data.
    
    This trainer implements standard supervised learning for image classification.
    It serves as a baseline for comparison with adversarial training methods.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to train on ('cpu' or 'cuda')
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
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
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
            'test_loss': [],
            'test_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch on clean data.
        
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Batch [{batch_idx:3d}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:6.2f}%")
        
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"  Epoch time: {epoch_time:.1f}s")
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def test(self) -> Dict[str, float]:
        """Evaluate model on clean test set.
        
        Returns:
            Dictionary with test loss and accuracy
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, num_epochs: int = 100, save_best: bool = True, save_path: str = "outputs/baseline_model.pt") -> Dict[str, List[float]]:
        """Train model for multiple epochs and track all metrics.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: If True, save model with best test accuracy
            save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*70}")
        print(f"Starting Baseline Training")
        print(f"{'='*70}")
        print(f"Epochs:       {num_epochs}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Device:       {self.device}")
        print(f"{'='*70}\n")
        
        best_acc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Test
            test_metrics = self.test()
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['test_acc'].append(test_metrics['accuracy'])
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\n  Epoch {epoch} Summary:")
            print(f"    Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
            print(f"    Test  Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']:.2f}%")
            print(f"    Learning Rate: {current_lr:.6f}")
            
            # Save best model based on test accuracy
            if save_best and test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                self.save_model(save_path)
                print(f"    ✅ New best accuracy: {best_acc:.2f}%")
            
            print("=" * 70)
            print()
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Final Train Accuracy: {self.history['train_acc'][-1]:.2f}%")
        print(f"Final Test Accuracy:  {self.history['test_acc'][-1]:.2f}%")
        print(f"Best Test Accuracy:   {best_acc:.2f}%")
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
            'history': self.history
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
        print(f"✅ Model loaded from {path}")
        return checkpoint
