# Training Components for Project Cerberus
## Making It a Complete ML Project (Not Just Using Existing Tools)

**Created:** December 27, 2025  
**Purpose:** Add model training to demonstrate ML engineering skills, not just tool usage

---

## 🎯 CRITICAL ADDITIONS NEEDED

Your project currently **LACKS** these essential ML components:

### ❌ **What's Missing:**
1. **No model training from scratch**
2. **No adversarial retraining (defense)**  
3. **No fine-tuning or transfer learning**
4. **No custom optimization algorithms**
5. **No learning curves or training metrics**

### ✅ **What You Need to Add:**

---

## 🏋️ **OPTION 1: Adversarial Training (ESSENTIAL)**
**Time:** 2-3 weeks  
**Impact:** ⭐⭐⭐⭐⭐⭐ (Core defense mechanism)

### **What It Is:**
Train your model on **adversarial examples** to make it robust to attacks.

### **Implementation:**

```python
# New file: cerberus/adversarial_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier


class AdversarialTrainer:
    """Train models with adversarial examples for improved robustness."""
    
    def __init__(
        self, 
        model: nn.Module,
        train_loader: Any,
        test_loader: Any,
        device: str = "cpu",
        epsilon: float = 0.03,
        alpha: float = 0.5  # Mix ratio: clean vs adversarial
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup ART classifier for generating adversarial examples
        self.art_classifier = PyTorchClassifier(
            model=model,
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
            device_type=device
        )
        self.attack = FastGradientMethod(estimator=self.art_classifier, eps=epsilon)
        
        # Track metrics
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_clean_acc': [],
            'test_adv_acc': []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with mixed clean + adversarial examples."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Generate adversarial examples for this batch
            inputs_np = inputs.cpu().numpy()
            adv_inputs_np = self.attack.generate(x=inputs_np)
            adv_inputs = torch.from_numpy(adv_inputs_np).to(self.device)
            
            # Mix clean and adversarial examples
            batch_size = inputs.size(0)
            num_clean = int(batch_size * self.alpha)
            num_adv = batch_size - num_clean
            
            mixed_inputs = torch.cat([inputs[:num_clean], adv_inputs[:num_adv]], dim=0)
            mixed_labels = torch.cat([labels[:num_clean], labels[:num_adv]], dim=0)
            
            # Shuffle the mixed batch
            perm = torch.randperm(batch_size)
            mixed_inputs = mixed_inputs[perm]
            mixed_labels = mixed_labels[perm]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(mixed_inputs)
            loss = self.criterion(outputs, mixed_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += mixed_labels.size(0)
            correct += predicted.eq(mixed_labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}%")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def evaluate(self, adversarial: bool = False) -> float:
        """Evaluate model on clean or adversarial test set."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                if adversarial:
                    # Generate adversarial examples
                    inputs_np = inputs.cpu().numpy()
                    adv_inputs_np = self.attack.generate(x=inputs_np)
                    inputs = torch.from_numpy(adv_inputs_np).to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, num_epochs: int = 10) -> Dict[str, list]:
        """Train model for multiple epochs and track all metrics."""
        print(f"\n{'='*60}")
        print(f"Starting Adversarial Training")
        print(f"Epochs: {num_epochs} | Epsilon: {self.epsilon} | Alpha: {self.alpha}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Evaluate
            clean_acc = self.evaluate(adversarial=False)
            adv_acc = self.evaluate(adversarial=True)
            self.history['test_clean_acc'].append(clean_acc)
            self.history['test_adv_acc'].append(adv_acc)
            
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc:  {train_metrics['accuracy']:.2f}%")
            print(f"  Test Clean: {clean_acc:.2f}%")
            print(f"  Test Adv:   {adv_acc:.2f}%")
            print(f"  Robustness: {adv_acc/clean_acc*100:.1f}%")
            print("-" * 60)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Final Clean Accuracy:       {self.history['test_clean_acc'][-1]:.2f}%")
        print(f"Final Adversarial Accuracy: {self.history['test_adv_acc'][-1]:.2f}%")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'epsilon': self.epsilon,
            'alpha': self.alpha
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"✅ Model loaded from {path}")
        return checkpoint


# CLI integration
def run_adversarial_training_from_config(config_path: str):
    """Run adversarial training based on config file."""
    from .config import load_config
    from .model import load_pytorch_model
    from .dataset import get_cifar10_loaders
    
    cfg = load_config(config_path)
    
    # Load model
    model_path = cfg.get('model', {}).get('path')
    if model_path:
        model = load_pytorch_model(model_path)
    else:
        # Use TinyCNN as starting point
        import torch.nn as nn
        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 10)
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = TinyCNN()
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(
        root=cfg.get('dataset', {}).get('root', './data'),
        batch_size=cfg.get('dataset', {}).get('batch_size', 64)
    )
    
    # Setup trainer
    trainer = AdversarialTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device='cpu',
        epsilon=cfg.get('training', {}).get('epsilon', 0.03),
        alpha=cfg.get('training', {}).get('alpha', 0.5)
    )
    
    # Train
    num_epochs = cfg.get('training', {}).get('epochs', 10)
    history = trainer.train(num_epochs=num_epochs)
    
    # Save
    output_path = cfg.get('training', {}).get('model_save_path', 'outputs/adversarially_trained_model.pt')
    trainer.save_model(output_path)
    
    return history
```

### **Config File Addition:**

```yaml
# configs/adversarial_training_config.yaml
model:
  path: null  # Start with untrained TinyCNN, or provide path to pre-trained model

dataset:
  name: cifar10
  root: ./data
  batch_size: 128
  num_workers: 2

training:
  epochs: 20
  epsilon: 0.03        # Adversarial perturbation strength
  alpha: 0.5           # Mix ratio: 0.5 = 50% clean, 50% adversarial
  learning_rate: 0.01
  momentum: 0.9
  model_save_path: outputs/adversarially_trained_model.pt

attack:
  name: fgsm
  eps: 0.03

output:
  report_path: outputs/adversarial_training_report.html
  figures_dir: figures/training/
```

### **CLI Command:**

```bash
# New command in run_demo.py
python run_demo.py --mode train --config configs/adversarial_training_config.yaml
```

---

## 📊 **OPTION 2: Train TinyCNN from Scratch**
**Time:** 1-2 weeks  
**Impact:** ⭐⭐⭐⭐ (Shows basic ML training skills)

### **Why:**
Currently your TinyCNN has **random weights**. Train it properly on CIFAR-10!

### **Implementation:**

```python
# New file: cerberus/baseline_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any


class BaselineTrainer:
    """Train baseline models from scratch on clean data."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        test_loader: Any,
        device: str = "cpu",
        learning_rate: float = 0.01
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}%')
        
        return {
            'loss': train_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
    
    def test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        test_loss = 0
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
        
        return {
            'loss': test_loss / len(self.test_loader),
            'accuracy': 100. * correct / total
        }
    
    def train(self, num_epochs: int = 100) -> Dict[str, list]:
        """Train for multiple epochs."""
        print(f"\n{'='*60}")
        print(f"Starting Baseline Training")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*60}\n")
        
        best_acc = 0
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Test
            test_metrics = self.test()
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['test_acc'].append(test_metrics['accuracy'])
            
            # Update learning rate
            self.scheduler.step()
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Test  Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']:.2f}%")
            
            # Save best model
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                self.save_model('outputs/best_baseline_model.pt')
                print(f"  ✅ New best accuracy: {best_acc:.2f}%")
            print("-" * 60)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Test Accuracy: {best_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)
```

---

## 🎓 **OPTION 3: Curriculum Adversarial Training**
**Time:** 3-4 weeks  
**Impact:** ⭐⭐⭐⭐⭐⭐ (NOVEL! Research-grade!)

### **Why This is Innovative:**
- Standard adversarial training uses **fixed epsilon**
- Your approach **adapts** epsilon based on model performance
- Similar to curriculum learning (easy → hard tasks)
- **VERY FEW PAPERS** on this specific approach

### **Key Innovation:**
```python
# Progressive epsilon schedule
epoch 1-5:   epsilon = 0.01  (easy attacks)
epoch 6-10:  epsilon = 0.02  (medium attacks)
epoch 11-15: epsilon = 0.03  (hard attacks)
epoch 16-20: epsilon = 0.05  (very hard attacks)

# Adaptive based on performance
if model_accuracy > 85%:
    epsilon *= 1.1  # Increase difficulty
elif model_accuracy < 70%:
    epsilon *= 0.9  # Decrease difficulty
```

**(See INNOVATION_IDEAS.md for full implementation)**

---

## 📈 **Comparison: Before vs After Training**

### **Current (Phase 1):**
```
Load pre-trained ResNet-50 → Test with FGSM → Report results
```
**Problem:** You didn't build anything ML-related!

### **With Training (Phase 2):**
```
1. Train TinyCNN from scratch (60-70% accuracy) [BASELINE]
2. Test with FGSM (accuracy drops to 30-40%)    [ATTACK]
3. Adversarial retraining (recover to 55-65%)   [DEFENSE]
4. Compare all three                            [ANALYSIS]
```
**Now you have:** ML pipeline, training skills, novel defense!

---

## 🎯 **Recommended Timeline**

### **Week 1-2: Baseline Training**
- Train TinyCNN from scratch on CIFAR-10
- Achieve 60-70% test accuracy
- Plot training curves (loss, accuracy)

### **Week 3-5: Adversarial Training**
- Implement adversarial training loop
- Mix clean + adversarial examples
- Show robustness improvement

### **Week 6-8: Curriculum Training (Optional)**
- Implement adaptive epsilon schedule
- Compare with standard adversarial training
- Show improved generalization

### **Week 9-10: Evaluation & Documentation**
- Comprehensive benchmarks
- Training curves visualization
- Write research report section

---

## 🏆 **Why This Makes Your Project Stand Out**

### **Without Training:**
❌ "We used existing tools (ART, pre-trained models)"  
❌ "Framework for testing adversarial attacks"  
❌ Limited originality

### **With Training:**
✅ "We implemented adversarial training from scratch"  
✅ "Novel curriculum learning approach"  
✅ "Improved robustness by 25%"  
✅ **Complete ML pipeline: data → train → attack → defend**

---

## 📊 **Deliverables After Adding Training**

1. **Trained Models:**
   - `baseline_tinycnn.pt` (60-70% clean accuracy)
   - `adversarially_trained_tinycnn.pt` (55-65% clean, 45-55% adversarial)
   - `curriculum_trained_tinycnn.pt` (65-75% clean, 50-60% adversarial)

2. **Training Curves:**
   - Loss over epochs
   - Accuracy over epochs
   - Robustness improvement over time

3. **Comparison Table:**
   ```
   Model                    Clean Acc  Adv Acc (ε=0.03)  Robustness
   ───────────────────────────────────────────────────────────────
   Baseline (untrained)     10%        10%               100%
   Baseline (trained)       68%        32%               47%
   Adversarial Training     62%        48%               77%
   Curriculum Training      70%        55%               79%
   ```

4. **Code:**
   - `cerberus/baseline_training.py` (200 lines)
   - `cerberus/adversarial_training.py` (300 lines)
   - `cerberus/curriculum_training.py` (400 lines - NOVEL!)

---

## 🚀 **Next Steps**

1. **Choose your training approach:**
   - Minimum: Baseline + Adversarial Training (Options 1 & 2)
   - Maximum: Add Curriculum Training (Option 3) for novelty

2. **Implement training loop:**
   - Start with baseline training (simplest)
   - Add adversarial training next
   - Curriculum training last (if time permits)

3. **Document everything:**
   - Training hyperparameters
   - Learning curves
   - Comparison metrics

4. **Create visualizations:**
   - Training curves
   - Before/after robustness
   - Epsilon vs accuracy plots

---

**Bottom line:** Your project currently just **uses** existing ML models and tools. Adding training components shows you can **build** ML systems, not just use them! 🎓

*Want me to help implement any of these training modules? Just pick one!*
