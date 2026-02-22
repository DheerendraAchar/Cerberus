# Cerberus API Documentation
## Complete User Guide and API Reference

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Core API Reference](#core-api-reference)
4. [Attack Algorithms](#attack-algorithms)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Examples](#examples)
8. [Advanced Usage](#advanced-usage)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/DheerendraAchar/Cerberus.git
cd Cerberus

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Your First Adversarial Attack

```python
import torch
from cerberus.attacks import FSGMAttack
from cerberus.models import load_model
from torchvision import datasets, transforms

# Load model and data
model = load_model("resnet18", pretrained=True)
data = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
images, labels = data[0:32]  # Load batch

# Generate adversarial examples
attack = FSGMAttack(epsilon=8/255)
adversarial_examples = attack.generate(images, labels)

# Evaluate
clean_output = model(images)
adv_output = model(adversarial_examples)
print(f"Clean accuracy: {(clean_output.argmax(1) == labels).float().mean():.2%}")
print(f"Adversarial accuracy: {(adv_output.argmax(1) == labels).float().mean():.2%}")
```

---

## Installation

### System Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, CPU supported)
- 2GB RAM minimum

### Full Installation

```bash
# Create virtual environment (recommended)
python -m venv cerberus_env
source cerberus_env/bin/activate  # On Windows: cerberus_env\Scripts\activate

# Clone and install
git clone https://github.com/DheerendraAchar/Cerberus.git
cd Cerberus
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```python
import cerberus
print(cerberus.__version__)

# Check all attack types available
from cerberus.attacks import FSGMAttack, PGDAttack, CWAttack, DeepFoolAttack, JSMAAttack
print("✓ All attacks imported successfully")
```

---

## Core API Reference

### Base Attack Class

All attacks inherit from `BaseAttack`:

```python
class BaseAttack:
    """
    Base class for all adversarial attacks.
    
    Attributes:
        device (torch.device): Computation device (cpu/cuda)
        epsilon (float): Perturbation budget
    """
    
    def generate(self, images: torch.Tensor, labels: torch.Tensor, 
                 model: torch.nn.Module) -> torch.Tensor:
        """
        Generate adversarial examples.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            labels: True labels [batch_size]
            model: Target model for attack
            
        Returns:
            Adversarial examples with same shape as input
            
        Raises:
            ValueError: If input shapes don't match
            RuntimeError: If model is in training mode
        """
        pass
    
    def evaluate(self, model: torch.nn.Module, 
                 images: torch.Tensor, 
                 labels: torch.Tensor) -> dict:
        """
        Evaluate attack success rate.
        
        Returns:
            {
                'success_rate': float,      # % of successful attacks
                'avg_perturbation': float,  # Average L∞ perturbation
                'misclassified': int,       # Number of misclassifications
            }
        """
        pass
```

### Available Attack Classes

```python
from cerberus.attacks import (
    FSGMAttack,      # Fast Gradient Sign Method
    PGDAttack,       # Projected Gradient Descent
    CWAttack,        # Carlini & Wagner
    DeepFoolAttack,  # DeepFool
    JSMAAttack       # Jacobian Saliency Map Attack
)
```

---

## Attack Algorithms

### 1. FGSM - Fast Gradient Sign Method

**Best for:** Quick attacks, baseline comparison

```python
from cerberus.attacks import FSGMAttack

attack = FSGMAttack(
    epsilon=8/255,      # Perturbation budget
    device='cuda'        # Computation device
)

# Single batch
adversarial = attack.generate(images, labels, model)

# Evaluate
results = attack.evaluate(model, images, labels)
print(f"Success rate: {results['success_rate']:.2%}")
```

**Complexity:** O(1) forward passes per image  
**Strength:** Weak-medium (single gradient step)  
**Use case:** Baseline, quick benchmarking

---

### 2. PGD - Projected Gradient Descent

**Best for:** Strong attacks, adversarial training

```python
from cerberus.attacks import PGDAttack

attack = PGDAttack(
    epsilon=8/255,           # Perturbation budget
    steps=20,                # Iteration count
    step_size=2/255,         # Step size per iteration
    random_start=True,       # Random initialization
    device='cuda'
)

adversarial = attack.generate(images, labels, model)
```

**Complexity:** O(steps) forward passes per image  
**Strength:** Very strong (multi-step iterative)  
**Use case:** Adversarial training, robustness testing

---

### 3. C&W - Carlini & Wagner

**Best for:** Strongest white-box attacks

```python
from cerberus.attacks import CWAttack

attack = CWAttack(
    epsilon=8/255,           # Perturbation budget
    learning_rate=0.01,      # Optimization LR
    steps=100,               # Max optimization steps
    confidence=0,            # Confidence margin
    device='cuda'
)

adversarial = attack.generate(images, labels, model)
```

**Complexity:** O(steps) optimization iterations  
**Strength:** Extremely strong (optimization-based)  
**Use case:** Breaking defenses, finding minimum perturbations

---

### 4. DeepFool

**Best for:** Minimal perturbations

```python
from cerberus.attacks import DeepFoolAttack

attack = DeepFoolAttack(
    overshoot=0.02,          # Overshoot parameter
    max_iter=50,             # Maximum iterations
    device='cuda'
)

adversarial = attack.generate(images, labels, model)
results = attack.evaluate(model, images, labels)
print(f"Avg perturbation: {results['avg_perturbation']:.6f}")
```

**Complexity:** O(iterations) forward passes  
**Strength:** Medium-strong (boundary-seeking)  
**Use case:** Minimal perturbation analysis

---

### 5. JSMA - Jacobian Saliency Map Attack

**Best for:** Targeted attacks, feature analysis

```python
from cerberus.attacks import JSMAAttack

attack = JSMAAttack(
    theta=0.1,               # Perturbation threshold
    gamma=0.15,              # Saliency threshold
    device='cuda'
)

# Targeted attack
target_labels = torch.ones_like(labels)  # Target class
adversarial = attack.generate(
    images, labels, model, 
    target=target_labels
)
```

**Complexity:** O(1) Jacobian computation  
**Strength:** Medium (feature-targeted)  
**Use case:** Targeted attacks, interpretability

---

## Training

### Basic Adversarial Training

```python
from cerberus.training import AdversarialTrainer
from torch.utils.data import DataLoader

# Setup
trainer = AdversarialTrainer(
    model=model,
    attack_type='fgsm',      # 'fgsm', 'pgd', 'cw', 'deepfool', 'jsma'
    epsilon=8/255,
    alpha=0.5,               # Mix ratio: 50% adversarial + 50% clean
    epochs=100,
    batch_size=128,
    learning_rate=0.1
)

# Train
history = trainer.train(train_loader, val_loader)

# Save model
trainer.save_checkpoint("adversarial_model.pth")
```

### Custom Training Loop

```python
from cerberus.attacks import PGDAttack
from torch.optim import SGD
import torch.nn as nn

model = model.train()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
attack = PGDAttack(epsilon=8/255, steps=7)

for epoch in range(100):
    for images, labels in train_loader:
        # Random choice: clean or adversarial
        if torch.rand(1) > 0.5:
            # Adversarial training
            with torch.enable_grad():
                adversarial = attack.generate(images, labels, model)
            outputs = model(adversarial)
        else:
            # Clean training
            outputs = model(images)
        
        # Update
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Evaluation

### Individual Model Evaluation

```python
from cerberus.evaluation import RobustnessEvaluator

evaluator = RobustnessEvaluator(model, device='cuda')

# Evaluate against single attack
results = evaluator.evaluate_single_attack(
    images=test_images,
    labels=test_labels,
    attack_type='pgd',
    epsilon=8/255
)

print(f"Clean accuracy: {results['clean_acc']:.2%}")
print(f"Adversarial accuracy: {results['adv_acc']:.2%}")
print(f"Robustness: {results['robustness_gain']:.2%}")
```

### Multi-Attack Comparison

```python
# Evaluate against multiple attacks
attacks = ['fgsm', 'pgd', 'cw', 'deepfool', 'jsma']
comparison = evaluator.compare_attacks(
    images=test_images,
    labels=test_labels,
    attacks=attacks,
    epsilon=8/255
)

# Results
for attack_name, results in comparison.items():
    print(f"{attack_name:10s} -> Adv Acc: {results['adv_acc']:.2%}")
```

### Transfer Attack Analysis

```python
from cerberus.evaluation import TransferAnalyzer

analyzer = TransferAnalyzer()

# Generate transfer matrix
transfer_matrix = analyzer.compute_transfer_matrix(
    models={'resnet18': model1, 'vgg16': model2, ...},
    images=test_images,
    labels=test_labels,
    attack_type='fgsm',
    epsilon=8/255
)

# Analyze
summary = analyzer.summarize(transfer_matrix)
print(f"Self-attack rate (diagonal): {summary['self_rate']:.2%}")
print(f"Transfer rate (off-diagonal): {summary['transfer_rate']:.2%}")
```

---

## Examples

### Example 1: Complete Pipeline

```python
import torch
from cerberus.attacks import FSGMAttack, PGDAttack
from cerberus.evaluation import RobustnessEvaluator
from cerberus.training import AdversarialTrainer

# 1. Load data and model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
train_loader = get_train_loader()  # Your loader
val_loader = get_val_loader()

# 2. Train adversarially
trainer = AdversarialTrainer(model, attack_type='pgd', epochs=100)
trainer.train(train_loader, val_loader)

# 3. Evaluate robustness
evaluator = RobustnessEvaluator(model)
results = evaluator.evaluate_single_attack(test_images, test_labels, 'pgd')

# 4. Report
print(f"""
Clean Accuracy:      {results['clean_acc']:.2%}
Adversarial Accuracy: {results['adv_acc']:.2%}
Robustness Gain:     {results['robustness_gain']:.2%}
""")
```

### Example 2: Adversarial Attack Comparison

```python
from cerberus.attacks import (
    FSGMAttack, PGDAttack, CWAttack, 
    DeepFoolAttack, JSMAAttack
)

attacks = [
    ('FGSM', FSGMAttack(epsilon=8/255)),
    ('PGD', PGDAttack(epsilon=8/255, steps=20)),
    ('C&W', CWAttack(epsilon=8/255, steps=100)),
    ('DeepFool', DeepFoolAttack()),
    ('JSMA', JSMAAttack())
]

print(f"{'Attack':<10} {'Success Rate':<15} {'Avg Perturbation':<20}")
print("-" * 45)

for name, attack in attacks:
    results = attack.evaluate(model, images, labels)
    print(f"{name:<10} {results['success_rate']:>13.2%} {results['avg_perturbation']:>18.6f}")
```

### Example 3: Transfer Attack Visualization

```python
import matplotlib.pyplot as plt
from cerberus.evaluation import TransferAnalyzer

# Compute transfer matrix
analyzer = TransferAnalyzer()
models = {
    'ResNet-18': resnet_model,
    'VGG-16': vgg_model,
    'MobileNet V2': mobilenet_model
}

matrix = analyzer.compute_transfer_matrix(models, test_images, test_labels)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(matrix, cmap='RdYlGn')
ax.set_xticks(range(len(models)))
ax.set_yticks(range(len(models)))
ax.set_xticklabels(models.keys())
ax.set_yticklabels(models.keys())
ax.set_xlabel('Target Model')
ax.set_ylabel('Source Model')
plt.colorbar(im, ax=ax, label='Success Rate (%)')
plt.tight_layout()
plt.show()
```

---

## Advanced Usage

### Custom Attack Implementation

```python
from cerberus.attacks import BaseAttack
import torch
import torch.nn.functional as F

class CustomAttack(BaseAttack):
    """Your custom adversarial attack."""
    
    def __init__(self, epsilon=8/255, custom_param=0.5, device='cpu'):
        super().__init__(epsilon=epsilon, device=device)
        self.custom_param = custom_param
    
    def generate(self, images, labels, model):
        """Generate adversarial examples."""
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Your attack logic
        perturbation = self._compute_perturbation(images, labels, model)
        adversarial = torch.clamp(
            images + perturbation,
            min=0, max=1
        )
        return adversarial
    
    def _compute_perturbation(self, images, labels, model):
        """Compute custom perturbation."""
        # Implementation here
        pass

# Use your custom attack
custom_attack = CustomAttack(epsilon=8/255, custom_param=0.7)
adversarial = custom_attack.generate(images, labels, model)
```

### Batch Processing Large Datasets

```python
from cerberus.utils import batch_generator

# Process in chunks
for batch_idx, (images, labels) in enumerate(batch_generator(dataset, batch_size=1000)):
    # Generate adversarial
    adversarial = attack.generate(images, labels, model)
    
    # Evaluate
    predictions = model(adversarial).argmax(1)
    success = (predictions != labels).sum().item()
    
    print(f"Batch {batch_idx}: {success/len(labels):.2%} attack success")
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group("nccl")
model = DistributedDataParallel(model)

# Training proceeds as normal
trainer = AdversarialTrainer(model)
trainer.train(train_loader, val_loader)
```

---

## Configuration

### YAML Configuration Files

Create `config.yaml`:

```yaml
# Model Configuration
model:
  architecture: "resnet18"
  pretrained: true
  num_classes: 10

# Dataset Configuration
dataset:
  name: "cifar10"
  root: "./data"
  train_split: 0.8
  
# Attack Configuration
attack:
  type: "pgd"           # fgsm, pgd, cw, deepfool, jsma
  epsilon: 0.0314       # 8/255
  iterations: 20
  step_size: 0.0078     # 2/255

# Training Configuration
training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0001
  
  # Adversarial training
  adversarial:
    enabled: true
    mix_ratio: 0.5      # 50% clean, 50% adversarial

# Evaluation Configuration
evaluation:
  test_batch_size: 256
  num_workers: 4
  
# Checkpoint Configuration
checkpoint:
  save_dir: "./checkpoints"
  save_freq: 10  # every 10 epochs
  keep_best: true
```

Load in Python:

```python
from cerberus.utils import load_config

config = load_config('config.yaml')
model = build_model(config.model)
trainer = AdversarialTrainer(model, config=config.training)
```

---

## Troubleshooting

### Out of Memory (OOM)

**Problem:** CUDA out of memory error

**Solutions:**
1. Reduce batch size
2. Use gradient accumulation
3. Switch to CPU mode
4. Use model quantization

```python
# Reduce batch size
trainer = AdversarialTrainer(model, batch_size=32)  # Instead of 128

# Or use gradient accumulation
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Slow Training

**Problem:** Adversarial training is very slow

**Solutions:**
1. Use faster attack (FGSM instead of PGD)
2. Reduce attack iterations
3. Use multiple GPUs
4. Enable mixed precision training

```python
# Fast but weak attack
fast_attack = FSGMAttack(epsilon=8/255)  # O(1) cost

# Slow but strong attack
strong_attack = PGDAttack(epsilon=8/255, steps=20)  # O(20) cost
```

### Model Not Learning

**Problem:** Training loss doesn't decrease

**Possible causes:**
- Learning rate too high/low
- Model gradient issue
- Batch normalization in training

**Solution:**
```python
# Check learning rate
for param_group in optimizer.param_groups:
    print(f"Current LR: {param_group['lr']}")

# Set model to train mode
model.train()

# Use appropriate batch norm behavior
model = model.train() if training else model.eval()
```

### Inconsistent Results

**Problem:** Different results on different runs

**Solution:** Set random seeds

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## FAQ

**Q: What's the difference between FGSM and PGD?**
A: FGSM is one-step (fast but weak), PGD is multi-step (slow but stronger). Use FGSM for quick tests, PGD for robust training.

**Q: Should I use adversarial training?**
A: Yes, if security/robustness is important. It costs ~2x training time but improves robustness ~50%.

**Q: What epsilon should I use?**
A: ε=8/255 is standard for CIFAR-10. For ImageNet, ε=4/255 is typical. Adjust based on your security requirements.

**Q: Can I use adversarial training with my own model?**
A: Yes, as long as your model is a PyTorch `nn.Module`.

**Q: What's the computational cost of different attacks?**
A: FGSM: 1x, PGD (20 steps): 20x, C&W (100 steps): 100x, DeepFool: ~10x, JSMA: ~5x

---

## Citation

If you use Cerberus in your research, please cite:

```bibtex
@inproceedings{Cerberus2026,
  title={Cerberus: A Comprehensive Framework for Adversarial Machine Learning Training and Transfer Attack Analysis},
  author={Achar, BD and Sharma, C and Bhandare, G and Kapoor, C},
  booktitle={Proc. IEEE Symposium Series on Computational Intelligence (SSCI)},
  year={2026}
}
```

---

## Support

- **GitHub Issues:** https://github.com/DheerendraAchar/Cerberus/issues
- **Email:** cerberus@dsu.edu.in
- **Documentation:** Full docs at https://cerberus.dsu.edu.in

---

*Last Updated: February 2026*
*Version: 1.0*
