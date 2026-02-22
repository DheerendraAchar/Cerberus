# 🚀 PHASE 3 QUICK START GUIDE
## Getting Started with Multiple Attacks Implementation

**Created:** February 19, 2026  
**Status:** Ready to implement  
**Estimated Duration:** 4-8 weeks

---

## ✅ WHAT'S BEEN CREATED

### **1. Attack Implementations (4 files)**
- ✅ `cerberus/attacks/pgd_attack.py` - Projected Gradient Descent
- ✅ `cerberus/attacks/cw_attack.py` - Carlini & Wagner
- ✅ `cerberus/attacks/deepfool_attack.py` - DeepFool
- ✅ `cerberus/attacks/jsma_attack.py` - JSMA

### **2. Test Suite**
- ✅ `tests/test_phase3_attacks.py` - Comprehensive unit tests

### **3. Documentation**
- ✅ `PHASE3_IMPLEMENTATION_PLAN.md` - Complete roadmap

---

## 🎯 NEXT STEPS (IN ORDER)

### **STEP 1: Fix PyTorch Installation** (Critical!)
The attacks need PyTorch to run. Current issue: Import error.

```bash
# Try one of these (in order):

# Option A: Fresh pip install
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Option B: Using conda (if you have it)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Option C: Check current environment
python3 -c "import torch; print(torch.__version__)"
```

**Verification:**
```bash
python3 -c "import torch; print('✅ PyTorch installed:', torch.__version__)"
```

---

### **STEP 2: Run Attack Tests** (Verify implementations)

Once PyTorch is installed:

```bash
cd /Users/admin/Desktop/major_projekt
python3 tests/test_phase3_attacks.py
```

**Expected output:**
```
🚀 PHASE 3 - ATTACK IMPLEMENTATIONS TEST SUITE 🚀
============================================================
🧪 Testing PGD Attack
============================================================
✅ PGD Attack Test PASSED
   Input shape: torch.Size([4, 10])
   Output shape: torch.Size([4, 10])
   Max perturbation: 0.2847

============================================================
📊 TEST SUMMARY
============================================================
✅ Passed: 4/4
❌ Failed: 0/4

🎉 ALL TESTS PASSED! Phase 3 attacks ready for use.
```

---

### **STEP 3: Integrate Attacks into CLI** (Days 1-2)

Update `cerberus/cli.py` to add attack options:

**Location:** `cerberus/cli.py` - in `run_from_config()` function

Add around line 100-150:

```python
def run_from_config(config_path: str):
    """Run attacks from configuration."""
    config = load_config(config_path)
    
    # Existing FGSM code...
    if config['attack_type'] == 'fgsm':
        # ... existing FGSM code ...
    
    # NEW: Add other attacks
    elif config['attack_type'] == 'pgd':
        from cerberus.attacks.pgd_attack import PGDAttack
        pgd = PGDAttack(
            model=model,
            eps=config.get('epsilon', 0.03),
            eps_step=config.get('eps_step', 0.01),
            max_iter=config.get('max_iter', 20)
        )
        results = pgd.evaluate(test_loader)
        
    elif config['attack_type'] == 'cw':
        from cerberus.attacks.cw_attack import CWAttack
        cw = CWAttack(
            model=model,
            c=config.get('c', 1.0),
            learning_rate=config.get('learning_rate', 0.01),
            max_iterations=config.get('max_iterations', 100)
        )
        results = cw.evaluate(test_loader)
    
    # ... similarly for deepfool and jsma ...
```

---

### **STEP 4: Create Attack Comparison Script** (Days 2-3)

**File to create:** `scripts/compare_all_attacks.py`

```python
"""Compare all attack types on the same model."""

import torch
import json
from cerberus.attacks.pgd_attack import PGDAttack
from cerberus.attacks.cw_attack import CWAttack
from cerberus.attacks.deepfool_attack import DeepFoolAttack
from cerberus.attacks.jsma_attack import JSMAAttack
from cerberus.dataset import get_cifar10_data
from cerberus.model import get_resnet18

def compare_attacks():
    """Compare all attacks on baseline model."""
    
    # Load data and model
    _, test_loader = get_cifar10_data(batch_size=128)
    model = get_resnet18(pretrained=True)
    
    # Create attacks
    attacks = {
        'FGSM': None,  # Existing from Phase 2
        'PGD': PGDAttack(model, eps=0.03, eps_step=0.01, max_iter=20),
        'C&W': CWAttack(model, c=1.0, learning_rate=0.01, max_iterations=100),
        'DeepFool': DeepFoolAttack(model, max_iterations=100),
        'JSMA': JSMAAttack(model, theta=1.0, max_pixels=100),
    }
    
    # Evaluate each attack
    results = {}
    for attack_name, attack in attacks.items():
        print(f"\n🔥 Evaluating {attack_name}...")
        try:
            result = attack.evaluate(test_loader)
            results[attack_name] = result
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Save and display results
    with open('outputs/attack_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*80)
    print("ATTACK COMPARISON RESULTS")
    print("="*80)
    print(f"{'Attack':<15} {'Accuracy':<12} {'Success Rate':<15} {'Time (s)':<10}")
    print("-"*80)
    
    for attack_name, result in results.items():
        accuracy = result.get('accuracy', 0)
        success = result.get('attack_success_rate', 0)
        time_sec = result.get('time_seconds', 0)
        print(f"{attack_name:<15} {accuracy:<12.2f}% {success:<15.2f}% {time_sec:<10.2f}")
    
    print("="*80)

if __name__ == "__main__":
    compare_attacks()
```

---

### **STEP 5: Create Transfer Analysis Framework** (Days 4-7)

**File to create:** `scripts/train_all_architectures.py`

This will train 6 different architectures:
- ResNet-18 (you have this)
- VGG-16
- MobileNet V2
- EfficientNet-B0
- DenseNet-121
- Vision Transformer (ViT)

```python
"""Train multiple architectures for transfer analysis."""

import torch
import torchvision.models as models
from cerberus.adversarial_training import AdversarialTrainer
from cerberus.dataset import get_cifar10_data
import os

ARCHITECTURES = [
    'resnet18',
    'vgg16',
    'mobilenet_v2',
    'efficientnet_b0',
    'densenet121',
]

def train_architecture(arch_name: str, num_epochs: int = 50):
    """Train one architecture."""
    print(f"\n{'='*60}")
    print(f"Training {arch_name.upper()}")
    print(f"{'='*60}")
    
    # Load model
    if arch_name == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif arch_name == 'vgg16':
        model = models.vgg16(pretrained=False)
    elif arch_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
    elif arch_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
    elif arch_name == 'densenet121':
        model = models.densenet121(pretrained=False)
    
    # Fix last layer for CIFAR-10 (10 classes)
    num_classes = 10
    if hasattr(model, 'fc'):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        # For VGG
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    
    # Load data
    train_loader, test_loader = get_cifar10_data(batch_size=128)
    
    # Train with adversarial training
    trainer = AdversarialTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epsilon=0.03,
        alpha=0.5
    )
    
    for epoch in range(1, num_epochs + 1):
        trainer.train_epoch(epoch)
        
        if epoch % 10 == 0:
            clean_acc = trainer.evaluate(adversarial=False)
            adv_acc = trainer.evaluate(adversarial=True)
            print(f"Epoch {epoch}: Clean={clean_acc:.2f}%, Adv={adv_acc:.2f}%")
    
    # Save model
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(
        model.state_dict(),
        f'outputs/models/{arch_name}_adversarial.pt'
    )
    print(f"✅ Model saved: outputs/models/{arch_name}_adversarial.pt")

def train_all():
    """Train all architectures."""
    for arch in ARCHITECTURES:
        train_architecture(arch, num_epochs=50)

if __name__ == "__main__":
    train_all()
```

---

### **STEP 6: Transfer Attack Matrix** (Days 7-10)

Create `scripts/run_transfer_analysis.py`:

```python
"""Generate adversarial transfer matrix."""

import torch
import json
import numpy as np
from cerberus.attacks.pgd_attack import PGDAttack
# ... import other attacks ...

def generate_transfer_matrix():
    """Generate 6x6 transfer matrix."""
    architectures = [
        'resnet18', 'vgg16', 'mobilenet_v2',
        'efficientnet_b0', 'densenet121'
    ]
    
    matrix = np.zeros((len(architectures), len(architectures)))
    
    # Load all trained models
    models = {}
    for arch in architectures:
        model = load_model(f'outputs/models/{arch}_adversarial.pt')
        models[arch] = model
    
    # Generate attacks from each source
    for src_idx, src_arch in enumerate(architectures):
        print(f"\nGenerating attacks from {src_arch}...")
        
        # Generate PGD attacks from this architecture
        pgd = PGDAttack(models[src_arch], eps=0.03)
        adv_examples = pgd.generate_adversarial(images, labels)
        
        # Evaluate on each target architecture
        for tgt_idx, tgt_arch in enumerate(architectures):
            success_rate = evaluate_transfer(
                models[tgt_arch],
                adv_examples,
                labels
            )
            matrix[src_idx, tgt_idx] = success_rate
    
    # Save and visualize
    np.save('outputs/transfer_matrix.npy', matrix)
    visualize_matrix(matrix, architectures)

if __name__ == "__main__":
    generate_transfer_matrix()
```

---

## 📅 TIMELINE

| Week | Task | Status |
|------|------|--------|
| 1 | Fix PyTorch + Run tests | 🔄 TODO |
| 1-2 | Integrate attacks into CLI | 🔄 TODO |
| 2-3 | Create attack comparison script | 🔄 TODO |
| 3-4 | Train 6 architectures | 🔄 TODO |
| 4-5 | Transfer analysis framework | 🔄 TODO |
| 5-6 | Transfer matrix generation | 🔄 TODO |
| 6-8 | Documentation + Paper writing | 🔄 TODO |

---

## 📦 FILES CREATED THIS SESSION

```
✅ cerberus/attacks/
   ├── __init__.py
   ├── pgd_attack.py
   ├── cw_attack.py
   ├── deepfool_attack.py
   └── jsma_attack.py

✅ tests/
   └── test_phase3_attacks.py

✅ Documentation/
   ├── PHASE3_IMPLEMENTATION_PLAN.md
   └── PHASE3_QUICK_START_GUIDE.md (this file)
```

---

## 🚀 IMMEDIATE ACTION ITEMS

### **TODAY:**
1. [ ] Fix PyTorch installation
2. [ ] Run attack tests
3. [ ] Verify all 4 attacks work

### **THIS WEEK:**
4. [ ] Integrate attacks into CLI
5. [ ] Create attack comparison script
6. [ ] Benchmark each attack on baseline model

### **NEXT 2 WEEKS:**
7. [ ] Start training additional architectures (VGG, MobileNet, etc.)
8. [ ] Create transfer analysis framework

---

## 📚 REFERENCES FOR EACH ATTACK

### **PGD (Projected Gradient Descent)**
- Paper: Madry et al. (2018) - "Towards Deep Learning Models Resistant to Adversarial Attacks"
- Key: Iterative attack with perturbation projection
- Strength: Stronger than FGSM, good practical attack
- Weakness: Slower than FGSM

### **C&W (Carlini & Wagner)**
- Paper: Carlini & Wagner (2016) - "Towards Evaluating the Robustness of Neural Networks"
- Key: Optimization-based, finds minimal perturbation
- Strength: Very strong, good for worst-case evaluation
- Weakness: Slowest of all attacks

### **DeepFool**
- Paper: Moosavi-Dezfooli et al. (2016) - "DeepFool: a simple and accurate method to fool deep neural networks"
- Key: Finds minimal perturbation to decision boundary
- Strength: Efficient, interpretable
- Weakness: Single perturbation direction

### **JSMA**
- Paper: Papernot et al. (2015) - "The Limitations of Deep Learning in Adversarial Settings"
- Key: Modifies few pixels using saliency maps
- Strength: Targeted, sparse perturbations
- Weakness: Computationally expensive (requires Jacobian)

---

## 💡 TIPS FOR SUCCESS

1. **Start Small:** Test each attack on toy data first (small models)
2. **Benchmark Early:** Create comparison plots of attack strengths
3. **Document Results:** Save results after each attack runs
4. **GPU Acceleration:** If available, use GPU for faster training
5. **Version Control:** Commit code after each major step
6. **Incremental:** Don't try to do everything at once

---

## 🆘 TROUBLESHOOTING

### **PyTorch Import Error**
```
ModuleNotFoundError: No module named 'torch'
```
**Solution:** Run `pip3 install torch torchvision`

### **CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or use `device='cpu'`

### **Slow Training**
```
Training takes too long
```
**Solution:** Reduce number of epochs, use smaller model, use GPU

---

## 📞 NEXT STEPS

**Ready to implement?**
1. Go to Step 1: Fix PyTorch
2. Come back when you hit any issues
3. I can help debug or accelerate any part

**Questions?**
- About attacks? Check PHASE3_IMPLEMENTATION_PLAN.md
- About code? Check docstrings in each attack file
- About timeline? Adjust based on your availability

---

**You've got this! Let's build something novel! 🚀✨**

