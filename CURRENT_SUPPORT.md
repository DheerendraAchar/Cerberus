# 🎯 Project Cerberus - Current Support Matrix

**Last Updated:** December 28, 2025  
**Current Phase:** Phase 2 Complete  

---

## 📊 Quick Summary

| Category | Currently Supported | Status |
|----------|---------------------|--------|
| **Attacks** | FGSM | ✅ 1 attack |
| **Models** | ResNet-18, Custom PyTorch models | ✅ Flexible |
| **Datasets** | CIFAR-10 | ✅ 1 dataset |
| **Training** | Baseline, Adversarial | ✅ 2 methods |
| **Devices** | CPU | ✅ CPU-only |

---

## ⚔️ Supported Attacks

### 1. FGSM (Fast Gradient Sign Method) ✅

**Status:** Fully Implemented  
**Implementation:** `cerberus/attacks.py`  
**Library:** IBM Adversarial Robustness Toolbox (ART)

**Capabilities:**
- ✅ Single-step gradient-based attack
- ✅ Configurable epsilon (perturbation strength)
- ✅ Batch processing for efficiency
- ✅ Used for both evaluation and training

**Usage:**
```python
# Evaluation mode
python3 run_demo.py --mode eval --config configs/sample_config.yaml

# In training (adversarial training)
python3 run_demo.py --mode train --training-type adversarial --config configs/training_config.yaml
```

**Configuration:**
```yaml
attack:
  name: fgsm
  eps: 0.03  # Perturbation strength (0.03 ≈ 8/255)
```

**How It Works:**
```
x_adv = x + ε × sign(∇_x Loss(f(x), y))
```
- Adds small perturbation in direction of gradient
- Fast (single forward-backward pass)
- Effective for testing basic robustness

**Typical Results (CIFAR-10, ε=0.03):**
- Baseline model accuracy drop: 70% → 35% (48% relative drop)
- Adversarial trained model: 68% → 52% (23% relative drop)

---

### ⏳ Planned Future Attacks (Phase 3)

| Attack | Type | Status | Planned |
|--------|------|--------|---------|
| **PGD** | Multi-step iterative | 🔄 Planned | Phase 3 (Jan 2026) |
| **C&W** | Optimization-based | 🔄 Planned | Phase 3 (Jan 2026) |
| **DeepFool** | Minimal perturbation | 🔄 Planned | Phase 3 (Jan 2026) |
| **AutoAttack** | Ensemble | 🔄 Planned | Phase 4 (Feb 2026) |

---

## 🏗️ Supported Models

### 1. ResNet-18 (Custom Implementation) ✅

**Status:** Fully Implemented  
**Implementation:** `cerberus/cli.py` (lines 59-112)  
**Use Case:** Training from scratch on CIFAR-10

**Architecture Details:**
```
SimpleResNet (ResNet-18 variant)
├── Conv1: 3 → 64 channels (3×3, stride=1)
├── BatchNorm + ReLU
├── Layer 1: 2× BasicBlock (64 → 64, stride=1)
├── Layer 2: 2× BasicBlock (64 → 128, stride=2)
├── Layer 3: 2× BasicBlock (128 → 256, stride=2)
├── Layer 4: 2× BasicBlock (256 → 512, stride=2)
├── AdaptiveAvgPool2d(1)
└── FC: 512 → 10 classes
```

**Specifications:**
- **Parameters:** ~11.2 million
- **Input Size:** 32×32×3 (CIFAR-10)
- **Output:** 10 classes
- **Features:** Residual connections, Batch normalization
- **Optimized for:** CIFAR-10 32×32 images

**Usage:**
```python
# Automatically used in training mode
python3 run_demo.py --mode train --training-type baseline --config configs/training_config.yaml
```

---

### 2. Custom PyTorch Models ✅

**Status:** Fully Supported  
**Implementation:** `cerberus/model.py`  
**Use Case:** Load pre-trained models for evaluation

**Supported Formats:**
- ✅ `.pt` files (PyTorch saved models)
- ✅ `.pth` files (PyTorch state dictionaries)
- ✅ TorchScript models (`torch.jit`)
- ✅ Model instances with `.eval()` method

**Loading Capabilities:**
```python
from cerberus.model import load_pytorch_model

# Load any PyTorch model
model = load_pytorch_model("path/to/model.pt", device="cpu")
```

**Supported Model Types:**
- ✅ Convolutional Neural Networks (CNNs)
- ✅ Any PyTorch `nn.Module` subclass
- ✅ Pre-trained models from torchvision
- ✅ Custom architectures

**Requirements:**
- Must be compatible with PyTorch
- Should accept inputs shaped as `(N, C, H, W)`
- Should output logits for classification

---

### ⏳ Planned Future Models (Phase 3+)

| Model | Type | Status | Planned |
|-------|------|--------|---------|
| **VGG-16** | Deep CNN | 🔄 Planned | Phase 3 |
| **DenseNet** | Dense connections | 🔄 Planned | Phase 3 |
| **Transformer** | Attention-based | 🔄 Planned | Phase 4 |
| **NLP Models** | BERT, RoBERTa | 🔄 Planned | Phase 3 (NLP support) |

---

## 📦 Supported Datasets

### 1. CIFAR-10 ✅

**Status:** Fully Implemented  
**Implementation:** `cerberus/dataset.py`  
**Download:** Automatic via torchvision

**Dataset Details:**
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Images:** 50,000
- **Test Images:** 10,000
- **Image Size:** 32×32 RGB
- **Total Size:** ~170 MB
- **Source:** https://www.cs.toronto.edu/~kriz/cifar.html

**Capabilities:**
- ✅ Automatic download
- ✅ Train/test split
- ✅ Data normalization
- ✅ Configurable batch size
- ✅ Multi-worker loading

**Usage:**
```python
from cerberus.dataset import get_cifar10_loaders

train_loader, test_loader = get_cifar10_loaders(
    root="./data",
    batch_size=128,
    num_workers=2
)
```

**Configuration:**
```yaml
dataset:
  name: cifar10
  root: ./data
  batch_size: 128
  num_workers: 2
```

**Class Distribution:**
All classes have 5,000 training images and 1,000 test images (balanced).

---

### ⏳ Planned Future Datasets (Phase 3+)

| Dataset | Type | Images | Status | Planned |
|---------|------|--------|--------|---------|
| **MNIST** | Grayscale digits | 70k | 🔄 Planned | Phase 3 |
| **CIFAR-100** | Fine-grained | 60k | 🔄 Planned | Phase 3 |
| **ImageNet** | Large-scale | Subset | 🔄 Planned | Phase 4 |
| **IMDB** | Text sentiment | 50k | 🔄 Planned | Phase 3 (NLP) |
| **SST** | Text sentiment | 11k | 🔄 Planned | Phase 3 (NLP) |

---

## 🏋️ Supported Training Methods

### 1. Baseline Training (Standard) ✅

**Status:** Fully Implemented  
**Implementation:** `cerberus/baseline_training.py`  
**Use Case:** Train standard models on clean data

**Features:**
- ✅ Standard supervised learning
- ✅ Cross-entropy loss
- ✅ SGD optimizer with momentum (0.9)
- ✅ Cosine annealing learning rate
- ✅ Automatic checkpointing (best model)
- ✅ Real-time metrics tracking

**Usage:**
```bash
python3 run_demo.py \
    --mode train \
    --training-type baseline \
    --config configs/training_config.yaml
```

**Training Parameters:**
- Learning Rate: 0.01
- Momentum: 0.9
- Weight Decay: 5e-4
- Epochs: 100 (configurable)
- Batch Size: 128

**Expected Results (CIFAR-10):**
- Clean Accuracy: 70-75%
- Training Time: 2-3 hours (CPU)

---

### 2. Adversarial Training (Defense) ✅

**Status:** Fully Implemented  
**Implementation:** `cerberus/adversarial_training.py`  
**Use Case:** Train robust models with defense mechanism

**Features:**
- ✅ FGSM-based adversarial example generation
- ✅ Configurable mix ratio (clean + adversarial)
- ✅ Dual evaluation (clean + adversarial accuracy)
- ✅ Same optimizer/scheduler as baseline
- ✅ Extended metrics tracking

**Usage:**
```bash
python3 run_demo.py \
    --mode train \
    --training-type adversarial \
    --config configs/training_config.yaml
```

**Training Parameters:**
- Epsilon (ε): 0.03 (perturbation strength)
- Alpha (α): 0.5 (50% clean, 50% adversarial)
- Learning Rate: 0.01
- Momentum: 0.9
- Epochs: 100 (configurable)

**How It Works:**
```
For each batch:
1. Forward pass on clean images
2. Generate FGSM adversarial examples
3. Mix clean + adversarial (shuffle)
4. Train on mixed batch
5. Evaluate on clean & adversarial test sets
```

**Expected Results (CIFAR-10):**
- Clean Accuracy: 65-70%
- Adversarial Accuracy: 50-55%
- Robustness Improvement: +18% over baseline
- Training Time: 3-4 hours (CPU)

---

### ⏳ Planned Future Training Methods (Phase 3+)

| Method | Type | Status | Planned |
|--------|------|--------|---------|
| **TRADES** | Robust training | 🔄 Planned | Phase 3 |
| **MART** | Misclassification aware | 🔄 Planned | Phase 3 |
| **Curriculum Training** | Progressive difficulty | 🔄 Planned | Phase 3 |
| **Certified Training** | Provable robustness | 🔄 Planned | Phase 4 |

---

## 💻 Supported Devices

### CPU-Only ✅

**Status:** Fully Supported  
**Current Configuration:** CPU-only mode

**Capabilities:**
- ✅ All training operations
- ✅ All evaluation operations
- ✅ All visualization operations
- ✅ Docker containerization (CPU)

**Performance:**
- Baseline training: 2-3 hours (100 epochs)
- Adversarial training: 3-4 hours (100 epochs)
- Model evaluation: 2-3 minutes (10k samples)

**Configuration:**
```yaml
device:
  use_gpu: false
  gpu_id: 0
```

---

### GPU Support 🔄

**Status:** Planned for Phase 4  
**Current Limitation:** CPU-only implementation

**Planned GPU Features:**
- Multi-GPU training
- CUDA acceleration
- Mixed precision training
- Distributed data parallel

**Expected Performance Gain:**
- Training time: 15-30 minutes (vs 2-4 hours on CPU)
- Evaluation time: ~30 seconds (vs 2-3 minutes on CPU)

---

## 🔧 Technical Stack

### Core Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.9+ | Runtime |
| **PyTorch** | Latest | Deep learning framework |
| **torchvision** | Latest | Vision datasets and transforms |
| **IBM ART** | Latest | Adversarial attacks |
| **NumPy** | Latest | Numerical operations |
| **matplotlib** | Latest | Visualization |
| **PyYAML** | Latest | Configuration |

### Optional Dependencies

| Component | Purpose | Status |
|-----------|---------|--------|
| **Docker** | Containerization | ✅ Supported |
| **pytest** | Testing | ✅ Supported |
| **scikit-learn** | Metrics | ✅ Supported |

---

## 📊 Configuration System

### Supported Config Files

#### 1. Training Configuration ✅
**File:** `configs/training_config.yaml`  
**Purpose:** Training hyperparameters

**Configurable Parameters:**
```yaml
# Training type
training_type: baseline | adversarial

# General training
training:
  num_epochs: 100
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0005

# Adversarial training
adversarial:
  epsilon: 0.03
  alpha: 0.5

# Dataset
dataset:
  name: CIFAR10
  batch_size: 128
  num_workers: 2

# Model
model:
  architecture: resnet18
  num_classes: 10

# Device
device:
  use_gpu: false
```

#### 2. Evaluation Configuration ✅
**File:** `configs/sample_config.yaml`  
**Purpose:** Attack evaluation

**Configurable Parameters:**
```yaml
model:
  path: null  # or path to .pt/.pth

dataset:
  name: cifar10
  root: ./data
  batch_size: 64

attack:
  name: fgsm
  eps: 0.03

output:
  report_path: outputs/report.html
```

---

## 📈 Evaluation Metrics

### Supported Metrics ✅

| Metric | Description | Availability |
|--------|-------------|--------------|
| **Clean Accuracy** | Accuracy on original test data | ✅ All modes |
| **Adversarial Accuracy** | Accuracy on adversarial examples | ✅ All modes |
| **Accuracy Drop** | Clean - Adversarial accuracy | ✅ Comparison |
| **Robustness Ratio** | Adv_acc / Clean_acc | ✅ Comparison |
| **Training Loss** | Cross-entropy loss | ✅ Training |
| **Test Loss** | Validation loss | ✅ Training |
| **Learning Rate** | Current LR | ✅ Training |

---

## 🎯 Complete Capability Matrix

### What You Can Do Now

| Feature | Baseline | Adversarial | Evaluation | Status |
|---------|----------|-------------|------------|--------|
| **Train ResNet-18** | ✅ | ✅ | N/A | Complete |
| **Train on CIFAR-10** | ✅ | ✅ | N/A | Complete |
| **FGSM Attack** | N/A | ✅ Training | ✅ Eval | Complete |
| **Load Custom Models** | N/A | N/A | ✅ | Complete |
| **Model Comparison** | ✅ | ✅ | ✅ | Complete |
| **Training Visualization** | ✅ | ✅ | N/A | Complete |
| **Checkpointing** | ✅ | ✅ | N/A | Complete |
| **HTML Reports** | N/A | N/A | ✅ | Complete |
| **CPU Execution** | ✅ | ✅ | ✅ | Complete |
| **Docker Support** | N/A | N/A | ✅ | Complete |

---

## 🚀 Usage Examples

### Example 1: Train and Evaluate
```bash
# Train baseline
python3 run_demo.py --mode train --training-type baseline --config configs/training_config.yaml

# Train adversarial
python3 run_demo.py --mode train --training-type adversarial --config configs/training_config.yaml

# Compare
python3 scripts/compare_models.py \
    --baseline-checkpoint outputs/models/baseline_model.pt \
    --adversarial-checkpoint outputs/models/adversarial_model.pt
```

### Example 2: Evaluate Custom Model
```bash
# Edit configs/sample_config.yaml to point to your model
model:
  path: path/to/your/model.pt

# Run evaluation
python3 run_demo.py --mode eval --config configs/sample_config.yaml
```

### Example 3: Quick FGSM Test
```bash
# Uses default config with sample model
python3 run_demo.py --mode eval --config configs/sample_config.yaml
```

---

## 🔮 Roadmap

### Phase 3 (Planned: Jan 2026)
- Additional attacks: PGD, C&W, DeepFool
- Additional datasets: MNIST, CIFAR-100
- NLP support: Text attacks, BERT models
- Plugin architecture

### Phase 4 (Planned: Feb 2026)
- GPU acceleration
- ImageNet support
- Additional defense methods
- Comprehensive benchmarking
- Final documentation and submission

---

## 📝 Summary

**Currently Supported:**

✅ **1 Attack Type:** FGSM (with configurable epsilon)  
✅ **2 Model Types:** ResNet-18 (built-in), Custom PyTorch models  
✅ **1 Dataset:** CIFAR-10 (automatic download)  
✅ **2 Training Methods:** Baseline, Adversarial  
✅ **1 Device Type:** CPU (full support)  
✅ **Complete Pipeline:** Train → Attack → Defend → Compare  

**Coming Soon (Phase 3+):**
- 🔄 Additional attacks (PGD, C&W, DeepFool)
- 🔄 Additional datasets (MNIST, CIFAR-100, ImageNet)
- 🔄 NLP support (text attacks, BERT)
- 🔄 GPU acceleration
- 🔄 Plugin architecture

---

*For detailed usage information, see `PHASE2_IMPLEMENTATION.md`*  
*For complete project capabilities, see `PROJECT_CAPABILITIES.md`*
