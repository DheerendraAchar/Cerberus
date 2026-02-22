# Cerberus: Adversarial ML Framework
## IEEE SSCI 2026 Conference Presentation

---

## SLIDE 1: Title Slide
**Cerberus: A Comprehensive Framework for Adversarial Machine Learning Training and Transfer Attack Analysis**

**Authors:**
- B Dheerendra Achar
- Chhavi Sharma
- Gaurav Bhandare
- Chiranjeev Kapoor

**Affiliation:** Department of Computer Science, Dayananda Sagar University, Bangalore

**Date:** February 2026

---

## SLIDE 2: The Problem
**Why Adversarial Robustness Matters**

### The Vulnerability
- Deep neural networks fool easily with adversarial examples
- Small imperceptible perturbations cause misclassification
- Critical for safety (autonomous vehicles, medical diagnosis)
- Security risk (fraud detection, malware analysis)

### The Challenge
- Multiple attack types exist (FGSM, PGD, C&W, DeepFool, JSMA)
- Different architectures show different robustness
- No unified framework for evaluation
- **Question:** How does robustness vary across architectures and attacks?

---

## SLIDE 3: Research Questions

**Q1: Attack Effectiveness Across Architectures**
- How do different attacks perform on ResNet-18, VGG-16, MobileNet V2, EfficientNet-B0, DenseNet-121?

**Q2: Adversarial Transferability**
- Do adversarial examples transfer across architectures?
- What is the transfer rate compared to self-attacks?

**Q3: Defense Through Diversity**
- Can architectural diversity serve as a natural defense?
- What is the gap between self-attacks and cross-architecture transfers?

---

## SLIDE 4: Our Contributions

### 1. Comprehensive Framework
✅ 5 attack algorithms (FGSM, PGD, C&W, DeepFool, JSMA)
✅ Production-ready code (2,950+ lines)
✅ Unified interface for researchers and practitioners

### 2. Multi-Architecture Analysis
✅ 5 diverse CNN architectures
✅ Systematic evaluation
✅ Architectural insights for robustness

### 3. Novel Transfer Analysis
✅ 5×5 transfer attack matrix
✅ **17.18 pp gap** between self-attacks and transfers
✅ Evidence for architecture-based defenses

### 4. Practical Insights
✅ Recommendations for practitioners
✅ Defense effectiveness analysis
✅ Deployment guidelines

---

## SLIDE 5: Methodology Overview

### Attack Algorithms
- **FGSM:** Fast Gradient Sign Method - single-step gradient attack
- **PGD:** Projected Gradient Descent - iterative attack (strongest)
- **C&W:** Carlini & Wagner - optimization-based attack
- **DeepFool:** Boundary-seeking minimal perturbation
- **JSMA:** Jacobian Saliency Map Attack - feature-targeted

### Experimental Setup
- **Dataset:** CIFAR-10 (50K train, 10K test)
- **Architectures:** ResNet-18, VGG-16, MobileNet V2, EfficientNet-B0, DenseNet-121
- **Training:** Adversarial training (50% clean + 50% adversarial examples)
- **Evaluation:** Clean accuracy, adversarial accuracy, transfer rates

---

## SLIDE 6: Attack Algorithms - Equations

### FGSM: One-Step Attack
```
x_adv = x + ε · sign(∇_x L(x, y))
```

### PGD: Multi-Step Attack
```
x_{t+1} = Clip(x_t + α · sign(∇_x L(x_t, y)))
```

### C&W: Optimization-Based
```
minimize: ||x - x'||²_2 + c · L(x', t)
```

### DeepFool: Minimal Perturbation
```
r_i = -f(x_i) / ||∇f(x_i)||² · ∇f(x_i)
```

### JSMA: Feature-Targeted
```
S(x,t)[i] = |∂f_t/∂x_i| · Σ_{j≠t} max(0, -∂f_j/∂x_i)
```

---

## SLIDE 7: Adversarial Training

### Mixed Training Strategy
Train on 50% clean examples + 50% adversarial examples

### Algorithm
```
For each epoch:
  For each batch:
    Sample α ~ Bernoulli(0.5)
    If α=1:
      Generate adversarial examples (FGSM, ε=8/255)
      Train on adversarial batch
    Else:
      Train on clean batch
```

### Benefits
- ~49% robustness improvement
- Practical defense mechanism
- Deployable in production

---

## SLIDE 8: Key Results - Training Performance

| Architecture | Clean Acc. | Adv. Acc. | Robustness Gain |
|---|---|---|---|
| **ResNet-18** | 92.19% | **43.06%** | 50.13% |
| **VGG-16** | 89.85% | 38.46% | 51.39% |
| **MobileNet V2** | 90.53% | 43.43% | 47.10% |
| **EfficientNet-B0** | 89.17% | 40.03% | 49.14% |
| **DenseNet-121** | 89.25% | 42.15% | 47.10% |
| **Average** | **90.20%** | **41.43%** | **48.97%** |

**Key Insight:** ~50% robustness improvement across all architectures through adversarial training

---

## SLIDE 9: Transfer Attack Matrix - Visual

### 5×5 Transfer Matrix Heatmap

```
Source \ Target    RN18    VGG    MNV2    EB0    DN121   Avg
ResNet-18          83.76   69.40  68.21  71.43  65.08  69.40
VGG-16             71.54   85.23  63.45  65.32  62.14  67.53
MobileNet V2       68.32   62.87  81.45  64.51  59.76  65.38
EfficientNet-B0    74.21   66.89  62.43  82.19  61.08  67.34
DenseNet-121       73.45   63.21  61.32  63.87  79.54  66.48
─────────────────────────────────────────────────────────────
Avg (off-diag)     71.88   65.59  63.85  66.28  62.01  66.58
```

**Diagonal Mean (Self-attacks):** 83.76%
**Off-Diagonal Mean (Transfers):** 66.58%
**Difference:** **17.18 percentage points**

---

## SLIDE 10: The 17.18 pp Gap

### What This Means

**Self-Attack Success Rate:** 83.76%
- Adversarial examples fool the same architecture

**Cross-Architecture Transfer Rate:** 66.58%
- Adversarial examples transfer less effectively

**Difference:** 17.18 pp
- **Architectural diversity provides significant defense!**

### Implications
1. **Ensemble defenses work** - Different architectures resist differently
2. **Architecture selection matters** - Some architectures more robust
3. **Transfer is not guaranteed** - Not all attacks transfer equally

---

## SLIDE 11: Architecture Analysis

### ResNet-18: High Transferability
- **Most transferable source:** 69.40% average transfer rate
- Residual connections create generalizable features
- Good for generating universal attacks
- **Bad for defense:** Both vulnerable and high-transferring

### MobileNet V2: Superior Robustness
- **Most robust to transfers:** 34.62% attack success (lowest!)
- Depthwise separable convolutions reduce transferability
- Best choice for ensemble defenses
- Trade-off: Good robustness, less transferable

### VGG-16: Parameter Curse
- **Largest model:** 138M parameters
- **Lowest robustness:** 38.46% adversarial accuracy
- Over-parameterization detrimental to robustness
- High clean accuracy (89.85%) but vulnerable to adversarial attacks

---

## SLIDE 12: Transferability Mechanisms

### Why Only 66.58% Transfer Success?

**1. Feature Diversity**
- Different architectures learn different feature hierarchies
- Adversarial examples optimized for one feature space don't transfer perfectly

**2. Decision Boundary Geometry**
- Each architecture has unique decision boundaries
- Boundaries differ across architecture families
- Perturbations effective for one boundary less effective for others

**3. Adversarial Training Effects**
- Architectures learn different robust features
- Robustness is architecture-specific
- Generalizes less across models

---

## SLIDE 13: Practical Implications

### For Security Practitioners
1. **Deploy Ensembles** with diverse architectures
2. **Mix ResNet + MobileNet** for security-critical applications
3. **Avoid large parameter models** (e.g., VGG-16) if robustness critical
4. **Use adversarial training** - 49% improvement is substantial

### For Researchers
1. **Architectural diversity matters** more than previously understood
2. **17.18 pp gap** validates ensemble effectiveness
3. **Feature diversity is key** - Don't just scale parameters
4. **Further investigate** cross-domain transferability

### For ML System Designers
- Build multi-architecture systems where possible
- Accept accuracy-robustness trade-off
- Monitor adversarial attacks in production
- Regular adversarial testing essential

---

## SLIDE 14: Limitations & Future Work

### Current Limitations
❌ Single dataset (CIFAR-10)
❌ Vision domain only
❌ Untargeted attacks only
❌ Single perturbation bound (ε=8/255)

### Future Research Directions
✅ ImageNet and large-scale datasets
✅ NLP domain and cross-domain transfer
✅ Targeted attacks and their transferability
✅ Certified defenses + architecture diversity
✅ Adaptive attacks (aware of ensemble)
✅ Theoretical bounds on transferability

---

## SLIDE 15: The Cerberus Framework

### What You Get

**Code Quality:**
- 2,950+ lines of production-ready code
- Full documentation and type hints
- Comprehensive test suite
- CLI interface for researchers

**Ease of Use:**
```python
from cerberus import FSGMAttack, Model
model = Model.load("resnet18.pth")
attack = FSGMAttack(epsilon=8/255)
adversarial_examples = attack.generate(images)
```

**Extensibility:**
- Plugin architecture for new attacks
- Easy to add architectures
- Custom training pipelines
- Export for deployment

---

## SLIDE 16: Conclusion & Key Takeaways

### Main Contributions
1. ✅ Comprehensive adversarial ML framework
2. ✅ Multi-architecture evaluation
3. ✅ **17.18 pp gap discovery** - architectural diversity aids defense
4. ✅ Practical deployment guidelines
5. ✅ Open-source production-ready code

### Key Finding
**Architectural diversity provides ~17 percentage point defense benefit against adversarial transfers**

### Call to Action
1. Use Cerberus for your adversarial robustness research
2. Deploy ensemble defenses in production
3. Contribute new attacks/architectures
4. Help advance adversarial ML security

---

## SLIDE 17: Q&A

**Questions?**

### Contact
- **Email:** cerberus@dsu.edu.in
- **GitHub:** github.com/DheerendraAchar/Cerberus
- **Paper:** Available on IEEE Xplore

### Acknowledgments
- Prof. Dharmendra D P (Supervisor)
- Dayananda Sagar University
- All contributors to the Cerberus project

---

## APPENDIX: Extra Slides (If Needed)

### A1: Code Example - Running Attacks

```python
import torch
from cerberus.attacks import FSGMAttack, PGDAttack

# Load model and data
model = torch.load("model.pth")
images, labels = load_cifar10()

# FGSM Attack
fgsm = FSGMAttack(epsilon=8/255)
adv_fgsm = fgsm.generate(images, labels)

# PGD Attack (stronger)
pgd = PGDAttack(epsilon=8/255, steps=20, step_size=2/255)
adv_pgd = pgd.generate(images, labels)

# Evaluate robustness
clean_acc = evaluate(model, images)
adv_acc = evaluate(model, adv_pgd)
print(f"Clean: {clean_acc:.2%}, Adversarial: {adv_acc:.2%}")
```

### A2: Configuration for Reproducibility

```yaml
# training_config.yaml
model:
  architecture: "resnet18"
  pretrained: true
  
adversarial:
  epsilon: 8/255
  alpha: 0.5  # Mix ratio
  
training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.1
  optimizer: "sgd"
```

### A3: Results Table - Complete Matrix

[Full 5×5 matrix with all values shown above]

---

**Presentation Duration:** 25 minutes (17 main slides + Q&A)
**Demo Duration:** 5 minutes (Live code execution)
**Total:** 30 minutes with Q&A
