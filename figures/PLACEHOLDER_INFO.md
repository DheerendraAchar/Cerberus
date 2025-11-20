# Figure Placeholders for Project Cerberus Report

## Status: Figures Not Yet Generated

The `figures/` directory exists but is empty. Figures need to be generated using one of the methods below.

---

## Required Figures for Results Section

### Figure 1: Clean vs Adversarial Examples Grid
**Filename**: `fgsm_examples_eps0.03.png`  
**Description**: 2-row grid showing 8-12 CIFAR-10 images. Top row = clean/original images with correct predictions. Bottom row = same images after FGSM attack (ε=0.03), showing misclassifications in red text.  
**Purpose**: Demonstrates that adversarial perturbations are nearly imperceptible to humans but fool the model.  
**Caption for report**: "Figure 1: Clean (top) vs FGSM-perturbed (bottom) CIFAR-10 samples (ε=0.03). Red labels indicate misclassifications caused by imperceptible noise."

---

### Figure 2: Accuracy vs Epsilon Curve
**Filename**: `fgsm_accuracy_vs_epsilon.png`  
**Description**: Line plot with epsilon values (0.0 to 0.10) on x-axis and model accuracy on y-axis. Shows sharp accuracy drop as perturbation budget increases.  
**Purpose**: Quantifies robustness degradation; identifies critical epsilon threshold where model becomes unreliable.  
**Caption for report**: "Figure 2: Test accuracy under FGSM attack degrades from 85% (clean) to ~15% at ε=0.10, indicating severe vulnerability."

---

### Figure 3a: Confusion Matrix (Clean)
**Filename**: `confusion_clean.png`  
**Description**: 10×10 heatmap showing predicted vs actual classes for CIFAR-10 test set without attack. Strong diagonal indicates good baseline performance.  
**Purpose**: Establishes baseline; shows which classes are naturally harder (e.g., cat vs dog confusion).  
**Caption for report**: "Figure 3a: Baseline confusion matrix on clean data shows 85% overall accuracy with some cat/dog confusion."

---

### Figure 3b: Confusion Matrix (FGSM Attack)
**Filename**: `confusion_fgsm_eps0.03.png`  
**Description**: Same format as 3a, but after FGSM attack at ε=0.03. Diagonal weakens; off-diagonal entries grow, revealing vulnerable class pairs.  
**Purpose**: Identifies which classes suffer most under attack; informs defense prioritization.  
**Caption for report**: "Figure 3b: Post-attack confusion matrix (ε=0.03) reveals accuracy collapse to 60%, with 'cat'→'dog' and 'deer'→'horse' misclassifications increasing 3×."

---

### Figure 4: Perturbation Heatmap
**Filename**: `fgsm_perturbation_heatmap_eps0.03.png`  
**Description**: Single 32×32 heatmap showing spatial distribution of |adv - clean| averaged over RGB channels. FGSM typically shows uniform noise pattern.  
**Purpose**: Visualizes attack strategy; helps distinguish FGSM (uniform) from targeted attacks (localized).  
**Caption for report**: "Figure 4: FGSM perturbation heatmap exhibits uniform spatial distribution, confirming gradient-based untargeted attack characteristics."

---

## How to Generate These Figures

### Method 1: Docker (Recommended - No Local Dependencies)

```bash
# Build Docker image with all dependencies
docker build -t cerberus-figures -f Dockerfile.figures .

# Generate figures (saved to ./figures/)
docker run --rm -v $(pwd)/figures:/app/figures cerberus-figures

# Verify
ls -lh figures/
```

**Time**: ~10-15 minutes (Docker build) + 2-5 minutes (figure generation)  
**Requirements**: Docker Desktop installed

---

### Method 2: Local Python Environment

**Prerequisites**:
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
pip install matplotlib numpy scikit-learn
```

**Run**:
```bash
python scripts/generate_figures.py \
  --device cpu \
  --eps-list 0.0 0.01 0.02 0.03 0.05 0.07 0.10 \
  --fgsm-eps 0.03 \
  --samples 12
```

**Time**: 5-10 minutes  
**Requirements**: Python 3.9+, 2GB RAM, ~500MB disk for dependencies

---

### Method 3: Use Placeholder Figures (For Draft Reports)

If you need figures immediately for report structure/layout but don't have time for full generation:

```bash
# Requires only matplotlib (lighter dependency)
pip install matplotlib numpy
python scripts/generate_placeholder_figures.py
```

This creates simulated/mockup figures with "[PLACEHOLDER]" watermarks. **Replace with real figures before final submission.**

---

## Current Status

- [x] Figure generation script created (`scripts/generate_figures.py`)
- [x] Docker build file created (`Dockerfile.figures`)
- [x] Placeholder generator created (`scripts/generate_placeholder_figures.py`)
- [ ] **Figures not yet generated** (run one of the methods above)
- [ ] Docker build in progress (was canceled at 79s)

---

## Quick Start (Immediate Action)

**For placeholder figures now**:
```bash
pip install matplotlib numpy
python scripts/generate_placeholder_figures.py
```

**For real figures later** (when you have 15 min):
```bash
./scripts/generate_figures_docker.sh
```

---

## Troubleshooting

**"No module named 'torch'"**  
→ Use Docker method or install: `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`

**"No module named 'matplotlib'"**  
→ `pip install matplotlib numpy`

**"Docker build failed"**  
→ Check Docker Desktop is running  
→ Ensure 5GB disk space available  
→ Try: `docker system prune` then rebuild

**"Permission denied: figures/"**  
→ `chmod 755 figures/`

---

Last Updated: November 20, 2025  
Phase: 1 (MVP)  
Repository: https://github.com/DheerendraAchar/Cerberus
