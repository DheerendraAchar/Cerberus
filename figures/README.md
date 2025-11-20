# Figures Directory

This directory contains visual evaluation results for Project Cerberus adversarial testing framework.

## Generated Figures (Phase 1)

After running the figure generation script, you'll find:

### 1. **fgsm_examples_eps0.03.png**
- **Purpose**: Qualitative demonstration of adversarial perturbations
- **Content**: Grid showing original CIFAR-10 images (top row) vs FGSM-perturbed versions (bottom row)
- **Details**: Misclassifications highlighted in red, correct predictions in green
- **Use in report**: "Figure 1: Clean vs adversarial examples demonstrate near-imperceptible perturbations that fool the model"

### 2. **fgsm_accuracy_vs_epsilon.png**
- **Purpose**: Quantify robustness degradation as attack strength increases
- **Content**: Line plot showing model accuracy for various epsilon values (0.0 to 0.10)
- **Details**: X-axis = epsilon, Y-axis = test accuracy
- **Use in report**: "Figure 2: Model accuracy drops sharply beyond ε=0.05, indicating vulnerability to stronger perturbations"

### 3. **confusion_clean.png**
- **Purpose**: Baseline performance per class
- **Content**: 10×10 confusion matrix for clean (unperturbed) test set
- **Details**: CIFAR-10 classes on axes
- **Use in report**: "Figure 3a: Baseline confusion matrix shows balanced performance across classes"

### 4. **confusion_fgsm_eps0.03.png**
- **Purpose**: Show which classes are most vulnerable to attack
- **Content**: 10×10 confusion matrix after FGSM attack (ε=0.03)
- **Details**: Compare with clean version to identify class-specific weaknesses
- **Use in report**: "Figure 3b: Post-attack confusion reveals 'cat' and 'dog' are frequently misclassified"

### 5. **fgsm_perturbation_heatmap_eps0.03.png**
- **Purpose**: Visualize spatial distribution of perturbations
- **Content**: Heatmap of |adv_image - clean_image| for a single sample
- **Details**: Warmer colors = larger perturbations
- **Use in report**: "Figure 4: Perturbation heatmap shows FGSM distributes noise uniformly across the image"

## How to Generate

### Option A: Using Docker (Recommended, no local PyTorch needed)
```bash
./scripts/generate_figures_docker.sh
```

### Option B: Local Python (requires PyTorch, torchvision, matplotlib, scikit-learn)
```bash
python scripts/generate_figures.py \
  --device cpu \
  --eps-list 0.0 0.01 0.02 0.03 0.05 0.07 0.10 \
  --fgsm-eps 0.03 \
  --samples 12
```

## Customization

**Change epsilon for examples:**
```bash
python scripts/generate_figures.py --fgsm-eps 0.05
```

**Skip certain figures:**
```bash
python scripts/generate_figures.py --skip-confusion --skip-heatmap
```

**More epsilon points in curve:**
```bash
python scripts/generate_figures.py --eps-list 0.0 0.005 0.01 0.015 0.02 0.025 0.03 0.04 0.05 0.07 0.10
```

## Future Expansions (Phase 2+)

Planned additional figures:
- **defense_comparison.png**: Accuracy before/after adversarial retraining
- **pgd_vs_fgsm_comparison.png**: Attack success rates for different methods
- **per_class_robustness.png**: Bar chart of class-specific retention under attack
- **attack_runtime_profile.png**: Computational cost comparison
- **training_curves.png**: Loss during adversarial retraining

## Image Specifications

- **Format**: PNG
- **Resolution**: 160 DPI (suitable for reports and presentations)
- **Size**: Typically 500KB–2MB per image
- **Color**: Full color (RGB for samples, Blues/Inferno colormaps for matrices/heatmaps)

## Embedding in LaTeX Report

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/fgsm_examples_eps0.03.png}
  \caption{Clean (top) vs FGSM-perturbed (bottom) CIFAR-10 samples (ε=0.03). 
           Misclassifications highlighted in red.}
  \label{fig:fgsm_examples}
\end{figure}
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"**
- Use Docker method (`./scripts/generate_figures_docker.sh`)
- Or install locally: `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`

**"Out of memory"**
- Reduce batch size in script (edit line ~148: `batch_size=64` → `batch_size=32`)
- Use fewer samples: `--samples 8` instead of 12

**"CIFAR-10 download failed"**
- Check internet connection
- Or manually download and extract to `./data/cifar-10-batches-py/`

---

**Last Updated**: November 20, 2025  
**Phase**: 1 (MVP)  
**Repository**: https://github.com/DheerendraAchar/Cerberus
