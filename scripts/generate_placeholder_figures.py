"""Generate placeholder figures for report (no PyTorch required).

This creates mockup figure placeholders you can use in your report immediately.
Replace with real figures later by running scripts/generate_figures.py with PyTorch.
"""
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(path='figures'):
    os.makedirs(path, exist_ok=True)
    return path

def generate_placeholder_grid(save_path):
    """Placeholder for clean vs adversarial examples."""
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        # Top row: "clean"
        axes[0, i].text(0.5, 0.5, f'Clean\nImage {i+1}', ha='center', va='center', fontsize=8)
        axes[0, i].set_xlim(0, 1)
        axes[0, i].set_ylim(0, 1)
        axes[0, i].axis('off')
        
        # Bottom row: "adversarial"
        color = 'red' if i % 3 == 0 else 'green'
        axes[1, i].text(0.5, 0.5, f'FGSM\nImage {i+1}', ha='center', va='center', fontsize=8, color=color)
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, 1)
        axes[1, i].axis('off')
    
    fig.suptitle('Clean (top) vs FGSM (bottom), eps=0.03 [PLACEHOLDER]', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def generate_placeholder_curve(save_path):
    """Placeholder for accuracy vs epsilon curve."""
    eps_values = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    # Simulated accuracy drop
    accuracy = [0.85, 0.82, 0.75, 0.65, 0.45, 0.30, 0.15]
    
    plt.figure(figsize=(6, 4))
    plt.plot(eps_values, accuracy, marker='o', linewidth=2, markersize=8)
    plt.xlabel('FGSM epsilon', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Accuracy vs FGSM epsilon [PLACEHOLDER - Simulated Data]', fontsize=12)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def generate_placeholder_confusion(save_path, title):
    """Placeholder confusion matrix."""
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Simulated confusion matrix
    if 'Clean' in title:
        cm = np.diag([950, 940, 920, 910, 930, 925, 945, 935, 955, 940])
        cm += np.random.randint(0, 20, (10, 10))
        np.fill_diagonal(cm, [950, 940, 920, 910, 930, 925, 945, 935, 955, 940])
    else:
        cm = np.diag([650, 620, 580, 540, 600, 560, 640, 610, 680, 630])
        cm += np.random.randint(10, 80, (10, 10))
        np.fill_diagonal(cm, [650, 620, 580, 540, 600, 560, 640, 610, 680, 630])
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'{title} [PLACEHOLDER]', fontsize=11)
    plt.xticks(range(10), classes, rotation=90, fontsize=7)
    plt.yticks(range(10), classes, fontsize=7)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def generate_placeholder_heatmap(save_path):
    """Placeholder perturbation heatmap."""
    # Simulated perturbation pattern
    x, y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
    diff = np.abs(np.sin(3*x) * np.cos(3*y)) * 0.03
    
    plt.figure(figsize=(4, 4))
    plt.imshow(diff, cmap='inferno')
    plt.title('FGSM Perturbation Heatmap (eps=0.03)\n[PLACEHOLDER]', fontsize=10)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def main():
    out_dir = ensure_dir('figures')
    print(f"Generating placeholder figures in {out_dir}/...")
    
    generate_placeholder_grid(os.path.join(out_dir, 'fgsm_examples_eps0.03.png'))
    print("[OK] fgsm_examples_eps0.03.png (placeholder)")
    
    generate_placeholder_curve(os.path.join(out_dir, 'fgsm_accuracy_vs_epsilon.png'))
    print("[OK] fgsm_accuracy_vs_epsilon.png (placeholder)")
    
    generate_placeholder_confusion(os.path.join(out_dir, 'confusion_clean.png'), 'Confusion Matrix (Clean)')
    print("[OK] confusion_clean.png (placeholder)")
    
    generate_placeholder_confusion(os.path.join(out_dir, 'confusion_fgsm_eps0.03.png'), 'Confusion Matrix (FGSM eps=0.03)')
    print("[OK] confusion_fgsm_eps0.03.png (placeholder)")
    
    generate_placeholder_heatmap(os.path.join(out_dir, 'fgsm_perturbation_heatmap_eps0.03.png'))
    print("[OK] fgsm_perturbation_heatmap_eps0.03.png (placeholder)")
    
    print("\n✅ Placeholder figures generated!")
    print("These are simulated visualizations for your report structure.")
    print("To generate REAL figures with actual model evaluations:")
    print("  Option 1: Run ./scripts/generate_figures_docker.sh (uses Docker)")
    print("  Option 2: Install PyTorch and run scripts/generate_figures.py")

if __name__ == '__main__':
    main()
