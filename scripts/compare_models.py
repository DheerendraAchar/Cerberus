#!/usr/bin/env python3
"""Compare baseline and adversarially trained models.

This script evaluates and compares the robustness of baseline and
adversarially trained models under clean and adversarial conditions.
"""

import argparse
import os
import torch
import torch.nn as nn
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


def fgsm_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, 
                epsilon: float, device: str) -> torch.Tensor:
    """Generate FGSM adversarial examples.
    
    Args:
        model: Target model
        images: Clean images
        labels: True labels
        epsilon: Perturbation strength
        device: Device to run on
        
    Returns:
        Adversarial examples
    """
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
    
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial examples
    perturbed_images = images + epsilon * images.grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images.detach()


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: str,
    epsilon: float = 0.03,
    max_samples: int = 1000
) -> Dict[str, float]:
    """Evaluate model on clean and adversarial examples.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        epsilon: FGSM epsilon for adversarial examples
        max_samples: Maximum samples to evaluate (for speed)
        
    Returns:
        Dictionary with clean_acc and adv_acc
    """
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for images, labels in test_loader:
        if total >= max_samples:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            clean_correct += predicted.eq(labels).sum().item()
        
        # Adversarial accuracy
        adv_images = fgsm_attack(model, images, labels, epsilon, device)
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(labels).sum().item()
        
        total += labels.size(0)
    
    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'robustness_ratio': adv_acc / clean_acc if clean_acc > 0 else 0,
        'samples_evaluated': total
    }


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    from cerberus.cli import run_training  # Import to get model architecture
    import torch.nn as nn
    
    # Rebuild model architecture (same as in cli.py)
    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.relu(out)
            return out
    
    class SimpleResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(64, 64, 2, stride=1)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, num_classes)
        
        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            layers.append(BasicBlock(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(BasicBlock(out_channels, out_channels, 1))
            return nn.Sequential(*layers)
        
        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
    
    model = SimpleResNet(num_classes=10)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def plot_comparison(
    baseline_results: Dict[str, float],
    adversarial_results: Dict[str, float],
    epsilon: float,
    output_dir: str
):
    """Create comparison visualizations.
    
    Args:
        baseline_results: Results from baseline model
        adversarial_results: Results from adversarial model
        epsilon: FGSM epsilon used
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Bar chart comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    categories = ['Clean Accuracy', 'Adversarial Accuracy']
    baseline_vals = [baseline_results['clean_accuracy'], baseline_results['adversarial_accuracy']]
    adversarial_vals = [adversarial_results['clean_accuracy'], adversarial_results['adversarial_accuracy']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, adversarial_vals, width, label='Adversarial Training', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Model Accuracy Comparison (ε={epsilon})', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # Add value labels on bars
    for i, (b_val, a_val) in enumerate(zip(baseline_vals, adversarial_vals)):
        ax1.text(i - width/2, b_val + 1, f'{b_val:.1f}%', ha='center', fontsize=10)
        ax1.text(i + width/2, a_val + 1, f'{a_val:.1f}%', ha='center', fontsize=10)
    
    # Robustness ratio comparison
    models = ['Baseline', 'Adversarial\nTraining']
    robustness = [
        baseline_results['robustness_ratio'] * 100,
        adversarial_results['robustness_ratio'] * 100
    ]
    
    colors = ['#3498db', '#e74c3c']
    bars = ax2.bar(models, robustness, color=colors, alpha=0.8)
    
    ax2.set_ylabel('Robustness Ratio (%)', fontsize=12)
    ax2.set_title('Robustness: Adv/Clean Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 100])
    
    # Add value labels
    for bar, val in zip(bars, robustness):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1, 
                f'{val:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to {save_path}")
    plt.close()
    
    # Accuracy drop visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_drop = baseline_results['clean_accuracy'] - baseline_results['adversarial_accuracy']
    adversarial_drop = adversarial_results['clean_accuracy'] - adversarial_results['adversarial_accuracy']
    
    models = ['Baseline', 'Adversarial Training']
    drops = [baseline_drop, adversarial_drop]
    improvements = [0, baseline_drop - adversarial_drop]
    
    bars = ax.bar(models, drops, color=['#e74c3c', '#2ecc71'], alpha=0.7)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'Accuracy Drop Under Attack (ε={epsilon})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, drop, improvement in zip(bars, drops, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, drop + 0.5, 
                f'{drop:.1f}%', ha='center', fontsize=11, fontweight='bold')
        if improvement > 0:
            ax.text(bar.get_x() + bar.get_width()/2, drop/2,
                   f'↓{improvement:.1f}% better', ha='center', fontsize=10,
                   color='white', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'accuracy_drop.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Accuracy drop plot saved to {save_path}")
    plt.close()


def print_comparison_table(
    baseline_results: Dict[str, float],
    adversarial_results: Dict[str, float],
    epsilon: float
):
    """Print formatted comparison table.
    
    Args:
        baseline_results: Results from baseline model
        adversarial_results: Results from adversarial model
        epsilon: FGSM epsilon used
    """
    print("\n" + "="*80)
    print(" MODEL COMPARISON - ROBUSTNESS EVALUATION")
    print("="*80)
    print(f" Attack: FGSM with ε = {epsilon}")
    print(f" Samples Evaluated: {baseline_results['samples_evaluated']}")
    print("="*80)
    
    # Header
    print(f"\n{'Metric':<30} {'Baseline':<20} {'Adversarial':<20} {'Δ Improvement':<15}")
    print("-"*85)
    
    # Clean accuracy
    b_clean = baseline_results['clean_accuracy']
    a_clean = adversarial_results['clean_accuracy']
    delta_clean = a_clean - b_clean
    print(f"{'Clean Accuracy':<30} {b_clean:>6.2f}% {'':<12} {a_clean:>6.2f}% {'':<12} {delta_clean:>+6.2f}%")
    
    # Adversarial accuracy
    b_adv = baseline_results['adversarial_accuracy']
    a_adv = adversarial_results['adversarial_accuracy']
    delta_adv = a_adv - b_adv
    print(f"{'Adversarial Accuracy':<30} {b_adv:>6.2f}% {'':<12} {a_adv:>6.2f}% {'':<12} {delta_adv:>+6.2f}%")
    
    # Accuracy drop
    b_drop = b_clean - b_adv
    a_drop = a_clean - a_adv
    delta_drop = b_drop - a_drop
    print(f"{'Accuracy Drop':<30} {b_drop:>6.2f}% {'':<12} {a_drop:>6.2f}% {'':<12} {delta_drop:>+6.2f}%")
    
    # Robustness ratio
    b_ratio = baseline_results['robustness_ratio'] * 100
    a_ratio = adversarial_results['robustness_ratio'] * 100
    delta_ratio = a_ratio - b_ratio
    print(f"{'Robustness Ratio (Adv/Clean)':<30} {b_ratio:>6.2f}% {'':<12} {a_ratio:>6.2f}% {'':<12} {delta_ratio:>+6.2f}%")
    
    print("="*85)
    
    # Summary
    print("\n📊 SUMMARY:")
    if delta_adv > 0:
        print(f"   ✅ Adversarial training improved robustness by {delta_adv:.2f}%")
    else:
        print(f"   ⚠️  Adversarial training decreased robustness by {abs(delta_adv):.2f}%")
    
    if abs(delta_clean) < 5:
        print(f"   ✅ Clean accuracy maintained (Δ = {delta_clean:+.2f}%)")
    else:
        print(f"   ⚠️  Clean accuracy changed significantly (Δ = {delta_clean:+.2f}%)")
    
    print(f"   🛡️  Adversarial training reduced accuracy drop by {delta_drop:.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and adversarially trained models"
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        required=True,
        help="Path to baseline model checkpoint"
    )
    parser.add_argument(
        "--adversarial-checkpoint",
        type=str,
        required=True,
        help="Path to adversarial model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="CIFAR-10 data directory"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="FGSM epsilon for adversarial examples"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples to evaluate (for speed)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/comparison",
        help="Directory to save comparison plots"
    )
    
    args = parser.parse_args()
    
    # Import dataset utilities
    from cerberus.dataset import get_cifar10_loaders
    
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Load dataset
    print(f"📦 Loading CIFAR-10 dataset from {args.data_dir}...")
    _, test_loader = get_cifar10_loaders(root=args.data_dir, batch_size=64, num_workers=2)
    print(f"   ✅ Test batches: {len(test_loader)}")
    
    # Load models
    print(f"\n🔄 Loading baseline model from {args.baseline_checkpoint}...")
    baseline_model = load_model_from_checkpoint(args.baseline_checkpoint, device)
    print(f"   ✅ Baseline model loaded")
    
    print(f"\n🔄 Loading adversarial model from {args.adversarial_checkpoint}...")
    adversarial_model = load_model_from_checkpoint(args.adversarial_checkpoint, device)
    print(f"   ✅ Adversarial model loaded")
    
    # Evaluate baseline model
    print(f"\n📊 Evaluating baseline model...")
    baseline_results = evaluate_model(baseline_model, test_loader, device, args.epsilon, args.max_samples)
    print(f"   Clean Accuracy:       {baseline_results['clean_accuracy']:.2f}%")
    print(f"   Adversarial Accuracy: {baseline_results['adversarial_accuracy']:.2f}%")
    
    # Evaluate adversarial model
    print(f"\n📊 Evaluating adversarially trained model...")
    adversarial_results = evaluate_model(adversarial_model, test_loader, device, args.epsilon, args.max_samples)
    print(f"   Clean Accuracy:       {adversarial_results['clean_accuracy']:.2f}%")
    print(f"   Adversarial Accuracy: {adversarial_results['adversarial_accuracy']:.2f}%")
    
    # Print comparison table
    print_comparison_table(baseline_results, adversarial_results, args.epsilon)
    
    # Generate plots
    print(f"\n📊 Generating comparison plots...")
    plot_comparison(baseline_results, adversarial_results, args.epsilon, args.output_dir)
    
    print(f"\n✅ Comparison complete!")
    print(f"   Plots saved to: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
