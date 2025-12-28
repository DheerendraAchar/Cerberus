#!/usr/bin/env python3
"""Plot training curves and compare training methods.

This script visualizes training history from saved model checkpoints,
comparing baseline training vs adversarial training performance.
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
from typing import Dict, List, Optional


def load_training_history(checkpoint_path: str) -> Dict[str, List[float]]:
    """Load training history from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing training history
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'history' not in checkpoint:
        raise ValueError(f"No training history found in checkpoint: {checkpoint_path}")
    
    return checkpoint['history']


def plot_loss_curves(
    baseline_history: Optional[Dict] = None,
    adversarial_history: Optional[Dict] = None,
    save_path: str = "figures/training/loss_curves.png"
):
    """Plot training and test loss curves.
    
    Args:
        baseline_history: Training history from baseline model
        adversarial_history: Training history from adversarial model
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 5))
    
    # Training loss
    plt.subplot(1, 2, 1)
    if baseline_history:
        epochs = range(1, len(baseline_history['train_loss']) + 1)
        plt.plot(epochs, baseline_history['train_loss'], 'b-', label='Baseline', linewidth=2)
    if adversarial_history:
        epochs = range(1, len(adversarial_history['train_loss']) + 1)
        plt.plot(epochs, adversarial_history['train_loss'], 'r-', label='Adversarial', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Test loss
    plt.subplot(1, 2, 2)
    if baseline_history and 'test_loss' in baseline_history:
        epochs = range(1, len(baseline_history['test_loss']) + 1)
        plt.plot(epochs, baseline_history['test_loss'], 'b-', label='Baseline', linewidth=2)
    if adversarial_history and 'test_clean_acc' in adversarial_history:
        # Adversarial training doesn't track test_loss separately, skip
        pass
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Test Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss curves saved to {save_path}")
    plt.close()


def plot_accuracy_curves(
    baseline_history: Optional[Dict] = None,
    adversarial_history: Optional[Dict] = None,
    save_path: str = "figures/training/accuracy_curves.png"
):
    """Plot training and test accuracy curves.
    
    Args:
        baseline_history: Training history from baseline model
        adversarial_history: Training history from adversarial model
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 5))
    
    # Training accuracy
    plt.subplot(1, 2, 1)
    if baseline_history:
        epochs = range(1, len(baseline_history['train_acc']) + 1)
        plt.plot(epochs, baseline_history['train_acc'], 'b-', label='Baseline', linewidth=2)
    if adversarial_history:
        epochs = range(1, len(adversarial_history['train_acc']) + 1)
        plt.plot(epochs, adversarial_history['train_acc'], 'r-', label='Adversarial', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Test accuracy (clean)
    plt.subplot(1, 2, 2)
    if baseline_history and 'test_acc' in baseline_history:
        epochs = range(1, len(baseline_history['test_acc']) + 1)
        plt.plot(epochs, baseline_history['test_acc'], 'b-', label='Baseline (Clean)', linewidth=2)
    if adversarial_history and 'test_clean_acc' in adversarial_history:
        epochs = range(1, len(adversarial_history['test_clean_acc']) + 1)
        plt.plot(epochs, adversarial_history['test_clean_acc'], 'r-', label='Adversarial (Clean)', linewidth=2)
    if adversarial_history and 'test_adv_acc' in adversarial_history:
        epochs = range(1, len(adversarial_history['test_adv_acc']) + 1)
        plt.plot(epochs, adversarial_history['test_adv_acc'], 'r--', label='Adversarial (Adversarial)', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Test Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Accuracy curves saved to {save_path}")
    plt.close()


def plot_robustness_comparison(
    baseline_history: Optional[Dict] = None,
    adversarial_history: Optional[Dict] = None,
    save_path: str = "figures/training/robustness_comparison.png"
):
    """Plot robustness comparison between baseline and adversarial training.
    
    Args:
        baseline_history: Training history from baseline model
        adversarial_history: Training history from adversarial model
        save_path: Path to save the figure
    """
    if not adversarial_history or 'test_adv_acc' not in adversarial_history:
        print("⚠️  Adversarial test accuracy not available, skipping robustness plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot adversarial accuracy over epochs
    epochs = range(1, len(adversarial_history['test_adv_acc']) + 1)
    plt.plot(epochs, adversarial_history['test_adv_acc'], 'r-', 
             label='Adversarial Training', linewidth=2)
    
    # If we have baseline clean accuracy, show it as reference
    if baseline_history and 'test_acc' in baseline_history:
        baseline_epochs = range(1, len(baseline_history['test_acc']) + 1)
        plt.plot(baseline_epochs, baseline_history['test_acc'], 'b-', 
                 label='Baseline (Clean)', linewidth=2, alpha=0.7)
    
    # Add clean accuracy from adversarial training
    if 'test_clean_acc' in adversarial_history:
        plt.plot(epochs, adversarial_history['test_clean_acc'], 'g--', 
                 label='Adversarial Training (Clean)', linewidth=2, alpha=0.7)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Robustness: Adversarial vs Clean Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at 50% for reference
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50% Reference')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Robustness comparison saved to {save_path}")
    plt.close()


def plot_learning_rate(
    baseline_history: Optional[Dict] = None,
    adversarial_history: Optional[Dict] = None,
    save_path: str = "figures/training/learning_rate.png"
):
    """Plot learning rate schedule.
    
    Args:
        baseline_history: Training history from baseline model
        adversarial_history: Training history from adversarial model
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 5))
    
    if baseline_history and 'learning_rate' in baseline_history:
        epochs = range(1, len(baseline_history['learning_rate']) + 1)
        plt.plot(epochs, baseline_history['learning_rate'], 'b-', 
                 label='Baseline', linewidth=2)
    
    if adversarial_history and 'learning_rate' in adversarial_history:
        epochs = range(1, len(adversarial_history['learning_rate']) + 1)
        plt.plot(epochs, adversarial_history['learning_rate'], 'r-', 
                 label='Adversarial', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Learning rate schedule saved to {save_path}")
    plt.close()


def print_summary_statistics(
    baseline_history: Optional[Dict] = None,
    adversarial_history: Optional[Dict] = None
):
    """Print summary statistics from training history.
    
    Args:
        baseline_history: Training history from baseline model
        adversarial_history: Training history from adversarial model
    """
    print("\n" + "="*70)
    print("TRAINING SUMMARY STATISTICS")
    print("="*70)
    
    if baseline_history:
        print("\n📊 Baseline Training:")
        print(f"  Final Train Accuracy:  {baseline_history['train_acc'][-1]:.2f}%")
        print(f"  Final Test Accuracy:   {baseline_history['test_acc'][-1]:.2f}%")
        print(f"  Best Test Accuracy:    {max(baseline_history['test_acc']):.2f}%")
        print(f"  Final Train Loss:      {baseline_history['train_loss'][-1]:.4f}")
        print(f"  Final Test Loss:       {baseline_history['test_loss'][-1]:.4f}")
    
    if adversarial_history:
        print("\n📊 Adversarial Training:")
        print(f"  Final Train Accuracy:       {adversarial_history['train_acc'][-1]:.2f}%")
        print(f"  Final Test Clean Accuracy:  {adversarial_history['test_clean_acc'][-1]:.2f}%")
        print(f"  Final Test Adv Accuracy:    {adversarial_history['test_adv_acc'][-1]:.2f}%")
        print(f"  Best Test Clean Accuracy:   {max(adversarial_history['test_clean_acc']):.2f}%")
        print(f"  Best Test Adv Accuracy:     {max(adversarial_history['test_adv_acc']):.2f}%")
        print(f"  Final Train Loss:           {adversarial_history['train_loss'][-1]:.4f}")
        
        # Robustness improvement
        if baseline_history and 'test_acc' in baseline_history:
            baseline_clean = baseline_history['test_acc'][-1]
            adv_clean = adversarial_history['test_clean_acc'][-1]
            adv_robust = adversarial_history['test_adv_acc'][-1]
            
            print(f"\n🛡️  Robustness Analysis:")
            print(f"  Clean Accuracy Drop:        {baseline_clean - adv_clean:+.2f}%")
            print(f"  Adversarial Accuracy Gain:  {adv_robust:.2f}% (vs ~30-40% for baseline)")
            print(f"  Robustness/Clean Ratio:     {adv_robust/adv_clean:.2f}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves and compare training methods"
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        help="Path to baseline model checkpoint"
    )
    parser.add_argument(
        "--adversarial-checkpoint",
        type=str,
        help="Path to adversarial model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/training",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.baseline_checkpoint and not args.adversarial_checkpoint:
        print("❌ Error: Provide at least one checkpoint file")
        parser.print_help()
        return
    
    # Load training histories
    baseline_history = None
    adversarial_history = None
    
    if args.baseline_checkpoint:
        print(f"📂 Loading baseline checkpoint: {args.baseline_checkpoint}")
        baseline_history = load_training_history(args.baseline_checkpoint)
        print(f"   ✅ Loaded {len(baseline_history['train_acc'])} epochs")
    
    if args.adversarial_checkpoint:
        print(f"📂 Loading adversarial checkpoint: {args.adversarial_checkpoint}")
        adversarial_history = load_training_history(args.adversarial_checkpoint)
        print(f"   ✅ Loaded {len(adversarial_history['train_acc'])} epochs")
    
    print(f"\n📊 Generating training visualizations...")
    
    # Generate all plots
    plot_loss_curves(
        baseline_history, adversarial_history,
        save_path=os.path.join(args.output_dir, "loss_curves.png")
    )
    
    plot_accuracy_curves(
        baseline_history, adversarial_history,
        save_path=os.path.join(args.output_dir, "accuracy_curves.png")
    )
    
    plot_robustness_comparison(
        baseline_history, adversarial_history,
        save_path=os.path.join(args.output_dir, "robustness_comparison.png")
    )
    
    plot_learning_rate(
        baseline_history, adversarial_history,
        save_path=os.path.join(args.output_dir, "learning_rate.png")
    )
    
    # Print statistics
    print_summary_statistics(baseline_history, adversarial_history)
    
    print(f"✅ All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
