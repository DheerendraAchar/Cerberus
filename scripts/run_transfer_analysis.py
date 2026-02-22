#!/usr/bin/env python3
"""Generate 6x6 transfer attack matrix for Phase 3C analysis.

This script tests adversarial examples crafted on one architecture against all
other architectures. This produces a "transfer matrix" showing cross-model
attack effectiveness.

Matrix format:
- Rows: Source architectures (which model created the adversarial examples)
- Columns: Target architectures (which models are being tested)
- Values: Attack success rate (%) of adversarial examples on target models

Usage:
    python3 scripts/run_transfer_analysis.py \
        --source-models outputs/models/*.pt \
        --epsilon 0.03 \
        --num-samples 1000
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_model(model_path: str, architecture: str, device: str = "cpu") -> nn.Module:
    """Load a trained model."""
    from scripts.train_all_architectures import load_architecture
    
    model = load_architecture(architecture)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def get_test_loader(batch_size: int = 128, num_samples: int = None):
    """Get CIFAR-10 test loader."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Limit to num_samples if specified
    if num_samples:
        indices = np.random.choice(len(testset), min(num_samples, len(testset)), replace=False)
        testset = torch.utils.data.Subset(testset, indices)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return test_loader


def generate_adversarial_examples(source_model: nn.Module,
                                  test_loader,
                                  epsilon: float = 0.03,
                                  device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate adversarial examples using FGSM on source model."""
    
    all_adv_examples = []
    all_labels = []
    
    print("    Generating adversarial examples...")
    
    source_model.eval()
    for inputs, targets in tqdm(test_loader, desc="    FGSM", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs.requires_grad = True
        outputs = source_model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        source_model.zero_grad()
        loss.backward()
        
        # FGSM attack
        adv_inputs = inputs + epsilon * inputs.grad.sign()
        adv_inputs = torch.clamp(adv_inputs, 0, 1)
        
        all_adv_examples.append(adv_inputs.detach().cpu())
        all_labels.append(targets.detach().cpu())
    
    adv_examples = torch.cat(all_adv_examples, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return adv_examples, labels


def evaluate_on_adversarial(target_model: nn.Module,
                           adv_examples: torch.Tensor,
                           labels: torch.Tensor,
                           batch_size: int = 128,
                           device: str = "cpu") -> Tuple[float, float]:
    """Evaluate target model on adversarial examples."""
    
    target_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(adv_examples), batch_size):
            batch_adv = adv_examples[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size].to(device)
            
            outputs = target_model(batch_adv)
            _, predicted = outputs.max(1)
            
            correct += predicted.eq(batch_labels).sum().item()
            total += batch_labels.size(0)
    
    accuracy = 100. * correct / total
    attack_success = 100. - accuracy  # Attack succeeds if prediction is wrong
    
    return accuracy, attack_success


def build_transfer_matrix(model_paths: Dict[str, str],
                         architectures: List[str],
                         test_loader,
                         epsilon: float = 0.03,
                         device: str = "cpu") -> Dict:
    """Build the 6x6 transfer matrix."""
    
    num_architectures = len(architectures)
    transfer_matrix = np.zeros((num_architectures, num_architectures))
    accuracy_matrix = np.zeros((num_architectures, num_architectures))
    
    results = {}
    
    print(f"\n{'='*70}")
    print("🎯 BUILDING TRANSFER ATTACK MATRIX")
    print(f"{'='*70}")
    
    # For each source architecture
    for i, source_arch in enumerate(architectures):
        print(f"\n📍 Source: {source_arch}")
        
        source_model = load_model(model_paths[source_arch], source_arch, device)
        
        # Generate adversarial examples
        adv_examples, labels = generate_adversarial_examples(
            source_model, test_loader, epsilon, device
        )
        
        results[source_arch] = {}
        
        # Test against all target architectures
        for j, target_arch in enumerate(architectures):
            target_model = load_model(model_paths[target_arch], target_arch, device)
            
            accuracy, attack_success = evaluate_on_adversarial(
                target_model, adv_examples, labels, device=device
            )
            
            transfer_matrix[i, j] = attack_success
            accuracy_matrix[i, j] = accuracy
            
            results[source_arch][target_arch] = {
                'accuracy': accuracy,
                'attack_success_rate': attack_success
            }
            
            marker = "🎯" if i == j else ("✓" if attack_success > 50 else "✗")
            print(f"  → Target: {target_arch:<20} Attack: {attack_success:6.2f}%  {marker}")
    
    return transfer_matrix, accuracy_matrix, results


def plot_transfer_matrix(transfer_matrix: np.ndarray,
                        architectures: List[str],
                        output_path: str = "figures/transfer_matrix.png"):
    """Plot the transfer matrix as a heatmap."""
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(transfer_matrix, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn_r',
                xticklabels=architectures,
                yticklabels=architectures,
                cbar_kws={'label': 'Attack Success Rate (%)'},
                vmin=0, vmax=100)
    
    plt.title('Transfer Attack Matrix\n(Adversarial examples from source → target architecture)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Target Architecture', fontsize=12)
    plt.ylabel('Source Architecture', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Transfer matrix plot saved to: {output_path}")
    plt.close()


def plot_diagonal_analysis(transfer_matrix: np.ndarray,
                          architectures: List[str],
                          output_path: str = "figures/diagonal_analysis.png"):
    """Plot diagonal (self-attack) vs off-diagonal (transfer attack) analysis."""
    
    diagonal = np.diag(transfer_matrix)
    off_diagonal = []
    
    for i in range(len(architectures)):
        for j in range(len(architectures)):
            if i != j:
                off_diagonal.append(transfer_matrix[i, j])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Self-attack rates
    axes[0].bar(architectures, diagonal, color='crimson', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Attack Success Rate (%)', fontsize=11)
    axes[0].set_title('Self-Attack Rates (Diagonal)', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(diagonal):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
    
    # Transfer attack distribution
    axes[1].hist(off_diagonal, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(diagonal), color='crimson', linestyle='--', linewidth=2, label=f'Self-attack avg: {np.mean(diagonal):.1f}%')
    axes[1].axvline(np.mean(off_diagonal), color='steelblue', linestyle='--', linewidth=2, label=f'Transfer avg: {np.mean(off_diagonal):.1f}%')
    axes[1].set_xlabel('Attack Success Rate (%)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Transfer Attack Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Diagonal analysis plot saved to: {output_path}")
    plt.close()


def analyze_transfer_patterns(transfer_matrix: np.ndarray,
                             architectures: List[str]) -> Dict:
    """Analyze transfer attack patterns."""
    
    analysis = {
        'self_attack_rates': np.diag(transfer_matrix).tolist(),
        'mean_self_attack': float(np.mean(np.diag(transfer_matrix))),
        'mean_transfer_attack': float(np.mean(transfer_matrix[np.triu_indices_from(transfer_matrix, k=1)])),
        'most_transferable_source': None,
        'most_robust_target': None,
        'most_vulnerable_target': None
    }
    
    # Most transferable source (highest avg off-diagonal)
    off_diagonal_means = []
    for i in range(len(architectures)):
        mask = np.ones(len(architectures), dtype=bool)
        mask[i] = False
        off_diagonal_means.append(np.mean(transfer_matrix[i, mask]))
    
    most_transferable_idx = np.argmax(off_diagonal_means)
    analysis['most_transferable_source'] = {
        'architecture': architectures[most_transferable_idx],
        'avg_transfer_rate': float(off_diagonal_means[most_transferable_idx])
    }
    
    # Most robust target (lowest avg column)
    column_means = np.mean(transfer_matrix, axis=0)
    most_robust_idx = np.argmin(column_means)
    analysis['most_robust_target'] = {
        'architecture': architectures[most_robust_idx],
        'avg_attack_success': float(column_means[most_robust_idx])
    }
    
    # Most vulnerable target
    most_vulnerable_idx = np.argmax(column_means)
    analysis['most_vulnerable_target'] = {
        'architecture': architectures[most_vulnerable_idx],
        'avg_attack_success': float(column_means[most_vulnerable_idx])
    }
    
    return analysis


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate transfer attack matrix for Phase 3C"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/models",
        help="Directory containing trained model checkpoints"
    )
    parser.add_argument(
        "--architectures",
        type=str,
        default="resnet18,vgg16,mobilenet_v2,efficientnet_b0,densenet121",
        help="Comma-separated list of architectures"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Perturbation magnitude"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of test samples to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("\n" + "🔄 "*20)
    print("CERBERUS PHASE 3C - TRANSFER ATTACK ANALYSIS")
    print("🔄 "*20 + "\n")
    
    # Parse architectures
    architectures = [a.strip() for a in args.architectures.split(',')]
    
    # Find model files
    model_dir = Path(args.model_dir)
    model_paths = {}
    
    for arch in architectures:
        model_file = model_dir / f"{arch}_adversarial.pt"
        if model_file.exists():
            model_paths[arch] = str(model_file)
        else:
            print(f"⚠️  Warning: Model for {arch} not found at {model_file}")
    
    if len(model_paths) == 0:
        print("❌ Error: No models found. Train architectures first using:")
        print("   python3 scripts/train_all_architectures.py")
        return
    
    print(f"✅ Found models: {', '.join(model_paths.keys())}\n")
    
    # Load test data
    print("📊 Loading test data...")
    test_loader = get_test_loader(num_samples=args.num_samples)
    print(f"✅ Test loader ready ({args.num_samples} samples)\n")
    
    # Build transfer matrix
    transfer_matrix, accuracy_matrix, results = build_transfer_matrix(
        model_paths, architectures, test_loader,
        epsilon=args.epsilon, device=args.device
    )
    
    # Analyze patterns
    print(f"\n{'='*70}")
    print("📈 TRANSFER ATTACK ANALYSIS")
    print(f"{'='*70}")
    
    analysis = analyze_transfer_patterns(transfer_matrix, architectures)
    
    print(f"\n📊 Self-Attack Rates (diagonal):")
    for arch, rate in zip(architectures, analysis['self_attack_rates']):
        print(f"   {arch:<20}: {rate:6.2f}%")
    
    print(f"\n📈 Key Findings:")
    print(f"   Mean self-attack rate:      {analysis['mean_self_attack']:6.2f}%")
    print(f"   Mean transfer attack rate:  {analysis['mean_transfer_attack']:6.2f}%")
    print(f"   Most transferable source:   {analysis['most_transferable_source']['architecture']} "
          f"({analysis['most_transferable_source']['avg_transfer_rate']:.2f}%)")
    print(f"   Most robust target:         {analysis['most_robust_target']['architecture']} "
          f"({analysis['most_robust_target']['avg_attack_success']:.2f}% avg)")
    print(f"   Most vulnerable target:     {analysis['most_vulnerable_target']['architecture']} "
          f"({analysis['most_vulnerable_target']['avg_attack_success']:.2f}% avg)")
    
    # Save results
    print(f"\n{'='*70}")
    print("💾 SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save transfer matrix as JSON
    results['transfer_matrix'] = transfer_matrix.tolist()
    results['accuracy_matrix'] = accuracy_matrix.tolist()
    results['analysis'] = analysis
    
    results_path = "outputs/transfer_analysis.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {results_path}")
    
    # Generate plots
    plot_transfer_matrix(transfer_matrix, architectures)
    plot_diagonal_analysis(transfer_matrix, architectures)
    
    print(f"\n✅ Phase 3C transfer analysis complete!\n")


if __name__ == "__main__":
    main()
