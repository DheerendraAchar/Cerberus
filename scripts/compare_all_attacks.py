#!/usr/bin/env python3
"""Compare all attack types on the same model.

This script evaluates your model against all 5 attack types:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- C&W (Carlini & Wagner)
- DeepFool
- JSMA (Jacobian Saliency Map Attack)

Usage:
    python3 scripts/compare_all_attacks.py \
        --model outputs/models/baseline_model.pt \
        --epsilon 0.03
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def create_simple_resnet18():
    """Create a simple ResNet-18 for CIFAR-10."""
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
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    return SimpleResNet()


def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """Load a PyTorch model from checkpoint."""
    model = create_simple_resnet18()
    
    if Path(model_path).exists():
        print(f"📂 Loading model from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️  Model file not found at {model_path}")
        print(f"   Using randomly initialized model for demo")
    
    model = model.to(device)
    model.eval()
    return model


def get_test_loader(batch_size: int = 128):
    """Get CIFAR-10 test loader."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return test_loader


def evaluate_baseline(model: nn.Module, test_loader, device: str = "cpu") -> float:
    """Evaluate model accuracy on clean data."""
    print("\n" + "="*70)
    print("📊 BASELINE EVALUATION (Clean Accuracy)")
    print("="*70)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  [{batch_idx + 1}/{len(test_loader)}] Accuracy: {100. * correct / total:.2f}%")
    
    accuracy = 100.0 * correct / total
    print(f"\n✅ Baseline Accuracy: {accuracy:.2f}%")
    
    return accuracy


def run_all_attacks(model: nn.Module, test_loader, epsilon: float = 0.03, device: str = "cpu") -> Dict[str, Any]:
    """Run all attacks and return results."""
    
    from cerberus.attacks.pgd_attack import PGDAttack
    from cerberus.attacks.cw_attack import CWAttack
    from cerberus.attacks.deepfool_attack import DeepFoolAttack
    from cerberus.attacks.jsma_attack import JSMAAttack
    from cerberus.attacks import run_fgsm_attack
    
    results = {}
    
    # 1. FGSM (existing)
    print("\n" + "="*70)
    print("⚔️  ATTACK 1: FGSM (Fast Gradient Sign Method)")
    print("="*70)
    try:
        fgsm_results = run_fgsm_attack(model, test_loader, eps=epsilon, device=device)
        results['FGSM'] = fgsm_results
        print(f"✅ FGSM Attack Success Rate: {fgsm_results.get('attack_success_rate', 0):.2f}%")
    except Exception as e:
        print(f"❌ FGSM Error: {e}")
        results['FGSM'] = {'error': str(e)}
    
    # 2. PGD
    print("\n" + "="*70)
    print("⚔️  ATTACK 2: PGD (Projected Gradient Descent)")
    print("="*70)
    try:
        pgd = PGDAttack(model, eps=epsilon, eps_step=epsilon/3, max_iter=20, device=device)
        pgd_results = pgd.evaluate(test_loader)
        results['PGD'] = pgd_results
        print(f"✅ PGD Attack Success Rate: {pgd_results.get('attack_success_rate', 0):.2f}%")
    except Exception as e:
        print(f"❌ PGD Error: {e}")
        results['PGD'] = {'error': str(e)}
    
    # 3. C&W
    print("\n" + "="*70)
    print("⚔️  ATTACK 3: C&W (Carlini & Wagner)")
    print("="*70)
    try:
        cw = CWAttack(model, c=1.0, learning_rate=0.01, max_iterations=100, device=device)
        cw_results = cw.evaluate(test_loader)
        results['C&W'] = cw_results
        print(f"✅ C&W Attack Success Rate: {cw_results.get('attack_success_rate', 0):.2f}%")
    except Exception as e:
        print(f"❌ C&W Error: {e}")
        results['C&W'] = {'error': str(e)}
    
    # 4. DeepFool
    print("\n" + "="*70)
    print("⚔️  ATTACK 4: DeepFool")
    print("="*70)
    try:
        deepfool = DeepFoolAttack(model, max_iterations=100, overshoot=0.02, device=device)
        deepfool_results = deepfool.evaluate(test_loader)
        results['DeepFool'] = deepfool_results
        print(f"✅ DeepFool Attack Success Rate: {deepfool_results.get('attack_success_rate', 0):.2f}%")
    except Exception as e:
        print(f"❌ DeepFool Error: {e}")
        results['DeepFool'] = {'error': str(e)}
    
    # 5. JSMA
    print("\n" + "="*70)
    print("⚔️  ATTACK 5: JSMA (Jacobian Saliency Map Attack)")
    print("="*70)
    try:
        jsma = JSMAAttack(model, theta=1.0, max_pixels=100, device=device)
        jsma_results = jsma.evaluate(test_loader)
        results['JSMA'] = jsma_results
        print(f"✅ JSMA Attack Success Rate: {jsma_results.get('attack_success_rate', 0):.2f}%")
    except Exception as e:
        print(f"❌ JSMA Error: {e}")
        results['JSMA'] = {'error': str(e)}
    
    return results


def create_comparison_table(baseline_acc: float, results: Dict[str, Any]) -> str:
    """Create formatted comparison table."""
    
    table = "\n" + "="*90
    table += "\n📊 ATTACK COMPARISON RESULTS\n"
    table += "="*90 + "\n"
    table += f"{'Attack Type':<15} {'Accuracy':<12} {'Success Rate':<15} {'Time (s)':<12} {'Status':<10}\n"
    table += "-"*90 + "\n"
    
    for attack_name, result in results.items():
        if 'error' in result:
            status = "❌ Failed"
            accuracy = "N/A"
            success_rate = "N/A"
            time_sec = "N/A"
        else:
            status = "✅ Pass"
            accuracy = f"{result.get('accuracy', 0):.2f}%"
            success_rate = f"{result.get('attack_success_rate', 0):.2f}%"
            time_sec = f"{result.get('time_seconds', 0):.2f}"
        
        table += f"{attack_name:<15} {accuracy:<12} {success_rate:<15} {time_sec:<12} {status:<10}\n"
    
    table += "-"*90 + "\n"
    table += f"{'Baseline':<15} {baseline_acc:.2f}%{'':<6} N/A{'':<15} N/A{'':<12} ✅ Ref\n"
    table += "="*90 + "\n"
    
    return table


def create_comparison_plot(baseline_acc: float, results: Dict[str, Any], output_path: str = "figures/attack_comparison.png"):
    """Create comparison bar chart."""
    
    attack_names = []
    accuracies = []
    success_rates = []
    
    for attack_name, result in results.items():
        if 'error' not in result:
            attack_names.append(attack_name)
            accuracies.append(result.get('accuracy', 0))
            success_rates.append(result.get('attack_success_rate', 0))
    
    if not attack_names:
        print("⚠️  No successful attacks to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy under attack
    x_pos = np.arange(len(attack_names))
    colors = ['#ff7f0e' if acc < baseline_acc * 0.5 else '#2ca02c' for acc in accuracies]
    ax1.bar(x_pos, accuracies, color=colors, alpha=0.8)
    ax1.axhline(y=baseline_acc, color='blue', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.1f}%)')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Under Different Attacks', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(attack_names, rotation=45, ha='right')
    ax1.set_ylim([0, 100])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Attack success rate
    colors2 = ['#d62728' if sr > 50 else '#1f77b4' for sr in success_rates]
    ax2.bar(x_pos, success_rates, color=colors2, alpha=0.8)
    ax2.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax2.set_title('Attack Effectiveness (Success Rate)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(attack_names, rotation=45, ha='right')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare all attack types on a model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/baseline_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Perturbation magnitude for attacks"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Test batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/attack_comparison.json",
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    print("\n" + "🚀 "*20)
    print("CERBERUS PHASE 3 - ATTACK COMPARISON SUITE")
    print("🚀 "*20 + "\n")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = load_model(args.model, device=args.device)
    
    # Load test data
    print("Loading CIFAR-10 test set...")
    test_loader = get_test_loader(batch_size=args.batch_size)
    
    # Baseline evaluation
    baseline_acc = evaluate_baseline(model, test_loader, device=args.device)
    
    # Run all attacks
    results = run_all_attacks(model, test_loader, epsilon=args.epsilon, device=args.device)
    
    # Create comparison table
    table = create_comparison_table(baseline_acc, results)
    print(table)
    
    # Create comparison plot
    create_comparison_plot(baseline_acc, results)
    
    # Save results to JSON
    output_dict = {
        'baseline_accuracy': baseline_acc,
        'epsilon': args.epsilon,
        'device': args.device,
        'attack_results': results
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output}")
    
    # Summary
    print("\n" + "="*70)
    print("📈 SUMMARY")
    print("="*70)
    
    successful_attacks = [r for r in results.values() if 'error' not in r]
    if successful_attacks:
        avg_success = np.mean([r.get('attack_success_rate', 0) for r in successful_attacks])
        print(f"✅ Successful attacks: {len(successful_attacks)}/{len(results)}")
        print(f"📊 Average success rate: {avg_success:.2f}%")
        
        # Find strongest attack
        strongest_attack = max(successful_attacks, key=lambda x: x.get('attack_success_rate', 0))
        print(f"💪 Strongest attack: {[k for k, v in results.items() if v == strongest_attack][0]}")
    else:
        print("❌ No attacks succeeded")
    
    print("\n✨ Comparison complete!\n")


if __name__ == "__main__":
    main()
