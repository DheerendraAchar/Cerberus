#!/usr/bin/env python3
"""Train multiple architectures for transfer attack analysis.

This script trains 6 different architectures on CIFAR-10 with adversarial training:
1. ResNet-18
2. VGG-16
3. MobileNet V2
4. EfficientNet-B0
5. DenseNet-121
6. Vision Transformer (optional, requires timm)

Usage:
    python3 scripts/train_all_architectures.py \
        --epochs 50 \
        --batch-size 128 \
        --architectures resnet18,vgg16,mobilenet_v2,efficientnet_b0,densenet121
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from typing import Dict, Tuple, List
import time


def get_cifar10_loaders(batch_size: int = 128):
    """Get CIFAR-10 train and test loaders."""
    from torchvision import datasets, transforms
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def load_architecture(arch_name: str, num_classes: int = 10) -> nn.Module:
    """Load a pre-trained architecture (fine-tuned for CIFAR-10)."""
    
    if arch_name == 'resnet18':
        import torchvision.models as models
        model = models.resnet18(weights=None)
        # Adjust first layer for CIFAR-10 (32x32 images)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)
    
    elif arch_name == 'vgg16':
        import torchvision.models as models
        model = models.vgg16(weights=None)
        # Adjust first layer
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    
    elif arch_name == 'mobilenet_v2':
        import torchvision.models as models
        model = models.mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Linear(1280, num_classes)
    
    elif arch_name == 'efficientnet_b0':
        import torchvision.models as models
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(1280, num_classes)
    
    elif arch_name == 'densenet121':
        import torchvision.models as models
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, num_classes)
    
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    return model


def train_epoch(model: nn.Module, 
                train_loader, 
                criterion, 
                optimizer, 
                device: str,
                epsilon: float = 0.03,
                alpha: float = 0.5) -> Tuple[float, float]:
    """Train one epoch with adversarial training."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples using FGSM
        inputs.requires_grad = True
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        model.zero_grad()
        loss.backward()
        
        # FGSM perturbation
        data_grad = inputs.grad.data
        perturbed_data = inputs + epsilon * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        # Mix clean and adversarial examples
        batch_size = inputs.size(0)
        num_clean = int(batch_size * alpha)
        num_adv = batch_size - num_clean
        
        if num_clean > 0 and num_adv > 0:
            mixed_inputs = torch.cat([inputs[:num_clean], perturbed_data[num_clean:]], dim=0)
            mixed_targets = targets
        else:
            mixed_inputs = inputs
            mixed_targets = targets
        
        # Shuffle
        perm = torch.randperm(batch_size)
        mixed_inputs = mixed_inputs[perm]
        mixed_targets = mixed_targets[perm]
        
        # Train on mixed batch
        model.zero_grad()
        optimizer.zero_grad()
        outputs = model(mixed_inputs)
        loss = criterion(outputs, mixed_targets)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += mixed_targets.size(0)
        correct += predicted.eq(mixed_targets).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
    
    avg_loss = running_loss / len(train_loader)
    avg_acc = 100. * correct / total
    
    return avg_loss, avg_acc


def evaluate(model: nn.Module, test_loader, criterion, device: str) -> Tuple[float, float]:
    """Evaluate model on clean and adversarial test set."""
    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Clean accuracy
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct_clean += predicted.eq(targets).sum().item()
            
            # Adversarial accuracy (FGSM ε=0.03)
            inputs.requires_grad = True
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            
            adv_inputs = inputs + 0.03 * inputs.grad.sign()
            adv_inputs = torch.clamp(adv_inputs, 0, 1)
            
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            correct_adv += predicted.eq(targets).sum().item()
            
            total += targets.size(0)
    
    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    
    return clean_acc, adv_acc


def train_architecture(arch_name: str,
                      train_loader,
                      test_loader,
                      num_epochs: int = 50,
                      device: str = "cpu",
                      epsilon: float = 0.03,
                      alpha: float = 0.5) -> Dict:
    """Train one architecture with adversarial training."""
    
    print(f"\n{'='*70}")
    print(f"🏋️  Training {arch_name.upper()}")
    print(f"{'='*70}")
    
    # Load model
    model = load_architecture(arch_name).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'test_clean': [], 'test_adv': []}
    best_adv_acc = 0.0
    best_model_path = f"outputs/models/{arch_name}_adversarial.pt"
    
    # Training loop
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epsilon=epsilon, alpha=alpha
        )
        
        test_clean, test_adv = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_clean'].append(test_clean)
        history['test_adv'].append(test_adv)
        
        print(f"  Epoch [{epoch}/{num_epochs}] Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.2f}% | Clean: {test_clean:.2f}% | Adv: {test_adv:.2f}%")
        
        # Save best model
        if test_adv > best_adv_acc:
            best_adv_acc = test_adv
            Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"    💾 Best model saved (adversarial acc: {test_adv:.2f}%)")
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ {arch_name.upper()} training complete!")
    print(f"   Final clean accuracy:  {history['test_clean'][-1]:.2f}%")
    print(f"   Final adv accuracy:    {history['test_adv'][-1]:.2f}%")
    print(f"   Training time:         {elapsed:.1f}s")
    print(f"   Model saved to:        {best_model_path}")
    
    return {
        'architecture': arch_name,
        'final_clean_accuracy': history['test_clean'][-1],
        'final_adv_accuracy': history['test_adv'][-1],
        'training_time_seconds': elapsed,
        'model_path': best_model_path
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train multiple architectures for transfer analysis"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size"
    )
    parser.add_argument(
        "--architectures",
        type=str,
        default="resnet18,vgg16,mobilenet_v2,efficientnet_b0,densenet121",
        help="Comma-separated list of architectures to train"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Perturbation magnitude for adversarial training"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mix ratio (clean vs adversarial examples)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "🚀 "*20)
    print("CERBERUS PHASE 3 - MULTI-ARCHITECTURE TRAINING")
    print("🚀 "*20 + "\n")
    
    # Load data once
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    print(f"✅ Dataset loaded: {len(train_loader)} train batches, {len(test_loader)} test batches\n")
    
    # Parse architectures
    architectures = [a.strip() for a in args.architectures.split(',')]
    print(f"🏗️  Architectures to train: {', '.join(architectures)}\n")
    
    # Train each architecture
    results = []
    for arch in architectures:
        try:
            result = train_architecture(
                arch,
                train_loader,
                test_loader,
                num_epochs=args.epochs,
                device=args.device,
                epsilon=args.epsilon,
                alpha=args.alpha
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error training {arch}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    print(f"{'Architecture':<20} {'Clean Acc':<12} {'Adv Acc':<12} {'Time (s)':<12}")
    print("-"*70)
    
    for result in results:
        print(f"{result['architecture']:<20} "
              f"{result['final_clean_accuracy']:<12.2f}% "
              f"{result['final_adv_accuracy']:<12.2f}% "
              f"{result['training_time_seconds']:<12.1f}")
    
    print("="*70 + "\n")
    
    # Save results
    summary_path = "outputs/training_summary.json"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Training summary saved to: {summary_path}")
    print(f"✅ All models saved to: outputs/models/\n")


if __name__ == "__main__":
    main()
