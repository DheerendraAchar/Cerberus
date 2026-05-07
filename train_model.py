#!/usr/bin/env python3
"""Quick training script for CIFAR-10 models"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import os

device = torch.device('cpu')

# Data setup
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print("Loading CIFAR-10...")
train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

architectures = ['resnet18', 'resnet50', 'vgg16']

for arch in architectures:
    print(f"\n{'='*60}")
    print(f"Training {arch}...")
    print('='*60)
    
    # Create model
    if arch == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(512, 10)
    elif arch == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(2048, 10)
    elif arch == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    model = model.to(device)
    
    # Freeze backbone, only train classifier
    for param in list(model.parameters())[:-2]:  # Freeze all but last layer
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few epochs
    for epoch in range(3):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] Loss: {total_loss/(batch_idx+1):.4f} Acc: {100.*correct/total:.2f}%')
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)
        
        print(f'Epoch {epoch+1} Test Accuracy: {100.*test_correct/test_total:.2f}%')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{arch}_cifar10.pt')
    print(f"Saved models/{arch}_cifar10.pt")

print("\nTraining complete!")
