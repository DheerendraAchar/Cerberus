import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from cerberus.dataset import get_cifar10_loaders
import os

os.makedirs('models', exist_ok=True)

print("Loading data...")
train_loader, val_loader = get_cifar10_loaders('./data', batch_size=128, num_workers=0)
print("OK")

print("Testing ResNet-18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 10)
model = model.to('cpu')

criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
for ep in range(1, 3):
    correct, total = 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to('cpu'), lbls.to('cpu')
        opt.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        opt.step()
        _, pred = out.max(1)
        correct += pred.eq(lbls).sum().item()
        total += lbls.size(0)
    
    acc = correct / total * 100
    print(f"Epoch {ep}: {acc:.1f}%")

print("OK")
