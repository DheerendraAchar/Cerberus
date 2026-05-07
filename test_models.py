#!/usr/bin/env python3
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

device = torch.device('cpu')

# Load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

imgs, labels = next(iter(loader))
imgs = imgs.to(device)
labels = labels.to(device)

# Test ResNet-50
print("Testing ResNet-50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(2048, 10)
model = model.to(device).eval()

with torch.no_grad():
    out = model(imgs)
    preds = out.argmax(dim=1)
    correct = (preds == labels).sum().item()
    print(f"ResNet-50: {correct}/8")

# Test VGG16
print("\nTesting VGG16...")
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 10)
)
model = model.to(device).eval()

with torch.no_grad():
    out = model(imgs)
    preds = out.argmax(dim=1)
    correct = (preds == labels).sum().item()
    print(f"VGG16: {correct}/8")
