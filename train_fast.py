"""Fast fine-tuning - Only train classifier head (10x faster)"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from cerberus.dataset import get_cifar10_loaders
import os

CONFIG = {'epochs': 10, 'batch_size': 256, 'lr': 0.01, 'device': 'cpu'}
os.makedirs('models', exist_ok=True)

def build_model(arch):
    if arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, 10)
    elif arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(2048, 10)
    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, 10))
    elif arch == 'mobilenetv2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 10))
    elif arch == 'efficientnetb0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 10))
    elif arch == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(1024, 10)
    return model.to(CONFIG['device'])

def freeze_backbone(model, arch):
    """Freeze all layers except classifier"""
    if arch == 'resnet18' or arch == 'resnet50':
        for param in model.parameters():
            param.requires_grad = False
        model.fc.requires_grad_(True)
    elif arch == 'vgg16':
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif arch == 'mobilenetv2':
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif arch == 'efficientnetb0':
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif arch == 'densenet121':
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier.requires_grad_(True)

def train_model(arch, train_loader, val_loader):
    model = build_model(arch)
    freeze_backbone(model, arch)
    
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.SGD(params, lr=CONFIG['lr'], momentum=0.9)
    best_acc = 0
    
    for ep in range(1, CONFIG['epochs'] + 1):
        model.train()
        tc, tt = 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            opt.step()
            _, pred = out.max(1)
            tc += pred.eq(lbls).sum().item()
            tt += lbls.size(0)
        
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                out = model(imgs)
                _, pred = out.max(1)
                vc += pred.eq(lbls).sum().item()
                vt += lbls.size(0)
        
        tacc = tc / tt * 100
        vacc = vc / vt * 100
        print(f"[{arch:15s}] Ep {ep:2d}/{CONFIG['epochs']} | Train: {tacc:6.2f}% | Val: {vacc:6.2f}%", flush=True)
        
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), f'models/{arch}_cifar10.pt')
            print(f"                   ✓ Saved ({vacc:.2f}%)", flush=True)
    
    return best_acc

print("Loading CIFAR-10...", flush=True)
train_loader, val_loader = get_cifar10_loaders('./data', batch_size=CONFIG['batch_size'], num_workers=0)
print("✓ Loaded\n", flush=True)

results = {}
for i, arch in enumerate(['resnet18', 'resnet50', 'vgg16', 'mobilenetv2', 'efficientnetb0', 'densenet121'], 1):
    print(f"[{i}/6] {arch.upper()}", flush=True)
    results[arch] = train_model(arch, train_loader, val_loader)
    print()

print("\n" + "="*60)
print("FINAL RESULTS (Classifier-Only Fine-tuning)")
print("="*60)
for arch, acc in results.items():
    print(f"{arch:20s}: {acc:6.2f}%")
print("="*60)
