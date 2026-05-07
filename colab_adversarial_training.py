# Cerberus Adversarial Training - Google Colab
# Run this in Google Colab for fast GPU training

# STEP 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 2: Navigate to project
import os
os.chdir('/content/drive/MyDrive/major_projekt')  # Adjust path if needed
print("Current directory:", os.getcwd())

# STEP 3: Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install flask flask-cors matplotlib numpy scipy scikit-learn Pillow -q

# STEP 4: Test GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch Version: {torch.__version__}")

# STEP 5: Install cerberus library
!pip install -e ./cerberus -q

# STEP 6: Run adversarial training
print("\n" + "="*70)
print("STARTING ADVERSARIAL TRAINING ON GPU")
print("="*70)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from cerberus.dataset import get_cifar10_loaders
from cerberus.attacks import FSGMAttack, PGDAttack

CONFIG = {
    'epochs': 15,
    'batch_size': 256,
    'lr': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

print(f"Using device: {CONFIG['device']}")

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

def train_adversarial(arch, train_loader, val_loader):
    """Train model with adversarial examples (robust training)"""
    model = build_model(arch)
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9, weight_decay=5e-4)
    
    # Setup attacks for adversarial training
    fgsm = FSGMAttack(model, eps=0.03)
    pgd = PGDAttack(model, eps=0.03, alpha=0.01, steps=5)
    
    best_acc = 0
    
    for ep in range(1, CONFIG['epochs'] + 1):
        model.train()
        tc, tt = 0, 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            
            # Generate adversarial examples
            with torch.no_grad():
                adv_imgs = fgsm.generate(imgs, lbls)
            
            # Train on BOTH clean and adversarial examples
            opt.zero_grad()
            
            # Clean loss
            out_clean = model(imgs)
            loss_clean = criterion(out_clean, lbls)
            
            # Adversarial loss
            out_adv = model(adv_imgs)
            loss_adv = criterion(out_adv, lbls)
            
            # Combined loss (robust training)
            loss = (loss_clean + loss_adv) / 2
            loss.backward()
            opt.step()
            
            _, pred = out_clean.max(1)
            tc += pred.eq(lbls).sum().item()
            tt += lbls.size(0)
        
        tacc = tc / tt * 100
        
        # Validation
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                out = model(imgs)
                _, pred = out.max(1)
                vc += pred.eq(lbls).sum().item()
                vt += lbls.size(0)
        
        vacc = vc / vt * 100
        print(f"[{arch:15s}] Ep {ep:2d}/{CONFIG['epochs']} | Train: {tacc:6.2f}% | Val: {vacc:6.2f}%")
        
        if vacc > best_acc:
            best_acc = vacc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/{arch}_robust_cifar10.pt')
            print(f"                   ✓ Saved robust model ({vacc:.2f}%)")
    
    return best_acc

# Load data
print("\nLoading CIFAR-10...")
train_loader, val_loader = get_cifar10_loaders('./data', batch_size=CONFIG['batch_size'], num_workers=2)
print("✓ Loaded\n")

# Train all models with adversarial training
results = {}
architectures = ['resnet18', 'resnet50', 'vgg16', 'mobilenetv2', 'efficientnetb0', 'densenet121']

for i, arch in enumerate(architectures, 1):
    print(f"\n[{i}/6] Training {arch.upper()} (ROBUST)")
    results[arch] = train_adversarial(arch, train_loader, val_loader)

print("\n" + "="*70)
print("FINAL RESULTS - ROBUST MODELS (Adversarial Training)")
print("="*70)
for arch, acc in results.items():
    print(f"{arch:20s}: {acc:6.2f}%")
print("="*70)

print("\n✓ Models saved to 'models/' - Download them!")
