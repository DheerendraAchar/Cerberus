"""
Adversarial Training for CIFAR-10 Models
This trains models to be robust against adversarial attacks
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from cerberus.dataset import get_cifar10_loaders
from cerberus.attacks import FSGMAttack
import os

CONFIG = {
    'epochs': 30,  # More epochs for adversarial training
    'batch_size': 128,  # Smaller batch for stability
    'lr': 0.001,  # Smaller learning rate
    'wd': 5e-4,
    'epsilon': 0.03,  # Adversarial perturbation budget
    'device': 'cpu'
}

os.makedirs('models', exist_ok=True)

def build_model(arch):
    """Build model from torchvision"""
    if arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, 10)
    elif arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(2048, 10)
    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True), 
            nn.Dropout(0.5), nn.Linear(512, 10)
        )
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

def adversarial_train_model(arch, train_loader, val_loader):
    """Train model with adversarial examples"""
    print(f"\n{'='*80}")
    print(f"ADVERSARIAL TRAINING: {arch.upper()}")
    print(f"{'='*80}")
    print(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['lr']} | Epsilon: {CONFIG['epsilon']}")
    
    model = build_model(arch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9, weight_decay=CONFIG['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # For FGSM attack generation
    attack = FSGMAttack(model=model, eps=CONFIG['epsilon'], device=CONFIG['device'])
    
    best_val_acc = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        train_clean_correct = 0
        train_adv_correct = 0
        train_total = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            # Generate adversarial examples
            with torch.enable_grad():
                adv_imgs = attack.generate(imgs, labels)
            
            # Train on clean images
            optimizer.zero_grad()
            clean_out = model(imgs)
            clean_loss = criterion(clean_out, labels)
            
            # Train on adversarial images
            adv_out = model(adv_imgs)
            adv_loss = criterion(adv_out, labels)
            
            # Combined loss: 50% clean + 50% adversarial
            total_loss = (clean_loss + adv_loss) / 2
            total_loss.backward()
            optimizer.step()
            
            # Track accuracy
            _, clean_pred = clean_out.max(1)
            _, adv_pred = adv_out.max(1)
            train_clean_correct += clean_pred.eq(labels).sum().item()
            train_adv_correct += adv_pred.eq(labels).sum().item()
            train_total += labels.size(0)
            
            if batch_idx % 20 == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx:3d} | Clean: {train_clean_correct/train_total*100:5.2f}% | Adv: {train_adv_correct/train_total*100:5.2f}%")
        
        scheduler.step()
        
        # Validation on clean data
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
                out = model(imgs)
                _, pred = out.max(1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total * 100
        train_clean_acc = train_clean_correct / train_total * 100
        train_adv_acc = train_adv_correct / train_total * 100
        
        print(f"Ep {epoch:2d}/{CONFIG['epochs']} | Train Clean: {train_clean_acc:6.2f}% | Train Adv: {train_adv_acc:6.2f}% | Val: {val_acc:6.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f'models/{arch}_robust_cifar10.pt'
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model to {model_path}")
    
    print(f"\n✓ Best validation accuracy: {best_val_acc:.2f}%\n")
    return model

if __name__ == '__main__':
    print("\n" + "="*80)
    print(" ADVERSARIAL TRAINING - CIFAR-10")
    print("="*80 + "\n")
    
    architectures = ['resnet18', 'resnet50', 'vgg16', 'mobilenetv2', 'efficientnetb0', 'densenet121']
    
    # Get data loaders (only train and test available)
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=CONFIG['batch_size'],
        num_workers=4
    )
    val_loader = test_loader  # Use test set for validation
    
    for arch in architectures:
        print(f"\n→ Training {arch}...")
        try:
            adversarial_train_model(arch, train_loader, val_loader)
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*80)
    print(" TRAINING COMPLETE!")
    print("="*80)
    print("\nModels saved as: models/{arch}_robust_cifar10.pt")
    print("These models are trained to resist adversarial attacks!")
    print("Defense test will now use these robust models automatically.\n")
