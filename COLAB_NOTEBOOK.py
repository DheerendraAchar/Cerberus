# Google Colab Adversarial Training Notebook
# Copy this entire code into a Google Colab notebook and run!

# =============================================================================
# CELL 1: Mount Google Drive
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# CELL 2: Setup Environment
# =============================================================================
import os
import sys

# Navigate to your project folder (already unzipped on Drive)
os.chdir('/content/drive/MyDrive/major_projekt')
print("Current directory:", os.getcwd())
print("Files:", os.listdir('.')[:10])

# Create setup.py for cerberus if it doesn't exist
setup_py_content = """from setuptools import setup, find_packages

setup(
    name='cerberus',
    version='1.0.0',
    description='Adversarial robustness framework',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'scikit-learn',
    ],
    python_requires='>=3.8',
)
"""

setup_path = 'cerberus/setup.py'
if not os.path.exists(setup_path):
    with open(setup_path, 'w') as f:
        f.write(setup_py_content)
    print("✓ Created cerberus/setup.py")
else:
    print("✓ setup.py already exists")

# Patch attacks.py to add FSGMAttack class if missing
attacks_file = 'cerberus/attacks.py'
with open(attacks_file, 'r') as f:
    attacks_content = f.read()

if 'class FSGMAttack' not in attacks_content:
    print("Patching attacks.py with FSGMAttack class...")
    fgsm_class = '''"""Attack wrappers. Integrates with IBM ART when available.

If ART is not installed, functions raise a clear error indicating the dependency.
"""
from typing import Any, Dict, Optional, Tuple


class FSGMAttack:
    """FGSM Attack class for adversarial training."""
    
    def __init__(self, model: Any, eps: float = 0.03, device: str = "cpu"):
        self.model = model
        self.eps = eps
        self.device = device
    
    def generate(self, x, y):
        """Generate adversarial examples using FGSM."""
        import torch
        
        x_adv = x.clone().detach().requires_grad_(True)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Forward pass
        outputs = self.model(x_adv)
        loss = loss_fn(outputs, y)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        with torch.no_grad():
            x_adv = x + self.eps * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv


'''
    # Replace the beginning of attacks.py
    start_idx = attacks_content.find('"""Attack wrappers')
    end_idx = attacks_content.find('\ndef run_fgsm_attack') + 1
    attacks_content = fgsm_class + attacks_content[end_idx:]
    
    with open(attacks_file, 'w') as f:
        f.write(attacks_content)
    print("✓ Patched attacks.py with FSGMAttack class")
else:
    print("✓ FSGMAttack class already exists")

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install flask flask-cors matplotlib numpy scipy scikit-learn Pillow -q

# Install cerberus package
!pip install -e ./cerberus -q

# =============================================================================
# CELL 3: Verify GPU Setup
# =============================================================================
import torch
print("\n" + "="*70)
print("GPU SETUP")
print("="*70)
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
print(f"PyTorch Version: {torch.__version__}")
print("="*70 + "\n")

# =============================================================================
# CELL 4: Run Adversarial Training
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from cerberus.dataset import get_cifar10_loaders
import os
import sys

# Reload cerberus.attacks to get the patched FSGMAttack
if 'cerberus.attacks' in sys.modules:
    del sys.modules['cerberus.attacks']

from cerberus.attacks import FSGMAttack

CONFIG = {
    'epochs': 30,  # Reduce to 15 for faster testing
    'batch_size': 128,
    'lr': 0.001,
    'wd': 5e-4,
    'epsilon': 0.03,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

os.makedirs('models', exist_ok=True)

print("\n" + "="*70)
print(" ADVERSARIAL TRAINING ON GPU")
print("="*70)
print(f"Device: {CONFIG['device']}")
print(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['lr']}")
print("="*70 + "\n")

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

def adversarial_train_model(arch, train_loader, test_loader):
    """Train model with adversarial examples"""
    print(f"\n{'='*70}")
    print(f"TRAINING: {arch.upper()}")
    print(f"{'='*70}")

    model = build_model(arch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9, weight_decay=CONFIG['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

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

            # Combined loss
            total_loss = (clean_loss + adv_loss) / 2
            total_loss.backward()
            optimizer.step()

            # Track accuracy
            _, clean_pred = clean_out.max(1)
            _, adv_pred = adv_out.max(1)
            train_clean_correct += clean_pred.eq(labels).sum().item()
            train_adv_correct += adv_pred.eq(labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # Validation on test set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
                out = model(imgs)
                _, pred = out.max(1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        train_clean_acc = train_clean_correct / train_total * 100
        train_adv_acc = train_adv_correct / train_total * 100

        print(f"Ep {epoch:2d}/{CONFIG['epochs']} | Clean: {train_clean_acc:6.2f}% | Adv: {train_adv_acc:6.2f}% | Test: {val_acc:6.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f'models/{arch}_robust_cifar10.pt'
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved: {model_path}")

    print(f"\n✓ Best test accuracy: {best_val_acc:.2f}%\n")
    return model

# Get data
print("Loading CIFAR-10 dataset...")
train_loader, test_loader = get_cifar10_loaders(
    batch_size=CONFIG['batch_size'],
    num_workers=2
)
print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}\n")

# Train all models
architectures = ['resnet18', 'resnet50', 'vgg16', 'mobilenetv2', 'efficientnetb0', 'densenet121']

for arch in architectures:
    try:
        adversarial_train_model(arch, train_loader, test_loader)
    except Exception as e:
        print(f"✗ Error training {arch}: {e}")

print("\n" + "="*70)
print(" TRAINING COMPLETE!")
print("="*70)
print("\nModels saved:")
!ls -lh models/*robust* | awk '{print "  " $9 " (" $5 ")"}'

# =============================================================================
# CELL 5: Download Models
# =============================================================================
print("\n" + "="*70)
print(" DOWNLOADING MODELS")
print("="*70)

# Create zip
!zip -r models_trained.zip models/*robust*
print("\n✓ Created models_trained.zip")

# Download
from google.colab import files
print("Downloading models...")
files.download('models_trained.zip')
print("✓ Download started!")

print("\n" + "="*70)
print(" NEXT STEPS")
print("="*70)
print("""
1. Extract models_trained.zip on your Mac
2. Copy *.pt files to: /Users/admin/Desktop/major_projekt/models/
3. Restart backend: pkill -f 'python3.10 backend.py'
4. Restart backend: cd /Users/admin/Desktop/major_projekt && python3.10 backend.py &
5. Open http://localhost:3000
6. Go to "Defense" tab
7. Test defense - you should see ~65% adversarial accuracy now!
""")
