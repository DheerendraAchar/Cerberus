import torch
import torchvision.models as models
from torchvision import datasets, transforms
from cerberus.attacks import PGDAttack

# Load model (same as backend)
model = models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(512, 10)
model.eval()

# Get data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False)

# Get batch
imgs, labels = next(iter(loader))
print(f"Input shape: {imgs.shape}, Labels: {labels}")

# Try PGD
try:
    attack = PGDAttack(model=model, eps=0.03, eps_step=0.01, max_iter=2)
    adv = attack.generate_adversarial(imgs, labels)
    print(f"✓ PGD worked! Output shape: {adv.shape}")
except Exception as e:
    print(f"✗ PGD failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
