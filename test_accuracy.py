import torch
import torchvision.models as models
from cerberus.dataset import get_cifar10_loaders

# Get test data
_, loader = get_cifar10_loaders('./data', batch_size=32, num_workers=0)
imgs, labels = next(iter(loader))

print("Testing models with ImageNet weights on CIFAR-10...")
print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}\n")

architectures = {
    'resnet18': (models.ResNet18_Weights.DEFAULT, 512),
    'resnet50': (models.ResNet50_Weights.DEFAULT, 2048),
    'vgg16': (models.VGG16_Weights.DEFAULT, 512),
    'mobilenetv2': (models.MobileNet_V2_Weights.DEFAULT, 1280),
    'efficientnetb0': (models.EfficientNet_B0_Weights.DEFAULT, 1280),
    'densenet121': (models.DenseNet121_Weights.DEFAULT, 1024)
}

for arch_name, (weights, fc_dim) in architectures.items():
    try:
        if arch_name == 'resnet18':
            model = models.resnet18(weights=weights)
            model.fc = torch.nn.Linear(fc_dim, 10)
        elif arch_name == 'resnet50':
            model = models.resnet50(weights=weights)
            model.fc = torch.nn.Linear(fc_dim, 10)
        elif arch_name == 'vgg16':
            model = models.vgg16(weights=weights)
            model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 10)
            )
        elif arch_name == 'mobilenetv2':
            model = models.mobilenet_v2(weights=weights)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(1280, 10)
            )
        elif arch_name == 'efficientnetb0':
            model = models.efficientnet_b0(weights=weights)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(1280, 10)
            )
        elif arch_name == 'densenet121':
            model = models.densenet121(weights=weights)
            model.classifier = torch.nn.Linear(1024, 10)
        
        model.eval()
        
        with torch.no_grad():
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            accuracy = correct / len(labels) * 100
        
        print(f"{arch_name:20s}: {correct:2d}/32 correct ({accuracy:5.1f}%)")
    except Exception as e:
        print(f"{arch_name:20s}: ERROR - {e}")
