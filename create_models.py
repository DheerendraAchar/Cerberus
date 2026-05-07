"""Create models for VGG16, MobileNetV2, EfficientNet-B0, DenseNet-121"""
import torch
import torch.nn as nn
import torchvision.models as models
import os

os.makedirs('models', exist_ok=True)

architectures = {
    'vgg16': models.VGG16_Weights.DEFAULT,
    'mobilenetv2': models.MobileNet_V2_Weights.DEFAULT,
    'efficientnetb0': models.EfficientNet_B0_Weights.DEFAULT,
    'densenet121': models.DenseNet121_Weights.DEFAULT,
}

for arch_name, weights in architectures.items():
    print(f"Creating {arch_name}...")
    
    if arch_name == 'vgg16':
        model = models.vgg16(weights=weights)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    elif arch_name == 'mobilenetv2':
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 10)
        )
    elif arch_name == 'efficientnetb0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 10)
        )
    elif arch_name == 'densenet121':
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(1024, 10)
    
    save_path = f'models/{arch_name}_cifar10.pt'
    torch.save(model.state_dict(), save_path)
    print(f"✓ Saved {arch_name} to {save_path}")

print("\n✓ All models created!")
