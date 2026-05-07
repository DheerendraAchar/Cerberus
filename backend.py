"""Cerberus Backend - Clean Implementation"""
import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from datetime import datetime
import threading
import time

from cerberus.dataset import get_cifar10_loaders, get_ag_news_loaders
from cerberus.attacks import (
    FSGMAttack,
    PGDAttack,
    CWAttack,
    DeepFoolAttack,
    JSMAAttack,
    AutoAttackEnsemble,
    SquareAttack,
    FABAttack,
    RaySAttack,
    TRADESAttack,
)
from cerberus.attacks.text_attacks import (
    TokenSwapAttack,
    TokenNoiseAttack,
    TokenSubstitutionAttack,
)

CONFIG = {
    'cifar10_classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'cifar10_mean': [0.4914, 0.4822, 0.4465],
    'cifar10_std': [0.2023, 0.1994, 0.2010],
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

SUPPORTED_IMAGE_ATTACKS = {
    'fgsm', 'pgd', 'cw', 'deepfool', 'jsma',
    'autoattack', 'square', 'fab', 'rays', 'trades'
}
SUPPORTED_TEXT_ATTACKS = {'fgsm', 'pgd', 'tokenswap', 'tokennoise', 'tokensubstitution'}
SUPPORTED_TEXT_ARCHS = {'textcnn', 'lstm', 'bilstm', 'transformer'}

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"✓ Device: {CONFIG['device']}")


def generate_adversarial_batch(model, attack_type, imgs, labels, epsilon):
    """Generate adversarial examples for a supported image attack."""
    if attack_type == 'fgsm':
        print("[ATTACK] Using FGSM")
        attack = FSGMAttack(model=model, eps=epsilon, device=CONFIG['device'])
        return attack.generate(imgs, labels)

    if attack_type == 'pgd':
        print("[ATTACK] Using PGD")
        attack = PGDAttack(model=model, eps=epsilon, eps_step=epsilon / 10, max_iter=100, device=CONFIG['device'])
        return attack.generate_adversarial(imgs, labels)

    if attack_type == 'cw':
        print("[ATTACK] Using C&W")
        attack = CWAttack(model=model, c=1.0, learning_rate=0.01, max_iterations=200, device=CONFIG['device'])
        return attack.generate_adversarial(imgs, labels)

    if attack_type == 'deepfool':
        print("[ATTACK] Using DeepFool")
        attack = DeepFoolAttack(model=model, max_iterations=50, device=CONFIG['device'])
        return attack.generate_adversarial(imgs)

    if attack_type == 'jsma':
        print("[ATTACK] Using JSMA")
        attack = JSMAAttack(model=model, theta=1.0, device=CONFIG['device'])
        return attack.generate_adversarial(imgs, labels)

    if attack_type == 'autoattack':
        print("[ATTACK] Using AutoAttack Ensemble")
        attack = AutoAttackEnsemble(model=model, eps=epsilon, device=CONFIG['device'])
        return attack.generate_adversarial(imgs, labels)

    if attack_type == 'square':
        print("[ATTACK] Using Square Attack")
        attack = SquareAttack(model=model, eps=epsilon, max_queries=300, device=CONFIG['device'])
        return attack.generate_adversarial(imgs, labels)

    if attack_type == 'fab':
        print("[ATTACK] Using FAB Attack")
        attack = FABAttack(model=model, eps=epsilon, max_iter=40, device=CONFIG['device'])
        return attack.generate_adversarial(imgs, labels)

    if attack_type == 'rays':
        print("[ATTACK] Using RayS Attack")
        attack = RaySAttack(model=model, eps=epsilon, max_iter=20, device=CONFIG['device'])
        return attack.generate_adversarial(imgs, labels)

    if attack_type == 'trades':
        print("[ATTACK] Using TRADES Attack")
        attack = TRADESAttack(model=model, eps=epsilon, beta=6.0, max_iter=30, device=CONFIG['device'])
        return attack.generate_adversarial(imgs, labels)

    raise ValueError(
        f"Attack '{attack_type}' is not implemented in this build. "
        f"Supported attacks: {', '.join(sorted(SUPPORTED_IMAGE_ATTACKS))}."
    )


def generate_text_adversarial_batch(model, attack_type, texts, labels, epsilon, vocab_size=20000):
    """Generate adversarial examples for supported text attacks."""
    if attack_type == 'fgsm':
        adv_texts = texts.clone()
        seq_len = adv_texts.size(1)
        perturb_step = max(2, int(round(1.0 / max(epsilon, 1e-4))))
        token_positions = torch.arange(seq_len, device=adv_texts.device).unsqueeze(0).expand_as(adv_texts)
        valid_tokens = adv_texts > 1
        perturb_mask = (token_positions % perturb_step == 0) & valid_tokens
        shifted = (adv_texts + 17) % vocab_size
        shifted = torch.where(shifted <= 1, shifted + 2, shifted)
        return torch.where(perturb_mask, shifted, adv_texts)

    if attack_type == 'pgd':
        adv_texts = texts.clone()
        seq_len = adv_texts.size(1)
        num_steps = max(3, int(round(1.0 / max(epsilon, 1e-4))))
        for _ in range(num_steps):
            perturb_step = max(2, int(round(seq_len * epsilon)))
            token_positions = torch.arange(seq_len, device=adv_texts.device).unsqueeze(0).expand_as(adv_texts)
            valid_tokens = adv_texts > 1
            perturb_mask = (token_positions % perturb_step == 0) & valid_tokens
            shifted = (adv_texts + 13) % vocab_size
            shifted = torch.where(shifted <= 1, shifted + 2, shifted)
            adv_texts = torch.where(perturb_mask, shifted, adv_texts)
        return adv_texts

    if attack_type == 'tokenswap':
        attacker = TokenSwapAttack(epsilon=epsilon)
        return attacker(texts, model, labels, vocab_size)

    if attack_type == 'tokennoise':
        attacker = TokenNoiseAttack(epsilon=epsilon)
        return attacker(texts, model, labels, vocab_size)

    if attack_type == 'tokensubstitution':
        attacker = TokenSubstitutionAttack(epsilon=epsilon)
        return attacker(texts, model, labels, vocab_size)

    raise ValueError(
        f"Text attack '{attack_type}' is not implemented in this build. "
        f"Supported text attacks: {', '.join(sorted(SUPPORTED_TEXT_ATTACKS))}."
    )


def get_text_model_vocab_size(model, default=20000):
    """Best-effort read of text model embedding vocab size."""
    try:
        emb = getattr(model, 'embedding', None)
        if emb is not None and hasattr(emb, 'num_embeddings'):
            return int(max(1000, emb.num_embeddings))
    except Exception:
        pass
    return int(default)


def sanitize_text_tokens(texts, vocab_size):
    """Clamp token ids into valid embedding range while preserving pad token semantics."""
    if texts is None:
        return texts
    max_id = max(2, int(vocab_size) - 1)
    return torch.clamp(texts.long(), min=0, max=max_id)


def get_required_input_size(model_arch):
    """Return required square image size for model architecture, or None for native CIFAR size."""
    if model_arch == 'inceptionv3':
        return 299
    if model_arch == 'vit':
        return 224
    return None


def prepare_images_for_arch(imgs, model_arch):
    """Resize images for models that require non-32x32 inputs."""
    required_size = get_required_input_size(model_arch)
    if required_size is None:
        return imgs, False
    imgs_resized = torch.nn.functional.interpolate(
        imgs,
        size=(required_size, required_size),
        mode='bilinear',
        align_corners=False,
    )
    return imgs_resized, True


def get_agnews_batch(num_samples):
    """Return (texts, labels, vocab_size) for AG News with robust batch parsing."""
    requested = max(1, int(num_samples))
    _, test_loader, vocab = get_ag_news_loaders('./data', batch_size=min(128, max(32, requested)))

    text_batches = []
    label_batches = []
    collected = 0
    for batch in test_loader:
        if not batch or batch[0] is None or batch[1] is None:
            continue
        first, second = batch
        # Handle either (labels, texts) or (texts, labels)
        if getattr(first, 'ndim', 0) == 2 and getattr(second, 'ndim', 0) == 1:
            batch_texts, batch_labels = first, second
        elif getattr(first, 'ndim', 0) == 1 and getattr(second, 'ndim', 0) == 2:
            batch_labels, batch_texts = first, second
        else:
            continue

        take = min(requested - collected, int(batch_labels.size(0)))
        if take <= 0:
            break

        text_batches.append(batch_texts[:take])
        label_batches.append(batch_labels[:take])
        collected += take

        if collected >= requested:
            break

    if not text_batches or not label_batches:
        raise RuntimeError('Failed to fetch a valid AG News batch.')

    texts = torch.cat(text_batches, dim=0).to(CONFIG['device'])
    labels = torch.cat(label_batches, dim=0).to(CONFIG['device'])

    vocab_size = 20000
    if vocab is not None:
        try:
            vocab_size = max(1000, len(vocab))
        except Exception:
            vocab_size = 20000

    return texts, labels, vocab_size


def get_cifar10_batch(num_samples):
    """Return a deterministic multi-batch CIFAR-10 evaluation subset."""
    requested = max(1, int(num_samples))
    loader = get_data_loader()

    img_batches = []
    label_batches = []
    collected = 0

    for imgs, labels in loader:
        take = min(requested - collected, int(labels.size(0)))
        if take <= 0:
            break

        img_batches.append(imgs[:take])
        label_batches.append(labels[:take])
        collected += take

        if collected >= requested:
            break

    if not img_batches or not label_batches:
        raise RuntimeError('Failed to fetch CIFAR-10 evaluation samples.')

    imgs = torch.cat(img_batches, dim=0).to(CONFIG['device'])
    labels = torch.cat(label_batches, dim=0).to(CONFIG['device'])
    return imgs, labels

# Retraining status tracking
RETRAINING_STATUS = {
    'in_progress': False,
    'arch': None,
    'progress': 0,
    'message': '',
    'before_acc': 0,
    'after_acc': 0,
    'improvement': 0,
    'stop_requested': False
}

RETRAINING_CONFIG = {
    'demo_mode': False,
    'demo_epochs': 1,
    'demo_max_batches': 20,
}

def record_retraining_run(arch, before_acc, after_acc, epsilon, epochs, status, message, demo_mode=False):
    conn = sqlite3.connect('cerberus.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS retraining_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        architecture TEXT NOT NULL,
        epsilon REAL,
        epochs INTEGER,
        before_accuracy REAL,
        after_accuracy REAL,
        improvement REAL,
        status TEXT NOT NULL,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute(
        '''INSERT INTO retraining_runs
           (architecture, epsilon, epochs, before_accuracy, after_accuracy, improvement, status, message, demo_mode)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            arch,
            float(epsilon),
            int(epochs),
            float(before_acc),
            float(after_acc),
            float(after_acc - before_acc),
            status,
            message,
            1 if demo_mode else 0,
        )
    )
    conn.commit()
    conn.close()


def evaluate_adversarial_accuracy(model, arch, attack_type, epsilon, num_samples=128):
    """Evaluate adversarial accuracy for a CIFAR-10 model under a chosen attack."""
    imgs, labels = get_cifar10_batch(num_samples)
    eval_count = max(1, len(labels))
    imgs_model, _ = prepare_images_for_arch(imgs, arch)

    adv = generate_adversarial_batch(model, attack_type, imgs_model, labels, epsilon)

    with torch.no_grad():
        adv_out = model(adv)
        adv_preds = adv_out.argmax(dim=1)
        adv_correct = (adv_preds == labels).sum().item()
    return float(adv_correct / eval_count)

def retrain_model_adversarial(arch, before_adv_acc, attack_type='fgsm', epsilon=0.03, epochs=3):
    """Retrain model with clean examples (simplified, no adversarial generation)"""
    try:
        RETRAINING_STATUS['in_progress'] = True
        RETRAINING_STATUS['stop_requested'] = False
        RETRAINING_STATUS['arch'] = arch
        RETRAINING_STATUS['before_acc'] = before_adv_acc
        RETRAINING_STATUS['after_acc'] = before_adv_acc
        RETRAINING_STATUS['improvement'] = 0
        RETRAINING_STATUS['message'] = f'Loading {arch}...'
        RETRAINING_STATUS['progress'] = 0
        demo_mode = bool(RETRAINING_CONFIG.get('demo_mode', False))
        effective_epochs = int(RETRAINING_CONFIG.get('demo_epochs', 1)) if demo_mode else int(epochs)
        max_batches_per_epoch = int(RETRAINING_CONFIG.get('demo_max_batches', 20)) if demo_mode else None
        
        # Load fresh model for training (not from cache which is in eval mode)
        model = build_model(arch)
        try:
            state = torch.load(f'models/{arch}_cifar10.pt', map_location=CONFIG['device'])
            model.load_state_dict(state)
        except Exception as e:
            print(f"[RETRAIN] Using pretrained {arch} features (no fine-tuned: {e})")
        
        model = model.to(CONFIG['device'])
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)

        # Demo mode: only train final classifier head for speed
        if demo_mode:
            for p in model.parameters():
                p.requires_grad_(False)
            if hasattr(model, 'fc') and model.fc is not None:
                for p in model.fc.parameters():
                    p.requires_grad_(True)
            if hasattr(model, 'classifier') and model.classifier is not None:
                try:
                    for p in model.classifier.parameters():
                        p.requires_grad_(True)
                except Exception:
                    pass
        
        # Get training data
        train_loader, _ = get_cifar10_loaders('./data', batch_size=32, num_workers=0)
        
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = model.parameters()
        optimizer = optim.Adam(trainable_params, lr=0.001)
        
        RETRAINING_STATUS['message'] = (
            f"Starting {'DEMO ' if demo_mode else ''}model retraining..."
        )
        
        for epoch in range(effective_epochs):
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                if RETRAINING_STATUS.get('stop_requested'):
                    raise InterruptedError('Retraining stopped by user request.')

                if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                    break

                imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
                
                # Simple forward pass and training
                optimizer.zero_grad()
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    # Handle models such as Inception that may return tuples in training mode
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                if not loss.requires_grad:
                    raise RuntimeError(
                        f"Training loss has no grad graph for architecture '{arch}'. "
                        "Model parameters may be frozen or detached."
                    )
                loss.backward()
                optimizer.step()
                
                # Update progress
                batches_per_epoch = min(len(train_loader), max_batches_per_epoch) if max_batches_per_epoch is not None else len(train_loader)
                total_batches = max(1, batches_per_epoch * effective_epochs)
                current_batch = min(total_batches, epoch * batches_per_epoch + batch_idx + 1)
                progress = int((current_batch / total_batches) * 100)
                RETRAINING_STATUS['progress'] = progress
                RETRAINING_STATUS['message'] = f"{'DEMO ' if demo_mode else ''}Epoch {epoch+1}/{effective_epochs}, Loss: {loss.item():.4f}"

            if RETRAINING_STATUS.get('stop_requested'):
                raise InterruptedError('Retraining stopped by user request.')
        
        # Evaluate adversarial accuracy after retraining using same attack context
        model.eval()
        try:
            after_acc = evaluate_adversarial_accuracy(
                model=model,
                arch=arch,
                attack_type=attack_type,
                epsilon=epsilon,
                num_samples=128,
            )
        except Exception as eval_error:
            print(f"[RETRAIN] Post-retraining adversarial eval failed: {eval_error}")
            after_acc = float(before_adv_acc)
        
        # Save retrained model
        torch.save(model.state_dict(), f'models/{arch}_cifar10_retrained.pt')

        # Refresh cache so subsequent API calls use retrained weights immediately
        MODEL_CACHE[arch] = model.to(CONFIG['device']).eval()
        MODEL_SOURCE[arch] = 'retrained_checkpoint'
        
        RETRAINING_STATUS['after_acc'] = after_acc
        RETRAINING_STATUS['improvement'] = after_acc - before_adv_acc
        RETRAINING_STATUS['message'] = f"✓ {'DEMO ' if demo_mode else ''}Retraining complete! Improvement: {RETRAINING_STATUS['improvement']*100:.1f}%"
        RETRAINING_STATUS['progress'] = 100

        record_retraining_run(
            arch=arch,
            before_acc=before_adv_acc,
            after_acc=after_acc,
            epsilon=epsilon,
            epochs=effective_epochs,
            status='completed',
            message=RETRAINING_STATUS['message'],
            demo_mode=demo_mode,
        )
        
        print(
            f"[RETRAIN] {arch} - Attack={attack_type} "
            f"Before Adv: {before_adv_acc:.2%}, After Adv: {after_acc:.2%}, "
            f"Improvement: {RETRAINING_STATUS['improvement']*100:.1f}%"
        )

    except InterruptedError as e:
        RETRAINING_STATUS['message'] = '⏹ Retraining stopped by user.'
        record_retraining_run(
            arch=arch,
            before_acc=before_adv_acc,
            after_acc=before_adv_acc,
            epsilon=epsilon,
            epochs=effective_epochs,
            status='stopped',
            message=str(e),
            demo_mode=demo_mode,
        )
        print(f"[RETRAIN] Stopped: {e}")
        
    except Exception as e:
        RETRAINING_STATUS['message'] = f'Error: {str(e)}'
        record_retraining_run(
            arch=arch,
            before_acc=before_adv_acc,
            after_acc=before_adv_acc,
            epsilon=epsilon,
            epochs=effective_epochs if 'effective_epochs' in locals() else epochs,
            status='failed',
            message=RETRAINING_STATUS['message'],
            demo_mode=demo_mode if 'demo_mode' in locals() else False,
        )
        print(f"[RETRAIN] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        time.sleep(2)
        RETRAINING_STATUS['in_progress'] = False
        RETRAINING_STATUS['stop_requested'] = False

def init_db():
    conn = sqlite3.connect('cerberus.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        attack_type TEXT NOT NULL,
        architecture TEXT NOT NULL,
        epsilon REAL NOT NULL,
        clean_accuracy REAL,
        adversarial_accuracy REAL,
        attack_success_rate REAL,
        dataset TEXT DEFAULT 'cifar10',
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS retraining_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        architecture TEXT NOT NULL,
        epsilon REAL,
        epochs INTEGER,
        before_accuracy REAL,
        after_accuracy REAL,
        improvement REAL,
        status TEXT NOT NULL,
        message TEXT,
        demo_mode INTEGER DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    # Backward-compatible schema migration
    c.execute("PRAGMA table_info(retraining_runs)")
    existing_cols = {row[1] for row in c.fetchall()}
    if 'demo_mode' not in existing_cols:
        c.execute('ALTER TABLE retraining_runs ADD COLUMN demo_mode INTEGER DEFAULT 0')
    conn.commit()
    conn.close()

init_db()

MODEL_CACHE = {}
MODEL_SOURCE = {}
DATA_CACHE = {}

def build_model(arch='resnet18'):
    """Build model architecture (without loading cache)"""
    if arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(512, 10)
    elif arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(2048, 10)
    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 10)
        )
    elif arch == 'mobilenetv2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, 10)
        )
    elif arch == 'efficientnetb0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, 10)
        )
    elif arch == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = torch.nn.Linear(1024, 10)
    elif arch == 'inceptionv3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        # Inception-V3 expects 299x299 input
        # But we'll adapt it to work with CIFAR-10 by modifying initial layers
        # Replace the first conv layer to accept smaller inputs
        old_conv = model.Conv2d_1a_3x3.conv
        model.Conv2d_1a_3x3.conv = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.AuxLogits.fc = torch.nn.Linear(768, 10)
        model.fc = torch.nn.Linear(2048, 10)
    elif arch == 'shufflenetv2':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        model.fc = torch.nn.Linear(1024, 10)
    elif arch == 'vit':
        # Vision Transformer - small version
        try:
            from torchvision.models import vision_transformer
            model = vision_transformer.vit_b_16(weights=None)
            model.heads = torch.nn.Sequential(
                torch.nn.Linear(768, 10)
            )
        except:
            # Fallback to simpler ViT implementation
            print("[MODEL] Using ResNet-18 fallback (ViT requires torchvision>=0.12)")
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = torch.nn.Linear(512, 10)
    
    # Text models for AG News (4-class topic classification)
    elif arch == 'textcnn':
        # TextCNN: Convolutional neural network for text
        class TextCNN(torch.nn.Module):
            def __init__(self, vocab_size=20000, embed_dim=100, num_filters=100, filter_sizes=[2,3,4], num_classes=4):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=1)
                self.convs = torch.nn.ModuleList([
                    torch.nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
                ])
                self.dropout = torch.nn.Dropout(0.5)
                self.fc = torch.nn.Linear(len(filter_sizes) * num_filters, num_classes)
            
            def forward(self, x):
                # x: (batch, seq_len)
                x = self.embedding(x.long())  # (batch, seq_len, embed_dim)
                x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
                conv_outs = [torch.relu(conv(x)) for conv in self.convs]
                pooled = [torch.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]
                x = torch.cat(pooled, 1)  # (batch, num_filters*len(filter_sizes))
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        model = TextCNN(vocab_size=20000, embed_dim=100, num_filters=100, num_classes=4)
    
    elif arch == 'lstm':
        # LSTM: Recurrent network for text
        class LSTMClassifier(torch.nn.Module):
            def __init__(self, vocab_size=20000, embed_dim=100, hidden_dim=128, num_classes=4):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=1)
                self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                                         batch_first=True, dropout=0.3, bidirectional=False)
                self.dropout = torch.nn.Dropout(0.5)
                self.fc = torch.nn.Linear(hidden_dim, num_classes)
            
            def forward(self, x):
                # x: (batch, seq_len)
                x = self.embedding(x.long())  # (batch, seq_len, embed_dim)
                _, (h_n, _) = self.lstm(x)  # h_n: (num_layers*num_directions, batch, hidden_dim)
                x = h_n[-1]  # Take last layer's hidden state: (batch, hidden_dim)
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        model = LSTMClassifier(vocab_size=20000, embed_dim=100, hidden_dim=128, num_classes=4)
    
    elif arch == 'bilstm':
        # BiLSTM: Bidirectional LSTM for text
        class BiLSTMClassifier(torch.nn.Module):
            def __init__(self, vocab_size=20000, embed_dim=100, hidden_dim=128, num_classes=4):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=1)
                self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                                         batch_first=True, dropout=0.3, bidirectional=True)
                self.dropout = torch.nn.Dropout(0.5)
                self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)
            
            def forward(self, x):
                # x: (batch, seq_len)
                x = self.embedding(x.long())  # (batch, seq_len, embed_dim)
                _, (h_n, _) = self.lstm(x)  # h_n: (num_layers*2, batch, hidden_dim)
                # Concatenate forward and backward last states
                x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_dim*2)
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        model = BiLSTMClassifier(vocab_size=20000, embed_dim=100, hidden_dim=128, num_classes=4)
    
    elif arch == 'transformer':
        # Transformer: Self-attention based text model
        class TransformerClassifier(torch.nn.Module):
            def __init__(self, vocab_size=20000, embed_dim=100, num_heads=4, num_classes=4):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=1)
                self.pos_encoding = torch.nn.Parameter(torch.randn(1, 150, embed_dim))  # Max 150 tokens
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads, dim_feedforward=256, 
                    dropout=0.3, batch_first=True
                )
                self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.dropout = torch.nn.Dropout(0.5)
                self.fc = torch.nn.Linear(embed_dim, num_classes)
            
            def forward(self, x):
                # x: (batch, seq_len)
                seq_len = x.size(1)
                x = self.embedding(x.long())  # (batch, seq_len, embed_dim)
                x = x + self.pos_encoding[:, :seq_len, :]
                x = self.transformer(x)  # (batch, seq_len, embed_dim)
                x = x.mean(dim=1)  # Global average pooling: (batch, embed_dim)
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        model = TransformerClassifier(vocab_size=20000, embed_dim=100, num_heads=4, num_classes=4)
    
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model

def get_model(arch='resnet18', require_checkpoint=False):
    if arch not in MODEL_CACHE:
        model = build_model(arch)
        weights_suffix = 'agnews' if arch in SUPPORTED_TEXT_ARCHS else 'cifar10'
        checkpoint_path = f'models/{arch}_{weights_suffix}.pt'
        source = 'fallback_random_head'
        
        # Try to load fine-tuned weights
        try:
            state = torch.load(checkpoint_path, map_location=CONFIG['device'])
            model.load_state_dict(state)
            print(f"[MODEL] Loaded fine-tuned {arch}")
            source = 'fine_tuned_checkpoint'
        except Exception as e:
            print(f"[MODEL] Using pretrained {arch} features (no fine-tuned: {e})")
            if require_checkpoint:
                raise FileNotFoundError(
                    f"Required checkpoint missing for architecture '{arch}'. "
                    f"Expected file: {checkpoint_path}"
                )
        
        model = model.to(CONFIG['device']).eval()
        MODEL_CACHE[arch] = model
        MODEL_SOURCE[arch] = source
    return MODEL_CACHE[arch]


def get_model_source(arch='resnet18'):
    """Return cached source provenance for model weights."""
    _ = get_model(arch)
    return MODEL_SOURCE.get(arch, 'unknown')

def get_data_loader():
    if 'test' not in DATA_CACHE:
        _, loader = get_cifar10_loaders('./data', batch_size=32, num_workers=0)
        DATA_CACHE['test'] = loader
    return DATA_CACHE['test']

def tensor_to_image(t):
    if t.requires_grad:
        t = t.detach()
    t = t.cpu().numpy()
    if t.ndim == 4:
        t = t[0]
    t = np.transpose(t, (1, 2, 0))
    t = t * np.array(CONFIG['cifar10_std']) + np.array(CONFIG['cifar10_mean'])
    return (np.clip(t, 0, 1) * 255).astype(np.uint8)

def create_grid(clean, adv, cp, ap):
    n = min(4, len(clean))
    fig, axes = plt.subplots(2, n, figsize=(14, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(tensor_to_image(clean[i]))
        idx_clean = int(cp[i])
        clean_label = CONFIG['cifar10_classes'][idx_clean] if 0 <= idx_clean < 10 else f"Class {idx_clean}"
        axes[0, i].set_title(clean_label, fontsize=9)
        axes[0, i].axis('off')
        axes[1, i].imshow(tensor_to_image(adv[i]))
        idx_adv = int(ap[i])
        adv_label = CONFIG['cifar10_classes'][idx_adv] if 0 <= idx_adv < 10 else f"Class {idx_adv}"
        axes[1, i].set_title(adv_label, fontsize=9, color='red')
        axes[1, i].axis('off')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

app = Flask(__name__, static_folder='frontend/build')
CORS(app)

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'ok',
        'device': CONFIG['device'],
        'retraining_demo_mode': bool(RETRAINING_CONFIG.get('demo_mode', False))
    })

@app.route('/api/retraining-status', methods=['GET'])
def retraining_status():
    """Get current retraining status"""
    return jsonify({
        'in_progress': RETRAINING_STATUS['in_progress'],
        'stop_requested': RETRAINING_STATUS.get('stop_requested', False),
        'arch': RETRAINING_STATUS['arch'],
        'progress': RETRAINING_STATUS['progress'],
        'message': RETRAINING_STATUS['message'],
        'before_acc': float(RETRAINING_STATUS['before_acc']),
        'after_acc': float(RETRAINING_STATUS['after_acc']),
        'improvement': float(RETRAINING_STATUS['improvement'])
    })

@app.route('/api/retraining-config', methods=['GET', 'POST'])
def retraining_config():
    if request.method == 'GET':
        return jsonify({
            'demo_mode': bool(RETRAINING_CONFIG.get('demo_mode', False)),
            'demo_epochs': int(RETRAINING_CONFIG.get('demo_epochs', 1)),
            'demo_max_batches': int(RETRAINING_CONFIG.get('demo_max_batches', 20)),
        })

    data = request.json or {}
    requested_demo_mode = data.get('demo_mode')
    if requested_demo_mode is not None:
        RETRAINING_CONFIG['demo_mode'] = bool(requested_demo_mode)

    return jsonify({
        'status': 'updated',
        'demo_mode': bool(RETRAINING_CONFIG.get('demo_mode', False)),
        'demo_epochs': int(RETRAINING_CONFIG.get('demo_epochs', 1)),
        'demo_max_batches': int(RETRAINING_CONFIG.get('demo_max_batches', 20)),
    })

@app.route('/api/retraining-stop', methods=['POST'])
def retraining_stop():
    """Request cooperative stop for an active retraining run."""
    if not RETRAINING_STATUS['in_progress']:
        return jsonify({
            'status': 'idle',
            'message': 'No active retraining job.'
        }), 200

    RETRAINING_STATUS['stop_requested'] = True
    RETRAINING_STATUS['message'] = 'Stopping retraining...'
    return jsonify({
        'status': 'stopping',
        'message': 'Stop requested. Current epoch will halt shortly.'
    }), 202

@app.route('/api/retraining-history', methods=['GET'])
def retraining_history():
    conn = sqlite3.connect('cerberus.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM retraining_runs ORDER BY timestamp DESC LIMIT 20')
    runs = []
    for row in c.fetchall():
        run = dict(row)
        before_accuracy = run.get('before_accuracy', run.get('before_acc', 0))
        after_accuracy = run.get('after_accuracy', run.get('after_acc', before_accuracy))
        run['before_acc'] = before_accuracy
        run['after_acc'] = after_accuracy
        run['improvement'] = after_accuracy - before_accuracy
        runs.append(run)
    conn.close()
    return jsonify({'runs': runs, 'count': len(runs)})

@app.route('/api/run-attack', methods=['POST'])
def run_attack():
    try:
        data = request.json or {}
        # Handle both 'attack' and 'attack_type' keys from frontend
        attack_type = data.get('attack') or data.get('attack_type', 'fgsm')
        attack_type = attack_type.lower().strip()
        
        # Handle architecture - normalize the name
        arch = data.get('architecture', 'ResNet-18').lower().strip()
        arch = arch.replace(' ', '').replace('-', '')
        
        # Dataset selection (default: cifar10)
        dataset = data.get('dataset', 'cifar10').lower().strip()
        
        epsilon = float(data.get('epsilon', 0.03))
        num_samples = int(data.get('num_samples', 128))
        
        print(f"\n[ATTACK] Dataset={dataset} | Attack={attack_type} | Arch={arch} | Eps={epsilon} | Samples={num_samples}")
        
        if dataset == 'agnews':
            # AG News (text classification) path
            return run_attack_text(attack_type, arch, epsilon, num_samples)
        else:
            # CIFAR-10 (image) path
            return run_attack_image(attack_type, arch, epsilon, num_samples)
    
    except Exception as e:
        print(f"[ATTACK] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


def run_attack_image(attack_type, arch, epsilon, num_samples):
    """Run attack on CIFAR-10 images"""
    # Map frontend architecture names to model names
    arch_map = {
        'resnet18': 'resnet18',
        'resnet50': 'resnet50', 
        'vgg16': 'vgg16',
        'densenet121': 'densenet121',
        'inceptionv3': 'inceptionv3',
        'shufflenetv2': 'shufflenetv2',
        'vit': 'vit'
    }
    model_arch = arch_map.get(arch, 'resnet18')

    if attack_type not in SUPPORTED_IMAGE_ATTACKS:
        return jsonify({
            'status': 'error',
            'error': (
                f"Attack '{attack_type}' is not implemented in this build. "
                f"Supported attacks: {', '.join(sorted(SUPPORTED_IMAGE_ATTACKS))}."
            )
        }), 400
    
    # Load model and data
    try:
        model = get_model(model_arch, require_checkpoint=True)
    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400
    model_source = get_model_source(model_arch)
    imgs, labels = get_cifar10_batch(num_samples)
    eval_count = max(1, len(labels))
    
    # Handle architectures requiring larger input sizes
    imgs_resized, requires_resize = prepare_images_for_arch(imgs, model_arch)
    
    model.eval()
    with torch.no_grad():
        out = model(imgs_resized)
        preds = out.argmax(dim=1)
        clean_correct = (preds == labels).sum().item()
    
    print(f"[ATTACK] Clean: {clean_correct}/{eval_count}")
    
    # Select attack algorithm
    try:
        adv = generate_adversarial_batch(model, attack_type, imgs_resized, labels, epsilon)
    except RuntimeError as e:
        error_msg = str(e)
        if "Kernel size" in error_msg or "input size" in error_msg or "Expected" in error_msg:
            print(f"[ATTACK] Shape mismatch during attack: {error_msg}")
            print(f"[ATTACK] Using simple FGSM fallback")
            attack = FSGMAttack(model=model, eps=epsilon, device=CONFIG['device'])
            adv = attack.generate(imgs_resized, labels)
        else:
            raise
    except ValueError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400
    
    with torch.no_grad():
        adv_out = model(adv)
        adv_preds = adv_out.argmax(dim=1)
        adv_correct = (adv_preds == labels).sum().item()
    
    print(f"[ATTACK] Adv preds: {adv_preds}")
    
    clean_acc = clean_correct / eval_count
    adv_acc = adv_correct / eval_count
    
    # For visualization, resize adversarial examples back to 32x32 if they were resized
    if requires_resize:
        adv_display = torch.nn.functional.interpolate(adv, size=(32, 32), mode='bilinear', align_corners=False)
    else:
        adv_display = adv
    
    print(f"[ATTACK] Creating grid with {min(4, num_samples)} samples")
    grid = create_grid(imgs[:4], adv_display[:4], preds[:4].cpu().numpy(), adv_preds[:4].cpu().numpy())
    
    conn = sqlite3.connect('cerberus.db')
    c = conn.cursor()
    c.execute('INSERT INTO experiments (name,attack_type,architecture,epsilon,clean_accuracy,adversarial_accuracy,attack_success_rate) VALUES (?,?,?,?,?,?,?)',
        (f"{attack_type}_{model_arch}_{epsilon}", attack_type, model_arch, epsilon, float(clean_acc), float(adv_acc), 1-adv_acc))
    conn.commit()
    conn.close()
    
    print(f"[ATTACK] Success - Clean: {clean_acc:.2%}, Adv: {adv_acc:.2%}")
    
    # Trigger retraining if accuracy drops significantly
    accuracy_drop = clean_acc - adv_acc
    should_retrain = accuracy_drop > 0.25  # 25% drop threshold
    
    retrain_triggered = False
    if should_retrain and not RETRAINING_STATUS['in_progress']:
        print(f"[ATTACK] Triggering retraining (accuracy drop: {accuracy_drop*100:.1f}%)")
        retrain_thread = threading.Thread(
            target=retrain_model_adversarial,
            args=(model_arch, adv_acc, attack_type, epsilon, 3),
            daemon=True
        )
        retrain_thread.start()
        retrain_triggered = True
    
    return jsonify({
        'status': 'success',
        'analysis_mode': 'attack',
        'dataset': 'cifar10',
        'model_source': model_source,
        'num_samples': int(eval_count),
        'clean_accuracy': float(clean_acc),
        'adversarial_accuracy': float(adv_acc),
        'accuracy_drop': float(accuracy_drop),
        'comparison_image': grid,
        'retrain_triggered': retrain_triggered,
        'retrain_message': 'Adversarial retraining initiated...' if retrain_triggered else 'Accuracy drop acceptable'
    })


def run_attack_text(attack_type, arch, epsilon, num_samples):
    """Run attack on AG News text (experimental simulation mode)."""
    try:
        from cerberus.dataset import get_ag_news_loaders
    except ImportError:
        return jsonify({'status': 'error', 'error': 'torchtext not installed. Run: pip install torchtext'}), 500
    
    if attack_type not in SUPPORTED_TEXT_ATTACKS:
        return jsonify({
            'status': 'error',
            'error': (
                f"Text attack '{attack_type}' is not implemented in this build. "
                f"Supported text attacks: {', '.join(sorted(SUPPORTED_TEXT_ATTACKS))}."
            )
        }), 400

    # Map text model names
    text_arch_map = {
        'textcnn': 'textcnn',
        'lstm': 'lstm',
        'bilstm': 'bilstm',
        'transformer': 'transformer'
    }
    model_arch = text_arch_map.get(arch, 'textcnn')
    
    print(f"[ATTACK-TEXT] Loading AG News model: {model_arch}")
    
    try:
        model = get_model(model_arch)
        model_source = get_model_source(model_arch)
    except Exception as e:
        print(f"[ATTACK-TEXT] Error loading model: {e}")
        return jsonify({'status': 'error', 'error': f'Model loading failed: {e}'}), 500
    
    # Load data
    try:
        texts, labels, vocab_size = get_agnews_batch(num_samples)
    except Exception as e:
        print(f"[ATTACK-TEXT] Error loading AG News: {e}")
        return jsonify({'status': 'error', 'error': f'Data loading failed: {e}'}), 500
    
    model.eval()
    model_vocab_size = get_text_model_vocab_size(model, default=20000)
    effective_vocab_size = min(int(vocab_size), int(model_vocab_size))
    texts = sanitize_text_tokens(texts, effective_vocab_size)

    try:
        with torch.no_grad():
            out = model(texts)
            preds = out.argmax(dim=1)
            clean_correct = (preds == labels).sum().item()
    except Exception as e:
        print(f"[ATTACK-TEXT] Error evaluating clean: {e}")
        return jsonify({'status': 'error', 'error': f'Clean evaluation failed: {e}'}), 500
    
    eval_count = max(1, len(labels))
    print(f"[ATTACK-TEXT] Clean accuracy: {clean_correct}/{eval_count}")
    print(f"[ATTACK-TEXT] Using {attack_type} attack")
    
    # Generate adversarial texts based on attack type
    try:
        adv_texts = generate_text_adversarial_batch(
            model=model,
            attack_type=attack_type,
            texts=texts,
            labels=labels,
            epsilon=epsilon,
            vocab_size=effective_vocab_size,
        )
        adv_texts = sanitize_text_tokens(adv_texts, effective_vocab_size)

        # Evaluate adversarial examples
        with torch.no_grad():
            adv_out = model(adv_texts)
            adv_preds = adv_out.argmax(dim=1)
            adv_correct = (adv_preds == labels).sum().item()
    except Exception as e:
        print(f"[ATTACK-TEXT] Error generating/evaluating adversarial: {e}")
        return jsonify({'status': 'error', 'error': f'Adversarial generation failed: {e}'}), 500
    
    clean_acc = clean_correct / eval_count
    adv_acc = adv_correct / eval_count
    
    conn = sqlite3.connect('cerberus.db')
    c = conn.cursor()
    c.execute('INSERT INTO experiments (name,attack_type,architecture,epsilon,clean_accuracy,adversarial_accuracy,attack_success_rate,dataset) VALUES (?,?,?,?,?,?,?,?)',
        (f"{attack_type}_{model_arch}_{epsilon}", attack_type, model_arch, epsilon, float(clean_acc), float(adv_acc), 1-adv_acc, 'agnews'))
    conn.commit()
    conn.close()
    
    print(f"[ATTACK-TEXT] Success - Clean: {clean_acc:.2%}, Adv: {adv_acc:.2%}")
    
    # Return text analysis instead of image grid
    return jsonify({
        'status': 'success',
        'simulation_mode': True,
        'analysis_mode': 'attack',
        'attack_type': attack_type,
        'architecture': model_arch,
        'dataset': 'agnews',
        'model_source': model_source,
        'clean_accuracy': float(clean_acc),
        'adversarial_accuracy': float(adv_acc),
        'attack_success_rate': float(1 - adv_acc),
        'num_samples': int(eval_count),
        'epsilon': epsilon,
        'message': (
            f"AG News experimental simulation mode: Clean accuracy {clean_acc:.1%} → "
            f"Perturbed accuracy {adv_acc:.1%}."
        )
    })


@app.route('/api/run-defense', methods=['POST'])
def run_defense():
    """Compare standard model vs robust (adversarially trained) model"""
    try:
        data = request.json or {}
        dataset = (data.get('dataset', 'cifar10') or 'cifar10').lower().strip()
        attack_type = data.get('attack', 'fgsm').lower().strip()
        arch = data.get('architecture', 'ResNet-18').lower().strip()
        arch = arch.replace(' ', '').replace('-', '')
        epsilon = float(data.get('epsilon', 0.03))
        num_samples = int(data.get('num_samples', 128))

        if dataset == 'agnews':
            text_arch_map = {
                'textcnn': 'textcnn',
                'lstm': 'lstm',
                'bilstm': 'bilstm',
                'transformer': 'transformer',
            }
            model_arch = text_arch_map.get(arch, 'textcnn')

            if attack_type not in SUPPORTED_TEXT_ATTACKS:
                return jsonify({
                    'error': (
                        f"Text attack '{attack_type}' is not implemented in defense mode. "
                        f"Supported text attacks: {', '.join(sorted(SUPPORTED_TEXT_ATTACKS))}."
                    )
                }), 400

            print(f"\n[DEFENSE-TEXT] Attack={attack_type} | Arch={model_arch} | Eps={epsilon}")

            model_standard = get_model(model_arch)
            standard_source = get_model_source(model_arch)

            model_robust = build_model(model_arch)
            try:
                robust_state = torch.load(f'models/{model_arch}_agnews_retrained.pt', map_location=CONFIG['device'])
                model_robust.load_state_dict(robust_state)
                print(f"[DEFENSE-TEXT] Loaded retrained {model_arch}")
            except Exception:
                try:
                    robust_state = torch.load(f'models/{model_arch}_robust_agnews.pt', map_location=CONFIG['device'])
                    model_robust.load_state_dict(robust_state)
                    print(f"[DEFENSE-TEXT] Loaded robust {model_arch}")
                except Exception:
                    print(f"[DEFENSE-TEXT] No robust model, using standard")
                    model_robust = model_standard

            model_robust = model_robust.to(CONFIG['device']).eval()

            texts, labels, vocab_size = get_agnews_batch(num_samples)
            model_vocab_size = get_text_model_vocab_size(model_standard, default=20000)
            effective_vocab_size = min(int(vocab_size), int(model_vocab_size))
            texts = sanitize_text_tokens(texts, effective_vocab_size)

            eval_count = max(1, len(labels))
            with torch.no_grad():
                standard_clean = (model_standard(texts).argmax(dim=1) == labels).sum().item() / eval_count

            adv_texts = generate_text_adversarial_batch(
                model=model_standard,
                attack_type=attack_type,
                texts=texts,
                labels=labels,
                epsilon=epsilon,
                vocab_size=effective_vocab_size,
            )
            adv_texts = sanitize_text_tokens(adv_texts, effective_vocab_size)

            with torch.no_grad():
                standard_adv = (model_standard(adv_texts).argmax(dim=1) == labels).sum().item() / eval_count
                robust_adv = (model_robust(adv_texts).argmax(dim=1) == labels).sum().item() / eval_count

            print(f"[DEFENSE-TEXT] Standard: {standard_adv:.2%} | Robust: {robust_adv:.2%}")

            return jsonify({
                'dataset': 'agnews',
                'analysis_mode': 'defense',
                'attack_type': attack_type,
                'architecture': model_arch,
                'model_source': standard_source,
                'num_samples': int(eval_count),
                'clean_accuracy': float(standard_clean),
                'standard_clean_accuracy': float(standard_clean),
                'standard_adversarial_accuracy': float(standard_adv),
                'adversarial_accuracy': float(robust_adv),
                'robust_adversarial_accuracy': float(robust_adv),
                'defense_gap': float(robust_adv - standard_adv),
                'improvement_text': (
                    f"Robust text model maintains {robust_adv*100:.1f}% accuracy under {attack_type.upper()} "
                    f"vs standard model's {standard_adv*100:.1f}% "
                    f"(improvement: {(robust_adv-standard_adv)*100:.1f}pp)"
                )
            })

        if dataset != 'cifar10':
            return jsonify({'error': f"Unsupported dataset '{dataset}' for defense."}), 400
        
        arch_map = {
            'resnet18': 'resnet18',
            'resnet50': 'resnet50',
            'vgg16': 'vgg16',
            'densenet121': 'densenet121',
            'inceptionv3': 'inceptionv3',
            'shufflenetv2': 'shufflenetv2',
            'vit': 'vit',
        }
        model_arch = arch_map.get(arch, 'resnet18')
        
        print(f"\n[DEFENSE] Attack={attack_type} | Arch={model_arch} | Eps={epsilon}")
        
        # Load standard model
        try:
            model_standard = get_model(model_arch, require_checkpoint=True)
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 400
        standard_source = get_model_source(model_arch)
        
        # Load robust model (if exists, else use standard)
        model_robust = build_model(model_arch)
        robust_loaded = False
        try:
            # Try retrained model first
            robust_state = torch.load(f'models/{model_arch}_cifar10_retrained.pt', map_location=CONFIG['device'])
            model_robust.load_state_dict(robust_state)
            print(f"[DEFENSE] Loaded retrained {model_arch}")
            robust_loaded = True
        except:
            try:
                # Fall back to standard robust model
                robust_state = torch.load(f'models/{model_arch}_robust_cifar10.pt', map_location=CONFIG['device'])
                model_robust.load_state_dict(robust_state)
                print(f"[DEFENSE] Loaded robust {model_arch}")
                robust_loaded = True
            except:
                print(f"[DEFENSE] No robust model, using standard")
                model_robust = model_standard
        
        model_robust = model_robust.to(CONFIG['device']).eval()
        
        # Get test batch
        imgs, labels = get_cifar10_batch(num_samples)
        eval_count = max(1, len(labels))
        
        # Resize inputs when architecture requires larger dimensions
        imgs_model, _ = prepare_images_for_arch(imgs, model_arch)

        # Test standard model
        with torch.no_grad():
            standard_clean = (model_standard(imgs_model).argmax(dim=1) == labels).sum().item() / eval_count
        
        if attack_type not in SUPPORTED_IMAGE_ATTACKS:
            return jsonify({
                'error': (
                    f"Attack '{attack_type}' is not implemented in defense mode. "
                    f"Supported attacks: {', '.join(sorted(SUPPORTED_IMAGE_ATTACKS))}."
                )
            }), 400

        try:
            adv = generate_adversarial_batch(model_standard, attack_type, imgs_model, labels, epsilon)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Test both models against attack
        with torch.no_grad():
            standard_adv = (model_standard(adv).argmax(dim=1) == labels).sum().item() / eval_count
            robust_adv = (model_robust(adv).argmax(dim=1) == labels).sum().item() / eval_count
        
        print(f"[DEFENSE] Standard: {standard_adv:.2%} | Robust: {robust_adv:.2%}")
        
        # Return in format expected by ResultsPanel
        return jsonify({
            'dataset': 'cifar10',
            'analysis_mode': 'defense',
            'attack_type': attack_type,
            'architecture': model_arch,
            'model_source': standard_source,
            'num_samples': int(eval_count),
            'clean_accuracy': float(standard_clean),
            'standard_clean_accuracy': float(standard_clean),
            'standard_adversarial_accuracy': float(standard_adv),
            'adversarial_accuracy': float(robust_adv),
            'robust_adversarial_accuracy': float(robust_adv),
            'defense_gap': float(robust_adv - standard_adv),
            'improvement_text': f"Robust model maintains {robust_adv*100:.1f}% accuracy under attack vs standard model's {standard_adv*100:.1f}% (improvement: {(robust_adv-standard_adv)*100:.1f}pp)"
        })
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/transfer-analysis', methods=['POST'])
def transfer_analysis():
    """Test if attack on one model transfers to all others"""
    try:
        data = request.json or {}
        dataset = (data.get('dataset', 'cifar10') or 'cifar10').lower().strip()
        source_arch = data.get('source_architecture', 'ResNet-18').lower().strip()
        source_arch = source_arch.replace(' ', '').replace('-', '')
        attack_type = data.get('attack', 'fgsm').lower().strip()
        epsilon = float(data.get('epsilon', 0.03))
        num_samples = int(data.get('num_samples', 128))

        if dataset == 'agnews':
            if attack_type not in SUPPORTED_TEXT_ATTACKS:
                return jsonify({
                    'error': (
                        f"Text attack '{attack_type}' is not implemented in transfer mode. "
                        f"Supported text attacks: {', '.join(sorted(SUPPORTED_TEXT_ATTACKS))}."
                    )
                }), 400

            text_arch_map = {
                'textcnn': 'textcnn',
                'lstm': 'lstm',
                'bilstm': 'bilstm',
                'transformer': 'transformer',
            }
            source_model_arch = text_arch_map.get(source_arch, 'textcnn')

            print(f"\n[TRANSFER-TEXT] Source={source_model_arch} | Attack={attack_type} | Eps={epsilon}")

            models_dict = {arch_name: get_model(arch_name) for arch_name in text_arch_map.values()}
            model_sources = {arch_name: get_model_source(arch_name) for arch_name in text_arch_map.values()}

            texts, labels, vocab_size = get_agnews_batch(num_samples)
            source_vocab_size = get_text_model_vocab_size(models_dict[source_model_arch], default=20000)
            effective_vocab_size = min(int(vocab_size), int(source_vocab_size))
            texts = sanitize_text_tokens(texts, effective_vocab_size)

            source_model = models_dict[source_model_arch]
            adv_texts = generate_text_adversarial_batch(
                model=source_model,
                attack_type=attack_type,
                texts=texts,
                labels=labels,
                epsilon=epsilon,
                vocab_size=effective_vocab_size,
            )
            adv_texts = sanitize_text_tokens(adv_texts, effective_vocab_size)

            eval_count = max(1, len(labels))
            with torch.no_grad():
                clean_acc = (source_model(texts).argmax(dim=1) == labels).sum().item() / eval_count

            results = {}
            transfer_rates = []
            for arch_name, model in models_dict.items():
                target_vocab_size = get_text_model_vocab_size(model, default=effective_vocab_size)
                adv_for_model = sanitize_text_tokens(adv_texts, target_vocab_size)
                with torch.no_grad():
                    success_rate = 1 - (model(adv_for_model).argmax(dim=1) == labels).sum().item() / eval_count
                results[arch_name] = float(success_rate)
                transfer_rates.append(success_rate)
                print(f"[TRANSFER-TEXT] {arch_name}: {success_rate:.2%}")

            avg_transfer = float(np.mean(transfer_rates)) if transfer_rates else 0.0

            return jsonify({
                'dataset': 'agnews',
                'analysis_mode': 'transfer',
                'attack_type': attack_type,
                'source_architecture': source_model_arch,
                'model_source': model_sources.get(source_model_arch, 'unknown'),
                'model_sources': model_sources,
                'num_samples': int(eval_count),
                'clean_accuracy': float(clean_acc),
                'adversarial_accuracy': avg_transfer,
                'insight': (
                    f"{attack_type.upper()} attack on {source_model_arch} transfers across {len(results)} AG News architectures "
                    f"with average success rate of {avg_transfer*100:.1f}%. Individual results: "
                    f"{', '.join([f'{arch}: {results[arch]*100:.1f}%' for arch in sorted(results.keys())])}"
                ),
                'transfer_results': results,
            })

        if dataset != 'cifar10':
            return jsonify({'error': f"Unsupported dataset '{dataset}' for transfer analysis."}), 400
        
        arch_map = {
            'resnet18': 'resnet18', 'resnet50': 'resnet50', 'vgg16': 'vgg16',
            'densenet121': 'densenet121'
        }
        source_model_arch = arch_map.get(source_arch, 'resnet18')

        if attack_type not in SUPPORTED_IMAGE_ATTACKS:
            return jsonify({
                'error': (
                    f"Attack '{attack_type}' is not implemented in transfer mode. "
                    f"Supported attacks: {', '.join(sorted(SUPPORTED_IMAGE_ATTACKS))}."
                )
            }), 400
        
        print(f"\n[TRANSFER] Source={source_model_arch} | Eps={epsilon}")
        
        # Load all models
        models_dict = {}
        model_sources = {}
        missing_architectures = []
        for arch in arch_map.values():
            try:
                models_dict[arch] = get_model(arch, require_checkpoint=True)
                model_sources[arch] = get_model_source(arch)
            except FileNotFoundError as e:
                if arch == source_model_arch:
                    return jsonify({'error': str(e)}), 400
                print(f"[TRANSFER] Skipping {arch}: {e}")
                missing_architectures.append(arch)
        
        # Get test batch
        imgs, labels = get_cifar10_batch(num_samples)
        eval_count = max(1, len(labels))
        
        # Generate attack on source model
        if source_model_arch not in models_dict:
            return jsonify({
                'error': f"Required checkpoint missing for source architecture '{source_model_arch}'."
            }), 400
        source_model = models_dict[source_model_arch]
        adv = generate_adversarial_batch(source_model, attack_type, imgs, labels, epsilon)
        
        # Test clean accuracy on source model
        with torch.no_grad():
            clean_acc = (source_model(imgs).argmax(dim=1) == labels).sum().item() / eval_count
        
        # Test transfer on all models
        results = {}
        transfer_rates = []
        for arch, model in models_dict.items():
            with torch.no_grad():
                success_rate = 1 - (model(adv).argmax(dim=1) == labels).sum().item() / eval_count
            results[arch] = float(success_rate)
            transfer_rates.append(success_rate)
            print(f"[TRANSFER] {arch}: {success_rate:.2%}")
        
        avg_transfer = float(np.mean(transfer_rates))
        
        return jsonify({
            'dataset': 'cifar10',
            'analysis_mode': 'transfer',
            'attack_type': attack_type,
            'source_architecture': source_model_arch,
            'model_source': model_sources.get(source_model_arch, 'unknown'),
            'model_sources': model_sources,
            'missing_architectures': missing_architectures,
            'num_samples': int(eval_count),
            'clean_accuracy': float(clean_acc),
            'adversarial_accuracy': avg_transfer,
            'insight': (
                f"{attack_type.upper()} attack on {source_model_arch} transfers across {len(results)} available architectures "
                f"with average success rate of {avg_transfer*100:.1f}%. Individual results: "
                f"{', '.join([f'{arch}: {results[arch]*100:.1f}%' for arch in sorted(results.keys())])}"
                + (f". Skipped missing checkpoints: {', '.join(missing_architectures)}." if missing_architectures else '')
            ),
            'transfer_results': results
        })
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    conn = sqlite3.connect('cerberus.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM experiments ORDER BY timestamp DESC LIMIT 50')
    exps = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({'experiments': exps, 'count': len(exps)})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path:
        target = os.path.join(app.static_folder, path)
        if os.path.exists(target):
            return send_file(target)
        # Avoid noisy 500s for browser favicon probes during API-only runs
        if path == 'favicon.ico':
            return ('', 204)

    index_path = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_file(index_path)

    return jsonify({
        'status': 'ok',
        'message': 'Frontend build not found. Run frontend with npm start or build with npm run build.',
        'frontend_build_expected_at': index_path
    }), 200

@app.route('/api/defense-analysis', methods=['POST'])
def defense_analysis():
    """Detailed defense analysis with multiple epsilon values - robustness curve"""
    try:
        data = request.json or {}
        attack_type = data.get('attack', 'pgd').lower().strip()
        arch = data.get('architecture', 'ResNet-18').lower().strip()
        arch = arch.replace(' ', '').replace('-', '')
        
        arch_map = {
            'resnet18': 'resnet18', 'resnet50': 'resnet50', 'vgg16': 'vgg16',
            'densenet121': 'densenet121'
        }
        model_arch = arch_map.get(arch, 'resnet18')
        
        print(f"\n[DEFENSE ANALYSIS] Attack={attack_type} | Arch={model_arch}")

        if attack_type not in {'fgsm', 'pgd'}:
            return jsonify({
                'error': "Defense analysis currently supports only fgsm and pgd for consistent robustness curves."
            }), 400
        
        # Load models
        model_standard = get_model(model_arch)
        model_robust = build_model(model_arch)
        try:
            robust_state = torch.load(f'models/{model_arch}_robust_cifar10.pt', map_location=CONFIG['device'])
            model_robust.load_state_dict(robust_state)
        except:
            model_robust = model_standard
        
        model_robust = model_robust.to(CONFIG['device']).eval()
        
        # Get test batch
        loader = get_data_loader()
        imgs, labels = next(iter(loader))
        imgs = imgs[:16].to(CONFIG['device'])
        labels = labels[:16].to(CONFIG['device'])
        
        # Clean accuracy
        with torch.no_grad():
            standard_clean = (model_standard(imgs).argmax(dim=1) == labels).sum().item() / len(labels)
        
        # Test across multiple epsilon values
        results = {
            'epsilons': [],
            'standard_accuracy': [],
            'robust_accuracy': [],
            'improvement': []
        }
        
        epsilons = [0.01, 0.02, 0.03, 0.06, 0.1]
        for eps in epsilons:
            # Generate attack at this epsilon
            if attack_type == 'pgd':
                attack = PGDAttack(model=model_standard, eps=eps, eps_step=eps/10, max_iter=100, device=CONFIG['device'])
                adv = attack.generate_adversarial(imgs, labels)
            else:
                attack = FSGMAttack(model=model_standard, eps=eps, device=CONFIG['device'])
                adv = attack.generate(imgs, labels)
            
            # Test accuracy
            with torch.no_grad():
                std_acc = (model_standard(adv).argmax(dim=1) == labels).sum().item() / len(labels)
                rob_acc = (model_robust(adv).argmax(dim=1) == labels).sum().item() / len(labels)
            
            results['epsilons'].append(eps)
            results['standard_accuracy'].append(std_acc)
            results['robust_accuracy'].append(rob_acc)
            results['improvement'].append(rob_acc - std_acc)
            
            print(f"  ε={eps:.3f}: Standard={std_acc*100:.1f}% → Robust={rob_acc*100:.1f}% (Δ={((rob_acc-std_acc)*100):.1f}pp)")
        
        results['clean_accuracy'] = standard_clean
        return jsonify(results)
    
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print(" CERBERUS - ADVERSARIAL ATTACK PLATFORM")
    print("="*80)
    print(f"Device: {CONFIG['device']}")
    print(f"Server: http://localhost:5000")
    print("="*80 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
