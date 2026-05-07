"""
Cerberus Interactive Dashboard - Flask Web UI
For panel demonstrations of adversarial attacks and defenses
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64

# Import Cerberus modules
from cerberus.attacks import FSGMAttack, PGDAttack, CWAttack, DeepFoolAttack, JSMAAttack
from cerberus.models import load_pretrained_model
from cerberus.data import load_cifar10
import torchvision.transforms as transforms
from torchvision.utils import make_grid

app = Flask(__name__)
CORS(app)

# Global state
MODEL_CACHE = {}
DATA_CACHE = {}
RESULTS_CACHE = {}

# Configuration
CONFIG = {
    'epsilon': 8/255,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_samples': 10,
    'batch_size': 10
}

def load_model(arch_name, model_type='baseline'):
    """Load model with caching"""
    cache_key = f"{arch_name}_{model_type}"
    if cache_key not in MODEL_CACHE:
        model = load_pretrained_model(arch_name, model_type=model_type)
        model = model.to(CONFIG['device'])
        model.eval()
        MODEL_CACHE[cache_key] = model
    return MODEL_CACHE[cache_key]

def get_cifar10_data():
    """Load CIFAR-10 dataset with caching"""
    if 'cifar10' not in DATA_CACHE:
        _, test_loader = load_cifar10(batch_size=CONFIG['batch_size'])
        DATA_CACHE['cifar10'] = test_loader
    return DATA_CACHE['cifar10']

def get_attack_instance(attack_type, epsilon):
    """Get attack instance"""
    attacks = {
        'fgsm': FSGMAttack(epsilon=epsilon),
        'pgd': PGDAttack(epsilon=epsilon, alpha=epsilon/4, steps=7),
        'cw': CWAttack(c=0.1, steps=100, lr=0.01),
        'deepfool': DeepFoolAttack(max_iter=50),
        'jsma': JSMAAttack(theta=0.1, gamma=0.15)
    }
    return attacks.get(attack_type.lower())

def images_to_base64(images):
    """Convert tensor images to base64 for JSON"""
    # Denormalize if needed
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    images = (images * 255).astype(np.uint8)
    
    base64_images = []
    for img in images:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img)
        ax.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        base64_images.append(f"data:image/png;base64,{img_str}")
        plt.close(fig)
    
    return base64_images

def create_comparison_grid(clean_imgs, adv_imgs, labels, predictions_clean, predictions_adv):
    """Create side-by-side comparison visualization"""
    fig, axes = plt.subplots(2, min(5, len(clean_imgs)), figsize=(15, 6))
    
    if len(clean_imgs) == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(5, len(clean_imgs))):
        # Clean image
        clean_img = clean_imgs[i].cpu().numpy()
        clean_img = np.transpose(clean_img, (1, 2, 0))
        clean_img = (clean_img * 255).astype(np.uint8)
        
        axes[0, i].imshow(clean_img)
        axes[0, i].set_title(f"Clean\nPred: {predictions_clean[i]}", fontsize=9)
        axes[0, i].axis('off')
        
        # Adversarial image
        adv_img = adv_imgs[i].cpu().numpy()
        adv_img = np.transpose(adv_img, (1, 2, 0))
        adv_img = (adv_img * 255).astype(np.uint8)
        
        axes[1, i].imshow(adv_img)
        axes[1, i].set_title(f"Attacked\nPred: {predictions_adv[i]}", fontsize=9)
        axes[1, i].axis('off')
    
    plt.suptitle('Clean vs Adversarial Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return f"data:image/png;base64,{img_str}"

def create_metrics_chart(metrics):
    """Create metrics visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy comparison
    categories = ['Clean Model', 'Robust Model']
    clean_acc = [metrics['clean_accuracy_baseline'], metrics['clean_accuracy_robust']]
    adv_acc = [metrics['adv_accuracy_baseline'], metrics['adv_accuracy_robust']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, clean_acc, width, label='Clean Accuracy', color='#2ecc71')
    axes[0].bar(x + width/2, adv_acc, width, label='Adversarial Accuracy', color='#e74c3c')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Robustness Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Attack success rate
    attack_types = ['FGSM', 'PGD', 'C&W', 'DeepFool', 'JSMA']
    success_rates = [
        metrics.get('fgsm_success', 0),
        metrics.get('pgd_success', 0),
        metrics.get('cw_success', 0),
        metrics.get('deepfool_success', 0),
        metrics.get('jsma_success', 0)
    ]
    
    axes[1].bar(attack_types, success_rates, color=['#e74c3c', '#e67e22', '#f39c12', '#95a5a6', '#34495e'])
    axes[1].set_ylabel('Attack Success Rate (%)')
    axes[1].set_title('Attack Effectiveness Comparison')
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def status():
    """Get system status"""
    return jsonify({
        'status': 'ready',
        'device': CONFIG['device'],
        'cuda_available': torch.cuda.is_available(),
        'torch_version': torch.__version__,
        'available_attacks': ['FGSM', 'PGD', 'C&W', 'DeepFool', 'JSMA'],
        'available_architectures': ['ResNet-18', 'VGG-16', 'MobileNet V2', 'EfficientNet-B0', 'DenseNet-121']
    })

@app.route('/api/run-attack', methods=['POST'])
def run_attack():
    """Execute adversarial attack"""
    try:
        data = request.json
        attack_type = data.get('attack', 'fgsm').lower()
        architecture = data.get('architecture', 'ResNet-18')
        model_type = data.get('model_type', 'baseline')
        epsilon = float(data.get('epsilon', 8/255))
        num_samples = int(data.get('num_samples', 10))
        
        # Load model and data
        model = load_model(architecture, model_type)
        test_loader = get_cifar10_data()
        
        # Get batch
        images, labels = next(iter(test_loader))
        images = images[:num_samples].to(CONFIG['device'])
        labels = labels[:num_samples].to(CONFIG['device'])
        
        # Get clean predictions
        with torch.no_grad():
            clean_logits = model(images)
            clean_preds = torch.argmax(clean_logits, dim=1).cpu().numpy()
            clean_acc = (clean_preds == labels.cpu().numpy()).mean() * 100
        
        # Generate adversarial examples
        attack = get_attack_instance(attack_type, epsilon)
        if attack is None:
            return jsonify({'error': f'Unknown attack: {attack_type}'}), 400
        
        adv_images = attack.generate(images, labels, model)
        
        # Get adversarial predictions
        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
            adv_acc = (adv_preds == labels.cpu().numpy()).mean() * 100
            attack_success = ((adv_preds != labels.cpu().numpy()).sum() / len(labels)) * 100
        
        # Create visualizations
        comparison_img = create_comparison_grid(images, adv_images, labels, clean_preds, adv_preds)
        
        # Prepare response
        result = {
            'status': 'success',
            'attack': attack_type.upper(),
            'architecture': architecture,
            'model_type': model_type,
            'epsilon': epsilon,
            'metrics': {
                'clean_accuracy': float(clean_acc),
                'adversarial_accuracy': float(adv_acc),
                'attack_success_rate': float(attack_success),
                'accuracy_drop': float(clean_acc - adv_acc)
            },
            'visualization': comparison_img,
            'samples': {
                'clean_predictions': clean_preds.tolist(),
                'adv_predictions': adv_preds.tolist(),
                'true_labels': labels.cpu().numpy().tolist()
            }
        }
        
        RESULTS_CACHE[f"{attack_type}_{architecture}"] = result
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run-defense', methods=['POST'])
def run_defense():
    """Compare baseline vs robust model"""
    try:
        data = request.json
        attack_type = data.get('attack', 'fgsm').lower()
        architecture = data.get('architecture', 'ResNet-18')
        epsilon = float(data.get('epsilon', 8/255))
        num_samples = int(data.get('num_samples', 10))
        
        # Load both models
        baseline_model = load_model(architecture, 'baseline')
        robust_model = load_model(architecture, 'robust')
        test_loader = get_cifar10_data()
        
        # Get batch
        images, labels = next(iter(test_loader))
        images = images[:num_samples].to(CONFIG['device'])
        labels = labels[:num_samples].to(CONFIG['device'])
        
        # Generate adversarial examples (using baseline model)
        attack = get_attack_instance(attack_type, epsilon)
        adv_images = attack.generate(images, labels, baseline_model)
        
        # Evaluate on both models
        with torch.no_grad():
            # Baseline
            baseline_clean = (torch.argmax(baseline_model(images), dim=1) == labels).float().mean() * 100
            baseline_adv = (torch.argmax(baseline_model(adv_images), dim=1) == labels).float().mean() * 100
            
            # Robust
            robust_clean = (torch.argmax(robust_model(images), dim=1) == labels).float().mean() * 100
            robust_adv = (torch.argmax(robust_model(adv_images), dim=1) == labels).float().mean() * 100
        
        # Create visualization
        metrics = {
            'clean_accuracy_baseline': float(baseline_clean),
            'adv_accuracy_baseline': float(baseline_adv),
            'clean_accuracy_robust': float(robust_clean),
            'adv_accuracy_robust': float(robust_adv),
            'robustness_improvement': float(robust_adv - baseline_adv),
            'fgsm_success': 0,
            'pgd_success': 0,
            'cw_success': 0,
            'deepfool_success': 0,
            'jsma_success': 0
        }
        
        metrics_img = create_metrics_chart(metrics)
        
        result = {
            'status': 'success',
            'attack': attack_type.upper(),
            'architecture': architecture,
            'epsilon': epsilon,
            'metrics': metrics,
            'visualization': metrics_img,
            'improvement_text': f"Robustness improved by {metrics['robustness_improvement']:.2f}% with adversarial training!"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transfer-analysis', methods=['POST'])
def transfer_analysis():
    """Run transfer matrix analysis"""
    try:
        data = request.json
        source_arch = data.get('source_architecture', 'ResNet-18')
        epsilon = float(data.get('epsilon', 8/255))
        
        # Load architectures
        architectures = ['ResNet-18', 'VGG-16', 'MobileNet V2', 'EfficientNet-B0', 'DenseNet-121']
        attacks_types = ['fgsm', 'pgd', 'cw', 'deepfool', 'jsma']
        
        source_model = load_model(source_arch, 'baseline')
        test_loader = get_cifar10_data()
        
        images, labels = next(iter(test_loader))
        images = images[:20].to(CONFIG['device'])
        labels = labels[:20].to(CONFIG['device'])
        
        # Generate adversarial examples from source
        attack = get_attack_instance('pgd', epsilon)
        adv_images = attack.generate(images, labels, source_model)
        
        # Test on all target architectures
        results = {}
        for target_arch in architectures:
            target_model = load_model(target_arch, 'baseline')
            with torch.no_grad():
                predictions = torch.argmax(target_model(adv_images), dim=1)
                success_rate = ((predictions != labels).sum().float() / len(labels)).item() * 100
                results[target_arch] = float(success_rate)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        transfer_rates = list(results.values())
        colors = ['#2ecc71' if x < 50 else '#e74c3c' for x in transfer_rates]
        
        ax.barh(list(results.keys()), transfer_rates, color=colors)
        ax.set_xlabel('Attack Transfer Success Rate (%)')
        ax.set_title(f'Cross-Architecture Transfer - Attacks from {source_arch}')
        ax.set_xlim([0, 100])
        
        for i, v in enumerate(transfer_rates):
            ax.text(v + 2, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return jsonify({
            'status': 'success',
            'source_architecture': source_arch,
            'transfer_rates': results,
            'visualization': f"data:image/png;base64,{img_str}",
            'insight': f"Transfer success from {source_arch} ranges from {min(transfer_rates):.1f}% to {max(transfer_rates):.1f}%"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Cerberus Interactive Dashboard starting...")
    print(f"📊 Device: {CONFIG['device']}")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, host='0.0.0.0')
