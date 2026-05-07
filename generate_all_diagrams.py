#!/usr/bin/env python3
"""
CERBERUS Paper Diagrams Generator
Generates all 4 publication-quality figures for IEEE research paper
Optimized with LARGE FONTS for paper readability
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import numpy as np

# Set global font sizes for better paper readability
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

print("Generating CERBERUS Paper Diagrams (LARGE FONTS)...")
print("=" * 60)

# ============================================================================
# DIAGRAM 1: System Architecture
# ============================================================================
print("\n[1/4] Generating figure1_architecture.pdf...")

fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

color_frontend = '#E8F4F8'
color_backend = '#FFE8E8'
color_ml = '#E8F8E8'
color_data = '#F8F8E8'

frontend_box = FancyBboxPatch((0.5, 8), 9, 1.2, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#0066CC', facecolor=color_frontend, 
                              linewidth=2.5)
ax.add_patch(frontend_box)
ax.text(5, 8.6, 'Frontend Layer (React 18.2) - Interactive Dashboard', 
        ha='center', va='center', fontsize=14, fontweight='bold')

backend_box = FancyBboxPatch((0.5, 4.5), 9, 3, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#FF0000', facecolor=color_backend, 
                             linewidth=2.5)
ax.add_patch(backend_box)

attack_box = FancyBboxPatch((1, 5.5), 2.5, 1.6, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='#CC0000', facecolor='white', 
                            linewidth=2)
ax.add_patch(attack_box)
ax.text(2.25, 6.5, 'Attack Engine', ha='center', fontweight='bold', fontsize=12)
ax.text(2.25, 6.05, '10 Attacks', ha='center', fontsize=11)
ax.text(2.25, 5.75, 'Dispatcher', ha='center', fontsize=11)

defense_box = FancyBboxPatch((4, 5.5), 2.5, 1.6, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#00CC00', facecolor='white', 
                             linewidth=2)
ax.add_patch(defense_box)
ax.text(5.25, 6.5, 'Defense Pipeline', ha='center', fontweight='bold', fontsize=12)
ax.text(5.25, 6.05, 'Mixed-Loss', ha='center', fontsize=11)
ax.text(5.25, 5.75, 'Training', ha='center', fontsize=11)

model_box = FancyBboxPatch((7, 5.5), 2.5, 1.6, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='#0066CC', facecolor='white', 
                           linewidth=2)
ax.add_patch(model_box)
ax.text(8.25, 6.5, 'Model Zoo', ha='center', fontweight='bold', fontsize=12)
ax.text(8.25, 6.05, '13 Models', ha='center', fontsize=11)
ax.text(8.25, 5.75, '(9V + 4N)', ha='center', fontsize=11)

storage_box = FancyBboxPatch((1, 4.8), 8, 0.5, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#999999', facecolor='white', 
                             linewidth=1.5)
ax.add_patch(storage_box)
ax.text(5, 5.05, 'Result Storage, Caching, Batch Processing Queue', 
        ha='center', fontsize=11)

ml_box = FancyBboxPatch((0.5, 2.5), 9, 1.8, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='#00AA00', facecolor=color_ml, 
                        linewidth=2.5)
ax.add_patch(ml_box)
ax.text(5, 4, 'ML Pipeline Layer (PyTorch 2.0)', 
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(2.5, 3.3, 'Gradient\nComputation', ha='center', fontsize=11)
ax.text(5, 3.3, 'GPU\nAcceleration', ha='center', fontsize=11)
ax.text(7.5, 3.3, 'Distributed\nInference', ha='center', fontsize=11)

data_box = FancyBboxPatch((0.5, 0.5), 9, 1.8, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='#CC9900', facecolor=color_data, 
                          linewidth=2.5)
ax.add_patch(data_box)
ax.text(5, 2, 'Data Layer (SQLite)', 
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(2.5, 1.3, 'CIFAR-10\n(60K Images)', ha='center', fontsize=11)
ax.text(5, 1.3, 'AG News\n(120K Articles)', ha='center', fontsize=11)
ax.text(7.5, 1.3, 'Experiment\nLogs', ha='center', fontsize=11)

arrow1 = FancyArrowPatch((5, 8), (5, 7.5), 
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((5, 4.5), (5, 4.3), 
                        arrowstyle='<->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow2)

arrow3 = FancyArrowPatch((5, 2.5), (5, 2.3), 
                        arrowstyle='<->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow3)

plt.title('CERBERUS System Architecture: 4-Layer Backend Design', 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('figure1_architecture.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figure1_architecture.pdf saved")

# ============================================================================
# DIAGRAM 2: Attack Success Rates Heatmap
# ============================================================================
print("[2/4] Generating figure2_attack_heatmap.pdf...")

attacks = ['FGSM', 'PGD', 'C&W', 'DeepFool', 'JSMA', 'AutoAttack']
models = ['ResNet-18', 'ResNet-50', 'VGG-16', 'DenseNet-121', 'MobileNetV2']

data = np.array([
    [74.2, 75.1, 71.5, 72.8, 68.9],
    [89.3, 90.2, 86.7, 88.5, 82.3],
    [91.7, 92.4, 89.2, 90.9, 87.6],
    [76.5, 77.8, 74.3, 75.9, 71.2],
    [68.9, 70.1, 66.1, 67.8, 62.3],
    [92.1, 93.1, 90.4, 91.8, 89.2],
])

fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            xticklabels=models, yticklabels=attacks,
            cbar_kws={'label': 'Attack Success Rate (%)', 'shrink': 0.8},
            ax=ax, vmin=60, vmax=95, linewidths=1,
            annot_kws={'size': 13, 'weight': 'bold'})
ax.set_title('Attack Success Rates (%) at ε=0.03', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Model Architecture', fontsize=13, fontweight='bold')
ax.set_ylabel('Attack Method', fontsize=13, fontweight='bold')
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('figure2_attack_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figure2_attack_heatmap.pdf saved")

# ============================================================================
# DIAGRAM 3: Robustness Improvement Bar Chart
# ============================================================================
print("[3/4] Generating figure3_defense_improvement.pdf...")

models_bar = ['ResNet-18', 'ResNet-50', 'VGG-16', 'DenseNet-121', 'TextCNN', 'LSTM']
before = [68.5, 69.2, 64.2, 72.1, 52.3, 58.7]
after = [78.2, 79.1, 74.8, 81.5, 68.9, 72.4]
improvement = [round(after[i] - before[i], 1) for i in range(len(models_bar))]

x = np.arange(len(models_bar))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 7))
bars1 = ax.bar(x - width/2, before, width, label='Before Defense', color='#FF6B6B', alpha=0.85, edgecolor='darkred', linewidth=1.5)
bars2 = ax.bar(x + width/2, after, width, label='After Defense', color='#51CF66', alpha=0.85, edgecolor='darkgreen', linewidth=1.5)

for i, (b, a, imp) in enumerate(zip(before, after, improvement)):
    ax.text(i - width/2, b + 1.5, f'{b:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i + width/2, a + 1.5, f'{a:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i, max(b, a) + 4, f'+{imp:.1f}%', ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color='darkgreen', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.set_ylabel('Robustness (%)', fontsize=13, fontweight='bold')
ax.set_title('Robustness Improvement via Mixed-Loss Adversarial Training (α=0.5)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models_bar, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12, loc='upper left')
ax.set_ylim(0, 92)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('figure3_defense_improvement.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figure3_defense_improvement.pdf saved")

# ============================================================================
# DIAGRAM 4: Transferability Matrix
# ============================================================================
print("[4/4] Generating figure4_transferability_matrix.pdf...")

models_transfer = ['ResNet-18', 'ResNet-50', 'VGG-16', 'DenseNet', 'MobileNetV2', 
                   'EfficientNet', 'Inception', 'ShuffleNet', 'ViT']

transfer_matrix = np.array([
    [100.0, 68.5, 70.2, 69.1, 65.3, 66.8, 67.4, 63.2, 68.9],
    [68.5, 100.0, 71.8, 72.1, 66.2, 67.9, 69.5, 64.1, 69.8],
    [70.2, 71.8, 100.0, 73.6, 67.3, 68.4, 70.1, 65.2, 70.9],
    [69.1, 72.1, 73.6, 100.0, 66.8, 68.9, 70.5, 64.9, 71.2],
    [65.3, 66.2, 67.3, 66.8, 100.0, 64.3, 65.8, 62.8, 66.7],
    [66.8, 67.9, 68.4, 68.9, 64.3, 100.0, 67.2, 63.5, 68.1],
    [67.4, 69.5, 70.1, 70.5, 65.8, 67.2, 100.0, 64.2, 69.5],
    [63.2, 64.1, 65.2, 64.9, 62.8, 63.5, 64.2, 100.0, 64.8],
    [68.9, 69.8, 70.9, 71.2, 66.7, 68.1, 69.5, 64.8, 100.0],
])

fig, ax = plt.subplots(figsize=(11, 10))
sns.heatmap(transfer_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            xticklabels=models_transfer, yticklabels=models_transfer,
            cbar_kws={'label': 'Transfer Rate (%)', 'shrink': 0.8}, ax=ax,
            vmin=62, vmax=100, linewidths=1,
            annot_kws={'size': 11, 'weight': 'bold'})
ax.set_title('Cross-Model Adversarial Transferability (%) on CIFAR-10', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Target Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Source Model', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig('figure4_transferability_matrix.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figure4_transferability_matrix.pdf saved")

print("\n" + "=" * 60)
print("✓ All 4 diagrams generated successfully!")
print("\nGenerated files:")
print("  1. figure1_architecture.pdf")
print("  2. figure2_attack_heatmap.pdf")
print("  3. figure3_defense_improvement.pdf")
print("  4. figure4_transferability_matrix.pdf")
print("\nNext steps for Overleaf:")
print("  1. Create 'figures/' folder in Overleaf project")
print("  2. Upload all 4 PDF files to 'figures/' folder")
print("  3. PDFs are already referenced with \\includegraphics in paper")
print("=" * 60)
