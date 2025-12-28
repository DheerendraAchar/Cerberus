#!/usr/bin/env python3
"""
Quick script to generate the 2 new figures using existing data.
Run directly without Docker to use cached CIFAR-10 data.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_figures import (
    per_class_robustness,
    attack_success_rate_by_epsilon,
    build_model,
    ensure_figures_dir
)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not available. This script requires torch.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)

def main():
    print("Generating 2 new figures...")
    print("Using cached CIFAR-10 data from ./data/")
    
    device = torch.device("cpu")
    model = build_model(device)
    out_dir = ensure_figures_dir("figures")
    
    eps = 0.03
    eps_list = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    
    try:
        # Figure 1: Per-class robustness
        print("\n[1/2] Generating per-class robustness bar chart...")
        path1 = os.path.join(out_dir, f"per_class_robustness_eps{eps}.png")
        per_class_robustness(model, device, eps, path1)
        print(f"✅ Saved: {path1}")
        
        # Figure 2: Attack success rate vs epsilon
        print("\n[2/2] Generating attack success rate plot...")
        path2 = os.path.join(out_dir, "attack_success_vs_epsilon.png")
        attack_success_rate_by_epsilon(model, device, eps_list, path2)
        print(f"✅ Saved: {path2}")
        
        print("\n✨ Done! 2 new figures generated successfully.")
        print(f"\nNew figures:")
        print(f"  - {path1}")
        print(f"  - {path2}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
