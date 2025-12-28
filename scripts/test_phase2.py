#!/usr/bin/env python3
"""Quick test script to verify Phase 2 implementation.

This script performs sanity checks on all Phase 2 components without
running full training (which takes hours).
"""

import sys
import os


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import yaml
        print("   ✅ PyYAML")
    except ImportError:
        print("   ❌ PyYAML - install with: pip install pyyaml")
        return False
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch - install with: pip install torch")
        return False
    
    try:
        import torchvision
        print(f"   ✅ torchvision {torchvision.__version__}")
    except ImportError:
        print("   ❌ torchvision - install with: pip install torchvision")
        return False
    
    try:
        import matplotlib
        print(f"   ✅ matplotlib {matplotlib.__version__}")
    except ImportError:
        print("   ❌ matplotlib - install with: pip install matplotlib")
        return False
    
    try:
        import numpy
        print(f"   ✅ numpy {numpy.__version__}")
    except ImportError:
        print("   ❌ numpy - install with: pip install numpy")
        return False
    
    return True


def test_module_structure():
    """Test that all Phase 2 files exist."""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "cerberus/baseline_training.py",
        "cerberus/adversarial_training.py",
        "configs/training_config.yaml",
        "scripts/plot_training_curves.py",
        "scripts/compare_models.py",
        "PHASE2_IMPLEMENTATION.md",
        "run_demo.py",
        "cerberus/cli.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def test_config_loading():
    """Test that config file can be loaded."""
    print("\n🔍 Testing config loading...")
    
    try:
        from cerberus.config import load_config
        cfg = load_config("configs/training_config.yaml")
        
        # Verify key sections exist
        assert "training" in cfg, "Missing 'training' section"
        assert "adversarial" in cfg, "Missing 'adversarial' section"
        assert "dataset" in cfg, "Missing 'dataset' section"
        assert "model" in cfg, "Missing 'model' section"
        
        print("   ✅ Config loads successfully")
        print(f"   ✅ Training type: {cfg.get('training_type', 'not set')}")
        print(f"   ✅ Epochs: {cfg['training']['num_epochs']}")
        print(f"   ✅ Epsilon: {cfg['adversarial']['epsilon']}")
        print(f"   ✅ Alpha: {cfg['adversarial']['alpha']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        return False


def test_training_modules():
    """Test that training modules can be imported and instantiated."""
    print("\n🔍 Testing training modules...")
    
    try:
        import torch
        import torch.nn as nn
        from cerberus.baseline_training import BaselineTrainer
        from cerberus.adversarial_training import AdversarialTrainer
        
        # Create dummy model and loaders
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        
        model = DummyModel()
        
        # Create dummy dataset
        dummy_dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 2, (100,))
        )
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=10)
        
        # Test BaselineTrainer
        baseline_trainer = BaselineTrainer(
            model=model,
            train_loader=dummy_loader,
            test_loader=dummy_loader,
            device="cpu"
        )
        print("   ✅ BaselineTrainer instantiated")
        
        # Test AdversarialTrainer
        adv_trainer = AdversarialTrainer(
            model=model,
            train_loader=dummy_loader,
            test_loader=dummy_loader,
            device="cpu",
            epsilon=0.03,
            alpha=0.5
        )
        print("   ✅ AdversarialTrainer instantiated")
        
        return True
    except Exception as e:
        print(f"   ❌ Training modules failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_integration():
    """Test that CLI functions exist."""
    print("\n🔍 Testing CLI integration...")
    
    try:
        from cerberus.cli import run_from_config, run_training
        print("   ✅ run_from_config imported")
        print("   ✅ run_training imported")
        return True
    except Exception as e:
        print(f"   ❌ CLI integration failed: {e}")
        return False


def test_visualization_scripts():
    """Test that visualization scripts can be imported."""
    print("\n🔍 Testing visualization scripts...")
    
    try:
        # Check plot_training_curves.py
        with open("scripts/plot_training_curves.py", "r") as f:
            content = f.read()
            assert "plot_loss_curves" in content
            assert "plot_accuracy_curves" in content
            assert "plot_robustness_comparison" in content
        print("   ✅ plot_training_curves.py has required functions")
        
        # Check compare_models.py
        with open("scripts/compare_models.py", "r") as f:
            content = f.read()
            assert "evaluate_model" in content
            assert "plot_comparison" in content
            assert "print_comparison_table" in content
        print("   ✅ compare_models.py has required functions")
        
        return True
    except Exception as e:
        print(f"   ❌ Visualization scripts test failed: {e}")
        return False


def main():
    print("="*70)
    print("PHASE 2 IMPLEMENTATION - SANITY CHECK")
    print("="*70)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("File Structure", test_module_structure()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("Training Modules", test_training_modules()))
    results.append(("CLI Integration", test_cli_integration()))
    results.append(("Visualization Scripts", test_visualization_scripts()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✅ All tests passed! Phase 2 is ready to use.")
        print("\nNext steps:")
        print("  1. Train baseline model:")
        print("     python run_demo.py --mode train --training-type baseline --config configs/training_config.yaml")
        print("\n  2. Train adversarial model:")
        print("     python run_demo.py --mode train --training-type adversarial --config configs/training_config.yaml")
        print("\n  3. Compare models:")
        print("     python scripts/compare_models.py \\")
        print("       --baseline-checkpoint outputs/models/baseline_model.pt \\")
        print("       --adversarial-checkpoint outputs/models/adversarial_model.pt")
        print()
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("  - Install missing dependencies: pip install torch torchvision matplotlib numpy")
        print("  - Ensure all files are in the correct locations")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
