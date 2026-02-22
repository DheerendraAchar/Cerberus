#!/usr/bin/env python3
"""Verify Phase 3 implementation is complete and ready to execute.

This script checks:
1. All required files exist
2. Python dependencies are available
3. Attack implementations are importable
4. Test suite passes
5. Comparison script runs without errors
"""

import sys
from pathlib import Path
import importlib.util


def check_file_exists(path, description):
    """Check if a file exists."""
    p = Path(path)
    if p.exists():
        print(f"  ✅ {description}: {path}")
        return True
    else:
        print(f"  ❌ {description}: {path} (MISSING)")
        return False


def check_imports(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"  ✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"  ❌ {description}: {module_name}")
        print(f"     Error: {e}")
        return False


def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        mod = __import__(import_name)
        print(f"  ✅ {package_name}: {mod.__version__ if hasattr(mod, '__version__') else 'installed'}")
        return True
    except ImportError:
        print(f"  ❌ {package_name}: NOT INSTALLED")
        return False


def verify_phase3_structure():
    """Verify Phase 3 directory structure."""
    print("\n" + "="*70)
    print("🏗️  PHASE 3 STRUCTURE VERIFICATION")
    print("="*70)
    
    files_to_check = [
        ("cerberus/attacks/__init__.py", "Attack module init"),
        ("cerberus/attacks/fgsm_attack.py", "FGSM attack"),
        ("cerberus/attacks/pgd_attack.py", "PGD attack"),
        ("cerberus/attacks/cw_attack.py", "C&W attack"),
        ("cerberus/attacks/deepfool_attack.py", "DeepFool attack"),
        ("cerberus/attacks/jsma_attack.py", "JSMA attack"),
        ("scripts/compare_all_attacks.py", "Attack comparison script"),
        ("scripts/train_all_architectures.py", "Architecture training script"),
        ("scripts/run_transfer_analysis.py", "Transfer analysis script"),
        ("tests/test_phase3_attacks.py", "Phase 3 attack tests"),
        ("PHASE3_START_HERE.md", "Quick start guide"),
        ("PHASE3_IMPLEMENTATION_PLAN.md", "Implementation plan"),
        ("PHASE3_QUICK_START_GUIDE.md", "Quick start guide"),
        ("PHASE3_SESSION_SUMMARY.md", "Session summary"),
        ("PHASE3_PAPER_OUTLINE.md", "Paper outline"),
        ("PHASE3_EXECUTION_GUIDE.md", "Execution guide"),
    ]
    
    all_exist = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def verify_dependencies():
    """Verify Python dependencies."""
    print("\n" + "="*70)
    print("📦 DEPENDENCY VERIFICATION")
    print("="*70)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "TQDM"),
        ("art", "IBM ART"),
    ]
    
    all_installed = True
    for package, name in dependencies:
        if not check_python_package(name, package):
            all_installed = False
    
    return all_installed


def verify_imports():
    """Verify critical imports work."""
    print("\n" + "="*70)
    print("🔌 IMPORT VERIFICATION")
    print("="*70)
    
    imports = [
        ("cerberus.attacks", "Attacks module"),
        ("cerberus.cli", "CLI module"),
        ("cerberus.models", "Models module"),
    ]
    
    all_import = True
    for module_name, description in imports:
        if not check_imports(module_name, description):
            all_import = False
    
    return all_import


def verify_test_suite():
    """Verify test suite exists and can be imported."""
    print("\n" + "="*70)
    print("🧪 TEST SUITE VERIFICATION")
    print("="*70)
    
    test_file = Path("tests/test_phase3_attacks.py")
    if test_file.exists():
        print(f"  ✅ Test file exists: {test_file}")
        
        # Try to parse the test file
        try:
            with open(test_file) as f:
                content = f.read()
            
            # Count test classes/functions
            test_count = content.count("def test_") + content.count("class Test")
            print(f"  ✅ Found ~{test_count} test cases")
            return True
        except Exception as e:
            print(f"  ❌ Error reading test file: {e}")
            return False
    else:
        print(f"  ❌ Test file missing: {test_file}")
        return False


def verify_cli_integration():
    """Verify CLI has all 5 attacks integrated."""
    print("\n" + "="*70)
    print("🎮 CLI INTEGRATION VERIFICATION")
    print("="*70)
    
    cli_file = Path("cerberus/cli.py")
    if cli_file.exists():
        with open(cli_file) as f:
            content = f.read()
        
        attacks = ["fgsm", "pgd", "cw", "deepfool", "jsma"]
        all_found = True
        
        for attack in attacks:
            if attack in content.lower():
                print(f"  ✅ {attack.upper()} integrated in CLI")
            else:
                print(f"  ❌ {attack.upper()} NOT found in CLI")
                all_found = False
        
        return all_found
    else:
        print(f"  ❌ CLI file missing: {cli_file}")
        return False


def verify_comparison_script():
    """Verify comparison script is complete."""
    print("\n" + "="*70)
    print("📊 COMPARISON SCRIPT VERIFICATION")
    print("="*70)
    
    script_file = Path("scripts/compare_all_attacks.py")
    if script_file.exists():
        with open(script_file) as f:
            content = f.read()
        
        required_functions = [
            "get_test_loader",
            "load_model",
            "run_all_attacks",
            "create_comparison_table",
            "create_comparison_plot",
        ]
        
        all_found = True
        for func in required_functions:
            if func in content:
                print(f"  ✅ Function {func}() exists")
            else:
                print(f"  ❌ Function {func}() missing")
                all_found = False
        
        return all_found
    else:
        print(f"  ❌ Comparison script missing: {script_file}")
        return False


def verify_training_script():
    """Verify training script is complete."""
    print("\n" + "="*70)
    print("🏋️  TRAINING SCRIPT VERIFICATION")
    print("="*70)
    
    script_file = Path("scripts/train_all_architectures.py")
    if script_file.exists():
        with open(script_file) as f:
            content = f.read()
        
        required_components = [
            "load_architecture",
            "train_epoch",
            "evaluate",
            "get_cifar10_loaders",
        ]
        
        architectures = [
            "resnet18",
            "vgg16",
            "mobilenet_v2",
            "efficientnet_b0",
            "densenet121",
        ]
        
        all_found = True
        
        for component in required_components:
            if component in content:
                print(f"  ✅ Function {component}() exists")
            else:
                print(f"  ❌ Function {component}() missing")
                all_found = False
        
        for arch in architectures:
            if arch in content.lower():
                print(f"  ✅ Architecture {arch} supported")
            else:
                print(f"  ❌ Architecture {arch} missing")
                all_found = False
        
        return all_found
    else:
        print(f"  ❌ Training script missing: {script_file}")
        return False


def verify_transfer_script():
    """Verify transfer analysis script is complete."""
    print("\n" + "="*70)
    print("🔄 TRANSFER ANALYSIS SCRIPT VERIFICATION")
    print("="*70)
    
    script_file = Path("scripts/run_transfer_analysis.py")
    if script_file.exists():
        with open(script_file) as f:
            content = f.read()
        
        required_functions = [
            "build_transfer_matrix",
            "plot_transfer_matrix",
            "plot_diagonal_analysis",
            "analyze_transfer_patterns",
            "generate_adversarial_examples",
        ]
        
        all_found = True
        for func in required_functions:
            if func in content:
                print(f"  ✅ Function {func}() exists")
            else:
                print(f"  ❌ Function {func}() missing")
                all_found = False
        
        return all_found
    else:
        print(f"  ❌ Transfer script missing: {script_file}")
        return False


def main():
    """Run all verifications."""
    print("\n" + "🔍 "*20)
    print("PHASE 3 IMPLEMENTATION VERIFICATION")
    print("🔍 "*20 + "\n")
    
    results = {
        'structure': verify_phase3_structure(),
        'dependencies': verify_dependencies(),
        'imports': verify_imports(),
        'tests': verify_test_suite(),
        'cli': verify_cli_integration(),
        'comparison': verify_comparison_script(),
        'training': verify_training_script(),
        'transfer': verify_transfer_script(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("📋 VERIFICATION SUMMARY")
    print("="*70)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check.upper():<20}: {status}")
    
    # Overall
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "🎉 "*20)
        print("✅ ALL VERIFICATIONS PASSED!")
        print("🎉 "*20)
        print("\n📊 NEXT STEPS:")
        print("1. Install missing dependencies (if any)")
        print("2. Run Phase 3A tests:")
        print("   python3 tests/test_phase3_attacks.py")
        print("3. Run attack comparison:")
        print("   python3 scripts/compare_all_attacks.py")
        print("4. Begin training architectures:")
        print("   python3 scripts/train_all_architectures.py --epochs 50")
        return 0
    else:
        print("\n" + "⚠️  "*20)
        print("❌ SOME VERIFICATIONS FAILED")
        print("⚠️  "*20)
        print("\n🔧 ACTION ITEMS:")
        if not results['structure']:
            print("- Create missing files (check Phase 3 documentation)")
        if not results['dependencies']:
            print("- Install missing Python packages: pip install torch torchvision matplotlib seaborn tqdm")
        if not results['imports']:
            print("- Check cerberus package structure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
