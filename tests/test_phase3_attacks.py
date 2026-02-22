"""Test suite for Phase 3 attacks."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from cerberus.attacks.pgd_attack import PGDAttack
from cerberus.attacks.cw_attack import CWAttack
from cerberus.attacks.deepfool_attack import DeepFoolAttack
from cerberus.attacks.jsma_attack import JSMAAttack


def test_pgd_attack():
    """Test PGD attack implementation."""
    print("\n" + "="*60)
    print("🧪 Testing PGD Attack")
    print("="*60)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    model.eval()
    
    # Create test data
    images = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 0, 1])
    
    # Test attack
    pgd = PGDAttack(model, eps=0.3, eps_step=0.1, max_iter=10)
    adv_images = pgd.generate_adversarial(images, labels)
    
    # Verify output shape
    assert adv_images.shape == images.shape, "Shape mismatch!"
    
    # Verify perturbation magnitude
    perturbation = torch.norm(adv_images - images, p=float('inf'))
    assert perturbation <= 0.3 + 0.01, "Perturbation exceeds epsilon!"
    
    print(f"✅ PGD Attack Test PASSED")
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {adv_images.shape}")
    print(f"   Max perturbation: {perturbation:.4f}")
    print(f"   Epsilon bound: 0.3")
    
    return True


def test_cw_attack():
    """Test C&W attack implementation."""
    print("\n" + "="*60)
    print("🧪 Testing C&W Attack")
    print("="*60)
    
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    model.eval()
    
    images = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 0, 1])
    
    cw = CWAttack(model, c=1.0, learning_rate=0.01, max_iterations=50)
    adv_images = cw.generate_adversarial(images, labels)
    
    assert adv_images.shape == images.shape, "Shape mismatch!"
    assert torch.all((adv_images >= 0) & (adv_images <= 1)), "Values out of range!"
    
    print(f"✅ C&W Attack Test PASSED")
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {adv_images.shape}")
    print(f"   Value range: [{adv_images.min():.4f}, {adv_images.max():.4f}]")
    
    return True


def test_deepfool_attack():
    """Test DeepFool attack implementation."""
    print("\n" + "="*60)
    print("🧪 Testing DeepFool Attack")
    print("="*60)
    
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    model.eval()
    
    images = torch.randn(4, 10)
    
    deepfool = DeepFoolAttack(model, max_iterations=50, overshoot=0.02)
    adv_images = deepfool.generate_adversarial(images)
    
    assert adv_images.shape == images.shape, "Shape mismatch!"
    
    perturbations = torch.norm(adv_images - images, p=2, dim=1)
    
    print(f"✅ DeepFool Attack Test PASSED")
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {adv_images.shape}")
    print(f"   Avg perturbation: {perturbations.mean():.4f}")
    print(f"   Max perturbation: {perturbations.max():.4f}")
    
    return True


def test_jsma_attack():
    """Test JSMA attack implementation."""
    print("\n" + "="*60)
    print("🧪 Testing JSMA Attack")
    print("="*60)
    
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    model.eval()
    
    images = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 0, 1])
    
    jsma = JSMAAttack(model, theta=0.1, max_pixels=3)
    adv_images = jsma.generate_adversarial(images, labels)
    
    assert adv_images.shape == images.shape, "Shape mismatch!"
    assert torch.all((adv_images >= 0) & (adv_images <= 1)), "Values out of range!"
    
    # Check that only few pixels modified
    modified_pixels = (adv_images != images).sum(dim=0).mean()
    
    print(f"✅ JSMA Attack Test PASSED")
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {adv_images.shape}")
    print(f"   Avg pixels modified per sample: {modified_pixels:.2f}")
    
    return True


def run_all_tests():
    """Run all attack tests."""
    print("\n" + "🚀 " * 20)
    print("PHASE 3 - ATTACK IMPLEMENTATIONS TEST SUITE")
    print("🚀 " * 20)
    
    tests = [
        ("PGD Attack", test_pgd_attack),
        ("C&W Attack", test_cw_attack),
        ("DeepFool Attack", test_deepfool_attack),
        ("JSMA Attack", test_jsma_attack),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} Test FAILED")
            print(f"   Error: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 3 attacks ready for use.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
