"""Tests for attack wrappers (using mocks to avoid ART dependency)."""
import sys
from unittest.mock import MagicMock, patch


def test_run_fgsm_attack_raises_if_art_missing():
    """Test that run_fgsm_attack raises a clear error if ART is not available."""
    # Remove ART modules to simulate missing
    original_art = sys.modules.pop("art", None)
    original_art_attacks = sys.modules.pop("art.attacks", None)
    original_art_attacks_evasion = sys.modules.pop("art.attacks.evasion", None)
    original_art_estimators = sys.modules.pop("art.estimators", None)
    original_art_estimators_classification = sys.modules.pop("art.estimators.classification", None)

    try:
        from cerberus.attacks import run_fgsm_attack

        mock_model = MagicMock()
        mock_loader = MagicMock()

        try:
            run_fgsm_attack(mock_model, mock_loader)
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "ART is required" in str(e)
    finally:
        if original_art:
            sys.modules["art"] = original_art
        if original_art_attacks:
            sys.modules["art.attacks"] = original_art_attacks
        if original_art_attacks_evasion:
            sys.modules["art.attacks.evasion"] = original_art_attacks_evasion
        if original_art_estimators:
            sys.modules["art.estimators"] = original_art_estimators
        if original_art_estimators_classification:
            sys.modules["art.estimators.classification"] = original_art_estimators_classification
