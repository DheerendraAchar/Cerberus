# Text Attack Expansion - Summary

## Overview
Added **3 new text adversarial attacks** to complement the existing FGSM and PGD attacks on AG News dataset.

**Total text attacks now: 5** (was 2)

## New Attacks Implemented

### 1. **TokenSwap Attack** (`tokenswap`)
- **Location**: `cerberus/attacks/text_attacks.py`
- **Description**: Randomly swaps adjacent tokens in sequences to craft adversarial examples
- **Method**: Structural perturbation by reordering tokens
- **Epsilon parameter**: Controls swap intensity (0-1), higher = more swaps
- **Use case**: Tests model robustness to word order variations

### 2. **TokenNoise Attack** (`tokennoise`)
- **Location**: `cerberus/attacks/text_attacks.py`
- **Description**: Adds token-level noise by small random perturbations to token indices
- **Method**: Gaussian-like noise added to vocabulary indices within valid range
- **Epsilon parameter**: Noise magnitude scaled by vocab size
- **Use case**: Tests model robustness to character/token corruption

### 3. **TokenSubstitution Attack** (`tokensubstitution`)
- **Location**: `cerberus/attacks/text_attacks.py`
- **Description**: Replaces tokens with vocabulary-nearby alternatives (synonym-like)
- **Method**: Replaces selected tokens with nearby vocab indices (±20 offset)
- **Epsilon parameter**: Substitution rate - fraction of tokens to replace
- **Use case**: Tests model robustness to semantic word replacements

## Files Modified

### Backend
- **`backend.py`**:
  - Added imports for new attack classes
  - Updated `SUPPORTED_TEXT_ATTACKS` set: `{'fgsm', 'pgd', 'tokenswap', 'tokennoise', 'tokensubstitution'}`
  - Updated `run_attack_text()` function to handle all 5 attacks with dispatch logic

### New Files
- **`cerberus/attacks/text_attacks.py`** (NEW):
  - Implements `TokenSwapAttack` class
  - Implements `TokenNoiseAttack` class
  - Implements `TokenSubstitutionAttack` class
  - All follow consistent interface: `__call__(texts, model, labels, vocab_size)`

### Frontend
- **`frontend/src/components/AttackPanel.js`**:
  - Updated text attack options to show all 5 attacks organized in categories:
    - **Gradient-Based**: FGSM, PGD
    - **Structural**: Token Swap, Token Noise
    - **Semantic**: Token Substitution

- **`frontend/src/components/DefensePanel.js`**:
  - Updated defense comparison dropdown with same 5 text attacks

## Attack Categories (Frontend UI)

### For CIFAR-10 (Images) - 10 attacks:
- Gradient-Based (Basic): FGSM
- Gradient-Based (Iterative): PGD
- Optimization-Based: C&W
- Geometric: DeepFool
- Feature-Based: JSMA
- Advanced Ensemble: AutoAttack, Square, FAB, RayS, TRADES

### For AG News (Text) - 5 attacks:
- **Gradient-Based**: FGSM (token perturbation), PGD (iterative token perturbation)
- **Structural**: Token Swap (adjacent token swapping), Token Noise (random perturbation)
- **Semantic**: Token Substitution (vocabulary replacement)

## Verification

✓ All 5 text attacks imported successfully
✓ Attack classes instantiate correctly
✓ Backend SUPPORTED_TEXT_ATTACKS updated
✓ Frontend dropdowns show all 5 text attacks
✓ Frontend builds successfully (159.26 kB gzipped)

## Testing

To test the new attacks via API:
```bash
# Token Swap
curl -X POST http://localhost:5000/api/run-attack \
  -d '{"attack": "tokenswap", "architecture": "TextCNN", "epsilon": 0.15, "num_samples": 2, "dataset": "agnews"}'

# Token Noise
curl -X POST http://localhost:5000/api/run-attack \
  -d '{"attack": "tokennoise", "architecture": "TextCNN", "epsilon": 0.1, "num_samples": 2, "dataset": "agnews"}'

# Token Substitution
curl -X POST http://localhost:5000/api/run-attack \
  -d '{"attack": "tokensubstitution", "architecture": "TextCNN", "epsilon": 0.15, "num_samples": 2, "dataset": "agnews"}'
```

## Impact

- **Diversity**: Text attack coverage now matches image attacks (5 vs 10, proportional to domain complexity)
- **Demonstration**: Expo panel can now show variety of text perturbation strategies
- **Robustness Testing**: Models can be evaluated against different perturbation types (structural, semantic, gradient-based)
