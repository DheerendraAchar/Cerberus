"""Advanced text-based adversarial attacks for AG News."""

import torch
import torch.nn.functional as F


class TokenSwapAttack:
    """Swap adjacent tokens in sequences to craft adversarial examples."""
    
    def __init__(self, epsilon=0.1):
        """
        Args:
            epsilon: Controls swap intensity (0-1). Higher = more swaps.
        """
        self.epsilon = epsilon
    
    def __call__(self, texts, model, labels, vocab_size=20000):
        """
        Apply token swap attack.
        
        Args:
            texts: Tensor of shape [batch_size, seq_len]
            model: Text model to attack
            labels: Ground truth labels
            vocab_size: Size of vocabulary
        
        Returns:
            Adversarial texts with same shape as input
        """
        adv_texts = texts.clone().detach()
        seq_len = adv_texts.size(1)
        
        # Determine number of swaps based on epsilon
        num_swaps = max(1, int(round(seq_len * self.epsilon)))
        
        for i in range(adv_texts.size(0)):  # for each sample in batch
            for _ in range(num_swaps):
                # Pick random position to swap with next position
                pos = torch.randint(0, seq_len - 1, (1,)).item()
                
                # Swap tokens (preserve PAD/UNK tokens at pos 0-1)
                if adv_texts[i, pos] > 1 and adv_texts[i, pos + 1] > 1:
                    adv_texts[i, pos], adv_texts[i, pos + 1] = adv_texts[i, pos + 1].clone(), adv_texts[i, pos].clone()
        
        return adv_texts


class TokenNoiseAttack:
    """Add token-level noise to sequences by small random perturbations."""
    
    def __init__(self, epsilon=0.05):
        """
        Args:
            epsilon: Noise magnitude (0-1).
        """
        self.epsilon = epsilon
    
    def __call__(self, texts, model, labels, vocab_size=20000):
        """
        Apply token noise attack.
        
        Args:
            texts: Tensor of shape [batch_size, seq_len]
            model: Text model to attack
            labels: Ground truth labels
            vocab_size: Size of vocabulary
        
        Returns:
            Adversarial texts with noisy token indices
        """
        adv_texts = texts.clone().detach().float()
        seq_len = adv_texts.size(1)
        
        # Generate noise proportional to epsilon
        noise_magnitude = int(round(vocab_size * self.epsilon))
        noise = torch.randint(-noise_magnitude, noise_magnitude + 1, adv_texts.shape, 
                              device=adv_texts.device, dtype=adv_texts.dtype)
        
        # Apply noise but keep tokens valid (> 1 for PAD/UNK)
        perturbed = adv_texts + noise
        perturbed = torch.clamp(perturbed, min=2, max=vocab_size - 1)
        
        # Only perturb non-PAD/UNK tokens
        mask = texts > 1
        adv_texts = torch.where(mask, perturbed, texts)
        
        return adv_texts.long()


class TokenSubstitutionAttack:
    """Replace tokens with vocabulary-nearby alternatives (synonym-like)."""
    
    def __init__(self, epsilon=0.15):
        """
        Args:
            epsilon: Substitution rate (0-1). Fraction of tokens to replace.
        """
        self.epsilon = epsilon
    
    def __call__(self, texts, model, labels, vocab_size=20000):
        """
        Apply token substitution attack.
        
        Args:
            texts: Tensor of shape [batch_size, seq_len]
            model: Text model to attack
            labels: Ground truth labels
            vocab_size: Size of vocabulary
        
        Returns:
            Adversarial texts with substituted tokens
        """
        adv_texts = texts.clone().detach()
        seq_len = adv_texts.size(1)
        
        # Determine number of tokens to substitute based on epsilon
        num_substitutions = max(1, int(round(seq_len * self.epsilon)))
        
        for i in range(adv_texts.size(0)):  # for each sample in batch
            # Get valid token positions (not PAD/UNK)
            valid_mask = adv_texts[i] > 1
            valid_positions = torch.where(valid_mask)[0]
            
            if len(valid_positions) > 0:
                # Randomly select positions to substitute
                num_to_sub = min(num_substitutions, len(valid_positions))
                positions = valid_positions[torch.randperm(len(valid_positions))[:num_to_sub]]
                
                # Replace with nearby vocab indices (±20 offset for semantic proximity)
                for pos in positions:
                    original_id = adv_texts[i, pos].item()
                    offset = torch.randint(-20, 21, (1,)).item()
                    new_id = max(2, min(vocab_size - 1, original_id + offset))
                    adv_texts[i, pos] = new_id
        
        return adv_texts
