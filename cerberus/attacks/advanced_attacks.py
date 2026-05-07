"""Advanced attack implementations used by the Flask backend.

These are lightweight in-house versions intended for demo/benchmark mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pgd_attack import PGDAttack
from .cw_attack import CWAttack
from .deepfool_attack import DeepFoolAttack


class AutoAttackEnsemble:
    """AutoAttack-style ensemble using multiple strong white-box attacks."""

    def __init__(self, model, eps=0.03, device="cpu"):
        self.model = model.to(device)
        self.eps = eps
        self.device = device

    def generate_adversarial(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        attacks = [
            PGDAttack(self.model, eps=self.eps, eps_step=max(self.eps / 10, 1e-4), max_iter=40, device=self.device),
            CWAttack(self.model, c=1.0, learning_rate=0.01, max_iterations=120, device=self.device),
            DeepFoolAttack(self.model, max_iterations=30, device=self.device),
            FABAttack(self.model, eps=self.eps, max_iter=30, device=self.device),
        ]

        candidates = []
        candidates.append(attacks[0].generate_adversarial(images, labels))
        candidates.append(attacks[1].generate_adversarial(images, labels))
        candidates.append(attacks[2].generate_adversarial(images))
        candidates.append(attacks[3].generate_adversarial(images, labels))

        # Keep the strongest candidate per sample (lowest true-class logit)
        best_adv = candidates[0].clone().detach()
        with torch.no_grad():
            logits = self.model(best_adv)
            best_score = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

            for candidate in candidates[1:]:
                cand_logits = self.model(candidate)
                cand_score = cand_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                better_mask = cand_score < best_score
                if better_mask.any():
                    best_adv[better_mask] = candidate[better_mask]
                    best_score[better_mask] = cand_score[better_mask]

        return torch.clamp(best_adv, 0, 1).detach()


class SquareAttack:
    """Query-based random square perturbation attack."""

    def __init__(self, model, eps=0.03, max_queries=300, device="cpu"):
        self.model = model.to(device)
        self.eps = eps
        self.max_queries = max_queries
        self.device = device

    def generate_adversarial(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        x_adv = images.clone().detach()
        delta = torch.zeros_like(images, device=self.device)
        h, w = images.shape[2], images.shape[3]

        self.model.eval()
        for query in range(self.max_queries):
            size = max(1, int((1.0 - query / self.max_queries) * min(h, w) / 2))
            top = torch.randint(0, max(1, h - size + 1), (1,), device=self.device).item()
            left = torch.randint(0, max(1, w - size + 1), (1,), device=self.device).item()

            square_pert = torch.zeros_like(images, device=self.device)
            square_pert[:, :, top:top + size, left:left + size] = torch.randn_like(
                square_pert[:, :, top:top + size, left:left + size]
            ) * self.eps

            x_test = torch.clamp(images + delta + square_pert, 0, 1)
            with torch.no_grad():
                preds = self.model(x_test).argmax(dim=1)
                improved = preds != labels
                if improved.any():
                    delta[improved] = (delta + square_pert)[improved]

        x_adv = torch.clamp(images + delta, 0, 1)
        return x_adv.detach()


class FABAttack:
    """Simplified FAB-style boundary attack."""

    def __init__(self, model, eps=0.03, max_iter=40, device="cpu"):
        self.model = model.to(device)
        self.eps = eps
        self.max_iter = max_iter
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def generate_adversarial(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        delta = torch.zeros_like(images, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=0.01)

        self.model.eval()
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            x_adv = torch.clamp(images + delta, 0, 1)
            logits = self.model(x_adv)

            ce_loss = self.criterion(logits, labels)
            l2_penalty = torch.norm(delta.reshape(delta.size(0), -1), p=2, dim=1).mean()
            loss = ce_loss + 0.02 * l2_penalty

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)

        return torch.clamp(images + delta.detach(), 0, 1).detach()


class RaySAttack:
    """Ray search style attack with PGD initialization."""

    def __init__(self, model, eps=0.03, max_iter=20, device="cpu"):
        self.model = model.to(device)
        self.eps = eps
        self.max_iter = max_iter
        self.device = device

    def generate_adversarial(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        init_attack = PGDAttack(self.model, eps=self.eps, eps_step=max(self.eps / 10, 1e-4), max_iter=20, device=self.device)
        x_adv = init_attack.generate_adversarial(images, labels)

        self.model.eval()
        for _ in range(self.max_iter):
            for t_val in [0.5, 0.25, 0.75, 0.125, 0.875]:
                x_test = torch.clamp(images + t_val * (x_adv - images), 0, 1)
                with torch.no_grad():
                    preds = self.model(x_test).argmax(dim=1)
                    is_adv = preds != labels
                    if is_adv.any():
                        x_adv[is_adv] = x_test[is_adv]

        x_adv = torch.max(torch.min(x_adv, images + self.eps), images - self.eps)
        return torch.clamp(x_adv, 0, 1).detach()


class TRADESAttack:
    """TRADES-style KL-regularized attack."""

    def __init__(self, model, eps=0.03, beta=6.0, max_iter=30, device="cpu"):
        self.model = model.to(device)
        self.eps = eps
        self.beta = beta
        self.max_iter = max_iter
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def generate_adversarial(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

        x_adv = images.clone().detach()
        alpha = max(self.eps / 10, 1e-4)

        self.model.eval()
        for _ in range(self.max_iter):
            x_adv.requires_grad_(True)
            logits_clean = self.model(images)
            logits_adv = self.model(x_adv)

            ce = self.criterion(logits_adv, labels)
            kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean, dim=1),
                reduction='batchmean'
            )
            loss = ce + self.beta * kl

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.max(torch.min(x_adv, images + self.eps), images - self.eps)
                x_adv = torch.clamp(x_adv, 0, 1).detach()

        return x_adv
