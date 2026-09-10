"""Custom adversarial training method for MLS-Bench."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════
# EDITABLE — implement AdversarialTrainer below
# ═══════════════════════════════════════════════════════════════════
class AdversarialTrainer:
    """
    Adversarial training method.

    The agent should modify this class to implement a better adversarial
    training procedure that improves model robustness against L_inf attacks.

    Args:
        model (nn.Module): The model to train.
        eps (float): L_inf perturbation budget.
        alpha (float): Step size for adversarial perturbation generation.
        attack_steps (int): Number of PGD steps for adversarial example generation.
        num_classes (int): Number of output classes.
    """

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes

    def train_step(self, images, labels, optimizer):
        """
        Perform one adversarial training step.

        Args:
            images: Clean images, shape (N, C, H, W), values in [0, 1].
            labels: Ground truth labels, shape (N,).
            optimizer: Model optimizer (already configured).

        Returns:
            dict: Must contain 'loss' key (float).
        """
        # Default: standard (non-adversarial) training
        self.model.train()
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}

# ═══════════════════════════════════════════════════════════════════
# END EDITABLE
# ═══════════════════════════════════════════════════════════════════
