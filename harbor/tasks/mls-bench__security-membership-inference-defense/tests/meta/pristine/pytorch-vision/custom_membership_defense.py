"""Editable membership-inference defense for MLS-Bench."""

import torch
import torch.nn.functional as F

# ============================================================
# EDITABLE
# ============================================================
class MembershipDefense:
    """Training-time regularizer for privacy-utility tradeoffs.

    The compute_loss method replaces nn.CrossEntropyLoss() in the
    fixed training loop.  Design a loss that reduces membership
    inference leakage (lower MIA AUC) while preserving test accuracy.

    Args:
        logits: raw model outputs, shape (batch_size, num_classes)
        labels: ground-truth class indices, shape (batch_size,)
        epoch:  current training epoch (0-indexed)

    Returns:
        scalar loss tensor
    """

    def __init__(self):
        pass

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels)
# ============================================================
# END EDITABLE
# ============================================================
