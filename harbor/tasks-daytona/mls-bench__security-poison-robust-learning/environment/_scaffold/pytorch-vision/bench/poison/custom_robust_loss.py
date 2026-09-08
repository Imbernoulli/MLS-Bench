"""Editable poison-robust loss for MLS-Bench."""

import torch
import torch.nn.functional as F

# ============================================================
# EDITABLE
# ============================================================
class RobustLoss:
    """Default cross-entropy objective."""

    def __init__(self):
        pass

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels)
# ============================================================
# END EDITABLE
# ============================================================
