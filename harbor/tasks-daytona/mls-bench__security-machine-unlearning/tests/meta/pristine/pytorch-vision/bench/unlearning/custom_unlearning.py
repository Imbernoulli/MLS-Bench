"""Editable unlearning method for MLS-Bench."""

import torch
import torch.nn.functional as F

# ============================================================
# EDITABLE
# ============================================================
class UnlearningMethod:
    """Default retain-only finetuning update."""

    def __init__(self):
        self.forget_weight = 0.0

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        retain_x, retain_y = retain_batch
        logits = model(retain_x)
        loss = F.cross_entropy(logits, retain_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
# ============================================================
# END EDITABLE
# ============================================================
