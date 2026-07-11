"""Agent-editable solution surface for person re-identification.

Keep the public callable names and signatures below. Follow the fixed harness
contract for return types, shapes, devices, value ranges, finiteness, and
determinism. The selected implementation is evaluated directly; contract or
runtime failures invalidate the run.
"""
from __future__ import annotations


# EDITABLE REGION
def build_loss(num_train_ids):
    from torchreid.losses import CrossEntropyLoss

    xent = CrossEntropyLoss(num_classes=num_train_ids)

    def loss_fn(logits, features, labels):
        return xent(logits, labels)

    loss_fn.name = "default_loss"
    return loss_fn
# END EDITABLE REGION

