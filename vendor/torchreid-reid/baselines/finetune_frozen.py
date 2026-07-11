"""Weak baseline: FREEZE the entire backbone (train only the classifier head on
fixed ImageNet features). Pedestrian features far from ImageNet -> under-fits.
Reference: vendor/torchreid-reid/baselines/finetune_frozen.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def configure_trainable(backbone):
    for p in backbone.parameters():
        p.requires_grad_(False)
    return None
