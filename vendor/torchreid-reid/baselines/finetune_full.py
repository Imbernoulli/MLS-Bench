"""Strong baseline: FULL fine-tuning (all backbone layers trainable).
Adapts every layer to the pedestrian domain -> strongest.
Reference: vendor/torchreid-reid/baselines/finetune_full.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def configure_trainable(backbone):
    for p in backbone.parameters():
        p.requires_grad_(True)
    return None
