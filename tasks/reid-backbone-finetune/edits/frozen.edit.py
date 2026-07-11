"""Weak baseline: FREEZE the entire backbone (train only the classifier head on
fixed ImageNet features). Pedestrian features far from ImageNet -> under-fits.
Reference: vendor/torchreid-reid/baselines/finetune_frozen.py
"""
_FILE = "torchreid-reid/solution/finetune.py"
_CONTENT = '''def configure_trainable(backbone):
    for p in backbone.parameters():
        p.requires_grad_(False)
    return None'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 15, "content": _CONTENT},
]
