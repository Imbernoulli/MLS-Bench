"""Strong baseline: FULL fine-tuning (all backbone layers trainable).
Adapts every layer to the pedestrian domain -> strongest.
Reference: vendor/torchreid-reid/baselines/finetune_full.py
"""
_FILE = "torchreid-reid/solution/finetune.py"
_CONTENT = '''def configure_trainable(backbone):
    for p in backbone.parameters():
        p.requires_grad_(True)
    return None'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 15, "content": _CONTENT},
]
