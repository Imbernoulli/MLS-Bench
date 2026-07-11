"""Medium baseline: freeze the EARLY layers (conv1, bn1, layer1, layer2) and
fine-tune the later, more task-specific layers (layer3, layer4).
Reference: vendor/torchreid-reid/baselines/finetune_partial.py
"""
_FILE = "torchreid-reid/solution/finetune.py"
_CONTENT = '''def configure_trainable(backbone):
    # first fine-tune everything, then freeze the early stem + first two stages
    for p in backbone.parameters():
        p.requires_grad_(True)
    for mod in (backbone.conv1, backbone.bn1, backbone.layer1, backbone.layer2):
        for p in mod.parameters():
            p.requires_grad_(False)
    return None'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 15, "content": _CONTENT},
]
