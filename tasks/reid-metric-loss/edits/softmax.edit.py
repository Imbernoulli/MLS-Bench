"""Weak loss baseline: identity classification (softmax / cross-entropy) ONLY.
Reference: vendor/torchreid-reid/baselines/loss_softmax.py
"""
_FILE = "torchreid-reid/solution/loss.py"
_CONTENT = '''def build_loss(num_train_ids):
    from torchreid.losses import CrossEntropyLoss

    xent = CrossEntropyLoss(num_classes=num_train_ids)

    def loss_fn(logits, features, labels):
        return xent(logits, labels)

    loss_fn.name = "softmax_only"
    return loss_fn'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 21, "content": _CONTENT},
]
