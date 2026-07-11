"""Strong loss baseline: batch-hard TRIPLET + label-smoothed CE (canonical ReID).
Reference: vendor/torchreid-reid/baselines/loss_triplet_softmax.py
"""
_FILE = "torchreid-reid/solution/loss.py"
_CONTENT = '''def build_loss(num_train_ids):
    from torchreid.losses import CrossEntropyLoss, TripletLoss

    xent = CrossEntropyLoss(num_classes=num_train_ids)
    triplet = TripletLoss(margin=0.3)

    def loss_fn(logits, features, labels):
        return xent(logits, labels) + triplet(features, labels)

    loss_fn.name = "triplet_softmax"
    return loss_fn'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 21, "content": _CONTENT},
]
