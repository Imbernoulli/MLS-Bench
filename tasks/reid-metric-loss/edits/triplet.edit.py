"""Metric loss baseline: batch-hard TRIPLET ONLY.
Reference: vendor/torchreid-reid/baselines/loss_triplet.py
"""
_FILE = "torchreid-reid/solution/loss.py"
_CONTENT = '''def build_loss(num_train_ids):
    from torchreid.losses import TripletLoss

    triplet = TripletLoss(margin=0.3)

    def loss_fn(logits, features, labels):
        return triplet(features, labels)

    loss_fn.name = "triplet_only"
    return loss_fn'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 21, "content": _CONTENT},
]
