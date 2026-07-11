"""Unconfigured diagnostic: batch-hard TRIPLET + label-smoothed CE + CENTER loss.

This variant adds center loss (Wen et al., ECCV 2016) on top of ID + triplet.
It is retained for diagnosis only because the measured retrieval metrics do not
strictly dominate the canonical triplet + softmax objective. Reference:
vendor/torchreid-reid/baselines/loss_triplet_softmax_center.py
"""
_FILE = "torchreid-reid/solution/loss.py"
_CONTENT = '''def build_loss(num_train_ids):
    import torch
    from torchreid.losses import CrossEntropyLoss, TripletLoss

    xent = CrossEntropyLoss(num_classes=num_train_ids)
    triplet = TripletLoss(margin=0.3)

    class _Center:
        """Center loss with EMA-updated (non-parametric) class centers.
        Centers live on the feature device and are updated in-place (detached),
        so they need no optimiser slot (Wen et al., ECCV 2016)."""
        def __init__(self, num_ids, alpha=0.5, weight=0.0005):
            self.centers = None
            self.num_ids = num_ids
            self.alpha = alpha
            self.weight = weight

        def __call__(self, features, labels):
            if self.centers is None:
                self.centers = torch.zeros(
                    self.num_ids, features.size(1), device=features.device)
            c = self.centers[labels]                       # [B, D]
            loss = ((features - c) ** 2).sum(dim=1).mean()
            # EMA update of centers toward the batch means (no gradient)
            with torch.no_grad():
                for cid in labels.unique():
                    m = labels == cid
                    self.centers[cid] = (
                        (1 - self.alpha) * self.centers[cid]
                        + self.alpha * features[m].mean(0))
            return self.weight * loss

    center = _Center(num_train_ids)

    def loss_fn(logits, features, labels):
        return xent(logits, labels) + triplet(features, labels) + center(features, labels)

    loss_fn.name = "triplet_softmax_center"
    return loss_fn'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 21, "content": _CONTENT},
]
