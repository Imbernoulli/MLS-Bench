"""SOTA reference: TRIPLET + softmax + CENTER loss (Luo "Bag of Tricks" 2019).

Center loss (Wen et al., ECCV 2016) with EMA-updated non-parametric class centers,
added on top of the canonical id + triplet objective. Centers are updated in-place
(detached) so they require no optimiser slot.
"""


def build_loss(num_train_ids):
    import torch
    from torchreid.losses import CrossEntropyLoss, TripletLoss

    xent = CrossEntropyLoss(num_classes=num_train_ids)
    triplet = TripletLoss(margin=0.3)

    class _Center:
        def __init__(self, num_ids, alpha=0.5, weight=0.0005):
            self.centers = None
            self.num_ids = num_ids
            self.alpha = alpha
            self.weight = weight

        def __call__(self, features, labels):
            if self.centers is None:
                self.centers = torch.zeros(
                    self.num_ids, features.size(1), device=features.device)
            c = self.centers[labels]
            loss = ((features - c) ** 2).sum(dim=1).mean()
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
    return loss_fn
