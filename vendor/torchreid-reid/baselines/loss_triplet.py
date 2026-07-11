"""Metric loss baseline: batch-hard TRIPLET ONLY (no id classifier)."""
def build_loss(num_train_ids):
    from torchreid.losses import TripletLoss
    triplet = TripletLoss(margin=0.3)
    def loss_fn(logits, features, labels):
        return triplet(features, labels)
    loss_fn.name = "triplet_only"
    return loss_fn
