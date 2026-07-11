"""Weak loss baseline: identity classification (softmax / CE) ONLY."""
def build_loss(num_train_ids):
    from torchreid.losses import CrossEntropyLoss
    xent = CrossEntropyLoss(num_classes=num_train_ids)
    def loss_fn(logits, features, labels):
        return xent(logits, labels)
    loss_fn.name = "softmax_only"
    return loss_fn
