"""GradNorm baseline (Chen et al., ICML 2018).

Automatically balances task losses by dynamically adjusting per-task
weights based on inverse training rate so that all tasks train at
similar rates.

Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive
Loss Balancing in Deep Multitask Networks" (ICML 2018)

Since the training loop is fixed and we cannot access model parameters
inside forward(), we implement GradNorm's weight-update rule using
buffers updated via exponential moving averages of the inverse training
rate.  Weights are stored as buffers (not Parameters) to avoid
interference from the main SGD optimizer.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_mtl.py"

_CONTENT = """\
class MultiTaskLoss(nn.Module):
    \"\"\"GradNorm: dynamic task weight balancing (Chen et al., ICML 2018).

    Adjusts per-task weights each step based on inverse training rate
    so that tasks that are lagging behind receive higher weight.
    Alpha=1.5 (asymmetry parameter from the paper).
    \"\"\"

    def __init__(self, num_tasks=2):
        super().__init__()
        # Use buffers, not parameters, so the main optimizer ignores them
        self.register_buffer('weights', torch.ones(num_tasks))
        self.register_buffer('initial_losses', torch.zeros(num_tasks))
        self.has_initial = False
        self.alpha = 1.5   # restoring force strength
        self.lr_w = 0.025  # weight update learning rate

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        losses = torch.stack([fine_loss, coarse_loss])
        losses_d = losses.detach()

        # Record initial losses at epoch 0
        if not self.has_initial:
            self.initial_losses.copy_(losses_d)
            self.has_initial = True

        # Inverse training rate: r_i = L_i(t) / L_i(0)
        r = losses_d / (self.initial_losses + 1e-8)
        # Relative inverse training rate (normalized by mean)
        r_mean = r.mean()
        r_rel = r / (r_mean + 1e-8)

        # Target weight: tasks with higher relative inverse training rate
        # (training slower) should get higher weight
        target_w = r_rel ** self.alpha
        # Normalize targets so they sum to num_tasks
        target_w = target_w * (2.0 / (target_w.sum() + 1e-8))

        # Update weights with a small step towards the target
        self.weights.copy_(
            (1 - self.lr_w) * self.weights + self.lr_w * target_w
        )
        # Renormalize so weights sum to num_tasks (mean = 1)
        self.weights.copy_(self.weights * (2.0 / (self.weights.sum() + 1e-8)))

        # Weighted loss (detach weights so gradients don't flow through them)
        weighted_loss = (self.weights.detach() * losses).sum()
        return weighted_loss
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 195,
        "end_line": 216,
        "content": _CONTENT,
    },
]
