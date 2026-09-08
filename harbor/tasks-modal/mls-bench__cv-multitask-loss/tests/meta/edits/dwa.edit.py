"""Dynamic Weight Average baseline (Liu et al., 2019).

Weights tasks by their relative loss change rate with temperature scaling.

Reference: Liu et al., "End-to-End Multi-Task Learning with Attention"
(CVPR 2019)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_mtl.py"

_CONTENT = """\
class MultiTaskLoss(nn.Module):
    \"\"\"Dynamic Weight Average (Liu et al., 2019).

    Weights tasks by relative loss change rate with temperature.
    \"\"\"

    def __init__(self, num_tasks=2):
        super().__init__()
        self.prev_losses = None
        self.T = 2.0  # temperature

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        losses = torch.stack([fine_loss, coarse_loss])
        if self.prev_losses is None or epoch == 0:
            weights = torch.ones(2, device=losses.device)
        else:
            ratios = losses.detach() / (self.prev_losses + 1e-8)
            weights = 2 * F.softmax(ratios / self.T, dim=0)
        self.prev_losses = losses.detach().clone()
        return (weights * losses).sum()
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
