"""Uncertainty weighting baseline (Kendall et al., 2018).

Learns per-task log-variance parameters. Each task loss is weighted by
exp(-log_var) with a log_var regularization term.

Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
Losses for Scene Geometry and Semantics" (CVPR 2018)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_mtl.py"

_CONTENT = """\
class MultiTaskLoss(nn.Module):
    \"\"\"Uncertainty weighting (Kendall et al., 2018).

    Learns per-task log-variance: loss_i / exp(log_var_i) + log_var_i.
    \"\"\"

    def __init__(self, num_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        losses = [fine_loss, coarse_loss]
        total = sum(
            torch.exp(-self.log_vars[i]) * losses[i] + self.log_vars[i]
            for i in range(2)
        )
        return total
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
