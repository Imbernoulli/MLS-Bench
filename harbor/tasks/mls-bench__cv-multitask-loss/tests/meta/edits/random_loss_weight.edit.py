"""Random Loss Weighting (RLW) baseline.

Samples task weights from a symmetric Dirichlet distribution each step,
providing implicit regularization through stochastic loss weighting.
Surprisingly competitive with more complex adaptive methods.

Reference: Lin et al., "Reasonable Effectiveness of Random Weighting:
A Litmus Test for Multi-Task Learning" (TMLR 2022)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_mtl.py"

_CONTENT = """\
class MultiTaskLoss(nn.Module):
    \"\"\"Random Loss Weighting (Lin et al., TMLR 2022).

    Samples task weights from Dir(1, 1) each step. The stochastic
    weighting acts as implicit regularization and is surprisingly
    effective as a multi-task learning baseline.
    \"\"\"

    def __init__(self, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        # Sample from Dirichlet(1, 1) = Uniform on simplex
        if self.training:
            weights = torch.distributions.Dirichlet(
                torch.ones(self.num_tasks, device=fine_loss.device)
            ).sample()
            # Scale so mean weight = 1 (weights sum to num_tasks)
            weights = weights * self.num_tasks
        else:
            weights = torch.ones(self.num_tasks, device=fine_loss.device)

        return weights[0] * fine_loss + weights[1] * coarse_loss
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
