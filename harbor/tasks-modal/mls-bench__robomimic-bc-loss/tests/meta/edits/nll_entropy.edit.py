"""NLL + entropy bonus baseline -- rigorous codebase edit ops.

Replaces the default NLL loss with NLL plus an entropy regularization term.
The entropy bonus encourages broader GMM modes, preventing premature
collapse to a single mode and improving exploration during training.

loss = -log_prob(actions).mean() - alpha * entropy.mean()

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "robomimic/custom_bc_loss.py"

# ── Replace CustomBCLoss class (lines 20-42) ────────────────────────────

_NLL_ENTROPY_CLASS = """\
class CustomBCLoss(nn.Module):
    \"\"\"NLL + entropy bonus for GMM-based behavioral cloning.

    Adds an entropy regularization term to the standard NLL loss.
    The entropy bonus encourages broader mixture components,
    preventing premature mode collapse.
    \"\"\"

    def __init__(self, action_dim=7, alpha=0.01):
        super().__init__()
        self.action_dim = action_dim
        self.alpha = alpha

    def forward(self, dist, target_actions):
        nll = -dist.log_prob(target_actions).mean()
        # MixtureSameFamily has no closed-form entropy; approximate via sampling
        with torch.no_grad():
            samples = dist.sample()
        entropy = -dist.log_prob(samples).mean()
        return nll - self.alpha * entropy
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 41,
        "content": _NLL_ENTROPY_CLASS,
    },
]
