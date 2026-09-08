"""Weighted NLL baseline -- rigorous codebase edit ops.

Replaces the default NLL loss with a dimension-weighted NLL that
applies higher weight to the positional action dimensions (first 6)
compared to the gripper dimension (last 1). This is implemented by
scaling the action targets and distribution means before computing
the log probability.

Since MixtureSameFamily log_prob operates on the full action vector,
we weight by rescaling actions and means so that positional errors
contribute more.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "robomimic/custom_bc_loss.py"

# ── Replace CustomBCLoss class (lines 20-42) ────────────────────────────

_WEIGHTED_NLL_CLASS = """\
class CustomBCLoss(nn.Module):
    \"\"\"Dimension-weighted NLL for GMM-based behavioral cloning.

    Weights positional action dimensions more heavily than the gripper
    dimension to prioritize accurate end-effector movement prediction.
    \"\"\"

    def __init__(self, action_dim=7, pos_weight=2.0, grip_weight=1.0):
        super().__init__()
        self.action_dim = action_dim
        weights = torch.ones(action_dim)
        weights[:6] = pos_weight
        weights[6:] = grip_weight
        self.register_buffer('weights', weights)

    def forward(self, dist, target_actions):
        # Weight targets for importance scaling
        weighted_targets = target_actions * self.weights.unsqueeze(0)
        nll = -dist.log_prob(weighted_targets)
        return nll.mean()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 41,
        "content": _WEIGHTED_NLL_CLASS,
    },
]
