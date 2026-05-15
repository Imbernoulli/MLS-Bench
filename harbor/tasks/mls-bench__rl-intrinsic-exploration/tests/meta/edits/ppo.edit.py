"""Vanilla PPO baseline for rl-intrinsic-exploration."""

_FILE = "cleanrl/cleanrl/custom_intrinsic_exploration.py"

_CONTENT = '''\
class IntrinsicBonusModule(nn.Module):
    """Baseline: no intrinsic reward."""

    def __init__(self, action_dim: int, device: torch.device, args: Args):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args

    def initialize(self, envs) -> None:
        return None

    def trainable_parameters(self):
        return []

    def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
        return None

    def compute_bonus(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(obs.shape[0], device=self.device)

    def normalize_rollout_rewards(self, rollout_intrinsic: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(rollout_intrinsic)

    def loss(
        self,
        batch_obs: torch.Tensor,
        batch_next_obs: torch.Tensor,
        batch_actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros((), device=self.device)


def mix_advantages(ext_advantages: torch.Tensor, int_advantages: torch.Tensor, args: Args) -> torch.Tensor:
    return args.ext_coef * ext_advantages
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 179,
        "end_line": 219,
        "content": _CONTENT,
    },
]
