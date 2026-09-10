"""IQL (Implicit Q-Learning) baseline — rigorous codebase edit ops.

Changes from template:
  1. Replace OfflineAlgorithm with IQL implementation (bottom-most)
  2. Replace Actor with GaussianPolicy-style (state-independent log_std, returns Normal)
  3. Insert helpers (CosineAnnealingLR import, EXP_ADV_MAX, asymmetric_l2_loss) at top

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CORL/algorithms/offline/custom.py"

# ── 1. Replace OfflineAlgorithm (lines 272-344) ──────────────────────────────

_IQL_ALGORITHM = """\
class OfflineAlgorithm:
    \"\"\"IQL — Implicit Q-Learning with expectile regression and advantage-weighted actor.\"\"\"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        replay_buffer=None,
        discount: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        orthogonal_init: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        # IQL hyperparameters
        self.beta = 3.0
        self.iql_tau = 0.7

        # Actor (GaussianPolicy-style via replaced Actor class)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(1e6))

        # Twin Q via two separate Critic instances + targets
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1).requires_grad_(False).to(device)
        self.critic_2_target = deepcopy(self.critic_2).requires_grad_(False).to(device)
        self.q_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=critic_lr,
        )

        # Value function V(s)
        self.vf = ValueFunction(state_dim, orthogonal_init).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)

    def _update_v(self, observations, actions, log_dict):
        with torch.no_grad():
            target_q = torch.min(
                self.critic_1_target(observations, actions),
                self.critic_2_target(observations, actions),
            )
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(self, next_v, observations, actions, rewards, dones, log_dict):
        targets = rewards + (1.0 - dones.float()) * self.discount * next_v.detach()
        q1 = self.critic_1(observations, actions)
        q2 = self.critic_2(observations, actions)
        q_loss = (F.mse_loss(q1, targets) + F.mse_loss(q2, targets)) / 2.0
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones, *_ = batch
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)

        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict
"""

# ── 2. Replace Actor (lines 192-235) with GaussianPolicy-style ───────────────

_IQL_ACTOR = """\
class Actor(nn.Module):
    \"\"\"GaussianPolicy for IQL — state-independent log_std, forward returns Normal.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 orthogonal_init: bool = False):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        # 2-hidden-layer MLP with Tanh output (matching IQL reference)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def forward(self, state: torch.Tensor) -> Normal:
        mean = self.net(state)
        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()
"""

# ── 3. Insert helpers at top of editable region (after line 169) ─────────────

_IQL_HELPERS = """\
from torch.optim.lr_scheduler import CosineAnnealingLR

EXP_ADV_MAX = 100.0

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

"""

_IQL_CRITIC = """\
class Critic(nn.Module):
    \"\"\"Q-function Q(s, a). 2 × 256 MLP (IQL reference architecture).\"\"\"

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
"""

_IQL_VALUEFUNC = """\
class ValueFunction(nn.Module):
    \"\"\"State value function V(s). 2 × 256 MLP (IQL reference architecture).\"\"\"

    def __init__(self, state_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 306,
        "end_line": 397,
        "content": _IQL_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 289,
        "end_line": 304,
        "content": _IQL_VALUEFUNC,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 272,
        "end_line": 287,
        "content": _IQL_CRITIC,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 226,
        "end_line": 270,
        "content": _IQL_ACTOR,
    },
    {
        "op": "insert",
        "file": _FILE,
        "after_line": 203,
        "content": _IQL_HELPERS,
    },
]
