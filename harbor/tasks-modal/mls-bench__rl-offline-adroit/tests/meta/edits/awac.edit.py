"""AWAC (Advantage Weighted Actor-Critic) baseline — rigorous codebase edit ops.

Reference: CORL/algorithms/offline/awac.py
Key differences from template defaults:
  - Actor: 3x256 MLP, state-independent log_std (nn.Parameter), min_log_std=-20.0,
    Normal distribution with clamp (NOT TanhTransform)
  - Critic: 3x256 MLP, output NOT squeezed (returns (batch, 1))
  - Algorithm: SEPARATE critic optimizers, advantage-weighted actor loss

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CORL/algorithms/offline/custom_adroit.py"

_AWAC_CONFIG = """\
CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": True}
"""

# ── 1. Replace OfflineAlgorithm (lines 272-363) ──────────────────────────────

_AWAC_ALGORITHM = """\
class OfflineAlgorithm:
    \"\"\"AWAC — Advantage Weighted Actor-Critic for offline RL.\"\"\"

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

        # AWAC hyperparameters (match CORL reference: awac_lambda=0.1)
        self.awac_lambda = 0.1
        self.exp_adv_max = 100.0

        # Actor (GaussianPolicy-style with state-independent log_std)
        self.actor = Actor(state_dim, action_dim, 256).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critics + targets (SEPARATE optimizers)
        self.critic_1 = Critic(state_dim, action_dim, 256).to(device)
        self.critic_2 = Critic(state_dim, action_dim, 256).to(device)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        log_dict: Dict[str, float] = {}

        # Critic update
        with torch.no_grad():
            next_actions, _ = self.actor(next_states)
            q_next = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self.discount * (1.0 - dones) * q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Actor update (advantage-weighted)
        with torch.no_grad():
            pi_action, _ = self.actor(states)
            v = torch.min(
                self.critic_1(states, pi_action),
                self.critic_2(states, pi_action),
            )
            q = torch.min(
                self.critic_1(states, actions),
                self.critic_2(states, actions),
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self.awac_lambda), self.exp_adv_max
            )

        action_log_prob = self.actor.log_prob(states, actions)
        actor_loss = (-action_log_prob * weights).mean()
        log_dict["actor_loss"] = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target update
        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        return log_dict
"""

# ── 2. Replace Critic (lines 238-252) with non-squeezing version ─────────────

_AWAC_CRITIC = """\
class Critic(nn.Module):
    \"\"\"Q-function Q(s, a). 3x256 MLP, returns (batch, 1).\"\"\"

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._mlp(torch.cat([state, action], dim=-1))
"""

# ── 3. Replace Actor (lines 192-235) with reference-style AWAC Actor ─────────

_AWAC_ACTOR = """\
class Actor(nn.Module):
    \"\"\"AWAC GaussianPolicy — 3x256 MLP, state-independent log_std, Normal + clamp.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 min_log_std: float = -20.0, max_log_std: float = 2.0):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std

    def _get_policy(self, state: torch.Tensor):
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        return torch.distributions.Normal(mean, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        return policy.log_prob(action).sum(-1, keepdim=True)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(-1.0, 1.0)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        return action_t[0].cpu().numpy()
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 319,
        "end_line": 416,
        "content": _AWAC_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 285,
        "end_line": 301,
        "content": _AWAC_CRITIC,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 239,
        "end_line": 284,
        "content": _AWAC_ACTOR,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 214,
        "end_line": 214,
        "content": _AWAC_CONFIG,
    },
]
