"""SAC-N (SAC with N critics) baseline — rigorous codebase edit ops.

Changes from template:
  1. Replace OfflineAlgorithm with SAC-N implementation (bottom-most)
  2. Replace Actor with EDAC-style (separate mu/log_sigma heads, same as SAC-N reference)
  3. Insert helpers (VectorizedLinear, VectorizedCritic) at top of editable region

SAC-N is a simpler variant of EDAC without the diversity penalty.  It uses an
ensemble of N critics and takes the minimum Q across all N for the target
(instead of the standard 2 critics in vanilla SAC).

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CORL/algorithms/offline/custom.py"

# ── 1. Replace OfflineAlgorithm (lines 272-357) ──────────────────────────────

_SACN_ALGORITHM = """\
class OfflineAlgorithm:
    \"\"\"SAC-N — Soft Actor-Critic with N-critic ensemble (min over all N).\"\"\"

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

        # SAC-N hyperparameters
        self.num_critics = 10
        self.target_entropy = -float(action_dim)

        # Actor (EDAC/SAC-N style stochastic policy)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Vectorized critic ensemble + target
        self.critic = VectorizedCritic(
            state_dim, action_dim, 256, self.num_critics
        ).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        # Adaptive entropy coefficient
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        state, action, reward, next_state, done, *_ = batch

        # ── Alpha update ───────────────────────────────────────────────
        with torch.no_grad():
            _, action_log_prob = self.actor(state, need_log_prob=True)
        alpha_loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ── Actor update ───────────────────────────────────────────────
        pi_action, pi_log_prob = self.actor(state, need_log_prob=True)
        q_pi = self.critic(state, pi_action)
        q_pi_min = q_pi.min(0).values
        actor_loss = (self.alpha * pi_log_prob - q_pi_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Critic update ──────────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state, need_log_prob=True)
            q_next = self.target_critic(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_log_prob
            q_target = reward.squeeze(-1) + (1 - done.squeeze(-1)) * self.discount * q_next

        q_values = self.critic(state, action)
        # [ensemble_size, batch_size] - [1, batch_size]
        critic_loss = ((q_values - q_target.unsqueeze(0)) ** 2).mean(dim=1).sum(dim=0)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Target update ──────────────────────────────────────────────
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, self.tau)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha.item(),
        }
"""

# ── 2. Replace Actor (lines 192-235) with SAC-N style ────────────────────────

_SACN_ACTOR = """\
class Actor(nn.Module):
    \"\"\"SAC-N stochastic policy — separate mu/log_sigma heads with EDAC-style init.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 orthogonal_init: bool = False):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_sigma = nn.Linear(256, action_dim)

        # EDAC-style initialization
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)
        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

    def forward(self, state: torch.Tensor, deterministic: bool = False,
                need_log_prob: bool = False):
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)
        log_sigma = torch.clip(log_sigma, -5, 2)
        dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = dist.rsample()

        tanh_action = torch.tanh(action)
        log_prob = None
        if need_log_prob:
            log_prob = dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0]
        return action.cpu().data.numpy().flatten()
"""

# ── 3. Insert helpers at top of editable region (after line 169) ─────────────

_SACN_HELPERS = """\
import math

class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias

class VectorizedCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)
        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)
        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        q_values = self.critic(state_action).squeeze(-1)
        return q_values

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 306,
        "end_line": 397,
        "content": _SACN_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 226,
        "end_line": 270,
        "content": _SACN_ACTOR,
    },
    {
        "op": "insert",
        "file": _FILE,
        "after_line": 203,
        "content": _SACN_HELPERS,
    },
]
