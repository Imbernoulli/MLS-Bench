"""TD3+BC (TD3 with Behavior Cloning regularization) baseline — rigorous codebase edit ops.

Reference: CORL/algorithms/offline/td3_bc.py
Key differences from template defaults:
  - Actor: deterministic, 2x256 MLP with Tanh output (same as DeterministicActor)
  - Critic: 2x256 MLP, returns (batch, 1) NOT squeezed (for proper Q normalization)
  - Algorithm: TD3 with BC regularization in actor loss
    actor_loss = -lambda * Q(s, pi(s)) + MSE(pi(s), a)
    lambda = alpha / mean(|Q|)    (alpha=2.5 default)
  - Twin critics, delayed policy updates (policy_freq=2)
  - Target policy smoothing noise

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CORL/algorithms/offline/custom_adroit.py"

# ── 1. Replace OfflineAlgorithm (lines 272-363) ──────────────────────────────

_TD3BC_ALGORITHM = """\
class OfflineAlgorithm:
    \"\"\"TD3+BC — TD3 with Behavior Cloning regularization for offline RL.\"\"\"

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

        # TD3+BC hyperparameters (match CORL reference)
        self.alpha = 2.5
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

        # Actor (deterministic)
        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critics + targets
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        log_dict: Dict[str, float] = {}

        # Critic update
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.discount * target_q

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            q = self.critic_1(states, pi)
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, actions)
            log_dict["actor_loss"] = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict
"""

# ── 2. Replace Critic (lines 238-252) with 2x256 MLP returning (batch, 1) ───

_TD3BC_CRITIC = """\
class Critic(nn.Module):
    \"\"\"Q-function Q(s, a). 2x256 MLP, returns (batch, 1) for TD3+BC Q-normalization.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 319,
        "end_line": 416,
        "content": _TD3BC_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 285,
        "end_line": 301,
        "content": _TD3BC_CRITIC,
    },
]
