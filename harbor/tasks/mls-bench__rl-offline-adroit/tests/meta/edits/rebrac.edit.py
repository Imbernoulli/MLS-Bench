"""ReBRAC (Regularized BRAC) baseline — rigorous codebase edit ops.

Reference: CORL/algorithms/offline/rebrac.py (JAX/Flax)
Key config: actor_bc_coef=1.0, critic_bc_coef=1.0, LR=1e-3, 3 hidden layers,
    actor_ln=False (no LayerNorm), critic_ln=True (post-activation LayerNorm),
    normalize_q=True, policy_noise=0.2, noise_clip=0.5, policy_freq=2

Critic BC: penalizes (next_actions_policy - next_actions_data)^2 in Bellman target
Actor BC: penalizes (pi - actions)^2 with sum (not mean) over action dims

Uses next_actions (action at next timestep) from the batch (6th element),
precomputed in ReplayBuffer.load_d4rl_dataset.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CORL/algorithms/offline/custom_adroit.py"

# ── 1. Replace OfflineAlgorithm (lines 272-363) ──────────────────────────────

_REBRAC_ALGORITHM = """\
class OfflineAlgorithm:
    \"\"\"ReBRAC — TD3+BC with critic BC regularization in Bellman target.\"\"\"

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

        # ReBRAC hyperparameters (per-env from CORL reference configs)
        env_name = os.environ.get("ENV", "")
        if "hammer" in env_name:
            self.actor_bc_coef = 0.01
            self.critic_bc_coef = 0.5
        elif "door-cloned" in env_name:
            self.actor_bc_coef = 0.01
            self.critic_bc_coef = 0.1
        elif "door" in env_name:
            self.actor_bc_coef = 0.1
            self.critic_bc_coef = 0.1
        else:  # pen (default)
            self.actor_bc_coef = 0.1
            self.critic_bc_coef = 0.5
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.normalize_q = True

        # Actor (deterministic, 3x256, NO LayerNorm)
        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # Twin critics (3x256, WITH post-activation LayerNorm) + targets
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions_data = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        log_dict: Dict[str, float] = {}

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions_policy = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            # Critic BC: subtract penalty from next_q
            bc_penalty = ((next_actions_policy - next_actions_data) ** 2).sum(-1)
            target_q1 = self.critic_1_target(next_states, next_actions_policy)
            target_q2 = self.critic_2_target(next_states, next_actions_policy)
            next_q = torch.min(target_q1, target_q2)
            next_q = next_q - self.critic_bc_coef * bc_penalty
            target_q = rewards + (1.0 - dones) * self.discount * next_q

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
            bc_penalty_actor = ((pi - actions) ** 2).sum(-1)
            q_values = torch.min(
                self.critic_1(states, pi),
                self.critic_2(states, pi),
            )

            lmbda = 1.0
            if self.normalize_q:
                lmbda = 1.0 / q_values.abs().mean().detach()

            actor_loss = (self.actor_bc_coef * bc_penalty_actor - lmbda * q_values).mean()
            log_dict["actor_loss"] = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict
"""

# ── 2. Replace Critic (lines 238-252) with 3x256 post-activation LayerNorm ───

_REBRAC_CRITIC = """\
class Critic(nn.Module):
    \"\"\"Q-function with post-activation LayerNorm (ReBRAC critic_ln=True). 3x256 MLP.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )
        # CORL-style init: pytorch_init for hidden, uniform_init(3e-3) for output
        import math
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                fan_in = layer.in_features
                if i < len(self.net) - 1:  # hidden layers
                    bound = math.sqrt(1.0 / fan_in)
                    nn.init.uniform_(layer.weight, -bound, bound)
                    nn.init.constant_(layer.bias, 0.1)
                else:  # output layer
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                    nn.init.uniform_(layer.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
"""

# ── 3. Replace DeterministicActor (lines 170-189) with 3x256 NO LayerNorm ────

_REBRAC_ACTOR = """\
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        # CORL-style init: pytorch_init for hidden, uniform_init(1e-3) for output
        import math
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                fan_in = layer.in_features
                if i < len(self.net) - 2:  # hidden layers
                    bound = math.sqrt(1.0 / fan_in)
                    nn.init.uniform_(layer.weight, -bound, bound)
                    nn.init.constant_(layer.bias, 0.1)
                else:  # output layer
                    nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                    nn.init.uniform_(layer.bias, -1e-3, 1e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 319,
        "end_line": 416,
        "content": _REBRAC_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 285,
        "end_line": 301,
        "content": _REBRAC_CRITIC,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 221,
        "end_line": 237,
        "content": _REBRAC_ACTOR,
    },
]
