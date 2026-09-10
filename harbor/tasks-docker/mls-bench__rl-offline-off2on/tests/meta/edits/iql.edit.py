"""IQL baseline — rigorous codebase edit ops for offline-to-online.

Reference: CORL/algorithms/finetune/iql.py
IQL (Implicit Q-Learning) for offline-to-online finetuning:
  - Actor: GaussianPolicy — 2x256 MLP with Tanh output activation, state-independent
    log_std (nn.Parameter), Normal distribution (NOT TanhTransform), clamp action
  - Critic: TwinQ — two 2x256 MLPs with squeeze, .both() for twin outputs
  - ValueFunction: 2x256 MLP with squeeze
  - V(s): expectile regression with asymmetric L2 loss (iql_tau=0.8)
  - Q(s,a): standard Bellman backup, averaged over twin Q, target network
  - Policy: advantage-weighted regression with temperature beta=3.0
  - CosineAnnealingLR scheduler for actor
  - IQL needs no special handling at offline→online transition

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CORL/algorithms/finetune/custom_finetune.py"

# ── 1. Replace OfflineOnlineAlgorithm (lines 303-412) ────────────────────────

_IQL_ALGORITHM = """\
class OfflineOnlineAlgorithm:
    \"\"\"IQL — Implicit Q-Learning for offline-to-online RL.\"\"\"

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
        device: str = "cuda",
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        # IQL hyperparameters (match CORL reference)
        self.iql_tau = 0.8       # expectile for asymmetric V loss (CORL hammer/pen-cloned config)
        self.beta = 3.0          # inverse temperature for advantage weighting
        self.exp_adv_max = 100.0

        # Actor (GaussianPolicy-style)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=int(1e6)
        )

        # Twin Q-network + target
        self.qf = TwinQ(state_dim, action_dim).to(device)
        self.q_target = deepcopy(self.qf).requires_grad_(False).to(device)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=critic_lr)

        # Value function
        self.vf = ValueFunction(state_dim).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        log_dict: Dict[str, float] = {}

        # V(s) update — expectile regression
        with torch.no_grad():
            target_q = self.q_target(states, actions)
        v = self.vf(states)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Q(s,a) update — standard Bellman with target V
        with torch.no_grad():
            next_v = self.vf(next_states)
        targets = rewards + (1.0 - dones) * self.discount * next_v.detach()
        qs = self.qf.both(states, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Target Q update
        soft_update(self.q_target, self.qf, self.tau)

        # Policy update — advantage-weighted regression
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.exp_adv_max)
        policy_out = self.actor(states)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        return log_dict

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, self.device)

    def on_online_start(self):
        # IQL needs no special handling at offline-to-online transition
        pass
"""

# ── 2. Replace ValueFunction (lines 286-301) with 2x256 ──────────────────────

_IQL_VALUE = """\
class ValueFunction(nn.Module):
    \"\"\"State value function V(s). 2x256 MLP, squeezed output.\"\"\"

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim] + [hidden_dim] * n_hidden + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.v = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state).squeeze(-1)
"""

# ── 3. Replace Critic (lines 269-283) with TwinQ ─────────────────────────────

_IQL_CRITIC = """\
class TwinQ(nn.Module):
    \"\"\"Twin Q-functions Q1(s,a), Q2(s,a). 2x256 MLPs, squeezed output.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + action_dim] + [hidden_dim] * n_hidden + [1]

        def _build_mlp():
            layers = []
            for i in range(len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            return nn.Sequential(*layers)

        self.q1 = _build_mlp()
        self.q2 = _build_mlp()

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))
"""

# ── 4. Replace Actor (lines 223-266) with GaussianPolicy ─────────────────────

_IQL_ACTOR = """\
class Actor(nn.Module):
    \"\"\"IQL GaussianPolicy — 2x256 MLP with Tanh output, state-independent log_std, Normal dist.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dim: int = 256, n_hidden: int = 2, dropout: float = 0.1):
        super().__init__()
        dims = [state_dim] + [hidden_dim] * n_hidden + [action_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.max_action = max_action
        self._log_std_min = -20.0
        self._log_std_max = 2.0

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(self._log_std_min, self._log_std_max))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()
"""

# ── 5. Insert asymmetric_l2_loss helper after line 200 ───────────────────────

_IQL_HELPERS = """\
def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 361,
        "end_line": 477,
        "content": _IQL_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 344,
        "end_line": 359,
        "content": _IQL_VALUE,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 327,
        "end_line": 342,
        "content": _IQL_CRITIC,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 281,
        "end_line": 324,
        "content": _IQL_ACTOR,
    },
    {
        "op": "insert",
        "file": _FILE,
        "after_line": 258,
        "content": _IQL_HELPERS,
    },
]
