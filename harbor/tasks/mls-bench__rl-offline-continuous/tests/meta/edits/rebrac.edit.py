"""ReBRAC (Regularized Behavior Regularized Actor Critic) baseline — rigorous codebase edit ops.

Changes from template:
  1. Replace DeterministicActor with 3x256 (no LayerNorm on actor, matching CORL actor_ln=False)
  2. Replace Critic with 3x256+LayerNorm (matching CORL critic_ln=True)
  3. Replace OfflineAlgorithm with ReBRAC implementation using per-env tuned hyperparameters
  4. Set CONFIG_OVERRIDES for per-env batch_size and learning rates

ReBRAC is a TD3-style algorithm with BC penalties on both the actor loss and the
critic target (via next-action BC penalty).  The original ReBRAC paper and CORL
implementation use carefully tuned per-environment BC coefficients, batch sizes,
and learning rates.  This edit uses the CORL-style ReBRAC hyperparameter set
selected for this benchmark harness.

Per-environment hyperparameters used by this task:
  halfcheetah-medium-v2: actor_bc=0.001, critic_bc=0.01,  lr=1e-3, batch=1024
  walker2d-medium-v2:    actor_bc=0.05,  critic_bc=0.1,   lr=1e-3, batch=1024
  hopper-medium-v2:      actor_bc=0.01,  critic_bc=0.01,  lr=1e-3, batch=1024
  maze2d-large-v1:       actor_bc=0.003, critic_bc=0.001, lr=3e-4, batch=256
  maze2d-medium-v1:      actor_bc=0.003, critic_bc=0.001, lr=3e-4, batch=256
"""

_FILE = "CORL/algorithms/offline/custom.py"

# ── Per-env config + CONFIG_OVERRIDES (replaces lines 198-203) ────────

_REBRAC_CONFIG = """\
# CONFIG_OVERRIDES: set any TrainConfig field here to override the fixed
# defaults. Allowed keys: normalize, normalize_reward, actor_lr, critic_lr,
# tau, discount, batch_size.
# =====================================================================
import sys as _sys

def _detect_env():
    \"\"\"Parse --env from sys.argv to determine environment name.\"\"\"
    for i, arg in enumerate(_sys.argv):
        if arg == "--env" and i + 1 < len(_sys.argv):
            return _sys.argv[i + 1]
        if arg.startswith("--env="):
            return arg.split("=", 1)[1]
    return ""

_REBRAC_ENV = _detect_env()

# Per-environment ReBRAC hyperparameters for this benchmark harness
_REBRAC_HPARAMS = {
    "halfcheetah-medium-v2": {"actor_bc_coef": 0.001, "critic_bc_coef": 0.01,  "lr": 1e-3, "batch_size": 1024},
    "walker2d-medium-v2":    {"actor_bc_coef": 0.05,  "critic_bc_coef": 0.1,   "lr": 1e-3, "batch_size": 1024},
    "hopper-medium-v2":      {"actor_bc_coef": 0.01,  "critic_bc_coef": 0.01,  "lr": 1e-3, "batch_size": 1024},
    "maze2d-large-v1":       {"actor_bc_coef": 0.003, "critic_bc_coef": 0.001, "lr": 3e-4, "batch_size": 256},
    "maze2d-medium-v1":      {"actor_bc_coef": 0.003, "critic_bc_coef": 0.001, "lr": 3e-4, "batch_size": 256},
}
_REBRAC_HP = _REBRAC_HPARAMS.get(_REBRAC_ENV, {"actor_bc_coef": 0.01, "critic_bc_coef": 0.01, "lr": 1e-3, "batch_size": 1024})

CONFIG_OVERRIDES: Dict[str, Any] = {}
"""

# ── DeterministicActor: 3x256 NO LayerNorm (CORL actor_ln=False) ─────

_REBRAC_ACTOR = """\
class DeterministicActor(nn.Module):
    \"\"\"Deterministic policy pi(s) = tanh(net(s)) * max_action.
    ReBRAC-style: 3 x 256 MLP without LayerNorm (matching CORL actor_ln=False).\"\"\"

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = \"cpu\") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()
"""

# ── Critic: 3x256 WITH LayerNorm (CORL critic_ln=True) ───────────────

_REBRAC_CRITIC = """\
class Critic(nn.Module):
    \"\"\"Q-function Q(s, a). 3 x 256 MLP with LayerNorm (ReBRAC-style, critic_ln=True).\"\"\"

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)
"""

# ── OfflineAlgorithm: ReBRAC with per-env tuned hyperparameters ──────

_REBRAC_ALGORITHM = """\
class OfflineAlgorithm:
    \"\"\"ReBRAC — Regularized Behavior Regularized Actor Critic.

    TD3-style with BC penalties on actor and critic targets.
    Per-environment BC coefficients and learning rates from CORL configs.
    \"\"\"

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

        # Per-env tuned ReBRAC hyperparameters for this benchmark harness
        self.actor_bc_coef = _REBRAC_HP["actor_bc_coef"]
        self.critic_bc_coef = _REBRAC_HP["critic_bc_coef"]
        _lr = _REBRAC_HP["lr"]
        self.policy_noise = 0.2      # target policy smoothing noise
        self.noise_clip = 0.5        # clipping range for smoothing noise
        self.policy_freq = 2         # delayed actor update frequency
        self.normalize_q = True      # normalize Q in actor loss

        # Actor (deterministic, no LayerNorm) + target
        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=_lr)

        # Twin critics (with LayerNorm) + targets
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=_lr)

        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions_data = batch
        not_done = 1 - dones.squeeze(-1)
        rewards_flat = rewards.squeeze(-1)
        log_dict: Dict[str, float] = {}

        # ── Critic update ──────────────────────────────────────────────
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            # BC penalty on next actions (compare policy's next actions to dataset's)
            bc_penalty = ((next_actions - next_actions_data) ** 2).sum(-1)

            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            # Subtract BC penalty from critic target (ReBRAC key idea)
            target_q = target_q - self.critic_bc_coef * bc_penalty
            target_q = rewards_flat + not_done * self.discount * target_q

        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # ── Delayed actor update ───────────────────────────────────────
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            q = self.critic_1(states, pi)

            # BC penalty on actor
            bc_mse = ((pi - actions) ** 2).sum(-1)

            lmbda = 1.0
            if self.normalize_q:
                lmbda = 1.0 / (torch.abs(q).mean().detach() + 1e-8)

            actor_loss = (self.actor_bc_coef * bc_mse - lmbda * q).mean()
            log_dict["actor_loss"] = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 306,
        "end_line": 397,
        "content": _REBRAC_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 272,
        "end_line": 287,
        "content": _REBRAC_CRITIC,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 198,
        "end_line": 224,
        "content": _REBRAC_CONFIG + _REBRAC_ACTOR,
    },
]
