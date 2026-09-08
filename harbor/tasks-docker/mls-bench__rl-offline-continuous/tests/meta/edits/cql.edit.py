"""CQL (Conservative Q-Learning) baseline — rigorous codebase edit ops.

Changes from template:
  1. Replace OfflineAlgorithm with CQL implementation (bottom-most)
  2. Insert helpers (extend_and_repeat, Scalar) at top of editable region

The template Actor (Tanh-Gaussian) and Critic are reused.  Multi-action Q
evaluation for the CQL penalty is handled by reshaping in train().

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CORL/algorithms/offline/custom.py"

# ── 1. Replace OfflineAlgorithm (lines 272-344) ──────────────────────────────

_CQL_ALGORITHM = """\
class OfflineAlgorithm:
    \"\"\"CQL — Conservative Q-Learning with SAC-style entropy tuning.\"\"\"

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

        # CQL hyperparameters
        self.cql_n_actions = 10
        self.cql_temp = 1.0
        self.cql_alpha = 10.0
        self.cql_importance_sample = True
        self.target_entropy = -float(action_dim)
        self.alpha_multiplier = 1.0
        self.bc_steps = 0
        self.policy_lr = 3e-5

        # Actor (stochastic, Tanh-Gaussian from template)
        self.actor = Actor(state_dim, action_dim, max_action, orthogonal_init).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.policy_lr)

        # Twin critics + targets
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # Automatic entropy tuning
        self.log_alpha = Scalar(0.0)
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=self.policy_lr)

    def _q_forward(self, critic, observations, actions):
        \"\"\"Evaluate Q(s, a) handling both 2D and 3D action tensors.\"\"\"
        if actions.ndim == 3 and observations.ndim == 2:
            B, N, A = actions.shape
            obs_exp = observations.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            act_exp = actions.reshape(B * N, A)
            return critic(obs_exp, act_exp).reshape(B, N)
        return critic(observations, actions)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones, *_ = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)

        # ── Policy forward ────────────────────────────────────────────
        new_actions, log_pi = self.actor(observations)

        # ── Alpha (entropy temperature) ───────────────────────────────
        alpha_loss = -(
            self.log_alpha() * (log_pi + self.target_entropy).detach()
        ).mean()
        alpha = self.log_alpha().exp() * self.alpha_multiplier

        # ── Policy loss ───────────────────────────────────────────────
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new).mean()

        log_dict: Dict[str, float] = {
            "policy_loss": policy_loss.item(),
            "alpha": alpha.item(),
        }

        # ── Q function loss (TD + CQL penalty) ───────────────────────
        q1_pred = self.critic_1(observations, actions)
        q2_pred = self.critic_2(observations, actions)

        # Target Q
        with torch.no_grad():
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )
        td_target = rewards + (1.0 - dones) * self.discount * target_q.detach()
        qf1_loss = F.mse_loss(q1_pred, td_target)
        qf2_loss = F.mse_loss(q2_pred, td_target)

        # CQL conservative penalty
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]

        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)

        # Sample multiple actions from current policy for current and next states
        obs_rep = extend_and_repeat(observations, 1, self.cql_n_actions)
        next_obs_rep = extend_and_repeat(next_observations, 1, self.cql_n_actions)
        cql_current_actions, cql_current_log_pis = self.actor(obs_rep)
        cql_next_actions, cql_next_log_pis = self.actor(next_obs_rep)
        cql_current_actions = cql_current_actions.detach()
        cql_current_log_pis = cql_current_log_pis.detach()
        cql_next_actions = cql_next_actions.detach()
        cql_next_log_pis = cql_next_log_pis.detach()

        # Evaluate Q on all action sets
        cql_q1_rand = self._q_forward(self.critic_1, observations, cql_random_actions)
        cql_q2_rand = self._q_forward(self.critic_2, observations, cql_random_actions)
        cql_q1_current = self._q_forward(self.critic_1, observations, cql_current_actions)
        cql_q2_current = self._q_forward(self.critic_2, observations, cql_current_actions)
        cql_q1_next = self._q_forward(self.critic_1, observations, cql_next_actions)
        cql_q2_next = self._q_forward(self.critic_2, observations, cql_next_actions)

        # Importance sampling
        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cql_cat_q1 = torch.cat([
                cql_q1_rand - random_density,
                cql_q1_next - cql_next_log_pis.detach(),
                cql_q1_current - cql_current_log_pis.detach(),
            ], dim=1)
            cql_cat_q2 = torch.cat([
                cql_q2_rand - random_density,
                cql_q2_next - cql_next_log_pis.detach(),
                cql_q2_current - cql_current_log_pis.detach(),
            ], dim=1)
        else:
            cql_cat_q1 = torch.cat([
                cql_q1_rand, q1_pred.unsqueeze(1),
                cql_q1_next, cql_q1_current,
            ], dim=1)
            cql_cat_q2 = torch.cat([
                cql_q2_rand, q2_pred.unsqueeze(1),
                cql_q2_next, cql_q2_current,
            ], dim=1)

        cql_qf1_ood = (
            torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        )
        cql_qf2_ood = (
            torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp
        )

        cql_qf1_diff = (cql_qf1_ood - q1_pred).mean()
        cql_qf2_diff = (cql_qf2_ood - q2_pred).mean()
        cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
        cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss
        log_dict["critic_loss"] = qf_loss.item()

        # ── Optimization ──────────────────────────────────────────────
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Target network update
        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        return log_dict
"""

# ── 2. Insert helpers at top of editable region (after line 169) ─────────────

_CQL_HELPERS = """\
def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    def forward(self) -> nn.Parameter:
        return self.constant

"""

_CQL_ACTOR = """\
class Actor(nn.Module):
    \"\"\"TanhGaussianPolicy for CQL — learnable log_std scaling via Scalar modules.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 orthogonal_init: bool = False):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )
        init_module_weights(self.net)
        self.log_std_multiplier = Scalar(1.0)
        self.log_std_offset = Scalar(-1.0)
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def _get_dist(self, state: torch.Tensor):
        out = self.net(state)
        mean, log_std = torch.split(out, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return TransformedDistribution(
            Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
        ), mean

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        dist, mean = self._get_dist(state)
        action = torch.tanh(mean) if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return self.max_action * action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        \"\"\"Log-probability of a dataset action under the current policy.\"\"\"
        dist, _ = self._get_dist(state)
        action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
        return dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = \"cpu\") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 306,
        "end_line": 397,
        "content": _CQL_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 226,
        "end_line": 270,
        "content": _CQL_ACTOR,
    },
    {
        "op": "insert",
        "file": _FILE,
        "after_line": 203,
        "content": _CQL_HELPERS,
    },
]
