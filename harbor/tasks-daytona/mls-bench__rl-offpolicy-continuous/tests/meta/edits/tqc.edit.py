"""TQC (Truncated Quantile Critics) baseline — rigorous codebase edit ops.

Ensemble of N quantile critics with truncation for pessimistic value estimation,
combined with SAC-style stochastic actor and automatic entropy tuning.

Reference: Kuznetsov et al., "Controlling Overestimation Bias with Truncated
Mixture of Continuous Distributional Quantile Critics", ICML 2020.

Changes from template:
  1. Replace OffPolicyAlgorithm with TQC (bottom-most)
  2. Replace QNetwork with quantile critic ensemble
  3. Replace Actor with stochastic Tanh-Gaussian actor (same as SAC)

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_offpolicy_continuous.py"

# -- 1. Replace OffPolicyAlgorithm (lines 215-267) -------------------------

_TQC_ALGORITHM = """\
class OffPolicyAlgorithm:
    \"\"\"TQC — Truncated Quantile Critics with SAC-style entropy tuning.\"\"\"

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_frequency = args.policy_frequency
        self.total_it = 0

        self.n_critics = 5
        self.n_quantiles = 25
        self.top_quantiles_to_drop = 2 * self.n_quantiles  # 50 out of 125

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)

        self.critic = QuantileCriticEnsemble(obs_dim, action_dim, self.n_critics, self.n_quantiles).to(device)
        self.critic_target = QuantileCriticEnsemble(obs_dim, action_dim, self.n_critics, self.n_quantiles).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        # Auto entropy tuning (same as SAC)
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.learning_rate)

        # Precompute quantile midpoints: tau_i = (2i - 1) / (2 * n_quantiles)
        self.quantile_tau = torch.arange(1, self.n_quantiles + 1, device=device, dtype=torch.float32)
        self.quantile_tau = (2 * self.quantile_tau - 1) / (2.0 * self.n_quantiles)  # shape [n_quantiles]

    def select_action(self, obs):
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action, _, _ = self.actor.get_action(obs_t)
        return action.cpu().numpy().flatten()

    def _quantile_huber_loss(self, quantiles_pred, target):
        \"\"\"Quantile Huber loss.

        quantiles_pred: [batch, n_quantiles] from one critic
        target: [batch, n_target_quantiles] target quantile values
        \"\"\"
        # Expand for pairwise computation
        # pred: [batch, n_quantiles, 1], target: [batch, 1, n_target_quantiles]
        pred = quantiles_pred.unsqueeze(2)
        tgt = target.unsqueeze(1)
        td_error = tgt - pred  # [batch, n_quantiles, n_target_quantiles]

        # Huber loss with kappa=1
        huber = torch.where(td_error.abs() <= 1.0,
                            0.5 * td_error.pow(2),
                            td_error.abs() - 0.5)

        # Quantile weights: tau_i for underestimation, (1 - tau_i) for overestimation
        tau = self.quantile_tau.view(1, -1, 1)  # [1, n_quantiles, 1]
        quantile_loss = (tau - (td_error < 0).float()).abs() * huber
        # Mean over target quantiles, sum over pred quantiles, mean over batch
        return quantile_loss.mean(2).sum(1).mean()

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        # -- Critic update --
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(next_obs)
            # Get all target quantile values: [batch, n_critics, n_quantiles]
            all_target_quantiles = self.critic_target(next_obs, next_actions)
            # Reshape to [batch, n_critics * n_quantiles] and sort
            batch_size = obs.shape[0]
            flat_quantiles = all_target_quantiles.reshape(batch_size, -1)
            sorted_quantiles, _ = flat_quantiles.sort(dim=1)
            # Truncate: drop the top quantiles (most optimistic)
            n_total = self.n_critics * self.n_quantiles
            n_keep = n_total - self.top_quantiles_to_drop
            truncated = sorted_quantiles[:, :n_keep]  # [batch, n_keep]
            # Target = mean of remaining quantiles - entropy bonus
            target_q = truncated.mean(dim=1) - self.alpha * next_log_pi.view(-1)
            td_target = rewards + (1 - dones) * self.gamma * target_q
            # Expand target for quantile loss: [batch, 1]
            td_target_expanded = td_target.unsqueeze(1)  # [batch, 1] as "1 target quantile"

        # Get predicted quantiles: [batch, n_critics, n_quantiles]
        pred_quantiles = self.critic(obs, actions)
        critic_loss = 0.0
        for i in range(self.n_critics):
            critic_loss = critic_loss + self._quantile_huber_loss(
                pred_quantiles[:, i, :], td_target_expanded
            )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -- Actor update --
        actor_loss_val = 0.0
        if self.total_it % self.policy_frequency == 0:
            pi, log_pi, _ = self.actor.get_action(obs)
            # Mean Q across all quantiles from all critics
            all_q = self.critic(obs, pi)  # [batch, n_critics, n_quantiles]
            mean_q = all_q.mean(dim=(1, 2))  # [batch]
            actor_loss = (self.alpha * log_pi.view(-1) - mean_q).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # Update alpha
            with torch.no_grad():
                _, log_pi_alpha, _ = self.actor.get_action(obs)
            alpha_loss = (-self.log_alpha.exp() * (log_pi_alpha.view(-1) + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # -- Target network update --
        soft_update(self.critic_target, self.critic, self.tau)

        return {"critic_loss": critic_loss.item() if torch.is_tensor(critic_loss) else critic_loss,
                "actor_loss": actor_loss_val, "alpha": self.alpha}
"""

# -- 2. Replace QNetwork (lines 199-213) with quantile critic ensemble -----

_TQC_QNETWORK = """\
class QNetwork(nn.Module):
    \"\"\"Single quantile critic: Q(s, a) -> n_quantiles quantile values.\"\"\"

    def __init__(self, obs_dim, action_dim, n_quantiles=25):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_quantiles)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # [batch, n_quantiles]


class QuantileCriticEnsemble(nn.Module):
    \"\"\"Ensemble of N quantile critics for TQC.\"\"\"

    def __init__(self, obs_dim, action_dim, n_critics=5, n_quantiles=25):
        super().__init__()
        self.critics = nn.ModuleList([
            QNetwork(obs_dim, action_dim, n_quantiles)
            for _ in range(n_critics)
        ])

    def forward(self, obs, action):
        # Returns [batch, n_critics, n_quantiles]
        return torch.stack([c(obs, action) for c in self.critics], dim=1)
"""

# -- 3. Replace Actor (lines 174-197) with stochastic Tanh-Gaussian --------

_TQC_ACTOR = """\
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    \"\"\"Stochastic Tanh-Gaussian actor for TQC (same as SAC).\"\"\"

    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor(max_action, dtype=torch.float32))

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale
        return action, log_prob, mean_action
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 194,
        "end_line": 244,
        "content": _TQC_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 178,
        "end_line": 192,
        "content": _TQC_QNETWORK,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 153,
        "end_line": 176,
        "content": _TQC_ACTOR,
    },
]
