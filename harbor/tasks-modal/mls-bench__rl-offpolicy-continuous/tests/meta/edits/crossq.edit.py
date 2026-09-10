"""CrossQ (Bhatt et al., 2024) baseline — rigorous codebase edit ops.

Eliminates target networks by using Batch Renormalization across combined
(current, next) batches.  Twin Q-networks with BatchRenorm, stochastic
Tanh-Gaussian actor, automatic entropy tuning.

Key idea: concatenate (s,a) and (s',a') into one batch, pass through Q-network
with BatchRenorm so that the batch statistics computed over the joint batch
prevent value bootstrapping leakage.  No target networks or Polyak averaging
needed.

Uses BatchRenorm (Ioffe, 2017) instead of standard BatchNorm for stability
with non-stationary RL data, following the CrossQ paper's recommendation.

Changes from template:
  1. Replace OffPolicyAlgorithm with CrossQ (bottom-most)
  2. Replace QNetwork with BatchRenorm variant (middle)
  3. Replace Actor with stochastic Tanh-Gaussian actor (top-most)

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_offpolicy_continuous.py"

# -- 1. Replace OffPolicyAlgorithm (lines 215-267) -------------------------

_CROSSQ_ALGORITHM = """\
class OffPolicyAlgorithm:
    \"\"\"CrossQ — BatchRenorm-based off-policy actor-critic without target networks.\"\"\"

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.policy_frequency = 1  # CrossQ paper default: update actor every step
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)

        # Twin Q-networks with BatchRenorm — NO target networks
        self.qf1 = QNetwork(obs_dim, action_dim).to(device)
        self.qf2 = QNetwork(obs_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=args.learning_rate,
        )

        # Auto entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.learning_rate)

    def select_action(self, obs):
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action, _, _ = self.actor.get_action(obs_t)
        return action.cpu().numpy().flatten()

    def _q_forward_cross(self, qf, obs, actions, next_obs, next_actions):
        \"\"\"Forward through Q-network on concatenated (current, next) batch.

        BatchRenorm sees both sub-batches, preventing bootstrap leakage.
        Returns (q_current, q_next) after splitting.
        \"\"\"
        cat_obs = torch.cat([obs, next_obs], dim=0)
        cat_act = torch.cat([actions, next_actions], dim=0)
        q_all = qf(cat_obs, cat_act)
        q_current, q_next = torch.split(q_all, [obs.shape[0], next_obs.shape[0]], dim=0)
        return q_current, q_next

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(next_obs)

        # CrossQ: forward current and next through Q-networks together
        self.qf1.train()
        self.qf2.train()
        q1_current, q1_next = self._q_forward_cross(self.qf1, obs, actions, next_obs, next_actions)
        q2_current, q2_next = self._q_forward_cross(self.qf2, obs, actions, next_obs, next_actions)

        # Detach q_next to prevent any gradient flow and free memory
        with torch.no_grad():
            min_q_next = torch.min(q1_next.detach().view(-1), q2_next.detach().view(-1)) - self.alpha * next_log_pi.view(-1)
            td_target = rewards + (1 - dones) * self.gamma * min_q_next

        critic_loss = F.mse_loss(q1_current.view(-1), td_target) + F.mse_loss(q2_current.view(-1), td_target)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # --- Actor update ---
        actor_loss_val = 0.0
        if self.total_it % self.policy_frequency == 0:
            pi, log_pi, _ = self.actor.get_action(obs)
            self.qf1.eval()
            self.qf2.eval()
            q1_pi = self.qf1(obs, pi).view(-1)
            q2_pi = self.qf2(obs, pi).view(-1)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_pi.view(-1) - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # Update alpha (reuse log_pi from actor update, detached)
            alpha_loss = (-self.log_alpha.exp() * (log_pi.detach().view(-1) + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val, "alpha": self.alpha}
"""

# -- 2. Replace QNetwork (lines 199-213) with BatchRenorm variant ----------

_CROSSQ_QNETWORK = """\
class BatchRenorm1d(nn.Module):
    \"\"\"Batch Renormalization (Ioffe, 2017) for 1-D inputs.

    More stable than standard BatchNorm for non-stationary data (e.g. RL)
    because it clips the correction terms r and d, preventing the running
    statistics from causing large deviations during training.
    \"\"\"

    def __init__(self, num_features, momentum=0.01, eps=1e-5, rmax=3.0, dmax=5.0):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.rmax = rmax
        self.dmax = dmax
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_std = (batch_var + self.eps).sqrt()
            running_std = (self.running_var + self.eps).sqrt()

            # Renormalization corrections (clamped for stability)
            r = (batch_std / running_std).detach().clamp(1.0 / self.rmax, self.rmax)
            d = ((batch_mean - self.running_mean) / running_std).detach().clamp(-self.dmax, self.dmax)

            # Normalize using batch stats, then correct
            x_hat = (x - batch_mean) / batch_std * r + d

            # Update running statistics
            self.running_mean.mul_(1 - self.momentum).add_(batch_mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(batch_var.detach() * self.momentum)
        else:
            # Inference: use running statistics (same as standard BN)
            x_hat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()

        return self.weight * x_hat + self.bias


class QNetwork(nn.Module):
    \"\"\"Q-function Q(s, a) -> scalar with BatchRenorm (CrossQ).\"\"\"

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.bn1 = BatchRenorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = BatchRenorm1d(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
"""

# -- 3. Replace Actor (lines 174-197) with stochastic Tanh-Gaussian ------

_CROSSQ_ACTOR = """\
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    \"\"\"Stochastic Tanh-Gaussian actor for CrossQ.\"\"\"

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
        "content": _CROSSQ_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 178,
        "end_line": 192,
        "content": _CROSSQ_QNETWORK,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 153,
        "end_line": 176,
        "content": _CROSSQ_ACTOR,
    },
]
