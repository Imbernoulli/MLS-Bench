"""SAC (Soft Actor-Critic) baseline — rigorous codebase edit ops.

Stochastic actor with tanh squashing, twin Q-networks, auto entropy tuning.

Reference: cleanrl/cleanrl/sac_continuous_action.py

Changes from template:
  1. Replace OffPolicyAlgorithm with SAC (bottom-most)
  2. Replace Actor with stochastic Tanh-Gaussian actor

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_offpolicy_continuous.py"

# -- 1. Replace OffPolicyAlgorithm (lines 215-267) -------------------------

_SAC_ALGORITHM = """\
class OffPolicyAlgorithm:
    \"\"\"SAC — Soft Actor-Critic with automatic entropy tuning.\"\"\"

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_frequency = args.policy_frequency
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)

        self.qf1 = QNetwork(obs_dim, action_dim).to(device)
        self.qf2 = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target = QNetwork(obs_dim, action_dim).to(device)
        self.qf2_target = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

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

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        # Update critics
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(next_obs)
            q1_next = self.qf1_target(next_obs, next_actions).view(-1)
            q2_next = self.qf2_target(next_obs, next_actions).view(-1)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi.view(-1)
            td_target = rewards + (1 - dones) * self.gamma * min_q_next

        q1 = self.qf1(obs, actions).view(-1)
        q2 = self.qf2(obs, actions).view(-1)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # Update actor
        actor_loss_val = 0.0
        if self.total_it % self.policy_frequency == 0:
            pi, log_pi, _ = self.actor.get_action(obs)
            q1_pi = self.qf1(obs, pi).view(-1)
            q2_pi = self.qf2(obs, pi).view(-1)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_pi.view(-1) - min_q_pi).mean()

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

        # Update target networks
        soft_update(self.qf1_target, self.qf1, self.tau)
        soft_update(self.qf2_target, self.qf2, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val, "alpha": self.alpha}
"""

# -- 2. Replace Actor (lines 174-197) with stochastic Tanh-Gaussian --------

_SAC_ACTOR = """\
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    \"\"\"Stochastic Tanh-Gaussian actor for SAC.\"\"\"

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
        "content": _SAC_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 153,
        "end_line": 176,
        "content": _SAC_ACTOR,
    },
]
