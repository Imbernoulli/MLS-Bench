"""DDPG (Deep Deterministic Policy Gradient) baseline — rigorous codebase edit ops.

Single Q-network, deterministic actor, soft target updates.
Simplest off-policy actor-critic baseline.

Reference: cleanrl/cleanrl/ddpg_continuous_action.py

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_offpolicy_continuous.py"

_DDPG_ALGORITHM = """\
class OffPolicyAlgorithm:
    \"\"\"DDPG — Deep Deterministic Policy Gradient.\"\"\"

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.exploration_noise = args.exploration_noise
        self.policy_frequency = args.policy_frequency
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.qf1 = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=args.learning_rate)

    def select_action(self, obs):
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.exploration_noise, size=action.shape)
        return np.clip(action + noise, -self.max_action, self.max_action)

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            target_q = self.qf1_target(next_obs, next_actions).view(-1)
            td_target = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.qf1(obs, actions).view(-1)
        critic_loss = F.mse_loss(current_q, td_target)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        actor_loss_val = 0.0
        if self.total_it % self.policy_frequency == 0:
            actor_loss = -self.qf1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.qf1_target, self.qf1, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 194,
        "end_line": 244,
        "content": _DDPG_ALGORITHM,
    },
]
