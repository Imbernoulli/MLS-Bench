"""DoubleDQN (Double Deep Q-Network) baseline -- rigorous codebase edit ops.

Double Q-learning: online network selects action, target network evaluates.
Reduces overestimation bias compared to standard DQN.
Uses the same MLPEncoder (fixed) + linear head as DQN.

Reference: cleanrl/cleanrl/dqn.py (standard DQN) + Double Q-learning from
    cleanrl/cleanrl/rainbow_atari.py (rainbow_atari uses double Q internally)

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_value_discrete.py"

_DOUBLE_DQN_ALGORITHM = """\
class ValueAlgorithm:
    \"\"\"DoubleDQN -- Double Deep Q-Network.\"\"\"

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.total_it = 0

        self.q_network = QNetwork(obs_dim, n_actions).to(device)
        self.target_network = QNetwork(obs_dim, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        q_values = self.q_network(obs_t)
        return torch.argmax(q_values, dim=1).item()

    def update(self, batch, global_step):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            # Double Q-learning: online net selects action, target net evaluates
            best_actions = self.q_network(next_obs).argmax(dim=1)
            target_q = self.target_network(next_obs).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            td_target = rewards + (1 - dones) * self.gamma * target_q

        old_val = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return {"td_loss": td_loss.item(), "q_values": old_val.mean().item()}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 191,
        "end_line": 242,
        "content": _DOUBLE_DQN_ALGORITHM,
    },
]
