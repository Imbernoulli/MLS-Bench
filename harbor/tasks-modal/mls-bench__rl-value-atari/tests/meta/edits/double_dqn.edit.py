"""DoubleDQN (Double Deep Q-Network) baseline -- rigorous codebase edit ops.

Double Q-learning for Atari: online network selects action, target network evaluates.
Uses NatureDQNEncoder (fixed) + linear head (same as DQN). Reduces overestimation bias.

Reference: cleanrl/cleanrl/dqn_atari.py (DQN) + Double Q-learning from
    cleanrl/cleanrl/rainbow_atari.py (rainbow uses double Q internally)

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_value_atari.py"

_DOUBLE_DQN_CODE = """\
class ValueAlgorithm:
    \"\"\"DoubleDQN -- Double Deep Q-Network with hard target updates.\"\"\"

    def __init__(self, envs, device, args):
        self.device = device
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_network_frequency = args.target_network_frequency

        self.q_network = QNetwork(envs).to(device)
        self.target_network = QNetwork(envs).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        \"\"\"Epsilon-greedy action selection.\"\"\"
        q_values = self.q_network(torch.Tensor(obs).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def update(self, batch, global_step):
        \"\"\"DoubleDQN update: online net selects action, target net evaluates.\"\"\"
        with torch.no_grad():
            # Double Q-learning: online net selects best action, target net evaluates it
            best_actions = self.q_network(batch.next_observations).argmax(dim=1, keepdim=True)
            target_q = self.target_network(batch.next_observations).gather(1, best_actions).squeeze()
            td_target = batch.rewards.flatten() + self.gamma * target_q * (1 - batch.dones.flatten())

        old_val = self.q_network(batch.observations).gather(1, batch.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard target update
        if global_step % self.target_network_frequency == 0:
            for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(
                    self.tau * q_param.data + (1.0 - self.tau) * target_param.data
                )

        return {"td_loss": loss.item(), "q_values": old_val.mean().item()}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 204,
        "end_line": 249,
        "content": _DOUBLE_DQN_CODE,
    },
]
