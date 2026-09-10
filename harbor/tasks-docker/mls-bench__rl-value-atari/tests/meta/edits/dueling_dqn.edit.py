"""DuelingDQN (Dueling Deep Q-Network) baseline -- rigorous codebase edit ops.

Dueling architecture for Atari: NatureDQNEncoder (fixed) + separate value and
advantage heads. Q(s,a) = V(s) + A(s,a) - mean(A(s,a)).

Reference: Wang et al. 2016 "Dueling Network Architectures for Deep RL"
           cleanrl/cleanrl/dqn_atari.py (base DQN)

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_value_atari.py"

_DUELING_DQN_CODE = """\
class QNetwork(nn.Module):
    \"\"\"Dueling Q-network: NatureDQNEncoder (fixed) + separate V and A heads.\"\"\"

    def __init__(self, envs):
        super().__init__()
        n_actions = envs.single_action_space.n
        self.encoder = NatureDQNEncoder()
        # Value stream
        self.value_head = nn.Linear(ENCODER_FEATURE_DIM, 1)
        # Advantage stream
        self.advantage_head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)

    def forward(self, x):
        features = self.encoder(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ValueAlgorithm:
    \"\"\"DuelingDQN -- Dueling Deep Q-Network with hard target updates.\"\"\"

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
        \"\"\"DuelingDQN update: MSE TD loss with hard target network update.\"\"\"
        with torch.no_grad():
            target_max, _ = self.target_network(batch.next_observations).max(dim=1)
            td_target = batch.rewards.flatten() + self.gamma * target_max * (1 - batch.dones.flatten())

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
        "start_line": 186,
        "end_line": 249,
        "content": _DUELING_DQN_CODE,
    },
]
