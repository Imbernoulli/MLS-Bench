"""DuelingDQN (Dueling Deep Q-Network) baseline -- rigorous codebase edit ops.

Dueling architecture: MLPEncoder (fixed) + separate value and advantage heads.
Q(s,a) = V(s) + A(s,a) - mean(A). Same loss as DQN but better value estimation.

Reference: Wang et al. 2016 "Dueling Network Architectures for Deep RL"
           cleanrl/cleanrl/dqn.py (base DQN)

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_value_discrete.py"

_DUELING_DQN_CODE = """\
class QNetwork(nn.Module):
    \"\"\"Dueling Q-network: MLPEncoder (fixed) + separate value and advantage heads.\"\"\"

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim)
        # Value stream
        self.value_head = nn.Linear(ENCODER_FEATURE_DIM, 1)
        # Advantage stream
        self.advantage_head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)

    def forward(self, obs):
        features = self.encoder(obs)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ValueAlgorithm:
    \"\"\"DuelingDQN -- Dueling Deep Q-Network.\"\"\"

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
            target_max, _ = self.target_network(next_obs).max(dim=1)
            td_target = rewards + (1 - dones) * self.gamma * target_max

        old_val = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        td_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {"td_loss": td_loss.item(), "q_values": old_val.mean().item()}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 174,
        "end_line": 242,
        "content": _DUELING_DQN_CODE,
    },
]
