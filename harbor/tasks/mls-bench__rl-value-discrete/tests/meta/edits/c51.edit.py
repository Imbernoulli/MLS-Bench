"""C51 (Categorical DQN) baseline -- rigorous codebase edit ops.

Distributional RL: learns full return distribution rather than mean Q-value.
Uses MLPEncoder (fixed) + distributional head outputting n_actions x n_atoms.
Uses n_atoms=51, v_min=-500, v_max=500 (wide enough for classic control envs:
CartPole returns ~0-500, LunarLander ~-400 to +300, Acrobot ~-500 to 0).

Reference: cleanrl/cleanrl/c51.py

Changes from template:
  1. Replace QNetwork with distributional head on top of MLPEncoder
  2. Replace ValueAlgorithm with C51 distributional update

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_value_discrete.py"

_C51_CODE = """\
class QNetwork(nn.Module):
    \"\"\"Distributional Q-network for C51: MLPEncoder (fixed) + n_actions x n_atoms head.\"\"\"

    def __init__(self, obs_dim, n_actions, n_atoms=51, v_min=-500, v_max=500):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.encoder = MLPEncoder(obs_dim)
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions * n_atoms)

    def forward(self, obs):
        features = self.encoder(obs)
        logits = self.head(features)
        pmfs = torch.softmax(logits.view(len(obs), self.n_actions, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        return q_values

    def get_action(self, obs, action=None):
        features = self.encoder(obs)
        logits = self.head(features)
        pmfs = torch.softmax(logits.view(len(obs), self.n_actions, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(obs)), action]


class ValueAlgorithm:
    \"\"\"C51 -- Categorical DQN with distributional value learning.\"\"\"

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.n_atoms = 51
        self.v_min = -500.0
        self.v_max = 500.0
        self.total_it = 0

        self.q_network = QNetwork(obs_dim, n_actions, self.n_atoms, self.v_min, self.v_max).to(device)
        self.target_network = QNetwork(obs_dim, n_actions, self.n_atoms, self.v_min, self.v_max).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        action, _ = self.q_network.get_action(obs_t)
        return action.item()

    def update(self, batch, global_step):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            _, next_pmfs = self.target_network.get_action(next_obs)
            next_atoms = rewards.unsqueeze(1) + self.gamma * self.target_network.atoms * (1 - dones.unsqueeze(1))
            # Projection
            delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
            tz = next_atoms.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / delta_z
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.q_network.get_action(obs, actions)
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        q_values = (old_pmfs * self.q_network.atoms).sum(1)
        return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 174,
        "end_line": 242,
        "content": _C51_CODE,
    },
]
