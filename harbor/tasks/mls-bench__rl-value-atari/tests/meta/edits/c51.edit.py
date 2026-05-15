"""C51 (Categorical DQN) baseline -- rigorous codebase edit ops.

Distributional RL with categorical projection and cross-entropy loss.
Uses NatureDQNEncoder (fixed) + distributional head (n_actions x n_atoms).

Reference: cleanrl/cleanrl/c51_atari.py

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_value_atari.py"

_C51_CODE = """\
class QNetwork(nn.Module):
    \"\"\"C51 distributional Q-network: NatureDQNEncoder (fixed) + distributional head.\"\"\"

    def __init__(self, envs, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.n_atoms = n_atoms
        self.n = envs.single_action_space.n
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.encoder = NatureDQNEncoder()
        self.head = nn.Linear(ENCODER_FEATURE_DIM, self.n * n_atoms)

    def forward(self, x):
        \"\"\"Return Q-values (expected values under the learned distribution).\"\"\"
        features = self.encoder(x)
        logits = self.head(features)
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        return q_values

    def get_action(self, x, action=None):
        \"\"\"Return (action, pmf_for_action). If action is None, use greedy.\"\"\"
        features = self.encoder(x)
        logits = self.head(features)
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]


class ValueAlgorithm:
    \"\"\"C51 -- Categorical Distributional DQN.\"\"\"

    def __init__(self, envs, device, args):
        self.device = device
        self.gamma = args.gamma
        # CleanRL's c51_atari.py uses a slower target refresh and larger
        # learning rate than DQN. Keeping the template/DQN values makes C51
        # systematically underperform on harder Atari games such as Seaquest.
        self.target_network_frequency = 10000
        self.n_atoms = 51
        self.v_min = -10.0
        self.v_max = 10.0

        self.q_network = QNetwork(envs, n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max).to(device)
        self.target_network = QNetwork(envs, n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=2.5e-4, eps=0.01 / args.batch_size)

    def select_action(self, obs, epsilon):
        \"\"\"Greedy action selection using distributional Q-values.\"\"\"
        action, _ = self.q_network.get_action(torch.Tensor(obs).to(self.device))
        return action.cpu().numpy()

    def update(self, batch, global_step):
        \"\"\"C51 distributional update: categorical projection + cross-entropy loss.\"\"\"
        with torch.no_grad():
            _, next_pmfs = self.target_network.get_action(batch.next_observations)
            next_atoms = batch.rewards + self.gamma * self.target_network.atoms * (1 - batch.dones)
            # Projection
            delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
            tz = next_atoms.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / delta_z
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)
            # Handle case where b is exactly an integer
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.q_network.get_action(batch.observations, batch.actions.flatten())
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard target update
        if global_step % self.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        old_val = (old_pmfs * self.q_network.atoms).sum(1)
        return {"td_loss": loss.item(), "q_values": old_val.mean().item()}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 186,
        "end_line": 249,
        "content": _C51_CODE,
    },
]
