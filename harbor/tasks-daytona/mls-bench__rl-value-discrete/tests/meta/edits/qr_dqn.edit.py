"""QR-DQN (Quantile Regression DQN) baseline -- rigorous codebase edit ops.

Distributional RL: learns quantile values of the return distribution.
Uses MLPEncoder (fixed) + quantile head outputting n_actions x n_quantiles.
Unlike C51, does not require v_min/v_max -- uses quantile Huber loss instead.
Uses n_quantiles=50 for classic control envs.

Reference: Dabney et al., 2018 "Distributional Reinforcement Learning with
Quantile Regression"

Changes from template:
  1. Replace QNetwork with quantile head on top of MLPEncoder
  2. Replace ValueAlgorithm with QR-DQN distributional update

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_value_discrete.py"

_QRDQN_CODE = """\
class QNetwork(nn.Module):
    \"\"\"Quantile Q-network for QR-DQN: MLPEncoder (fixed) + n_actions x n_quantiles head.\"\"\"

    def __init__(self, obs_dim, n_actions, n_quantiles=50):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.encoder = MLPEncoder(obs_dim)
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions * n_quantiles)

    def forward(self, obs):
        \"\"\"Return Q-values as mean of quantile values per action.\"\"\"
        features = self.encoder(obs)
        quantiles = self.head(features).view(len(obs), self.n_actions, self.n_quantiles)
        q_values = quantiles.mean(dim=2)
        return q_values

    def get_quantiles(self, obs):
        \"\"\"Return raw quantile values: [batch, n_actions, n_quantiles].\"\"\"
        features = self.encoder(obs)
        return self.head(features).view(len(obs), self.n_actions, self.n_quantiles)


class ValueAlgorithm:
    \"\"\"QR-DQN -- Quantile Regression DQN with distributional value learning.\"\"\"

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.n_quantiles = 50
        self.kappa = 1.0  # Huber loss threshold
        self.total_it = 0

        self.q_network = QNetwork(obs_dim, n_actions, self.n_quantiles).to(device)
        self.target_network = QNetwork(obs_dim, n_actions, self.n_quantiles).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

        # Fixed quantile midpoints: tau_i = (2i - 1) / (2N) for i = 1, ..., N
        self.tau = torch.arange(1, self.n_quantiles + 1, dtype=torch.float32, device=device)
        self.tau = (2 * self.tau - 1) / (2 * self.n_quantiles)

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
            # Get quantile values for next state from target network
            next_quantiles = self.target_network.get_quantiles(next_obs)  # [batch, n_actions, n_quantiles]
            next_q = next_quantiles.mean(dim=2)  # [batch, n_actions]
            next_actions = next_q.argmax(dim=1)  # [batch]
            # Select quantiles for best actions
            next_quantiles_best = next_quantiles[torch.arange(len(next_obs)), next_actions]  # [batch, n_quantiles]
            # Compute target quantile values
            target_quantiles = rewards.unsqueeze(1) + self.gamma * next_quantiles_best * (1 - dones.unsqueeze(1))

        # Get current quantile values for taken actions
        current_quantiles = self.q_network.get_quantiles(obs)  # [batch, n_actions, n_quantiles]
        current_quantiles = current_quantiles[torch.arange(len(obs)), actions]  # [batch, n_quantiles]

        # Quantile Huber loss
        # current_quantiles: [batch, n_quantiles] (predictions at each quantile)
        # target_quantiles:  [batch, n_quantiles] (targets)
        # Pairwise TD errors: [batch, n_quantiles (pred), n_quantiles (target)]
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)  # [batch, N, N]

        # Huber loss element-wise
        abs_td = td_errors.abs()
        huber = torch.where(abs_td <= self.kappa,
                            0.5 * td_errors ** 2,
                            self.kappa * (abs_td - 0.5 * self.kappa))

        # Asymmetric weighting by quantile level
        # tau shape: [N] -> [1, N, 1] for broadcasting
        tau = self.tau.view(1, -1, 1)
        quantile_weights = torch.abs(tau - (td_errors < 0).float())
        loss = (quantile_weights * huber / self.kappa).sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        q_values = current_quantiles.mean(dim=1)
        return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 174,
        "end_line": 242,
        "content": _QRDQN_CODE,
    },
]
