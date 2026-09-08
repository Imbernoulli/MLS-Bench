"""QR-DQN (Quantile Regression DQN) baseline -- rigorous codebase edit ops.

Distributional RL: learns quantile values of the return distribution.
Uses NatureDQNEncoder (fixed) + quantile head (n_actions x n_quantiles).
Unlike C51, does not require v_min/v_max -- uses quantile Huber loss instead.
Uses n_quantiles=200 for Atari envs (as in the original paper).

Reference: Dabney et al., 2018 "Distributional Reinforcement Learning with
Quantile Regression"

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_value_atari.py"

_QRDQN_CODE = """\
class QNetwork(nn.Module):
    \"\"\"QR-DQN quantile Q-network: NatureDQNEncoder (fixed) + quantile head.\"\"\"

    def __init__(self, envs, n_quantiles=200):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n = envs.single_action_space.n
        self.encoder = NatureDQNEncoder()
        self.head = nn.Linear(ENCODER_FEATURE_DIM, self.n * n_quantiles)

    def forward(self, x):
        \"\"\"Return Q-values as mean of quantile values per action.\"\"\"
        features = self.encoder(x)
        quantiles = self.head(features).view(len(x), self.n, self.n_quantiles)
        q_values = quantiles.mean(dim=2)
        return q_values

    def get_quantiles(self, x):
        \"\"\"Return raw quantile values: [batch, n_actions, n_quantiles].\"\"\"
        features = self.encoder(x)
        return self.head(features).view(len(x), self.n, self.n_quantiles)


class ValueAlgorithm:
    \"\"\"QR-DQN -- Quantile Regression DQN with distributional value learning.\"\"\"

    def __init__(self, envs, device, args):
        self.device = device
        self.gamma = args.gamma
        self.target_network_frequency = args.target_network_frequency
        self.n_quantiles = 200
        self.kappa = 1.0  # Huber loss threshold

        self.q_network = QNetwork(envs, n_quantiles=self.n_quantiles).to(device)
        self.target_network = QNetwork(envs, n_quantiles=self.n_quantiles).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)

        # Fixed quantile midpoints: tau_i = (2i - 1) / (2N) for i = 1, ..., N
        self.tau = torch.arange(1, self.n_quantiles + 1, dtype=torch.float32, device=device)
        self.tau = (2 * self.tau - 1) / (2 * self.n_quantiles)

    def select_action(self, obs, epsilon):
        \"\"\"Greedy action selection using mean of quantile values.\"\"\"
        q_values = self.q_network(torch.Tensor(obs).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def update(self, batch, global_step):
        \"\"\"QR-DQN update: quantile Huber loss.\"\"\"
        with torch.no_grad():
            # Get quantile values for next state from target network
            next_quantiles = self.target_network.get_quantiles(batch.next_observations)  # [batch, n_actions, N]
            next_q = next_quantiles.mean(dim=2)  # [batch, n_actions]
            next_actions = next_q.argmax(dim=1)  # [batch]
            # Select quantiles for best actions
            next_quantiles_best = next_quantiles[torch.arange(len(batch.next_observations)), next_actions]  # [batch, N]
            # Compute target quantile values
            target_quantiles = batch.rewards + self.gamma * next_quantiles_best * (1 - batch.dones)

        # Get current quantile values for taken actions
        current_quantiles_all = self.q_network.get_quantiles(batch.observations)  # [batch, n_actions, N]
        current_quantiles = current_quantiles_all[torch.arange(len(batch.observations)), batch.actions.flatten()]  # [batch, N]

        # Quantile Huber loss
        # Pairwise TD errors: [batch, N (pred), N (target)]
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        # Huber loss element-wise
        abs_td = td_errors.abs()
        huber = torch.where(abs_td <= self.kappa,
                            0.5 * td_errors ** 2,
                            self.kappa * (abs_td - 0.5 * self.kappa))

        # Asymmetric weighting by quantile level
        tau = self.tau.view(1, -1, 1)
        quantile_weights = torch.abs(tau - (td_errors < 0).float())
        loss = (quantile_weights * huber / self.kappa).sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard target update
        if global_step % self.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        q_values = current_quantiles.mean(dim=1)
        return {"td_loss": loss.item(), "q_values": q_values.mean().item()}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 186,
        "end_line": 249,
        "content": _QRDQN_CODE,
    },
]
