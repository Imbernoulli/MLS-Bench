"""Cal-QL baseline — rigorous codebase edit ops for offline-to-online.

Reference: CORL/algorithms/finetune/cal_ql.py
Cal-QL is CQL + calibration using Monte Carlo return-to-go.
Key: torch.maximum(cql_q_values, mc_returns) prevents CQL from pushing Q-values
below the actual return, which stabilizes offline-to-online transition.

Hyperparameters matching CORL YAML configs (pen/hammer cloned_v1.yaml):
  cql_alpha=1.0, cql_alpha_online=1.0, policy_lr=1e-4, qf_lr=3e-4,
  bc_steps=0, cql_n_actions=10, cql_importance_sample=True,
  alpha_multiplier=1.0, soft_target_update_rate=5e-3, q_n_hidden_layers=3,
  cql_max_target_backup=True, cql_clip_diff_min=-200,
  orthogonal_init=True, mixing_ratio=0.5, normalize=False
Note: CORL reference uses 256-wide networks (same as template constraint).
  All hyperparameters match the reference YAML configs exactly.

Actor: TanhGaussianPolicy (3x256, TransformedDistribution, log_std_multiplier/offset)
       Orthogonal init via init_module_weights (module.apply style).
Critic: FullyConnectedQFunction (3 hidden layers per config, multi-action support)
        Orthogonal init via init_module_weights (module.apply style).

MC returns are precomputed in __init__ from replay buffer data.
Buffer's sample() is monkey-patched to return mc_returns as 7th element.
Calibration is toggled off at online transition (switch_calibration).
During online phase, 50/50 mixing of offline and online data.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CORL/algorithms/finetune/custom_finetune.py"

# ── 1. Replace OfflineOnlineAlgorithm (lines 345-453) ────────────────────────

_CALQL_ALGORITHM = """\
class OfflineOnlineAlgorithm:
    \"\"\"Cal-QL — Calibrated Q-Learning for offline-to-online RL.\"\"\"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        replay_buffer=None,
        discount: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        device: str = "cuda",
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        # CQL/Cal-QL hyperparameters (matching CORL YAML configs exactly)
        self.cql_n_actions = 10
        self.cql_temp = 1.0
        self.cql_alpha = 1.0
        self.cql_alpha_online = 1.0
        self.cql_importance_sample = True
        self.cql_max_target_backup = True
        self.cql_clip_diff_min = -200.0
        self.cql_clip_diff_max = float('inf')
        self.target_entropy = -float(action_dim)
        self.alpha_multiplier = 1.0
        self.use_automatic_entropy_tuning = True
        self.backup_entropy = False
        self.bc_steps = 0
        self.policy_lr = 1e-4
        self.qf_lr = critic_lr
        self.mixing_ratio = 0.5
        self._calibration_enabled = True
        self._offline_size = 0

        # Precompute MC returns from offline data and monkey-patch buffer
        self._replay_buffer = replay_buffer
        if replay_buffer is not None:
            self._offline_size = replay_buffer._size
            self._setup_mc_returns(replay_buffer, discount)

        # Actor (TanhGaussianPolicy with log_std multiplier/offset, orthogonal init)
        self.actor = Actor(state_dim, action_dim, max_action, orthogonal_init=True).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.policy_lr)

        # Twin critics (3 hidden layers per config, orthogonal init) + targets
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init=True).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init=True).to(device)
        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.qf_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.qf_lr)

        # Entropy tuning
        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled

    def _setup_mc_returns(self, buf, discount):
        \"\"\"Compute MC returns for offline data and monkey-patch buffer.\"\"\"
        n = buf._size
        rewards = buf._rewards[:n].squeeze(-1).cpu().numpy()
        dones = buf._dones[:n].squeeze(-1).cpu().numpy()
        states = buf._states[:n].cpu().numpy()
        next_states = buf._next_states[:n].cpu().numpy()

        # Detect max_episode_steps from data (Adroit=200, fallback=1000)
        _ep_lens = []
        _el = 0
        for t in range(n):
            _el += 1
            if dones[t] or t == n - 1 or (t < n - 1 and np.linalg.norm(states[t + 1] - next_states[t]) > 1e-6):
                _ep_lens.append(_el)
                _el = 0
        _max_ep_steps = max(_ep_lens) if _ep_lens else 200

        mc_returns = np.zeros(n, dtype=np.float32)
        ep_start = 0
        ep_len = 0
        cur_rewards = []
        terminals = []
        for t in range(n):
            cur_rewards.append(float(rewards[t]))
            terminals.append(float(dones[t]))
            ep_len += 1
            is_last_step = (
                (t == n - 1)
                or (t < n - 1 and np.linalg.norm(states[t + 1] - next_states[t]) > 1e-6)
                or ep_len == _max_ep_steps
            )
            if dones[t] or is_last_step:
                prev_return = 0.0
                for i in reversed(range(ep_len)):
                    cur_rewards[i] = cur_rewards[i] + discount * prev_return * (1 - terminals[i])
                    prev_return = cur_rewards[i]
                mc_returns[ep_start:ep_start + ep_len] = cur_rewards
                ep_start = t + 1
                ep_len = 0
                cur_rewards = []
                terminals = []

        # Store mc_returns in buffer
        buf._mc_returns = torch.zeros(
            (buf._buffer_size, 1), dtype=torch.float32, device=buf._device
        )
        buf._mc_returns[:n] = torch.tensor(
            mc_returns, dtype=torch.float32, device=buf._device
        ).unsqueeze(-1)

        # Monkey-patch sample to return 6 elements with offline/online mixing
        offline_size = n
        mixing_ratio = self.mixing_ratio
        def _sample_with_mc(batch_size, is_online=False):
            if is_online and buf._size > offline_size:
                # mixing_ratio portion from offline, rest from online
                n_offline = int(batch_size * mixing_ratio)
                n_online = batch_size - n_offline
                off_idx = np.random.randint(0, offline_size, size=n_offline)
                on_idx = np.random.randint(offline_size, buf._size, size=n_online)
                indices = np.concatenate([off_idx, on_idx])
            else:
                indices = np.random.randint(0, buf._size, size=batch_size)
            return [
                buf._states[indices],
                buf._actions[indices],
                buf._rewards[indices],
                buf._next_states[indices],
                buf._dones[indices],
                buf._next_actions[indices],
                buf._mc_returns[indices],
            ]
        buf.sample = _sample_with_mc

        # Monkey-patch add_transition to set mc_returns=0 for online entries
        _orig_add = buf.add_transition
        def _add_with_mc(state, action, reward, next_state, done):
            idx = buf._pointer
            _orig_add(state, action, reward, next_state, done)
            buf._mc_returns[idx] = 0.0
        buf.add_transition = _add_with_mc

    def _alpha_and_alpha_loss(self, observations, log_pi):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = torch.clamp(
                self.log_alpha().exp() * self.alpha_multiplier,
                min=1e-6, max=100.0,
            )
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(self, observations, actions, new_actions, alpha, log_pi):
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(self, observations, actions, next_observations, rewards, dones,
                mc_returns, alpha, log_dict):
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(
            1, cql_q1_current_actions.shape[1]
        )

        # Cal-QL: bound Q-values with MC return-to-go
        if self._calibration_enabled:
            cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds)
            cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds)
            cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds)
            cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds)

        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
        else:
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand,
                    torch.unsqueeze(q1_predicted, 1),
                    cql_q1_next_actions,
                    cql_q1_current_actions,
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand,
                    torch.unsqueeze(q2_predicted, 1),
                    cql_q2_next_actions,
                    cql_q2_current_actions,
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        # Subtract the log likelihood of data
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
        cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(dict(
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
            cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
        ))

        return qf_loss

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1

        # Self-sample with mixing when online (ignore provided batch)
        if is_online and self._replay_buffer is not None:
            batch = self._replay_buffer.sample(256, is_online=True)
            batch = [b.to(self.device) for b in batch]

        if len(batch) == 7:
            observations, actions, rewards, next_observations, dones, _next_act, mc_returns = batch
        elif len(batch) == 6:
            observations, actions, rewards, next_observations, dones, _next_act = batch
            mc_returns = torch.zeros_like(rewards)
        else:
            observations, actions, rewards, next_observations, dones = batch
            mc_returns = torch.zeros_like(rewards)

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        # Policy loss
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        # Q function loss
        qf_loss = self._q_loss(
            observations, actions, next_observations, rewards, dones,
            mc_returns, alpha, log_dict,
        )
        log_dict["critic_loss"] = qf_loss.item()

        # Optimize — all losses computed first, then backward
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        return log_dict

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, self.device)

    def on_online_start(self):
        # Disable Cal-QL calibration at offline-to-online transition
        self.switch_calibration()
        self.cql_alpha = self.cql_alpha_online
"""

# ── 2. Replace Critic (lines 311-325) with 3-hidden-layer multi-action Q ─────

_CALQL_CRITIC = """\
class Critic(nn.Module):
    \"\"\"FullyConnectedQFunction — 3 hidden layers (q_n_hidden_layers=3), multi-action support.
    Uses init_module_weights (module.apply style) matching CORL reference.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False,
                 n_hidden_layers: int = 3):
        super().__init__()
        layers = [
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))
        self.network = nn.Sequential(*layers)
        if orthogonal_init:
            self.network.apply(lambda m: _calql_init_module_weights(m, True))
        else:
            _calql_init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
"""

# ── 3. Replace Actor (lines 265-308) with TanhGaussianPolicy ─────────────────

_CALQL_ACTOR = """\
class Actor(nn.Module):
    \"\"\"TanhGaussianPolicy with learnable log_std multiplier/offset and orthogonal init.
    Uses init_module_weights (module.apply style) matching CORL reference.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 log_std_multiplier: float = 1.0, log_std_offset: float = -1.0,
                 orthogonal_init: bool = False):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )
        if orthogonal_init:
            self.base_network.apply(lambda m: _calql_init_module_weights(m, True))
        else:
            _calql_init_module_weights(self.base_network[-1], False)
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def _get_dist(self, observations, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        output = self.base_network(observations)
        mean, log_std = torch.split(output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return TransformedDistribution(
            Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
        ), mean

    def forward(self, observations: torch.Tensor, deterministic: bool = False,
                repeat: int = None):
        dist, mean = self._get_dist(observations, repeat)
        action = torch.tanh(mean) if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return self.max_action * action, log_prob

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        dist = TransformedDistribution(
            Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
        )
        # Score the provided actions (clamp to valid tanh range)
        scaled = torch.clamp(actions / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
        log_probs = dist.log_prob(scaled).sum(-1)
        return log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()
"""

# ── 4. Insert helpers at top of editable region (after line 242) ─────────────

_CALQL_HELPERS = """\
def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

def _calql_init_module_weights(module, orthogonal_init=False):
    \"\"\"Per-module init matching CORL reference: orthogonal sqrt(2) for all layers
    when orthogonal_init=True, xavier 1e-2 for output layer when False.\"\"\"
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    def forward(self) -> nn.Parameter:
        return self.constant

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 361,
        "end_line": 477,
        "content": _CALQL_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 327,
        "end_line": 342,
        "content": _CALQL_CRITIC,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 281,
        "end_line": 324,
        "content": _CALQL_ACTOR,
    },
    {
        "op": "insert",
        "file": _FILE,
        "after_line": 258,
        "content": _CALQL_HELPERS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 258,
        "end_line": 258,
        "content": 'CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": False}\n',
    },
]
