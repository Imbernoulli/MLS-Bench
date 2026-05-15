"""SPOT baseline — rigorous codebase edit ops for offline-to-online.

Reference: CORL/algorithms/finetune/spot.py
SPOT uses TD3 + VAE support constraint. Key components:
  - VAE (Variational Auto-Encoder) trained on offline data for density estimation
  - Actor loss: -norm_q * Q.mean() + lambda * neg_log_beta.mean()
  - norm_q = 1 / |Q|.mean() for Q-value normalization
  - neg_log_beta = ELBO loss from VAE (support constraint)
  - Lambda cooling during online phase (if enabled)
  - Critic: 2x256, returns (batch, 1) NOT squeezed
  - Actor: 2x256 deterministic (template's DeterministicActor matches)

VAE training runs in a separate pretrain() phase before offline policy/critic
training, matching CORL's offline-to-online flow.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "CORL/algorithms/finetune/custom_finetune.py"

# ── 1. Replace OfflineOnlineAlgorithm (lines 303-412) ────────────────────────

_SPOT_ALGORITHM = """\
class OfflineOnlineAlgorithm:
    \"\"\"SPOT — Support constraint Policy Optimization via online Training (TD3 + VAE).\"\"\"

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
        self.online_discount = 0.99
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0
        self.replay_buffer = replay_buffer

        # SPOT hyperparameters
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2
        self.expl_noise = 0.1
        self.beta = 0.5
        self.lambd = 1.0
        self.lambd_cool = True
        self.lambd_end = 0.5
        self.num_samples = 1
        self.is_online = False
        self.online_it = 0
        self.max_online_steps = int(1e6)
        self.vae_iterations = 100_000
        self._actor_lr = 1e-4
        self._critic_lr = critic_lr

        # VAE for support constraint (pretrained before offline policy training)
        latent_dim = 2 * action_dim
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dim=750).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self._vae_trained = False

        # Actor (deterministic) + target — init head weights small (reference: 0.001)
        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        _actor_head = self.actor.net[-2]  # Linear before Tanh
        _actor_head.weight.data.uniform_(-0.001, 0.001)
        _actor_head.bias.data.uniform_(-0.001, 0.001)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)

        # Twin critics (2x256, unsqueezed) + targets — init head weights small (reference: 0.003)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        _c1_head = self.critic_1.net[-1]  # Last Linear (output layer)
        _c1_head.weight.data.uniform_(-0.003, 0.003)
        _c1_head.bias.data.uniform_(-0.003, 0.003)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self._critic_lr)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        _c2_head = self.critic_2.net[-1]
        _c2_head.weight.data.uniform_(-0.003, 0.003)
        _c2_head.bias.data.uniform_(-0.003, 0.003)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self._critic_lr)

    def _elbo_loss(self, state, action):
        \"\"\"ELBO loss for VAE support constraint (neg log beta).\"\"\"
        mean, std = self.vae.encode(state, action)
        N = self.num_samples
        mean_s = mean.repeat(N, 1, 1).permute(1, 0, 2)
        std_s = std.repeat(N, 1, 1).permute(1, 0, 2)
        z = mean_s + std_s * torch.randn_like(std_s)
        state_r = state.repeat(N, 1, 1).permute(1, 0, 2)
        action_r = action.repeat(N, 1, 1).permute(1, 0, 2)
        u = self.vae.decode(state_r, z)
        recon_loss = ((u - action_r) ** 2).mean(dim=(1, 2))
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        return recon_loss + self.beta * KL_loss

    def _vae_train_step(self, batch):
        \"\"\"One VAE training step.\"\"\"
        state, action, *_ = batch
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.beta * KL_loss
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        return {"vae_loss": vae_loss.item(), "vae_recon": recon_loss.item()}

    def pretrain(self, replay_buffer, batch_size: int) -> Dict[str, float]:
        \"\"\"Train the VAE before the fixed offline-to-online loop starts.\"\"\"
        print(f"Pretraining SPOT VAE for {self.vae_iterations} steps")
        log_dict: Dict[str, float] = {}
        self.vae.train()
        for t in range(self.vae_iterations):
            batch = replay_buffer.sample(batch_size)
            batch = [b.to(self.device) for b in batch]
            log_dict = self._vae_train_step(batch)
            if (t + 1) % 1000 == 0:
                metrics_str = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in log_dict.items()
                )
                print(f"TRAIN_METRICS step=vae_{t+1} {metrics_str}", flush=True)
        self.vae.eval()
        self._vae_trained = True
        return log_dict

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1
        if not self._vae_trained:
            if self.replay_buffer is not None:
                self.pretrain(self.replay_buffer, batch[0].shape[0])
            else:
                self.vae.eval()
                self._vae_trained = True

        if is_online:
            self.online_it += 1
        state, action, reward, next_state, done, *_ = batch
        not_done = 1 - done
        log_dict: Dict[str, float] = {}

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates with VAE support constraint
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic_1(state, pi)

            # VAE support constraint (neg log beta)
            neg_log_beta = self._elbo_loss(state, pi)

            # Lambda cooling
            if self.lambd_cool:
                lambd = self.lambd * max(
                    self.lambd_end, (1.0 - self.online_it / self.max_online_steps)
                )
            else:
                lambd = self.lambd

            # Q-value normalization
            norm_q = 1.0 / q.abs().mean().detach()

            actor_loss = -norm_q * q.mean() + lambd * neg_log_beta.mean()
            log_dict["actor_loss"] = actor_loss.item()
            log_dict["lambd"] = lambd

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.tensor(
                state.reshape(1, -1), device=self.device, dtype=torch.float32
            )
            action = self.actor(state_t)
            noise = (torch.randn_like(action) * self.expl_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            action = (action + noise).clamp(-self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()

    def on_online_start(self):
        self.is_online = True
        self.discount = self.online_discount
        # Reset optimizers at transition
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self._actor_lr
        )
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=self._critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=self._critic_lr
        )
"""

# ── 2. Replace Critic (lines 269-283) with 2x256 unsqueezed ──────────────────

_SPOT_CRITIC = """\
class Critic(nn.Module):
    \"\"\"Q-function Q(s, a). 2x256 MLP, returns (batch, 1).\"\"\"

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))
"""

# ── 3. Insert VAE class after line 200 ───────────────────────────────────────

_SPOT_VAE = """\
class VAE(nn.Module):
    \"\"\"Variational Auto-Encoder for SPOT support constraint.\"\"\"

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int,
                 max_action: float, hidden_dim: int = 750):
        super().__init__()
        self.encoder_shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def encode(self, state, action):
        z = self.encoder_shared(torch.cat([state, action], -1))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(self, state, z):
        return self.max_action * self.decoder(torch.cat([state, z], -1))

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 361,
        "end_line": 477,
        "content": _SPOT_ALGORITHM,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 327,
        "end_line": 342,
        "content": _SPOT_CRITIC,
    },
    {
        "op": "insert",
        "file": _FILE,
        "after_line": 258,
        "content": _SPOT_VAE,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 258,
        "end_line": 258,
        "content": 'CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": False}\n',
    },
]
