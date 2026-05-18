"""ICM baseline for rl-intrinsic-exploration."""

_FILE = "cleanrl/cleanrl/custom_intrinsic_exploration.py"

_CONTENT = '''\
class IntrinsicBonusModule(nn.Module):
    """Intrinsic Curiosity Module baseline."""

    def __init__(self, action_dim: int, device: torch.device, args: Args):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args
        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        self.reward_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(args.int_gamma)

        feature_output = 7 * 7 * 64
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 256)),
            nn.ReLU(),
        )
        self.inverse_model = nn.Sequential(
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.forward_model = nn.Sequential(
            layer_init(nn.Linear(256 + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
        )

    def initialize(self, envs) -> None:
        bootstrap = []
        total_steps = self.args.num_steps * self.args.num_iterations_obs_norm_init
        for _ in range(total_steps):
            random_actions = np.random.randint(0, envs.single_action_space.n, size=(self.args.num_envs,))
            sampled_obs, _, _, _ = envs.step(random_actions)
            bootstrap.append(sampled_obs[:, 3:4, :, :])
            if len(bootstrap) >= self.args.num_steps:
                stacked = np.concatenate(bootstrap, axis=0)
                self.obs_rms.update(stacked)
                bootstrap.clear()

    def trainable_parameters(self):
        return list(self.parameters())

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.obs_rms.mean).to(self.device)
        var = torch.from_numpy(self.obs_rms.var).to(self.device)
        return ((last_frame(obs) - mean) / torch.sqrt(var)).clip(-5, 5).float()

    def _one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions.long(), num_classes=self.action_dim).float()

    def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
        self.obs_rms.update(last_frame(batch_next_obs).cpu().numpy())

    def compute_bonus(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        obs_feat = self.encoder(self._normalize_obs(obs))
        next_feat = self.encoder(self._normalize_obs(next_obs))
        pred_next_feat = self.forward_model(torch.cat([obs_feat, self._one_hot(actions)], dim=1))
        return 0.5 * (pred_next_feat - next_feat).pow(2).mean(dim=1).detach()

    def normalize_rollout_rewards(self, rollout_intrinsic: torch.Tensor) -> torch.Tensor:
        discounted = np.stack(
            [self.discounted_reward.update(reward_per_step) for reward_per_step in rollout_intrinsic.cpu().numpy()],
            axis=0,
        )
        flat_discounted = discounted.reshape(-1)
        self.reward_rms.update_from_moments(
            float(flat_discounted.mean()),
            float(flat_discounted.var()),
            int(flat_discounted.size),
        )
        return rollout_intrinsic / float(np.sqrt(self.reward_rms.var + 1e-8))

    def loss(
        self,
        batch_obs: torch.Tensor,
        batch_next_obs: torch.Tensor,
        batch_actions: torch.Tensor,
    ) -> torch.Tensor:
        obs_feat = self.encoder(self._normalize_obs(batch_obs))
        next_feat = self.encoder(self._normalize_obs(batch_next_obs))
        pred_next_feat = self.forward_model(torch.cat([obs_feat, self._one_hot(batch_actions)], dim=1))
        pred_action = self.inverse_model(torch.cat([obs_feat, next_feat], dim=1))
        inverse_loss = F.cross_entropy(pred_action, batch_actions.long())
        forward_loss = 0.5 * (pred_next_feat - next_feat.detach()).pow(2).mean()
        return inverse_loss + 0.2 * forward_loss


def mix_advantages(ext_advantages: torch.Tensor, int_advantages: torch.Tensor, args: Args) -> torch.Tensor:
    return args.ext_coef * ext_advantages + args.int_coef * int_advantages
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 179,
        "end_line": 219,
        "content": _CONTENT,
    },
]
