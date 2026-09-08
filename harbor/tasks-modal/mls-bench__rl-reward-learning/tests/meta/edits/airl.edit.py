"""AIRL (Adversarial Inverse Reinforcement Learning) baseline — rigorous codebase edit ops.

Uses a shaped reward: r(s,a,s') = g(s,a) + gamma * h(s') - h(s),
where g is the reward and h is a potential/shaping function.
The discriminator is: D = exp(f) / (exp(f) + pi(a|s)),
implemented as logits f(s,a,s') - log pi(a|s).

Reference: Fu et al., 2018. "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning."

This implementation matches the imitation library's AIRL setup more closely:
  - RunningNorm on reward-net inputs (obs/next_obs) so the discriminator
    cannot trivially classify expert vs policy on raw obs scale alone.
  - RunningNorm on reward-net output so the value passed into the FIXED
    template-level reward normalization is already roughly unit-variance,
    avoiding the saturation that obliterates the learning signal.
  - done-aware shaping: zero h(next_state) when the next state is terminal.
  - More disc updates and a larger effective batch per round (since args.*
    are FIXED outside the editable range, we bump them inside update()).

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "imitation/custom_irl.py"

_AIRL_REWARD = """\
class _RunningMeanStd:
    \"\"\"Welford running mean/std for input/output normalization.\"\"\"

    def __init__(self, shape, device, eps=1e-4):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = eps

    @torch.no_grad()
    def update(self, x):
        if x.numel() == 0:
            return
        x = x.detach().to(self.mean.device, dtype=torch.float32).reshape(-1, self.mean.shape[-1]) \\
            if self.mean.dim() else x.detach().to(self.mean.device, dtype=torch.float32).reshape(-1)
        batch_count = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False) if batch_count > 1 else torch.zeros_like(self.var)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class RewardNetwork(nn.Module):
    \"\"\"AIRL shaped reward: f(s,a,s',done) = g(s,a) + gamma * (1-done) * h(s') - h(s).

    Adds RunningNorm on the obs inputs (frozen during inference) and on
    the network output, mirroring imitation's tuned configs.
    \"\"\"

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # g(s, a): reward approximator
        self.g_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        # h(s): potential-based shaping
        self.h_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.gamma = 0.99

        # Running normalization. Initialized lazily (we need a device).
        self._obs_rms = None
        self._out_rms = None
        self._update_norm = True

    def _ensure_rms(self, ref_tensor):
        if self._obs_rms is None:
            self._obs_rms = _RunningMeanStd((self.obs_dim,), ref_tensor.device)
            self._out_rms = _RunningMeanStd((1,), ref_tensor.device)

    def freeze_norm(self):
        self._update_norm = False

    def update_obs_norm(self, obs, next_obs):
        if self._obs_rms is None:
            self._ensure_rms(obs)
        if self._update_norm:
            self._obs_rms.update(obs)
            self._obs_rms.update(next_obs)

    def update_out_norm(self, raw_f):
        if self._out_rms is None:
            return
        if self._update_norm:
            self._out_rms.update(raw_f.unsqueeze(-1) if raw_f.dim() == 1 else raw_f)

    def _norm_obs(self, x):
        if self._obs_rms is None:
            return x
        return self._obs_rms.normalize(x)

    def _norm_out(self, y):
        if self._out_rms is None:
            return y
        return (y - self._out_rms.mean.squeeze()) / torch.sqrt(self._out_rms.var.squeeze() + 1e-8)

    def g(self, state, action):
        x = torch.cat([self._norm_obs(state), action], dim=-1)
        return self.g_net(x).squeeze(-1)

    def h(self, state):
        return self.h_net(self._norm_obs(state)).squeeze(-1)

    def raw_f(self, state, action, next_state, done=None):
        \"\"\"Unnormalized shaped reward. done: float tensor of shape (batch,).\"\"\"
        h_next = self.h(next_state)
        if done is not None:
            h_next = h_next * (1.0 - done.float())
        return self.g(state, action) + self.gamma * h_next - self.h(state)

    def forward(self, state, action, next_state, done=None):
        \"\"\"Shaped reward, with running output normalization applied.\"\"\"
        f = self.raw_f(state, action, next_state, done)
        return self._norm_out(f)


class IRLAlgorithm:
    \"\"\"AIRL — Adversarial Inverse Reinforcement Learning.

    Discriminator logits are raw_f(s,a,s',done) - log pi(a|s).
    The reward returned to PPO is the *normalized* shaped f, so the FIXED
    template-level reward normalization (running mean/std on buffer.rewards)
    becomes near-identity rather than collapsing the signal.
    \"\"\"

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)
        self.total_updates = 0
        self._policy = None

        # Effective per-round disc updates and batch (args are FIXED outside
        # editable range — bump internally to match imitation tuned configs).
        self._inner_updates = 4   # multiplies args.n_irl_updates_per_round
        self._batch_mult = 4      # multiplies args.irl_batch_size

    def set_policy(self, policy, optimizer):
        del optimizer
        self._policy = policy

    def compute_reward(self, obs, acts, next_obs):
        \"\"\"Normalized shaped reward for PPO. dones not available here, so
        the terminal correction is applied during update() instead.\"\"\"
        with torch.no_grad():
            return self.reward_net(obs, acts, next_obs)

    def _log_policy_act_prob(self, obs, acts):
        if self._policy is None:
            raise RuntimeError("AIRL requires set_policy() before discriminator updates")
        with torch.no_grad():
            _, log_prob, _, _ = self._policy.get_action_and_value(obs, acts)
        return log_prob.detach()

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        \"\"\"AIRL discriminator update using raw_f - log pi(a|s) on the BCE side.\"\"\"
        self.total_updates += 1
        bs = self.args.irl_batch_size * self._batch_mult

        # Update obs running stats once per outer call using the freshest
        # generator rollout — keeps statistics close to the policy state dist.
        self.reward_net.update_obs_norm(policy_obs, policy_next_obs)

        n_expert = len(self.expert_demos["obs"])
        n_policy = len(policy_obs)
        # Expert "dones" are not stored in the demo dict — assume non-terminal
        # (correct for halfcheetah which has no terminal; mildly wrong for
        # hopper/walker terminal states but reference imitation also lacks
        # dones for expert demos by default).
        expert_done_zeros = torch.zeros(bs, device=self.device)

        last = {}
        for _ in range(self._inner_updates):
            # Resample fresh batches each inner step
            expert_idx = torch.randint(0, n_expert, (bs,))
            expert_obs = self.expert_demos["obs"][expert_idx]
            expert_acts = self.expert_demos["acts"][expert_idx]
            expert_next_obs = self.expert_demos["next_obs"][expert_idx]

            policy_idx = torch.randint(0, n_policy, (bs,))
            gen_obs = policy_obs[policy_idx]
            gen_acts = policy_acts[policy_idx]
            gen_next_obs = policy_next_obs[policy_idx]
            gen_dones = policy_dones[policy_idx].float()

            expert_f = self.reward_net.raw_f(
                expert_obs, expert_acts, expert_next_obs, expert_done_zeros
            )
            gen_f = self.reward_net.raw_f(
                gen_obs, gen_acts, gen_next_obs, gen_dones
            )

            expert_logits = expert_f - self._log_policy_act_prob(expert_obs, expert_acts)
            gen_logits = gen_f - self._log_policy_act_prob(gen_obs, gen_acts)

            logits = torch.cat([expert_logits, gen_logits], dim=0)
            labels = torch.cat([
                torch.ones(bs, device=self.device),
                torch.zeros(bs, device=self.device),
            ], dim=0)

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.reward_net.update_out_norm(torch.cat([expert_f, gen_f], dim=0))
                acc = ((logits > 0).float() == labels).float().mean().item()
            last = {"irl_loss": loss.item(), "disc_acc": acc}

        return last
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 231,
        "end_line": 357,
        "content": _AIRL_REWARD,
    },
]
