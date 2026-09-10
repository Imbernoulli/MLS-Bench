"""BC (Behavioral Cloning) baseline — rigorous codebase edit ops.

Instead of learning a reward and then training a policy, BC directly
clones the expert's behavior by supervised learning on (state, action) pairs.
The "reward" is the negative BC loss so the PPO loop still runs but is
secondary to the BC pretraining done in the update step.

Reference: Pomerleau, 1991. "Efficient Training of Artificial Neural Networks for Autonomous Navigation."

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "imitation/custom_irl.py"

_BC_CLASSES = """\
class RewardNetwork(nn.Module):
    \"\"\"Dummy reward network for BC (not used for reward shaping).

    BC does not learn a reward; this returns a constant so the PPO
    loop runs but does not meaningfully update from reward signal.
    The policy is trained via supervised loss in IRLAlgorithm.update().
    \"\"\"

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Unused parameters to keep interface consistent
        self.dummy = nn.Linear(1, 1)

    def forward(self, state, action, next_state):
        return torch.zeros(state.shape[0], device=state.device)


class IRLAlgorithm:
    \"\"\"BC — Behavioral Cloning.

    Directly trains the policy network to mimic expert actions via
    supervised MSE loss. The reward network is unused.
    Policy is trained both via BC loss in update() and via PPO in the
    main loop (with near-zero reward), but BC dominates learning.
    \"\"\"

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.total_updates = 0
        # BC does not need a reward optimizer; policy is trained externally
        # We store a reference to policy that gets set during training
        self._policy = None
        self._policy_optimizer = None

    def set_policy(self, policy, optimizer):
        \"\"\"Set reference to policy for BC updates.\"\"\"
        self._policy = policy
        self._policy_optimizer = optimizer

    def compute_reward(self, obs, acts, next_obs):
        \"\"\"BC uses constant reward (PPO loop is secondary).\"\"\"
        return torch.zeros(obs.shape[0], device=self.device)

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        \"\"\"BC supervised update: minimize negative log-probability of expert actions.

        Uses the full policy distribution (mean + log_std) to compute log prob,
        matching the reference imitation library approach. This trains both the
        mean and the variance of the policy, giving better action coverage.
        \"\"\"
        self.total_updates += 1

        if self._policy is None:
            return {"irl_loss": 0.0}

        batch_size = self.args.irl_batch_size

        # Sample expert data
        n_expert = len(self.expert_demos["obs"])

        total_bc_loss = 0.0
        n_bc_steps = 20  # more BC gradient steps per IRL update

        for _ in range(n_bc_steps):
            expert_idx = torch.randint(0, n_expert, (batch_size,))
            expert_obs = self.expert_demos["obs"][expert_idx]
            expert_acts = self.expert_demos["acts"][expert_idx]

            # Use get_action_and_value to get log_prob of expert actions
            # This trains both actor_mean and actor_logstd
            _, log_prob, entropy, _ = self._policy.get_action_and_value(
                expert_obs, expert_acts,
            )

            # Negative log-likelihood loss (matching reference BC)
            neglogp = -log_prob.mean()
            # Entropy bonus for exploration (prevents policy from collapsing)
            ent_bonus = -0.001 * entropy.mean()

            bc_loss = neglogp + ent_bonus

            self._policy_optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
            self._policy_optimizer.step()
            total_bc_loss += bc_loss.item()

        return {"irl_loss": total_bc_loss / n_bc_steps}
"""

# Replace RewardNetwork + IRLAlgorithm (lines 183-309)
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 231,
        "end_line": 357,
        "content": _BC_CLASSES,
    },
]
