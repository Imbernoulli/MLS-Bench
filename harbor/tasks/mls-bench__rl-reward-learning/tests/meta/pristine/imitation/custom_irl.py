# Custom IRL / Reward Learning algorithm for MLS-Bench
#
# EDITABLE section: RewardNetwork and IRLAlgorithm classes.
# FIXED sections: everything else (config, env, demo loading, PPO training, evaluation).
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =====================================================================
# FIXED: Configuration
# =====================================================================
@dataclass
class Args:
    env_id: str = "HalfCheetah-v4"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    # IRL training
    irl_epochs: int = 200
    irl_batch_size: int = 256
    irl_lr: float = 3e-4
    demo_path: str = ""  # set from env or CLI
    # Policy training (PPO via custom loop)
    total_timesteps: int = 1000000
    policy_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    n_epochs: int = 10
    minibatch_size: int = 64
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Evaluation
    eval_freq: int = 50000
    eval_episodes: int = 10
    # IRL-specific
    n_gen_steps_per_irl_update: int = 2048
    n_irl_updates_per_round: int = 5


# =====================================================================
# FIXED: Environment setup
# =====================================================================
def make_env(env_id, seed, idx=0):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk


# =====================================================================
# FIXED: Expert demonstration generation & loading
# =====================================================================
def generate_expert_demos(demo_path, env_id, total_timesteps=2_000_000, n_demos=25000):
    """Train PPO expert and collect demonstrations on GPU."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy as sb3_eval

    os.makedirs(demo_path, exist_ok=True)
    print(f"Training expert for {env_id} ({total_timesteps} steps)...", flush=True)

    train_env = SubprocVecEnv([lambda eid=env_id, i=i: gym.make(eid) for i in range(4)])
    sb3_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO("MlpPolicy", train_env, verbose=0,
                n_steps=2048, batch_size=64, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                max_grad_norm=0.5, device=sb3_device)
    model.learn(total_timesteps=total_timesteps)
    train_env.close()

    eval_env = DummyVecEnv([lambda eid=env_id: gym.make(eid)])
    mean_reward, std_reward = sb3_eval(model, eval_env, n_eval_episodes=20)
    print(f"  Expert {env_id}: {mean_reward:.1f} +/- {std_reward:.1f}", flush=True)
    model.save(os.path.join(demo_path, f"{env_id}_expert"))

    all_obs, all_acts, all_next_obs, all_dones = [], [], [], []
    obs = eval_env.reset()
    for _ in range(n_demos):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = eval_env.step(action)
        all_obs.append(obs[0].copy())
        all_acts.append(action[0].copy())
        all_next_obs.append(next_obs[0].copy())
        all_dones.append(float(done[0]))
        obs = next_obs

    demos = {
        "obs": np.array(all_obs, dtype=np.float32),
        "acts": np.array(all_acts, dtype=np.float32),
        "next_obs": np.array(all_next_obs, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
    }
    np.savez(os.path.join(demo_path, f"{env_id}_demos.npz"), **demos)
    print(f"  Saved {n_demos} transitions for {env_id}", flush=True)
    eval_env.close()


def load_expert_demos(demo_path, env_id, device):
    """Load expert demonstrations, generating them if needed."""
    path = os.path.join(demo_path, f"{env_id}_demos.npz")
    if not os.path.exists(path):
        generate_expert_demos(demo_path, env_id)
    data = np.load(path)
    demos = {
        "obs": torch.tensor(data["obs"], dtype=torch.float32, device=device),
        "acts": torch.tensor(data["acts"], dtype=torch.float32, device=device),
        "next_obs": torch.tensor(data["next_obs"], dtype=torch.float32, device=device),
        "dones": torch.tensor(data["dones"], dtype=torch.float32, device=device),
    }
    print(f"Loaded {len(demos['obs'])} expert transitions from {path}")
    return demos


# =====================================================================
# FIXED: Policy network (PPO actor-critic, not editable)
# =====================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    """PPO Actor-Critic policy. FIXED — not editable."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)


# =====================================================================
# FIXED: Rollout buffer for PPO
# =====================================================================
class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self, n_steps, obs_dim, action_dim, device):
        self.obs = torch.zeros((n_steps, obs_dim), device=device)
        self.actions = torch.zeros((n_steps, action_dim), device=device)
        self.logprobs = torch.zeros(n_steps, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.next_obs = torch.zeros((n_steps, obs_dim), device=device)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value, next_obs):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.next_obs[self.ptr] = next_obs
        self.ptr += 1

    def reset(self):
        self.ptr = 0


# =====================================================================
# FIXED: Evaluation
# =====================================================================
@torch.no_grad()
def evaluate_policy(env_id, policy, device, n_episodes, seed):
    """Evaluate policy for n_episodes; returns array of episode rewards."""
    eval_env = gym.make(env_id)
    episode_rewards = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        while not done:
            obs_t = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)
            action, _, _, _ = policy.get_action_and_value(obs_t)
            action = action.cpu().numpy().flatten()
            action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)
    eval_env.close()
    return np.asarray(episode_rewards)


# =====================================================================
# EDITABLE: Reward Network and IRL Algorithm
# =====================================================================
class RewardNetwork(nn.Module):
    """Reward network R(s, a, s') -> scalar.

    Takes state, action, next_state as input and outputs a scalar reward.
    This is the discriminator/reward model used in IRL.

    You may redesign this architecture entirely. The forward signature must remain:
        forward(state, action, next_state) -> reward_tensor of shape (batch,)
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        input_dim = obs_dim + action_dim + obs_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action, next_state):
        """Compute reward for a batch of transitions.

        Args:
            state: (batch, obs_dim) current observations
            action: (batch, action_dim) actions taken
            next_state: (batch, obs_dim) next observations

        Returns:
            Reward tensor of shape (batch,)
        """
        x = torch.cat([state, action, next_state], dim=-1)
        return self.net(x).squeeze(-1)


class IRLAlgorithm:
    """Inverse RL / Reward Learning algorithm.

    Responsible for:
      1. Training the reward network to distinguish expert from policy data.
      2. Providing learned rewards for policy training.

    The main training loop calls:
        irl = IRLAlgorithm(reward_net, expert_demos, obs_dim, action_dim, device, args)
        ...
        # After collecting on-policy rollout data:
        irl.update(policy_obs, policy_acts, policy_next_obs, policy_dones)
        ...
        # To compute rewards for PPO:
        rewards = irl.compute_reward(obs, acts, next_obs)

    Available classes (defined above, editable):
        RewardNetwork — R(s, a, s') -> scalar

    You MUST keep:
        - self.reward_net set to a RewardNetwork instance
        - compute_reward(obs, acts, next_obs) -> tensor of shape (batch,)
        - update(...) that trains the reward network
    """

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)
        self.total_updates = 0

    def compute_reward(self, obs, acts, next_obs):
        """Compute learned reward for given transitions (used during PPO rollout).

        Args:
            obs: (batch, obs_dim) observations
            acts: (batch, action_dim) actions
            next_obs: (batch, obs_dim) next observations

        Returns:
            Reward tensor of shape (batch,)
        """
        with torch.no_grad():
            return self.reward_net(obs, acts, next_obs)

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        """Update reward network using expert demos and on-policy generator data.

        Args:
            policy_obs: (N, obs_dim) observations from current policy rollout
            policy_acts: (N, action_dim) actions from current policy rollout
            policy_next_obs: (N, obs_dim) next observations from policy rollout
            policy_dones: (N,) done flags from policy rollout

        Returns:
            dict of scalar metrics for logging

        TODO: Implement your IRL reward learning algorithm here.
        """
        self.total_updates += 1
        batch_size = self.args.irl_batch_size

        # Sample expert data
        n_expert = len(self.expert_demos["obs"])
        expert_idx = torch.randint(0, n_expert, (batch_size,))
        expert_obs = self.expert_demos["obs"][expert_idx]
        expert_acts = self.expert_demos["acts"][expert_idx]
        expert_next_obs = self.expert_demos["next_obs"][expert_idx]

        # Sample policy data
        n_policy = len(policy_obs)
        policy_idx = torch.randint(0, n_policy, (batch_size,))
        gen_obs = policy_obs[policy_idx]
        gen_acts = policy_acts[policy_idx]
        gen_next_obs = policy_next_obs[policy_idx]

        # Placeholder — replace with your IRL algorithm
        loss = torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"irl_loss": loss.item()}


# =====================================================================
# FIXED: PPO update step
# =====================================================================
def ppo_update(policy, optimizer, buffer, args, device):
    """Run PPO update on the rollout buffer. Returns metrics dict."""
    n_steps = buffer.ptr
    obs = buffer.obs[:n_steps]
    actions = buffer.actions[:n_steps]
    logprobs = buffer.logprobs[:n_steps]
    rewards = buffer.rewards[:n_steps]
    dones = buffer.dones[:n_steps]
    values = buffer.values[:n_steps]

    # Compute GAE
    with torch.no_grad():
        next_value = policy.get_value(buffer.next_obs[n_steps - 1].unsqueeze(0)).squeeze()
    advantages = torch.zeros(n_steps, device=device)
    lastgaelam = 0
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalue = next_value
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalue = values[t + 1]
        delta = rewards[t] + args.gamma * nextvalue * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values

    # PPO epochs
    indices = np.arange(n_steps)
    total_pg_loss = 0.0
    total_v_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for epoch in range(args.n_epochs):
        np.random.shuffle(indices)
        for start in range(0, n_steps, args.minibatch_size):
            end = start + args.minibatch_size
            if end > n_steps:
                break
            mb_idx = indices[start:end]

            _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                obs[mb_idx], actions[mb_idx]
            )
            logratio = newlogprob - logprobs[mb_idx]
            ratio = logratio.exp()

            mb_advantages = advantages[mb_idx]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = F.mse_loss(newvalue.squeeze(), returns[mb_idx])

            # Entropy loss
            entropy_loss = entropy.mean()

            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_entropy += entropy_loss.item()
            n_updates += 1

    return {
        "pg_loss": total_pg_loss / max(n_updates, 1),
        "v_loss": total_v_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }


# =====================================================================
# FIXED: Main training loop
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--demo-path", type=str, default="")
    cli_args = parser.parse_args()

    args = Args()
    args.env_id = cli_args.env_id
    args.seed = cli_args.seed
    args.total_timesteps = cli_args.total_timesteps
    # Demo path: CLI > env SAVE_PATH > fallback
    args.demo_path = cli_args.demo_path or os.path.join(
        os.environ.get("SAVE_PATH", "/workspace"), "irl_experts"
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # Load expert demonstrations
    expert_demos = load_expert_demos(args.demo_path, args.env_id, device)

    # Initialize reward network and IRL algorithm
    reward_net = RewardNetwork(obs_dim, action_dim).to(device)
    irl = IRLAlgorithm(reward_net, expert_demos, obs_dim, action_dim, device, args)

    # ── FIXED: Parameter count check ────────────────────────────────
    # Budget based on 1.05x largest baseline (AIRL with g_net + h_net).
    # AIRL: g_net(obs+act -> 256 -> 256 -> 1) + h_net(obs -> 256 -> 256 -> 1)
    _g_params = (obs_dim + action_dim) * 256 + 256 + 256 * 256 + 256 + 256 + 1
    _h_params = obs_dim * 256 + 256 + 256 * 256 + 256 + 256 + 1
    _budget = int((_g_params + _h_params + 100) * 1.05)
    _total_params = sum(p.numel() for p in reward_net.parameters())
    print(f"Total reward net params: {_total_params:,} (budget: {_budget:,})")

    # Initialize policy
    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=args.policy_lr)

    # Allow IRL algorithm to access policy (used by BC baseline)
    if hasattr(irl, "set_policy"):
        irl.set_policy(policy, policy_optimizer)

    # Rollout buffer
    buffer = RolloutBuffer(args.n_steps, obs_dim, action_dim, device)

    start_time = time.time()
    global_step = 0
    obs, _ = env.reset(seed=args.seed)

    # Running reward normalization for IRL (stabilizes non-stationary rewards)
    rew_running_mean = 0.0
    rew_running_var = 1.0
    rew_count = 1e-4

    while global_step < args.total_timesteps:
        # ── Collect rollout ──
        buffer.reset()
        for step in range(args.n_steps):
            global_step += 1
            obs_t = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(obs_t)

            action_np = action.cpu().numpy().flatten()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            next_obs, env_reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Use LEARNED reward instead of environment reward
            next_obs_t = torch.tensor(next_obs.reshape(1, -1), device=device, dtype=torch.float32)
            action_t = torch.tensor(action_np.reshape(1, -1), device=device, dtype=torch.float32)
            learned_reward = irl.compute_reward(obs_t, action_t, next_obs_t).item()

            buffer.add(
                obs_t.squeeze(0), action.squeeze(0), logprob.squeeze(),
                learned_reward, float(done), value.squeeze(), next_obs_t.squeeze(0),
            )

            if "episode" in info:
                ep_return = info["episode"]["r"]
                print(f"global_step={global_step}, episodic_return={ep_return}", flush=True)

            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

            if global_step >= args.total_timesteps:
                break

        # ── IRL update: train reward network ──
        for _ in range(args.n_irl_updates_per_round):
            irl_metrics = irl.update(
                buffer.obs[:buffer.ptr],
                buffer.actions[:buffer.ptr],
                buffer.next_obs[:buffer.ptr],
                buffer.dones[:buffer.ptr],
            )

        # ── Normalize learned rewards (running mean/std) ──
        n = buffer.ptr
        raw_rewards = buffer.rewards[:n]
        batch_mean = raw_rewards.mean().item()
        batch_var = raw_rewards.var().item()
        delta = batch_mean - rew_running_mean
        new_count = rew_count + n
        rew_running_mean += delta * n / new_count
        m_a = rew_running_var * rew_count
        m_b = batch_var * n
        M2 = m_a + m_b + delta ** 2 * rew_count * n / new_count
        rew_running_var = M2 / new_count
        rew_count = new_count
        rew_std = max(rew_running_var ** 0.5, 1e-8)
        buffer.rewards[:n] = (raw_rewards - rew_running_mean) / rew_std

        # ── PPO update: improve policy using learned reward ──
        ppo_metrics = ppo_update(policy, policy_optimizer, buffer, args, device)

        # ── Logging ──
        all_metrics = {**ppo_metrics, **irl_metrics}
        metrics_str = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in all_metrics.items()
        )
        print(f"TRAIN_METRICS step={global_step} {metrics_str}", flush=True)

        # ── Periodic evaluation ──
        if global_step % args.eval_freq < args.n_steps or global_step >= args.total_timesteps:
            eval_returns = evaluate_policy(
                args.env_id, policy, device,
                n_episodes=args.eval_episodes, seed=args.seed + 1000,
            )
            mean_return = eval_returns.mean()
            print(f"Eval episodic_return: {mean_return:.2f}", flush=True)

    env.close()
    print("Training complete.", flush=True)
