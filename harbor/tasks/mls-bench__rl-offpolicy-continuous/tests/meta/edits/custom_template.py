# Custom off-policy continuous RL algorithm for MLS-Bench
#
# EDITABLE section: Actor, QNetwork, and OffPolicyAlgorithm classes.
# FIXED sections: everything else (config, env, buffer, eval, training loop).
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
import tyro


# =====================================================================
# FIXED: Configuration
# =====================================================================
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 25000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    eval_freq: int = 10000
    """evaluation frequency (timesteps)"""
    eval_episodes: int = 10
    """number of evaluation episodes"""


# =====================================================================
# FIXED: Environment setup
# =====================================================================
def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


# =====================================================================
# FIXED: Replay Buffer
# =====================================================================
class SimpleReplayBuffer:
    """Numpy-based replay buffer for continuous actions."""

    def __init__(self, obs_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx], device=device),
            torch.tensor(self.next_obs[idx], device=device),
            torch.tensor(self.actions[idx], device=device),
            torch.tensor(self.rewards[idx], device=device),
            torch.tensor(self.dones[idx], device=device),
        )


# =====================================================================
# FIXED: Utilities
# =====================================================================
def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def _mlp_factory(input_dim, output_dim, hidden=256):
    """Build a 2-hidden-layer MLP. Use this as a building block for actors/critics."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )


@torch.no_grad()
def eval_actor(env_id, actor, device, n_episodes, seed):
    """Evaluate actor for n_episodes in a fresh env; returns array of episode rewards."""
    eval_env = gym.make(env_id)
    episode_rewards = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        while not done:
            obs_t = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)
            action = actor.get_action(obs_t)
            if isinstance(action, tuple):
                action = action[0]
            action = action.cpu().numpy().flatten()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)
    eval_env.close()
    return np.asarray(episode_rewards)


# =====================================================================
# EDITABLE: Network definitions and OffPolicyAlgorithm
# =====================================================================
class Actor(nn.Module):
    """Actor network. Must implement forward(obs) and get_action(obs).

    forward(obs) -> action tensor (used for training).
    get_action(obs) -> action tensor (used for evaluation, no grad).
    """

    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor(max_action, dtype=torch.float32))

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x)) * self.action_scale

    @torch.no_grad()
    def get_action(self, obs):
        return self.forward(obs)


class QNetwork(nn.Module):
    """Q-function Q(s, a) -> scalar."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OffPolicyAlgorithm:
    """Off-policy actor-critic algorithm -- implement your approach here.

    The training loop calls:
        algorithm = OffPolicyAlgorithm(obs_dim, action_dim, max_action, device, args)
        action = algorithm.select_action(obs)        # during data collection
        metrics = algorithm.update(batch)             # after each env step
        eval_actor(env_id, algorithm.actor, ...)      # periodic evaluation

    You MUST set self.actor to an nn.Module with a .get_action(obs) method.

    Available classes (defined above, editable):
        Actor    -- deterministic policy with tanh squashing
        QNetwork -- Q(s, a) critic

    Available utilities (fixed): soft_update, _mlp_factory
    """

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.total_it = 0

        # Build networks -- modify or replace as needed
        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.qf1 = QNetwork(obs_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=args.learning_rate)

    def select_action(self, obs):
        """Select action for environment interaction (with exploration noise)."""
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        action = self.actor(obs_t).cpu().numpy().flatten()
        noise = np.random.normal(0, self.max_action * 0.1, size=action.shape)
        return np.clip(action + noise, -self.max_action, self.max_action)

    def update(self, batch):
        """Single gradient update. Returns a dict of scalar metrics.

        batch = (obs, next_obs, actions, rewards, dones) -- torch.Tensor on device

        TODO: implement your off-policy RL algorithm here.
        """
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        # Placeholder -- replace with your algorithm
        return {"critic_loss": 0.0, "actor_loss": 0.0}


# =====================================================================
# FIXED: Training loop
# =====================================================================
if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.prod(envs.single_action_space.shape)
    max_action = float(envs.single_action_space.high[0])

    # Algorithm
    algorithm = OffPolicyAlgorithm(obs_dim, action_dim, max_action, device, args)

    # Replay buffer
    rb = SimpleReplayBuffer(obs_dim, action_dim, args.buffer_size)

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()])
        else:
            actions = algorithm.select_action(obs[0]).reshape(1, -1)

        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    break

        # Handle truncation
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs[0], real_next_obs[0], actions[0], rewards[0], terminations[0])
        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            batch = rb.sample(args.batch_size, device)
            log_dict = algorithm.update(batch)

            if global_step % 1000 == 0:
                metrics_str = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in log_dict.items()
                )
                print(f"TRAIN_METRICS step={global_step} {metrics_str}", flush=True)

        # Evaluation
        if (global_step + 1) % args.eval_freq == 0:
            eval_returns = eval_actor(
                args.env_id, algorithm.actor, device,
                n_episodes=args.eval_episodes, seed=args.seed + 1000,
            )
            mean_return = eval_returns.mean()
            print(f"Eval episodic_return: {mean_return:.2f}", flush=True)

    envs.close()
