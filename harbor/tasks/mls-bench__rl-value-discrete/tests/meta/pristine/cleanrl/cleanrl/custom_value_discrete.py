# Custom value-based discrete RL algorithm for MLS-Bench
#
# EDITABLE section: QNetwork head and ValueAlgorithm classes.
# FIXED sections: everything else (config, env, buffer, encoder, utility, training loop).
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
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the replay memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
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
    """Numpy-based replay buffer for discrete actions."""

    def __init__(self, obs_dim, max_size=10000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size,), dtype=np.int64)
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
            torch.tensor(self.actions[idx], dtype=torch.long, device=device),
            torch.tensor(self.rewards[idx], device=device),
            torch.tensor(self.dones[idx], device=device),
        )


# =====================================================================
# FIXED: Utilities
# =====================================================================
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear epsilon schedule from start_e to end_e over duration steps."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@torch.no_grad()
def eval_qnetwork(env_id, q_network, device, n_episodes, seed):
    """Evaluate Q-network greedily for n_episodes in a fresh env; returns array of episode rewards."""
    eval_env = gym.make(env_id)
    episode_rewards = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        while not done:
            obs_t = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)
            q_values = q_network(obs_t)
            action = torch.argmax(q_values, dim=1).item()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)
    eval_env.close()
    return np.asarray(episode_rewards)


# =====================================================================
# FIXED: MLP Encoder (network capacity is controlled here)
# =====================================================================
ENCODER_HIDDEN_DIMS = [120, 84]
ENCODER_FEATURE_DIM = ENCODER_HIDDEN_DIMS[-1]  # 84


class MLPEncoder(nn.Module):
    """Fixed 2-layer MLP encoder: obs_dim -> 120 -> 84.

    All algorithms share this backbone. Only the head (defined in the
    EDITABLE section) may differ.
    """

    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


# =====================================================================
# EDITABLE: QNetwork head and ValueAlgorithm
# =====================================================================
class QNetwork(nn.Module):
    """Q-network: MLPEncoder (fixed) + head. Output: Q-values per action (batch x n_actions).

    The encoder is FIXED (120->84 MLP). Only the head layer(s) on top of
    the 84-dim features may be changed.
    """

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim)
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)

    def forward(self, obs):
        features = self.encoder(obs)
        return self.head(features)


class ValueAlgorithm:
    """Value-based RL algorithm -- implement your approach here.

    The training loop calls:
        algorithm = ValueAlgorithm(obs_dim, n_actions, device, args)
        action = algorithm.select_action(obs, epsilon)   # during data collection
        metrics = algorithm.update(batch, global_step)    # after each training step
        eval_qnetwork(env_id, algorithm.q_network, ...)   # periodic evaluation

    You MUST set self.q_network to an nn.Module with forward(obs) -> Q-values.

    Available classes:
        MLPEncoder (fixed) -- 2-layer MLP encoder, obs_dim -> 84-dim features
        QNetwork   (editable) -- MLPEncoder + head
    ENCODER_FEATURE_DIM = 84 (feature dimension from MLPEncoder)

    Available utilities (fixed): linear_schedule, eval_qnetwork
    """

    def __init__(self, obs_dim, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.total_it = 0

        # Build networks -- modify or replace as needed
        self.q_network = QNetwork(obs_dim, n_actions).to(device)
        self.target_network = QNetwork(obs_dim, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        """Select action with epsilon-greedy exploration."""
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        q_values = self.q_network(obs_t)
        return torch.argmax(q_values, dim=1).item()

    def update(self, batch, global_step):
        """Single gradient update. Returns a dict of scalar metrics.

        batch = (obs, next_obs, actions, rewards, dones) -- torch.Tensor on device

        TODO: implement your value-based RL algorithm here.
        """
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        # Placeholder -- replace with your algorithm
        return {"td_loss": 0.0, "q_values": 0.0}


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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    n_actions = envs.single_action_space.n

    # Algorithm
    algorithm = ValueAlgorithm(obs_dim, n_actions, device, args)

    # Replay buffer
    rb = SimpleReplayBuffer(obs_dim, args.buffer_size)

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # Epsilon-greedy action selection
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        action = algorithm.select_action(obs[0], epsilon)
        actions = np.array([action])

        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
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
            if global_step % args.train_frequency == 0:
                batch = rb.sample(args.batch_size, device)
                log_dict = algorithm.update(batch, global_step)

                if global_step % 1000 == 0:
                    metrics_str = " ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in log_dict.items()
                    )
                    print(f"TRAIN_METRICS step={global_step} {metrics_str}", flush=True)

            # Update target network
            if global_step % args.target_network_frequency == 0:
                for target_param, q_param in zip(algorithm.target_network.parameters(), algorithm.q_network.parameters()):
                    target_param.data.copy_(
                        args.tau * q_param.data + (1.0 - args.tau) * target_param.data
                    )

        # Evaluation
        if (global_step + 1) % args.eval_freq == 0:
            eval_returns = eval_qnetwork(
                args.env_id, algorithm.q_network, device,
                n_episodes=args.eval_episodes, seed=args.seed + 1000,
            )
            mean_return = eval_returns.mean()
            print(f"Eval episodic_return: {mean_return:.2f}", flush=True)

    envs.close()
