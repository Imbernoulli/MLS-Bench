# Custom value-based RL algorithm for Atari -- MLS-Bench
#
# EDITABLE section: QNetwork head and ValueAlgorithm classes.
# FIXED sections: everything else (config, env, buffer, encoder, eval, training loop).
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

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer


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
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the replay memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    eval_freq: int = 100000
    """evaluation frequency (timesteps)"""
    eval_episodes: int = 10
    """number of evaluation episodes"""


# =====================================================================
# FIXED: Environment setup
# =====================================================================
def make_env(env_id, seed):
    """Create a training environment with the full Atari wrapper stack."""
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk


def make_eval_env(env_id, seed):
    """Create an evaluation environment (no EpisodicLifeEnv, no ClipRewardEnv)."""
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk


# =====================================================================
# FIXED: Replay Buffer (uses cleanrl_utils.buffers.ReplayBuffer)
# =====================================================================
# The ReplayBuffer is instantiated in the training loop below using
# cleanrl_utils.buffers.ReplayBuffer with optimize_memory_usage=True.


# =====================================================================
# FIXED: Utilities
# =====================================================================
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@torch.no_grad()
def eval_qnetwork(env_id, algorithm, device, n_episodes, seed):
    """Evaluate value algorithm for n_episodes in a fresh eval env; returns array of episode rewards."""
    eval_envs = gym.vector.SyncVectorEnv([make_eval_env(env_id, seed)])
    episode_rewards = []
    obs, _ = eval_envs.reset(seed=seed)
    while len(episode_rewards) < n_episodes:
        q_values = algorithm.q_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        obs, rewards, terminations, truncations, infos = eval_envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_rewards.append(float(info["episode"]["r"]))
    eval_envs.close()
    return np.asarray(episode_rewards[:n_episodes])


# =====================================================================
# FIXED: Nature DQN Encoder (network capacity is controlled here)
# =====================================================================
ENCODER_FEATURE_DIM = 512


class NatureDQNEncoder(nn.Module):
    """Nature DQN CNN encoder (Mnih et al. 2015).

    Input: (B, 4, 84, 84) uint8 frames
    Output: (B, 512) feature vector

    All algorithms share this backbone. Only the head (defined in the
    EDITABLE section) may differ.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, ENCODER_FEATURE_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(self.conv(x / 255.0))


# =====================================================================
# EDITABLE: QNetwork head and ValueAlgorithm
# =====================================================================
class QNetwork(nn.Module):
    """Q-network: NatureDQNEncoder (fixed) + head. Output: Q-values per action.

    The encoder is FIXED (Nature DQN CNN -> 512-dim features). Only the
    head layer(s) on top of the 512-dim features may be changed.
    """

    def __init__(self, envs):
        super().__init__()
        n_actions = envs.single_action_space.n
        self.encoder = NatureDQNEncoder()
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class ValueAlgorithm:
    """Value-based algorithm for Atari -- implement your approach here.

    The training loop calls:
        algorithm = ValueAlgorithm(envs, device, args)
        action = algorithm.select_action(obs, epsilon)
        metrics = algorithm.update(batch, global_step)
        eval_qnetwork(env_id, algorithm, device, ...)

    You MUST set self.q_network and self.target_network to nn.Module instances.

    Available classes:
        NatureDQNEncoder (fixed) -- Nature DQN CNN encoder, -> 512-dim features
        QNetwork         (editable) -- NatureDQNEncoder + head
    ENCODER_FEATURE_DIM = 512 (feature dimension from NatureDQNEncoder)

    Available utilities (fixed): linear_schedule
    """

    def __init__(self, envs, device, args):
        self.device = device
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_network_frequency = args.target_network_frequency

        self.q_network = QNetwork(envs).to(device)
        self.target_network = QNetwork(envs).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return np.array([self.q_network.head.out_features])  # placeholder
        q_values = self.q_network(torch.Tensor(obs).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def update(self, batch, global_step):
        """Single gradient update. Returns a dict of scalar metrics.

        batch: cleanrl ReplayBuffer sample with .observations, .next_observations,
               .actions, .rewards, .dones

        TODO: implement your value-based RL algorithm here.
        """
        return {"td_loss": 0.0, "q_values": 0.0}


# =====================================================================
# FIXED: Parameter count assertion
# =====================================================================
def _check_param_budget(q_network, n_actions):
    """Ensure the Q-network does not exceed the parameter budget.

    The budget is the NatureDQNEncoder params + a generous head allowance.
    This prevents capacity hacking by adding hidden layers.
    """
    encoder_params = sum(p.numel() for p in NatureDQNEncoder().parameters())
    # Largest head: QR-DQN with 200 quantiles: n_actions * 200 outputs
    max_head_output = n_actions * 200
    max_head_params = ENCODER_FEATURE_DIM * max_head_output + max_head_output
    max_total = int((encoder_params + max_head_params) * 1.05)
    actual = sum(p.numel() for p in q_network.parameters())
    print(
        f"QNetwork parameters: {actual:,} / {max_total:,} "
        f"(1.05x largest baseline, informational only)",
        flush=True,
    )


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

    # Algorithm
    algorithm = ValueAlgorithm(envs, device, args)

    # Parameter budget check
    _check_param_budget(algorithm.q_network, envs.single_action_space.n)

    # Replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # Epsilon-greedy action selection
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = algorithm.select_action(obs, epsilon=0.0)

        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    break

        # Handle truncation
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                batch = rb.sample(args.batch_size)
                log_dict = algorithm.update(batch, global_step)

                if global_step % 1000 == 0:
                    metrics_str = " ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in log_dict.items()
                    )
                    print(f"TRAIN_METRICS step={global_step} {metrics_str}", flush=True)

        # Evaluation
        if (global_step + 1) % args.eval_freq == 0:
            eval_returns = eval_qnetwork(
                args.env_id, algorithm, device,
                n_episodes=args.eval_episodes, seed=args.seed + 1000,
            )
            mean_return = eval_returns.mean()
            print(f"Eval episodic_return: {mean_return:.2f}", flush=True)

    envs.close()
