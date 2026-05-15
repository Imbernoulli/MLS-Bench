# Custom offline RL algorithm for MLS-Bench
#
# EDITABLE section: network definitions + OfflineAlgorithm class.
# FIXED sections: everything else (config, utilities, data, eval, training loop).
import os
import random
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution

TensorBatch = List[torch.Tensor]


# =====================================================================
# FIXED: Configuration
# batch_size, eval_freq, n_episodes, max_timesteps are enforced here.
# =====================================================================
@dataclass
class TrainConfig:
    device: str = "cuda"
    env: str = "halfcheetah-medium-v2"
    seed: int = 0
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    max_timesteps: int = int(1e6)
    checkpoints_path: Optional[str] = None
    buffer_size: int = 2_000_000
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 5e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    normalize: bool = True
    orthogonal_init: bool = True
    project: str = "CORL"
    group: str = "custom-D4RL"
    name: str = "custom"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# =====================================================================
# FIXED: Utilities
# =====================================================================
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    return states.mean(0), states.std(0) + eps


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    env = gym.wrappers.TransformObservation(env, lambda s: (s - state_mean) / state_std)
    return env


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def init_module_weights(module: nn.Sequential, orthogonal_init: bool = False):
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)
    last = module[-1]
    if orthogonal_init:
        nn.init.orthogonal_(last.weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(last.weight, gain=1e-2)
    nn.init.constant_(last.bias, 0.0)


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._next_actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n = data["observations"].shape[0]
        if n > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        self._states[:n] = self._to_tensor(data["observations"])
        self._actions[:n] = self._to_tensor(data["actions"])
        self._rewards[:n] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n] = self._to_tensor(data["next_observations"])
        self._dones[:n] = self._to_tensor(data["terminals"][..., None])
        # Compute next_actions: action at the next timestep in the dataset.
        # At episode boundaries (terminal=1) or the last transition, use the current action.
        next_actions = np.concatenate([data["actions"][1:], data["actions"][-1:]], axis=0)
        terminals = data["terminals"].astype(bool)
        next_actions[terminals] = data["actions"][terminals]
        self._next_actions[:n] = self._to_tensor(next_actions)
        self._size += n
        self._pointer = min(self._size, n)
        print(f"Dataset size: {n}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        return [
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            self._next_actions[indices],
        ]


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    """Evaluate actor for n_episodes; returns array of episode rewards."""
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    actor.train()
    return np.asarray(episode_rewards)


def _mlp(input_dim: int, output_dim: int, hidden_dim: int = 256,
         n_layers: int = 3) -> nn.Sequential:
    """FIXED MLP factory -- hidden width is locked at 256. Do NOT modify."""
    assert hidden_dim == 256, "hidden_dim must be 256"
    layers: list = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


# =====================================================================
# EDITABLE: Network definitions and OfflineAlgorithm
#
# CONSTRAINTS:
# - Total trainable parameter count is soft-capped.
# - Total parameter count is checked at runtime and must not exceed
#   1.2x the largest baseline. Focus on algorithmic improvements, not
#   network capacity.
#
# CONFIG_OVERRIDES: override method-specific TrainConfig fields here.
# Allowed keys: normalize, normalize_reward, actor_lr, critic_lr, tau, discount.
# Example: CONFIG_OVERRIDES = {"normalize": False, "actor_lr": 1e-3}
# =====================================================================
CONFIG_OVERRIDES: Dict[str, Any] = {}


class DeterministicActor(nn.Module):
    """Deterministic policy pi(s) = tanh(net(s)) * max_action.
    Suitable for BC, TD3+BC style algorithms. Default: 2 x 256 MLP."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Actor(nn.Module):
    """Tanh-Gaussian stochastic policy. Default: 3 x 256 MLP.
    Suitable for CQL, IQL style algorithms."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 orthogonal_init: bool = False):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )
        init_module_weights(self.net, orthogonal_init)
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def _get_dist(self, state: torch.Tensor):
        out = self.net(state)
        mean, log_std = torch.split(out, self.action_dim, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return TransformedDistribution(
            Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
        ), mean

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        dist, mean = self._get_dist(state)
        action = torch.tanh(mean) if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return self.max_action * action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-probability of a dataset action under the current policy."""
        dist, _ = self._get_dist(state)
        action = torch.clamp(action / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
        return dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q-function Q(s, a). Default: 3 x 256 MLP."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        init_module_weights(self.net, orthogonal_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ValueFunction(nn.Module):
    """State value function V(s). Default: 3 x 256 MLP. Useful for IQL-style algorithms."""

    def __init__(self, state_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        init_module_weights(self.net, orthogonal_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class OfflineAlgorithm:
    """Offline RL algorithm -- implement your approach here.

    Goal: suppress Q-value overestimation while learning from offline data
    on continuous locomotion tasks (HalfCheetah, Hopper, Walker2d).

    The training loop calls:
        trainer = OfflineAlgorithm(state_dim, action_dim, max_action, **kwargs)
        log_dict = trainer.train(batch)          # called at every timestep
        eval_actor(env, trainer.actor, ...)      # called every eval_freq steps

    You MUST set self.actor to an nn.Module that has an .act(state, device) method.
    Build all your networks here; the training loop only provides environment dimensions
    and the hyperparameters from TrainConfig.

    actor_lr, critic_lr, alpha_lr are passed from TrainConfig as starting points --
    you may use different values by creating optimizers with your own lr in __init__.
    Any additional algorithm-specific hyperparameters should be hardcoded in __init__.

    Dataset access:
        replay_buffer is a ReplayBuffer instance containing the full offline dataset.
        You can use it to compute dataset-level statistics, e.g.:
            replay_buffer._states[:replay_buffer._size]   -- all states  (Tensor)
            replay_buffer._actions[:replay_buffer._size]  -- all actions (Tensor)
            replay_buffer._rewards[:replay_buffer._size]  -- all rewards (Tensor)
            replay_buffer._next_actions[:replay_buffer._size] -- next actions (Tensor)
            replay_buffer._size                           -- number of transitions

    Available network classes (defined above, editable):
        DeterministicActor  -- deterministic tanh policy (for BC / TD3+BC approaches)
        Actor               -- stochastic Tanh-Gaussian policy (for CQL / IQL approaches)
        Critic              -- Q(s, a) network
        ValueFunction       -- V(s) network (for IQL-style approaches)
    Available utilities (fixed): soft_update, init_module_weights
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        replay_buffer: "ReplayBuffer" = None,
        discount: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        orthogonal_init: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0
        # Full offline dataset -- use for computing global statistics
        # (e.g. action mean/std for normalization, reward scaling, etc.)
        self.replay_buffer = replay_buffer

        # Build networks -- modify or replace as needed
        self.actor = Actor(state_dim, action_dim, max_action, orthogonal_init).to(device)
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        """Update networks on one batch. Return a dict of scalar metrics for logging.

        batch = [states, actions, rewards, next_states, dones, next_actions]
        (torch.Tensor, on device)

        next_actions is the action at the next timestep in the dataset
        (useful for algorithms like ReBRAC that penalize deviations from
        the data's next action).

        TODO: implement your offline RL algorithm here.
        """
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions = batch

        # -- Placeholder: replace with your algorithm --
        log_dict: Dict[str, float] = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
        }
        return log_dict


# =====================================================================
# FIXED: Training loop
# =====================================================================
@pyrallis.wrap()
def train(config: TrainConfig):
    # Apply editable config overrides
    for _k, _v in CONFIG_OVERRIDES.items():
        if hasattr(config, _k):
            setattr(config, _k, _v)

    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0.0, 1.0

    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)

    set_seed(config.seed, env)

    trainer = OfflineAlgorithm(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        replay_buffer=replay_buffer,
        discount=config.discount,
        tau=config.tau,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        orthogonal_init=config.orthogonal_init,
        device=config.device,
    )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        if (t + 1) % 1000 == 0:
            metrics_str = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in log_dict.items()
            )
            print(f"TRAIN_METRICS step={t+1} {metrics_str}", flush=True)

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env, trainer.actor, device=config.device,
                n_episodes=config.n_episodes, seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                pass  # checkpoint saving disabled (crashes on tmpfs)


if __name__ == "__main__":
    train()
