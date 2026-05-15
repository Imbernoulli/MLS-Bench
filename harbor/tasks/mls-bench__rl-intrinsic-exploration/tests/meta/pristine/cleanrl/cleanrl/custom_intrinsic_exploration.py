# Custom sparse-reward Atari exploration benchmark for MLS-Bench.
#
# FIXED sections: PPO loop, Atari preprocessing, policy/value architecture,
# evaluation, logging, and optimizer wiring.
# EDITABLE section: IntrinsicBonusModule + mix_advantages.

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    env_id: str = "MontezumaRevenge-v5"
    total_timesteps: int = 10000000
    learning_rate: float = 1e-4
    num_envs: int = 32
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    int_coef: float = 1.0
    ext_coef: float = 2.0
    int_gamma: float = 0.99
    update_proportion: float = 0.25
    num_iterations_obs_norm_init: int = 10

    eval_interval: int = 500000
    eval_episodes: int = 5
    eval_max_episode_steps: int = 27000

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, dones, infos


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def last_frame(obs: torch.Tensor) -> torch.Tensor:
    return obs[:, 3:4, :, :].float()


class RewardForwardFilter:
    def __init__(self, gamma: float):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.1),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def _hidden(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network(obs / 255.0)
        features = self.extra_layer(hidden)
        return hidden, features

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self._hidden(obs)
        return self.actor(hidden)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        hidden, features = self._hidden(obs)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, features = self._hidden(obs)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.get_logits(obs), dim=1)


# =====================================================================
# EDITABLE: intrinsic reward design
# =====================================================================
class IntrinsicBonusModule(nn.Module):
    """Default baseline: no intrinsic reward."""

    def __init__(self, action_dim: int, device: torch.device, args: Args):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args

    def initialize(self, envs) -> None:
        return None

    def trainable_parameters(self):
        return []

    def update_batch_stats(self, batch_obs: torch.Tensor, batch_next_obs: torch.Tensor) -> None:
        return None

    def compute_bonus(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(obs.shape[0], device=self.device)

    def normalize_rollout_rewards(self, rollout_intrinsic: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(rollout_intrinsic)

    def loss(
        self,
        batch_obs: torch.Tensor,
        batch_next_obs: torch.Tensor,
        batch_actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros((), device=self.device)


def mix_advantages(ext_advantages: torch.Tensor, int_advantages: torch.Tensor, args: Args) -> torch.Tensor:
    return args.ext_coef * ext_advantages


# =====================================================================
# FIXED: evaluation and training loop
# =====================================================================
@torch.no_grad()
def evaluate_policy(args: Args, agent: Agent, device: torch.device, seed: int) -> tuple[float, float]:
    # Cap Atari evaluation episodes so deterministic no-op / survival loops cannot
    # stall the whole benchmark at the first eval checkpoint.
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=1,
        episodic_life=False,
        reward_clip=True,
        repeat_action_probability=0.25,
        seed=seed,
    )
    envs.num_envs = 1
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)

    returns = []
    obs = torch.tensor(envs.reset(), device=device)
    episode_steps = 0
    while len(returns) < args.eval_episodes:
        action = agent.get_deterministic_action(obs)
        next_obs, _, done, info = envs.step(action.cpu().numpy())
        obs = torch.tensor(next_obs, device=device)
        episode_steps += 1
        if done[0] or episode_steps >= args.eval_max_episode_steps:
            returns.append(float(info["r"][0]))
            episode_steps = 0
            if len(returns) < args.eval_episodes:
                obs = torch.tensor(envs.reset(), device=device)

    envs.close()
    returns_np = np.asarray(returns, dtype=np.float32)
    return float(returns_np.mean()), float((returns_np != 0.0).mean())


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        repeat_action_probability=0.25,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    bonus_module = IntrinsicBonusModule(envs.single_action_space.n, device, args).to(device)
    bonus_params = list(bonus_module.trainable_parameters())
    optimizer = optim.Adam(list(agent.parameters()) + bonus_params, lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    next_obs_buf = torch.zeros_like(obs)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.int64)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    int_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    ext_values = torch.zeros((args.num_steps, args.num_envs), device=device)
    int_values = torch.zeros((args.num_steps, args.num_envs), device=device)

    recent_returns: deque[float] = deque(maxlen=20)
    eval_steps: list[int] = []
    eval_returns: list[float] = []
    eval_nonzero_rates: list[float] = []
    next_eval_step = args.eval_interval

    global_step = 0
    start_time = time.time()
    # Reset once before any bootstrap rollout so the wrapper's episode buffers exist,
    # then reset again to start actual training from a clean state.
    envs.reset()
    bonus_module.initialize(envs)
    next_obs = torch.tensor(envs.reset(), device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    eval_seed = args.seed + 10_000

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step] = value_ext.flatten()
                int_values[step] = value_int.flatten()
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            stepped_obs, reward, done, info = envs.step(action.cpu().numpy())
            next_obs = torch.tensor(stepped_obs, device=device)
            next_done = torch.tensor(done, device=device, dtype=torch.float32)
            next_obs_buf[step] = next_obs
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            with torch.no_grad():
                rollout_bonus = bonus_module.compute_bonus(obs[step], next_obs, action)
            int_rewards[step] = rollout_bonus * (1.0 - next_done)

            for idx, terminated in enumerate(done):
                if terminated and info["lives"][idx] == 0:
                    recent_returns.append(float(info["r"][idx]))

        int_rewards = bonus_module.normalize_rollout_rewards(int_rewards)

        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext = next_value_ext.reshape(1, -1)
            next_value_int = next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(int_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = ext_nextnonterminal
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = ext_nextnonterminal
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = int_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)

        b_advantages = mix_advantages(b_ext_advantages, b_int_advantages, args)
        bonus_module.update_batch_stats(b_obs, b_next_obs)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_ext_values = new_ext_values.view(-1)
                new_int_values = new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss = 0.5 * torch.max(ext_v_loss_unclipped, ext_v_loss_clipped).mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()
                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss

                entropy_loss = entropy.mean()
                bonus_loss = bonus_module.loss(
                    b_obs[mb_inds],
                    b_next_obs[mb_inds],
                    b_actions[mb_inds],
                )
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + bonus_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(agent.parameters()) + bonus_params, args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        latest_eval_return = float("nan")
        latest_nonzero = float("nan")
        if global_step >= next_eval_step or iteration == args.num_iterations:
            latest_eval_return, latest_nonzero = evaluate_policy(args, agent, device, eval_seed)
            eval_steps.append(global_step)
            eval_returns.append(latest_eval_return)
            eval_nonzero_rates.append(latest_nonzero)
            next_eval_step += args.eval_interval

        avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        avg_intrinsic = float(int_rewards.mean().item())
        sps = int(global_step / max(time.time() - start_time, 1e-6))
        print(
            f"TRAIN_METRICS step={global_step} avg_return={avg_return:.4f} "
            f"avg_intrinsic={avg_intrinsic:.6f} eval_return={latest_eval_return:.4f} "
            f"nonzero_rate={latest_nonzero:.4f} sps={sps}",
            flush=True,
        )

    if not eval_returns:
        final_eval_return, final_nonzero = evaluate_policy(args, agent, device, eval_seed)
        eval_steps.append(global_step)
        eval_returns.append(final_eval_return)
        eval_nonzero_rates.append(final_nonzero)

    auc = float(np.trapz(np.asarray(eval_returns, dtype=np.float32), np.asarray(eval_steps, dtype=np.float32)))
    auc /= max(float(eval_steps[-1]), 1.0)
    final_eval_return = float(eval_returns[-1])
    final_nonzero = float(eval_nonzero_rates[-1])
    best_eval_return = float(np.max(eval_returns))
    print(
        f"TEST_METRICS eval_return={final_eval_return:.4f} auc={auc:.6f} "
        f"nonzero_rate={final_nonzero:.4f} best_eval_return={best_eval_return:.4f}",
        flush=True,
    )
