#!/bin/bash
# PPO baseline — uses stable-baselines3 PPO on HumanoidBench environments
cd fast_td3
python -c "
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import random
import numpy as np
import torch
import gymnasium as gym
import humanoid_bench
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

env_name = '${ENV}'
seed = int('${SEED:-1}')

# Total frames budget: 100000 steps * 128 envs = 12.8M frames
# PPO with 16 envs: 12.8M / 16 = 800000 steps
num_envs = 16
total_timesteps = 800000

if env_name in ['h1hand-push-v0', 'h1-push-v0', 'h1hand-cube-v0', 'h1hand-basketball-v0']:
    max_episode_steps = 500
else:
    max_episode_steps = 1000

def make_env(rank):
    def _init():
        import humanoid_bench  # re-register envs in subprocess
        import gymnasium as gym
        from gymnasium.wrappers import TimeLimit
        env = gym.make(env_name)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)
        return env
    return _init

envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])


class EvalCallback(BaseCallback):
    def __init__(self, eval_freq=50000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate using a few episodes
            eval_env = SubprocVecEnv([make_env(100 + i) for i in range(8)])
            episode_returns = []
            obs = eval_env.reset()
            ep_rewards = np.zeros(8)
            ep_dones = np.zeros(8, dtype=bool)
            for _ in range(max_episode_steps):
                actions, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env.step(actions)
                ep_rewards = np.where(~ep_dones, ep_rewards + rewards, ep_rewards)
                ep_dones = ep_dones | dones
                if ep_dones.all():
                    break
            eval_env.close()
            mean_return = ep_rewards.mean()
            print(f'TRAIN_METRICS step={self.n_calls} eval_return={mean_return:.2f}')
        return True

eval_callback = EvalCallback(eval_freq=50000)

model = PPO(
    'MlpPolicy',
    envs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=seed,
    verbose=0,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
)

model.learn(total_timesteps=total_timesteps, callback=eval_callback)

# Final evaluation
eval_env = SubprocVecEnv([make_env(200 + i) for i in range(8)])
final_returns = []
for trial in range(3):
    obs = eval_env.reset()
    ep_rewards = np.zeros(8)
    ep_dones = np.zeros(8, dtype=bool)
    for _ in range(max_episode_steps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(actions)
        ep_rewards = np.where(~ep_dones, ep_rewards + rewards, ep_rewards)
        ep_dones = ep_dones | dones
        if ep_dones.all():
            break
    final_returns.append(ep_rewards.mean())
eval_env.close()
envs.close()

mean_reward = np.mean(final_returns)
std_reward = np.std(final_returns)
print(f'TEST_METRICS mean_reward={mean_reward:.4f} std_reward={std_reward:.4f}')
"
