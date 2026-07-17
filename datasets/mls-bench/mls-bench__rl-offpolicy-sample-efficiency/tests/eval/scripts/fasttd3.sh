#!/bin/bash
# FastTD3 baseline — runs the original FastTD3 training pipeline
cd fast_td3
python -c "
import os, sys
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import random, time, math, argparse
import tqdm, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tensordict import TensorDict

from fast_td3_utils import EmpiricalNormalization, SimpleReplayBuffer, mark_step
from fast_td3 import Actor, Critic

torch.set_float32_matmul_precision('high')

env_name = '${ENV}'
seed = int('${SEED:-1}')
total_timesteps = 100000
num_envs = 128
batch_size = 32768
buffer_size = 1024 * 50
gamma = 0.99
tau = 0.1
policy_noise = 0.001
noise_clip = 0.5
learning_starts = 10
policy_frequency = 2
num_updates = 2
eval_interval = 5000
actor_hidden_dim = 512
critic_hidden_dim = 1024
init_scale = 0.01
num_atoms = 101
v_min = -250.0
v_max = 250.0
std_min = 0.001
std_max = 0.4
actor_lr = 3e-4
critic_lr = 3e-4
weight_decay = 0.1

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

from environments.humanoid_bench_env import HumanoidBenchEnv
envs = HumanoidBenchEnv(env_name, num_envs, device=device)
eval_envs = envs

n_act = envs.num_actions
n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]

obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)

actor = Actor(n_obs=n_obs, n_act=n_act, num_envs=num_envs, device=device,
              hidden_dim=actor_hidden_dim, init_scale=init_scale, std_min=std_min, std_max=std_max)
qnet = Critic(n_obs=n_obs, n_act=n_act, num_atoms=num_atoms, v_min=v_min, v_max=v_max,
              hidden_dim=critic_hidden_dim, device=device)
qnet_target = Critic(n_obs=n_obs, n_act=n_act, num_atoms=num_atoms, v_min=v_min, v_max=v_max,
                     hidden_dim=critic_hidden_dim, device=device)
qnet_target.load_state_dict(qnet.state_dict())

q_optimizer = optim.AdamW(qnet.parameters(), lr=torch.tensor(critic_lr, device=device), weight_decay=weight_decay)
actor_optimizer = optim.AdamW(actor.parameters(), lr=torch.tensor(actor_lr, device=device), weight_decay=weight_decay)
q_scheduler = optim.lr_scheduler.CosineAnnealingLR(q_optimizer, T_max=total_timesteps, eta_min=torch.tensor(critic_lr, device=device))
actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=total_timesteps, eta_min=torch.tensor(actor_lr, device=device))

rb = SimpleReplayBuffer(n_env=num_envs, buffer_size=buffer_size, n_obs=n_obs, n_act=n_act,
                        n_critic_obs=n_obs, asymmetric_obs=False, n_steps=1, gamma=gamma, device=device)

amp_enabled = torch.cuda.is_available()
amp_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
amp_dtype = torch.bfloat16
scaler = GradScaler(enabled=False)

policy = actor.explore
normalize_obs = obs_normalizer.forward
if True:
    policy = torch.compile(policy, mode=None)
    normalize_obs = torch.compile(obs_normalizer.forward, mode=None)

def evaluate():
    num_eval_envs = eval_envs.num_envs
    episode_returns = torch.zeros(num_eval_envs, device=device)
    done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
    obs_e = eval_envs.reset()
    for i in range(eval_envs.max_episode_steps):
        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            obs_n = normalize_obs(obs_e, update=False)
            actions = actor(obs_n)
        next_obs_e, rewards_e, dones_e, infos_e = eval_envs.step(actions.float())
        episode_returns = torch.where(~done_masks, episode_returns + rewards_e, episode_returns)
        done_masks = torch.logical_or(done_masks, dones_e)
        if done_masks.all():
            break
        obs_e = next_obs_e
    return episode_returns.mean().item()

@torch.no_grad()
def soft_update(src, tgt, tau_val):
    src_ps = [p.data for p in src.parameters()]
    tgt_ps = [p.data for p in tgt.parameters()]
    torch._foreach_mul_(tgt_ps, 1.0 - tau_val)
    torch._foreach_add_(tgt_ps, src_ps, alpha=tau_val)

obs = envs.reset()
dones = None
global_step = 0
pbar = tqdm.tqdm(total=total_timesteps)
start_time = None

while global_step < total_timesteps:
    mark_step()
    if start_time is None and global_step >= 3 + learning_starts:
        start_time = time.time()
        measure_start = global_step

    with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
        norm_obs = normalize_obs(obs)
        actions = policy(obs=norm_obs, dones=dones)

    next_obs, rewards, dones, infos = envs.step(actions.float())
    truncations = infos['time_outs']
    true_next_obs = torch.where(dones[:, None] > 0, infos['observations']['raw']['obs'], next_obs)
    transition = TensorDict({
        'observations': obs,
        'actions': torch.as_tensor(actions, device=device, dtype=torch.float),
        'next': {
            'observations': true_next_obs,
            'rewards': torch.as_tensor(rewards, device=device, dtype=torch.float),
            'truncations': truncations.long(),
            'dones': dones.long(),
        },
    }, batch_size=(num_envs,), device=device)
    rb.extend(transition)
    obs = next_obs

    if global_step > learning_starts:
        for i in range(num_updates):
            data = rb.sample(max(1, batch_size // num_envs))
            data['observations'] = normalize_obs(data['observations'])
            data['next']['observations'] = normalize_obs(data['next']['observations'])

            with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                observations_ = data['observations']
                next_observations_ = data['next']['observations']
                actions_ = data['actions']
                rewards_ = data['next']['rewards']
                dones_ = data['next']['dones'].bool()
                truncations_ = data['next']['truncations'].bool()
                bootstrap_ = (truncations_ | ~dones_).float()
                clipped_noise_ = torch.randn_like(actions_).mul(policy_noise).clamp(-noise_clip, noise_clip)
                next_state_actions_ = (actor(next_observations_) + clipped_noise_).clamp(-1.0, 1.0)
                discount_ = gamma ** data['next']['effective_n_steps']
                with torch.no_grad():
                    qf1_proj, qf2_proj = qnet_target.projection(next_observations_, next_state_actions_, rewards_, bootstrap_, discount_)
                    qf1_val = qnet_target.get_value(qf1_proj)
                    qf2_val = qnet_target.get_value(qf2_proj)
                    qf_next_dist_ = torch.where(qf1_val.unsqueeze(1) < qf2_val.unsqueeze(1), qf1_proj, qf2_proj)
                qf1_, qf2_ = qnet(observations_, actions_)
                qf1_loss_ = -torch.sum(qf_next_dist_ * F.log_softmax(qf1_, dim=1), dim=1).mean()
                qf2_loss_ = -torch.sum(qf_next_dist_ * F.log_softmax(qf2_, dim=1), dim=1).mean()
                qf_loss_ = qf1_loss_ + qf2_loss_
            q_optimizer.zero_grad(set_to_none=True)
            scaler.scale(qf_loss_).backward()
            scaler.unscale_(q_optimizer)
            scaler.step(q_optimizer)
            scaler.update()

            should_update = (num_updates > 1 and i % policy_frequency == 1) or (num_updates == 1 and global_step % policy_frequency == 0)
            if should_update:
                with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                    qf1_a, qf2_a = qnet(data['observations'], actor(data['observations']))
                    qf1_v = qnet.get_value(F.softmax(qf1_a, dim=1))
                    qf2_v = qnet.get_value(F.softmax(qf2_a, dim=1))
                    actor_loss_ = -torch.minimum(qf1_v, qf2_v).mean()
                actor_optimizer.zero_grad(set_to_none=True)
                scaler.scale(actor_loss_).backward()
                scaler.unscale_(actor_optimizer)
                scaler.step(actor_optimizer)
                scaler.update()

            soft_update(qnet, qnet_target, tau)

        if global_step % 100 == 0 and start_time is not None:
            speed = (global_step - measure_start) / (time.time() - start_time)
            pbar.set_description(f'{speed:.1f} sps')
            if eval_interval > 0 and global_step % eval_interval == 0:
                eval_return = evaluate()
                obs = envs.reset()
                print(f'TRAIN_METRICS step={global_step} eval_return={eval_return:.2f}')

    global_step += 1
    actor_scheduler.step()
    q_scheduler.step()
    pbar.update(1)

# Final evaluation
final_returns = []
for _ in range(3):
    ret = evaluate()
    final_returns.append(ret)
    obs = envs.reset()
mean_reward = np.mean(final_returns)
std_reward = np.std(final_returns)
print(f'TEST_METRICS mean_reward={mean_reward:.4f} std_reward={std_reward:.4f}')
"
