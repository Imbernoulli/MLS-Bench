"""Custom off-policy RL algorithm for HumanoidBench locomotion tasks.

This script is adapted from FastTD3's training pipeline. The EDITABLE section
contains the full algorithm: Actor, Critic, update functions, and exploration
strategy. The FIXED sections handle environment setup, evaluation, replay buffer
infrastructure, and metric printing.

The agent should design a sample-efficient off-policy (or hybrid) RL algorithm
that outperforms FastTD3, FastSAC, and PPO on humanoid locomotion tasks.
"""

import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"

import argparse
import random
import time
import math

import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict

# Import utilities from FastTD3
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fast_td3_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    mark_step,
)

torch.set_float32_matmul_precision("high")


# ═══════════════════════════════════════════════════════════════════════
# ██ EDITABLE SECTION START — Design your off-policy RL algorithm here
# ═══════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """Deterministic actor network.

    Design a better actor architecture for sample-efficient learning.
    Consider: normalization layers, activation functions, initialization,
    residual connections, spectral normalization, etc.
    """
    def __init__(self, n_obs, n_act, num_envs, device, hidden_dim=512,
                 init_scale=0.01, std_min=0.001, std_max=0.4):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=device),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs
        self.device_ = device

    def forward(self, obs):
        x = self.net(obs)
        return self.fc_mu(x)

    def explore(self, obs, dones=None, deterministic=False):
        if dones is not None and dones.sum() > 0:
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min) + self.std_min
            )
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )
        act = self(obs)
        if deterministic:
            return act
        noise = torch.randn_like(act) * self.noise_scales
        return act + noise


class DistributionalQNetwork(nn.Module):
    """Distributional Q-network for value estimation.

    Design a better critic architecture for more accurate value estimation.
    Consider: distributional RL atoms, network width/depth, normalization, etc.
    """
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], 1)
        return self.net(x)

    def projection(self, obs, actions, rewards, bootstrap, discount, q_support, device):
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()
        is_int = (l == u)
        l_mask = is_int & (l > 0)
        u_mask = is_int & (l == 0)
        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)
        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1).expand(batch_size, self.num_atoms).long()
        )
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        return proj_dist


class Critic(nn.Module):
    """Twin distributional critic with clipped double Q-learning.

    Design improvements to the critic: number of Q-networks, ensemble methods,
    target computation strategy, etc.
    """
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
        self.qnet2 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))
        self.device = device

    def forward(self, obs, actions):
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(self, obs, actions, rewards, bootstrap, discount):
        q1_proj = self.qnet1.projection(obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device)
        q2_proj = self.qnet2.projection(obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device)
        return q1_proj, q2_proj

    def get_value(self, probs):
        return torch.sum(probs * self.q_support, dim=1)


def build_algorithm(n_obs, n_act, num_envs, device, args):
    """Build all algorithm components: actor, critic, optimizers, schedulers.

    This function creates and returns all components needed for training.
    You can modify hyperparameters, add new components (e.g., entropy tuning,
    auxiliary networks, prioritized replay modifications), etc.

    Returns a dict with keys:
        actor, critic, critic_target, actor_optimizer, critic_optimizer,
        actor_scheduler, critic_scheduler, and any additional components.
    """
    actor = Actor(
        n_obs=n_obs, n_act=n_act, num_envs=num_envs, device=device,
        hidden_dim=args.actor_hidden_dim, init_scale=args.init_scale,
        std_min=args.std_min, std_max=args.std_max,
    )
    critic = Critic(
        n_obs=n_obs, n_act=n_act, num_atoms=args.num_atoms,
        v_min=args.v_min, v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim, device=device,
    )
    critic_target = Critic(
        n_obs=n_obs, n_act=n_act, num_atoms=args.num_atoms,
        v_min=args.v_min, v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim, device=device,
    )
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.AdamW(
        actor.parameters(),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    critic_optimizer = optim.AdamW(
        critic.parameters(),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer, T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        critic_optimizer, T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )

    return {
        "actor": actor,
        "critic": critic,
        "critic_target": critic_target,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "actor_scheduler": actor_scheduler,
        "critic_scheduler": critic_scheduler,
    }


def update_critic(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype):
    """Update the critic network(s).

    Modify the critic loss, target computation, or add auxiliary objectives.
    Consider: different distributional RL losses, n-step returns, reward shaping, etc.
    """
    actor = components["actor"]
    critic = components["critic"]
    critic_target = components["critic_target"]
    critic_optimizer = components["critic_optimizer"]

    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
        observations = data["observations"]
        next_observations = data["next"]["observations"]
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        clipped_noise = torch.randn_like(actions)
        clipped_noise = clipped_noise.mul(args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
        next_state_actions = (actor(next_observations) + clipped_noise).clamp(-1.0, 1.0)
        discount = args.gamma ** data["next"]["effective_n_steps"]

        with torch.no_grad():
            qf1_next_proj, qf2_next_proj = critic_target.projection(
                next_observations, next_state_actions, rewards, bootstrap, discount,
            )
            qf1_next_val = critic_target.get_value(qf1_next_proj)
            qf2_next_val = critic_target.get_value(qf2_next_proj)
            qf_next_dist = torch.where(
                qf1_next_val.unsqueeze(1) < qf2_next_val.unsqueeze(1),
                qf1_next_proj, qf2_next_proj,
            )
            qf1_next_dist = qf2_next_dist = qf_next_dist

        qf1, qf2 = critic(observations, actions)
        qf1_loss = -torch.sum(qf1_next_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
        qf2_loss = -torch.sum(qf2_next_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
        qf_loss = qf1_loss + qf2_loss

    critic_optimizer.zero_grad(set_to_none=True)
    scaler.scale(qf_loss).backward()
    scaler.unscale_(critic_optimizer)
    scaler.step(critic_optimizer)
    scaler.update()

    return {"qf_loss": qf_loss.detach(), "qf1_next_val": qf1_next_val}


def update_actor(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype):
    """Update the actor (policy) network.

    Modify the policy objective, add entropy regularization, or implement
    other policy improvement techniques.
    """
    actor = components["actor"]
    critic = components["critic"]
    actor_optimizer = components["actor_optimizer"]

    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
        qf1, qf2 = critic(data["observations"], actor(data["observations"]))
        qf1_value = critic.get_value(F.softmax(qf1, dim=1))
        qf2_value = critic.get_value(F.softmax(qf2, dim=1))
        qf_value = torch.minimum(qf1_value, qf2_value)
        actor_loss = -qf_value.mean()

    actor_optimizer.zero_grad(set_to_none=True)
    scaler.scale(actor_loss).backward()
    scaler.unscale_(actor_optimizer)
    scaler.step(actor_optimizer)
    scaler.update()

    return {"actor_loss": actor_loss.detach()}


@torch.no_grad()
def soft_update(src, tgt, tau):
    """Soft update target network parameters."""
    src_ps = [p.data for p in src.parameters()]
    tgt_ps = [p.data for p in tgt.parameters()]
    torch._foreach_mul_(tgt_ps, 1.0 - tau)
    torch._foreach_add_(tgt_ps, src_ps, alpha=tau)


# ═══════════════════════════════════════════════════════════════════════
# ██ EDITABLE SECTION END
# ═══════════════════════════════════════════════════════════════════════


# ─── FIXED: Argument parsing ───────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="h1hand-stand-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--buffer_size", type=int, default=1024 * 50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--policy_noise", type=float, default=0.001)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--learning_starts", type=int, default=10)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--num_updates", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=5000)
    # Network
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--critic_hidden_dim", type=int, default=1024)
    parser.add_argument("--init_scale", type=float, default=0.01)
    parser.add_argument("--num_atoms", type=int, default=101)
    parser.add_argument("--v_min", type=float, default=-250.0)
    parser.add_argument("--v_max", type=float, default=250.0)
    # Exploration
    parser.add_argument("--std_min", type=float, default=0.001)
    parser.add_argument("--std_max", type=float, default=0.4)
    # Optimizer
    parser.add_argument("--actor_learning_rate", type=float, default=3e-4)
    parser.add_argument("--critic_learning_rate", type=float, default=3e-4)
    parser.add_argument("--actor_learning_rate_end", type=float, default=3e-4)
    parser.add_argument("--critic_learning_rate_end", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    # AMP
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--amp_dtype", type=str, default="bf16")
    # Misc
    parser.add_argument("--obs_normalization", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--device_rank", type=int, default=0)
    return parser.parse_args()


# ─── FIXED: Main training loop ────────────────────────────────────────
def main():
    args = get_args()
    print(f"Args: {args}")

    amp_enabled = args.amp and torch.cuda.is_available()
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(f"cuda:{args.device_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ─── Environment setup (FIXED) ────────────────────────────────────
    from environments.humanoid_bench_env import HumanoidBenchEnv
    envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
    eval_envs = envs

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = nn.Identity()

    # ─── Build algorithm (EDITABLE function) ──────────────────────────
    components = build_algorithm(n_obs, n_act, args.num_envs, device, args)
    actor = components["actor"]
    critic = components["critic"]
    critic_target = components["critic_target"]

    # ─── Replay buffer (FIXED) ────────────────────────────────────────
    rb = SimpleReplayBuffer(
        n_env=args.num_envs, buffer_size=args.buffer_size,
        n_obs=n_obs, n_act=n_act, n_critic_obs=n_obs,
        asymmetric_obs=False, n_steps=args.num_steps,
        gamma=args.gamma, device=device,
    )

    # ─── Compile (FIXED) ──────────────────────────────────────────────
    policy = actor.explore
    normalize_obs = obs_normalizer.forward
    if args.compile:
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)

    # ─── Evaluation (FIXED) ───────────────────────────────────────────
    def evaluate():
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
        obs = eval_envs.reset()
        for i in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                obs_norm = normalize_obs(obs, update=False)
                actions = actor(obs_norm)
            next_obs, rewards, dones, infos = eval_envs.step(actions.float())
            episode_returns = torch.where(~done_masks, episode_returns + rewards, episode_returns)
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs
        return episode_returns.mean().item()

    # ─── Training loop (FIXED structure, calls EDITABLE update functions) ─
    obs = envs.reset()
    dones = None
    global_step = 0
    pbar = tqdm.tqdm(total=args.total_timesteps)
    start_time = None
    measure_burnin = 3
    eval_results = []

    while global_step < args.total_timesteps:
        mark_step()

        if start_time is None and global_step >= measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_start = global_step

        # Collect experience
        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]

        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        transition = TensorDict({
            "observations": obs,
            "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
            "next": {
                "observations": true_next_obs,
                "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                "truncations": truncations.long(),
                "dones": dones.long(),
            },
        }, batch_size=(envs.num_envs,), device=device)
        rb.extend(transition)
        obs = next_obs

        # Update
        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(data["next"]["observations"])

                critic_info = update_critic(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype)

                should_update_actor = (
                    (args.num_updates > 1 and i % args.policy_frequency == 1)
                    or (args.num_updates == 1 and global_step % args.policy_frequency == 0)
                )
                if should_update_actor:
                    actor_info = update_actor(data, components, args, scaler, amp_enabled, amp_device_type, amp_dtype)

                soft_update(critic, critic_target, args.tau)

            # Logging and evaluation
            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_start) / (time.time() - start_time)
                pbar.set_description(f"{speed:.1f} sps")

                if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                    eval_return = evaluate()
                    eval_results.append((global_step, eval_return))
                    obs = envs.reset()  # Reset after eval for HumanoidBench

                    # Print training metrics
                    print(f"TRAIN_METRICS step={global_step} eval_return={eval_return:.2f}")

        global_step += 1
        components["actor_scheduler"].step()
        components["critic_scheduler"].step()
        pbar.update(1)

    # Final evaluation
    final_returns = []
    for _ in range(3):
        ret = evaluate()
        final_returns.append(ret)
        obs = envs.reset()
    mean_reward = np.mean(final_returns)
    std_reward = np.std(final_returns)

    print(f"TEST_METRICS mean_reward={mean_reward:.4f} std_reward={std_reward:.4f}")


if __name__ == "__main__":
    main()
