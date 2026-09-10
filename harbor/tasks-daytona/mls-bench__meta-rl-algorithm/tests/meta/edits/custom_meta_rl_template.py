"""Custom meta-RL algorithm template for meta-rl-algorithm task.

FIXED infrastructure (not editable): environment setup, network building blocks,
replay buffers, sampler, evaluation protocol, and outer training loop.
EDITABLE region: CustomMetaRLAgent and CustomMetaRLAlgorithm classes.
"""
import os
import sys
import copy
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, np_ify
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.samplers.util import rollout
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch.sac.policies import MakeDeterministic


# =====================================================================
# FIXED — Per-environment configurations
# =====================================================================
ENV_CONFIGS = {
    "cheetah-vel": {
        "env_name": "cheetah-vel",
        "n_train_tasks": 30,
        "n_eval_tasks": 10,
        "env_params": {"n_tasks": 40, "randomize_tasks": True},
        "algo_params": {
            "num_iterations": 50,
            "num_initial_steps": 2000,
            "num_tasks_sample": 5,
            "num_steps_prior": 400,
            "num_steps_posterior": 0,
            "num_extra_rl_steps_posterior": 600,
            "num_train_steps_per_itr": 600,
            "num_evals": 1,
            "num_steps_per_eval": 600,
            "embedding_batch_size": 100,
            "embedding_mini_batch_size": 100,
            "max_path_length": 200,
            "batch_size": 256,
            "meta_batch": 16,
            "discount": 0.99,
            "reward_scale": 5.0,
            "sparse_rewards": False,
            "num_exp_traj_eval": 1,
        },
    },
    "sparse-point-robot": {
        "env_name": "sparse-point-robot",
        "n_train_tasks": 40,
        "n_eval_tasks": 10,
        "env_params": {"n_tasks": 50, "randomize_tasks": True},
        "algo_params": {
            "num_iterations": 40,
            "num_initial_steps": 200,
            "num_tasks_sample": 10,
            "num_steps_prior": 100,
            "num_steps_posterior": 900,
            "num_extra_rl_steps_posterior": 0,
            "num_train_steps_per_itr": 1000,
            "num_evals": 1,
            "num_steps_per_eval": 400,
            "embedding_batch_size": 1024,
            "embedding_mini_batch_size": 1024,
            "max_path_length": 20,
            "batch_size": 256,
            "meta_batch": 16,
            "discount": 0.90,
            "reward_scale": 100.0,
            "sparse_rewards": True,
            "kl_lambda": 1.0,
            "num_exp_traj_eval": 5,
        },
    },
    "point-robot": {
        "env_name": "point-robot",
        "n_train_tasks": 40,
        "n_eval_tasks": 10,
        "env_params": {"n_tasks": 50, "randomize_tasks": True},
        "algo_params": {
            "num_iterations": 30,
            "num_initial_steps": 200,
            "num_tasks_sample": 10,
            "num_steps_prior": 200,
            "num_steps_posterior": 0,
            "num_extra_rl_steps_posterior": 200,
            "num_train_steps_per_itr": 1000,
            "num_evals": 1,
            "num_steps_per_eval": 60,
            "embedding_batch_size": 100,
            "embedding_mini_batch_size": 100,
            "max_path_length": 20,
            "batch_size": 256,
            "meta_batch": 16,
            "discount": 0.99,
            "reward_scale": 100.0,
            "sparse_rewards": False,
            "num_exp_traj_eval": 1,
        },
    },
}


# =====================================================================
# FIXED — Network building blocks
# =====================================================================
def build_mlp(input_dim, output_dim, hidden_dim=200, n_layers=3, init_w=3e-3):
    """Build a simple MLP with ReLU activations."""
    layers = []
    in_dim = input_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    last = nn.Linear(in_dim, output_dim)
    last.weight.data.uniform_(-init_w, init_w)
    last.bias.data.uniform_(-init_w, init_w)
    layers.append(last)
    return nn.Sequential(*layers)


def build_policy(obs_dim, action_dim, latent_dim, net_size=300):
    """Build a TanhGaussianPolicy conditioned on (obs, z)."""
    return TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )


def build_qf(obs_dim, action_dim, latent_dim, net_size=300):
    """Build a FlattenMlp Q-function: Q(obs, action, z)."""
    return FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )


def build_vf(obs_dim, latent_dim, net_size=300):
    """Build a FlattenMlp V-function: V(obs, z)."""
    return FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )


# =====================================================================
# FIXED — Replay buffer helpers
# =====================================================================
def create_replay_buffers(env, tasks, max_size=1000000):
    """Create two MultiTaskReplayBuffers: one for RL, one for encoder."""
    replay_buffer = MultiTaskReplayBuffer(max_size, env, tasks)
    enc_replay_buffer = MultiTaskReplayBuffer(max_size, env, tasks)
    return replay_buffer, enc_replay_buffer


def unpack_batch(batch, sparse_reward=False):
    """Unpack a batch dict to [obs, actions, rewards, next_obs, terminals] with leading task dim."""
    o = batch['observations'][None, ...]
    a = batch['actions'][None, ...]
    if sparse_reward:
        r = batch['sparse_rewards'][None, ...]
    else:
        r = batch['rewards'][None, ...]
    no = batch['next_observations'][None, ...]
    t = batch['terminals'][None, ...]
    return [o, a, r, no, t]


def sample_context_from_buffer(enc_replay_buffer, indices, embedding_batch_size,
                               sparse_rewards=False, use_next_obs_in_context=False):
    """Sample context batch from encoder replay buffer.

    Returns tensor of shape (num_tasks, embedding_batch_size, context_dim).
    """
    if not hasattr(indices, '__iter__'):
        indices = [indices]
    batches = [ptu.np_to_pytorch_batch(enc_replay_buffer.random_batch(idx, batch_size=embedding_batch_size))
               for idx in indices]
    context = [unpack_batch(batch, sparse_reward=sparse_rewards) for batch in batches]
    # group like elements together
    context = [[x[i] for x in context] for i in range(len(context[0]))]
    context = [torch.cat(x, dim=0) for x in context]
    # full context consists of [obs, act, rewards, next_obs, terms]
    if use_next_obs_in_context:
        context = torch.cat(context[:-1], dim=2)
    else:
        context = torch.cat(context[:-2], dim=2)
    return context


def sample_sac_batch(replay_buffer, indices, batch_size):
    """Sample RL training batch from replay buffer.

    Returns [obs, actions, rewards, next_obs, terminals] each (num_tasks, batch_size, dim).
    """
    batches = [ptu.np_to_pytorch_batch(replay_buffer.random_batch(idx, batch_size=batch_size))
               for idx in indices]
    unpacked = [unpack_batch(batch) for batch in batches]
    unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
    unpacked = [torch.cat(x, dim=0) for x in unpacked]
    return unpacked


# =====================================================================
# FIXED — Evaluation protocol
# (Faithfully adapted from oyster's rl_algorithm.py evaluate() and
#  _do_eval() / collect_paths(), using agent.adapt() instead of
#  agent.infer_posterior(agent.context))
# =====================================================================
def collect_eval_paths(agent, env, sampler, task_idx, config, epoch=0, run=0):
    """Collect evaluation paths for a single task, using online adaptation.

    Follows oyster's collect_paths(): collect trajectories with accum_context=True,
    after num_exp_traj_eval trajectories call agent.adapt() then continue
    with deterministic policy.
    """
    env.reset_task(task_idx)
    agent.clear_context()

    eval_deterministic = True
    num_steps_per_eval = config['num_steps_per_eval']
    num_exp_traj_eval = config.get('num_exp_traj_eval', 1)
    max_path_length = config['max_path_length']
    sparse_rewards = config.get('sparse_rewards', False)

    paths = []
    num_transitions = 0
    num_trajs = 0
    while num_transitions < num_steps_per_eval:
        # Use deterministic policy after adaptation
        policy = MakeDeterministic(agent) if eval_deterministic else agent
        path = rollout(env, policy, max_path_length=max_path_length, accum_context=True)
        # Save the latent context z
        path['context'] = agent.z.detach().cpu().numpy()
        paths.append(path)
        num_transitions += len(path['observations'])
        num_trajs += 1
        if num_trajs >= num_exp_traj_eval:
            agent.adapt()

    if sparse_rewards:
        for p in paths:
            sparse_rew = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
            p['rewards'] = sparse_rew

    return paths


def do_eval(agent, env, sampler, task_indices, config, epoch=0):
    """Evaluate on a set of tasks, returning final and online returns.

    Follows oyster's _do_eval().
    """
    num_evals = config.get('num_evals', 1)
    final_returns = []
    online_returns = []
    for idx in task_indices:
        all_rets = []
        for r in range(num_evals):
            paths = collect_eval_paths(agent, env, sampler, idx, config, epoch, r)
            all_rets.append([get_average_returns([p]) for p in paths])
        final_returns.append(np.mean([a[-1] for a in all_rets]))
        n = min([len(a) for a in all_rets])
        all_rets = [a[:n] for a in all_rets]
        all_rets = np.mean(np.stack(all_rets), axis=0)
        online_returns.append(all_rets)
    n = min([len(t) for t in online_returns])
    online_returns = [t[:n] for t in online_returns]
    return final_returns, online_returns


def get_average_returns(paths):
    """Compute average return across paths."""
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def run_evaluation(agent, env, train_tasks, eval_tasks, sampler, config, epoch):
    """Full evaluation: train tasks + test tasks.

    Follows oyster's evaluate() method. Prints TRAIN_METRICS and TEST_METRICS.
    """
    # --- Evaluate on a subset of train tasks ---
    indices = np.random.choice(train_tasks, len(eval_tasks))

    # Online evaluation on train tasks
    train_final_returns, train_online_returns = do_eval(
        agent, env, sampler, indices, config, epoch
    )
    avg_train_return = np.mean(train_final_returns)

    # --- Evaluate on test tasks ---
    test_final_returns, test_online_returns = do_eval(
        agent, env, sampler, eval_tasks, config, epoch
    )
    avg_test_return = np.mean(test_final_returns)

    print(f'TRAIN_METRICS iteration={epoch} avg_train_return={avg_train_return:.4f}', flush=True)
    print(f'TEST_METRICS iteration={epoch} meta_test_return={avg_test_return:.4f}', flush=True)

    return avg_train_return, avg_test_return


# =====================================================================
# FIXED — Data collection helpers
# =====================================================================
def collect_data(agent, env, sampler, replay_buffer, enc_replay_buffer,
                 task_idx, num_samples, resample_z_rate, update_posterior_rate,
                 add_to_enc_buffer=True, config=None):
    """Collect data for a single task, following oyster's collect_data().

    Uses sampler to get trajectories, adds to replay buffers.
    """
    agent.clear_context()
    num_transitions = 0
    while num_transitions < num_samples:
        paths, n_samples = sampler.obtain_samples(
            max_samples=num_samples - num_transitions,
            max_trajs=update_posterior_rate,
            accum_context=False,
            resample=resample_z_rate,
        )
        num_transitions += n_samples
        replay_buffer.add_paths(task_idx, paths)
        if add_to_enc_buffer:
            enc_replay_buffer.add_paths(task_idx, paths)
        if update_posterior_rate != np.inf:
            # Sample context from buffer and adapt
            sparse_rewards = config.get('sparse_rewards', False) if config else False
            use_next_obs = config.get('use_next_obs_in_context', False) if config else False
            ctx = sample_context_from_buffer(
                enc_replay_buffer, task_idx,
                config.get('embedding_batch_size', 100),
                sparse_rewards=sparse_rewards,
                use_next_obs_in_context=use_next_obs,
            )
            agent.infer_posterior(ctx)


# =====================================================================
# EDITABLE — Custom imports
# =====================================================================


# =====================================================================
# EDITABLE — Custom Meta-RL Agent
# =====================================================================
class CustomMetaRLAgent(nn.Module):
    """Custom meta-RL agent.

    Must implement:
      - get_action(obs, deterministic=False) -> (action_np, agent_info)
      - update_context(transition_tuple) -> None
      - adapt() -> None  (called after context collection; performs task inference)
      - clear_context(num_tasks=1) -> None
      - infer_posterior(context_tensor) -> None  (for training-time context encoding)
      - context: property or attribute returning collected context
      - z: tensor attribute for latent task variable
      - networks: list of nn.Module for GPU transfer and param counting
    """

    def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context

        # Simple MLP policy (no task conditioning, placeholder)
        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)

        # Latent variable z (dummy zeros for this placeholder)
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self._context = None

    def clear_context(self, num_tasks=1):
        self.z = ptu.zeros(num_tasks, self.latent_dim)
        self._context = None

    # alias used by InPlacePathSampler via sample_z
    def clear_z(self, num_tasks=1):
        self.clear_context(num_tasks)

    def sample_z(self):
        pass

    @property
    def context(self):
        return self._context

    def update_context(self, inputs):
        """Append single transition to context. inputs = [obs, action, reward, next_obs, done, env_info]."""
        o, a, r, no, d, info = inputs
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self._context is None:
            self._context = data
        else:
            self._context = torch.cat([self._context, data], dim=1)

    def adapt(self):
        """Perform task inference from collected context. Override this."""
        pass

    def infer_posterior(self, context):
        """Encode context tensor (from replay buffer) for training. Override this."""
        pass

    def get_action(self, obs, deterministic=False):
        z = self.z
        obs_t = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs_t, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def detach_z(self):
        self.z = self.z.detach()

    @property
    def networks(self):
        return [self.policy]


class CustomMetaRLAlgorithm:
    """Custom meta-RL training algorithm.

    Must implement:
      - __init__(agent, env, train_tasks, eval_tasks, replay_buffer, enc_replay_buffer, config)
      - collect_initial_data() -> None
      - train_iteration(iteration_idx) -> dict  (one meta-training iteration)
      - agent: attribute referencing the CustomMetaRLAgent
    """

    def __init__(self, agent, env, train_tasks, eval_tasks,
                 replay_buffer, enc_replay_buffer, config):
        self.agent = agent
        self.env = env
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.replay_buffer = replay_buffer
        self.enc_replay_buffer = enc_replay_buffer
        self.config = config

        self.sampler = InPlacePathSampler(
            env=env, policy=agent,
            max_path_length=config['max_path_length'],
        )

    def collect_initial_data(self):
        """Collect initial exploration data for all training tasks."""
        num_initial_steps = self.config.get('num_initial_steps', 200)
        for idx in self.train_tasks:
            self.env.reset_task(idx)
            collect_data(
                self.agent, self.env, self.sampler,
                self.replay_buffer, self.enc_replay_buffer,
                idx, num_initial_steps, 1, np.inf,
                add_to_enc_buffer=True, config=self.config,
            )

    def train_iteration(self, iteration_idx):
        """One meta-training iteration: collect data + gradient updates."""
        return {}

    @property
    def networks(self):
        """Return all networks for GPU transfer."""
        return self.agent.networks


# =====================================================================
# FIXED — Main entry point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='Custom Meta-RL Algorithm')
    parser.add_argument('--env', type=str, default='cheetah-vel',
                        choices=list(ENV_CONFIGS.keys()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Environment config
    env_cfg = ENV_CONFIGS[args.env]
    algo_params = env_cfg['algo_params']

    # Create environment
    env = NormalizedBoxEnv(ENVS[env_cfg['env_name']](**env_cfg['env_params']))
    tasks = env.get_all_task_idx()
    train_tasks = list(tasks[:env_cfg['n_train_tasks']])
    eval_tasks = list(tasks[-env_cfg['n_eval_tasks']:])

    # Dimensions
    sample_obs = env.reset()
    obs_dim = sample_obs.shape[0]
    action_dim = int(np.prod(env.action_space.shape))

    # GPU
    ptu.set_gpu_mode(True, args.gpu)

    # Create agent
    agent = CustomMetaRLAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=5,
        net_size=300,
        reward_dim=1,
        use_next_obs_in_context=False,
        sparse_rewards=algo_params.get('sparse_rewards', False),
    )

    # Create replay buffers
    replay_buffer, enc_replay_buffer = create_replay_buffers(env, train_tasks)

    # Create algorithm
    algorithm = CustomMetaRLAlgorithm(
        agent=agent,
        env=env,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        replay_buffer=replay_buffer,
        enc_replay_buffer=enc_replay_buffer,
        config=algo_params,
    )

    # Move to GPU
    if ptu.gpu_enabled():
        for net in algorithm.networks:
            net.to(ptu.device)
        agent.to(ptu.device)

    # Create sampler for evaluation
    eval_sampler = InPlacePathSampler(
        env=env, policy=agent,
        max_path_length=algo_params['max_path_length'],
    )

    # Collect initial data
    print('Collecting initial pool of data...', flush=True)
    algorithm.collect_initial_data()

    # Meta-training loop
    num_iterations = algo_params['num_iterations']
    for it in range(num_iterations):
        # Training mode
        for net in algorithm.networks:
            net.train(True)
        agent.train(True)

        # Train one iteration
        train_info = algorithm.train_iteration(it)

        # Eval mode
        for net in algorithm.networks:
            net.train(False)
        agent.train(False)

        # Evaluation
        run_evaluation(
            agent, env, train_tasks, eval_tasks,
            eval_sampler, algo_params, it,
        )

    print('Training complete.', flush=True)


if __name__ == '__main__':
    main()
