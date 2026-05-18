"""Custom PEARL experiment launcher for meta-rl task.

This script is FIXED (not editable). It imports CustomContextEncoder
from custom_encoder.py and runs meta-training on the specified environment.
"""
import os
import pathlib
import numpy as np
import click
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

from custom_encoder import CustomContextEncoder


# ── Per-environment configurations ──────────────────────────────────────
ENV_CONFIGS = {
    "cheetah-vel": {
        "env_name": "cheetah-vel",
        "n_train_tasks": 30,
        "n_eval_tasks": 10,
        "env_params": {
            "n_tasks": 40,
            "randomize_tasks": True,
        },
        "algo_params": {
            "num_iterations": 20,
            "num_initial_steps": 2000,
            "num_steps_prior": 400,
            "num_steps_posterior": 0,
            "num_extra_rl_steps_posterior": 600,
            "num_train_steps_per_itr": 600,
            "num_evals": 1,
            "num_steps_per_eval": 600,
            "embedding_batch_size": 100,
            "embedding_mini_batch_size": 100,
        },
    },
    "sparse-point-robot": {
        "env_name": "sparse-point-robot",
        "n_train_tasks": 40,
        "n_eval_tasks": 10,
        "env_params": {
            "n_tasks": 50,
            "randomize_tasks": True,
        },
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
            "discount": 0.90,
            "reward_scale": 100.0,
            "sparse_rewards": 1,
            "kl_lambda": 1.0,
            "num_exp_traj_eval": 5,
        },
    },
    "point-robot": {
        "env_name": "point-robot",
        "n_train_tasks": 40,
        "n_eval_tasks": 10,
        "env_params": {
            "n_tasks": 50,
            "randomize_tasks": True,
        },
        "algo_params": {
            "num_iterations": 20,
            "num_initial_steps": 200,
            "num_tasks_sample": 10,
            "num_steps_prior": 200,
            "num_steps_posterior": 0,
            "num_extra_rl_steps_posterior": 200,
            "num_train_steps_per_itr": 1000,
            "num_evals": 1,
            "num_steps_per_eval": 60,
            "max_path_length": 20,
            "reward_scale": 100.0,
        },
    },
}


def experiment(variant):
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    # Use actual obs shape (oyster envs may override _get_obs without updating obs space)
    sample_obs = env.reset()
    obs_dim = sample_obs.shape[0]
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    latent_dim = variant['latent_size']
    context_encoder_input_dim = (
        2 * obs_dim + action_dim + reward_dim
        if variant['algo_params']['use_next_obs_in_context']
        else obs_dim + action_dim + reward_dim
    )
    context_encoder_output_dim = (
        latent_dim * 2
        if variant['algo_params']['use_information_bottleneck']
        else latent_dim
    )
    net_size = variant['net_size']
    variant['algo_params']['recurrent'] = bool(
        getattr(CustomContextEncoder, 'IS_RECURRENT', False)
    )
    # Recurrent encoders process embedding_batch_size as a single concatenated
    # sequence through an LSTM. The default config (sparse-point-robot=1024)
    # was tuned for permutation-invariant encoders and makes the LSTM ~10x
    # slower than the time budget allows. Cap at 100 for recurrent baselines.
    if variant['algo_params']['recurrent']:
        ap = variant['algo_params']
        cap = 100
        if ap.get('embedding_batch_size', 0) > cap:
            ap['embedding_batch_size'] = cap
        if ap.get('embedding_mini_batch_size', 0) > cap:
            ap['embedding_mini_batch_size'] = cap

    context_encoder = CustomContextEncoder(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    ptu.set_gpu_mode(
        variant['util_params']['use_gpu'],
        variant['util_params']['gpu_id'],
    )
    if ptu.gpu_enabled():
        algorithm.to()

    os.environ['DEBUG'] = str(int(variant['util_params']['debug']))
    exp_id = 'debug' if variant['util_params']['debug'] else None
    experiment_log_dir = setup_logger(
        variant['env_name'],
        variant=variant,
        exp_id=exp_id,
        base_log_dir=variant['util_params']['base_log_dir'],
    )

    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    algorithm.train()


def deep_update_dict(fr, to):
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


@click.command()
@click.option('--env', default='cheetah-vel', type=click.Choice(list(ENV_CONFIGS.keys())))
@click.option('--gpu', default=0)
@click.option('--seed', default=42)
def main(env, gpu, seed):
    variant = default_config
    exp_params = ENV_CONFIGS[env]
    variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    np.random.seed(seed)
    torch.manual_seed(seed)

    experiment(variant)


if __name__ == "__main__":
    main()
