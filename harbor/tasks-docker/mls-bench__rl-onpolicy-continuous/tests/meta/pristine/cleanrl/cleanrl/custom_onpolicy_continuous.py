# Custom on-policy continuous RL algorithm for MLS-Bench
#
# FIXED sections: config, env, utilities, network architecture, training loop.
# EDITABLE section: get_action_and_value method and compute_losses function.
import copy
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
from torch.distributions.normal import Normal


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
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    eval_freq: int = 50000
    """evaluation frequency (timesteps)"""
    eval_episodes: int = 10
    """number of evaluation episodes"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# =====================================================================
# FIXED: Environment setup
# =====================================================================
def make_env(env_id, idx, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


# =====================================================================
# FIXED: Utilities
# =====================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@torch.no_grad()
def eval_agent(env_id, agent, device, n_episodes, seed, gamma=0.99, obs_rms=None):
    """Evaluate agent for n_episodes using same env wrappers as training.
    If obs_rms provided, copies training normalization stats to eval env.
    Returns array of raw (un-normalized) episode returns."""
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, gamma)]
    )
    if obs_rms is not None:
        _eval_env = eval_envs.envs[0]
        while hasattr(_eval_env, 'env'):
            if isinstance(_eval_env, gym.wrappers.NormalizeObservation):
                _eval_env.obs_rms = copy.deepcopy(obs_rms)
                break
            _eval_env = _eval_env.env
    episode_rewards = []
    obs, _ = eval_envs.reset(seed=seed)
    while len(episode_rewards) < n_episodes:
        obs_t = torch.Tensor(obs).to(device)
        action, _, _, _ = agent.get_action_and_value(obs_t)
        obs, _, _, _, infos = eval_envs.step(action.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_rewards.append(float(info["episode"]["r"]))
    eval_envs.close()
    return np.asarray(episode_rewards)


# =====================================================================
# FIXED: Agent architecture (network capacity is fixed)
# =====================================================================
class Agent(nn.Module):
    """On-policy actor-critic agent.

    Architecture is FIXED (2x64 MLP for both actor and critic).
    Only get_action_and_value is editable — this is where algorithmic
    innovation happens (distribution type, squashing, etc.).
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        h = 64
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs):
        return self.critic(obs)

    # =================================================================
    # EDITABLE: get_action_and_value and compute_losses
    # =================================================================
    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    """Compute policy and value losses for a minibatch.

    Args:
        agent: the Agent instance
        mb_obs: minibatch observations
        mb_actions: minibatch actions
        mb_logprobs: minibatch old log probabilities
        mb_advantages: minibatch advantages
        mb_returns: minibatch returns
        mb_values: minibatch old values
        args: hyperparameters

    Returns:
        (loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac)
    """
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Policy loss -- placeholder
    pg_loss = (-mb_advantages * ratio).mean()

    # Value loss -- placeholder
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac


# =====================================================================
# FIXED: Training loop
# =====================================================================
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.prod(envs.single_action_space.shape)

    agent = Agent(obs_dim, action_dim).to(device)

    # Parameter count guard: prevent network capacity hacking
    _param_count = sum(p.numel() for p in agent.parameters())
    _expected_params = (obs_dim * 64 + 64) + (64 * 64 + 64) + (64 * 1 + 1) \
                     + (obs_dim * 64 + 64) + (64 * 64 + 64) + (64 * action_dim + action_dim) \
                     + action_dim  # actor_logstd
    assert _param_count == _expected_params, (
        f"Parameter count mismatch: got {_param_count}, expected {_expected_params}. "
        f"Do not modify the network architecture — only get_action_and_value and compute_losses are editable."
    )

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.norm_adv:
                    mb_advantages = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)
                else:
                    mb_advantages = b_advantages[mb_inds]

                loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = compute_losses(
                    agent, b_obs[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds],
                    mb_advantages, b_returns[mb_inds], b_values[mb_inds], args,
                )
                clipfracs.append(clipfrac)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Log training metrics
        print(
            f"TRAIN_METRICS step={global_step} "
            f"pg_loss={pg_loss.item():.4f} vf_loss={v_loss.item():.4f} "
            f"entropy={entropy_loss.item():.4f} approx_kl={approx_kl.item():.4f} "
            f"clipfrac={np.mean(clipfracs):.4f}",
            flush=True,
        )

        # Evaluation
        if global_step % args.eval_freq < args.batch_size:
            _env = envs.envs[0]
            _obs_rms = None
            while hasattr(_env, 'env'):
                if isinstance(_env, gym.wrappers.NormalizeObservation):
                    _obs_rms = _env.obs_rms
                    break
                _env = _env.env
            eval_returns = eval_agent(
                args.env_id, agent, device,
                n_episodes=args.eval_episodes, seed=args.seed + 1000,
                gamma=args.gamma, obs_rms=_obs_rms,
            )
            mean_return = eval_returns.mean()
            print(f"Eval episodic_return: {mean_return:.2f}", flush=True)

    envs.close()
