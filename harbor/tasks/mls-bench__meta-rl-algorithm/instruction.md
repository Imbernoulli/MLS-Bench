# MLS-Bench: meta-rl-algorithm

# Meta-RL Algorithm Design

## Research Question
Design a complete meta-reinforcement learning algorithm for fast
adaptation to unseen tasks from limited interaction data. You must
implement both the **agent** (how to encode context and condition the
policy on it) and the **training algorithm** (how to meta-train the agent
across tasks).

## Background
Meta-RL trains across a distribution of tasks so that at test time an
agent can quickly adapt to a new task from a few interactions. The key
design choices are:

1. **Task inference** — how to encode past experience (context) into a
   compact task representation.
2. **Policy conditioning** — how to condition the policy on this task
   representation.
3. **Meta-training** — how to optimize the agent across tasks so that it
   generalizes to held-out tasks.

Existing approaches span this design space. PEARL (Rakelly et al.,
"Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic
Context Variables", arXiv:1903.08254, ICML 2019) uses a probabilistic
encoder with product-of-Gaussians aggregation and an SAC backbone. FOCAL
(Li et al., "FOCAL: Efficient Fully-Offline Meta-Reinforcement Learning
via Distance Metric Learning and Behavior Regularization",
arXiv:2010.01112, ICLR 2021) uses a deterministic encoder trained with a
contrastive distance metric. VariBAD (Zintgraf et al., "VariBAD: A Very
Good Method for Bayes-Adaptive Deep RL via Meta-Learning",
arXiv:1910.08348, ICLR 2020) uses a recurrent encoder trained with reward
and transition reconstruction, and conditions the policy on the recurrent
posterior.

You will modify the `CustomMetaRLAgent` and `CustomMetaRLAlgorithm`
classes in `custom_meta_rl.py`. The template provides fixed
infrastructure (environment setup, evaluation, replay buffers, trajectory
sampler, network building blocks) — you design the algorithm.

## Agent Interface (`CustomMetaRLAgent`)
Your agent must implement:
- `get_action(obs, deterministic=False) -> (action_np, agent_info)` —
  sample an action conditioned on the current task belief.
- `update_context(transition_tuple) -> None` — accumulate online
  experience (called during rollout).
- `adapt() -> None` — perform task inference from collected context
  (called after exploration).
- `clear_context(num_tasks=1) -> None` — reset context and task belief.
- `infer_posterior(context_tensor) -> None` — encode context drawn from
  the replay buffer (used during training).
- `context` property — the collected context.
- `z` attribute — latent task variable tensor.
- `networks` property — list of `nn.Module` for GPU transfer.

## Algorithm Interface (`CustomMetaRLAlgorithm`)
Your algorithm must implement:
- `collect_initial_data()` — gather initial exploration data for all
  training tasks.
- `train_iteration(iteration_idx) -> dict` — one meta-training iteration
  (data collection + gradient updates).
- `networks` property — all networks for GPU transfer.

## Available Utilities
The template provides these fixed utilities:
- `build_mlp(input_dim, output_dim, hidden_dim, n_layers)` — simple MLP.
- `build_policy(obs_dim, action_dim, latent_dim, net_size)` —
  TanhGaussian policy.
- `build_qf(obs_dim, action_dim, latent_dim, net_size)` — Q-function.
- `build_vf(obs_dim, latent_dim, net_size)` — V-function.
- `create_replay_buffers(env, tasks)` — replay buffer pair.
- `sample_context_from_buffer(enc_replay_buffer, indices, batch_size,
  ...)` — sample context.
- `sample_sac_batch(replay_buffer, indices, batch_size)` — sample an RL
  batch.
- `collect_data(agent, env, sampler, replay_buffer, enc_replay_buffer,
  ...)` — collect trajectories.
- `InPlacePathSampler` from rlkit — trajectory sampler.

## Environments
Three meta-RL task families with different challenges:

1. **Half-Cheetah Velocity** (`cheetah-vel`) — 30 train / 10 test tasks,
   target velocities in `[0, 3]` m/s. Obs dim 20, action dim 6. Dense
   reward from velocity matching. High-dim observations require strong
   encoding.

2. **Sparse Point Robot** (`sparse-point-robot`) — 40 train / 10 test
   tasks. Goals on a half-circle, sparse reward (+1 near goal, 0
   otherwise). Obs dim 2, action dim 2. Sparse reward makes inference
   particularly hard.

3. **Point Robot** (`point-robot`) — 40 train / 10 test tasks. Goals in
   `[-1, 1]^2`. Dense reward (negative L2 distance). Obs dim 2, action
   dim 2. Tests basic meta-learning quality.

## Evaluation
Performance is measured by `meta_test_return` on each environment:
average return on held-out test tasks after meta-training. The evaluation
protocol collects exploration trajectories, calls `agent.adapt()`, then
evaluates with a deterministic policy. Higher is better.

## Key Design Dimensions
- **Context encoding** — permutation-invariant (MLP + aggregation),
  sequential (RNN/GRU), or attention-based.
- **Task variable** — probabilistic (with an information bottleneck) vs
  deterministic.
- **Encoder loss** — KL divergence, contrastive, reward prediction, or
  reconstruction.
- **RL algorithm** — SAC variants, on-policy gradient, or alternatives.

## Note on Training Budget
This task intentionally uses a short fixed meta-training budget (20
outer iterations) to keep wall time per environment near 1 hour. This is
much shorter than the 500+ iteration budgets in the PEARL/VariBAD/FOCAL
papers, so absolute returns are not comparable to those papers; only
relative ordering within this benchmark is meaningful.

On `sparse-point-robot`, returns of 0 indicate that no goal was reached
in the budget rather than algorithmic failure, since the reward is
binary.

The companion [`meta-rl`](../meta-rl/task_description.md) task uses the
same budget convention.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/oyster/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `oyster/custom_meta_rl.py`
- editable lines **357–494**


Other files you may **read** for context (do not modify):
- `oyster/rlkit/torch/networks.py`
- `oyster/rlkit/torch/sac/policies.py`
- `oyster/configs/default.py`


## Readable Context


### `oyster/custom_meta_rl.py`  [EDITABLE — lines 357–494 only]

```python
     1: """Custom meta-RL algorithm template for meta-rl-algorithm task.
     2: 
     3: FIXED infrastructure (not editable): environment setup, network building blocks,
     4: replay buffers, sampler, evaluation protocol, and outer training loop.
     5: EDITABLE region: CustomMetaRLAgent and CustomMetaRLAlgorithm classes.
     6: """
     7: import os
     8: import sys
     9: import copy
    10: import argparse
    11: import numpy as np
    12: 
    13: import torch
    14: import torch.nn as nn
    15: import torch.nn.functional as F
    16: import torch.optim as optim
    17: 
    18: import rlkit.torch.pytorch_util as ptu
    19: from rlkit.torch.core import PyTorchModule, np_ify
    20: from rlkit.torch.networks import FlattenMlp
    21: from rlkit.torch.sac.policies import TanhGaussianPolicy
    22: from rlkit.envs import ENVS
    23: from rlkit.envs.wrappers import NormalizedBoxEnv
    24: from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
    25: from rlkit.samplers.util import rollout
    26: from rlkit.samplers.in_place import InPlacePathSampler
    27: from rlkit.torch.sac.policies import MakeDeterministic
    28: 
    29: 
    30: # =====================================================================
    31: # FIXED — Per-environment configurations
    32: # =====================================================================
    33: ENV_CONFIGS = {
    34:     "cheetah-vel": {
    35:         "env_name": "cheetah-vel",
    36:         "n_train_tasks": 30,
    37:         "n_eval_tasks": 10,
    38:         "env_params": {"n_tasks": 40, "randomize_tasks": True},
    39:         "algo_params": {
    40:             "num_iterations": 50,
    41:             "num_initial_steps": 2000,
    42:             "num_tasks_sample": 5,
    43:             "num_steps_prior": 400,
    44:             "num_steps_posterior": 0,
    45:             "num_extra_rl_steps_posterior": 600,
    46:             "num_train_steps_per_itr": 600,
    47:             "num_evals": 1,
    48:             "num_steps_per_eval": 600,
    49:             "embedding_batch_size": 100,
    50:             "embedding_mini_batch_size": 100,
    51:             "max_path_length": 200,
    52:             "batch_size": 256,
    53:             "meta_batch": 16,
    54:             "discount": 0.99,
    55:             "reward_scale": 5.0,
    56:             "sparse_rewards": False,
    57:             "num_exp_traj_eval": 1,
    58:         },
    59:     },
    60:     "sparse-point-robot": {
    61:         "env_name": "sparse-point-robot",
    62:         "n_train_tasks": 40,
    63:         "n_eval_tasks": 10,
    64:         "env_params": {"n_tasks": 50, "randomize_tasks": True},
    65:         "algo_params": {
    66:             "num_iterations": 40,
    67:             "num_initial_steps": 200,
    68:             "num_tasks_sample": 10,
    69:             "num_steps_prior": 100,
    70:             "num_steps_posterior": 900,
    71:             "num_extra_rl_steps_posterior": 0,
    72:             "num_train_steps_per_itr": 1000,
    73:             "num_evals": 1,
    74:             "num_steps_per_eval": 400,
    75:             "embedding_batch_size": 1024,
    76:             "embedding_mini_batch_size": 1024,
    77:             "max_path_length": 20,
    78:             "batch_size": 256,
    79:             "meta_batch": 16,
    80:             "discount": 0.90,
    81:             "reward_scale": 100.0,
    82:             "sparse_rewards": True,
    83:             "kl_lambda": 1.0,
    84:             "num_exp_traj_eval": 5,
    85:         },
    86:     },
    87:     "point-robot": {
    88:         "env_name": "point-robot",
    89:         "n_train_tasks": 40,
    90:         "n_eval_tasks": 10,
    91:         "env_params": {"n_tasks": 50, "randomize_tasks": True},
    92:         "algo_params": {
    93:             "num_iterations": 30,
    94:             "num_initial_steps": 200,
    95:             "num_tasks_sample": 10,
    96:             "num_steps_prior": 200,
    97:             "num_steps_posterior": 0,
    98:             "num_extra_rl_steps_posterior": 200,
    99:             "num_train_steps_per_itr": 1000,
   100:             "num_evals": 1,
   101:             "num_steps_per_eval": 60,
   102:             "embedding_batch_size": 100,
   103:             "embedding_mini_batch_size": 100,
   104:             "max_path_length": 20,
   105:             "batch_size": 256,
   106:             "meta_batch": 16,
   107:             "discount": 0.99,
   108:             "reward_scale": 100.0,
   109:             "sparse_rewards": False,
   110:             "num_exp_traj_eval": 1,
   111:         },
   112:     },
   113: }
   114: 
   115: 
   116: # =====================================================================
   117: # FIXED — Network building blocks
   118: # =====================================================================
   119: def build_mlp(input_dim, output_dim, hidden_dim=200, n_layers=3, init_w=3e-3):
   120:     """Build a simple MLP with ReLU activations."""
   121:     layers = []
   122:     in_dim = input_dim
   123:     for _ in range(n_layers):
   124:         layers.append(nn.Linear(in_dim, hidden_dim))
   125:         layers.append(nn.ReLU())
   126:         in_dim = hidden_dim
   127:     last = nn.Linear(in_dim, output_dim)
   128:     last.weight.data.uniform_(-init_w, init_w)
   129:     last.bias.data.uniform_(-init_w, init_w)
   130:     layers.append(last)
   131:     return nn.Sequential(*layers)
   132: 
   133: 
   134: def build_policy(obs_dim, action_dim, latent_dim, net_size=300):
   135:     """Build a TanhGaussianPolicy conditioned on (obs, z)."""
   136:     return TanhGaussianPolicy(
   137:         hidden_sizes=[net_size, net_size, net_size],
   138:         obs_dim=obs_dim + latent_dim,
   139:         latent_dim=latent_dim,
   140:         action_dim=action_dim,
   141:     )
   142: 
   143: 
   144: def build_qf(obs_dim, action_dim, latent_dim, net_size=300):
   145:     """Build a FlattenMlp Q-function: Q(obs, action, z)."""
   146:     return FlattenMlp(
   147:         hidden_sizes=[net_size, net_size, net_size],
   148:         input_size=obs_dim + action_dim + latent_dim,
   149:         output_size=1,
   150:     )
   151: 
   152: 
   153: def build_vf(obs_dim, latent_dim, net_size=300):
   154:     """Build a FlattenMlp V-function: V(obs, z)."""
   155:     return FlattenMlp(
   156:         hidden_sizes=[net_size, net_size, net_size],
   157:         input_size=obs_dim + latent_dim,
   158:         output_size=1,
   159:     )
   160: 
   161: 
   162: # =====================================================================
   163: # FIXED — Replay buffer helpers
   164: # =====================================================================
   165: def create_replay_buffers(env, tasks, max_size=1000000):
   166:     """Create two MultiTaskReplayBuffers: one for RL, one for encoder."""
   167:     replay_buffer = MultiTaskReplayBuffer(max_size, env, tasks)
   168:     enc_replay_buffer = MultiTaskReplayBuffer(max_size, env, tasks)
   169:     return replay_buffer, enc_replay_buffer
   170: 
   171: 
   172: def unpack_batch(batch, sparse_reward=False):
   173:     """Unpack a batch dict to [obs, actions, rewards, next_obs, terminals] with leading task dim."""
   174:     o = batch['observations'][None, ...]
   175:     a = batch['actions'][None, ...]
   176:     if sparse_reward:
   177:         r = batch['sparse_rewards'][None, ...]
   178:     else:
   179:         r = batch['rewards'][None, ...]
   180:     no = batch['next_observations'][None, ...]
   181:     t = batch['terminals'][None, ...]
   182:     return [o, a, r, no, t]
   183: 
   184: 
   185: def sample_context_from_buffer(enc_replay_buffer, indices, embedding_batch_size,
   186:                                sparse_rewards=False, use_next_obs_in_context=False):
   187:     """Sample context batch from encoder replay buffer.
   188: 
   189:     Returns tensor of shape (num_tasks, embedding_batch_size, context_dim).
   190:     """
   191:     if not hasattr(indices, '__iter__'):
   192:         indices = [indices]
   193:     batches = [ptu.np_to_pytorch_batch(enc_replay_buffer.random_batch(idx, batch_size=embedding_batch_size))
   194:                for idx in indices]
   195:     context = [unpack_batch(batch, sparse_reward=sparse_rewards) for batch in batches]
   196:     # group like elements together
   197:     context = [[x[i] for x in context] for i in range(len(context[0]))]
   198:     context = [torch.cat(x, dim=0) for x in context]
   199:     # full context consists of [obs, act, rewards, next_obs, terms]
   200:     if use_next_obs_in_context:
   201:         context = torch.cat(context[:-1], dim=2)
   202:     else:
   203:         context = torch.cat(context[:-2], dim=2)
   204:     return context
   205: 
   206: 
   207: def sample_sac_batch(replay_buffer, indices, batch_size):
   208:     """Sample RL training batch from replay buffer.
   209: 
   210:     Returns [obs, actions, rewards, next_obs, terminals] each (num_tasks, batch_size, dim).
   211:     """
   212:     batches = [ptu.np_to_pytorch_batch(replay_buffer.random_batch(idx, batch_size=batch_size))
   213:                for idx in indices]
   214:     unpacked = [unpack_batch(batch) for batch in batches]
   215:     unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
   216:     unpacked = [torch.cat(x, dim=0) for x in unpacked]
   217:     return unpacked
   218: 
   219: 
   220: # =====================================================================
   221: # FIXED — Evaluation protocol
   222: # (Faithfully adapted from oyster's rl_algorithm.py evaluate() and
   223: #  _do_eval() / collect_paths(), using agent.adapt() instead of
   224: #  agent.infer_posterior(agent.context))
   225: # =====================================================================
   226: def collect_eval_paths(agent, env, sampler, task_idx, config, epoch=0, run=0):
   227:     """Collect evaluation paths for a single task, using online adaptation.
   228: 
   229:     Follows oyster's collect_paths(): collect trajectories with accum_context=True,
   230:     after num_exp_traj_eval trajectories call agent.adapt() then continue
   231:     with deterministic policy.
   232:     """
   233:     env.reset_task(task_idx)
   234:     agent.clear_context()
   235: 
   236:     eval_deterministic = True
   237:     num_steps_per_eval = config['num_steps_per_eval']
   238:     num_exp_traj_eval = config.get('num_exp_traj_eval', 1)
   239:     max_path_length = config['max_path_length']
   240:     sparse_rewards = config.get('sparse_rewards', False)
   241: 
   242:     paths = []
   243:     num_transitions = 0
   244:     num_trajs = 0
   245:     while num_transitions < num_steps_per_eval:
   246:         # Use deterministic policy after adaptation
   247:         policy = MakeDeterministic(agent) if eval_deterministic else agent
   248:         path = rollout(env, policy, max_path_length=max_path_length, accum_context=True)
   249:         # Save the latent context z
   250:         path['context'] = agent.z.detach().cpu().numpy()
   251:         paths.append(path)
   252:         num_transitions += len(path['observations'])
   253:         num_trajs += 1
   254:         if num_trajs >= num_exp_traj_eval:
   255:             agent.adapt()
   256: 
   257:     if sparse_rewards:
   258:         for p in paths:
   259:             sparse_rew = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
   260:             p['rewards'] = sparse_rew
   261: 
   262:     return paths
   263: 
   264: 
   265: def do_eval(agent, env, sampler, task_indices, config, epoch=0):
   266:     """Evaluate on a set of tasks, returning final and online returns.
   267: 
   268:     Follows oyster's _do_eval().
   269:     """
   270:     num_evals = config.get('num_evals', 1)
   271:     final_returns = []
   272:     online_returns = []
   273:     for idx in task_indices:
   274:         all_rets = []
   275:         for r in range(num_evals):
   276:             paths = collect_eval_paths(agent, env, sampler, idx, config, epoch, r)
   277:             all_rets.append([get_average_returns([p]) for p in paths])
   278:         final_returns.append(np.mean([a[-1] for a in all_rets]))
   279:         n = min([len(a) for a in all_rets])
   280:         all_rets = [a[:n] for a in all_rets]
   281:         all_rets = np.mean(np.stack(all_rets), axis=0)
   282:         online_returns.append(all_rets)
   283:     n = min([len(t) for t in online_returns])
   284:     online_returns = [t[:n] for t in online_returns]
   285:     return final_returns, online_returns
   286: 
   287: 
   288: def get_average_returns(paths):
   289:     """Compute average return across paths."""
   290:     returns = [sum(path["rewards"]) for path in paths]
   291:     return np.mean(returns)
   292: 
   293: 
   294: def run_evaluation(agent, env, train_tasks, eval_tasks, sampler, config, epoch):
   295:     """Full evaluation: train tasks + test tasks.
   296: 
   297:     Follows oyster's evaluate() method. Prints TRAIN_METRICS and TEST_METRICS.
   298:     """
   299:     # --- Evaluate on a subset of train tasks ---
   300:     indices = np.random.choice(train_tasks, len(eval_tasks))
   301: 
   302:     # Online evaluation on train tasks
   303:     train_final_returns, train_online_returns = do_eval(
   304:         agent, env, sampler, indices, config, epoch
   305:     )
   306:     avg_train_return = np.mean(train_final_returns)
   307: 
   308:     # --- Evaluate on test tasks ---
   309:     test_final_returns, test_online_returns = do_eval(
   310:         agent, env, sampler, eval_tasks, config, epoch
   311:     )
   312:     avg_test_return = np.mean(test_final_returns)
   313: 
   314:     print(f'TRAIN_METRICS iteration={epoch} avg_train_return={avg_train_return:.4f}', flush=True)
   315:     print(f'TEST_METRICS iteration={epoch} meta_test_return={avg_test_return:.4f}', flush=True)
   316: 
   317:     return avg_train_return, avg_test_return
   318: 
   319: 
   320: # =====================================================================
   321: # FIXED — Data collection helpers
   322: # =====================================================================
   323: def collect_data(agent, env, sampler, replay_buffer, enc_replay_buffer,
   324:                  task_idx, num_samples, resample_z_rate, update_posterior_rate,
   325:                  add_to_enc_buffer=True, config=None):
   326:     """Collect data for a single task, following oyster's collect_data().
   327: 
   328:     Uses sampler to get trajectories, adds to replay buffers.
   329:     """
   330:     agent.clear_context()
   331:     num_transitions = 0
   332:     while num_transitions < num_samples:
   333:         paths, n_samples = sampler.obtain_samples(
   334:             max_samples=num_samples - num_transitions,
   335:             max_trajs=update_posterior_rate,
   336:             accum_context=False,
   337:             resample=resample_z_rate,
   338:         )
   339:         num_transitions += n_samples
   340:         replay_buffer.add_paths(task_idx, paths)
   341:         if add_to_enc_buffer:
   342:             enc_replay_buffer.add_paths(task_idx, paths)
   343:         if update_posterior_rate != np.inf:
   344:             # Sample context from buffer and adapt
   345:             sparse_rewards = config.get('sparse_rewards', False) if config else False
   346:             use_next_obs = config.get('use_next_obs_in_context', False) if config else False
   347:             ctx = sample_context_from_buffer(
   348:                 enc_replay_buffer, task_idx,
   349:                 config.get('embedding_batch_size', 100),
   350:                 sparse_rewards=sparse_rewards,
   351:                 use_next_obs_in_context=use_next_obs,
   352:             )
   353:             agent.infer_posterior(ctx)
   354: 
   355: 
   356: # =====================================================================
   357: # EDITABLE — Custom imports
   358: # =====================================================================
   359: 
   360: 
   361: # =====================================================================
   362: # EDITABLE — Custom Meta-RL Agent
   363: # =====================================================================
   364: class CustomMetaRLAgent(nn.Module):
   365:     """Custom meta-RL agent.
   366: 
   367:     Must implement:
   368:       - get_action(obs, deterministic=False) -> (action_np, agent_info)
   369:       - update_context(transition_tuple) -> None
   370:       - adapt() -> None  (called after context collection; performs task inference)
   371:       - clear_context(num_tasks=1) -> None
   372:       - infer_posterior(context_tensor) -> None  (for training-time context encoding)
   373:       - context: property or attribute returning collected context
   374:       - z: tensor attribute for latent task variable
   375:       - networks: list of nn.Module for GPU transfer and param counting
   376:     """
   377: 
   378:     def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
   379:                  reward_dim=1, use_next_obs_in_context=False, **kwargs):
   380:         super().__init__()
   381:         self.obs_dim = obs_dim
   382:         self.action_dim = action_dim
   383:         self.latent_dim = latent_dim
   384:         self.use_next_obs_in_context = use_next_obs_in_context
   385: 
   386:         # Simple MLP policy (no task conditioning, placeholder)
   387:         self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
   388: 
   389:         # Latent variable z (dummy zeros for this placeholder)
   390:         self.register_buffer('z', torch.zeros(1, latent_dim))
   391:         self._context = None
   392: 
   393:     def clear_context(self, num_tasks=1):
   394:         self.z = ptu.zeros(num_tasks, self.latent_dim)
   395:         self._context = None
   396: 
   397:     # alias used by InPlacePathSampler via sample_z
   398:     def clear_z(self, num_tasks=1):
   399:         self.clear_context(num_tasks)
   400: 
   401:     def sample_z(self):
   402:         pass
   403: 
   404:     @property
   405:     def context(self):
   406:         return self._context
   407: 
   408:     def update_context(self, inputs):
   409:         """Append single transition to context. inputs = [obs, action, reward, next_obs, done, env_info]."""
   410:         o, a, r, no, d, info = inputs
   411:         o = ptu.from_numpy(o[None, None, ...])
   412:         a = ptu.from_numpy(a[None, None, ...])
   413:         r = ptu.from_numpy(np.array([r])[None, None, ...])
   414:         no = ptu.from_numpy(no[None, None, ...])
   415:         if self.use_next_obs_in_context:
   416:             data = torch.cat([o, a, r, no], dim=2)
   417:         else:
   418:             data = torch.cat([o, a, r], dim=2)
   419:         if self._context is None:
   420:             self._context = data
   421:         else:
   422:             self._context = torch.cat([self._context, data], dim=1)
   423: 
   424:     def adapt(self):
   425:         """Perform task inference from collected context. Override this."""
   426:         pass
   427: 
   428:     def infer_posterior(self, context):
   429:         """Encode context tensor (from replay buffer) for training. Override this."""
   430:         pass
   431: 
   432:     def get_action(self, obs, deterministic=False):
   433:         z = self.z
   434:         obs_t = ptu.from_numpy(obs[None])
   435:         in_ = torch.cat([obs_t, z], dim=1)
   436:         return self.policy.get_action(in_, deterministic=deterministic)
   437: 
   438:     def set_num_steps_total(self, n):
   439:         self.policy.set_num_steps_total(n)
   440: 
   441:     def detach_z(self):
   442:         self.z = self.z.detach()
   443: 
   444:     @property
   445:     def networks(self):
   446:         return [self.policy]
   447: 
   448: 
   449: class CustomMetaRLAlgorithm:
   450:     """Custom meta-RL training algorithm.
   451: 
   452:     Must implement:
   453:       - __init__(agent, env, train_tasks, eval_tasks, replay_buffer, enc_replay_buffer, config)
   454:       - collect_initial_data() -> None
   455:       - train_iteration(iteration_idx) -> dict  (one meta-training iteration)
   456:       - agent: attribute referencing the CustomMetaRLAgent
   457:     """
   458: 
   459:     def __init__(self, agent, env, train_tasks, eval_tasks,
   460:                  replay_buffer, enc_replay_buffer, config):
   461:         self.agent = agent
   462:         self.env = env
   463:         self.train_tasks = train_tasks
   464:         self.eval_tasks = eval_tasks
   465:         self.replay_buffer = replay_buffer
   466:         self.enc_replay_buffer = enc_replay_buffer
   467:         self.config = config
   468: 
   469:         self.sampler = InPlacePathSampler(
   470:             env=env, policy=agent,
   471:             max_path_length=config['max_path_length'],
   472:         )
   473: 
   474:     def collect_initial_data(self):
   475:         """Collect initial exploration data for all training tasks."""
   476:         num_initial_steps = self.config.get('num_initial_steps', 200)
   477:         for idx in self.train_tasks:
   478:             self.env.reset_task(idx)
   479:             collect_data(
   480:                 self.agent, self.env, self.sampler,
   481:                 self.replay_buffer, self.enc_replay_buffer,
   482:                 idx, num_initial_steps, 1, np.inf,
   483:                 add_to_enc_buffer=True, config=self.config,
   484:             )
   485: 
   486:     def train_iteration(self, iteration_idx):
   487:         """One meta-training iteration: collect data + gradient updates."""
   488:         return {}
   489: 
   490:     @property
   491:     def networks(self):
   492:         """Return all networks for GPU transfer."""
   493:         return self.agent.networks
   494: 
   495: 
   496: # =====================================================================
   497: # FIXED — Main entry point
   498: # =====================================================================
   499: def main():
   500:     parser = argparse.ArgumentParser(description='Custom Meta-RL Algorithm')

[truncated: showing at most 500 lines / 60000 bytes from oyster/custom_meta_rl.py]
```

## Parameter Budget

This task enforces a parameter-count cap. Your edits will be rejected if
the resulting model exceeds **1.05×** the strongest
baseline's parameter count. The check runs automatically inside the eval
scripts — you don't need to invoke it.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `pearl` baseline — editable region  [READ-ONLY — reference implementation]

In `oyster/custom_meta_rl.py`:

```python
Lines 357–679:
   354: 
   355: 
   356: # =====================================================================
   357: # =====================================================================
   358: # EDITABLE — Custom imports
   359: # =====================================================================
   360: from rlkit.torch.distributions import TanhNormal
   361: 
   362: 
   363: def _product_of_gaussians(mus, sigmas_squared):
   364:     sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
   365:     sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
   366:     mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
   367:     return mu, sigma_squared
   368: 
   369: 
   370: # =====================================================================
   371: # EDITABLE — Custom Meta-RL Agent
   372: # =====================================================================
   373: class CustomMetaRLAgent(nn.Module):
   374:     """PEARL agent with product-of-Gaussians context encoder."""
   375: 
   376:     def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
   377:                  reward_dim=1, use_next_obs_in_context=False, **kwargs):
   378:         super().__init__()
   379:         self.obs_dim = obs_dim
   380:         self.action_dim = action_dim
   381:         self.latent_dim = latent_dim
   382:         self.use_next_obs_in_context = use_next_obs_in_context
   383:         self.sparse_rewards = kwargs.get('sparse_rewards', False)
   384: 
   385:         # Context encoder input: (obs, action, reward [, next_obs])
   386:         context_input_dim = obs_dim + action_dim + reward_dim
   387:         if use_next_obs_in_context:
   388:             context_input_dim += obs_dim
   389:         context_output_dim = latent_dim * 2  # mean + logvar for IB
   390: 
   391:         # Context encoder: 3-layer MLP
   392:         self.context_encoder = build_mlp(
   393:             context_input_dim, context_output_dim,
   394:             hidden_dim=200, n_layers=3,
   395:         )
   396:         self.context_encoder_output_size = context_output_dim
   397: 
   398:         # Policy, Q-functions, V-function (z-conditioned)
   399:         self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
   400:         self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
   401:         self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
   402:         self.vf = build_vf(obs_dim, latent_dim, net_size)
   403:         self.target_vf = copy.deepcopy(self.vf)
   404: 
   405:         # z distribution
   406:         self.register_buffer('z', torch.zeros(1, latent_dim))
   407:         self.register_buffer('z_means', torch.zeros(1, latent_dim))
   408:         self.register_buffer('z_vars', torch.ones(1, latent_dim))
   409:         self._context = None
   410: 
   411:     def clear_context(self, num_tasks=1):
   412:         mu = ptu.zeros(num_tasks, self.latent_dim)
   413:         var = ptu.ones(num_tasks, self.latent_dim)
   414:         self.z_means = mu
   415:         self.z_vars = var
   416:         self.sample_z()
   417:         self._context = None
   418: 
   419:     def clear_z(self, num_tasks=1):
   420:         self.clear_context(num_tasks)
   421: 
   422:     @property
   423:     def context(self):
   424:         return self._context
   425: 
   426:     def update_context(self, inputs):
   427:         o, a, r, no, d, info = inputs
   428:         if self.sparse_rewards:
   429:             r = info.get('sparse_reward', r)
   430:         o = ptu.from_numpy(o[None, None, ...])
   431:         a = ptu.from_numpy(a[None, None, ...])
   432:         r = ptu.from_numpy(np.array([r])[None, None, ...])
   433:         no = ptu.from_numpy(no[None, None, ...])
   434:         if self.use_next_obs_in_context:
   435:             data = torch.cat([o, a, r, no], dim=2)
   436:         else:
   437:             data = torch.cat([o, a, r], dim=2)
   438:         if self._context is None:
   439:             self._context = data
   440:         else:
   441:             self._context = torch.cat([self._context, data], dim=1)
   442: 
   443:     def infer_posterior(self, context):
   444:         params = self.context_encoder(context)
   445:         params = params.view(context.size(0), -1, self.context_encoder_output_size)
   446:         mu = params[..., :self.latent_dim]
   447:         sigma_squared = F.softplus(params[..., self.latent_dim:])
   448:         z_params = [_product_of_gaussians(m, s)
   449:                     for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
   450:         self.z_means = torch.stack([p[0] for p in z_params])
   451:         self.z_vars = torch.stack([p[1] for p in z_params])
   452:         self.sample_z()
   453: 
   454:     def sample_z(self):
   455:         posteriors = [torch.distributions.Normal(m, torch.sqrt(s))
   456:                       for m, s in zip(torch.unbind(self.z_means),
   457:                                       torch.unbind(self.z_vars))]
   458:         z = [d.rsample() for d in posteriors]
   459:         self.z = torch.stack(z)
   460: 
   461:     def adapt(self):
   462:         if self._context is not None:
   463:             self.infer_posterior(self._context)
   464: 
   465:     def compute_kl_div(self):
   466:         prior = torch.distributions.Normal(
   467:             ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
   468:         posteriors = [torch.distributions.Normal(mu, torch.sqrt(var))
   469:                       for mu, var in zip(torch.unbind(self.z_means),
   470:                                          torch.unbind(self.z_vars))]
   471:         kl_divs = [torch.distributions.kl.kl_divergence(post, prior)
   472:                    for post in posteriors]
   473:         return torch.sum(torch.stack(kl_divs))
   474: 
   475:     def get_action(self, obs, deterministic=False):
   476:         z = self.z
   477:         obs_t = ptu.from_numpy(obs[None])
   478:         in_ = torch.cat([obs_t, z], dim=1)
   479:         return self.policy.get_action(in_, deterministic=deterministic)
   480: 
   481:     def forward(self, obs, context):
   482:         self.infer_posterior(context)
   483:         self.sample_z()
   484:         task_z = self.z
   485:         t, b, _ = obs.size()
   486:         obs = obs.view(t * b, -1)
   487:         task_z = [z.repeat(b, 1) for z in task_z]
   488:         task_z = torch.cat(task_z, dim=0)
   489:         in_ = torch.cat([obs, task_z.detach()], dim=1)
   490:         policy_outputs = self.policy(
   491:             in_, reparameterize=True, return_log_prob=True)
   492:         return policy_outputs, task_z
   493: 
   494:     def set_num_steps_total(self, n):
   495:         self.policy.set_num_steps_total(n)
   496: 
   497:     def detach_z(self):
   498:         self.z = self.z.detach()
   499: 
   500:     @property
   501:     def networks(self):
   502:         return [self.policy, self.qf1, self.qf2,
   503:                 self.vf, self.target_vf]
   504: 
   505: 
   506: class CustomMetaRLAlgorithm:
   507:     """PEARL SAC meta-training algorithm."""
   508: 
   509:     def __init__(self, agent, env, train_tasks, eval_tasks,
   510:                  replay_buffer, enc_replay_buffer, config):
   511:         self.agent = agent
   512:         self.env = env
   513:         self.train_tasks = train_tasks
   514:         self.eval_tasks = eval_tasks
   515:         self.replay_buffer = replay_buffer
   516:         self.enc_replay_buffer = enc_replay_buffer
   517:         self.config = config
   518: 
   519:         self.sampler = InPlacePathSampler(
   520:             env=env, policy=agent,
   521:             max_path_length=config['max_path_length'],
   522:         )
   523: 
   524:         # Hyperparameters
   525:         self.batch_size = config.get('batch_size', 256)
   526:         self.meta_batch = config.get('meta_batch', 16)
   527:         self.discount = config.get('discount', 0.99)
   528:         self.reward_scale = config.get('reward_scale', 5.0)
   529:         self.kl_lambda = config.get('kl_lambda', 0.1)
   530:         self.soft_target_tau = config.get('soft_target_tau', 0.005)
   531:         self.sparse_rewards = config.get('sparse_rewards', False)
   532:         self.use_next_obs_in_context = config.get('use_next_obs_in_context', False)
   533:         self.embedding_batch_size = config.get('embedding_batch_size', 64)
   534:         self.embedding_mini_batch_size = config.get('embedding_mini_batch_size', 64)
   535:         self.num_tasks_sample = config.get('num_tasks_sample', 5)
   536:         self.num_steps_prior = config.get('num_steps_prior', 400)
   537:         self.num_steps_posterior = config.get('num_steps_posterior', 0)
   538:         self.num_extra_rl_steps_posterior = config.get('num_extra_rl_steps_posterior', 400)
   539:         self.num_train_steps_per_itr = config.get('num_train_steps_per_itr', 2000)
   540:         self.update_post_train = config.get('update_post_train', 1)
   541: 
   542:         # Optimizers
   543:         lr = 3e-4
   544:         self.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
   545:         self.qf1_optimizer = optim.Adam(agent.qf1.parameters(), lr=lr)
   546:         self.qf2_optimizer = optim.Adam(agent.qf2.parameters(), lr=lr)
   547:         self.vf_optimizer = optim.Adam(agent.vf.parameters(), lr=lr)
   548:         self.context_optimizer = optim.Adam(
   549:             agent.context_encoder.parameters(), lr=lr)
   550: 
   551:     def collect_initial_data(self):
   552:         num_initial_steps = self.config.get('num_initial_steps', 200)
   553:         for idx in self.train_tasks:
   554:             self.env.reset_task(idx)
   555:             collect_data(
   556:                 self.agent, self.env, self.sampler,
   557:                 self.replay_buffer, self.enc_replay_buffer,
   558:                 idx, num_initial_steps, 1, np.inf,
   559:                 add_to_enc_buffer=True, config=self.config,
   560:             )
   561: 
   562:     def train_iteration(self, iteration_idx):
   563:         # --- Collect data from sampled tasks ---
   564:         for i in range(self.num_tasks_sample):
   565:             idx = np.random.randint(len(self.train_tasks))
   566:             self.env.reset_task(idx)
   567:             self.enc_replay_buffer.task_buffers[idx].clear()
   568:             if self.num_steps_prior > 0:
   569:                 collect_data(
   570:                     self.agent, self.env, self.sampler,
   571:                     self.replay_buffer, self.enc_replay_buffer,
   572:                     idx, self.num_steps_prior, 1, np.inf,
   573:                     config=self.config,
   574:                 )
   575:             if self.num_steps_posterior > 0:
   576:                 collect_data(
   577:                     self.agent, self.env, self.sampler,
   578:                     self.replay_buffer, self.enc_replay_buffer,
   579:                     idx, self.num_steps_posterior, 1, self.update_post_train,
   580:                     config=self.config,
   581:                 )
   582:             if self.num_extra_rl_steps_posterior > 0:
   583:                 collect_data(
   584:                     self.agent, self.env, self.sampler,
   585:                     self.replay_buffer, self.enc_replay_buffer,
   586:                     idx, self.num_extra_rl_steps_posterior, 1,
   587:                     self.update_post_train,
   588:                     add_to_enc_buffer=False, config=self.config,
   589:                 )
   590: 
   591:         # --- Meta-gradient updates ---
   592:         for _ in range(self.num_train_steps_per_itr):
   593:             indices = np.random.choice(
   594:                 self.train_tasks, self.meta_batch)
   595:             self._take_step(indices)
   596:         return {}
   597: 
   598:     def _take_step(self, indices):
   599:         mb_size = self.embedding_mini_batch_size
   600:         num_updates = self.embedding_batch_size // mb_size
   601: 
   602:         context_batch = sample_context_from_buffer(
   603:             self.enc_replay_buffer, indices, self.embedding_batch_size,
   604:             sparse_rewards=self.sparse_rewards,
   605:             use_next_obs_in_context=self.use_next_obs_in_context,
   606:         )
   607:         self.agent.clear_z(num_tasks=len(indices))
   608: 
   609:         for i in range(num_updates):
   610:             context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
   611:             self._update(indices, context)
   612:             self.agent.detach_z()
   613: 
   614:     def _update(self, indices, context):
   615:         num_tasks = len(indices)
   616:         obs, actions, rewards, next_obs, terms = sample_sac_batch(
   617:             self.replay_buffer, indices, self.batch_size)
   618: 
   619:         # Forward pass through agent
   620:         policy_outputs, task_z = self.agent(obs, context)
   621:         new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
   622: 
   623:         t, b, _ = obs.size()
   624:         obs_flat = obs.view(t * b, -1)
   625:         actions_flat = actions.view(t * b, -1)
   626:         next_obs_flat = next_obs.view(t * b, -1)
   627: 
   628:         # Q and V predictions
   629:         q1_pred = self.agent.qf1(obs_flat, actions_flat, task_z)
   630:         q2_pred = self.agent.qf2(obs_flat, actions_flat, task_z)
   631:         v_pred = self.agent.vf(obs_flat, task_z.detach())
   632:         with torch.no_grad():
   633:             target_v = self.agent.target_vf(next_obs_flat, task_z)
   634: 
   635:         # KL loss
   636:         self.context_optimizer.zero_grad()
   637:         kl_div = self.agent.compute_kl_div()
   638:         kl_loss = self.kl_lambda * kl_div
   639:         kl_loss.backward(retain_graph=True)
   640: 
   641:         # Q-function loss
   642:         self.qf1_optimizer.zero_grad()
   643:         self.qf2_optimizer.zero_grad()
   644:         rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
   645:         rewards_flat = rewards_flat * self.reward_scale
   646:         terms_flat = terms.view(self.batch_size * num_tasks, -1)
   647:         q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v
   648:         qf_loss = (torch.mean((q1_pred - q_target) ** 2) +
   649:                    torch.mean((q2_pred - q_target) ** 2))
   650:         qf_loss.backward()
   651:         self.qf1_optimizer.step()
   652:         self.qf2_optimizer.step()
   653:         self.context_optimizer.step()
   654: 
   655:         # V-function loss
   656:         min_q = torch.min(
   657:             self.agent.qf1(obs_flat, new_actions, task_z.detach()),
   658:             self.agent.qf2(obs_flat, new_actions, task_z.detach()),
   659:         )
   660:         v_target = min_q - log_pi
   661:         vf_loss = F.mse_loss(v_pred, v_target.detach())
   662:         self.vf_optimizer.zero_grad()
   663:         vf_loss.backward()
   664:         self.vf_optimizer.step()
   665:         ptu.soft_update_from_to(
   666:             self.agent.vf, self.agent.target_vf, self.soft_target_tau)
   667: 
   668:         # Policy loss
   669:         policy_loss = (log_pi - min_q).mean()
   670:         mean_reg = 1e-3 * (policy_mean ** 2).mean()
   671:         std_reg = 1e-3 * (policy_log_std ** 2).mean()
   672:         policy_loss = policy_loss + mean_reg + std_reg
   673:         self.policy_optimizer.zero_grad()
   674:         policy_loss.backward()
   675:         self.policy_optimizer.step()
   676: 
   677:     @property
   678:     def networks(self):
   679:         return self.agent.networks
   680: 
   681: # =====================================================================
   682: # FIXED — Main entry point
```

### `focal` baseline — editable region  [READ-ONLY — reference implementation]

In `oyster/custom_meta_rl.py`:

```python
Lines 357–673:
   354: 
   355: 
   356: # =====================================================================
   357: # =====================================================================
   358: # EDITABLE — Custom imports
   359: # =====================================================================
   360: 
   361: 
   362: # =====================================================================
   363: # EDITABLE — Custom Meta-RL Agent
   364: # =====================================================================
   365: class CustomMetaRLAgent(nn.Module):
   366:     """FOCAL agent: deterministic encoder with mean aggregation."""
   367: 
   368:     def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
   369:                  reward_dim=1, use_next_obs_in_context=False, **kwargs):
   370:         super().__init__()
   371:         self.obs_dim = obs_dim
   372:         self.action_dim = action_dim
   373:         self.latent_dim = latent_dim
   374:         self.use_next_obs_in_context = use_next_obs_in_context
   375:         self.sparse_rewards = kwargs.get('sparse_rewards', False)
   376: 
   377:         context_input_dim = obs_dim + action_dim + reward_dim
   378:         if use_next_obs_in_context:
   379:             context_input_dim += obs_dim
   380: 
   381:         # Deterministic encoder: output is latent_dim (no IB)
   382:         self.context_encoder = build_mlp(
   383:             context_input_dim, latent_dim,
   384:             hidden_dim=200, n_layers=3,
   385:         )
   386: 
   387:         # Policy, Q-functions, V-function (z-conditioned)
   388:         self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
   389:         self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
   390:         self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
   391:         self.vf = build_vf(obs_dim, latent_dim, net_size)
   392:         self.target_vf = copy.deepcopy(self.vf)
   393: 
   394:         self.register_buffer('z', torch.zeros(1, latent_dim))
   395:         self._context = None
   396: 
   397:     def clear_context(self, num_tasks=1):
   398:         self.z = ptu.zeros(num_tasks, self.latent_dim)
   399:         self._context = None
   400: 
   401:     def clear_z(self, num_tasks=1):
   402:         self.clear_context(num_tasks)
   403: 
   404:     def sample_z(self):
   405:         pass  # z is deterministic
   406: 
   407:     @property
   408:     def context(self):
   409:         return self._context
   410: 
   411:     def update_context(self, inputs):
   412:         o, a, r, no, d, info = inputs
   413:         if self.sparse_rewards:
   414:             r = info.get('sparse_reward', r)
   415:         o = ptu.from_numpy(o[None, None, ...])
   416:         a = ptu.from_numpy(a[None, None, ...])
   417:         r = ptu.from_numpy(np.array([r])[None, None, ...])
   418:         no = ptu.from_numpy(no[None, None, ...])
   419:         if self.use_next_obs_in_context:
   420:             data = torch.cat([o, a, r, no], dim=2)
   421:         else:
   422:             data = torch.cat([o, a, r], dim=2)
   423:         if self._context is None:
   424:             self._context = data
   425:         else:
   426:             self._context = torch.cat([self._context, data], dim=1)
   427: 
   428:     def infer_posterior(self, context):
   429:         """Encode context and take mean over transitions."""
   430:         embeddings = self.context_encoder(context)
   431:         # embeddings: (num_tasks, seq_len, latent_dim)
   432:         embeddings = embeddings.view(context.size(0), -1, self.latent_dim)
   433:         self.z = torch.mean(embeddings, dim=1)  # (num_tasks, latent_dim)
   434: 
   435:     def adapt(self):
   436:         if self._context is not None:
   437:             self.infer_posterior(self._context)
   438: 
   439:     def get_action(self, obs, deterministic=False):
   440:         z = self.z
   441:         obs_t = ptu.from_numpy(obs[None])
   442:         in_ = torch.cat([obs_t, z], dim=1)
   443:         return self.policy.get_action(in_, deterministic=deterministic)
   444: 
   445:     def forward(self, obs, context):
   446:         self.infer_posterior(context)
   447:         task_z = self.z
   448:         t, b, _ = obs.size()
   449:         obs = obs.view(t * b, -1)
   450:         task_z = [z.repeat(b, 1) for z in task_z]
   451:         task_z = torch.cat(task_z, dim=0)
   452:         in_ = torch.cat([obs, task_z.detach()], dim=1)
   453:         policy_outputs = self.policy(
   454:             in_, reparameterize=True, return_log_prob=True)
   455:         return policy_outputs, task_z
   456: 
   457:     def set_num_steps_total(self, n):
   458:         self.policy.set_num_steps_total(n)
   459: 
   460:     def detach_z(self):
   461:         self.z = self.z.detach()
   462: 
   463:     @property
   464:     def networks(self):
   465:         return [self.policy, self.qf1, self.qf2,
   466:                 self.vf, self.target_vf]
   467: 
   468: 
   469: class CustomMetaRLAlgorithm:
   470:     """FOCAL SAC with deep metric encoder loss."""
   471: 
   472:     def __init__(self, agent, env, train_tasks, eval_tasks,
   473:                  replay_buffer, enc_replay_buffer, config):
   474:         self.agent = agent
   475:         self.env = env
   476:         self.train_tasks = train_tasks
   477:         self.eval_tasks = eval_tasks
   478:         self.replay_buffer = replay_buffer
   479:         self.enc_replay_buffer = enc_replay_buffer
   480:         self.config = config
   481: 
   482:         self.sampler = InPlacePathSampler(
   483:             env=env, policy=agent,
   484:             max_path_length=config['max_path_length'],
   485:         )
   486: 
   487:         self.batch_size = config.get('batch_size', 256)
   488:         self.meta_batch = config.get('meta_batch', 16)
   489:         self.discount = config.get('discount', 0.99)
   490:         self.reward_scale = config.get('reward_scale', 5.0)
   491:         self.soft_target_tau = config.get('soft_target_tau', 0.005)
   492:         self.sparse_rewards = config.get('sparse_rewards', False)
   493:         self.use_next_obs_in_context = config.get('use_next_obs_in_context', False)
   494:         self.embedding_batch_size = config.get('embedding_batch_size', 64)
   495:         self.embedding_mini_batch_size = config.get('embedding_mini_batch_size', 64)
   496:         self.num_tasks_sample = config.get('num_tasks_sample', 5)
   497:         self.num_steps_prior = config.get('num_steps_prior', 400)
   498:         self.num_steps_posterior = config.get('num_steps_posterior', 0)
   499:         self.num_extra_rl_steps_posterior = config.get('num_extra_rl_steps_posterior', 400)
   500:         self.num_train_steps_per_itr = config.get('num_train_steps_per_itr', 2000)
   501:         self.update_post_train = config.get('update_post_train', 1)
   502:         self.contrastive_weight = 1.0
   503:         self.dml_beta = config.get('dml_beta', 1.0)
   504:         self.dml_epsilon = config.get('dml_epsilon', 1e-3)
   505:         self.dml_power = config.get('dml_power', 2.0)
   506: 
   507:         lr = 3e-4
   508:         self.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
   509:         self.qf1_optimizer = optim.Adam(agent.qf1.parameters(), lr=lr)
   510:         self.qf2_optimizer = optim.Adam(agent.qf2.parameters(), lr=lr)
   511:         self.vf_optimizer = optim.Adam(agent.vf.parameters(), lr=lr)
   512:         self.context_optimizer = optim.Adam(
   513:             agent.context_encoder.parameters(), lr=lr)
   514: 
   515:     def collect_initial_data(self):
   516:         num_initial_steps = self.config.get('num_initial_steps', 200)
   517:         for idx in self.train_tasks:
   518:             self.env.reset_task(idx)
   519:             collect_data(
   520:                 self.agent, self.env, self.sampler,
   521:                 self.replay_buffer, self.enc_replay_buffer,
   522:                 idx, num_initial_steps, 1, np.inf,
   523:                 add_to_enc_buffer=True, config=self.config,
   524:             )
   525: 
   526:     def train_iteration(self, iteration_idx):
   527:         for i in range(self.num_tasks_sample):
   528:             idx = np.random.randint(len(self.train_tasks))
   529:             self.env.reset_task(idx)
   530:             self.enc_replay_buffer.task_buffers[idx].clear()
   531:             if self.num_steps_prior > 0:
   532:                 collect_data(
   533:                     self.agent, self.env, self.sampler,
   534:                     self.replay_buffer, self.enc_replay_buffer,
   535:                     idx, self.num_steps_prior, 1, np.inf,
   536:                     config=self.config,
   537:                 )
   538:             if self.num_steps_posterior > 0:
   539:                 collect_data(
   540:                     self.agent, self.env, self.sampler,
   541:                     self.replay_buffer, self.enc_replay_buffer,
   542:                     idx, self.num_steps_posterior, 1, self.update_post_train,
   543:                     config=self.config,
   544:                 )
   545:             if self.num_extra_rl_steps_posterior > 0:
   546:                 collect_data(
   547:                     self.agent, self.env, self.sampler,
   548:                     self.replay_buffer, self.enc_replay_buffer,
   549:                     idx, self.num_extra_rl_steps_posterior, 1,
   550:                     self.update_post_train,
   551:                     add_to_enc_buffer=False, config=self.config,
   552:                 )
   553: 
   554:         for _ in range(self.num_train_steps_per_itr):
   555:             indices = np.random.choice(
   556:                 self.train_tasks, self.meta_batch)
   557:             self._take_step(indices)
   558:         return {}
   559: 
   560:     def _take_step(self, indices):
   561:         mb_size = self.embedding_mini_batch_size
   562:         num_updates = self.embedding_batch_size // mb_size
   563: 
   564:         context_batch = sample_context_from_buffer(
   565:             self.enc_replay_buffer, indices, self.embedding_batch_size,
   566:             sparse_rewards=self.sparse_rewards,
   567:             use_next_obs_in_context=self.use_next_obs_in_context,
   568:         )
   569:         self.agent.clear_z(num_tasks=len(indices))
   570: 
   571:         for i in range(num_updates):
   572:             context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
   573:             self._update(indices, context)
   574:             self.agent.detach_z()
   575: 
   576:     def _encode_task_context(self, context):
   577:         embeddings = self.agent.context_encoder(context)
   578:         embeddings = embeddings.view(context.size(0), -1, self.agent.latent_dim)
   579:         return torch.mean(embeddings, dim=1)
   580: 
   581:     def _compute_contrastive_loss(self, indices, context):
   582:         """FOCAL Eq. 13 deep metric learning loss."""
   583:         half = context.size(1) // 2
   584:         labels_base = np.asarray(indices)
   585:         if labels_base.ndim == 0:
   586:             labels_base = labels_base[None]
   587:         if half == 0:
   588:             z = self._encode_task_context(context)
   589:             labels_np = labels_base
   590:         else:
   591:             ctx_a = context[:, :half, :]
   592:             ctx_b = context[:, half:2*half, :]
   593:             z = torch.cat([
   594:                 self._encode_task_context(ctx_a),
   595:                 self._encode_task_context(ctx_b),
   596:             ], dim=0)
   597:             labels_np = np.concatenate([labels_base, labels_base])
   598: 
   599:         labels = torch.as_tensor(labels_np, device=context.device)
   600:         same_task = labels[:, None].eq(labels[None, :]).float()
   601:         dist_sq = torch.sum((z[:, None, :] - z[None, :, :]) ** 2, dim=-1)
   602:         positive_loss = same_task * dist_sq
   603:         dist = torch.sqrt(dist_sq + 1e-12)
   604:         negative_loss = (1.0 - same_task) * (
   605:             self.dml_beta /
   606:             (torch.pow(dist, self.dml_power) + self.dml_epsilon)
   607:         )
   608:         return (positive_loss + negative_loss).mean()
   609: 
   610:     def _update(self, indices, context):
   611:         num_tasks = len(indices)
   612:         obs, actions, rewards, next_obs, terms = sample_sac_batch(
   613:             self.replay_buffer, indices, self.batch_size)
   614: 
   615:         policy_outputs, task_z = self.agent(obs, context)
   616:         new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
   617: 
   618:         t, b, _ = obs.size()
   619:         obs_flat = obs.view(t * b, -1)
   620:         actions_flat = actions.view(t * b, -1)
   621:         next_obs_flat = next_obs.view(t * b, -1)
   622: 
   623:         q1_pred = self.agent.qf1(obs_flat, actions_flat, task_z)
   624:         q2_pred = self.agent.qf2(obs_flat, actions_flat, task_z)
   625:         v_pred = self.agent.vf(obs_flat, task_z.detach())
   626:         with torch.no_grad():
   627:             target_v = self.agent.target_vf(next_obs_flat, task_z)
   628: 
   629:         # Eq. 13 deep metric encoder loss
   630:         self.context_optimizer.zero_grad()
   631:         contrastive_loss = self._compute_contrastive_loss(indices, context)
   632:         encoder_loss = self.contrastive_weight * contrastive_loss
   633:         encoder_loss.backward(retain_graph=True)
   634: 
   635:         # Q-function loss
   636:         self.qf1_optimizer.zero_grad()
   637:         self.qf2_optimizer.zero_grad()
   638:         rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
   639:         rewards_flat = rewards_flat * self.reward_scale
   640:         terms_flat = terms.view(self.batch_size * num_tasks, -1)
   641:         q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v
   642:         qf_loss = (torch.mean((q1_pred - q_target) ** 2) +
   643:                    torch.mean((q2_pred - q_target) ** 2))
   644:         qf_loss.backward()
   645:         self.qf1_optimizer.step()
   646:         self.qf2_optimizer.step()
   647:         self.context_optimizer.step()
   648: 
   649:         # V-function loss
   650:         min_q = torch.min(
   651:             self.agent.qf1(obs_flat, new_actions, task_z.detach()),
   652:             self.agent.qf2(obs_flat, new_actions, task_z.detach()),
   653:         )
   654:         v_target = min_q - log_pi
   655:         vf_loss = F.mse_loss(v_pred, v_target.detach())
   656:         self.vf_optimizer.zero_grad()
   657:         vf_loss.backward()
   658:         self.vf_optimizer.step()
   659:         ptu.soft_update_from_to(
   660:             self.agent.vf, self.agent.target_vf, self.soft_target_tau)
   661: 
   662:         # Policy loss
   663:         policy_loss = (log_pi - min_q).mean()
   664:         mean_reg = 1e-3 * (policy_mean ** 2).mean()
   665:         std_reg = 1e-3 * (policy_log_std ** 2).mean()
   666:         policy_loss = policy_loss + mean_reg + std_reg
   667:         self.policy_optimizer.zero_grad()
   668:         policy_loss.backward()
   669:         self.policy_optimizer.step()
   670: 
   671:     @property
   672:     def networks(self):
   673:         return self.agent.networks
   674: 
   675: # =====================================================================
   676: # FIXED — Main entry point
```

### `varibad` baseline — editable region  [READ-ONLY — reference implementation]

In `oyster/custom_meta_rl.py`:

```python
Lines 357–760:
   354: 
   355: 
   356: # =====================================================================
   357: # =====================================================================
   358: # EDITABLE — Custom imports
   359: # =====================================================================
   360: 
   361: 
   362: # =====================================================================
   363: # EDITABLE — Custom Meta-RL Agent
   364: # =====================================================================
   365: class GRUContextEncoder(nn.Module):
   366:     """GRU-based context encoder for sequential processing."""
   367: 
   368:     def __init__(self, input_dim, hidden_dim, latent_dim):
   369:         super().__init__()
   370:         self.hidden_dim = hidden_dim
   371:         self.latent_dim = latent_dim
   372:         # Pre-process transitions with MLP
   373:         self.fc_pre = nn.Sequential(
   374:             nn.Linear(input_dim, hidden_dim),
   375:             nn.ReLU(),
   376:             nn.Linear(hidden_dim, hidden_dim),
   377:             nn.ReLU(),
   378:         )
   379:         self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
   380:         # Output mean and logvar for latent
   381:         self.fc_mean = nn.Linear(hidden_dim, latent_dim)
   382:         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
   383:         self.register_buffer('hidden', torch.zeros(1, 1, hidden_dim))
   384: 
   385:     def reset(self, num_tasks=1):
   386:         self.hidden = self.hidden.new_zeros(1, num_tasks, self.hidden_dim)
   387: 
   388:     def forward(self, context, return_sequence=False):
   389:         """context: (num_tasks, seq_len, feat_dim)."""
   390:         t, s, f = context.size()
   391:         x = context.reshape(t * s, f)
   392:         x = self.fc_pre(x)
   393:         x = x.view(t, s, -1)
   394:         out, hn = self.gru(x, self.hidden)
   395:         self.hidden = hn.detach()
   396:         mean = self.fc_mean(out)
   397:         logvar = self.fc_logvar(out)
   398:         if return_sequence:
   399:             return mean, logvar
   400:         return mean[:, -1, :], logvar[:, -1, :]
   401: 
   402: 
   403: class RewardDecoder(nn.Module):
   404:     """Predict reward from (state, action, belief/z)."""
   405: 
   406:     def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim=200):
   407:         super().__init__()
   408:         self.net = nn.Sequential(
   409:             nn.Linear(obs_dim + action_dim + latent_dim, hidden_dim),
   410:             nn.ReLU(),
   411:             nn.Linear(hidden_dim, hidden_dim),
   412:             nn.ReLU(),
   413:             nn.Linear(hidden_dim, 1),
   414:         )
   415: 
   416:     def forward(self, obs, action, z):
   417:         return self.net(torch.cat([obs, action, z], dim=-1))
   418: 
   419: 
   420: class CustomMetaRLAgent(nn.Module):
   421:     """VariBAD agent: GRU encoder + reward decoder."""
   422: 
   423:     def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
   424:                  reward_dim=1, use_next_obs_in_context=False, **kwargs):
   425:         super().__init__()
   426:         self.obs_dim = obs_dim
   427:         self.action_dim = action_dim
   428:         self.latent_dim = latent_dim
   429:         self.use_next_obs_in_context = use_next_obs_in_context
   430:         self.sparse_rewards = kwargs.get('sparse_rewards', False)
   431: 
   432:         context_input_dim = obs_dim + action_dim + reward_dim
   433:         if use_next_obs_in_context:
   434:             context_input_dim += obs_dim
   435: 
   436:         self.encoder = GRUContextEncoder(
   437:             context_input_dim, hidden_dim=200, latent_dim=latent_dim)
   438:         self.reward_decoder = RewardDecoder(
   439:             obs_dim, action_dim, latent_dim, hidden_dim=200)
   440: 
   441:         self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
   442:         self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
   443:         self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
   444:         self.vf = build_vf(obs_dim, latent_dim, net_size)
   445:         self.target_vf = copy.deepcopy(self.vf)
   446: 
   447:         self.register_buffer('z', torch.zeros(1, latent_dim))
   448:         self.register_buffer('z_means', torch.zeros(1, latent_dim))
   449:         self.register_buffer('z_logvars', torch.zeros(1, latent_dim))
   450:         self._context = None
   451: 
   452:     def clear_context(self, num_tasks=1):
   453:         self.z = ptu.zeros(num_tasks, self.latent_dim)
   454:         self.z_means = ptu.zeros(num_tasks, self.latent_dim)
   455:         self.z_logvars = ptu.zeros(num_tasks, self.latent_dim)
   456:         self.encoder.reset(num_tasks)
   457:         self._context = None
   458: 
   459:     def clear_z(self, num_tasks=1):
   460:         self.clear_context(num_tasks)
   461: 
   462:     def sample_z(self):
   463:         std = torch.exp(0.5 * self.z_logvars)
   464:         eps = torch.randn_like(std)
   465:         self.z = self.z_means + eps * std
   466: 
   467:     @property
   468:     def context(self):
   469:         return self._context
   470: 
   471:     def update_context(self, inputs):
   472:         o, a, r, no, d, info = inputs
   473:         if self.sparse_rewards:
   474:             r = info.get('sparse_reward', r)
   475:         o = ptu.from_numpy(o[None, None, ...])
   476:         a = ptu.from_numpy(a[None, None, ...])
   477:         r = ptu.from_numpy(np.array([r])[None, None, ...])
   478:         no = ptu.from_numpy(no[None, None, ...])
   479:         if self.use_next_obs_in_context:
   480:             data = torch.cat([o, a, r, no], dim=2)
   481:         else:
   482:             data = torch.cat([o, a, r], dim=2)
   483:         if self._context is None:
   484:             self._context = data
   485:         else:
   486:             self._context = torch.cat([self._context, data], dim=1)
   487: 
   488:     def infer_posterior(self, context):
   489:         self.encoder.reset(context.size(0))
   490:         mean, logvar = self.encoder(context)
   491:         self.z_means = mean
   492:         self.z_logvars = logvar
   493:         self.sample_z()
   494: 
   495:     def adapt(self):
   496:         if self._context is not None:
   497:             self.infer_posterior(self._context)
   498: 
   499:     def compute_kl_div(self):
   500:         """KL(q(z|c) || N(0, I))."""
   501:         kl = -0.5 * torch.sum(
   502:             1 + self.z_logvars - self.z_means.pow(2) - self.z_logvars.exp())
   503:         return kl
   504: 
   505:     def get_action(self, obs, deterministic=False):
   506:         z = self.z
   507:         obs_t = ptu.from_numpy(obs[None])
   508:         in_ = torch.cat([obs_t, z], dim=1)
   509:         return self.policy.get_action(in_, deterministic=deterministic)
   510: 
   511:     def forward(self, obs, context):
   512:         self.infer_posterior(context)
   513:         task_z = self.z
   514:         t, b, _ = obs.size()
   515:         obs = obs.view(t * b, -1)
   516:         task_z_rep = [z.repeat(b, 1) for z in task_z]
   517:         task_z_rep = torch.cat(task_z_rep, dim=0)
   518:         in_ = torch.cat([obs, task_z_rep.detach()], dim=1)
   519:         policy_outputs = self.policy(
   520:             in_, reparameterize=True, return_log_prob=True)
   521:         return policy_outputs, task_z_rep
   522: 
   523:     def set_num_steps_total(self, n):
   524:         self.policy.set_num_steps_total(n)
   525: 
   526:     def detach_z(self):
   527:         self.z = self.z.detach()
   528: 
   529:     @property
   530:     def networks(self):
   531:         return [self.policy, self.qf1, self.qf2,
   532:                 self.vf, self.target_vf, self.reward_decoder]
   533: 
   534: 
   535: class CustomMetaRLAlgorithm:
   536:     """VariBAD: SAC + ELBO (reward prediction + KL) loss."""
   537: 
   538:     def __init__(self, agent, env, train_tasks, eval_tasks,
   539:                  replay_buffer, enc_replay_buffer, config):
   540:         self.agent = agent
   541:         self.env = env
   542:         self.train_tasks = train_tasks
   543:         self.eval_tasks = eval_tasks
   544:         self.replay_buffer = replay_buffer
   545:         self.enc_replay_buffer = enc_replay_buffer
   546:         self.config = config
   547: 
   548:         self.sampler = InPlacePathSampler(
   549:             env=env, policy=agent,
   550:             max_path_length=config['max_path_length'],
   551:         )
   552: 
   553:         self.batch_size = config.get('batch_size', 256)
   554:         self.meta_batch = config.get('meta_batch', 16)
   555:         self.discount = config.get('discount', 0.99)
   556:         self.reward_scale = config.get('reward_scale', 5.0)
   557:         self.kl_lambda = config.get('kl_lambda', 0.1)
   558:         self.reward_pred_weight = 1.0
   559:         self.soft_target_tau = config.get('soft_target_tau', 0.005)
   560:         self.sparse_rewards = config.get('sparse_rewards', False)
   561:         self.use_next_obs_in_context = config.get('use_next_obs_in_context', False)
   562:         self.embedding_batch_size = config.get('embedding_batch_size', 64)
   563:         self.embedding_mini_batch_size = config.get('embedding_mini_batch_size', 64)
   564:         self.num_tasks_sample = config.get('num_tasks_sample', 5)
   565:         self.num_steps_prior = config.get('num_steps_prior', 400)
   566:         self.num_steps_posterior = config.get('num_steps_posterior', 0)
   567:         self.num_extra_rl_steps_posterior = config.get('num_extra_rl_steps_posterior', 400)
   568:         self.num_train_steps_per_itr = config.get('num_train_steps_per_itr', 2000)
   569:         self.update_post_train = config.get('update_post_train', 1)
   570: 
   571:         lr = 3e-4
   572:         encoder_params = list(agent.encoder.parameters()) + list(agent.reward_decoder.parameters())
   573:         self.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
   574:         self.qf1_optimizer = optim.Adam(agent.qf1.parameters(), lr=lr)
   575:         self.qf2_optimizer = optim.Adam(agent.qf2.parameters(), lr=lr)
   576:         self.vf_optimizer = optim.Adam(agent.vf.parameters(), lr=lr)
   577:         self.encoder_optimizer = optim.Adam(encoder_params, lr=lr)
   578: 
   579:     def collect_initial_data(self):
   580:         num_initial_steps = self.config.get('num_initial_steps', 200)
   581:         for idx in self.train_tasks:
   582:             self.env.reset_task(idx)
   583:             collect_data(
   584:                 self.agent, self.env, self.sampler,
   585:                 self.replay_buffer, self.enc_replay_buffer,
   586:                 idx, num_initial_steps, 1, np.inf,
   587:                 add_to_enc_buffer=True, config=self.config,
   588:             )
   589: 
   590:     def train_iteration(self, iteration_idx):
   591:         for i in range(self.num_tasks_sample):
   592:             idx = np.random.randint(len(self.train_tasks))
   593:             self.env.reset_task(idx)
   594:             self.enc_replay_buffer.task_buffers[idx].clear()
   595:             if self.num_steps_prior > 0:
   596:                 collect_data(
   597:                     self.agent, self.env, self.sampler,
   598:                     self.replay_buffer, self.enc_replay_buffer,
   599:                     idx, self.num_steps_prior, 1, np.inf,
   600:                     config=self.config,
   601:                 )
   602:             if self.num_steps_posterior > 0:
   603:                 collect_data(
   604:                     self.agent, self.env, self.sampler,
   605:                     self.replay_buffer, self.enc_replay_buffer,
   606:                     idx, self.num_steps_posterior, 1, self.update_post_train,
   607:                     config=self.config,
   608:                 )
   609:             if self.num_extra_rl_steps_posterior > 0:
   610:                 collect_data(
   611:                     self.agent, self.env, self.sampler,
   612:                     self.replay_buffer, self.enc_replay_buffer,
   613:                     idx, self.num_extra_rl_steps_posterior, 1,
   614:                     self.update_post_train,
   615:                     add_to_enc_buffer=False, config=self.config,
   616:                 )
   617: 
   618:         for _ in range(self.num_train_steps_per_itr):
   619:             indices = np.random.choice(
   620:                 self.train_tasks, self.meta_batch)
   621:             self._take_step(indices)
   622:         return {}
   623: 
   624:     def _sample_ordered_context_from_buffer(self, indices, batch_size):
   625:         """Sample chronological trajectory context for VariBAD.
   626: 
   627:         Cap batch_size at max_path_length so each task's context comes
   628:         from a SINGLE trajectory. oyster's random_sequence concatenates
   629:         independent trajectories without resetting the GRU hidden state
   630:         at episode boundaries, polluting the per-step posterior used by
   631:         the ELBO reward decoder.
   632:         """
   633:         if not hasattr(indices, '__iter__'):
   634:             indices = [indices]
   635:         max_path = self.config.get('max_path_length', batch_size)
   636:         bsz = min(batch_size, max_path)
   637:         batches = [ptu.np_to_pytorch_batch(
   638:             self.enc_replay_buffer.random_batch(
   639:                 idx, batch_size=bsz, sequence=True))
   640:             for idx in indices]
   641:         context = [unpack_batch(batch, sparse_reward=self.sparse_rewards)
   642:                    for batch in batches]
   643:         context = [[x[i] for x in context] for i in range(len(context[0]))]
   644:         context = [torch.cat(x, dim=0) for x in context]
   645:         if self.use_next_obs_in_context:
   646:             return torch.cat(context[:-1], dim=2)
   647:         return torch.cat(context[:-2], dim=2)
   648: 
   649:     def _take_step(self, indices):
   650:         mb_size = self.embedding_mini_batch_size
   651:         num_updates = self.embedding_batch_size // mb_size
   652: 
   653:         context_batch = self._sample_ordered_context_from_buffer(
   654:             indices, self.embedding_batch_size)
   655:         self.agent.clear_z(num_tasks=len(indices))
   656: 
   657:         for i in range(num_updates):
   658:             context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
   659:             self._update(indices, context)
   660:             self.agent.detach_z()
   661: 
   662:     def _sample_sac_batch_with_sparse(self, indices):
   663:         """Sample RL batch; also return sparse rewards when needed."""
   664:         batches = [ptu.np_to_pytorch_batch(
   665:             self.replay_buffer.random_batch(idx, batch_size=self.batch_size))
   666:             for idx in indices]
   667:         unpacked = [unpack_batch(b) for b in batches]
   668:         unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
   669:         unpacked = [torch.cat(x, dim=0) for x in unpacked]
   670:         obs, actions, rewards, next_obs, terms = unpacked
   671:         if self.sparse_rewards:
   672:             sparse_r = torch.cat(
   673:                 [b['sparse_rewards'][None, ...] for b in batches], dim=0)
   674:         else:
   675:             sparse_r = rewards
   676:         return obs, actions, rewards, next_obs, terms, sparse_r
   677: 
   678:     def _update(self, indices, context):
   679:         num_tasks = len(indices)
   680:         obs, actions, rewards, next_obs, terms, _ = \
   681:             self._sample_sac_batch_with_sparse(indices)
   682: 
   683:         # Forward: encode context and get policy outputs
   684:         policy_outputs, task_z = self.agent(obs, context)
   685:         new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
   686: 
   687:         t, b, _ = obs.size()
   688:         obs_flat = obs.view(t * b, -1)
   689:         actions_flat = actions.view(t * b, -1)
   690:         next_obs_flat = next_obs.view(t * b, -1)
   691: 
   692:         q1_pred = self.agent.qf1(obs_flat, actions_flat, task_z)
   693:         q2_pred = self.agent.qf2(obs_flat, actions_flat, task_z)
   694:         v_pred = self.agent.vf(obs_flat, task_z.detach())
   695:         with torch.no_grad():
   696:             target_v = self.agent.target_vf(next_obs_flat, task_z)
   697: 
   698:         # --- Encoder loss: KL + reward prediction ---
   699:         # Per-step ELBO with z_seq forces the GRU to predict rewards
   700:         # from partial-belief z_seq[k] (built from only first k context
   701:         # transitions) on cheetah-vel where target velocity is barely
   702:         # observable from a single (s,a) — encoder fails to converge.
   703:         # Fall back to OLD's design: single-posterior z (last step) +
   704:         # decoder predicts the SAC batch's rewards. Ordered context
   705:         # still helps the GRU build a coherent posterior.
   706:         self.encoder_optimizer.zero_grad()
   707: 
   708:         kl_div = -0.5 * torch.sum(
   709:             1 + self.agent.z_logvars - self.agent.z_means.pow(2)
   710:             - self.agent.z_logvars.exp())
   711: 
   712:         z_for_pred = self.agent.z
   713:         z_rep = z_for_pred.unsqueeze(1).expand(-1, b, -1).reshape(t * b, -1)
   714:         rewards_flat = rewards.view(t * b, -1)
   715:         reward_pred = self.agent.reward_decoder(
   716:             obs_flat, actions_flat, z_rep)
   717:         reward_pred_loss = F.mse_loss(reward_pred, rewards_flat)
   718: 
   719:         encoder_loss = (self.kl_lambda * kl_div +
   720:                         self.reward_pred_weight * reward_pred_loss)
   721:         encoder_loss.backward(retain_graph=True)
   722: 
   723:         # Q-function loss
   724:         self.qf1_optimizer.zero_grad()
   725:         self.qf2_optimizer.zero_grad()
   726:         rewards_scaled = rewards_flat * self.reward_scale
   727:         terms_flat = terms.view(self.batch_size * num_tasks, -1)
   728:         q_target = rewards_scaled + (1. - terms_flat) * self.discount * target_v
   729:         qf_loss = (torch.mean((q1_pred - q_target) ** 2) +
   730:                    torch.mean((q2_pred - q_target) ** 2))
   731:         qf_loss.backward()
   732:         self.qf1_optimizer.step()
   733:         self.qf2_optimizer.step()
   734:         self.encoder_optimizer.step()
   735: 
   736:         # V-function loss
   737:         min_q = torch.min(
   738:             self.agent.qf1(obs_flat, new_actions, task_z.detach()),
   739:             self.agent.qf2(obs_flat, new_actions, task_z.detach()),
   740:         )
   741:         v_target = min_q - log_pi
   742:         vf_loss = F.mse_loss(v_pred, v_target.detach())
   743:         self.vf_optimizer.zero_grad()
   744:         vf_loss.backward()
   745:         self.vf_optimizer.step()
   746:         ptu.soft_update_from_to(
   747:             self.agent.vf, self.agent.target_vf, self.soft_target_tau)
   748: 
   749:         # Policy loss
   750:         policy_loss = (log_pi - min_q).mean()
   751:         mean_reg = 1e-3 * (policy_mean ** 2).mean()
   752:         std_reg = 1e-3 * (policy_log_std ** 2).mean()
   753:         policy_loss = policy_loss + mean_reg + std_reg
   754:         self.policy_optimizer.zero_grad()
   755:         policy_loss.backward()
   756:         self.policy_optimizer.step()
   757: 
   758:     @property
   759:     def networks(self):
   760:         return self.agent.networks
   761: 
   762: # =====================================================================
   763: # FIXED — Main entry point
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
