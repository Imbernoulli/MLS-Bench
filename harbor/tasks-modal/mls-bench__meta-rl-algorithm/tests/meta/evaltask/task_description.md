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
