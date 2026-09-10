# Meta-RL: Context Encoder for PEARL Task Inference

## Research Question
Design a context encoder for the PEARL meta-reinforcement learning
algorithm that maps transition tuples `(state, action, reward, next_state)`
to latent task representations. The encoder should enable effective task
inference from limited interaction data so that the agent can adapt
quickly to unseen tasks.

## Background
PEARL (Probabilistic Embeddings for Actor-Critic RL), introduced in
Rakelly et al., "Efficient Off-Policy Meta-Reinforcement Learning via
Probabilistic Context Variables" (arXiv:1903.08254, ICML 2019), is a
meta-RL algorithm that learns a probabilistic latent task variable `z`
from context transitions. During meta-testing, the agent collects a few
transitions from a new task, encodes them into a posterior distribution
`q(z|c)`, and conditions an SAC-style policy on samples of `z`.

The context encoder processes individual transition tuples and outputs
Gaussian parameters (mean and log-variance). The PEARLAgent aggregates
per-transition outputs via product of Gaussians to form the task
posterior. The SAC policy/value backbone, replay buffers, task sampling,
and outer training loop are all fixed; only the encoder architecture is
open.

You will modify the `CustomContextEncoder` class and may add custom
imports inside the editable region of `custom_encoder.py`.

## Interface
Your `CustomContextEncoder` must:
- Extend `PyTorchModule` and call `self.save_init_params(locals())` in
  `__init__`.
- Accept `hidden_sizes`, `input_size`, `output_size` as constructor
  arguments.
- Set the `self.output_size` attribute in `__init__`.
- Implement `forward(self, input, return_preactivations=False)` returning
  a tensor of shape `(*, output_size)`.
- Implement `reset(self, num_tasks=1)` to reset any stateful components
  such as recurrent hidden state.

## Reference Architectures
- **MLP encoder** — independent per-transition MLP (the original PEARL
  encoder; Rakelly et al., 2019).
- **Recurrent encoder** — GRU over the context sequence in the spirit of
  VariBAD (Zintgraf et al., "VariBAD: A Very Good Method for Bayes-Adaptive
  Deep RL via Meta-Learning", arXiv:1910.08348, ICLR 2020).
- **Attention encoder** — a small Transformer-style aggregator over the
  context tuples.

## Environments
The encoder is evaluated across MuJoCo and point-robot meta-RL task
families with different reward structures:

1. **Half-Cheetah Velocity** (`cheetah-vel`): 30 train / 10 test tasks,
   target velocities in `[0, 3]` m/s. Obs dim 20, action dim 6. Dense
   reward based on velocity matching. Tests encoding quality on a
   continuous task distribution with high-dimensional observations.

2. **Sparse Point Robot** (`sparse-point-robot`): 40 train / 10 test
   tasks. Goals on a half-circle, sparse reward (+1 within goal radius,
   0 otherwise). Obs dim 2, action dim 2. Tests the encoder's ability to
   extract task information from sparse reward signals.

3. **Point Robot** (`point-robot`): 40 train / 10 test tasks. Goals
   sampled uniformly from `[-1, 1]^2`. Dense reward (negative L2 distance
   to goal). Obs dim 2, action dim 2. A simpler diverse continuous task
   distribution.

## Evaluation
Performance is measured by `meta_test_return` on each environment:
average return on held-out test tasks after meta-training under this
benchmark's fixed budget. Higher is better.

## Note on Training Budget
This task intentionally uses a short fixed meta-training budget (20 outer
iterations) to keep wall time per environment near 1 hour. This is far
shorter than the 500+ iteration budgets used in the PEARL/VariBAD/FOCAL
papers (roughly 1.5e6–2.0e6 environment steps), so absolute returns are
not directly comparable to those papers; only relative ordering across
baselines and agents within this fixed budget is meaningful.

On `sparse-point-robot`, methods that report 0 indicate no goal was
reached within the budget rather than algorithmic failure, since the
environment reward is binary.

The companion [`meta-rl-algorithm`](../meta-rl-algorithm/task_description.md)
task uses the same budget convention.
