# Cooperative MARL: Centralized Critic Architecture for MAPPO

## Research Question
Improve cooperative multi-agent reinforcement learning by designing a better
**centralized critic architecture** for MAPPO (Multi-Agent PPO). You will
modify the `CustomCritic` class and may add custom imports inside the
editable region of `custom_critic.py`.

## Background
In cooperative MARL with partial observability, each agent only sees a
local observation but the team shares a common reward. Centralized-Training-
with-Decentralized-Execution (CTDE) methods train a centralized value
function during training (which can see the global state and possibly all
agents' information) and use it to reduce variance when computing
advantages for each agent's decentralized policy gradient update. The
architecture of this centralized critic — what it conditions on and how it
mixes per-agent features — directly determines the bias-variance tradeoff
and how well MAPPO scales to hard cooperation tasks.

The training uses EPyMARL's `ppo_learner` with the MAPPO default
hyperparameters from Yu et al. (2022) on cooperative SMAC maps via
**smaclite**, a pure-Python reimplementation of the StarCraft Multi-Agent
Challenge benchmark that does not require the StarCraft II binary. Each map
trains for roughly 5M environment steps. The actor architecture, learner,
optimizer, GAE settings, and environment interface are fixed.

## Interface
Your `CustomCritic` must:
- Inherit from `nn.Module`.
- Accept `(scheme, args)` in `__init__`, where:
  - `scheme["state"]["vshape"]` — global state dim
  - `scheme["obs"]["vshape"]` — per-agent observation dim
  - `args.n_agents`, `args.n_actions`, `args.hidden_dim`,
    `args.obs_last_action`, `args.obs_individual_obs`
- Set `self.output_type = "v"` in `__init__`.
- Implement `forward(self, batch, t=None)` where:
  - `batch["state"]` has shape `(B, T, state_dim)`
  - `batch["obs"]` has shape `(B, T, n_agents, obs_dim)`
  - `batch.batch_size`, `batch.max_seq_length`, `batch.device` are
    available
  - `t=None` means "whole sequence"; otherwise `t` is an integer time
    index
  - Returns `q` with shape `(B, T, n_agents, 1)`. The learner later does
    `.squeeze(3)`, so the trailing singleton is mandatory.

## Reference Implementations
The following baselines are provided as `*.edit.py` files for reference and
serve as design points spanning the literature:

- **IPPO critic** — per-agent MLP over `batch["obs"]` ⊕ agent-one-hot, no
  centralization. Floor baseline corresponding to the IPPO ablation in Yu
  et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent
  Games" (arXiv:2103.01955, NeurIPS Datasets and Benchmarks 2022). See
  also `epymarl/src/modules/critics/ac.py`.
- **MAPPO critic** — shared MLP over `(batch["state"] ⊕ agent-one-hot)`.
  The standard MAPPO central V from the same paper. See also
  `epymarl/src/modules/critics/centralV.py`.
- **MAT-style attention critic** — projects per-agent features
  `(obs ⊕ broadcast state)` into tokens, then a single
  `TransformerEncoder` layer with self-attention across the agent axis
  produces a per-agent value. Adapted (critic-only form) from Wen et al.,
  "Multi-Agent Reinforcement Learning is a Sequence Modeling Problem"
  (arXiv:2205.14953, NeurIPS 2022); the MAPPO actor is kept unchanged.

## Evaluation
Performance is measured by **test win rate** (`battle_won_mean`) averaged
over the SMAC test episodes with the greedy policy, evaluated separately
per map and recorded under setup-specific metric keys:

- Primary: `test_battle_won_mean_<map>` (higher is better)
- Secondary: `test_return_mean_<map>` (higher is better)

A strong centralized critic should generalize across cooperative maps of
varying difficulty rather than specialize to one scenario.
