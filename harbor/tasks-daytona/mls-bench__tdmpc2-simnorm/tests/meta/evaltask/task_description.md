# Latent Representation Normalization for Model-Based RL

## Objective
Design and implement a custom normalization technique for latent state representations in model-based reinforcement learning. Your code goes in the `CustomSimNorm` class in `custom_simnorm.py`. This normalization is applied as the final activation in both the encoder and dynamics networks of the TD-MPC2 world model.

## Background
**TD-MPC2** (Hansen, Su, Wang, ICLR 2024, arXiv:2310.16828) learns an implicit world model in a latent space and uses it for planning. The latent representation geometry is critical for stable learning. The default approach uses **SimNorm (Simplicial Normalization)**, introduced in the TD-MPC2 paper, which reshapes the latent vector into groups of 8 and applies softmax within each group, constraining each group to lie on a simplex.

Alternative normalization strategies could improve learning stability, representation quality, or computational efficiency:
- **L2 normalization**: projects onto a hypersphere.
- **RMSNorm**: root-mean-square normalization without mean centering.
- **Spectral normalization**: controls the Lipschitz constant.
- **Gumbel-softmax**: adds stochasticity to the simplex projection.
- **Hybrid approaches**: combining different normalization strategies.

## What You Can Modify
The `CustomSimNorm` class in `custom_simnorm.py`:
- `__init__(self, cfg)`: initialize parameters (`cfg.simnorm_dim = 8`)
- `forward(self, x)`: normalize the latent vector (must preserve shape)

## Evaluation
- **Metric**: episode reward (higher is better)
- **Environments**: DMControl walker-walk and cheetah-run
- **Model**: TD-MPC2 with 1M parameters, 200K training steps

## Architecture Context
The normalization is used in:
1. **Encoder** (`layers.py: enc()`): maps observations to latent states.
2. **Dynamics** (`world_model.py: __init__`): predicts next latent state from current state + action.

Both use SimNorm as the final activation in their MLP stacks. The latent dimension is 128 with `simnorm_dim = 8` (16 groups).
