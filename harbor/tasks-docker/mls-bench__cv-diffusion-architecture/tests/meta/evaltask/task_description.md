# Diffusion Model Architecture Design

## Objective

Design a UNet backbone for unconditional CIFAR-10 diffusion that achieves
lower FID than the standard DDPM-style architectures, under a fixed training
target (epsilon prediction), DDIM sampler, optimizer, and noise schedule.

## Background

The UNet (Ronneberger et al., 2015) is the standard architecture for the
denoising network in DDPMs (Ho et al., 2020, arXiv:2006.11239). Key
architectural choices include:

- **Block types**: pure convolutional residual blocks (`DownBlock2D` /
  `UpBlock2D`) or blocks with self-attention (`AttnDownBlock2D` /
  `AttnUpBlock2D`), and at which resolution levels they are placed.
- **Attention placement**: self-attention is expensive at high spatial
  resolutions (32×32) but may improve global coherence. The original DDPM
  places self-attention only at the 16×16 resolution stage.
- **Depth and normalization**: `layers_per_block`, `norm_num_groups`,
  `attention_head_dim`, channel multipliers, etc.
- **Custom modules**: hybrid convolution / transformer blocks, gated blocks,
  multi-scale fusion, or new architectures entirely, as long as they satisfy
  the input / output interface.

## Implementation Contract

You are given `custom_train.py`, a self-contained unconditional DDPM training
script on CIFAR-10. Everything is fixed except the `build_model(device)`
function, which must return a denoiser satisfying:

- **Input**: `(x, timestep)` where `x` is `[B, 3, 32, 32]`, `timestep` is
  `[B]`.
- **Output**: an object with a `.sample` attribute of shape `[B, 3, 32, 32]`
  representing the predicted epsilon.

`UNet2DModel` from `diffusers` already satisfies this interface, but you may
also build a fully custom `nn.Module`.

Channel widths are passed via the `BLOCK_OUT_CHANNELS` environment variable
(e.g. `"128,256,256,256"`) so that the same architecture scales across
evaluation tiers. `LAYERS_PER_BLOCK` (default 2) is also available.

## Fixed Pipeline

The following are fixed across baselines and submissions:

- Dataset: CIFAR-10 (32×32, unconditional).
- Training target: epsilon prediction with MSE loss.
- Optimizer: AdamW, learning rate 2e-4, EMA rate 0.9995.
- Training: 35,000 steps per scale.
- Inference: 50-step DDIM sampling (Song et al., 2020, arXiv:2010.02502).
- Metric: FID computed by clean-fid against the CIFAR-10 train set
  (50,000 samples), lower is better.
- Channel scales:
  - Small:  `block_out_channels=(64, 128, 128, 128)`, ~9M params, batch 128.
  - Medium: `block_out_channels=(128, 256, 256, 256)`, ~36M params, batch 128.
  - Large:  `block_out_channels=(160, 320, 320, 320)`, ~55M params, batch 128.

## Baselines

| Baseline    | Description |
|-------------|-------------|
| `standard`  | Original DDPM architecture (Ho et al., 2020, arXiv:2006.11239). Self-attention only at the 16×16 resolution. Matches the `google/ddpm-cifar10-32` configuration. |
| `full-attn` | Self-attention at every resolution (32×32, 16×16, 8×8, 4×4). More expressive but significantly more compute and memory per step. |
| `no-attn`   | Pure convolutional UNet with no per-resolution self-attention; only the mid-block retains its default self-attention layer. Smallest and fastest. |

## Evaluation

Evaluation trains the candidate architecture at the multiple channel scales
above and scores generated samples with clean-fid against the CIFAR-10 train
set (50,000 samples); lower FID is better. The architecture must preserve the
denoising interface: it receives images and timesteps and returns a same-shaped
noise prediction.

Improvements should come from transferable architecture design, not from
changes to data, loss target, optimizer, sampler, or evaluation.
