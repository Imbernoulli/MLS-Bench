# Class-Conditional Diffusion: Conditioning Injection Methods

## Objective

Design a conditioning injection method that improves class-conditional
CIFAR-10 diffusion FID under a fixed denoiser scaling, training procedure, and
DDIM sampler.

## Background

Class-conditional diffusion models generate images conditioned on a class
label. The key design choice is **how** the class information is injected into
the denoiser. Three established families:

- **Cross-Attention.** Class embedding serves as key / value in a
  cross-attention layer after each ResBlock; this is the mechanism used by
  Stable Diffusion (Rombach et al., CVPR 2022) for text conditioning.
- **Adaptive LayerNorm — AdaLN-Zero** (Peebles & Xie, ICCV 2023, DiT,
  arXiv:2212.09748). Class embedding generates per-layer scale, shift, and
  residual-gate parameters that modulate LayerNorm; the gate is initialized
  to zero so each block starts as the identity.
- **FiLM-style conditioning** (Perez et al., AAAI 2018, "FiLM: Visual
  Reasoning with a General Conditioning Layer"). Class embedding is added to
  the timestep embedding and injected via adaptive GroupNorm (scale / shift)
  inside ResBlocks.

## Implementation Contract

You are given `custom_train.py`, a self-contained class-conditional DDPM
training script with a small UNet on CIFAR-10 (32×32, 10 classes). The
editable region exposes two pieces:

1. `prepare_conditioning(time_emb, class_emb)` — controls how class embedding
   is combined with the timestep embedding before entering ResBlocks.
2. `ClassConditioner(nn.Module)` — a conditioning module applied after each
   ResBlock, enabling methods like cross-attention or adaptive normalization.

Both pieces must keep the denoising interface (`(x, timestep, class_id)` →
predicted epsilon of the same shape as `x`) and the class-conditioning
semantics.

## Fixed Pipeline

The following are fixed across baselines and submissions:

- Dataset: CIFAR-10 (32×32, 10 classes).
- Model: `UNet2DModel` (diffusers backbone) at three channel scales:
  - Small:  `block_out_channels=(64, 128, 128, 128)`, ~9M params, batch 128.
  - Medium: `block_out_channels=(128, 256, 256, 256)`, ~36M params, batch 128.
  - Large:  `block_out_channels=(160, 320, 320, 320)`, ~55M params, batch 128.
- Training: 35,000 steps per scale, AdamW lr=2e-4, EMA rate 0.9995.
- Inference: 50-step DDIM sampling (Song et al., 2020, arXiv:2010.02502),
  class-conditional.
- Metric: FID computed by clean-fid against the CIFAR-10 train set
  (50,000 samples), lower is better.

## Baselines

| Baseline      | Description |
|---------------|-------------|
| `concat-film` | FiLM-style conditioning (Perez et al., AAAI 2018): add class embedding to timestep embedding, inject via adaptive GroupNorm in ResBlocks. Simplest. |
| `cross-attn`  | Cross-attention conditioning: class embedding is key / value in cross-attention layers after each ResBlock. Most expressive. |
| `adanorm`     | DiT-style AdaLN-Zero conditioning (Peebles & Xie, ICCV 2023, arXiv:2212.09748): class embedding generates scale / shift / gate parameters for adaptive normalization, with the residual gate initialized to zero. |

## Evaluation

Evaluation trains the candidate conditioning at the channel scales above and
scores generated samples with clean-fid against CIFAR-10; lower FID is better.
The improvement should come from a transferable conditioning design, not from
changes to the dataset, labels, loss, optimizer, sampler, or metric.
