# VAE Loss Function Design for Image Reconstruction

## Objective

Design a training loss function for a Variational Autoencoder (VAE) that
achieves the best reconstruction quality on CIFAR-10, under a fixed
`AutoencoderKL` architecture, optimizer, and evaluation protocol.

## Background

Variational Autoencoders encode images into a compressed latent representation
and decode them back. Reconstruction quality depends critically on the
training loss. Standard ingredients combine:

- **Pixel reconstruction loss** — L1 or L2 between reconstruction and target.
- **KL divergence** — regularizes the encoder posterior toward a standard
  normal prior.
- **Perceptual loss** — LPIPS (Zhang et al., CVPR 2018) or VGG feature
  matching, encouraging perceptual rather than pixel-exact match.
- **Adversarial loss** — a PatchGAN discriminator (Isola et al., 2017) for
  sharpness, as in VQGAN (Esser et al., CVPR 2021, arXiv:2012.09841).
- **Frequency-domain loss** — FFT-based weighting to preserve fine detail.

Recent work on the **Prism Hypothesis** (Fan et al., UAE,
arXiv:2512.19693) argues that natural images decompose into a compact
low-frequency semantic component and residual higher-frequency detail, and
that explicitly handling these bands during training improves both semantic
and pixel-level reconstruction quality. The key intuition is that semantic
content concentrates at low frequencies while fine perceptual detail lives in
higher bands, motivating frequency-aware loss design.

## Implementation Contract

Implement the `VAELoss` class in `custom_train.py`. The loss is used to train
an `AutoencoderKL` (from `diffusers`) on CIFAR-10 32×32 images.

```python
class VAELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Initialize loss components here.

    def forward(self, recon, target, posterior, step):
        # recon:     [B, 3, 32, 32] reconstructed images in [-1, 1].
        # target:    [B, 3, 32, 32] original images in [-1, 1].
        # posterior: DiagonalGaussianDistribution.
        #              - posterior.kl() -> KL divergence per sample.
        #              - posterior.mean, posterior.logvar.
        # step:      current training step (int).
        # Return: (loss_tensor, metrics_dict).
        ...
```

### Available Libraries

- `torch`, `torch.nn`, `torch.nn.functional`.
- `torch.fft` — frequency-domain ops (`fft2`, `ifft2`, `fftshift`, …).
- `lpips` — learned perceptual loss (`lpips.LPIPS(net='vgg').to(device)`).
- `numpy`, `math`.

## Fixed Pipeline

Architecture (fixed):

- `AutoencoderKL` from `diffusers`, 3 blocks and 2 downsample stages, latent
  resolution 8×8 (compression factor f = 4) for 32×32 input.
- `latent_channels=4`, `layers_per_block=2`.
- GroupNorm (32 groups) + SiLU activation.

Channel widths and latent channels scale via environment variables across
training scales:

- Small:  `BLOCK_OUT_CHANNELS=(64, 128, 256)`,  `LATENT_CHANNELS=4`,  20,000 steps.
- Medium: `BLOCK_OUT_CHANNELS=(96, 192, 384)`,  `LATENT_CHANNELS=8`,  30,000 steps.
- Large:  `BLOCK_OUT_CHANNELS=(128, 256, 512)`, `LATENT_CHANNELS=16`, 30,000 steps.

Training (fixed):

- Optimizer: AdamW, lr = 2e-4, weight_decay = 1e-6.
- LR schedule: 5% warmup followed by constant LR.
- Mixed precision (autocast + GradScaler).
- Gradient clipping at 1.0.
- EMA with rate 0.999.

## Baselines

| Baseline     | Description |
|--------------|-------------|
| `l2-kl`      | Simplest VAE loss: pixel-level L2 reconstruction + KL regularization (Kingma & Welling, ICLR 2014). |
| `perceptual` | MSE + LPIPS (Zhang et al., CVPR 2018) + KL — adds learned perceptual similarity over VGG features. |
| `vqgan`      | Multi-objective VQGAN-style recipe (Esser et al., CVPR 2021, arXiv:2012.09841): L1 reconstruction + LPIPS perceptual + PatchGAN adversarial loss + KL. |

## Evaluation

Reconstruction quality is measured on the full CIFAR-10 test set
(10,000 images):

| Metric  | Direction        | Description |
|---------|------------------|-------------|
| **rFID**  | lower is better  | Reconstruction FID between original and reconstructed test images (primary metric). |
| **PSNR**  | higher is better | Peak signal-to-noise ratio in dB. |
| **SSIM**  | higher is better | Structural similarity index. |

Task scoring uses best reconstruction FID per scale; PSNR and SSIM are
supporting diagnostics. The contribution should be the loss design only — not
changes to architecture, data pipeline, training schedule, or evaluation.
