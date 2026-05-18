# MLS-Bench: cv-vae-loss

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

- Optimizer: AdamW, lr = 4e-4, weight_decay = 1e-4.
- LR schedule: 5% warmup + cosine decay.
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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/diffusers-main/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `diffusers-main/custom_train.py`
- editable lines **32–76**




## Readable Context


### `diffusers-main/custom_train.py`  [EDITABLE — lines 32–76 only]

```python
     1: """VAE Training on CIFAR-10 with configurable loss function.
     2: 
     3: Uses AutoencoderKL architecture (fixed). Only the loss function is editable.
     4: """
     5: 
     6: import copy
     7: import math
     8: import os
     9: import shutil
    10: import sys
    11: import time
    12: from datetime import timedelta
    13: 
    14: import numpy as np
    15: import torch
    16: import torch.nn as nn
    17: import torch.distributed as dist
    18: import torch.nn.functional as F
    19: from PIL import Image
    20: from torch.nn.parallel import DistributedDataParallel as DDP
    21: from torchvision import datasets, transforms
    22: 
    23: # Use diffusers from the external package
    24: sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    25: from diffusers import AutoencoderKL
    26: 
    27: 
    28: # ============================================================================
    29: # Loss Function (EDITABLE REGION)
    30: # ============================================================================
    31: 
    32: class VAELoss(nn.Module):
    33:     """VAE training loss function.
    34: 
    35:     TODO: Design your loss function for training a KL-regularized VAE.
    36: 
    37:     The loss function receives:
    38:         recon:     Reconstructed images [B, 3, 32, 32] in [-1, 1]
    39:         target:    Original images [B, 3, 32, 32] in [-1, 1]
    40:         posterior: DiagonalGaussianDistribution from the encoder
    41:                    - posterior.kl()       -> KL divergence [B, ...]
    42:                    - posterior.mean       -> latent mean
    43:                    - posterior.logvar     -> latent log-variance
    44:         step:      Current training step (int)
    45: 
    46:     Must return: (total_loss, metrics_dict)
    47:         total_loss: scalar tensor (for backpropagation)
    48:         metrics_dict: dict with string keys and float values for logging
    49: 
    50:     Available imports: torch, torch.nn, torch.nn.functional, numpy, lpips
    51:     The lpips package provides learned perceptual loss:
    52:         loss_fn = lpips.LPIPS(net='vgg').to(device)
    53:         p_loss = loss_fn(recon, target).mean()
    54: 
    55:     You may also use torch.fft for frequency-domain operations, or any
    56:     other approach you think will improve reconstruction quality.
    57: 
    58:     Evaluation metrics (for reference, you do NOT compute these):
    59:         - rFID: Reconstruction FID (lower is better)
    60:         - PSNR: Peak signal-to-noise ratio in dB (higher is better)
    61:         - SSIM: Structural similarity (higher is better)
    62:     """
    63: 
    64:     def __init__(self, device):
    65:         super().__init__()
    66:         raise NotImplementedError("Implement VAELoss.__init__")
    67: 
    68:     def forward(self, recon, target, posterior, step):
    69:         """Compute total VAE loss.
    70: 
    71:         Returns:
    72:             loss: scalar tensor (total training loss for backpropagation)
    73:             metrics: dict of {str: float} for logging
    74:         """
    75:         raise NotImplementedError("Implement VAELoss.forward")
    76: 
    77: 
    78: # ============================================================================
    79: # Fixed: Model Architecture
    80: # ============================================================================
    81: 
    82: def build_model(device):
    83:     """Build AutoencoderKL with channels from environment variable.
    84: 
    85:     Uses 3 blocks (2 downsample stages) for 32x32 input → 8x8 latent (f=4).
    86:     """
    87:     channels = (128, 256, 512)
    88:     if os.environ.get('BLOCK_OUT_CHANNELS'):
    89:         channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    90:     layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))
    91:     latent_ch = int(os.environ.get('LATENT_CHANNELS', 4))
    92: 
    93:     return AutoencoderKL(
    94:         in_channels=3,
    95:         out_channels=3,
    96:         latent_channels=latent_ch,
    97:         block_out_channels=channels,
    98:         down_block_types=tuple(["DownEncoderBlock2D"] * len(channels)),
    99:         up_block_types=tuple(["UpDecoderBlock2D"] * len(channels)),
   100:         layers_per_block=layers,
   101:         norm_num_groups=32,
   102:         act_fn="silu",
   103:         sample_size=32,
   104:         scaling_factor=0.18215,
   105:     ).to(device)
   106: 
   107: 
   108: class VAEWrapper(nn.Module):
   109:     """Wrapper for encode+decode in a single forward pass (needed for DDP)."""
   110: 
   111:     def __init__(self, vae):
   112:         super().__init__()
   113:         self.vae = vae
   114: 
   115:     def forward(self, x):
   116:         posterior = self.vae.encode(x).latent_dist
   117:         z = posterior.sample()
   118:         recon = self.vae.decode(z).sample
   119:         return recon, posterior
   120: 
   121: 
   122: # ============================================================================
   123: # Fixed: Evaluation Metrics
   124: # ============================================================================
   125: 
   126: def compute_psnr(img1, img2):
   127:     """PSNR between image tensors in [0, 1]. Returns dB value."""
   128:     mse = F.mse_loss(img1, img2)
   129:     if mse == 0:
   130:         return torch.tensor(100.0)
   131:     return (10 * torch.log10(1.0 / mse)).clamp(max=100.0)
   132: 
   133: 
   134: def _gaussian_window(size, sigma, channels, device):
   135:     coords = torch.arange(size, device=device).float() - size // 2
   136:     g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
   137:     g = g / g.sum()
   138:     window = g.outer(g).unsqueeze(0).unsqueeze(0)
   139:     return window.expand(channels, 1, size, size).contiguous()
   140: 
   141: 
   142: def compute_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
   143:     """SSIM between image tensors in [0, 1]."""
   144:     channels = img1.shape[1]
   145:     window = _gaussian_window(window_size, 1.5, channels, img1.device)
   146:     pad = window_size // 2
   147: 
   148:     mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
   149:     mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
   150:     mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
   151:     mu1_mu2 = mu1 * mu2
   152: 
   153:     sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
   154:     sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
   155:     sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2
   156: 
   157:     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
   158:                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
   159:     return ssim_map.mean()
   160: 
   161: 
   162: def evaluate_reconstruction(model, test_loader, device, output_dir,
   163:                             rank=0, world_size=1):
   164:     """Evaluate VAE reconstruction: rFID, PSNR, SSIM on test set.
   165: 
   166:     Saves images to /dev/shm (RAM tmpfs) to avoid slow disk I/O,
   167:     then uses cleanfid's proven compute_fid path.
   168:     """
   169:     from cleanfid import fid as cleanfid
   170: 
   171:     model.eval()
   172:     psnr_sum, ssim_sum, count = 0.0, 0.0, 0
   173: 
   174:     # Use RAM-backed tmpfs for image I/O (much faster than disk)
   175:     orig_dir = "/dev/shm/_eval_orig"
   176:     recon_dir = "/dev/shm/_eval_recon"
   177: 
   178:     if rank == 0:
   179:         for d in [orig_dir, recon_dir]:
   180:             if os.path.exists(d):
   181:                 shutil.rmtree(d)
   182:             os.makedirs(d)
   183:     if world_size > 1:
   184:         dist.barrier()
   185: 
   186:     sample_pairs = []
   187:     idx = 0
   188: 
   189:     with torch.no_grad():
   190:         for x, _ in test_loader:
   191:             x = x.to(device)
   192:             with torch.amp.autocast(device_type='cuda'):
   193:                 recon, _ = model(x)
   194: 
   195:             x_01 = (x * 0.5 + 0.5).float()
   196:             recon_01 = recon.clamp(-1, 1).float() * 0.5 + 0.5
   197: 
   198:             psnr_sum += compute_psnr(recon_01, x_01).item() * x.shape[0]
   199:             ssim_sum += compute_ssim(recon_01, x_01).item() * x.shape[0]
   200:             count += x.shape[0]
   201: 
   202:             if rank == 0:
   203:                 # Collect first 10 pairs for visual comparison
   204:                 if len(sample_pairs) < 10:
   205:                     n = min(10 - len(sample_pairs), x.shape[0])
   206:                     for j in range(n):
   207:                         sample_pairs.append((
   208:                             (x_01[j] * 255).clamp(0, 255).byte().cpu(),
   209:                             (recon_01[j] * 255).clamp(0, 255).byte().cpu(),
   210:                         ))
   211: 
   212:                 # Save to RAM tmpfs for FID computation
   213:                 x_uint8 = (x_01 * 255).clamp(0, 255).byte().cpu()
   214:                 r_uint8 = (recon_01 * 255).clamp(0, 255).byte().cpu()
   215:                 for j in range(x.shape[0]):
   216:                     Image.fromarray(x_uint8[j].permute(1, 2, 0).numpy()).save(
   217:                         os.path.join(orig_dir, f'{idx:05d}.png'))
   218:                     Image.fromarray(r_uint8[j].permute(1, 2, 0).numpy()).save(
   219:                         os.path.join(recon_dir, f'{idx:05d}.png'))
   220:                     idx += 1
   221: 
   222:     if world_size > 1:
   223:         dist.barrier()
   224: 
   225:     avg_psnr = psnr_sum / max(count, 1)
   226:     avg_ssim = ssim_sum / max(count, 1)
   227: 
   228:     rfid = None
   229:     if rank == 0:
   230:         import cleanfid.features as _feat
   231: 
   232:         cache_dir = "/data/cleanfid"
   233:         os.makedirs(cache_dir, exist_ok=True)
   234: 
   235:         # Patch cleanfid to load inception from image cache (no network needed)
   236:         _orig_build = _feat.build_feature_extractor
   237:         def _patched_build(mode, device=device, use_dataparallel=True):
   238:             from cleanfid.inception_torchscript import InceptionV3W
   239:             m = InceptionV3W(cache_dir, download=False,
   240:                              resize_inside=(mode == "legacy_tensorflow")).to(device)
   241:             m.eval()
   242:             if use_dataparallel:
   243:                 m = torch.nn.DataParallel(m)
   244:             return lambda x: m(x)
   245:         _feat.build_feature_extractor = _patched_build
   246: 
   247:         rfid = cleanfid.compute_fid(
   248:             orig_dir, recon_dir,
   249:             device=device, batch_size=64, verbose=False,
   250:         )
   251: 
   252:         _feat.build_feature_extractor = _orig_build
   253: 
   254:         # Save 10 sample comparisons
   255:         sample_dir = os.path.join(output_dir, 'samples')
   256:         os.makedirs(sample_dir, exist_ok=True)
   257:         for i, (orig_t, recon_t) in enumerate(sample_pairs):
   258:             o = Image.fromarray(orig_t.permute(1, 2, 0).numpy())
   259:             r = Image.fromarray(recon_t.permute(1, 2, 0).numpy())
   260:             cmp = Image.new('RGB', (o.width * 2 + 4, o.height), (128, 128, 128))
   261:             cmp.paste(o, (0, 0))
   262:             cmp.paste(r, (o.width + 4, 0))
   263:             cmp.save(os.path.join(sample_dir, f'cmp_{i:02d}.png'))
   264: 
   265:         shutil.rmtree(orig_dir, ignore_errors=True)
   266:         shutil.rmtree(recon_dir, ignore_errors=True)
   267: 
   268:     if world_size > 1:
   269:         dist.barrier()
   270: 
   271:     model.train()
   272:     return rfid, avg_psnr, avg_ssim
   273: 
   274: 
   275: # ============================================================================
   276: # Training Script
   277: # ============================================================================
   278: 
   279: if __name__ == '__main__':
   280:     seed = int(os.environ.get('SEED', 42))
   281:     data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
   282:     output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
   283:     max_steps = int(os.environ.get('MAX_STEPS', 10000))
   284:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 10000))
   285:     batch_size = int(os.environ.get('BATCH_SIZE', 128))
   286:     lr = float(os.environ.get('LR', 1e-4))
   287:     ema_rate = float(os.environ.get('EMA_RATE', 0.999))
   288: 
   289:     # ── DDP setup ──────────────────────────────────────────────────────────
   290:     use_ddp = 'RANK' in os.environ
   291:     if use_ddp:
   292:         dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
   293:         local_rank = int(os.environ['LOCAL_RANK'])
   294:         rank = int(os.environ['RANK'])
   295:         world_size = int(os.environ['WORLD_SIZE'])
   296:         device = torch.device(f'cuda:{local_rank}')
   297:         torch.cuda.set_device(device)
   298:         is_main = (rank == 0)
   299:     else:
   300:         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   301:         rank = 0
   302:         world_size = 1
   303:         is_main = True
   304: 
   305:     torch.manual_seed(seed + rank)
   306:     os.makedirs(output_dir, exist_ok=True)
   307: 
   308:     # ── Data ────────────────────────────────────────────────────────────────
   309:     transform = transforms.Compose([
   310:         transforms.RandomHorizontalFlip(),
   311:         transforms.ToTensor(),
   312:         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
   313:     ])
   314:     train_dataset = datasets.CIFAR10(data_dir, train=True, transform=transform,
   315:                                      download=False)
   316:     test_transform = transforms.Compose([
   317:         transforms.ToTensor(),
   318:         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
   319:     ])
   320:     test_dataset = datasets.CIFAR10(data_dir, train=False,
   321:                                     transform=test_transform, download=False)
   322: 
   323:     if use_ddp:
   324:         sampler = torch.utils.data.DistributedSampler(
   325:             train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
   326:         train_loader = torch.utils.data.DataLoader(
   327:             train_dataset, batch_size=batch_size, sampler=sampler,
   328:             num_workers=4, pin_memory=True, drop_last=True)
   329:     else:
   330:         train_loader = torch.utils.data.DataLoader(
   331:             train_dataset, batch_size=batch_size, shuffle=True,
   332:             num_workers=4, pin_memory=True, drop_last=True)
   333: 
   334:     test_loader = torch.utils.data.DataLoader(
   335:         test_dataset, batch_size=256, shuffle=False,
   336:         num_workers=4, pin_memory=True)
   337:     data_iter = iter(train_loader)
   338: 
   339:     # ── Model ───────────────────────────────────────────────────────────────
   340:     vae = build_model(device)
   341:     wrapper = VAEWrapper(vae)
   342:     ema_vae = copy.deepcopy(vae)
   343:     ema_vae.requires_grad_(False)
   344:     ema_wrapper = VAEWrapper(ema_vae)
   345: 
   346:     if use_ddp:
   347:         wrapper = DDP(wrapper, device_ids=[local_rank])
   348: 
   349:     # ── Loss & optimizer ────────────────────────────────────────────────────
   350:     criterion = VAELoss(device)
   351:     optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-6)
   352:     warmup_steps = int(max_steps * 0.05)
   353:     def lr_lambda(step):
   354:         if step < warmup_steps:
   355:             return step / max(warmup_steps, 1)
   356:         return 1.0  # constant LR after warmup
   357:     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
   358:     scaler = torch.amp.GradScaler()
   359: 
   360:     num_params = sum(p.numel() for p in vae.parameters())
   361:     if is_main:
   362:         print(f"Model parameters: {num_params/1e6:.1f}M | GPUs: {world_size}",
   363:               flush=True)
   364: 
   365:     # ── Training loop ────────────────────────────────────────────────────────
   366:     best_rfid = float('inf')
   367:     t0 = time.time()
   368:     epoch = 0
   369: 
   370:     for step in range(1, max_steps + 1):
   371:         try:
   372:             x, _ = next(data_iter)
   373:         except StopIteration:
   374:             epoch += 1
   375:             if use_ddp:
   376:                 sampler.set_epoch(epoch)
   377:             data_iter = iter(train_loader)
   378:             x, _ = next(data_iter)
   379: 
   380:         x = x.to(device)
   381: 
   382:         with torch.amp.autocast(device_type='cuda'):
   383:             recon, posterior = wrapper(x)
   384:             loss, metrics = criterion(recon, x, posterior, step)
   385: 
   386:         # GAN support: both VAE and disc update every step (not alternating)
   387:         has_disc = hasattr(criterion, 'disc') and hasattr(criterion, 'disc_opt')
   388:         disc_factor = 0.0
   389:         if has_disc:
   390:             disc_start = getattr(criterion, 'disc_start', 5000)
   391:             disc_factor = 1.0 if step >= disc_start else 0.0
   392: 
   393:         # 1) Generator (VAE) update — every step
   394:         if has_disc and disc_factor > 0:
   395:             # Add adaptive GAN loss (diffusers-aligned: perceptual grad for weight)
   396:             logits_fake = criterion.disc(recon.float())
   397:             g_loss = -logits_fake.mean()
   398:             last_layer = vae.decoder.conv_out.weight
   399:             p_loss = getattr(criterion, '_perceptual_loss', None)
   400:             ref_loss = p_loss if p_loss is not None else loss
   401:             ref_grads = torch.autograd.grad(ref_loss, last_layer, retain_graph=True)[0].detach()
   402:             g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0].detach()
   403:             disc_weight = torch.clamp(
   404:                 ref_grads.norm(p=2) / g_grads.norm(p=2).clamp(min=1e-8), 0.0, 1e4)
   405:             loss = loss + disc_weight * g_loss
   406:             metrics['g_loss'] = g_loss.item()
   407:             metrics['disc_w'] = disc_weight.item()
   408: 
   409:         optimizer.zero_grad()
   410:         scaler.scale(loss).backward()
   411:         scaler.unscale_(optimizer)
   412:         torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
   413:         scaler.step(optimizer)
   414:         scaler.update()
   415:         scheduler.step()
   416: 
   417:         # 2) Discriminator update — every step after disc_start (FP32, no AMP)
   418:         if has_disc and disc_factor > 0:
   419:             criterion.disc_opt.zero_grad()
   420:             x_real = x.float().detach().requires_grad_(True)
   421:             logits_real = criterion.disc(x_real)
   422:             logits_fake_d = criterion.disc(recon.float().detach())
   423:             d_loss = (F.relu(1 + logits_fake_d) + F.relu(1 - logits_real)).mean()
   424:             # Gradient penalty (diffusers-style)
   425:             gp_grads = torch.autograd.grad(
   426:                 outputs=logits_real, inputs=x_real,
   427:                 grad_outputs=torch.ones_like(logits_real),
   428:                 create_graph=True, retain_graph=True, only_inputs=True,
   429:             )[0]
   430:             gp = 10.0 * ((gp_grads.reshape(gp_grads.shape[0], -1).norm(2, dim=1) - 1) ** 2).mean()
   431:             d_loss = d_loss + gp
   432:             d_loss.backward()
   433:             criterion.disc_opt.step()
   434:             metrics['d_loss'] = d_loss.item()
   435: 
   436:         with torch.no_grad():
   437:             for p_ema, p in zip(ema_vae.parameters(), vae.parameters()):
   438:                 p_ema.mul_(ema_rate).add_(p, alpha=1 - ema_rate)
   439: 
   440:         if is_main and step % 200 == 0:
   441:             dt = time.time() - t0
   442:             m_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
   443:             print(f"step {step}/{max_steps} | loss={loss.item():.4f} {m_str} "
   444:                   f"| {dt:.1f}s", flush=True)
   445:             t0 = time.time()
   446: 
   447:         if step % eval_interval == 0 or step == max_steps:
   448:             if is_main:
   449:                 print(f"Eval at step {step}...", flush=True)
   450:             eval_model = ema_wrapper if step >= max_steps // 2 else \
   451:                 VAEWrapper(vae)
   452:             rfid, psnr, ssim_val = evaluate_reconstruction(
   453:                 eval_model, test_loader, device, output_dir, rank, world_size)
   454:             if is_main:
   455:                 if rfid < best_rfid:
   456:                     best_rfid = rfid
   457:                 if step < max_steps:
   458:                     print(f"TRAIN_METRICS: step={step}, rfid={rfid:.2f}, "
   459:                           f"psnr={psnr:.2f}, ssim={ssim_val:.4f}", flush=True)
   460:                 else:
   461:                     print(f"TEST_METRICS: rfid={rfid:.2f}, psnr={psnr:.2f}, "
   462:                           f"ssim={ssim_val:.4f}, best_rfid={best_rfid:.2f}",
   463:                           flush=True)
   464: 
   465:     # ── Save ──────────────────────────────────────────────────────────────────
   466:     if is_main:
   467:         print(f"Saving checkpoint to {output_dir}/checkpoint.pth", flush=True)
   468:         torch.save({
   469:             'step': max_steps,
   470:             'model_state_dict': vae.state_dict(),
   471:             'ema_model_state_dict': ema_vae.state_dict(),
   472:             'best_rfid': best_rfid,
   473:         }, os.path.join(output_dir, 'checkpoint.pth'))
   474: 
   475:     if use_ddp:
   476:         dist.destroy_process_group()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **train_small** — wall-clock budget `1:00:00`, compute share `8.0`
- **train_medium** — wall-clock budget `2:00:00`, compute share `8.0`
- **train_large** — wall-clock budget `4:00:00`, compute share `8.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `l2-kl` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 32–48:
    29: # Loss Function (EDITABLE REGION)
    30: # ============================================================================
    31: 
    32: 
    33: class VAELoss(nn.Module):
    34:     """Basic VAE loss: L1 reconstruction + KL divergence."""
    35: 
    36:     def __init__(self, device):
    37:         super().__init__()
    38:         self.kl_weight = 1e-6
    39: 
    40:     def forward(self, recon, target, posterior, step):
    41:         rec_loss = F.l1_loss(recon, target)
    42:         kl_loss = posterior.kl().mean()
    43:         loss = rec_loss + self.kl_weight * kl_loss
    44:         return loss, {
    45:             "rec_loss": rec_loss.item(),
    46:             "kl_loss": kl_loss.item(),
    47:         }
    48: 
    49: 
    50: # ============================================================================
    51: # Fixed: Model Architecture
```

### `perceptual` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 32–56:
    29: # Loss Function (EDITABLE REGION)
    30: # ============================================================================
    31: 
    32: 
    33: class VAELoss(nn.Module):
    34:     """Perceptual VAE loss: L1 + LPIPS + KL."""
    35: 
    36:     def __init__(self, device):
    37:         super().__init__()
    38:         import lpips
    39:         self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
    40:         self.lpips_fn.eval()
    41:         for p in self.lpips_fn.parameters():
    42:             p.requires_grad_(False)
    43:         self.kl_weight = 1e-6
    44:         self.perceptual_weight = 0.5
    45: 
    46:     def forward(self, recon, target, posterior, step):
    47:         rec_loss = F.l1_loss(recon, target)
    48:         p_loss = self.lpips_fn(recon.float(), target.float()).mean()
    49:         kl_loss = posterior.kl().mean()
    50:         loss = rec_loss + self.perceptual_weight * p_loss + self.kl_weight * kl_loss
    51:         return loss, {
    52:             "rec_loss": rec_loss.item(),
    53:             "p_loss": p_loss.item(),
    54:             "kl_loss": kl_loss.item(),
    55:         }
    56: 
    57: 
    58: # ============================================================================
    59: # Fixed: Model Architecture
```

### `vqgan` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 32–130:
    29: # Loss Function (EDITABLE REGION)
    30: # ============================================================================
    31: 
    32: 
    33: class VAELoss(nn.Module):
    34:     """VAE loss with adversarial training and feature matching."""
    35: 
    36:     def __init__(self, device):
    37:         super().__init__()
    38:         import lpips
    39:         self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
    40:         self.lpips_fn.eval()
    41:         for p in self.lpips_fn.parameters():
    42:             p.requires_grad_(False)
    43: 
    44:         self.device = device
    45:         self.perceptual_weight = 0.5
    46:         self.kl_weight = 1e-6
    47:         self.feat_match_weight = 1.0
    48: 
    49:         self.disc = NLayerDiscriminatorWithFeatures().to(device)
    50:         self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
    51:         self.disc_start = 5000
    52: 
    53:     def forward(self, recon, target, posterior, step):
    54:         rec_loss = F.l1_loss(recon, target)
    55:         p_loss = self.lpips_fn(recon.float(), target.float()).mean()
    56:         self._perceptual_loss = p_loss
    57:         kl_loss = posterior.kl().mean()
    58: 
    59:         disc_factor = 1.0 if step >= self.disc_start else 0.0
    60:         feat_match_loss = 0.0
    61:         if disc_factor > 0:
    62:             _, real_feats = self.disc(target, return_features=True)
    63:             _, fake_feats = self.disc(recon, return_features=True)
    64:             for real_f, fake_f in zip(real_feats, fake_feats):
    65:                 feat_match_loss += F.l1_loss(fake_f, real_f.detach())
    66:             feat_match_loss = feat_match_loss / len(real_feats)
    67: 
    68:         total_rec_loss = rec_loss + self.perceptual_weight * p_loss
    69:         if disc_factor > 0:
    70:             total_rec_loss = total_rec_loss + self.feat_match_weight * feat_match_loss
    71: 
    72:         loss = total_rec_loss + self.kl_weight * kl_loss
    73: 
    74:         metrics = {
    75:             "rec_loss": rec_loss.item(),
    76:             "p_loss": p_loss.item(),
    77:             "kl_loss": kl_loss.item(),
    78:         }
    79:         if disc_factor > 0:
    80:             metrics["feat_match"] = feat_match_loss.item()
    81: 
    82:         return loss, metrics
    83: 
    84: 
    85: class NLayerDiscriminatorWithFeatures(nn.Module):
    86:     """PatchGAN discriminator with intermediate feature extraction."""
    87: 
    88:     def __init__(self, input_nc=3, ndf=64, n_layers=3):
    89:         super().__init__()
    90:         from torch.nn.utils import spectral_norm
    91:         self.n_layers = n_layers
    92: 
    93:         layers = []
    94:         layers.append(spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1)))
    95:         layers.append(nn.LeakyReLU(0.2, True))
    96: 
    97:         nf_mult = 1
    98:         for n in range(1, n_layers):
    99:             nf_mult_prev = nf_mult
   100:             nf_mult = min(2 ** n, 8)
   101:             layers.append(spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False)))
   102:             layers.append(nn.BatchNorm2d(ndf * nf_mult))
   103:             layers.append(nn.LeakyReLU(0.2, True))
   104: 
   105:         nf_mult_prev = nf_mult
   106:         nf_mult = min(2 ** n_layers, 8)
   107:         layers.append(spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False)))
   108:         layers.append(nn.BatchNorm2d(ndf * nf_mult))
   109:         layers.append(nn.LeakyReLU(0.2, True))
   110: 
   111:         layers.append(spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)))
   112: 
   113:         self.model = nn.Sequential(*layers)
   114:         self.features = []
   115:         self._register_hooks()
   116: 
   117:     def _register_hooks(self):
   118:         def hook(module, input, output):
   119:             self.features.append(output)
   120: 
   121:         for layer in self.model:
   122:             if isinstance(layer, nn.LeakyReLU):
   123:                 layer.register_forward_hook(hook)
   124: 
   125:     def forward(self, x, return_features=False):
   126:         self.features.clear()
   127:         out = self.model(x)
   128:         if return_features:
   129:             return out, self.features.copy()
   130:         return out
   131: 
   132: # ============================================================================
   133: # Fixed: Model Architecture
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
