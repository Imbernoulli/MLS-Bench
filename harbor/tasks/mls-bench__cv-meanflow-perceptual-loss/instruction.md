# MLS-Bench: cv-meanflow-perceptual-loss

# Flow Matching with Perceptual Loss

## Objective

Design an auxiliary training loss for CIFAR-10 flow matching that improves
sample FID under a fixed DiT backbone, MeanFlow training objective, and
ten-step Euler sampler.

## Background

Flow matching trains a network to predict velocity fields that transport
samples from noise to data. **MeanFlow** (Geng et al., 2025,
arXiv:2505.13447, "Mean Flows for One-step Generative Modeling") is a
flow-matching variant that learns the *average* velocity field over a time
interval and supports very-low-step (down to single-step) generation. The
canonical training loss is mean squared error on the predicted mean velocity:

```
loss_mse = || v_pred - v_target ||^2
```

However, the predicted velocity also implies a denoised image at every step:

```
x_denoised = x_t - t * v_pred
```

so auxiliary losses can be applied on `x_denoised` (image-space, perceptual,
gradient, multiscale, frequency-domain) to encourage the network to produce
high-quality images, not only accurate velocities.

## Implementation Contract

You are given `custom_train_perceptual.py`, a self-contained training script
that trains a small DiT (Peebles & Xie, ICCV 2023, arXiv:2212.09748) on
CIFAR-10 using MeanFlow. The editable region is the loss computation in the
training loop, e.g.:

```python
# Current: pure MSE on velocity.
loss_mse = ((pred_mean_vel - mean_vel_target) ** 2).mean()
loss = loss_mse
```

The fixed code already exposes helpers you may use to build perceptual /
auxiliary losses on `x_denoised`:

- `lpips_fn(x_denoised, x_target)` — LPIPS perceptual loss.
- `compute_gradient_loss(x_denoised, x_target)` — Sobel-style gradient-domain
  loss.
- `compute_multiscale_loss(x_denoised, x_target)` — multi-resolution loss.

**Stability constraint:** apply auxiliary losses only when `t > 0.1`. At very
small `t` the implied `x_denoised` becomes ill-conditioned and auxiliary
gradients dominate the velocity target.

## Fixed Pipeline

- Dataset: CIFAR-10 (32×32).
- Model: SmallDiT (~512 hidden, ~8 layers, ~40M params).
- Training: 10,000 steps, batch size 128.
- Inference: 10-step Euler sampler.
- Metric: FID computed by clean-fid against the CIFAR-10 train set, lower is
  better.

## Baselines

| Baseline         | Description |
|------------------|-------------|
| `mse_base`       | Pure MSE on velocity — clean linear formulation, the floor reference. |
| `lpips_grad`     | MSE + Charbonnier-smoothed L1 on velocity + LPIPS + Sobel gradient + multiscale L1 on `x_denoised`, with a `(1 − t)^2` perceptual schedule and a `t ≤ 0.1` mask (spatial-domain perceptual recipe). |
| `lpips_spectral` | `lpips_grad` stack augmented with an FFT-magnitude L1 term on `x_denoised` (spatial + frequency-domain recipe). |

## Evaluation

Evaluation trains on CIFAR-10 at the configured scales / budgets and samples
with the fixed ten-step Euler sampler. Scoring uses FID per scale; lower is
better.

A useful method should improve visual sample quality without destabilizing the
velocity target. Auxiliary losses must be applied only where `x_denoised` is
numerically meaningful. Do not change the architecture, data pipeline,
sampler, number of evaluation steps, or metric computation.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/alphaflow-main/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `alphaflow-main/custom_train_perceptual.py`
- editable lines **384–401**


Other files you may **read** for context (do not modify):
- `alphaflow-main/perceptual_utils.py`


## Readable Context


### `alphaflow-main/custom_train_perceptual.py`  [EDITABLE — lines 384–401 only]

```python
     1: """Custom Flow Matching Training Script — Perceptual Loss Variant
     2: Small-scale flow matching training on CIFAR-10 with a lightweight DiT.
     3: The training objective (MeanFlow) is pre-implemented; your task is to
     4: design an improved loss function, optionally using perceptual losses.
     5: """
     6: 
     7: import math
     8: import os
     9: import time
    10: 
    11: import lpips
    12: import numpy as np
    13: import torch
    14: import torch.nn as nn
    15: import torch.nn.functional as F
    16: from torch.autograd.functional import jvp
    17: from torchvision import datasets, transforms
    18: from torchvision.utils import save_image
    19: from perceptual_utils import compute_gradient_loss, compute_multiscale_loss
    20: 
    21: # ============================================================================
    22: # Model: Lightweight DiT for CIFAR-10 (32x32)
    23: # ============================================================================
    24: 
    25: def modulate(x, shift, scale):
    26:     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    27: 
    28: 
    29: class TimestepEmbedder(nn.Module):
    30:     def __init__(self, hidden_size, freq_embed_size=256):
    31:         super().__init__()
    32:         self.mlp = nn.Sequential(
    33:             nn.Linear(freq_embed_size, hidden_size),
    34:             nn.SiLU(),
    35:             nn.Linear(hidden_size, hidden_size),
    36:         )
    37:         self.freq_embed_size = freq_embed_size
    38: 
    39:     @staticmethod
    40:     def timestep_embedding(t, dim):
    41:         half = dim // 2
    42:         freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    43:         args = t[:, None] * freqs[None]
    44:         return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    45: 
    46:     def forward(self, t):
    47:         t_freq = self.timestep_embedding(t, self.freq_embed_size)
    48:         return self.mlp(t_freq)
    49: 
    50: 
    51: class DiTBlock(nn.Module):
    52:     def __init__(self, hidden_size, num_heads):
    53:         super().__init__()
    54:         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    55:         self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
    56:         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    57:         self.mlp = nn.Sequential(
    58:             nn.Linear(hidden_size, hidden_size * 4),
    59:             nn.GELU(),
    60:             nn.Linear(hidden_size * 4, hidden_size),
    61:         )
    62:         self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
    63: 
    64:     def forward(self, x, c):
    65:         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)
    66:         x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
    67:         attn_out, _ = self.attn(x_norm, x_norm, x_norm)
    68:         x = x + gate_msa.unsqueeze(1) * attn_out
    69:         x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
    70:         return x
    71: 
    72: 
    73: class SmallDiT(nn.Module):
    74:     """Lightweight DiT for CIFAR-10 (32x32 images, patch_size=4 -> 64 tokens)."""
    75:     def __init__(self, img_size=32, patch_size=4, in_channels=3, hidden_size=256, depth=6, num_heads=4):
    76:         super().__init__()
    77:         self.patch_size = patch_size
    78:         self.num_patches = (img_size // patch_size) ** 2
    79:         self.hidden_size = hidden_size
    80: 
    81:         self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, patch_size)
    82:         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
    83:         self.t_embedder = TimestepEmbedder(hidden_size)
    84:         # Two timestep embeddings: t (current) and t_next (target)
    85:         self.t_next_embedder = TimestepEmbedder(hidden_size)
    86:         self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
    87:         self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    88:         self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
    89: 
    90:         nn.init.normal_(self.pos_embed, std=0.02)
    91:         nn.init.zeros_(self.proj_out.weight)
    92:         nn.init.zeros_(self.proj_out.bias)
    93: 
    94:     def forward(self, x, sigma, sigma_next=None, **kwargs):
    95:         """
    96:         x: [b, c, h, w]
    97:         sigma: [b, 1, 1, 1] or [b] - current timestep t
    98:         sigma_next: [b, 1, 1, 1] or [b] - target timestep t_next (for mean velocity)
    99:         Returns: mean velocity [b, c, h, w]
   100:         """
   101:         B = x.shape[0]
   102:         t = sigma.view(B) if sigma.numel() == B else sigma.view(B, -1).mean(-1)
   103:         t_next = sigma_next.view(B) if sigma_next is not None and sigma_next.numel() == B \
   104:             else (sigma_next.view(B, -1).mean(-1) if sigma_next is not None else torch.zeros_like(t))
   105: 
   106:         # Patch embed
   107:         h = self.patch_embed(x)  # [b, hidden, h/p, w/p]
   108:         h = h.flatten(2).transpose(1, 2)  # [b, num_patches, hidden]
   109:         h = h + self.pos_embed
   110: 
   111:         # Condition: t + t_next embeddings
   112:         c = self.t_embedder(t) + self.t_next_embedder(t_next)  # [b, hidden]
   113: 
   114:         for block in self.blocks:
   115:             h = block(h, c)
   116: 
   117:         h = self.norm_out(h)
   118:         h = self.proj_out(h)  # [b, num_patches, p*p*c]
   119: 
   120:         # Unpatchify
   121:         p = self.patch_size
   122:         hw = int(self.num_patches ** 0.5)
   123:         h = h.reshape(B, hw, hw, p, p, -1)
   124:         h = h.permute(0, 5, 1, 3, 2, 4).contiguous()
   125:         h = h.reshape(B, -1, hw * p, hw * p)
   126:         return h  # [b, c, h, w]
   127: 
   128: 
   129: # ============================================================================
   130: # Sampling utilities
   131: # ============================================================================
   132: 
   133: def sample_logit_norm(batch_size, device, loc=0.0, scale=1.0, eps=1e-5):
   134:     """Sample t ~ logit-normal distribution, clipped to [eps, 1-eps]."""
   135:     u = torch.randn(batch_size, device=device) * scale + loc
   136:     t = torch.sigmoid(u)
   137:     return t.clamp(eps, 1 - eps)
   138: 
   139: 
   140: # ============================================================================
   141: # Training objective (MeanFlow — pre-implemented)
   142: # ============================================================================
   143: 
   144: def sample_traj_params(batch_size, cur_step, max_steps, device):
   145:     """MeanFlow: alpha=0, ratio_fm=0.75. Only JVP-based continuous training."""
   146:     ratio_fm = 0.75
   147:     alpha = 0.0
   148: 
   149:     batch_size_fm = int(batch_size * ratio_fm)
   150:     batch_size_mf = batch_size - batch_size_fm
   151: 
   152:     t_fm = sample_logit_norm(batch_size_fm, device, loc=-0.4)
   153:     t_next_fm = t_fm.clone()
   154:     dt_fm = torch.zeros_like(t_fm)
   155: 
   156:     t_1 = sample_logit_norm(batch_size_mf, device, loc=-0.4)
   157:     t_2 = sample_logit_norm(batch_size_mf, device, loc=-0.4)
   158:     t_mf = torch.maximum(t_1, t_2)
   159:     t_next_mf = torch.minimum(t_1, t_2)
   160:     dt_mf = torch.zeros_like(t_mf)  # alpha=0 -> dt=0
   161: 
   162:     t = torch.cat([t_fm, t_mf]).view(batch_size, 1, 1, 1)
   163:     t_next = torch.cat([t_next_fm, t_next_mf]).view(batch_size, 1, 1, 1)
   164:     dt = torch.cat([dt_fm, dt_mf]).view(batch_size, 1, 1, 1)
   165: 
   166:     return t, t_next, dt, alpha
   167: 
   168: 
   169: def compute_mean_velocity_target(net, x_t, t, t_next, dt, velocity, device):
   170:     """MeanFlow: use JVP for all MeanFlow samples, no discrete path."""
   171:     B = x_t.shape[0]
   172:     t_flat = t.view(B)
   173:     t_next_flat = t_next.view(B)
   174: 
   175:     mask_fm = torch.isclose(t_flat, t_next_flat)
   176:     mask_c = ~mask_fm
   177: 
   178:     mean_velocity = velocity.clone()
   179: 
   180:     if mask_c.any():
   181:         idx = mask_c.nonzero(as_tuple=True)[0]
   182:         x_c = x_t[idx]
   183:         t_c = t_flat[idx]
   184:         t_next_c = t_next_flat[idx]
   185: 
   186:         def wrap_net(x, t_in):
   187:             t_in_5d = t_in.view(-1, 1, 1, 1)
   188:             t_next_5d = t_next_c.view(-1, 1, 1, 1)
   189:             return net(x, sigma=t_in_5d, sigma_next=t_next_5d)
   190: 
   191:         _, dudt = jvp(wrap_net, (x_c, t_c), (velocity[idx], torch.ones_like(t_c)))
   192:         u_c = velocity[idx] - (t_c - t_next_c).view(-1, 1, 1, 1) * dudt
   193:         mean_velocity[idx] = u_c
   194: 
   195:     return mean_velocity
   196: 
   197: 
   198: # ============================================================================
   199: # Inference: reverse flow sampling
   200: # ============================================================================
   201: 
   202: @torch.no_grad()
   203: def sample_images(net, num_samples, num_steps, device, img_size=32, channels=3):
   204:     """Generate images via reverse flow (Euler steps)."""
   205:     net.eval()
   206:     x = torch.randn(num_samples, channels, img_size, img_size, device=device)
   207:     t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
   208: 
   209:     for i in range(num_steps):
   210:         t_cur = t_steps[i].expand(num_samples).view(num_samples, 1, 1, 1)
   211:         t_next = t_steps[i + 1].expand(num_samples).view(num_samples, 1, 1, 1)
   212:         v = net(x, sigma=t_cur, sigma_next=t_next)
   213:         x = x - (t_cur - t_next) * v
   214: 
   215:     net.train()
   216:     return x.clamp(-1, 1)
   217: 
   218: 
   219: # ============================================================================
   220: # FID computation (using clean-fid)
   221: # ============================================================================
   222: 
   223: def compute_fid(net, device, num_samples=2048, num_steps=10, img_size=32, batch_size=128):
   224:     """Compute FID against CIFAR-10 train set using clean-fid."""
   225:     import tempfile, shutil, numpy as np
   226:     from cleanfid import fid as cleanfid
   227:     import cleanfid.features as _feat
   228: 
   229:     # Use bind-mounted persistent cache dir for inception weights and stats
   230:     cache_dir = "/data/cleanfid"
   231:     os.makedirs(cache_dir, exist_ok=True)
   232: 
   233:     # Check if files exist, only download if missing
   234:     inception_path = os.path.join(cache_dir, "inception-2015-12-05.pt")
   235:     stats_path = os.path.join(cache_dir, "cifar10_clean_train_32.npz")
   236: 
   237:     missing = [p for p in (inception_path, stats_path) if not os.path.exists(p)]
   238:     if missing:
   239:         raise FileNotFoundError(
   240:             "Missing clean-fid cache files prepared by `mlsbench data alphaflow-main`: "
   241:             + ", ".join(missing)
   242:         )
   243: 
   244:     # Patch cleanfid to load inception from our cache dir instead of /tmp
   245:     _orig_build = _feat.build_feature_extractor
   246:     def _patched_build(mode, device=device, use_dataparallel=True):
   247:         from cleanfid.inception_torchscript import InceptionV3W
   248:         model = InceptionV3W(cache_dir, download=False, resize_inside=(mode=="legacy_tensorflow")).to(device)
   249:         model.eval()
   250:         if use_dataparallel:
   251:             model = torch.nn.DataParallel(model)
   252:         return lambda x: model(x)
   253:     _feat.build_feature_extractor = _patched_build
   254: 
   255:     # Patch stats lookup to use our cache dir
   256:     _orig_ref = _feat.get_reference_statistics
   257:     def _patched_ref(name, res, mode="clean", model_name="inception_v3", seed=0, split="train", metric="FID"):
   258:         fpath = os.path.join(cache_dir, f"{name}_{mode}_{split}_{res}.npz".lower())
   259:         stats = np.load(fpath)
   260:         return stats["mu"], stats["sigma"]
   261:     _feat.get_reference_statistics = _patched_ref
   262: 
   263:     net.eval()
   264:     gen_dir = tempfile.mkdtemp()
   265: 
   266:     generated = 0
   267:     idx = 0
   268:     while generated < num_samples:
   269:         bs = min(batch_size, num_samples - generated)
   270:         imgs = sample_images(net, bs, num_steps, device, img_size)
   271:         imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   272:         for j in range(bs):
   273:             from PIL import Image
   274:             img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
   275:             Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
   276:             idx += 1
   277:         generated += bs
   278: 
   279:     score = cleanfid.compute_fid(
   280:         gen_dir,
   281:         dataset_name="cifar10",
   282:         dataset_res=32,
   283:         dataset_split="train",
   284:         device=device,
   285:         batch_size=batch_size,
   286:         verbose=False,
   287:     )
   288:     shutil.rmtree(gen_dir)
   289: 
   290:     # Restore original functions
   291:     _feat.build_feature_extractor = _orig_build
   292:     _feat.get_reference_statistics = _orig_ref
   293: 
   294:     net.train()
   295:     return score
   296: 
   297: 
   298: # ============================================================================
   299: # Training Script
   300: # ============================================================================
   301: 
   302: if __name__ == '__main__':
   303:     # ── Config ──────────────────────────────────────────────────────────────
   304:     seed = int(os.environ.get('SEED', 42))
   305:     data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
   306:     output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
   307:     max_steps = int(os.environ.get('MAX_STEPS', 1000))
   308:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1000))
   309:     batch_size = int(os.environ.get('BATCH_SIZE', 128))
   310:     lr = float(os.environ.get('LR', 2e-4))
   311:     num_fid_samples = int(os.environ.get('NUM_FID_SAMPLES', 2048))
   312:     num_eval_steps = int(os.environ.get('NUM_EVAL_STEPS', 10))
   313: 
   314:     torch.manual_seed(seed)
   315:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   316:     os.makedirs(output_dir, exist_ok=True)
   317: 
   318:     # ── Data ────────────────────────────────────────────────────────────────
   319:     transform = transforms.Compose([
   320:         transforms.RandomHorizontalFlip(),
   321:         transforms.ToTensor(),
   322:         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
   323:     ])
   324:     dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=False)
   325:     loader = torch.utils.data.DataLoader(
   326:         dataset, batch_size=batch_size, shuffle=True,
   327:         num_workers=4, pin_memory=True, drop_last=True
   328:     )
   329:     data_iter = iter(loader)
   330: 
   331:     # ── Model ────────────────────────────────────────────────────────────────
   332:     hidden_size = int(os.environ.get('MODEL_HIDDEN_SIZE', 512))
   333:     depth       = int(os.environ.get('MODEL_DEPTH', 8))
   334:     num_heads   = int(os.environ.get('MODEL_NUM_HEADS', 8))
   335:     net = SmallDiT(img_size=32, patch_size=4, in_channels=3,
   336:                    hidden_size=hidden_size, depth=depth, num_heads=num_heads).to(device)
   337:     optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
   338:     scaler = torch.amp.GradScaler()
   339: 
   340:     num_params = sum(p.numel() for p in net.parameters())
   341:     print(f"Model parameters: {num_params/1e6:.1f}M")
   342: 
   343:     # ── LPIPS perceptual loss model ──────────────────────────────────────────
   344:     lpips_fn = lpips.LPIPS(net='vgg').to(device)
   345:     lpips_fn.eval()
   346:     for p in lpips_fn.parameters():
   347:         p.requires_grad_(False)
   348: 
   349:     # ── Training loop ────────────────────────────────────────────────────────
   350:     best_fid = float('inf')
   351:     t0 = time.time()
   352: 
   353:     for step in range(1, max_steps + 1):
   354:         try:
   355:             x, _ = next(data_iter)
   356:         except StopIteration:
   357:             data_iter = iter(loader)
   358:             x, _ = next(data_iter)
   359: 
   360:         x = x.to(device)
   361:         B = x.shape[0]
   362: 
   363:         # Sample trajectory params
   364:         t, t_next, dt, alpha = sample_traj_params(B, step, max_steps, device)
   365: 
   366:         # Add noise: x_t = (1-t)*x + t*noise
   367:         noise = torch.randn_like(x)
   368:         x_t = (1 - t) * x + t * noise
   369: 
   370:         # Instantaneous velocity target: v = noise - x
   371:         velocity = noise - x
   372: 
   373:         # Compute mean velocity target
   374:         with torch.amp.autocast(device_type='cuda'):
   375:             mean_vel_target = compute_mean_velocity_target(
   376:                 net, x_t, t, t_next, dt, velocity, device
   377:             )
   378: 
   379:             # Predict mean velocity
   380:             pred_mean_vel = net(x_t, sigma=t, sigma_next=t_next)
   381: 
   382:             # TODO: Implement your loss function here.
   383:             #
   384:             # You have access to:
   385:             #   pred_mean_vel : [B, C, H, W] — model's predicted mean velocity
   386:             #   mean_vel_target: [B, C, H, W] — ground-truth mean velocity target
   387:             #   x              : [B, C, H, W] — clean image (normalized to [-1, 1])
   388:             #   x_t            : [B, C, H, W] — noisy image at timestep t
   389:             #   t              : [B, 1, 1, 1] — current timestep
   390:             #   t_next         : [B, 1, 1, 1] — target timestep
   391:             #   dt             : [B, 1, 1, 1] — step size
   392:             #   alpha          : float         — discrete path weight
   393:             #   lpips_fn       : LPIPS model (VGG backbone), expects input in [-1, 1]
   394:             #   device         : torch.device
   395:             #
   396:             # Your loss must assign a scalar `loss` (the value that will be
   397:             # back-propagated). You may also use perceptual (LPIPS) losses,
   398:             # frequency-domain losses, or any combination thereof.
   399:             raise NotImplementedError("Implement the loss function")
   400: 
   401:         optimizer.zero_grad()
   402:         scaler.scale(loss).backward()
   403:         scaler.unscale_(optimizer)
   404:         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
   405:         scaler.step(optimizer)
   406:         scaler.update()
   407: 
   408:         if step % 200 == 0:
   409:             dt_elapsed = time.time() - t0
   410:             print(f"step {step}/{max_steps} | loss {loss.item():.4f} | {dt_elapsed:.1f}s", flush=True)
   411:             t0 = time.time()
   412: 
   413:         if step % eval_interval == 0 or step == max_steps:
   414:             print(f"Computing FID at step {step}...", flush=True)
   415:             fid = compute_fid(net, device, num_samples=num_fid_samples, num_steps=num_eval_steps)
   416:             print(f"TRAIN_METRICS: step={step}, loss={loss.item():.4f}, fid={fid:.2f}", flush=True)
   417:             if fid < best_fid:
   418:                 best_fid = fid
   419: 
   420:     # ── Save checkpoint ──────────────────────────────────────────────────────
   421:     print(f"Saving checkpoint to {output_dir}/checkpoint.pth", flush=True)
   422:     torch.save({
   423:         'step': max_steps,
   424:         'model_state_dict': net.state_dict(),
   425:         'optimizer_state_dict': optimizer.state_dict(),
   426:         'best_fid': best_fid,
   427:     }, os.path.join(output_dir, 'checkpoint.pth'))
   428: 
   429:     # ── Final eval ───────────────────────────────────────────────────────────
   430:     fid = compute_fid(net, device, num_samples=num_fid_samples, num_steps=num_eval_steps)
   431:     print(f"TEST_METRICS: fid={fid:.2f}, best_fid={best_fid:.2f}", flush=True)
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `mse_base` baseline — editable region  [READ-ONLY — reference implementation]

In `alphaflow-main/custom_train_perceptual.py`:

```python
Lines 384–388:
   381: 
   382:             # TODO: Implement your loss function here.
   383:             #
   384:             # Pure MSE on mean velocity prediction.
   385:             # No inverse-loss reweighting (which would amplify easy samples
   386:             # and destabilise training around step 35k).
   387:             loss_mse_unscaled = ((pred_mean_vel - mean_vel_target) ** 2).flatten(1).mean(1)
   388:             loss = loss_mse_unscaled.mean()
   389:         scaler.scale(loss).backward()
   390:         scaler.unscale_(optimizer)
   391:         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
```

### `lpips_grad` baseline — editable region  [READ-ONLY — reference implementation]

In `alphaflow-main/custom_train_perceptual.py`:

```python
Lines 384–410:
   381: 
   382:             # TODO: Implement your loss function here.
   383:             #
   384:             # MSE on velocity + Charbonnier smooth-L1 pixel loss on velocity
   385:             err = pred_mean_vel - mean_vel_target
   386:             loss_mse_unscaled = (err ** 2).flatten(1).mean(1)
   387:             loss_charb = torch.sqrt(err ** 2 + 1e-6).flatten(1).mean(1)
   388: 
   389:             # Auxiliary perceptual losses on denoised image (mask t<=0.1 edge case)
   390:             x_denoised = x_t - t * pred_mean_vel
   391:             t_flat = t.view(B)
   392:             mask = (t_flat > 0.1)
   393:             perceptual_w = ((1.0 - t_flat) ** 2) * mask.float()
   394: 
   395:             loss_lpips = torch.zeros(B, device=device)
   396:             loss_grad = torch.zeros(B, device=device)
   397:             loss_multi = torch.zeros(B, device=device)
   398:             if mask.any():
   399:                 xd = x_denoised[mask].clamp(-1, 1).float()
   400:                 xc = x[mask].clamp(-1, 1).float()
   401:                 loss_lpips[mask] = lpips_fn(xd, xc).view(-1).float()
   402:                 loss_grad[mask] = compute_gradient_loss(xd, xc).float()
   403:                 loss_multi[mask] = compute_multiscale_loss(xd, xc).float()
   404: 
   405:             loss_total = (
   406:                 loss_mse_unscaled
   407:                 + 0.1 * loss_charb
   408:                 + perceptual_w * (0.5 * loss_lpips + 0.3 * loss_grad + 0.2 * loss_multi)
   409:             )
   410:             loss = loss_total.mean()
   411:         scaler.scale(loss).backward()
   412:         scaler.unscale_(optimizer)
   413:         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
```

### `lpips_spectral` baseline — editable region  [READ-ONLY — reference implementation]

In `alphaflow-main/custom_train_perceptual.py`:

```python
Lines 384–418:
   381: 
   382:             # TODO: Implement your loss function here.
   383:             #
   384:             # MSE on velocity
   385:             err = pred_mean_vel - mean_vel_target
   386:             loss_mse_unscaled = (err ** 2).flatten(1).mean(1)
   387: 
   388:             # Auxiliary perceptual losses on denoised image (mask t<=0.1 edge case)
   389:             x_denoised = x_t - t * pred_mean_vel
   390:             t_flat = t.view(B)
   391:             mask = (t_flat > 0.1)
   392:             perceptual_w = ((1.0 - t_flat) ** 2) * mask.float()
   393: 
   394:             loss_lpips = torch.zeros(B, device=device)
   395:             loss_grad = torch.zeros(B, device=device)
   396:             loss_multi = torch.zeros(B, device=device)
   397:             loss_spec = torch.zeros(B, device=device)
   398:             if mask.any():
   399:                 xd = x_denoised[mask].clamp(-1, 1).float()
   400:                 xc = x[mask].clamp(-1, 1).float()
   401:                 loss_lpips[mask] = lpips_fn(xd, xc).view(-1).float()
   402:                 loss_grad[mask] = compute_gradient_loss(xd, xc).float()
   403:                 loss_multi[mask] = compute_multiscale_loss(xd, xc).float()
   404:                 # FFT magnitude L1: per-channel rfft2, abs, L1 of difference
   405:                 fd = torch.fft.rfft2(xd, dim=(-2, -1)).abs()
   406:                 fc = torch.fft.rfft2(xc, dim=(-2, -1)).abs()
   407:                 loss_spec[mask] = (fd - fc).abs().mean(dim=(1, 2, 3)).float()
   408: 
   409:             loss_total = (
   410:                 loss_mse_unscaled
   411:                 + perceptual_w * (
   412:                     0.5 * loss_lpips
   413:                     + 0.3 * loss_grad
   414:                     + 0.2 * loss_multi
   415:                     + 0.2 * loss_spec
   416:                 )
   417:             )
   418:             loss = loss_total.mean()
   419:         scaler.scale(loss).backward()
   420:         scaler.unscale_(optimizer)
   421:         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
