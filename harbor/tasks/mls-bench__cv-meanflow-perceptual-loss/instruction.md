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

## Baselines

| Baseline         | Description |
|------------------|-------------|
| `mse_base`       | Pure MSE on velocity — clean linear formulation, the floor reference. |
| `lpips_grad`     | MSE + Charbonnier-smoothed L1 on velocity + LPIPS + Sobel gradient + multiscale L1 on `x_denoised`, with a `(1 − t)^2` perceptual schedule and a `t ≤ 0.1` mask (spatial-domain perceptual recipe). |
| `lpips_spectral` | `lpips_grad` stack augmented with an FFT-magnitude L1 term on `x_denoised` (spatial + frequency-domain recipe). |


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
- editable lines **445–462**


Other files you may **read** for context (do not modify):
- `alphaflow-main/perceptual_utils.py`


## Readable Context


### `alphaflow-main/custom_train_perceptual.py`  [EDITABLE — lines 445–462 only]

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
   263:     # cleanfid.fid (aliased `cleanfid` here) did `from .features import
   264:     # build_feature_extractor, get_reference_statistics` at import time, so
   265:     # cleanfid.compute_fid resolves those names from the fid module's own
   266:     # globals, NOT from cleanfid.features. Patching only cleanfid.features
   267:     # (above) is therefore a no-op for compute_fid -> it would hit the default
   268:     # download path (InceptionV3W -> /tmp, stats -> CMU URL) and fail on a
   269:     # no-network compute node, so no FID is produced and the task scores 0.
   270:     # Patch the fid module's names too so the staged /data/cleanfid weights +
   271:     # reference stats are actually used.
   272:     _orig_build_fid = getattr(cleanfid, "build_feature_extractor", None)
   273:     _orig_ref_fid = getattr(cleanfid, "get_reference_statistics", None)
   274:     cleanfid.build_feature_extractor = _patched_build
   275:     cleanfid.get_reference_statistics = _patched_ref
   276: 
   277:     net.eval()
   278:     gen_dir = tempfile.mkdtemp()
   279: 
   280:     generated = 0
   281:     idx = 0
   282:     while generated < num_samples:
   283:         bs = min(batch_size, num_samples - generated)
   284:         imgs = sample_images(net, bs, num_steps, device, img_size)
   285:         imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   286:         for j in range(bs):
   287:             from PIL import Image
   288:             img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
   289:             Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
   290:             idx += 1
   291:         generated += bs
   292: 
   293:     score = cleanfid.compute_fid(
   294:         gen_dir,
   295:         dataset_name="cifar10",
   296:         dataset_res=32,
   297:         dataset_split="train",
   298:         device=device,
   299:         batch_size=batch_size,
   300:         verbose=False,
   301:     )
   302:     shutil.rmtree(gen_dir)
   303: 
   304:     # Restore original functions
   305:     _feat.build_feature_extractor = _orig_build
   306:     _feat.get_reference_statistics = _orig_ref
   307:     if _orig_build_fid is not None:
   308:         cleanfid.build_feature_extractor = _orig_build_fid
   309:     if _orig_ref_fid is not None:
   310:         cleanfid.get_reference_statistics = _orig_ref_fid
   311: 
   312:     net.train()
   313:     return score
   314: 
   315: 
   316: # ============================================================================
   317: # Training Script
   318: # ============================================================================
   319: 
   320: if __name__ == '__main__':
   321:     use_ddp = 'RANK' in os.environ
   322:     if use_ddp:
   323:         import torch.distributed as dist
   324:         from torch.nn.parallel import DistributedDataParallel as DDP
   325:         from torch.utils.data.distributed import DistributedSampler
   326: 
   327:         backend = 'nccl' if torch.cuda.is_available() else 'gloo'
   328:         dist.init_process_group(backend)
   329:         rank = dist.get_rank()
   330:         world_size = dist.get_world_size()
   331:         local_rank = int(os.environ.get('LOCAL_RANK', 0))
   332:     else:
   333:         dist = None
   334:         DDP = None
   335:         DistributedSampler = None
   336:         rank = 0
   337:         world_size = 1
   338:         local_rank = 0
   339:     is_main = rank == 0
   340: 
   341:     # ── Config ──────────────────────────────────────────────────────────────
   342:     seed = int(os.environ.get('SEED', 42))
   343:     data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
   344:     output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
   345:     max_steps = int(os.environ.get('MAX_STEPS', 1000))
   346:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 1000))
   347:     batch_size = int(os.environ.get('BATCH_SIZE', 128))
   348:     lr = float(os.environ.get('LR', 2e-4))
   349:     num_fid_samples = int(os.environ.get('NUM_FID_SAMPLES', 2048))
   350:     num_eval_steps = int(os.environ.get('NUM_EVAL_STEPS', 10))
   351:     ema_decay = float(os.environ.get('EMA_DECAY', 0.0))
   352: 
   353:     torch.manual_seed(seed + rank)
   354:     device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
   355:     if torch.cuda.is_available():
   356:         torch.cuda.set_device(device)
   357:     if is_main:
   358:         os.makedirs(output_dir, exist_ok=True)
   359:     if use_ddp:
   360:         dist.barrier()
   361: 
   362:     # ── Data ────────────────────────────────────────────────────────────────
   363:     transform = transforms.Compose([
   364:         transforms.RandomHorizontalFlip(),
   365:         transforms.ToTensor(),
   366:         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
   367:     ])
   368:     dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=False)
   369:     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
   370:     loader = torch.utils.data.DataLoader(
   371:         dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
   372:         num_workers=4, pin_memory=True, drop_last=True
   373:     )
   374:     data_iter = iter(loader)
   375: 
   376:     # ── Model ────────────────────────────────────────────────────────────────
   377:     hidden_size = int(os.environ.get('MODEL_HIDDEN_SIZE', 512))
   378:     depth       = int(os.environ.get('MODEL_DEPTH', 8))
   379:     num_heads   = int(os.environ.get('MODEL_NUM_HEADS', 8))
   380:     net = SmallDiT(img_size=32, patch_size=4, in_channels=3,
   381:                    hidden_size=hidden_size, depth=depth, num_heads=num_heads).to(device)
   382:     ema_net = None
   383:     if ema_decay > 0:
   384:         import copy
   385:         ema_net = copy.deepcopy(net)
   386:         ema_net.eval()
   387:         for p in ema_net.parameters():
   388:             p.requires_grad_(False)
   389:     if use_ddp:
   390:         ddp_kwargs = {"find_unused_parameters": True}
   391:         if device.type == 'cuda':
   392:             ddp_kwargs["device_ids"] = [local_rank]
   393:         net = DDP(net, **ddp_kwargs)
   394:     optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
   395:     scaler = torch.amp.GradScaler()
   396: 
   397:     raw_net = net.module if hasattr(net, 'module') else net
   398:     num_params = sum(p.numel() for p in raw_net.parameters())
   399:     if is_main:
   400:         ema_msg = f", ema={ema_decay}" if ema_net is not None else ""
   401:         print(f"Model parameters: {num_params/1e6:.1f}M | GPUs: {world_size}{ema_msg}")
   402: 
   403:     # ── LPIPS perceptual loss model ──────────────────────────────────────────
   404:     lpips_fn = lpips.LPIPS(net='vgg').to(device)
   405:     lpips_fn.eval()
   406:     for p in lpips_fn.parameters():
   407:         p.requires_grad_(False)
   408: 
   409:     # ── Training loop ────────────────────────────────────────────────────────
   410:     best_fid = float('inf')
   411:     t0 = time.time()
   412: 
   413:     for step in range(1, max_steps + 1):
   414:         try:
   415:             x, _ = next(data_iter)
   416:         except StopIteration:
   417:             if sampler is not None:
   418:                 sampler.set_epoch(step)
   419:             data_iter = iter(loader)
   420:             x, _ = next(data_iter)
   421: 
   422:         x = x.to(device)
   423:         B = x.shape[0]
   424: 
   425:         # Sample trajectory params
   426:         t, t_next, dt, alpha = sample_traj_params(B, step, max_steps, device)
   427: 
   428:         # Add noise: x_t = (1-t)*x + t*noise
   429:         noise = torch.randn_like(x)
   430:         x_t = (1 - t) * x + t * noise
   431: 
   432:         # Instantaneous velocity target: v = noise - x
   433:         velocity = noise - x
   434: 
   435:         # Compute mean velocity target
   436:         with torch.amp.autocast(device_type='cuda'):
   437:             raw_net = net.module if hasattr(net, 'module') else net
   438:             mean_vel_target = compute_mean_velocity_target(
   439:                 raw_net, x_t, t, t_next, dt, velocity, device
   440:             )
   441: 
   442:             # Predict mean velocity
   443:             pred_mean_vel = net(x_t, sigma=t, sigma_next=t_next)
   444: 
   445:             # TODO: Implement your loss function here.
   446:             #
   447:             # You have access to:
   448:             #   pred_mean_vel : [B, C, H, W] — model's predicted mean velocity
   449:             #   mean_vel_target: [B, C, H, W] — ground-truth mean velocity target
   450:             #   x              : [B, C, H, W] — clean image (normalized to [-1, 1])
   451:             #   x_t            : [B, C, H, W] — noisy image at timestep t
   452:             #   t              : [B, 1, 1, 1] — current timestep
   453:             #   t_next         : [B, 1, 1, 1] — target timestep
   454:             #   dt             : [B, 1, 1, 1] — step size
   455:             #   alpha          : float         — discrete path weight
   456:             #   lpips_fn       : LPIPS model (VGG backbone), expects input in [-1, 1]
   457:             #   device         : torch.device
   458:             #
   459:             # Your loss must assign a scalar `loss` (the value that will be
   460:             # back-propagated). You may also use perceptual (LPIPS) losses,
   461:             # frequency-domain losses, or any combination thereof.
   462:             raise NotImplementedError("Implement the loss function")
   463: 
   464:         warmup_steps = 1000
   465:         cur_lr = lr * step / warmup_steps if step <= warmup_steps else lr
   466:         for pg in optimizer.param_groups:
   467:             pg['lr'] = cur_lr
   468: 
   469:         optimizer.zero_grad()
   470:         scaler.scale(loss).backward()
   471:         scaler.unscale_(optimizer)
   472:         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
   473:         scaler.step(optimizer)
   474:         scaler.update()
   475: 
   476:         if ema_net is not None:
   477:             with torch.no_grad():
   478:                 raw_net = net.module if hasattr(net, 'module') else net
   479:                 for p_ema, p_net in zip(ema_net.parameters(), raw_net.parameters()):
   480:                     p_ema.lerp_(p_net, 1 - ema_decay)
   481: 
   482:         if step % 200 == 0 and is_main:
   483:             dt_elapsed = time.time() - t0
   484:             print(f"step {step}/{max_steps} | loss {loss.item():.4f} | lr {cur_lr:.2e} | {dt_elapsed:.1f}s", flush=True)
   485:             t0 = time.time()
   486: 
   487:         if step % eval_interval == 0 or step == max_steps:
   488:             if is_main:
   489:                 eval_net = ema_net if ema_net is not None else (net.module if hasattr(net, 'module') else net)
   490:                 eval_label = " (EMA)" if ema_net is not None else ""
   491:                 print(f"Computing FID at step {step}{eval_label}...", flush=True)
   492:                 fid = compute_fid(eval_net, device, num_samples=num_fid_samples, num_steps=num_eval_steps)
   493:                 print(f"TRAIN_METRICS: step={step}, loss={loss.item():.4f}, fid={fid:.2f}", flush=True)
   494:                 if fid < best_fid:
   495:                     best_fid = fid
   496:             if use_ddp:
   497:                 dist.barrier()
   498: 
   499:     # ── Save checkpoint ──────────────────────────────────────────────────────
   500:     if is_main:

[truncated: showing at most 500 lines / 60000 bytes from alphaflow-main/custom_train_perceptual.py]
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
Lines 445–449:
   442:             # Predict mean velocity
   443:             pred_mean_vel = net(x_t, sigma=t, sigma_next=t_next)
   444: 
   445:             # Pure MSE on mean velocity prediction.
   446:             # No inverse-loss reweighting (which would amplify easy samples
   447:             # and destabilise training around step 35k).
   448:             loss_mse_unscaled = ((pred_mean_vel - mean_vel_target) ** 2).flatten(1).mean(1)
   449:             loss = loss_mse_unscaled.mean()
   450: 
   451:         warmup_steps = 1000
   452:         cur_lr = lr * step / warmup_steps if step <= warmup_steps else lr
```

### `lpips_grad` baseline — editable region  [READ-ONLY — reference implementation]

In `alphaflow-main/custom_train_perceptual.py`:

```python
Lines 445–471:
   442:             # Predict mean velocity
   443:             pred_mean_vel = net(x_t, sigma=t, sigma_next=t_next)
   444: 
   445:             # MSE on velocity + Charbonnier smooth-L1 pixel loss on velocity
   446:             err = pred_mean_vel - mean_vel_target
   447:             loss_mse_unscaled = (err ** 2).flatten(1).mean(1)
   448:             loss_charb = torch.sqrt(err ** 2 + 1e-6).flatten(1).mean(1)
   449: 
   450:             # Auxiliary perceptual losses on denoised image (mask t<=0.1 edge case)
   451:             x_denoised = x_t - t * pred_mean_vel
   452:             t_flat = t.view(B)
   453:             mask = (t_flat > 0.1)
   454:             perceptual_w = ((1.0 - t_flat) ** 2) * mask.float()
   455: 
   456:             loss_lpips = torch.zeros(B, device=device)
   457:             loss_grad = torch.zeros(B, device=device)
   458:             loss_multi = torch.zeros(B, device=device)
   459:             if mask.any():
   460:                 xd = x_denoised[mask].clamp(-1, 1).float()
   461:                 xc = x[mask].clamp(-1, 1).float()
   462:                 loss_lpips[mask] = lpips_fn(xd, xc).view(-1).float()
   463:                 loss_grad[mask] = compute_gradient_loss(xd, xc).float()
   464:                 loss_multi[mask] = compute_multiscale_loss(xd, xc).float()
   465: 
   466:             loss_total = (
   467:                 loss_mse_unscaled
   468:                 + 0.1 * loss_charb
   469:                 + perceptual_w * (0.5 * loss_lpips + 0.3 * loss_grad + 0.2 * loss_multi)
   470:             )
   471:             loss = loss_total.mean()
   472: 
   473:         warmup_steps = 1000
   474:         cur_lr = lr * step / warmup_steps if step <= warmup_steps else lr
```

### `lpips_spectral` baseline — editable region  [READ-ONLY — reference implementation]

In `alphaflow-main/custom_train_perceptual.py`:

```python
Lines 445–479:
   442:             # Predict mean velocity
   443:             pred_mean_vel = net(x_t, sigma=t, sigma_next=t_next)
   444: 
   445:             # MSE on velocity
   446:             err = pred_mean_vel - mean_vel_target
   447:             loss_mse_unscaled = (err ** 2).flatten(1).mean(1)
   448: 
   449:             # Auxiliary perceptual losses on denoised image (mask t<=0.1 edge case)
   450:             x_denoised = x_t - t * pred_mean_vel
   451:             t_flat = t.view(B)
   452:             mask = (t_flat > 0.1)
   453:             perceptual_w = ((1.0 - t_flat) ** 2) * mask.float()
   454: 
   455:             loss_lpips = torch.zeros(B, device=device)
   456:             loss_grad = torch.zeros(B, device=device)
   457:             loss_multi = torch.zeros(B, device=device)
   458:             loss_spec = torch.zeros(B, device=device)
   459:             if mask.any():
   460:                 xd = x_denoised[mask].clamp(-1, 1).float()
   461:                 xc = x[mask].clamp(-1, 1).float()
   462:                 loss_lpips[mask] = lpips_fn(xd, xc).view(-1).float()
   463:                 loss_grad[mask] = compute_gradient_loss(xd, xc).float()
   464:                 loss_multi[mask] = compute_multiscale_loss(xd, xc).float()
   465:                 # FFT magnitude L1: per-channel rfft2, abs, L1 of difference
   466:                 fd = torch.fft.rfft2(xd, dim=(-2, -1)).abs()
   467:                 fc = torch.fft.rfft2(xc, dim=(-2, -1)).abs()
   468:                 loss_spec[mask] = (fd - fc).abs().mean(dim=(1, 2, 3)).float()
   469: 
   470:             loss_total = (
   471:                 loss_mse_unscaled
   472:                 + perceptual_w * (
   473:                     0.5 * loss_lpips
   474:                     + 0.3 * loss_grad
   475:                     + 0.2 * loss_multi
   476:                     + 0.2 * loss_spec
   477:                 )
   478:             )
   479:             loss = loss_total.mean()
   480: 
   481:         warmup_steps = 1000
   482:         cur_lr = lr * step / warmup_steps if step <= warmup_steps else lr
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
