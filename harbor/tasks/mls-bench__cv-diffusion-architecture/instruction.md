# MLS-Bench: cv-diffusion-architecture

# Diffusion Model Architecture Design

## Objective

Design a UNet backbone for unconditional image diffusion that achieves
better generation quality than the standard DDPM-style architectures, under
a fixed training target (epsilon prediction), DDIM sampler, optimizer, and
noise schedule.

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
(e.g. `"128,256,256,256"`) so that the same architecture can scale to
different channel widths. `LAYERS_PER_BLOCK` (default 2) is also available.

## Fixed Pipeline

The following are fixed across baselines and submissions:

- Training target: epsilon prediction with MSE loss.
- Optimizer: AdamW, learning rate 2e-4, EMA rate 0.9995.
- Inference: 50-step DDIM sampling (Song et al., 2020, arXiv:2010.02502).
- Channel widths are passed via `BLOCK_OUT_CHANNELS` env var.

## Baselines

| Baseline    | Description |
|-------------|-------------|
| `standard`  | Original DDPM architecture (Ho et al., 2020, arXiv:2006.11239). Self-attention only at the 16×16 resolution. Matches the `google/ddpm-cifar10-32` configuration. |
| `full-attn` | Self-attention at every resolution (32×32, 16×16, 8×8, 4×4). More expressive but significantly more compute and memory per step. |
| `no-attn`   | Pure convolutional UNet with no per-resolution self-attention; only the mid-block retains its default self-attention layer. Smallest and fastest. |

Improvements should come from transferable architecture design, not from
changes to data, loss target, optimizer, sampler, or evaluation.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/diffusers-main/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `diffusers-main/custom_train.py`
- editable lines **31–58**




## Readable Context


### `diffusers-main/custom_train.py`  [EDITABLE — lines 31–58 only]

```python
     1: """Unconditional DDPM Training on CIFAR-10 with configurable UNet architecture.
     2: 
     3: Uses epsilon prediction (fixed). Only the model architecture is editable.
     4: """
     5: 
     6: import copy
     7: import math
     8: import os
     9: import sys
    10: import time
    11: from datetime import timedelta
    12: 
    13: import numpy as np
    14: import torch
    15: import torch.nn as nn
    16: import torch.distributed as dist
    17: import torch.nn.functional as F
    18: from PIL import Image
    19: from torch.nn.parallel import DistributedDataParallel as DDP
    20: from torchvision import datasets, transforms
    21: 
    22: # Use diffusers from the external package
    23: sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    24: from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
    25: 
    26: 
    27: # ============================================================================
    28: # Model Architecture (EDITABLE REGION)
    29: # ============================================================================
    30: 
    31: def build_model(device):
    32:     """Build a UNet model for unconditional CIFAR-10 diffusion.
    33: 
    34:     TODO: Design your UNet architecture here.
    35: 
    36:     The model must satisfy:
    37:     - Input:  (x, timestep) where x is [B, 3, 32, 32], timestep is [B]
    38:     - Output: object with .sample attribute of shape [B, 3, 32, 32]
    39:     - UNet2DModel from diffusers satisfies this interface
    40: 
    41:     The channel widths are provided via env var BLOCK_OUT_CHANNELS (e.g.
    42:     "128,256,256,256") so the same architecture scales across evaluation
    43:     tiers.  LAYERS_PER_BLOCK (default 2) is also available.
    44: 
    45:     Available from diffusers UNet2DModel:
    46:         down_block_types / up_block_types — choose from:
    47:             "DownBlock2D"     / "UpBlock2D"      (pure convolution)
    48:             "AttnDownBlock2D" / "AttnUpBlock2D"  (conv + self-attention)
    49:         Other knobs: layers_per_block, norm_num_groups, attention_head_dim,
    50:                      resnet_time_scale_shift, act_fn, etc.
    51: 
    52:     You may also build a fully custom nn.Module as long as it exposes
    53:     the same (x, timestep) → .sample interface.
    54: 
    55:     Returns:
    56:         nn.Module on the given device
    57:     """
    58:     raise NotImplementedError("Implement build_model")
    59: 
    60: 
    61: # ============================================================================
    62: # Fixed: epsilon prediction
    63: # ============================================================================
    64: 
    65: def get_schedule_tensors(noise_scheduler, device):
    66:     acp = noise_scheduler.alphas_cumprod.to(device)
    67:     return {
    68:         "alphas_cumprod": acp,
    69:         "sqrt_alpha": acp.sqrt(),
    70:         "sqrt_one_minus_alpha": (1.0 - acp).sqrt(),
    71:     }
    72: 
    73: 
    74: def compute_training_target(x_0, noise, timesteps, schedule):
    75:     """Epsilon prediction — fixed, not editable."""
    76:     return noise
    77: 
    78: 
    79: def predict_x0(model_output, x_t, timesteps, schedule):
    80:     """Recover x_0 from epsilon prediction — fixed, not editable."""
    81:     sa = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    82:     soma = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    83:     return (x_t - soma * model_output) / sa
    84: 
    85: 
    86: # ============================================================================
    87: # Sampling — DDIM with epsilon prediction
    88: # ============================================================================
    89: 
    90: @torch.no_grad()
    91: def sample_images(model, schedule, num_samples, device, num_steps=1000,
    92:                   sample_steps=50, img_size=32, channels=3):
    93:     model.eval()
    94:     scheduler = DDIMScheduler(
    95:         num_train_timesteps=num_steps,
    96:         beta_schedule="linear",
    97:         beta_start=0.0001,
    98:         beta_end=0.02,
    99:         clip_sample=True,
   100:         set_alpha_to_one=False,
   101:         prediction_type="epsilon",
   102:     )
   103:     scheduler.set_timesteps(sample_steps)
   104: 
   105:     x = torch.randn(num_samples, channels, img_size, img_size, device=device)
   106: 
   107:     for t in scheduler.timesteps:
   108:         t_batch = t.expand(num_samples).to(device)
   109:         with torch.amp.autocast(device_type='cuda'):
   110:             noise_pred = model(x, t_batch).sample
   111:         x = scheduler.step(noise_pred, t, x).prev_sample
   112: 
   113:     model.train()
   114:     return x.clamp(-1, 1)
   115: 
   116: 
   117: # ============================================================================
   118: # FID computation (using clean-fid)
   119: # ============================================================================
   120: 
   121: def compute_fid(model, schedule, device, num_samples=2048, num_steps=1000,
   122:                 sample_steps=50, img_size=32, batch_size=128,
   123:                 rank=0, world_size=1):
   124:     import shutil
   125:     from cleanfid import fid as cleanfid
   126:     import cleanfid.features as _feat
   127: 
   128:     gen_dir = os.path.join(os.environ.get('OUTPUT_DIR', '/tmp/output'), '_fid_tmp')
   129:     if rank == 0:
   130:         if os.path.exists(gen_dir):
   131:             shutil.rmtree(gen_dir)
   132:         os.makedirs(gen_dir)
   133:     if world_size > 1:
   134:         dist.barrier()
   135: 
   136:     per_rank = (num_samples + world_size - 1) // world_size
   137:     my_start = rank * per_rank
   138:     my_count = min(per_rank, num_samples - my_start)
   139: 
   140:     model.eval()
   141:     generated = 0
   142:     idx = my_start
   143:     while generated < my_count:
   144:         bs = min(batch_size, my_count - generated)
   145:         imgs = sample_images(model, schedule, bs, device, num_steps,
   146:                              sample_steps, img_size)
   147:         imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   148:         for j in range(bs):
   149:             img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
   150:             Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
   151:             idx += 1
   152:         generated += bs
   153: 
   154:     if world_size > 1:
   155:         dist.barrier()
   156: 
   157:     score = None
   158:     if rank == 0:
   159:         cache_dir = "/data/cleanfid"
   160:         os.makedirs(cache_dir, exist_ok=True)
   161: 
   162:         inception_path = os.path.join(cache_dir, "inception-2015-12-05.pt")
   163:         stats_path = os.path.join(cache_dir, "cifar10_clean_train_32.npz")
   164: 
   165:         missing = [p for p in (inception_path, stats_path) if not os.path.exists(p)]
   166:         if missing:
   167:             raise FileNotFoundError(
   168:                 "Missing clean-fid cache files prepared by `mlsbench data diffusers-main`: "
   169:                 + ", ".join(missing)
   170:             )
   171: 
   172:         _orig_build = _feat.build_feature_extractor
   173:         def _patched_build(mode, device=device, use_dataparallel=True):
   174:             from cleanfid.inception_torchscript import InceptionV3W
   175:             m = InceptionV3W(cache_dir, download=False,
   176:                              resize_inside=(mode == "legacy_tensorflow")).to(device)
   177:             m.eval()
   178:             if use_dataparallel:
   179:                 m = torch.nn.DataParallel(m)
   180:             return lambda x: m(x)
   181:         _feat.build_feature_extractor = _patched_build
   182: 
   183:         _orig_ref = _feat.get_reference_statistics
   184:         def _patched_ref(name, res, mode="clean", model_name="inception_v3",
   185:                          seed=0, split="train", metric="FID"):
   186:             fpath = os.path.join(cache_dir, f"{name}_{mode}_{split}_{res}.npz".lower())
   187:             stats = np.load(fpath)
   188:             return stats["mu"], stats["sigma"]
   189:         _feat.get_reference_statistics = _patched_ref
   190:         import cleanfid.fid as _fid_mod
   191:         _fid_mod.get_reference_statistics = _patched_ref
   192: 
   193:         score = cleanfid.compute_fid(
   194:             gen_dir, dataset_name="cifar10", dataset_res=32,
   195:             dataset_split="train", device=device, batch_size=batch_size, verbose=False,
   196:         )
   197: 
   198:         _feat.build_feature_extractor = _orig_build
   199:         _feat.get_reference_statistics = _orig_ref
   200:         _fid_mod.get_reference_statistics = _orig_ref
   201: 
   202:         shutil.rmtree(gen_dir)
   203: 
   204:     if world_size > 1:
   205:         dist.barrier()
   206: 
   207:     model.train()
   208:     return score
   209: 
   210: 
   211: def save_sample_images(model, schedule, device, output_dir, step, num_images=16,
   212:                        num_steps=1000, sample_steps=50, tag=""):
   213:     imgs = sample_images(model, schedule, num_images, device, num_steps, sample_steps)
   214:     imgs = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   215: 
   216:     nrow = int(math.sqrt(num_images))
   217:     grid_h = nrow * 32
   218:     grid_w = nrow * 32
   219:     grid = Image.new('RGB', (grid_w, grid_h))
   220:     for i in range(num_images):
   221:         img_np = imgs[i].permute(1, 2, 0).numpy()
   222:         img = Image.fromarray(img_np)
   223:         row, col = divmod(i, nrow)
   224:         grid.paste(img, (col * 32, row * 32))
   225: 
   226:     suffix = f"_{tag}" if tag else ""
   227:     path = os.path.join(output_dir, f'samples_step{step}{suffix}.png')
   228:     grid.save(path)
   229:     print(f"Saved sample images to {path}", flush=True)
   230: 
   231: 
   232: # ============================================================================
   233: # Training Script
   234: # ============================================================================
   235: 
   236: if __name__ == '__main__':
   237:     seed = int(os.environ.get('SEED', 42))
   238:     data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
   239:     output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
   240:     max_steps = int(os.environ.get('MAX_STEPS', 10000))
   241:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 10000))
   242:     batch_size = int(os.environ.get('BATCH_SIZE', 128))
   243:     lr = float(os.environ.get('LR', 2e-4))
   244:     num_fid_samples = int(os.environ.get('NUM_FID_SAMPLES', 2048))
   245:     diffusion_steps = int(os.environ.get('DIFFUSION_STEPS', 1000))
   246:     sample_steps = int(os.environ.get('SAMPLE_STEPS', 50))
   247:     ema_rate = float(os.environ.get('EMA_RATE', 0.9999))
   248: 
   249:     # ── DDP setup ──────────────────────────────────────────────────────────
   250:     use_ddp = 'RANK' in os.environ
   251:     if use_ddp:
   252:         dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
   253:         local_rank = int(os.environ['LOCAL_RANK'])
   254:         rank = int(os.environ['RANK'])
   255:         world_size = int(os.environ['WORLD_SIZE'])
   256:         device = torch.device(f'cuda:{local_rank}')
   257:         torch.cuda.set_device(device)
   258:         is_main = (rank == 0)
   259:     else:
   260:         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   261:         rank = 0
   262:         world_size = 1
   263:         is_main = True
   264: 
   265:     torch.manual_seed(seed + rank)
   266:     os.makedirs(output_dir, exist_ok=True)
   267: 
   268:     # ── Data ────────────────────────────────────────────────────────────────
   269:     transform = transforms.Compose([
   270:         transforms.RandomHorizontalFlip(),
   271:         transforms.ToTensor(),
   272:         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
   273:     ])
   274:     dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=False)
   275:     if use_ddp:
   276:         sampler = torch.utils.data.DistributedSampler(
   277:             dataset, num_replicas=world_size, rank=rank, shuffle=True)
   278:         loader = torch.utils.data.DataLoader(
   279:             dataset, batch_size=batch_size, sampler=sampler,
   280:             num_workers=4, pin_memory=True, drop_last=True,
   281:         )
   282:     else:
   283:         loader = torch.utils.data.DataLoader(
   284:             dataset, batch_size=batch_size, shuffle=True,
   285:             num_workers=4, pin_memory=True, drop_last=True,
   286:         )
   287:     data_iter = iter(loader)
   288: 
   289:     # ── Noise scheduler ────────────────────────────────────────────────────
   290:     noise_scheduler = DDPMScheduler(
   291:         num_train_timesteps=diffusion_steps,
   292:         beta_schedule="linear",
   293:         beta_start=0.0001,
   294:         beta_end=0.02,
   295:         clip_sample=True,
   296:         variance_type="fixed_large",
   297:     )
   298:     schedule = get_schedule_tensors(noise_scheduler, device)
   299: 
   300:     # ── Model ───────────────────────────────────────────────────────────────
   301:     net = build_model(device)
   302: 
   303:     ema_net = copy.deepcopy(net)
   304:     ema_net.requires_grad_(False)
   305: 
   306:     if use_ddp:
   307:         net = DDP(net, device_ids=[local_rank])
   308:     net_raw = net.module if use_ddp else net
   309: 
   310:     optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
   311:     scaler = torch.amp.GradScaler()
   312: 
   313:     num_params = sum(p.numel() for p in net_raw.parameters())
   314:     if is_main:
   315:         print(f"Model parameters: {num_params/1e6:.1f}M | GPUs: {world_size}", flush=True)
   316: 
   317:     # ── Training loop ────────────────────────────────────────────────────────
   318:     best_fid = float('inf')
   319:     t0 = time.time()
   320:     epoch = 0
   321: 
   322:     for step in range(1, max_steps + 1):
   323:         try:
   324:             x, _ = next(data_iter)
   325:         except StopIteration:
   326:             epoch += 1
   327:             if use_ddp:
   328:                 sampler.set_epoch(epoch)
   329:             data_iter = iter(loader)
   330:             x, _ = next(data_iter)
   331: 
   332:         x = x.to(device)
   333:         B = x.shape[0]
   334: 
   335:         t = torch.randint(0, diffusion_steps, (B,), device=device).long()
   336:         noise = torch.randn_like(x)
   337:         x_t = noise_scheduler.add_noise(x, noise, t)
   338: 
   339:         target = compute_training_target(x, noise, t, schedule)
   340: 
   341:         with torch.amp.autocast(device_type='cuda'):
   342:             pred = net(x_t, t).sample
   343:             loss = F.mse_loss(pred, target)
   344: 
   345:         optimizer.zero_grad()
   346:         scaler.scale(loss).backward()
   347:         scaler.unscale_(optimizer)
   348:         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
   349:         scaler.step(optimizer)
   350:         scaler.update()
   351: 
   352:         with torch.no_grad():
   353:             for p_ema, p in zip(ema_net.parameters(), net_raw.parameters()):
   354:                 p_ema.mul_(ema_rate).add_(p, alpha=1 - ema_rate)
   355: 
   356:         if is_main and step % 200 == 0:
   357:             dt_elapsed = time.time() - t0
   358:             print(f"step {step}/{max_steps} | loss {loss.item():.4f} | {dt_elapsed:.1f}s",
   359:                   flush=True)
   360:             t0 = time.time()
   361: 
   362:         if step % eval_interval == 0 or step == max_steps:
   363:             if is_main:
   364:                 print(f"Eval at step {step}...", flush=True)
   365:                 save_sample_images(net_raw, schedule, device, output_dir, step,
   366:                                    num_steps=diffusion_steps, sample_steps=sample_steps,
   367:                                    tag="net")
   368:                 save_sample_images(ema_net, schedule, device, output_dir, step,
   369:                                    num_steps=diffusion_steps, sample_steps=sample_steps,
   370:                                    tag="ema")
   371:             eval_model = ema_net if step >= 20000 else net_raw
   372:             fid = compute_fid(eval_model, schedule, device,
   373:                               num_samples=num_fid_samples,
   374:                               num_steps=diffusion_steps,
   375:                               sample_steps=sample_steps,
   376:                               rank=rank, world_size=world_size)
   377:             if is_main:
   378:                 print(f"TRAIN_METRICS: step={step}, loss={loss.item():.4f}, fid={fid:.2f}",
   379:                       flush=True)
   380:                 if fid < best_fid:
   381:                     best_fid = fid
   382: 
   383:     # ── Save & final eval ────────────────────────────────────────────────────
   384:     if is_main:
   385:         print(f"Saving checkpoint to {output_dir}/checkpoint.pth", flush=True)
   386:         torch.save({
   387:             'step': max_steps,
   388:             'model_state_dict': net_raw.state_dict(),
   389:             'ema_model_state_dict': ema_net.state_dict(),
   390:             'optimizer_state_dict': optimizer.state_dict(),
   391:             'best_fid': best_fid,
   392:         }, os.path.join(output_dir, 'checkpoint.pth'))
   393: 
   394:         save_sample_images(net_raw, schedule, device, output_dir, max_steps,
   395:                            num_steps=diffusion_steps, sample_steps=sample_steps,
   396:                            tag="net_final")
   397:         save_sample_images(ema_net, schedule, device, output_dir, max_steps,
   398:                            num_steps=diffusion_steps, sample_steps=sample_steps,
   399:                            tag="ema_final")
   400: 
   401:     eval_model = ema_net if max_steps >= 20000 else net_raw
   402:     fid = compute_fid(eval_model, schedule, device,
   403:                       num_samples=num_fid_samples,
   404:                       num_steps=diffusion_steps,
   405:                       sample_steps=sample_steps,
   406:                       rank=rank, world_size=world_size)
   407:     if is_main:
   408:         print(f"TEST_METRICS: fid={fid:.2f}, best_fid={best_fid:.2f}", flush=True)
   409: 
   410:     if use_ddp:
   411:         dist.destroy_process_group()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `standard` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 31–54:
    28: # Model Architecture (EDITABLE REGION)
    29: # ============================================================================
    30: 
    31: 
    32: def build_model(device):
    33:     """Standard DDPM architecture: attention at 16x16 only."""
    34:     channels = (128, 256, 256, 256)
    35:     if os.environ.get('BLOCK_OUT_CHANNELS'):
    36:         channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    37:     layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))
    38: 
    39:     return UNet2DModel(
    40:         sample_size=32,
    41:         in_channels=3,
    42:         out_channels=3,
    43:         block_out_channels=channels,
    44:         down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
    45:         up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    46:         layers_per_block=layers,
    47:         norm_num_groups=32,
    48:         norm_eps=1e-6,
    49:         act_fn="silu",
    50:         time_embedding_type="positional",
    51:         flip_sin_to_cos=False,
    52:         freq_shift=1,
    53:         downsample_padding=0,
    54:     ).to(device)
    55: 
    56: 
    57: # ============================================================================
```

### `full-attn` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 31–54:
    28: # Model Architecture (EDITABLE REGION)
    29: # ============================================================================
    30: 
    31: 
    32: def build_model(device):
    33:     """Full-attention: self-attention at every resolution."""
    34:     channels = (128, 256, 256, 256)
    35:     if os.environ.get('BLOCK_OUT_CHANNELS'):
    36:         channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    37:     layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))
    38: 
    39:     return UNet2DModel(
    40:         sample_size=32,
    41:         in_channels=3,
    42:         out_channels=3,
    43:         block_out_channels=channels,
    44:         down_block_types=("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    45:         up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
    46:         layers_per_block=layers,
    47:         norm_num_groups=32,
    48:         norm_eps=1e-6,
    49:         act_fn="silu",
    50:         time_embedding_type="positional",
    51:         flip_sin_to_cos=False,
    52:         freq_shift=1,
    53:         downsample_padding=0,
    54:     ).to(device)
    55: 
    56: 
    57: # ============================================================================
```

### `no-attn` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 31–54:
    28: # Model Architecture (EDITABLE REGION)
    29: # ============================================================================
    30: 
    31: 
    32: def build_model(device):
    33:     """No-attention: pure convolutional UNet (no per-resolution attention)."""
    34:     channels = (128, 256, 256, 256)
    35:     if os.environ.get('BLOCK_OUT_CHANNELS'):
    36:         channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    37:     layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))
    38: 
    39:     return UNet2DModel(
    40:         sample_size=32,
    41:         in_channels=3,
    42:         out_channels=3,
    43:         block_out_channels=channels,
    44:         down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    45:         up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    46:         layers_per_block=layers,
    47:         norm_num_groups=32,
    48:         norm_eps=1e-6,
    49:         act_fn="silu",
    50:         time_embedding_type="positional",
    51:         flip_sin_to_cos=False,
    52:         freq_shift=1,
    53:         downsample_padding=0,
    54:     ).to(device)
    55: 
    56: 
    57: # ============================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
