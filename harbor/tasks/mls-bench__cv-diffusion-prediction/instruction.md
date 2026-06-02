# MLS-Bench: cv-diffusion-prediction

# Diffusion Prediction Parameterization

## Objective

Design a prediction parameterization for unconditional CIFAR-10 diffusion that
improves FID under a fixed UNet backbone, training procedure, and DDIM sampler.

## Background

In DDPM training (Ho et al., 2020, arXiv:2006.11239), the model is shown a
noisy sample

```
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
```

and trained to predict a chosen target. Three standard parameterizations:

1. **Epsilon prediction** (Ho et al., 2020, arXiv:2006.11239) — predict the
   noise `epsilon`. Standard DDPM choice.
2. **`x_0` prediction** — directly predict the clean image `x_0`.
3. **`v` prediction** (Salimans & Ho, ICLR 2022, arXiv:2202.00512,
   "Progressive Distillation for Fast Sampling of Diffusion Models") —
   predict the velocity `v = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * x_0`.

The three are mathematically interchangeable (any one can be converted to the
others), but they give different loss landscapes, signal scaling across
timesteps, and gradient magnitudes, leading to different FID under a finite
training budget.

## Implementation Contract

You are given `custom_train.py`, a self-contained training script that trains
a UNet (`google/ddpm-cifar10-32` style architecture) on CIFAR-10. The
editable region contains two coupled functions:

1. `compute_training_target(x_0, noise, timesteps, schedule)` — defines what
   the model should predict during training.
2. `predict_x0(model_output, x_t, timesteps, schedule)` — recovers the
   predicted clean image from the model's output. Used during DDIM sampling.

These two functions must be **consistent**: the sampling procedure must
correctly invert the training parameterization.

The `schedule` dict provides precomputed noise-schedule tensors:

- `alphas_cumprod` — cumulative product of `(1 - beta)`.
- `sqrt_alpha` — `sqrt(alphas_cumprod)`.
- `sqrt_one_minus_alpha` — `sqrt(1 - alphas_cumprod)`.

## Fixed Pipeline

The following are fixed across baselines and submissions:

- Dataset: CIFAR-10 (32×32, unconditional).
- Backbone: `UNet2DModel` (diffusers) at multiple channel scales.
- Training: AdamW with EMA, multi-GPU DDP.
- Inference: DDIM (Song et al., 2020, arXiv:2010.02502).
- The contribution should be a transferable target parameterization, not a change to architecture, dataset, optimizer, noise schedule, sampling procedure, or metric computation.

## Baselines

| Baseline  | Description |
|-----------|-------------|
| `epsilon` | Predict `epsilon` (Ho et al., 2020, arXiv:2006.11239). DDPM default. |
| `x0pred`  | Predict the clean image `x_0` directly. |
| `vpred`   | Predict the velocity `v = sqrt(alpha) * epsilon - sqrt(1 - alpha) * x_0` (Salimans & Ho, ICLR 2022, arXiv:2202.00512). |

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/diffusers-main/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `diffusers-main/custom_train.py`
- editable lines **83–118**




## Readable Context


### `diffusers-main/custom_train.py`  [EDITABLE — lines 83–118 only]

```python
     1: """Unconditional DDPM Training on CIFAR-10 with configurable prediction target.
     2: 
     3: Uses diffusers UNet2DModel (google/ddpm-cifar10-32 architecture).
     4: Only the prediction parameterization (training target + x0 recovery) is editable.
     5: """
     6: 
     7: import copy
     8: import math
     9: import os
    10: import sys
    11: import time
    12: from datetime import timedelta
    13: 
    14: import numpy as np
    15: import torch
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
    28: # Model: UNet2DModel (google/ddpm-cifar10-32 architecture)
    29: # ============================================================================
    30: 
    31: UNET_CONFIG = dict(
    32:     sample_size=32,
    33:     in_channels=3,
    34:     out_channels=3,
    35:     block_out_channels=(128, 256, 256, 256),
    36:     down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
    37:     up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    38:     layers_per_block=2,
    39:     norm_num_groups=32,
    40:     norm_eps=1e-6,
    41:     act_fn="silu",
    42:     time_embedding_type="positional",
    43:     flip_sin_to_cos=False,
    44:     freq_shift=1,
    45:     downsample_padding=0,
    46: )
    47: 
    48: 
    49: def build_model(device):
    50:     """Build UNet2DModel with optional env var overrides for scaling."""
    51:     config = dict(UNET_CONFIG)
    52:     if os.environ.get('BLOCK_OUT_CHANNELS'):
    53:         config['block_out_channels'] = tuple(
    54:             int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    55:     if os.environ.get('LAYERS_PER_BLOCK'):
    56:         config['layers_per_block'] = int(os.environ['LAYERS_PER_BLOCK'])
    57:     return UNet2DModel(**config).to(device)
    58: 
    59: 
    60: # ============================================================================
    61: # Noise schedule helpers (precomputed from DDPMScheduler)
    62: # ============================================================================
    63: 
    64: def get_schedule_tensors(noise_scheduler, device):
    65:     """Extract alpha/sigma tensors from DDPMScheduler for use in editable region.
    66: 
    67:     Returns dict with:
    68:         alphas_cumprod:    [T] cumulative product of (1 - beta)
    69:         sqrt_alpha:        [T] sqrt(alphas_cumprod)
    70:         sqrt_one_minus_alpha: [T] sqrt(1 - alphas_cumprod)
    71:     """
    72:     acp = noise_scheduler.alphas_cumprod.to(device)
    73:     return {
    74:         "alphas_cumprod": acp,
    75:         "sqrt_alpha": acp.sqrt(),
    76:         "sqrt_one_minus_alpha": (1.0 - acp).sqrt(),
    77:     }
    78: 
    79: 
    80: # ============================================================================
    81: # Prediction parameterization (EDITABLE REGION)
    82: # ============================================================================
    83: 
    84: def compute_training_target(x_0, noise, timesteps, schedule):
    85:     """Compute the training target given clean images and noise.
    86: 
    87:     TODO: Implement your prediction parameterization here.
    88: 
    89:     The model will be trained to predict this target via MSE loss.
    90:     Must be consistent with predict_x0() below.
    91: 
    92:     Args:
    93:         x_0:       [B, C, H, W] clean images
    94:         noise:     [B, C, H, W] sampled Gaussian noise
    95:         timesteps: [B] integer timesteps (0 to T-1)
    96:         schedule:  dict with keys 'alphas_cumprod', 'sqrt_alpha',
    97:                    'sqrt_one_minus_alpha', each [T] tensors
    98: 
    99:     Returns: [B, C, H, W] target tensor
   100:     """
   101:     raise NotImplementedError("Implement compute_training_target")
   102: 
   103: 
   104: def predict_x0(model_output, x_t, timesteps, schedule):
   105:     """Recover predicted x_0 from the model's output.
   106: 
   107:     TODO: Must be consistent with compute_training_target() above.
   108: 
   109:     Used during DDIM sampling to convert model prediction back to x_0.
   110: 
   111:     Args:
   112:         model_output: [B, C, H, W] model prediction
   113:         x_t:          [B, C, H, W] noisy sample
   114:         timesteps:    [B] integer timesteps
   115:         schedule:     dict (same as compute_training_target)
   116: 
   117:     Returns: [B, C, H, W] predicted clean image
   118:     """
   119:     raise NotImplementedError("Implement predict_x0")
   120: 
   121: 
   122: # ============================================================================
   123: # Sampling — uses diffusers DDIMScheduler with predict_x0 bridge
   124: # ============================================================================
   125: 
   126: @torch.no_grad()
   127: def sample_images(model, schedule, num_samples, device, num_steps=1000,
   128:                   sample_steps=50, img_size=32, channels=3):
   129:     """Generate images via DDIM sampling (diffusers).
   130: 
   131:     Uses predict_x0() to convert model output to x_0, then feeds it to
   132:     DDIMScheduler with prediction_type='sample' for the actual DDIM step.
   133:     """
   134:     model.eval()
   135:     scheduler = DDIMScheduler(
   136:         num_train_timesteps=num_steps,
   137:         beta_schedule="linear",
   138:         beta_start=0.0001,
   139:         beta_end=0.02,
   140:         clip_sample=True,
   141:         set_alpha_to_one=False,
   142:         prediction_type="sample",
   143:     )
   144:     scheduler.set_timesteps(sample_steps)
   145: 
   146:     x = torch.randn(num_samples, channels, img_size, img_size, device=device)
   147: 
   148:     for t in scheduler.timesteps:
   149:         t_batch = t.expand(num_samples).to(device)
   150: 
   151:         with torch.amp.autocast(device_type='cuda'):
   152:             pred = model(x, t_batch).sample
   153:         # Convert model output to x_0 via editable predict_x0
   154:         pred_x0 = predict_x0(pred, x, t_batch, schedule)
   155:         # DDIMScheduler treats prediction_type="sample" as direct x_0 input
   156:         x = scheduler.step(pred_x0, t, x).prev_sample
   157: 
   158:     model.train()
   159:     return x.clamp(-1, 1)
   160: 
   161: 
   162: # ============================================================================
   163: # FID computation (using clean-fid)
   164: # ============================================================================
   165: 
   166: def compute_fid(model, schedule, device, num_samples=2048, num_steps=1000,
   167:                 sample_steps=50, img_size=32, batch_size=128,
   168:                 rank=0, world_size=1):
   169:     """Compute FID against CIFAR-10 train set using clean-fid.
   170: 
   171:     Supports distributed sampling: each rank generates its share of samples,
   172:     then rank 0 computes FID on all samples. Returns FID on rank 0, None on others.
   173:     """
   174:     import shutil
   175:     import tempfile
   176: 
   177:     from cleanfid import fid as cleanfid
   178:     import cleanfid.features as _feat
   179: 
   180:     # Use a shared directory so all ranks write to the same place
   181:     gen_dir = os.path.join(os.environ.get('OUTPUT_DIR', '/tmp/output'), '_fid_tmp')
   182:     if rank == 0:
   183:         if os.path.exists(gen_dir):
   184:             shutil.rmtree(gen_dir)
   185:         os.makedirs(gen_dir)
   186:     if world_size > 1:
   187:         dist.barrier()
   188: 
   189:     # Each rank generates its portion
   190:     per_rank = (num_samples + world_size - 1) // world_size
   191:     my_start = rank * per_rank
   192:     my_count = min(per_rank, num_samples - my_start)
   193: 
   194:     model.eval()
   195:     generated = 0
   196:     idx = my_start
   197:     while generated < my_count:
   198:         bs = min(batch_size, my_count - generated)
   199:         imgs = sample_images(model, schedule, bs, device, num_steps,
   200:                              sample_steps, img_size)
   201:         imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   202:         for j in range(bs):
   203:             img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
   204:             Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
   205:             idx += 1
   206:         generated += bs
   207: 
   208:     # Wait for all ranks to finish sampling
   209:     if world_size > 1:
   210:         dist.barrier()
   211: 
   212:     score = None
   213:     if rank == 0:
   214:         cache_dir = "/data/cleanfid"
   215:         os.makedirs(cache_dir, exist_ok=True)
   216: 
   217:         inception_path = os.path.join(cache_dir, "inception-2015-12-05.pt")
   218:         stats_path = os.path.join(cache_dir, "cifar10_clean_train_32.npz")
   219: 
   220:         missing = [p for p in (inception_path, stats_path) if not os.path.exists(p)]
   221:         if missing:
   222:             raise FileNotFoundError(
   223:                 "Missing clean-fid cache files prepared by `mlsbench data diffusers-main`: "
   224:                 + ", ".join(missing)
   225:             )
   226: 
   227:         _orig_build = _feat.build_feature_extractor
   228:         def _patched_build(mode, device=device, use_dataparallel=True):
   229:             from cleanfid.inception_torchscript import InceptionV3W
   230:             m = InceptionV3W(cache_dir, download=False,
   231:                              resize_inside=(mode == "legacy_tensorflow")).to(device)
   232:             m.eval()
   233:             if use_dataparallel:
   234:                 m = torch.nn.DataParallel(m)
   235:             return lambda x: m(x)
   236:         _feat.build_feature_extractor = _patched_build
   237: 
   238:         _orig_ref = _feat.get_reference_statistics
   239:         def _patched_ref(name, res, mode="clean", model_name="inception_v3",
   240:                          seed=0, split="train", metric="FID"):
   241:             fpath = os.path.join(cache_dir, f"{name}_{mode}_{split}_{res}.npz".lower())
   242:             stats = np.load(fpath)
   243:             return stats["mu"], stats["sigma"]
   244:         _feat.get_reference_statistics = _patched_ref
   245:         import cleanfid.fid as _fid_mod
   246:         _fid_mod.get_reference_statistics = _patched_ref
   247: 
   248:         score = cleanfid.compute_fid(
   249:             gen_dir, dataset_name="cifar10", dataset_res=32,
   250:             dataset_split="train", device=device, batch_size=batch_size, verbose=False,
   251:         )
   252: 
   253:         _feat.build_feature_extractor = _orig_build
   254:         _feat.get_reference_statistics = _orig_ref
   255:         _fid_mod.get_reference_statistics = _orig_ref
   256: 
   257:         shutil.rmtree(gen_dir)
   258: 
   259:     # Wait for rank 0 to finish FID computation
   260:     if world_size > 1:
   261:         dist.barrier()
   262: 
   263:     model.train()
   264:     return score
   265: 
   266: 
   267: def save_sample_images(model, schedule, device, output_dir, step, num_images=16,
   268:                        num_steps=1000, sample_steps=50, tag=""):
   269:     """Save a grid of sample images for visual inspection."""
   270:     imgs = sample_images(model, schedule, num_images, device, num_steps, sample_steps)
   271:     imgs = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   272: 
   273:     nrow = int(math.sqrt(num_images))
   274:     grid_h = nrow * 32
   275:     grid_w = nrow * 32
   276:     grid = Image.new('RGB', (grid_w, grid_h))
   277:     for i in range(num_images):
   278:         img_np = imgs[i].permute(1, 2, 0).numpy()
   279:         img = Image.fromarray(img_np)
   280:         row, col = divmod(i, nrow)
   281:         grid.paste(img, (col * 32, row * 32))
   282: 
   283:     suffix = f"_{tag}" if tag else ""
   284:     path = os.path.join(output_dir, f'samples_step{step}{suffix}.png')
   285:     grid.save(path)
   286:     print(f"Saved sample images to {path}", flush=True)
   287: 
   288: 
   289: # ============================================================================
   290: # Training Script
   291: # ============================================================================
   292: 
   293: if __name__ == '__main__':
   294:     seed = int(os.environ.get('SEED', 42))
   295:     data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
   296:     output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
   297:     max_steps = int(os.environ.get('MAX_STEPS', 10000))
   298:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 10000))
   299:     batch_size = int(os.environ.get('BATCH_SIZE', 128))
   300:     lr = float(os.environ.get('LR', 2e-4))
   301:     num_fid_samples = int(os.environ.get('NUM_FID_SAMPLES', 2048))
   302:     diffusion_steps = int(os.environ.get('DIFFUSION_STEPS', 1000))
   303:     sample_steps = int(os.environ.get('SAMPLE_STEPS', 50))
   304:     ema_rate = float(os.environ.get('EMA_RATE', 0.9999))
   305: 
   306:     # ── DDP setup ──────────────────────────────────────────────────────────
   307:     use_ddp = 'RANK' in os.environ
   308:     if use_ddp:
   309:         dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
   310:         local_rank = int(os.environ['LOCAL_RANK'])
   311:         rank = int(os.environ['RANK'])
   312:         world_size = int(os.environ['WORLD_SIZE'])
   313:         device = torch.device(f'cuda:{local_rank}')
   314:         torch.cuda.set_device(device)
   315:         is_main = (rank == 0)
   316:     else:
   317:         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   318:         rank = 0
   319:         world_size = 1
   320:         is_main = True
   321: 
   322:     torch.manual_seed(seed + rank)
   323:     os.makedirs(output_dir, exist_ok=True)
   324: 
   325:     # ── Data ────────────────────────────────────────────────────────────────
   326:     transform = transforms.Compose([
   327:         transforms.RandomHorizontalFlip(),
   328:         transforms.ToTensor(),
   329:         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
   330:     ])
   331:     dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=False)
   332:     if use_ddp:
   333:         sampler = torch.utils.data.DistributedSampler(
   334:             dataset, num_replicas=world_size, rank=rank, shuffle=True)
   335:         loader = torch.utils.data.DataLoader(
   336:             dataset, batch_size=batch_size, sampler=sampler,
   337:             num_workers=4, pin_memory=True, drop_last=True,
   338:         )
   339:     else:
   340:         loader = torch.utils.data.DataLoader(
   341:             dataset, batch_size=batch_size, shuffle=True,
   342:             num_workers=4, pin_memory=True, drop_last=True,
   343:         )
   344:     data_iter = iter(loader)
   345: 
   346:     # ── Noise scheduler (diffusers) ────────────────────────────────────────
   347:     noise_scheduler = DDPMScheduler(
   348:         num_train_timesteps=diffusion_steps,
   349:         beta_schedule="linear",
   350:         beta_start=0.0001,
   351:         beta_end=0.02,
   352:         clip_sample=True,
   353:         variance_type="fixed_large",
   354:     )
   355:     schedule = get_schedule_tensors(noise_scheduler, device)
   356: 
   357:     # ── Model ───────────────────────────────────────────────────────────────
   358:     net = build_model(device)
   359: 
   360:     ema_net = copy.deepcopy(net)
   361:     ema_net.requires_grad_(False)
   362: 
   363:     if use_ddp:
   364:         net = DDP(net, device_ids=[local_rank])
   365:     net_raw = net.module if use_ddp else net
   366: 
   367:     optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
   368:     scaler = torch.amp.GradScaler()
   369: 
   370:     num_params = sum(p.numel() for p in net_raw.parameters())
   371:     if is_main:
   372:         print(f"Model parameters: {num_params/1e6:.1f}M | GPUs: {world_size}", flush=True)
   373: 
   374:     # ── Training loop ────────────────────────────────────────────────────────
   375:     best_fid = float('inf')
   376:     t0 = time.time()
   377:     epoch = 0
   378: 
   379:     for step in range(1, max_steps + 1):
   380:         try:
   381:             x, _ = next(data_iter)
   382:         except StopIteration:
   383:             epoch += 1
   384:             if use_ddp:
   385:                 sampler.set_epoch(epoch)
   386:             data_iter = iter(loader)
   387:             x, _ = next(data_iter)
   388: 
   389:         x = x.to(device)
   390:         B = x.shape[0]
   391: 
   392:         # Sample random timesteps and add noise
   393:         t = torch.randint(0, diffusion_steps, (B,), device=device).long()
   394:         noise = torch.randn_like(x)
   395:         x_t = noise_scheduler.add_noise(x, noise, t)
   396: 
   397:         # Compute target using editable parameterization
   398:         target = compute_training_target(x, noise, t, schedule)
   399: 
   400:         # Forward pass
   401:         with torch.amp.autocast(device_type='cuda'):
   402:             pred = net(x_t, t).sample
   403:             loss = F.mse_loss(pred, target)
   404: 
   405:         optimizer.zero_grad()
   406:         scaler.scale(loss).backward()
   407:         scaler.unscale_(optimizer)
   408:         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
   409:         scaler.step(optimizer)
   410:         scaler.update()
   411: 
   412:         # Update EMA
   413:         with torch.no_grad():
   414:             for p_ema, p in zip(ema_net.parameters(), net_raw.parameters()):
   415:                 p_ema.mul_(ema_rate).add_(p, alpha=1 - ema_rate)
   416: 
   417:         if is_main and step % 200 == 0:
   418:             dt_elapsed = time.time() - t0
   419:             print(f"step {step}/{max_steps} | loss {loss.item():.4f} | {dt_elapsed:.1f}s",
   420:                   flush=True)
   421:             t0 = time.time()
   422: 
   423:         if step % eval_interval == 0 or step == max_steps:
   424:             if is_main:
   425:                 print(f"Eval at step {step}...", flush=True)
   426:                 save_sample_images(net_raw, schedule, device, output_dir, step,
   427:                                    num_steps=diffusion_steps, sample_steps=sample_steps,
   428:                                    tag="net")
   429:                 save_sample_images(ema_net, schedule, device, output_dir, step,
   430:                                    num_steps=diffusion_steps, sample_steps=sample_steps,
   431:                                    tag="ema")
   432:             eval_model = ema_net if step >= 20000 else net_raw
   433:             fid = compute_fid(eval_model, schedule, device,
   434:                               num_samples=num_fid_samples,
   435:                               num_steps=diffusion_steps,
   436:                               sample_steps=sample_steps,
   437:                               rank=rank, world_size=world_size)
   438:             if is_main:
   439:                 print(f"TRAIN_METRICS: step={step}, loss={loss.item():.4f}, fid={fid:.2f}",
   440:                       flush=True)
   441:                 if fid < best_fid:
   442:                     best_fid = fid
   443: 
   444:     # ── Save & final eval ────────────────────────────────────────────────────
   445:     if is_main:
   446:         print(f"Saving checkpoint to {output_dir}/checkpoint.pth", flush=True)
   447:         torch.save({
   448:             'step': max_steps,
   449:             'model_state_dict': net_raw.state_dict(),
   450:             'ema_model_state_dict': ema_net.state_dict(),
   451:             'optimizer_state_dict': optimizer.state_dict(),
   452:             'best_fid': best_fid,
   453:         }, os.path.join(output_dir, 'checkpoint.pth'))
   454: 
   455:         save_sample_images(net_raw, schedule, device, output_dir, max_steps,
   456:                            num_steps=diffusion_steps, sample_steps=sample_steps,
   457:                            tag="net_final")
   458:         save_sample_images(ema_net, schedule, device, output_dir, max_steps,
   459:                            num_steps=diffusion_steps, sample_steps=sample_steps,
   460:                            tag="ema_final")
   461: 
   462:     eval_model = ema_net if max_steps >= 20000 else net_raw
   463:     fid = compute_fid(eval_model, schedule, device,
   464:                       num_samples=num_fid_samples,
   465:                       num_steps=diffusion_steps,
   466:                       sample_steps=sample_steps,
   467:                       rank=rank, world_size=world_size)
   468:     if is_main:
   469:         print(f"TEST_METRICS: fid={fid:.2f}, best_fid={best_fid:.2f}", flush=True)
   470: 
   471:     if use_ddp:
   472:         dist.destroy_process_group()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `epsilon` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 83–94:
    80: # ============================================================================
    81: # Prediction parameterization (EDITABLE REGION)
    82: # ============================================================================
    83: def compute_training_target(x_0, noise, timesteps, schedule):
    84:     # Epsilon prediction: model learns to predict the added noise
    85:     return noise
    86: 
    87: 
    88: def predict_x0(model_output, x_t, timesteps, schedule):
    89:     # Recover x_0 from epsilon prediction:
    90:     # x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * eps
    91:     # => x_0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
    92:     sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    93:     sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    94:     return (x_t - sqrt_one_minus_alpha * model_output) / sqrt_alpha.clamp(min=1e-8)
    95:     raise NotImplementedError("Implement predict_x0")
    96: 
    97: 
```

### `vpred` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 83–97:
    80: # ============================================================================
    81: # Prediction parameterization (EDITABLE REGION)
    82: # ============================================================================
    83: def compute_training_target(x_0, noise, timesteps, schedule):
    84:     # V-prediction: v = sqrt(alpha) * noise - sqrt(1-alpha) * x_0
    85:     sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    86:     sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    87:     return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
    88: 
    89: 
    90: def predict_x0(model_output, x_t, timesteps, schedule):
    91:     # Recover x_0 from v-prediction:
    92:     # v = sqrt(alpha) * eps - sqrt(1-alpha) * x_0
    93:     # x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * eps
    94:     # => x_0 = sqrt(alpha) * x_t - sqrt(1-alpha) * v
    95:     sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    96:     sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    97:     return sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
    98:     raise NotImplementedError("Implement predict_x0")
    99: 
   100: 
```

### `x0pred` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 83–90:
    80: # ============================================================================
    81: # Prediction parameterization (EDITABLE REGION)
    82: # ============================================================================
    83: def compute_training_target(x_0, noise, timesteps, schedule):
    84:     # X0-prediction: model directly predicts the clean image
    85:     return x_0
    86: 
    87: 
    88: def predict_x0(model_output, x_t, timesteps, schedule):
    89:     # Model output IS x_0, no conversion needed
    90:     return model_output
    91:     raise NotImplementedError("Implement predict_x0")
    92: 
    93: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
