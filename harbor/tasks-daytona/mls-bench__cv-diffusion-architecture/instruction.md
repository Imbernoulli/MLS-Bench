# MLS-Bench: cv-diffusion-architecture

# Diffusion Model Architecture Design

## Objective

Design a UNet backbone for unconditional image diffusion that achieves
better generation quality than the standard DDPM-style architectures, under
a fixed training and sampling pipeline.

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
The model must follow these widths: its size (parameters + buffers) may be at
most 5% above the `full-attn` baseline built at the same `BLOCK_OUT_CHANNELS`
and `LAYERS_PER_BLOCK`, and its prediction must come from its registered
parameters. The fixed script checks the returned model, the parameters the
optimizer trains and the model evaluated for FID, and rejects a run above the
cap (no metrics).

## Fixed Pipeline

The training and sampling pipeline (training target/loss, optimizer, EMA,
noise schedule, and the DDIM sampler of Song et al., 2020, arXiv:2010.02502)
is fixed by the harness and not editable. Generation quality is scored with
FID. The model receives channel widths via the `BLOCK_OUT_CHANNELS` env var
(and `LAYERS_PER_BLOCK`, default 2).

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
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

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
    62: # Fixed: per-tier parameter budget
    63: # ============================================================================
    64: 
    65: # Each evaluation tier fixes the channel widths (BLOCK_OUT_CHANNELS) and with
    66: # them the model size. The cap is the parameter count of the largest baseline
    67: # (full-attn: self-attention at every resolution) at the tier's widths, plus
    68: # 5%. It is counted on the returned model (parameters + buffers), on the
    69: # parameters the fixed optimizer trains, and on the model FID is computed
    70: # from; a model above it is rejected.
    71: _PARAM_CAP_TOLERANCE = 1.05
    72: 
    73: 
    74: def _launch_env(name, default=""):
    75:     """An environment variable as the launch script set it.
    76: 
    77:     Read from the process's initial environment, which later writes to
    78:     os.environ do not change.
    79:     """
    80:     try:
    81:         with open("/proc/self/environ", "rb") as f:
    82:             entries = f.read().split(b"\0")
    83:     except OSError:
    84:         return os.environ.get(name, default)
    85:     prefix = name.encode() + b"="
    86:     for entry in entries:
    87:         if entry.startswith(prefix):
    88:             return entry[len(prefix):].decode()
    89:     return default
    90: 
    91: 
    92: def _tier_param_cap():
    93:     """(cap, widths) for the tier this run was launched with."""
    94:     widths = _launch_env("BLOCK_OUT_CHANNELS").replace(" ", "")
    95:     if not widths:
    96:         return None, widths
    97:     channels = tuple(int(c) for c in widths.split(","))
    98:     layers = int(_launch_env("LAYERS_PER_BLOCK", "2"))
    99:     from diffusers.models.unets.unet_2d import UNet2DModel as _RefUNet
   100:     with torch.random.fork_rng(devices=[]), torch.device("meta"):
   101:         ref = _RefUNet(
   102:             sample_size=32, in_channels=3, out_channels=3,
   103:             block_out_channels=channels,
   104:             down_block_types=("AttnDownBlock2D",) * len(channels),
   105:             up_block_types=("AttnUpBlock2D",) * len(channels),
   106:             layers_per_block=layers, norm_num_groups=32, norm_eps=1e-6,
   107:             act_fn="silu", time_embedding_type="positional",
   108:             flip_sin_to_cos=False, freq_shift=1, downsample_padding=0,
   109:         )
   110:     return int(_model_size(ref) * _PARAM_CAP_TOLERANCE), widths
   111: 
   112: 
   113: def _model_size(model):
   114:     """Parameters + buffers of a module (shared tensors counted once)."""
   115:     n = 0
   116:     for p in model.parameters():
   117:         n += p.numel()
   118:     for b in model.buffers():
   119:         n += b.numel()
   120:     return n
   121: 
   122: 
   123: def _check_param_budget(what, n_params, cap, widths):
   124:     if cap is not None and n_params > cap:
   125:         raise RuntimeError(
   126:             f"PARAM_BUDGET_EXCEEDED: {what} has {n_params:,} parameters, above "
   127:             f"the {cap:,} cap for BLOCK_OUT_CHANNELS={widths} (full-attn UNet "
   128:             f"at these widths + 5%). The model must follow the given widths."
   129:         )
   130: 
   131: 
   132: def _untrained_grad_leaves(output, trained_ids):
   133:     """Numel of grad-requiring leaves feeding `output` that the fixed
   134:     optimizer does not train (weights hidden outside the registered module)."""
   135:     hidden = 0
   136:     seen = set()
   137:     stack = [output.grad_fn]
   138:     while stack:
   139:         fn = stack.pop()
   140:         if fn is None or fn in seen:
   141:             continue
   142:         seen.add(fn)
   143:         var = getattr(fn, "variable", None)
   144:         if var is not None and id(var) not in trained_ids:
   145:             hidden += var.numel()
   146:         for nxt, _ in fn.next_functions:
   147:             stack.append(nxt)
   148:     return hidden
   149: 
   150: 
   151: # ============================================================================
   152: # Fixed: epsilon prediction
   153: # ============================================================================
   154: 
   155: def get_schedule_tensors(noise_scheduler, device):
   156:     acp = noise_scheduler.alphas_cumprod.to(device)
   157:     return {
   158:         "alphas_cumprod": acp,
   159:         "sqrt_alpha": acp.sqrt(),
   160:         "sqrt_one_minus_alpha": (1.0 - acp).sqrt(),
   161:     }
   162: 
   163: 
   164: def compute_training_target(x_0, noise, timesteps, schedule):
   165:     """Epsilon prediction — fixed, not editable."""
   166:     return noise
   167: 
   168: 
   169: def predict_x0(model_output, x_t, timesteps, schedule):
   170:     """Recover x_0 from epsilon prediction — fixed, not editable."""
   171:     sa = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
   172:     soma = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
   173:     return (x_t - soma * model_output) / sa
   174: 
   175: 
   176: # ============================================================================
   177: # Sampling — DDIM with epsilon prediction
   178: # ============================================================================
   179: 
   180: @torch.no_grad()
   181: def sample_images(model, schedule, num_samples, device, num_steps=1000,
   182:                   sample_steps=50, img_size=32, channels=3):
   183:     model.eval()
   184:     scheduler = DDIMScheduler(
   185:         num_train_timesteps=num_steps,
   186:         beta_schedule="linear",
   187:         beta_start=0.0001,
   188:         beta_end=0.02,
   189:         clip_sample=True,
   190:         set_alpha_to_one=False,
   191:         prediction_type="epsilon",
   192:     )
   193:     scheduler.set_timesteps(sample_steps)
   194: 
   195:     x = torch.randn(num_samples, channels, img_size, img_size, device=device)
   196: 
   197:     for t in scheduler.timesteps:
   198:         t_batch = t.expand(num_samples).to(device)
   199:         with torch.amp.autocast(device_type='cuda'):
   200:             noise_pred = model(x, t_batch).sample
   201:         x = scheduler.step(noise_pred, t, x).prev_sample
   202: 
   203:     model.train()
   204:     return x.clamp(-1, 1)
   205: 
   206: 
   207: # ============================================================================
   208: # FID computation (using clean-fid)
   209: # ============================================================================
   210: 
   211: def compute_fid(model, schedule, device, num_samples=2048, num_steps=1000,
   212:                 sample_steps=50, img_size=32, batch_size=128,
   213:                 rank=0, world_size=1):
   214:     import shutil
   215:     from cleanfid import fid as cleanfid
   216:     import cleanfid.features as _feat
   217: 
   218:     gen_dir = os.path.join(os.environ.get('OUTPUT_DIR', '/tmp/output'), '_fid_tmp')
   219:     if rank == 0:
   220:         if os.path.exists(gen_dir):
   221:             shutil.rmtree(gen_dir)
   222:         os.makedirs(gen_dir)
   223:     if world_size > 1:
   224:         dist.barrier()
   225: 
   226:     per_rank = (num_samples + world_size - 1) // world_size
   227:     my_start = rank * per_rank
   228:     my_count = min(per_rank, num_samples - my_start)
   229: 
   230:     model.eval()
   231:     generated = 0
   232:     idx = my_start
   233:     while generated < my_count:
   234:         bs = min(batch_size, my_count - generated)
   235:         imgs = sample_images(model, schedule, bs, device, num_steps,
   236:                              sample_steps, img_size)
   237:         imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   238:         for j in range(bs):
   239:             img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
   240:             Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
   241:             idx += 1
   242:         generated += bs
   243: 
   244:     if world_size > 1:
   245:         dist.barrier()
   246: 
   247:     score = None
   248:     if rank == 0:
   249:         cache_dir = "/data/cleanfid"
   250:         os.makedirs(cache_dir, exist_ok=True)
   251: 
   252:         inception_path = os.path.join(cache_dir, "inception-2015-12-05.pt")
   253:         stats_path = os.path.join(cache_dir, "cifar10_clean_train_32.npz")
   254: 
   255:         missing = [p for p in (inception_path, stats_path) if not os.path.exists(p)]
   256:         if missing:
   257:             raise FileNotFoundError(
   258:                 "Missing clean-fid cache files prepared by `mlsbench data diffusers-main`: "
   259:                 + ", ".join(missing)
   260:             )
   261: 
   262:         _orig_build = _feat.build_feature_extractor
   263:         def _patched_build(mode, device=device, use_dataparallel=True):
   264:             from cleanfid.inception_torchscript import InceptionV3W
   265:             m = InceptionV3W(cache_dir, download=False,
   266:                              resize_inside=(mode == "legacy_tensorflow")).to(device)
   267:             m.eval()
   268:             if use_dataparallel:
   269:                 m = torch.nn.DataParallel(m)
   270:             return lambda x: m(x)
   271:         _feat.build_feature_extractor = _patched_build
   272: 
   273:         _orig_ref = _feat.get_reference_statistics
   274:         def _patched_ref(name, res, mode="clean", model_name="inception_v3",
   275:                          seed=0, split="train", metric="FID"):
   276:             fpath = os.path.join(cache_dir, f"{name}_{mode}_{split}_{res}.npz".lower())
   277:             stats = np.load(fpath)
   278:             return stats["mu"], stats["sigma"]
   279:         _feat.get_reference_statistics = _patched_ref
   280:         import cleanfid.fid as _fid_mod
   281:         _fid_mod.get_reference_statistics = _patched_ref
   282:         _orig_fid_build = _fid_mod.build_feature_extractor
   283:         _fid_mod.build_feature_extractor = _patched_build
   284: 
   285:         score = cleanfid.compute_fid(
   286:             gen_dir, dataset_name="cifar10", dataset_res=32,
   287:             dataset_split="train", device=device, batch_size=batch_size, verbose=False,
   288:         )
   289: 
   290:         _feat.build_feature_extractor = _orig_build
   291:         _feat.get_reference_statistics = _orig_ref
   292:         _fid_mod.get_reference_statistics = _orig_ref
   293:         _fid_mod.build_feature_extractor = _orig_fid_build
   294: 
   295:         shutil.rmtree(gen_dir)
   296: 
   297:     if world_size > 1:
   298:         dist.barrier()
   299: 
   300:     model.train()
   301:     return score
   302: 
   303: 
   304: def save_sample_images(model, schedule, device, output_dir, step, num_images=16,
   305:                        num_steps=1000, sample_steps=50, tag=""):
   306:     imgs = sample_images(model, schedule, num_images, device, num_steps, sample_steps)
   307:     imgs = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   308: 
   309:     nrow = int(math.sqrt(num_images))
   310:     grid_h = nrow * 32
   311:     grid_w = nrow * 32
   312:     grid = Image.new('RGB', (grid_w, grid_h))
   313:     for i in range(num_images):
   314:         img_np = imgs[i].permute(1, 2, 0).numpy()
   315:         img = Image.fromarray(img_np)
   316:         row, col = divmod(i, nrow)
   317:         grid.paste(img, (col * 32, row * 32))
   318: 
   319:     suffix = f"_{tag}" if tag else ""
   320:     path = os.path.join(output_dir, f'samples_step{step}{suffix}.png')
   321:     grid.save(path)
   322:     print(f"Saved sample images to {path}", flush=True)
   323: 
   324: 
   325: # ============================================================================
   326: # Training Script
   327: # ============================================================================
   328: 
   329: if __name__ == '__main__':
   330:     seed = int(os.environ.get('SEED', 42))
   331:     data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
   332:     output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
   333:     max_steps = int(os.environ.get('MAX_STEPS', 10000))
   334:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 10000))
   335:     batch_size = int(os.environ.get('BATCH_SIZE', 128))
   336:     lr = float(os.environ.get('LR', 2e-4))
   337:     num_fid_samples = int(os.environ.get('NUM_FID_SAMPLES', 2048))
   338:     diffusion_steps = int(os.environ.get('DIFFUSION_STEPS', 1000))
   339:     sample_steps = int(os.environ.get('SAMPLE_STEPS', 50))
   340:     ema_rate = float(os.environ.get('EMA_RATE', 0.9999))
   341: 
   342:     # ── DDP setup ──────────────────────────────────────────────────────────
   343:     use_ddp = 'RANK' in os.environ
   344:     if use_ddp:
   345:         dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
   346:         local_rank = int(os.environ['LOCAL_RANK'])
   347:         rank = int(os.environ['RANK'])
   348:         world_size = int(os.environ['WORLD_SIZE'])
   349:         device = torch.device(f'cuda:{local_rank}')
   350:         torch.cuda.set_device(device)
   351:         is_main = (rank == 0)
   352:     else:
   353:         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   354:         rank = 0
   355:         world_size = 1
   356:         is_main = True
   357: 
   358:     torch.manual_seed(seed + rank)
   359:     os.makedirs(output_dir, exist_ok=True)
   360: 
   361:     # ── Data ────────────────────────────────────────────────────────────────
   362:     transform = transforms.Compose([
   363:         transforms.RandomHorizontalFlip(),
   364:         transforms.ToTensor(),
   365:         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
   366:     ])
   367:     dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=False)
   368:     if use_ddp:
   369:         sampler = torch.utils.data.DistributedSampler(
   370:             dataset, num_replicas=world_size, rank=rank, shuffle=True)
   371:         loader = torch.utils.data.DataLoader(
   372:             dataset, batch_size=batch_size, sampler=sampler,
   373:             num_workers=4, pin_memory=True, drop_last=True,
   374:         )
   375:     else:
   376:         loader = torch.utils.data.DataLoader(
   377:             dataset, batch_size=batch_size, shuffle=True,
   378:             num_workers=4, pin_memory=True, drop_last=True,
   379:         )
   380:     data_iter = iter(loader)
   381: 
   382:     # ── Noise scheduler ────────────────────────────────────────────────────
   383:     noise_scheduler = DDPMScheduler(
   384:         num_train_timesteps=diffusion_steps,
   385:         beta_schedule="linear",
   386:         beta_start=0.0001,
   387:         beta_end=0.02,
   388:         clip_sample=True,
   389:         variance_type="fixed_large",
   390:     )
   391:     schedule = get_schedule_tensors(noise_scheduler, device)
   392: 
   393:     # ── Model ───────────────────────────────────────────────────────────────
   394:     net = build_model(device)
   395: 
   396:     ema_net = copy.deepcopy(net)
   397:     ema_net.requires_grad_(False)
   398: 
   399:     if use_ddp:
   400:         net = DDP(net, device_ids=[local_rank])
   401:     net_raw = net.module if use_ddp else net
   402: 
   403:     optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-8)
   404:     scaler = torch.amp.GradScaler()
   405: 
   406:     num_params = sum(p.numel() for p in net_raw.parameters())
   407:     if is_main:
   408:         print(f"Model parameters: {num_params/1e6:.1f}M | GPUs: {world_size}", flush=True)
   409: 
   410:     # ── Parameter budget (fixed) ────────────────────────────────────────────
   411:     if not isinstance(net_raw, nn.Module):
   412:         raise TypeError("build_model must return an nn.Module")
   413:     param_cap, budget_tier = _tier_param_cap()
   414:     trained_params = {}
   415:     for group in optimizer.param_groups:
   416:         for p in group["params"]:
   417:             trained_params[id(p)] = p
   418:     trained_ids = set(trained_params)
   419:     model_size = _model_size(net_raw)
   420:     _check_param_budget("the model returned by build_model", model_size,
   421:                         param_cap, budget_tier)
   422:     _check_param_budget("the trained parameter set",
   423:                         sum(p.numel() for p in trained_params.values()),
   424:                         param_cap, budget_tier)
   425:     _check_param_budget("the EMA model", _model_size(ema_net), param_cap,
   426:                         budget_tier)
   427:     if is_main:
   428:         if param_cap is None:
   429:             print(f"Parameter budget: {model_size:,} (no cap: "
   430:                   f"BLOCK_OUT_CHANNELS is not set)", flush=True)
   431:         else:
   432:             print(f"Parameter budget: {model_size:,} / {param_cap:,} "
   433:                   f"(BLOCK_OUT_CHANNELS={budget_tier})", flush=True)
   434: 
   435:     # ── Training loop ────────────────────────────────────────────────────────
   436:     best_fid = float('inf')
   437:     t0 = time.time()
   438:     epoch = 0
   439: 
   440:     for step in range(1, max_steps + 1):
   441:         try:
   442:             x, _ = next(data_iter)
   443:         except StopIteration:
   444:             epoch += 1
   445:             if use_ddp:
   446:                 sampler.set_epoch(epoch)
   447:             data_iter = iter(loader)
   448:             x, _ = next(data_iter)
   449: 
   450:         x = x.to(device)
   451:         B = x.shape[0]
   452: 
   453:         t = torch.randint(0, diffusion_steps, (B,), device=device).long()
   454:         noise = torch.randn_like(x)
   455:         x_t = noise_scheduler.add_noise(x, noise, t)
   456: 
   457:         target = compute_training_target(x, noise, t, schedule)
   458: 
   459:         with torch.amp.autocast(device_type='cuda'):
   460:             pred = net(x_t, t).sample
   461:             loss = F.mse_loss(pred, target)
   462: 
   463:         # The prediction must come from the registered, optimizer-trained
   464:         # weights only: checked on step 1 and on ~5% of steps at random.
   465:         if step == 1 or os.urandom(1)[0] < 13:
   466:             hidden = _untrained_grad_leaves(pred, trained_ids)
   467:             if hidden:
   468:                 raise RuntimeError(
   469:                     f"PARAM_BUDGET_EXCEEDED: the prediction depends on {hidden:,} "
   470:                     f"trainable weights that are not registered parameters of "
   471:                     f"the model returned by build_model")
   472: 
   473:         optimizer.zero_grad()
   474:         scaler.scale(loss).backward()
   475:         scaler.unscale_(optimizer)
   476:         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
   477:         scaler.step(optimizer)
   478:         scaler.update()
   479: 
   480:         with torch.no_grad():
   481:             for p_ema, p in zip(ema_net.parameters(), net_raw.parameters()):
   482:                 p_ema.mul_(ema_rate).add_(p, alpha=1 - ema_rate)
   483: 
   484:         if is_main and step % 200 == 0:
   485:             dt_elapsed = time.time() - t0
   486:             print(f"step {step}/{max_steps} | loss {loss.item():.4f} | {dt_elapsed:.1f}s",
   487:                   flush=True)
   488:             t0 = time.time()
   489: 
   490:         if step % eval_interval == 0 or step == max_steps:
   491:             if is_main:
   492:                 print(f"Eval at step {step}...", flush=True)
   493:                 save_sample_images(net_raw, schedule, device, output_dir, step,
   494:                                    num_steps=diffusion_steps, sample_steps=sample_steps,
   495:                                    tag="net")
   496:                 save_sample_images(ema_net, schedule, device, output_dir, step,
   497:                                    num_steps=diffusion_steps, sample_steps=sample_steps,
   498:                                    tag="ema")
   499:             eval_model = ema_net if step >= 20000 else net_raw
   500:             _check_param_budget("the evaluated model", _model_size(eval_model),
   501:                                 param_cap, budget_tier)
   502:             fid = compute_fid(eval_model, schedule, device,
   503:                               num_samples=num_fid_samples,
   504:                               num_steps=diffusion_steps,
   505:                               sample_steps=sample_steps,
   506:                               rank=rank, world_size=world_size)
   507:             if is_main:
   508:                 print(f"TRAIN_METRICS: step={step}, loss={loss.item():.4f}, fid={fid:.2f}",
   509:                       flush=True)
   510:                 if fid < best_fid:
   511:                     best_fid = fid
   512: 
   513:     # ── Save & final eval ────────────────────────────────────────────────────
   514:     if is_main:
   515:         print(f"Saving checkpoint to {output_dir}/checkpoint.pth", flush=True)
   516:         torch.save({
   517:             'step': max_steps,
   518:             'model_state_dict': net_raw.state_dict(),
   519:             'ema_model_state_dict': ema_net.state_dict(),
   520:             'optimizer_state_dict': optimizer.state_dict(),
   521:             'best_fid': best_fid,
   522:         }, os.path.join(output_dir, 'checkpoint.pth'))
   523: 
   524:         save_sample_images(net_raw, schedule, device, output_dir, max_steps,
   525:                            num_steps=diffusion_steps, sample_steps=sample_steps,
   526:                            tag="net_final")
   527:         save_sample_images(ema_net, schedule, device, output_dir, max_steps,
   528:                            num_steps=diffusion_steps, sample_steps=sample_steps,
   529:                            tag="ema_final")
   530: 
   531:     eval_model = ema_net if max_steps >= 20000 else net_raw
   532:     _check_param_budget("the evaluated model", _model_size(eval_model),
   533:                         param_cap, budget_tier)
   534:     fid = compute_fid(eval_model, schedule, device,
   535:                       num_samples=num_fid_samples,
   536:                       num_steps=diffusion_steps,
   537:                       sample_steps=sample_steps,
   538:                       rank=rank, world_size=world_size)
   539:     if is_main:
   540:         print(f"TEST_METRICS: fid={fid:.2f}, best_fid={best_fid:.2f}", flush=True)
   541: 
   542:     if use_ddp:
   543:         dist.destroy_process_group()
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

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
