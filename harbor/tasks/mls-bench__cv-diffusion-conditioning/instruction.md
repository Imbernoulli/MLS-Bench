# MLS-Bench: cv-diffusion-conditioning

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
  - Large:  `block_out_channels=(256, 512, 512, 512)`, ~140M params, batch 64.
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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/diffusers-main/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `diffusers-main/custom_train.py`
- editable lines **195–227**




## Readable Context


### `diffusers-main/custom_train.py`  [EDITABLE — lines 195–227 only]

```python
     1: """Class-Conditional DDPM Training on CIFAR-10.
     2: 
     3: Uses diffusers UNet2DModel backbone (same architecture as google/ddpm-cifar10-32)
     4: with configurable class-conditioning injection. Only the conditioning method
     5: (prepare_conditioning + ClassConditioner) is editable.
     6: """
     7: 
     8: import copy
     9: import math
    10: import os
    11: import sys
    12: import time
    13: 
    14: import numpy as np
    15: import torch
    16: import torch.distributed as dist
    17: import torch.nn as nn
    18: import torch.nn.functional as F
    19: from PIL import Image
    20: from torch.nn.parallel import DistributedDataParallel as DDP
    21: from torchvision import datasets, transforms
    22: 
    23: # Use diffusers from the external package
    24: sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    25: from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
    26: 
    27: 
    28: # ============================================================================
    29: # Conditioning utilities (available for baselines to use)
    30: # ============================================================================
    31: 
    32: def zero_module(module):
    33:     for p in module.parameters():
    34:         p.detach().zero_()
    35:     return module
    36: 
    37: 
    38: class CrossAttentionLayer(nn.Module):
    39:     """Cross-attention: features attend to class embedding as key/value."""
    40:     def __init__(self, channels, context_dim, num_heads=4):
    41:         super().__init__()
    42:         self.num_heads = num_heads
    43:         self.head_dim = channels // num_heads
    44:         self.norm = nn.GroupNorm(32, channels)
    45:         self.q_proj = nn.Linear(channels, channels)
    46:         self.k_proj = nn.Linear(context_dim, channels)
    47:         self.v_proj = nn.Linear(context_dim, channels)
    48:         self.out_proj = zero_module(nn.Linear(channels, channels))
    49: 
    50:     def forward(self, x, context):
    51:         B, C, H, W = x.shape
    52:         h = self.norm(x).view(B, C, -1).transpose(1, 2)
    53:         ctx = context.unsqueeze(1)
    54:         q = self.q_proj(h).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
    55:         k = self.k_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
    56:         v = self.v_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
    57:         attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
    58:         attn = F.softmax(attn, dim=-1)
    59:         out = torch.matmul(attn, v)
    60:         out = out.transpose(1, 2).reshape(B, H * W, C)
    61:         out = self.out_proj(out)
    62:         return x + out.transpose(1, 2).view(B, C, H, W)
    63: 
    64: 
    65: class AdaLNBlock(nn.Module):
    66:     """Adaptive LayerNorm-Zero: class embedding generates scale/shift/gate."""
    67:     def __init__(self, channels, cond_dim):
    68:         super().__init__()
    69:         self.norm = nn.GroupNorm(1, channels)
    70:         self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * channels))
    71:         nn.init.zeros_(self.proj[-1].weight)
    72:         nn.init.zeros_(self.proj[-1].bias)
    73: 
    74:     def forward(self, x, cond):
    75:         scale, shift, gate = self.proj(cond).unsqueeze(-1).unsqueeze(-1).chunk(3, dim=1)
    76:         h = self.norm(x.float()).type(x.dtype)
    77:         h = h * (1 + scale) + shift
    78:         return x + gate * (h - x)
    79: 
    80: 
    81: # ============================================================================
    82: # Conditional UNet: diffusers UNet2DModel backbone + class conditioning hooks
    83: # ============================================================================
    84: 
    85: class ConditionalUNet(nn.Module):
    86:     """Wraps UNet2DModel and adds configurable class-conditioning.
    87: 
    88:     The backbone is identical to google/ddpm-cifar10-32. Class conditioning
    89:     is injected via two editable components:
    90:       - prepare_conditioning(time_emb, class_emb): how class info enters time path
    91:       - ClassConditioner(channels, cond_dim): extra conditioning after each block
    92:     """
    93: 
    94:     # Architecture matching google/ddpm-cifar10-32
    95:     UNET_CONFIG = dict(
    96:         sample_size=32,
    97:         in_channels=3,
    98:         out_channels=3,
    99:         block_out_channels=(128, 256, 256, 256),
   100:         down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
   101:         up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
   102:         layers_per_block=2,
   103:         norm_num_groups=32,
   104:         norm_eps=1e-6,
   105:         act_fn="silu",
   106:         time_embedding_type="positional",
   107:         flip_sin_to_cos=False,
   108:         freq_shift=1,
   109:         downsample_padding=0,
   110:     )
   111: 
   112:     def __init__(self, num_classes=10):
   113:         super().__init__()
   114:         config = dict(self.UNET_CONFIG)
   115:         if os.environ.get('BLOCK_OUT_CHANNELS'):
   116:             config['block_out_channels'] = tuple(
   117:                 int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
   118:         if os.environ.get('LAYERS_PER_BLOCK'):
   119:             config['layers_per_block'] = int(os.environ['LAYERS_PER_BLOCK'])
   120:         self.unet = UNet2DModel(**config)
   121:         self.time_embed_dim = self.unet.config.block_out_channels[0] * 4
   122: 
   123:         # Class embedding: same dim as time_embed_dim for prepare_conditioning
   124:         self.class_embed = nn.Embedding(num_classes, self.time_embed_dim)
   125: 
   126:         # ClassConditioner per down block, mid block, up block
   127:         down_channels = [self.unet.config.block_out_channels[i]
   128:                          for i in range(len(self.unet.down_blocks))]
   129:         mid_ch = self.unet.config.block_out_channels[-1]
   130: 
   131:         reversed_ch = list(reversed(self.unet.config.block_out_channels))
   132:         up_channels = [reversed_ch[i] for i in range(len(self.unet.up_blocks))]
   133: 
   134:         self.down_cond = nn.ModuleList(
   135:             [ClassConditioner(ch, self.time_embed_dim) for ch in down_channels])
   136:         self.mid_cond = ClassConditioner(mid_ch, self.time_embed_dim)
   137:         self.up_cond = nn.ModuleList(
   138:             [ClassConditioner(ch, self.time_embed_dim) for ch in up_channels])
   139: 
   140:     def forward(self, sample, timestep, class_labels):
   141:         # --- Time embedding (reuse UNet internals) ---
   142:         timesteps = timestep
   143:         if not torch.is_tensor(timesteps):
   144:             timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
   145:         elif len(timesteps.shape) == 0:
   146:             timesteps = timesteps[None].to(sample.device)
   147:         timesteps = timesteps * torch.ones(
   148:             sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
   149: 
   150:         t_emb = self.unet.time_proj(timesteps)
   151:         t_emb = t_emb.to(dtype=self.unet.dtype)
   152:         emb = self.unet.time_embedding(t_emb)
   153: 
   154:         # --- Class embedding ---
   155:         class_emb = self.class_embed(class_labels)  # [B, time_embed_dim]
   156: 
   157:         # --- Editable: how class_emb interacts with time_emb ---
   158:         emb = prepare_conditioning(emb, class_emb)
   159: 
   160:         # --- UNet forward with ClassConditioner hooks ---
   161:         sample = self.unet.conv_in(sample)
   162: 
   163:         # Down
   164:         down_block_res_samples = (sample,)
   165:         for i, block in enumerate(self.unet.down_blocks):
   166:             sample, res_samples = block(hidden_states=sample, temb=emb)
   167:             sample = self.down_cond[i](sample, class_emb)
   168:             # Fix res_samples: last one should be the conditioned sample
   169:             res_samples = res_samples[:-1] + (sample,)
   170:             down_block_res_samples += res_samples
   171: 
   172:         # Mid
   173:         if self.unet.mid_block is not None:
   174:             sample = self.unet.mid_block(sample, emb)
   175:         sample = self.mid_cond(sample, class_emb)
   176: 
   177:         # Up
   178:         for i, block in enumerate(self.unet.up_blocks):
   179:             res_samples = down_block_res_samples[-len(block.resnets):]
   180:             down_block_res_samples = down_block_res_samples[:-len(block.resnets)]
   181:             sample = block(sample, res_samples, emb)
   182:             sample = self.up_cond[i](sample, class_emb)
   183: 
   184:         # Out
   185:         sample = self.unet.conv_norm_out(sample)
   186:         sample = self.unet.conv_act(sample)
   187:         sample = self.unet.conv_out(sample)
   188:         return sample
   189: 
   190: 
   191: # ============================================================================
   192: # Conditioning injection (EDITABLE REGION)
   193: # ============================================================================
   194: 
   195: def prepare_conditioning(time_emb, class_emb):
   196:     """Prepare the combined embedding used in ResBlocks.
   197: 
   198:     TODO: Implement your conditioning preparation here.
   199: 
   200:     Args:
   201:         time_emb:  [B, time_embed_dim] timestep embedding
   202:         class_emb: [B, time_embed_dim] class embedding
   203: 
   204:     Returns: [B, time_embed_dim] embedding used in ResBlocks
   205:     """
   206:     raise NotImplementedError("Implement prepare_conditioning")
   207: 
   208: 
   209: class ClassConditioner(nn.Module):
   210:     """Conditioning module applied after each UNet block.
   211: 
   212:     TODO: Implement your conditioning method here.
   213: 
   214:     Args (forward):
   215:         h:         [B, C, H, W] feature map
   216:         class_emb: [B, time_embed_dim] class embedding
   217: 
   218:     Available utilities:
   219:         CrossAttentionLayer(channels, context_dim, num_heads)
   220:         AdaLNBlock(channels, cond_dim)
   221:     """
   222:     def __init__(self, channels, cond_dim):
   223:         super().__init__()
   224:         raise NotImplementedError("Implement ClassConditioner.__init__")
   225: 
   226:     def forward(self, h, class_emb):
   227:         raise NotImplementedError("Implement ClassConditioner.forward")
   228: 
   229: 
   230: # ============================================================================
   231: # Sampling — uses diffusers DDIMScheduler
   232: # ============================================================================
   233: 
   234: @torch.no_grad()
   235: def sample_images(model, num_samples, device, num_classes=10, num_steps=1000,
   236:                   sample_steps=50, img_size=32, channels=3):
   237:     """Generate class-conditional images via DDIM sampling (diffusers)."""
   238:     model.eval()
   239:     scheduler = DDIMScheduler(
   240:         num_train_timesteps=num_steps,
   241:         beta_schedule="linear",
   242:         beta_start=0.0001,
   243:         beta_end=0.02,
   244:         clip_sample=True,
   245:         set_alpha_to_one=False,
   246:         prediction_type="epsilon",
   247:     )
   248:     scheduler.set_timesteps(sample_steps)
   249: 
   250:     x = torch.randn(num_samples, channels, img_size, img_size, device=device)
   251:     class_labels = torch.randint(0, num_classes, (num_samples,), device=device)
   252: 
   253:     for t in scheduler.timesteps:
   254:         t_batch = t.expand(num_samples).to(device)
   255:         with torch.amp.autocast(device_type='cuda'):
   256:             noise_pred = model(x, t_batch, class_labels)
   257:         x = scheduler.step(noise_pred, t, x).prev_sample
   258: 
   259:     model.train()
   260:     return x.clamp(-1, 1)
   261: 
   262: 
   263: # ============================================================================
   264: # FID computation (using clean-fid)
   265: # ============================================================================
   266: 
   267: def compute_fid(model, device, num_samples=2048, num_classes=10, num_steps=1000,
   268:                 sample_steps=50, img_size=32, batch_size=128,
   269:                 rank=0, world_size=1):
   270:     """Compute FID against CIFAR-10 train set using clean-fid.
   271: 
   272:     All ranks sample in parallel; rank 0 computes FID on the merged results.
   273:     """
   274:     import shutil
   275:     import tempfile
   276: 
   277:     model.eval()
   278: 
   279:     # Each rank samples its share
   280:     my_samples = num_samples // world_size
   281:     if rank < (num_samples % world_size):
   282:         my_samples += 1
   283:     start_idx = rank * (num_samples // world_size) + min(rank, num_samples % world_size)
   284: 
   285:     gen_dir = os.path.join(tempfile.gettempdir(), f"fid_gen_{os.getpid()}")
   286:     os.makedirs(gen_dir, exist_ok=True)
   287: 
   288:     generated = 0
   289:     idx = start_idx
   290:     while generated < my_samples:
   291:         bs = min(batch_size, my_samples - generated)
   292:         imgs = sample_images(model, bs, device, num_classes, num_steps,
   293:                              sample_steps, img_size)
   294:         imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   295:         for j in range(bs):
   296:             img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
   297:             Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
   298:             idx += 1
   299:         generated += bs
   300:         if rank == 0:
   301:             print(f"  Sampling: {generated}/{my_samples}", flush=True)
   302: 
   303:     # Sync all ranks before FID computation
   304:     if world_size > 1:
   305:         dist.barrier()
   306: 
   307:     # Rank 0 gathers all images and computes FID
   308:     score = 0.0
   309:     if rank == 0:
   310:         # Merge images from all ranks into one dir
   311:         merged_dir = os.path.join(tempfile.gettempdir(), "fid_merged")
   312:         if os.path.exists(merged_dir):
   313:             shutil.rmtree(merged_dir)
   314:         os.makedirs(merged_dir)
   315: 
   316:         # Copy from all per-rank dirs
   317:         for f in sorted(os.listdir(gen_dir)):
   318:             shutil.copy2(os.path.join(gen_dir, f), os.path.join(merged_dir, f))
   319: 
   320:         if world_size > 1:
   321:             # Other ranks wrote to /tmp on the same node
   322:             import glob
   323:             for other_dir in glob.glob(os.path.join(tempfile.gettempdir(), "fid_gen_*")):
   324:                 if other_dir == gen_dir:
   325:                     continue
   326:                 for f in os.listdir(other_dir):
   327:                     shutil.copy2(os.path.join(other_dir, f), os.path.join(merged_dir, f))
   328: 
   329:         from cleanfid import fid as cleanfid
   330:         import cleanfid.features as _feat
   331: 
   332:         cache_dir = "/data/cleanfid"
   333:         os.makedirs(cache_dir, exist_ok=True)
   334: 
   335:         inception_path = os.path.join(cache_dir, "inception-2015-12-05.pt")
   336:         stats_path = os.path.join(cache_dir, "cifar10_clean_train_32.npz")
   337:         missing = [p for p in (inception_path, stats_path) if not os.path.exists(p)]
   338:         if missing:
   339:             raise FileNotFoundError(
   340:                 "Missing clean-fid cache files prepared by `mlsbench data diffusers-main`: "
   341:                 + ", ".join(missing)
   342:             )
   343: 
   344:         _orig_build = _feat.build_feature_extractor
   345:         def _patched_build(mode, device=device, use_dataparallel=True):
   346:             from cleanfid.inception_torchscript import InceptionV3W
   347:             m = InceptionV3W(cache_dir, download=False,
   348:                              resize_inside=(mode == "legacy_tensorflow")).to(device)
   349:             m.eval()
   350:             if use_dataparallel:
   351:                 m = torch.nn.DataParallel(m)
   352:             return lambda x: m(x)
   353:         _feat.build_feature_extractor = _patched_build
   354: 
   355:         _orig_ref = _feat.get_reference_statistics
   356:         def _patched_ref(name, res, mode="clean", model_name="inception_v3",
   357:                          seed=0, split="train", metric="FID"):
   358:             fpath = os.path.join(cache_dir, f"{name}_{mode}_{split}_{res}.npz".lower())
   359:             stats = np.load(fpath)
   360:             return stats["mu"], stats["sigma"]
   361:         _feat.get_reference_statistics = _patched_ref
   362:         import cleanfid.fid as _fid_mod
   363:         _fid_mod.get_reference_statistics = _patched_ref
   364: 
   365:         print(f"  Computing FID on {len(os.listdir(merged_dir))} images...", flush=True)
   366:         score = cleanfid.compute_fid(
   367:             merged_dir, dataset_name="cifar10", dataset_res=32,
   368:             dataset_split="train", device=device, batch_size=batch_size, verbose=False,
   369:         )
   370: 
   371:         shutil.rmtree(merged_dir)
   372:         _feat.build_feature_extractor = _orig_build
   373:         _feat.get_reference_statistics = _orig_ref
   374:         _fid_mod.get_reference_statistics = _orig_ref
   375: 
   376:     # Clean up per-rank dir
   377:     shutil.rmtree(gen_dir, ignore_errors=True)
   378: 
   379:     if world_size > 1:
   380:         dist.barrier()
   381: 
   382:     model.train()
   383:     return score
   384: 
   385: 
   386: def save_sample_images(model, device, output_dir, step, num_images=16,
   387:                        num_classes=10, num_steps=1000, sample_steps=50, tag=""):
   388:     """Save a grid of sample images for visual inspection."""
   389:     imgs = sample_images(model, num_images, device, num_classes, num_steps, sample_steps)
   390:     imgs = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
   391: 
   392:     # Make a grid: 4x4
   393:     nrow = int(math.sqrt(num_images))
   394:     grid_h = nrow * 32
   395:     grid_w = nrow * 32
   396:     grid = Image.new('RGB', (grid_w, grid_h))
   397:     for i in range(num_images):
   398:         img_np = imgs[i].permute(1, 2, 0).numpy()
   399:         img = Image.fromarray(img_np)
   400:         row, col = divmod(i, nrow)
   401:         grid.paste(img, (col * 32, row * 32))
   402: 
   403:     suffix = f"_{tag}" if tag else ""
   404:     path = os.path.join(output_dir, f'samples_step{step}{suffix}.png')
   405:     grid.save(path)
   406:     print(f"Saved sample images to {path}", flush=True)
   407: 
   408: 
   409: # ============================================================================
   410: # Training Script
   411: # ============================================================================
   412: 
   413: if __name__ == '__main__':
   414:     seed = int(os.environ.get('SEED', 42))
   415:     data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
   416:     output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
   417:     max_steps = int(os.environ.get('MAX_STEPS', 10000))
   418:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 10000))
   419:     batch_size = int(os.environ.get('BATCH_SIZE', 128))
   420:     lr = float(os.environ.get('LR', 2e-4))
   421:     num_fid_samples = int(os.environ.get('NUM_FID_SAMPLES', 2048))
   422:     num_classes = int(os.environ.get('NUM_CLASSES', 10))
   423:     diffusion_steps = int(os.environ.get('DIFFUSION_STEPS', 1000))
   424:     sample_steps = int(os.environ.get('SAMPLE_STEPS', 50))
   425:     ema_rate = float(os.environ.get('EMA_RATE', 0.9999))
   426: 
   427:     # ── DDP setup ──────────────────────────────────────────────────────────
   428:     use_ddp = 'RANK' in os.environ
   429:     if use_ddp:
   430:         import datetime as _dt
   431:         dist.init_process_group(backend='nccl', timeout=_dt.timedelta(hours=20))
   432:         local_rank = int(os.environ['LOCAL_RANK'])
   433:         rank = int(os.environ['RANK'])
   434:         world_size = int(os.environ['WORLD_SIZE'])
   435:         device = torch.device(f'cuda:{local_rank}')
   436:         torch.cuda.set_device(device)
   437:         is_main = (rank == 0)
   438:     else:
   439:         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   440:         rank = 0
   441:         world_size = 1
   442:         is_main = True
   443: 
   444:     torch.manual_seed(seed + rank)
   445:     os.makedirs(output_dir, exist_ok=True)
   446: 
   447:     # ── Data ────────────────────────────────────────────────────────────────
   448:     transform = transforms.Compose([
   449:         transforms.RandomHorizontalFlip(),
   450:         transforms.ToTensor(),
   451:         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
   452:     ])
   453:     dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=False)
   454:     if use_ddp:
   455:         sampler = torch.utils.data.DistributedSampler(
   456:             dataset, num_replicas=world_size, rank=rank, shuffle=True)
   457:         loader = torch.utils.data.DataLoader(
   458:             dataset, batch_size=batch_size, sampler=sampler,
   459:             num_workers=4, pin_memory=True, drop_last=True,
   460:         )
   461:     else:
   462:         loader = torch.utils.data.DataLoader(
   463:             dataset, batch_size=batch_size, shuffle=True,
   464:             num_workers=4, pin_memory=True, drop_last=True,
   465:         )
   466:     data_iter = iter(loader)
   467: 
   468:     # ── Noise scheduler (diffusers) ────────────────────────────────────────
   469:     noise_scheduler = DDPMScheduler(
   470:         num_train_timesteps=diffusion_steps,
   471:         beta_schedule="linear",
   472:         beta_start=0.0001,
   473:         beta_end=0.02,
   474:         prediction_type="epsilon",
   475:         clip_sample=True,
   476:         variance_type="fixed_large",
   477:     )
   478: 
   479:     # ── Model ───────────────────────────────────────────────────────────────
   480:     net = ConditionalUNet(num_classes=num_classes).to(device)
   481: 
   482:     # Parameter budget check (1.05x largest baseline: Cross-Attention)
   483:     # Build a temporary reference model with the cross-attn baseline to compute budget.
   484:     def _budget_prepare_conditioning(time_emb, class_emb):
   485:         return time_emb
   486:     class _BudgetClassConditioner(nn.Module):
   487:         def __init__(self, channels, cond_dim):
   488:             super().__init__()
   489:             self.cross_attn = CrossAttentionLayer(channels, cond_dim, num_heads=4)
   490:         def forward(self, h, class_emb):
   491:             return self.cross_attn(h, class_emb)
   492:     _orig_prepare = globals()['prepare_conditioning']
   493:     _orig_conditioner = globals()['ClassConditioner']
   494:     globals()['prepare_conditioning'] = _budget_prepare_conditioning
   495:     globals()['ClassConditioner'] = _BudgetClassConditioner
   496:     _ref_net = ConditionalUNet(num_classes=num_classes)
   497:     _max_budget = int(sum(p.numel() for p in _ref_net.parameters()) * 1.05)
   498:     del _ref_net
   499:     globals()['prepare_conditioning'] = _orig_prepare
   500:     globals()['ClassConditioner'] = _orig_conditioner

[truncated: showing at most 500 lines / 60000 bytes from diffusers-main/custom_train.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **train_small** — wall-clock budget `1:30:00`, compute share `1.0`
- **train_medium** — wall-clock budget `4:00:00`, compute share `1.0`
- **train_large** — wall-clock budget `10:00:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `concat-film` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 195–206:
   192: # Conditioning injection (EDITABLE REGION)
   193: # ============================================================================
   194: 
   195: def prepare_conditioning(time_emb, class_emb):
   196:     # Concat-FiLM: add projected class_emb to time_emb
   197:     return time_emb + class_emb
   198: 
   199: 
   200: class ClassConditioner(nn.Module):
   201:     # No-op: all conditioning is via time_emb
   202:     def __init__(self, channels, cond_dim):
   203:         super().__init__()
   204: 
   205:     def forward(self, h, class_emb):
   206:         return h
   207: 
   208: 
   209: # ============================================================================
```

### `cross-attn` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 195–207:
   192: # Conditioning injection (EDITABLE REGION)
   193: # ============================================================================
   194: 
   195: def prepare_conditioning(time_emb, class_emb):
   196:     # Cross-attn: time_emb unchanged, conditioning via ClassConditioner
   197:     return time_emb
   198: 
   199: 
   200: class ClassConditioner(nn.Module):
   201:     # Cross-attention: class embedding as key/value
   202:     def __init__(self, channels, cond_dim):
   203:         super().__init__()
   204:         self.cross_attn = CrossAttentionLayer(channels, cond_dim, num_heads=4)
   205: 
   206:     def forward(self, h, class_emb):
   207:         return self.cross_attn(h, class_emb)
   208: 
   209: 
   210: # ============================================================================
```

### `adanorm` baseline — editable region  [READ-ONLY — reference implementation]

In `diffusers-main/custom_train.py`:

```python
Lines 195–207:
   192: # Conditioning injection (EDITABLE REGION)
   193: # ============================================================================
   194: 
   195: def prepare_conditioning(time_emb, class_emb):
   196:     # AdaNorm: time_emb unchanged, conditioning via ClassConditioner
   197:     return time_emb
   198: 
   199: 
   200: class ClassConditioner(nn.Module):
   201:     # Adaptive LayerNorm-Zero: class embedding modulates features
   202:     def __init__(self, channels, cond_dim):
   203:         super().__init__()
   204:         self.adaln = AdaLNBlock(channels, cond_dim)
   205: 
   206:     def forward(self, h, class_emb):
   207:         return self.adaln(h, class_emb)
   208: 
   209: 
   210: # ============================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
