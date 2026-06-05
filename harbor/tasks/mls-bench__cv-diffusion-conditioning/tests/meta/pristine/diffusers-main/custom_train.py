"""Class-Conditional DDPM Training on CIFAR-10.

Uses diffusers UNet2DModel backbone (same architecture as google/ddpm-cifar10-32)
with configurable class-conditioning injection. Only the conditioning method
(prepare_conditioning + ClassConditioner) is editable.
"""

import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

# Use diffusers from the external package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel


# ============================================================================
# Conditioning utilities (available for baselines to use)
# ============================================================================

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class CrossAttentionLayer(nn.Module):
    """Cross-attention: features attend to class embedding as key/value."""
    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(context_dim, channels)
        self.v_proj = nn.Linear(context_dim, channels)
        self.out_proj = zero_module(nn.Linear(channels, channels))

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1).transpose(1, 2)
        ctx = context.unsqueeze(1)
        q = self.q_proj(h).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)
        return x + out.transpose(1, 2).view(B, C, H, W)


class AdaLNBlock(nn.Module):
    """Adaptive LayerNorm-Zero: class embedding generates scale/shift/gate."""
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * channels))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond):
        scale, shift, gate = self.proj(cond).unsqueeze(-1).unsqueeze(-1).chunk(3, dim=1)
        h = self.norm(x.float()).type(x.dtype)
        h = h * (1 + scale) + shift
        return x + gate * (h - x)


# ============================================================================
# Conditional UNet: diffusers UNet2DModel backbone + class conditioning hooks
# ============================================================================

class ConditionalUNet(nn.Module):
    """Wraps UNet2DModel and adds configurable class-conditioning.

    The backbone is identical to google/ddpm-cifar10-32. Class conditioning
    is injected via two editable components:
      - prepare_conditioning(time_emb, class_emb): how class info enters time path
      - ClassConditioner(channels, cond_dim): extra conditioning after each block
    """

    # Architecture matching google/ddpm-cifar10-32
    UNET_CONFIG = dict(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        layers_per_block=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        act_fn="silu",
        time_embedding_type="positional",
        flip_sin_to_cos=False,
        freq_shift=1,
        downsample_padding=0,
    )

    def __init__(self, num_classes=10):
        super().__init__()
        config = dict(self.UNET_CONFIG)
        if os.environ.get('BLOCK_OUT_CHANNELS'):
            config['block_out_channels'] = tuple(
                int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
        if os.environ.get('LAYERS_PER_BLOCK'):
            config['layers_per_block'] = int(os.environ['LAYERS_PER_BLOCK'])
        self.unet = UNet2DModel(**config)
        self.time_embed_dim = self.unet.config.block_out_channels[0] * 4

        # Class embedding: same dim as time_embed_dim for prepare_conditioning
        self.class_embed = nn.Embedding(num_classes, self.time_embed_dim)

        # ClassConditioner per down block, mid block, up block
        down_channels = [self.unet.config.block_out_channels[i]
                         for i in range(len(self.unet.down_blocks))]
        mid_ch = self.unet.config.block_out_channels[-1]

        reversed_ch = list(reversed(self.unet.config.block_out_channels))
        up_channels = [reversed_ch[i] for i in range(len(self.unet.up_blocks))]

        self.down_cond = nn.ModuleList(
            [ClassConditioner(ch, self.time_embed_dim) for ch in down_channels])
        self.mid_cond = ClassConditioner(mid_ch, self.time_embed_dim)
        self.up_cond = nn.ModuleList(
            [ClassConditioner(ch, self.time_embed_dim) for ch in up_channels])

    def forward(self, sample, timestep, class_labels):
        # --- Time embedding (reuse UNet internals) ---
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.unet.dtype)
        emb = self.unet.time_embedding(t_emb)

        # --- Class embedding ---
        class_emb = self.class_embed(class_labels)  # [B, time_embed_dim]

        # --- Editable: how class_emb interacts with time_emb ---
        emb = prepare_conditioning(emb, class_emb)

        # --- UNet forward with ClassConditioner hooks ---
        sample = self.unet.conv_in(sample)

        # Down
        down_block_res_samples = (sample,)
        for i, block in enumerate(self.unet.down_blocks):
            sample, res_samples = block(hidden_states=sample, temb=emb)
            sample = self.down_cond[i](sample, class_emb)
            # Fix res_samples: last one should be the conditioned sample
            res_samples = res_samples[:-1] + (sample,)
            down_block_res_samples += res_samples

        # Mid
        if self.unet.mid_block is not None:
            sample = self.unet.mid_block(sample, emb)
        sample = self.mid_cond(sample, class_emb)

        # Up
        for i, block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(block.resnets)]
            sample = block(sample, res_samples, emb)
            sample = self.up_cond[i](sample, class_emb)

        # Out
        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        return sample


# ============================================================================
# Conditioning injection (EDITABLE REGION)
# ============================================================================

def prepare_conditioning(time_emb, class_emb):
    """Prepare the combined embedding used in ResBlocks.

    TODO: Implement your conditioning preparation here.

    Args:
        time_emb:  [B, time_embed_dim] timestep embedding
        class_emb: [B, time_embed_dim] class embedding

    Returns: [B, time_embed_dim] embedding used in ResBlocks
    """
    raise NotImplementedError("Implement prepare_conditioning")


class ClassConditioner(nn.Module):
    """Conditioning module applied after each UNet block.

    TODO: Implement your conditioning method here.

    Args (forward):
        h:         [B, C, H, W] feature map
        class_emb: [B, time_embed_dim] class embedding

    Available utilities:
        CrossAttentionLayer(channels, context_dim, num_heads)
        AdaLNBlock(channels, cond_dim)
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        raise NotImplementedError("Implement ClassConditioner.__init__")

    def forward(self, h, class_emb):
        raise NotImplementedError("Implement ClassConditioner.forward")


# ============================================================================
# Sampling — uses diffusers DDIMScheduler
# ============================================================================

@torch.no_grad()
def sample_images(model, num_samples, device, num_classes=10, num_steps=1000,
                  sample_steps=50, img_size=32, channels=3):
    """Generate class-conditional images via DDIM sampling (diffusers)."""
    model.eval()
    scheduler = DDIMScheduler(
        num_train_timesteps=num_steps,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(sample_steps)

    x = torch.randn(num_samples, channels, img_size, img_size, device=device)
    class_labels = torch.randint(0, num_classes, (num_samples,), device=device)

    for t in scheduler.timesteps:
        t_batch = t.expand(num_samples).to(device)
        with torch.amp.autocast(device_type='cuda'):
            noise_pred = model(x, t_batch, class_labels)
        x = scheduler.step(noise_pred, t, x).prev_sample

    model.train()
    return x.clamp(-1, 1)


# ============================================================================
# FID computation (using clean-fid)
# ============================================================================

def compute_fid(model, device, num_samples=2048, num_classes=10, num_steps=1000,
                sample_steps=50, img_size=32, batch_size=128,
                rank=0, world_size=1):
    """Compute FID against CIFAR-10 train set using clean-fid.

    All ranks sample in parallel; rank 0 computes FID on the merged results.
    """
    import shutil
    import tempfile

    model.eval()

    # Each rank samples its share
    my_samples = num_samples // world_size
    if rank < (num_samples % world_size):
        my_samples += 1
    start_idx = rank * (num_samples // world_size) + min(rank, num_samples % world_size)

    gen_dir = os.path.join(tempfile.gettempdir(), f"fid_gen_{os.getpid()}")
    os.makedirs(gen_dir, exist_ok=True)

    generated = 0
    idx = start_idx
    while generated < my_samples:
        bs = min(batch_size, my_samples - generated)
        imgs = sample_images(model, bs, device, num_classes, num_steps,
                             sample_steps, img_size)
        imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
        for j in range(bs):
            img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
            Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
            idx += 1
        generated += bs
        if rank == 0:
            print(f"  Sampling: {generated}/{my_samples}", flush=True)

    # Sync all ranks before FID computation
    if world_size > 1:
        dist.barrier()

    # Rank 0 gathers all images and computes FID
    score = 0.0
    if rank == 0:
        # Merge images from all ranks into one dir
        merged_dir = os.path.join(tempfile.gettempdir(), "fid_merged")
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
        os.makedirs(merged_dir)

        # Copy from all per-rank dirs
        for f in sorted(os.listdir(gen_dir)):
            shutil.copy2(os.path.join(gen_dir, f), os.path.join(merged_dir, f))

        if world_size > 1:
            # Other ranks wrote to /tmp on the same node
            import glob
            for other_dir in glob.glob(os.path.join(tempfile.gettempdir(), "fid_gen_*")):
                if other_dir == gen_dir:
                    continue
                for f in os.listdir(other_dir):
                    shutil.copy2(os.path.join(other_dir, f), os.path.join(merged_dir, f))

        from cleanfid import fid as cleanfid
        import cleanfid.features as _feat

        cache_dir = "/data/cleanfid"
        os.makedirs(cache_dir, exist_ok=True)

        inception_path = os.path.join(cache_dir, "inception-2015-12-05.pt")
        stats_path = os.path.join(cache_dir, "cifar10_clean_train_32.npz")
        missing = [p for p in (inception_path, stats_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing clean-fid cache files prepared by `mlsbench data diffusers-main`: "
                + ", ".join(missing)
            )

        _orig_build = _feat.build_feature_extractor
        def _patched_build(mode, device=device, use_dataparallel=True):
            from cleanfid.inception_torchscript import InceptionV3W
            m = InceptionV3W(cache_dir, download=False,
                             resize_inside=(mode == "legacy_tensorflow")).to(device)
            m.eval()
            if use_dataparallel:
                m = torch.nn.DataParallel(m)
            return lambda x: m(x)
        _feat.build_feature_extractor = _patched_build

        _orig_ref = _feat.get_reference_statistics
        def _patched_ref(name, res, mode="clean", model_name="inception_v3",
                         seed=0, split="train", metric="FID"):
            fpath = os.path.join(cache_dir, f"{name}_{mode}_{split}_{res}.npz".lower())
            stats = np.load(fpath)
            return stats["mu"], stats["sigma"]
        _feat.get_reference_statistics = _patched_ref
        import cleanfid.fid as _fid_mod
        _fid_mod.get_reference_statistics = _patched_ref

        print(f"  Computing FID on {len(os.listdir(merged_dir))} images...", flush=True)
        score = cleanfid.compute_fid(
            merged_dir, dataset_name="cifar10", dataset_res=32,
            dataset_split="train", device=device, batch_size=batch_size, verbose=False,
        )

        shutil.rmtree(merged_dir)
        _feat.build_feature_extractor = _orig_build
        _feat.get_reference_statistics = _orig_ref
        _fid_mod.get_reference_statistics = _orig_ref

    # Clean up per-rank dir
    shutil.rmtree(gen_dir, ignore_errors=True)

    if world_size > 1:
        dist.barrier()

    model.train()
    return score


def save_sample_images(model, device, output_dir, step, num_images=16,
                       num_classes=10, num_steps=1000, sample_steps=50, tag=""):
    """Save a grid of sample images for visual inspection."""
    imgs = sample_images(model, num_images, device, num_classes, num_steps, sample_steps)
    imgs = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()

    # Make a grid: 4x4
    nrow = int(math.sqrt(num_images))
    grid_h = nrow * 32
    grid_w = nrow * 32
    grid = Image.new('RGB', (grid_w, grid_h))
    for i in range(num_images):
        img_np = imgs[i].permute(1, 2, 0).numpy()
        img = Image.fromarray(img_np)
        row, col = divmod(i, nrow)
        grid.paste(img, (col * 32, row * 32))

    suffix = f"_{tag}" if tag else ""
    path = os.path.join(output_dir, f'samples_step{step}{suffix}.png')
    grid.save(path)
    print(f"Saved sample images to {path}", flush=True)


# ============================================================================
# Training Script
# ============================================================================

if __name__ == '__main__':
    seed = int(os.environ.get('SEED', 42))
    data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
    output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
    max_steps = int(os.environ.get('MAX_STEPS', 10000))
    eval_interval = int(os.environ.get('EVAL_INTERVAL', 10000))
    batch_size = int(os.environ.get('BATCH_SIZE', 128))
    lr = float(os.environ.get('LR', 2e-4))
    num_fid_samples = int(os.environ.get('NUM_FID_SAMPLES', 2048))
    num_classes = int(os.environ.get('NUM_CLASSES', 10))
    diffusion_steps = int(os.environ.get('DIFFUSION_STEPS', 1000))
    sample_steps = int(os.environ.get('SAMPLE_STEPS', 50))
    ema_rate = float(os.environ.get('EMA_RATE', 0.9999))

    # ── DDP setup ──────────────────────────────────────────────────────────
    use_ddp = 'RANK' in os.environ
    if use_ddp:
        import datetime as _dt
        dist.init_process_group(backend='nccl', timeout=_dt.timedelta(hours=20))
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_main = (rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
        world_size = 1
        is_main = True

    torch.manual_seed(seed + rank)
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=False)
    if use_ddp:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
    data_iter = iter(loader)

    # ── Noise scheduler (diffusers) ────────────────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_steps,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=True,
        variance_type="fixed_large",
    )

    # ── Model ───────────────────────────────────────────────────────────────
    net = ConditionalUNet(num_classes=num_classes).to(device)

    # Parameter budget check (1.05x largest baseline: Cross-Attention)
    # Build a temporary reference model with the cross-attn baseline to compute budget.
    def _budget_prepare_conditioning(time_emb, class_emb):
        return time_emb
    class _BudgetClassConditioner(nn.Module):
        def __init__(self, channels, cond_dim):
            super().__init__()
            self.cross_attn = CrossAttentionLayer(channels, cond_dim, num_heads=4)
        def forward(self, h, class_emb):
            return self.cross_attn(h, class_emb)
    _orig_prepare = globals()['prepare_conditioning']
    _orig_conditioner = globals()['ClassConditioner']
    globals()['prepare_conditioning'] = _budget_prepare_conditioning
    globals()['ClassConditioner'] = _BudgetClassConditioner
    _ref_net = ConditionalUNet(num_classes=num_classes)
    _max_budget = int(sum(p.numel() for p in _ref_net.parameters()) * 1.05)
    del _ref_net
    globals()['prepare_conditioning'] = _orig_prepare
    globals()['ClassConditioner'] = _orig_conditioner
    _total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameter count: {_total_params:,} / {_max_budget:,}", flush=True)

    # EMA model for evaluation (on main process only)
    ema_net = copy.deepcopy(net)
    ema_net.requires_grad_(False)

    if use_ddp:
        net = DDP(net, device_ids=[local_rank])
    net_raw = net.module if use_ddp else net

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-8)
    scaler = torch.amp.GradScaler()

    num_params = sum(p.numel() for p in net_raw.parameters())
    if is_main:
        print(f"Model parameters: {num_params/1e6:.1f}M | GPUs: {world_size}", flush=True)

    # ── Training loop ────────────────────────────────────────────────────────
    best_fid = float('inf')
    t0 = time.time()
    epoch = 0

    for step in range(1, max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            epoch += 1
            if use_ddp:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        B = x.shape[0]

        # Sample random timesteps and add noise (using diffusers scheduler)
        t = torch.randint(0, diffusion_steps, (B,), device=device).long()
        noise = torch.randn_like(x)
        x_t = noise_scheduler.add_noise(x, noise, t)

        # Predict noise
        with torch.amp.autocast(device_type='cuda'):
            pred_noise = net(x_t, t, y)
            loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update EMA (use raw model params)
        with torch.no_grad():
            for p_ema, p in zip(ema_net.parameters(), net_raw.parameters()):
                p_ema.mul_(ema_rate).add_(p, alpha=1 - ema_rate)

        if is_main and step % 200 == 0:
            dt_elapsed = time.time() - t0
            print(f"step {step}/{max_steps} | loss {loss.item():.4f} | {dt_elapsed:.1f}s",
                  flush=True)
            t0 = time.time()

        if step % eval_interval == 0 or step == max_steps:
            if is_main:
                print(f"Eval at step {step}...", flush=True)
                save_sample_images(net_raw, device, output_dir, step,
                                   num_classes=num_classes, num_steps=diffusion_steps,
                                   sample_steps=sample_steps, tag="net")
                save_sample_images(ema_net, device, output_dir, step,
                                   num_classes=num_classes, num_steps=diffusion_steps,
                                   sample_steps=sample_steps, tag="ema")
            eval_model = ema_net if step >= 20000 else net_raw
            fid = compute_fid(eval_model, device, num_samples=num_fid_samples,
                              num_classes=num_classes, num_steps=diffusion_steps,
                              sample_steps=sample_steps,
                              rank=rank, world_size=world_size)
            if is_main:
                print(f"TRAIN_METRICS: step={step}, loss={loss.item():.4f}, fid={fid:.2f}",
                      flush=True)
                if fid < best_fid:
                    best_fid = fid

    # ── Save & final eval ────────────────────────────────────────────────────
    if is_main:
        print(f"Saving checkpoint to {output_dir}/checkpoint.pth", flush=True)
        torch.save({
            'step': max_steps,
            'model_state_dict': net_raw.state_dict(),
            'ema_model_state_dict': ema_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_fid': best_fid,
        }, os.path.join(output_dir, 'checkpoint.pth'))

        save_sample_images(net_raw, device, output_dir, max_steps,
                           num_classes=num_classes, num_steps=diffusion_steps,
                           sample_steps=sample_steps, tag="net_final")
        save_sample_images(ema_net, device, output_dir, max_steps,
                           num_classes=num_classes, num_steps=diffusion_steps,
                           sample_steps=sample_steps, tag="ema_final")

    eval_model = ema_net if max_steps >= 20000 else net_raw
    fid = compute_fid(eval_model, device, num_samples=num_fid_samples,
                      num_classes=num_classes, num_steps=diffusion_steps,
                      sample_steps=sample_steps,
                      rank=rank, world_size=world_size)
    if is_main:
        print(f"TEST_METRICS: fid={fid:.2f}, best_fid={best_fid:.2f}", flush=True)

    if use_ddp:
        dist.destroy_process_group()
