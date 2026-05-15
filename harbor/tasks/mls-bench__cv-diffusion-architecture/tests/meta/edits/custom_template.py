"""Unconditional DDPM Training on CIFAR-10 with configurable UNet architecture.

Uses epsilon prediction (fixed). Only the model architecture is editable.
"""

import copy
import math
import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

# Use diffusers from the external package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel


# ============================================================================
# Model Architecture (EDITABLE REGION)
# ============================================================================

def build_model(device):
    """Build a UNet model for unconditional CIFAR-10 diffusion.

    TODO: Design your UNet architecture here.

    The model must satisfy:
    - Input:  (x, timestep) where x is [B, 3, 32, 32], timestep is [B]
    - Output: object with .sample attribute of shape [B, 3, 32, 32]
    - UNet2DModel from diffusers satisfies this interface

    The channel widths are provided via env var BLOCK_OUT_CHANNELS (e.g.
    "128,256,256,256") so the same architecture scales across evaluation
    tiers.  LAYERS_PER_BLOCK (default 2) is also available.

    Available from diffusers UNet2DModel:
        down_block_types / up_block_types — choose from:
            "DownBlock2D"     / "UpBlock2D"      (pure convolution)
            "AttnDownBlock2D" / "AttnUpBlock2D"  (conv + self-attention)
        Other knobs: layers_per_block, norm_num_groups, attention_head_dim,
                     resnet_time_scale_shift, act_fn, etc.

    You may also build a fully custom nn.Module as long as it exposes
    the same (x, timestep) → .sample interface.

    Returns:
        nn.Module on the given device
    """
    raise NotImplementedError("Implement build_model")


# ============================================================================
# Fixed: epsilon prediction
# ============================================================================

def get_schedule_tensors(noise_scheduler, device):
    acp = noise_scheduler.alphas_cumprod.to(device)
    return {
        "alphas_cumprod": acp,
        "sqrt_alpha": acp.sqrt(),
        "sqrt_one_minus_alpha": (1.0 - acp).sqrt(),
    }


def compute_training_target(x_0, noise, timesteps, schedule):
    """Epsilon prediction — fixed, not editable."""
    return noise


def predict_x0(model_output, x_t, timesteps, schedule):
    """Recover x_0 from epsilon prediction — fixed, not editable."""
    sa = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    soma = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return (x_t - soma * model_output) / sa


# ============================================================================
# Sampling — DDIM with epsilon prediction
# ============================================================================

@torch.no_grad()
def sample_images(model, schedule, num_samples, device, num_steps=1000,
                  sample_steps=50, img_size=32, channels=3):
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

    for t in scheduler.timesteps:
        t_batch = t.expand(num_samples).to(device)
        with torch.amp.autocast(device_type='cuda'):
            noise_pred = model(x, t_batch).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    model.train()
    return x.clamp(-1, 1)


# ============================================================================
# FID computation (using clean-fid)
# ============================================================================

def compute_fid(model, schedule, device, num_samples=2048, num_steps=1000,
                sample_steps=50, img_size=32, batch_size=128,
                rank=0, world_size=1):
    import shutil
    from cleanfid import fid as cleanfid
    import cleanfid.features as _feat

    gen_dir = os.path.join(os.environ.get('OUTPUT_DIR', '/tmp/output'), '_fid_tmp')
    if rank == 0:
        if os.path.exists(gen_dir):
            shutil.rmtree(gen_dir)
        os.makedirs(gen_dir)
    if world_size > 1:
        dist.barrier()

    per_rank = (num_samples + world_size - 1) // world_size
    my_start = rank * per_rank
    my_count = min(per_rank, num_samples - my_start)

    model.eval()
    generated = 0
    idx = my_start
    while generated < my_count:
        bs = min(batch_size, my_count - generated)
        imgs = sample_images(model, schedule, bs, device, num_steps,
                             sample_steps, img_size)
        imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
        for j in range(bs):
            img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
            Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
            idx += 1
        generated += bs

    if world_size > 1:
        dist.barrier()

    score = None
    if rank == 0:
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

        score = cleanfid.compute_fid(
            gen_dir, dataset_name="cifar10", dataset_res=32,
            dataset_split="train", device=device, batch_size=batch_size, verbose=False,
        )

        _feat.build_feature_extractor = _orig_build
        _feat.get_reference_statistics = _orig_ref
        _fid_mod.get_reference_statistics = _orig_ref

        shutil.rmtree(gen_dir)

    if world_size > 1:
        dist.barrier()

    model.train()
    return score


def save_sample_images(model, schedule, device, output_dir, step, num_images=16,
                       num_steps=1000, sample_steps=50, tag=""):
    imgs = sample_images(model, schedule, num_images, device, num_steps, sample_steps)
    imgs = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()

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
    diffusion_steps = int(os.environ.get('DIFFUSION_STEPS', 1000))
    sample_steps = int(os.environ.get('SAMPLE_STEPS', 50))
    ema_rate = float(os.environ.get('EMA_RATE', 0.9999))

    # ── DDP setup ──────────────────────────────────────────────────────────
    use_ddp = 'RANK' in os.environ
    if use_ddp:
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
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

    # ── Noise scheduler ────────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_steps,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True,
        variance_type="fixed_large",
    )
    schedule = get_schedule_tensors(noise_scheduler, device)

    # ── Model ───────────────────────────────────────────────────────────────
    net = build_model(device)

    ema_net = copy.deepcopy(net)
    ema_net.requires_grad_(False)

    if use_ddp:
        net = DDP(net, device_ids=[local_rank])
    net_raw = net.module if use_ddp else net

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
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
            x, _ = next(data_iter)
        except StopIteration:
            epoch += 1
            if use_ddp:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            x, _ = next(data_iter)

        x = x.to(device)
        B = x.shape[0]

        t = torch.randint(0, diffusion_steps, (B,), device=device).long()
        noise = torch.randn_like(x)
        x_t = noise_scheduler.add_noise(x, noise, t)

        target = compute_training_target(x, noise, t, schedule)

        with torch.amp.autocast(device_type='cuda'):
            pred = net(x_t, t).sample
            loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

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
                save_sample_images(net_raw, schedule, device, output_dir, step,
                                   num_steps=diffusion_steps, sample_steps=sample_steps,
                                   tag="net")
                save_sample_images(ema_net, schedule, device, output_dir, step,
                                   num_steps=diffusion_steps, sample_steps=sample_steps,
                                   tag="ema")
            eval_model = ema_net if step >= 20000 else net_raw
            fid = compute_fid(eval_model, schedule, device,
                              num_samples=num_fid_samples,
                              num_steps=diffusion_steps,
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

        save_sample_images(net_raw, schedule, device, output_dir, max_steps,
                           num_steps=diffusion_steps, sample_steps=sample_steps,
                           tag="net_final")
        save_sample_images(ema_net, schedule, device, output_dir, max_steps,
                           num_steps=diffusion_steps, sample_steps=sample_steps,
                           tag="ema_final")

    eval_model = ema_net if max_steps >= 20000 else net_raw
    fid = compute_fid(eval_model, schedule, device,
                      num_samples=num_fid_samples,
                      num_steps=diffusion_steps,
                      sample_steps=sample_steps,
                      rank=rank, world_size=world_size)
    if is_main:
        print(f"TEST_METRICS: fid={fid:.2f}, best_fid={best_fid:.2f}", flush=True)

    if use_ddp:
        dist.destroy_process_group()
