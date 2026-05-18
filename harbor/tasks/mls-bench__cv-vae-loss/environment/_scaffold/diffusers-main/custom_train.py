"""VAE Training on CIFAR-10 with configurable loss function.

Uses AutoencoderKL architecture (fixed). Only the loss function is editable.
"""

import copy
import math
import os
import shutil
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
from diffusers import AutoencoderKL


# ============================================================================
# Loss Function (EDITABLE REGION)
# ============================================================================

class VAELoss(nn.Module):
    """VAE training loss function.

    TODO: Design your loss function for training a KL-regularized VAE.

    The loss function receives:
        recon:     Reconstructed images [B, 3, 32, 32] in [-1, 1]
        target:    Original images [B, 3, 32, 32] in [-1, 1]
        posterior: DiagonalGaussianDistribution from the encoder
                   - posterior.kl()       -> KL divergence [B, ...]
                   - posterior.mean       -> latent mean
                   - posterior.logvar     -> latent log-variance
        step:      Current training step (int)

    Must return: (total_loss, metrics_dict)
        total_loss: scalar tensor (for backpropagation)
        metrics_dict: dict with string keys and float values for logging

    Available imports: torch, torch.nn, torch.nn.functional, numpy, lpips
    The lpips package provides learned perceptual loss:
        loss_fn = lpips.LPIPS(net='vgg').to(device)
        p_loss = loss_fn(recon, target).mean()

    You may also use torch.fft for frequency-domain operations, or any
    other approach you think will improve reconstruction quality.

    Evaluation metrics (for reference, you do NOT compute these):
        - rFID: Reconstruction FID (lower is better)
        - PSNR: Peak signal-to-noise ratio in dB (higher is better)
        - SSIM: Structural similarity (higher is better)
    """

    def __init__(self, device):
        super().__init__()
        raise NotImplementedError("Implement VAELoss.__init__")

    def forward(self, recon, target, posterior, step):
        """Compute total VAE loss.

        Returns:
            loss: scalar tensor (total training loss for backpropagation)
            metrics: dict of {str: float} for logging
        """
        raise NotImplementedError("Implement VAELoss.forward")


# ============================================================================
# Fixed: Model Architecture
# ============================================================================

def build_model(device):
    """Build AutoencoderKL with channels from environment variable.

    Uses 3 blocks (2 downsample stages) for 32x32 input → 8x8 latent (f=4).
    """
    channels = (128, 256, 512)
    if os.environ.get('BLOCK_OUT_CHANNELS'):
        channels = tuple(int(x) for x in os.environ['BLOCK_OUT_CHANNELS'].split(','))
    layers = int(os.environ.get('LAYERS_PER_BLOCK', 2))
    latent_ch = int(os.environ.get('LATENT_CHANNELS', 4))

    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=latent_ch,
        block_out_channels=channels,
        down_block_types=tuple(["DownEncoderBlock2D"] * len(channels)),
        up_block_types=tuple(["UpDecoderBlock2D"] * len(channels)),
        layers_per_block=layers,
        norm_num_groups=32,
        act_fn="silu",
        sample_size=32,
        scaling_factor=0.18215,
    ).to(device)


class VAEWrapper(nn.Module):
    """Wrapper for encode+decode in a single forward pass (needed for DDP)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample()
        recon = self.vae.decode(z).sample
        return recon, posterior


# ============================================================================
# Fixed: Evaluation Metrics
# ============================================================================

def compute_psnr(img1, img2):
    """PSNR between image tensors in [0, 1]. Returns dB value."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(100.0)
    return (10 * torch.log10(1.0 / mse)).clamp(max=100.0)


def _gaussian_window(size, sigma, channels, device):
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.outer(g).unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, size, size).contiguous()


def compute_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """SSIM between image tensors in [0, 1]."""
    channels = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def evaluate_reconstruction(model, test_loader, device, output_dir,
                            rank=0, world_size=1):
    """Evaluate VAE reconstruction: rFID, PSNR, SSIM on test set.

    Saves images to /dev/shm (RAM tmpfs) to avoid slow disk I/O,
    then uses cleanfid's proven compute_fid path.
    """
    from cleanfid import fid as cleanfid

    model.eval()
    psnr_sum, ssim_sum, count = 0.0, 0.0, 0

    # Use RAM-backed tmpfs for image I/O (much faster than disk)
    orig_dir = "/dev/shm/_eval_orig"
    recon_dir = "/dev/shm/_eval_recon"

    if rank == 0:
        for d in [orig_dir, recon_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
    if world_size > 1:
        dist.barrier()

    sample_pairs = []
    idx = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            with torch.amp.autocast(device_type='cuda'):
                recon, _ = model(x)

            x_01 = (x * 0.5 + 0.5).float()
            recon_01 = recon.clamp(-1, 1).float() * 0.5 + 0.5

            psnr_sum += compute_psnr(recon_01, x_01).item() * x.shape[0]
            ssim_sum += compute_ssim(recon_01, x_01).item() * x.shape[0]
            count += x.shape[0]

            if rank == 0:
                # Collect first 10 pairs for visual comparison
                if len(sample_pairs) < 10:
                    n = min(10 - len(sample_pairs), x.shape[0])
                    for j in range(n):
                        sample_pairs.append((
                            (x_01[j] * 255).clamp(0, 255).byte().cpu(),
                            (recon_01[j] * 255).clamp(0, 255).byte().cpu(),
                        ))

                # Save to RAM tmpfs for FID computation
                x_uint8 = (x_01 * 255).clamp(0, 255).byte().cpu()
                r_uint8 = (recon_01 * 255).clamp(0, 255).byte().cpu()
                for j in range(x.shape[0]):
                    Image.fromarray(x_uint8[j].permute(1, 2, 0).numpy()).save(
                        os.path.join(orig_dir, f'{idx:05d}.png'))
                    Image.fromarray(r_uint8[j].permute(1, 2, 0).numpy()).save(
                        os.path.join(recon_dir, f'{idx:05d}.png'))
                    idx += 1

    if world_size > 1:
        dist.barrier()

    avg_psnr = psnr_sum / max(count, 1)
    avg_ssim = ssim_sum / max(count, 1)

    rfid = None
    if rank == 0:
        import cleanfid.features as _feat

        cache_dir = "/data/cleanfid"
        os.makedirs(cache_dir, exist_ok=True)

        # Patch cleanfid to load inception from image cache (no network needed)
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

        rfid = cleanfid.compute_fid(
            orig_dir, recon_dir,
            device=device, batch_size=64, verbose=False,
        )

        _feat.build_feature_extractor = _orig_build

        # Save 10 sample comparisons
        sample_dir = os.path.join(output_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        for i, (orig_t, recon_t) in enumerate(sample_pairs):
            o = Image.fromarray(orig_t.permute(1, 2, 0).numpy())
            r = Image.fromarray(recon_t.permute(1, 2, 0).numpy())
            cmp = Image.new('RGB', (o.width * 2 + 4, o.height), (128, 128, 128))
            cmp.paste(o, (0, 0))
            cmp.paste(r, (o.width + 4, 0))
            cmp.save(os.path.join(sample_dir, f'cmp_{i:02d}.png'))

        shutil.rmtree(orig_dir, ignore_errors=True)
        shutil.rmtree(recon_dir, ignore_errors=True)

    if world_size > 1:
        dist.barrier()

    model.train()
    return rfid, avg_psnr, avg_ssim


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
    lr = float(os.environ.get('LR', 1e-4))
    ema_rate = float(os.environ.get('EMA_RATE', 0.999))

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
    train_dataset = datasets.CIFAR10(data_dir, train=True, transform=transform,
                                     download=False)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    test_dataset = datasets.CIFAR10(data_dir, train=False,
                                    transform=test_transform, download=False)

    if use_ddp:
        sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True)
    data_iter = iter(train_loader)

    # ── Model ───────────────────────────────────────────────────────────────
    vae = build_model(device)
    wrapper = VAEWrapper(vae)
    ema_vae = copy.deepcopy(vae)
    ema_vae.requires_grad_(False)
    ema_wrapper = VAEWrapper(ema_vae)

    if use_ddp:
        wrapper = DDP(wrapper, device_ids=[local_rank])

    # ── Loss & optimizer ────────────────────────────────────────────────────
    criterion = VAELoss(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-6)
    warmup_steps = int(max_steps * 0.05)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0  # constant LR after warmup
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler()

    num_params = sum(p.numel() for p in vae.parameters())
    if is_main:
        print(f"Model parameters: {num_params/1e6:.1f}M | GPUs: {world_size}",
              flush=True)

    # ── Training loop ────────────────────────────────────────────────────────
    best_rfid = float('inf')
    t0 = time.time()
    epoch = 0

    for step in range(1, max_steps + 1):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            epoch += 1
            if use_ddp:
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            x, _ = next(data_iter)

        x = x.to(device)

        with torch.amp.autocast(device_type='cuda'):
            recon, posterior = wrapper(x)
            loss, metrics = criterion(recon, x, posterior, step)

        # GAN support: both VAE and disc update every step (not alternating)
        has_disc = hasattr(criterion, 'disc') and hasattr(criterion, 'disc_opt')
        disc_factor = 0.0
        if has_disc:
            disc_start = getattr(criterion, 'disc_start', 5000)
            disc_factor = 1.0 if step >= disc_start else 0.0

        # 1) Generator (VAE) update — every step
        if has_disc and disc_factor > 0:
            # Add adaptive GAN loss (diffusers-aligned: perceptual grad for weight)
            logits_fake = criterion.disc(recon.float())
            g_loss = -logits_fake.mean()
            last_layer = vae.decoder.conv_out.weight
            p_loss = getattr(criterion, '_perceptual_loss', None)
            ref_loss = p_loss if p_loss is not None else loss
            ref_grads = torch.autograd.grad(ref_loss, last_layer, retain_graph=True)[0].detach()
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0].detach()
            disc_weight = torch.clamp(
                ref_grads.norm(p=2) / g_grads.norm(p=2).clamp(min=1e-8), 0.0, 1e4)
            loss = loss + disc_weight * g_loss
            metrics['g_loss'] = g_loss.item()
            metrics['disc_w'] = disc_weight.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # 2) Discriminator update — every step after disc_start (FP32, no AMP)
        if has_disc and disc_factor > 0:
            criterion.disc_opt.zero_grad()
            x_real = x.float().detach().requires_grad_(True)
            logits_real = criterion.disc(x_real)
            logits_fake_d = criterion.disc(recon.float().detach())
            d_loss = (F.relu(1 + logits_fake_d) + F.relu(1 - logits_real)).mean()
            # Gradient penalty (diffusers-style)
            gp_grads = torch.autograd.grad(
                outputs=logits_real, inputs=x_real,
                grad_outputs=torch.ones_like(logits_real),
                create_graph=True, retain_graph=True, only_inputs=True,
            )[0]
            gp = 10.0 * ((gp_grads.reshape(gp_grads.shape[0], -1).norm(2, dim=1) - 1) ** 2).mean()
            d_loss = d_loss + gp
            d_loss.backward()
            criterion.disc_opt.step()
            metrics['d_loss'] = d_loss.item()

        with torch.no_grad():
            for p_ema, p in zip(ema_vae.parameters(), vae.parameters()):
                p_ema.mul_(ema_rate).add_(p, alpha=1 - ema_rate)

        if is_main and step % 200 == 0:
            dt = time.time() - t0
            m_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"step {step}/{max_steps} | loss={loss.item():.4f} {m_str} "
                  f"| {dt:.1f}s", flush=True)
            t0 = time.time()

        if step % eval_interval == 0 or step == max_steps:
            if is_main:
                print(f"Eval at step {step}...", flush=True)
            eval_model = ema_wrapper if step >= max_steps // 2 else \
                VAEWrapper(vae)
            rfid, psnr, ssim_val = evaluate_reconstruction(
                eval_model, test_loader, device, output_dir, rank, world_size)
            if is_main:
                if rfid < best_rfid:
                    best_rfid = rfid
                if step < max_steps:
                    print(f"TRAIN_METRICS: step={step}, rfid={rfid:.2f}, "
                          f"psnr={psnr:.2f}, ssim={ssim_val:.4f}", flush=True)
                else:
                    print(f"TEST_METRICS: rfid={rfid:.2f}, psnr={psnr:.2f}, "
                          f"ssim={ssim_val:.4f}, best_rfid={best_rfid:.2f}",
                          flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    if is_main:
        print(f"Saving checkpoint to {output_dir}/checkpoint.pth", flush=True)
        torch.save({
            'step': max_steps,
            'model_state_dict': vae.state_dict(),
            'ema_model_state_dict': ema_vae.state_dict(),
            'best_rfid': best_rfid,
        }, os.path.join(output_dir, 'checkpoint.pth'))

    if use_ddp:
        dist.destroy_process_group()
