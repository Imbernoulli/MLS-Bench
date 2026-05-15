"""Custom Flow Matching Training Script — Perceptual Loss Variant
Small-scale flow matching training on CIFAR-10 with a lightweight DiT.
The training objective (MeanFlow) is pre-implemented; your task is to
design an improved loss function, optionally using perceptual losses.
"""

import math
import os
import time

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from torchvision import datasets, transforms
from torchvision.utils import save_image
from perceptual_utils import compute_gradient_loss, compute_multiscale_loss

# ============================================================================
# Model: Lightweight DiT for CIFAR-10 (32x32)
# ============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.freq_embed_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class SmallDiT(nn.Module):
    """Lightweight DiT for CIFAR-10 (32x32 images, patch_size=4 -> 64 tokens)."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, hidden_size=256, depth=6, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.hidden_size = hidden_size

        self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Two timestep embeddings: t (current) and t_next (target)
        self.t_next_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, sigma, sigma_next=None, **kwargs):
        """
        x: [b, c, h, w]
        sigma: [b, 1, 1, 1] or [b] - current timestep t
        sigma_next: [b, 1, 1, 1] or [b] - target timestep t_next (for mean velocity)
        Returns: mean velocity [b, c, h, w]
        """
        B = x.shape[0]
        t = sigma.view(B) if sigma.numel() == B else sigma.view(B, -1).mean(-1)
        t_next = sigma_next.view(B) if sigma_next is not None and sigma_next.numel() == B \
            else (sigma_next.view(B, -1).mean(-1) if sigma_next is not None else torch.zeros_like(t))

        # Patch embed
        h = self.patch_embed(x)  # [b, hidden, h/p, w/p]
        h = h.flatten(2).transpose(1, 2)  # [b, num_patches, hidden]
        h = h + self.pos_embed

        # Condition: t + t_next embeddings
        c = self.t_embedder(t) + self.t_next_embedder(t_next)  # [b, hidden]

        for block in self.blocks:
            h = block(h, c)

        h = self.norm_out(h)
        h = self.proj_out(h)  # [b, num_patches, p*p*c]

        # Unpatchify
        p = self.patch_size
        hw = int(self.num_patches ** 0.5)
        h = h.reshape(B, hw, hw, p, p, -1)
        h = h.permute(0, 5, 1, 3, 2, 4).contiguous()
        h = h.reshape(B, -1, hw * p, hw * p)
        return h  # [b, c, h, w]


# ============================================================================
# Sampling utilities
# ============================================================================

def sample_logit_norm(batch_size, device, loc=0.0, scale=1.0, eps=1e-5):
    """Sample t ~ logit-normal distribution, clipped to [eps, 1-eps]."""
    u = torch.randn(batch_size, device=device) * scale + loc
    t = torch.sigmoid(u)
    return t.clamp(eps, 1 - eps)


# ============================================================================
# Training objective (MeanFlow — pre-implemented)
# ============================================================================

def sample_traj_params(batch_size, cur_step, max_steps, device):
    """MeanFlow: alpha=0, ratio_fm=0.75. Only JVP-based continuous training."""
    ratio_fm = 0.75
    alpha = 0.0

    batch_size_fm = int(batch_size * ratio_fm)
    batch_size_mf = batch_size - batch_size_fm

    t_fm = sample_logit_norm(batch_size_fm, device, loc=-0.4)
    t_next_fm = t_fm.clone()
    dt_fm = torch.zeros_like(t_fm)

    t_1 = sample_logit_norm(batch_size_mf, device, loc=-0.4)
    t_2 = sample_logit_norm(batch_size_mf, device, loc=-0.4)
    t_mf = torch.maximum(t_1, t_2)
    t_next_mf = torch.minimum(t_1, t_2)
    dt_mf = torch.zeros_like(t_mf)  # alpha=0 -> dt=0

    t = torch.cat([t_fm, t_mf]).view(batch_size, 1, 1, 1)
    t_next = torch.cat([t_next_fm, t_next_mf]).view(batch_size, 1, 1, 1)
    dt = torch.cat([dt_fm, dt_mf]).view(batch_size, 1, 1, 1)

    return t, t_next, dt, alpha


def compute_mean_velocity_target(net, x_t, t, t_next, dt, velocity, device):
    """MeanFlow: use JVP for all MeanFlow samples, no discrete path."""
    B = x_t.shape[0]
    t_flat = t.view(B)
    t_next_flat = t_next.view(B)

    mask_fm = torch.isclose(t_flat, t_next_flat)
    mask_c = ~mask_fm

    mean_velocity = velocity.clone()

    if mask_c.any():
        idx = mask_c.nonzero(as_tuple=True)[0]
        x_c = x_t[idx]
        t_c = t_flat[idx]
        t_next_c = t_next_flat[idx]

        def wrap_net(x, t_in):
            t_in_5d = t_in.view(-1, 1, 1, 1)
            t_next_5d = t_next_c.view(-1, 1, 1, 1)
            return net(x, sigma=t_in_5d, sigma_next=t_next_5d)

        _, dudt = jvp(wrap_net, (x_c, t_c), (velocity[idx], torch.ones_like(t_c)))
        u_c = velocity[idx] - (t_c - t_next_c).view(-1, 1, 1, 1) * dudt
        mean_velocity[idx] = u_c

    return mean_velocity


# ============================================================================
# Inference: reverse flow sampling
# ============================================================================

@torch.no_grad()
def sample_images(net, num_samples, num_steps, device, img_size=32, channels=3):
    """Generate images via reverse flow (Euler steps)."""
    net.eval()
    x = torch.randn(num_samples, channels, img_size, img_size, device=device)
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_cur = t_steps[i].expand(num_samples).view(num_samples, 1, 1, 1)
        t_next = t_steps[i + 1].expand(num_samples).view(num_samples, 1, 1, 1)
        v = net(x, sigma=t_cur, sigma_next=t_next)
        x = x - (t_cur - t_next) * v

    net.train()
    return x.clamp(-1, 1)


# ============================================================================
# FID computation (using clean-fid)
# ============================================================================

def compute_fid(net, device, num_samples=2048, num_steps=10, img_size=32, batch_size=128):
    """Compute FID against CIFAR-10 train set using clean-fid."""
    import tempfile, shutil, numpy as np
    from cleanfid import fid as cleanfid
    import cleanfid.features as _feat

    # Use bind-mounted persistent cache dir for inception weights and stats
    cache_dir = "/data/cleanfid"
    os.makedirs(cache_dir, exist_ok=True)

    # Check if files exist, only download if missing
    inception_path = os.path.join(cache_dir, "inception-2015-12-05.pt")
    stats_path = os.path.join(cache_dir, "cifar10_clean_train_32.npz")

    missing = [p for p in (inception_path, stats_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing clean-fid cache files prepared by `mlsbench data alphaflow-main`: "
            + ", ".join(missing)
        )

    # Patch cleanfid to load inception from our cache dir instead of /tmp
    _orig_build = _feat.build_feature_extractor
    def _patched_build(mode, device=device, use_dataparallel=True):
        from cleanfid.inception_torchscript import InceptionV3W
        model = InceptionV3W(cache_dir, download=False, resize_inside=(mode=="legacy_tensorflow")).to(device)
        model.eval()
        if use_dataparallel:
            model = torch.nn.DataParallel(model)
        return lambda x: model(x)
    _feat.build_feature_extractor = _patched_build

    # Patch stats lookup to use our cache dir
    _orig_ref = _feat.get_reference_statistics
    def _patched_ref(name, res, mode="clean", model_name="inception_v3", seed=0, split="train", metric="FID"):
        fpath = os.path.join(cache_dir, f"{name}_{mode}_{split}_{res}.npz".lower())
        stats = np.load(fpath)
        return stats["mu"], stats["sigma"]
    _feat.get_reference_statistics = _patched_ref

    net.eval()
    gen_dir = tempfile.mkdtemp()

    generated = 0
    idx = 0
    while generated < num_samples:
        bs = min(batch_size, num_samples - generated)
        imgs = sample_images(net, bs, num_steps, device, img_size)
        imgs_uint8 = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu()
        for j in range(bs):
            from PIL import Image
            img_np = imgs_uint8[j].permute(1, 2, 0).numpy()
            Image.fromarray(img_np).save(os.path.join(gen_dir, f'{idx:05d}.png'))
            idx += 1
        generated += bs

    score = cleanfid.compute_fid(
        gen_dir,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        device=device,
        batch_size=batch_size,
        verbose=False,
    )
    shutil.rmtree(gen_dir)

    # Restore original functions
    _feat.build_feature_extractor = _orig_build
    _feat.get_reference_statistics = _orig_ref

    net.train()
    return score


# ============================================================================
# Training Script
# ============================================================================

if __name__ == '__main__':
    # ── Config ──────────────────────────────────────────────────────────────
    seed = int(os.environ.get('SEED', 42))
    data_dir = os.environ.get('DATA_DIR', '/data/cifar10')
    output_dir = os.environ.get('OUTPUT_DIR', '/tmp/output')
    max_steps = int(os.environ.get('MAX_STEPS', 1000))
    eval_interval = int(os.environ.get('EVAL_INTERVAL', 1000))
    batch_size = int(os.environ.get('BATCH_SIZE', 128))
    lr = float(os.environ.get('LR', 2e-4))
    num_fid_samples = int(os.environ.get('NUM_FID_SAMPLES', 2048))
    num_eval_steps = int(os.environ.get('NUM_EVAL_STEPS', 10))

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    data_iter = iter(loader)

    # ── Model ────────────────────────────────────────────────────────────────
    hidden_size = int(os.environ.get('MODEL_HIDDEN_SIZE', 512))
    depth       = int(os.environ.get('MODEL_DEPTH', 8))
    num_heads   = int(os.environ.get('MODEL_NUM_HEADS', 8))
    net = SmallDiT(img_size=32, patch_size=4, in_channels=3,
                   hidden_size=hidden_size, depth=depth, num_heads=num_heads).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()

    num_params = sum(p.numel() for p in net.parameters())
    print(f"Model parameters: {num_params/1e6:.1f}M")

    # ── LPIPS perceptual loss model ──────────────────────────────────────────
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # ── Training loop ────────────────────────────────────────────────────────
    best_fid = float('inf')
    t0 = time.time()

    for step in range(1, max_steps + 1):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, _ = next(data_iter)

        x = x.to(device)
        B = x.shape[0]

        # Sample trajectory params
        t, t_next, dt, alpha = sample_traj_params(B, step, max_steps, device)

        # Add noise: x_t = (1-t)*x + t*noise
        noise = torch.randn_like(x)
        x_t = (1 - t) * x + t * noise

        # Instantaneous velocity target: v = noise - x
        velocity = noise - x

        # Compute mean velocity target
        with torch.amp.autocast(device_type='cuda'):
            mean_vel_target = compute_mean_velocity_target(
                net, x_t, t, t_next, dt, velocity, device
            )

            # Predict mean velocity
            pred_mean_vel = net(x_t, sigma=t, sigma_next=t_next)

            # TODO: Implement your loss function here.
            #
            # You have access to:
            #   pred_mean_vel : [B, C, H, W] — model's predicted mean velocity
            #   mean_vel_target: [B, C, H, W] — ground-truth mean velocity target
            #   x              : [B, C, H, W] — clean image (normalized to [-1, 1])
            #   x_t            : [B, C, H, W] — noisy image at timestep t
            #   t              : [B, 1, 1, 1] — current timestep
            #   t_next         : [B, 1, 1, 1] — target timestep
            #   dt             : [B, 1, 1, 1] — step size
            #   alpha          : float         — discrete path weight
            #   lpips_fn       : LPIPS model (VGG backbone), expects input in [-1, 1]
            #   device         : torch.device
            #
            # Your loss must assign a scalar `loss` (the value that will be
            # back-propagated). You may also use perceptual (LPIPS) losses,
            # frequency-domain losses, or any combination thereof.
            raise NotImplementedError("Implement the loss function")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 200 == 0:
            dt_elapsed = time.time() - t0
            print(f"step {step}/{max_steps} | loss {loss.item():.4f} | {dt_elapsed:.1f}s", flush=True)
            t0 = time.time()

        if step % eval_interval == 0 or step == max_steps:
            print(f"Computing FID at step {step}...", flush=True)
            fid = compute_fid(net, device, num_samples=num_fid_samples, num_steps=num_eval_steps)
            print(f"TRAIN_METRICS: step={step}, loss={loss.item():.4f}, fid={fid:.2f}", flush=True)
            if fid < best_fid:
                best_fid = fid

    # ── Save checkpoint ──────────────────────────────────────────────────────
    print(f"Saving checkpoint to {output_dir}/checkpoint.pth", flush=True)
    torch.save({
        'step': max_steps,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_fid': best_fid,
    }, os.path.join(output_dir, 'checkpoint.pth'))

    # ── Final eval ───────────────────────────────────────────────────────────
    fid = compute_fid(net, device, num_samples=num_fid_samples, num_steps=num_eval_steps)
    print(f"TEST_METRICS: fid={fid:.2f}, best_fid={best_fid:.2f}", flush=True)
