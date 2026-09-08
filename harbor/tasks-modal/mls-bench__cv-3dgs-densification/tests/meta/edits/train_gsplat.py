"""Minimal 3DGS training script using gsplat with custom strategy.

Stripped-down version of gsplat's simple_trainer.py, removing viewer/tyro
dependencies. Only supports COLMAP datasets with DefaultStrategy-compatible
strategies.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

# Ensure we import the pip-installed gsplat, not the workspace source tree
_workspace_gsplat = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _workspace_gsplat]

# Make cuBLAS deterministic on Ampere (before torch imports internal state)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

# Match 2080Ti (Turing, no TF32) numerical behavior on Ampere+ hardware.
# Newer PyTorch already defaults matmul TF32 to False, but cudnn TF32 is
# still True by default — explicitly disable both.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("highest")

from gsplat import rasterization

# Import custom strategy from workspace
sys.path.insert(0, _workspace_gsplat)
from custom_strategy import CustomStrategy

# ── Dataset (COLMAP) ─────────────────────────────────────────────────

sys.path.insert(0, os.path.join(_workspace_gsplat, "examples"))
from datasets.colmap import Parser, Dataset


# ── Helpers ──────────────────────────────────────────────────────────

def _fused_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """Lightweight differentiable SSIM (memory-efficient, no torchmetrics)."""
    channels = img1.shape[1]
    # 1D Gaussian kernel
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    # Separable 2D window via two 1D convolutions
    w_h = g.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    w_v = g.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    pad = window_size // 2

    def conv(x):
        x = F.conv2d(x, w_h, padding=(0, pad), groups=channels)
        return F.conv2d(x, w_v, padding=(pad, 0), groups=channels)

    mu1, mu2 = conv(img1), conv(img2)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sig1_sq = conv(img1 * img1) - mu1_sq
    sig2_sq = conv(img2 * img2) - mu2_sq
    sig12 = conv(img1 * img2) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))
    return ssim_map.mean()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _knn_dists(points, K=4):
    """Compute K-nearest-neighbor distances using PyTorch (no sklearn needed).

    Chunked over query rows so peak memory is O(chunk x N) instead of the
    O(N x N) of a single torch.cdist, which materializes a [N, N] float matrix
    (~170 GB for N=2e5 SfM points -> OOM-kills the process at init before
    training even starts). Results are identical to the unchunked version."""
    # points: [N, 3]
    N = points.shape[0]
    chunk = 2048
    out = torch.empty(N, K, dtype=points.dtype, device=points.device)
    for i in range(0, N, chunk):
        block = points[i:i + chunk]  # [b, 3]
        d = torch.cdist(block.unsqueeze(0), points.unsqueeze(0)).squeeze(0)  # [b, N]
        d, _ = d.topk(K + 1, dim=-1, largest=False)  # include self (dist=0)
        out[i:i + chunk] = d[:, 1:]  # exclude self
    return out  # [N, K]


def init_splats(parser, device, init_opa=0.1, init_scale=1.0, sh_degree=3):
    """Initialize Gaussian parameters from SfM points."""
    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    N = points.shape[0]

    # Compute initial scales from nearest-neighbor distances
    dists = _knn_dists(points, 4)
    scales = torch.log(dists.mean(dim=-1, keepdim=True).repeat(1, 3) * init_scale)

    quats = torch.zeros(N, 4)
    quats[:, 0] = 1.0  # identity rotation

    opacities = torch.logit(torch.full((N,), init_opa))

    # SH coefficients
    dim_sh = (sh_degree + 1) ** 2
    sh0 = (rgbs - 0.5) / 0.2821  # RGB to DC component
    sh0 = sh0.unsqueeze(1)  # [N, 1, 3]
    shN = torch.zeros(N, dim_sh - 1, 3)

    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }).to(device)
    return splats


def create_optimizers(splats, lr_means=1.6e-4, lr_scales=5e-3, lr_quats=1e-3,
                      lr_opacities=5e-2, lr_sh0=2.5e-3, lr_shN=2.5e-3 / 20):
    """Create per-parameter optimizers."""
    optimizers = {}
    lrs = {
        "means": lr_means, "scales": lr_scales, "quats": lr_quats,
        "opacities": lr_opacities, "sh0": lr_sh0, "shN": lr_shN,
    }
    for name, param in splats.items():
        if param.requires_grad:
            optimizers[name] = torch.optim.Adam([param], lr=lrs.get(name, 1e-3))
    return optimizers


def create_schedulers(optimizers, max_steps, lr_final_factor=0.01):
    """Exponential decay scheduler for means."""
    schedulers = []
    if "means" in optimizers:
        gamma = lr_final_factor ** (1.0 / max_steps)
        schedulers.append(
            torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=gamma)
        )
    return schedulers


# ── Rendering ────────────────────────────────────────────────────────

def render_view(splats, camtoworld, K, width, height, sh_degree,
                near_plane=0.01, far_plane=1e10, render_mode="RGB"):
    """Render a single view using gsplat rasterization."""
    means = splats["means"]
    quats = F.normalize(splats["quats"], dim=-1)
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworld),
        Ks=K,
        width=width,
        height=height,
        sh_degree=sh_degree,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode="RGB",
        packed=False,
        absgrad=True,
    )
    colors_out = renders[..., :3]
    return colors_out, alphas, info


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(splats, valset, device, sh_degree, result_dir, step):
    """Evaluate on validation set, return PSNR/SSIM/LPIPS."""
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex",
                                                      normalize=True).to(device)

    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)
    metrics = defaultdict(list)

    for i, data in enumerate(valloader):
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]

        colors, _, _ = render_view(splats, camtoworld, K, width, height, sh_degree)
        colors = torch.clamp(colors, 0.0, 1.0)

        pixels_p = pixels.permute(0, 3, 1, 2)
        colors_p = colors.permute(0, 3, 1, 2)
        metrics["psnr"].append(psnr_fn(colors_p, pixels_p))
        metrics["ssim"].append(ssim_fn(colors_p, pixels_p))
        metrics["lpips"].append(lpips_fn(colors_p, pixels_p))

        # Save first 5 comparison images
        if i < 5:
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            os.makedirs(os.path.join(result_dir, "samples"), exist_ok=True)
            imageio.imwrite(
                os.path.join(result_dir, "samples", f"cmp_{step}_{i:02d}.png"),
                canvas,
            )

    stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
    stats["num_gs"] = len(splats["means"])
    return stats


# ── Training ─────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Load dataset
    parser_obj = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,
        test_every=8,
    )
    trainset = Dataset(parser_obj, split="train")
    valset = Dataset(parser_obj, split="val")

    scene_scale = parser_obj.scene_scale * 1.1
    print(f"Loaded {len(trainset)} train, {len(valset)} val images. "
          f"Scene scale: {scene_scale:.2f}", flush=True)

    # Initialize Gaussians
    splats = init_splats(parser_obj, device, sh_degree=args.sh_degree)
    print(f"Initialized {len(splats['means'])} Gaussians", flush=True)

    # Optimizers and schedulers
    optimizers = create_optimizers(splats)
    schedulers = create_schedulers(optimizers, args.max_steps)

    # Strategy
    strategy = CustomStrategy()
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    # Training loop — seeded generator + worker_init_fn so worker numpy RNGs
    # are deterministic across restarts (each worker seeds its own numpy RNG).
    _dl_generator = torch.Generator().manual_seed(args.seed)
    def _worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
        generator=_dl_generator, worker_init_fn=_worker_init_fn,
    )
    data_iter = iter(trainloader)
    best_psnr = 0.0

    os.makedirs(args.result_dir, exist_ok=True)

    pbar = tqdm(range(1, args.max_steps + 1), desc="Training", disable=True)
    for step in pbar:
        # Get batch
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(trainloader)
            data = next(data_iter)

        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]

        # SH degree schedule
        sh_degree_to_use = min(step // args.sh_degree_interval, args.sh_degree)

        # Render
        colors, alphas, info = render_view(
            splats, camtoworld, K, width, height, sh_degree_to_use,
        )

        # Pre-backward hook
        strategy.step_pre_backward(
            params=splats, optimizers=optimizers,
            state=strategy_state, step=step, info=info,
        )

        # A800 numerical guard. Rasterization can emit NaN/Inf for out-of-view
        # Gaussians on sparse scenes (bicycle 54k init, bonsai); torch.clamp by
        # itself propagates NaN through — use nan_to_num first, then clamp.
        colors = torch.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # Loss: 0.8 * L1 + 0.2 * SSIM
        l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - _fused_ssim(
            colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2),
        )
        loss = 0.8 * l1loss + 0.2 * ssimloss

        loss.backward()

        pbar.set_description(
            f"loss={loss.item():.4f} n_gs={len(splats['means'])}"
        )

        # Optimizer step
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        # Post-backward hook (densification)
        strategy.step_post_backward(
            params=splats, optimizers=optimizers,
            state=strategy_state, step=step, info=info,
        )

        # Logging
        if step % 5000 == 0:
            print(f"TRAIN_METRICS: step={step}, loss={loss.item():.4f}, "
                  f"l1={l1loss.item():.4f}, ssim_loss={ssimloss.item():.4f}, "
                  f"num_gs={len(splats['means'])}", flush=True)

        # Evaluation
        if step in args.eval_steps:
            print(f"Starting eval at step {step}...", flush=True)
            try:
                stats = evaluate(splats, valset, device, args.sh_degree,
                               args.result_dir, step)
            except Exception as e:
                print(f"Eval FAILED at step {step}: {e}", flush=True)
                continue
            if stats["psnr"] > best_psnr:
                best_psnr = stats["psnr"]
            if step < args.max_steps:
                print(f"EVAL step={step}: PSNR={stats['psnr']:.3f}, "
                      f"SSIM={stats['ssim']:.4f}, LPIPS={stats['lpips']:.3f}, "
                      f"num_GS={stats['num_gs']}", flush=True)
            else:
                print(f"TEST_METRICS: psnr={stats['psnr']:.3f}, ssim={stats['ssim']:.4f}, "
                      f"lpips={stats['lpips']:.3f}, num_gs={stats['num_gs']}, "
                      f"best_psnr={best_psnr:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--result_dir", type=str, default="/tmp/output")
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--eval_steps", type=int, nargs="+", default=[7000, 30000])
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--sh_degree_interval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)
