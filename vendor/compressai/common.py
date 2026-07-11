"""Shared, FIXED evaluation utilities for the CompressAI (compress-*) MLS-Bench tasks.

Un-gameable path: the training patch set, the eval image set (real Kodak images),
the optimizer/step budget, PSNR, and bits-per-pixel (bpp) are all computed HERE and
cannot be touched by the agent surface. The agent only controls a single, narrowly
scoped design choice (entropy-model wiring / quantization surrogate / R-D loss).

The model is small (a bmshj-style factorized/hyperprior transform with
N,M channels chosen so a few-hundred-step train finishes in ~1-2 min on one GPU),
trained from scratch on fixed patches, then scored on real Kodak images.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_surface(sol_path: str, attr: str):
    p = Path(sol_path)
    spec = importlib.util.spec_from_file_location("agent_surface", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solution surface from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    if not hasattr(mod, attr):
        raise AttributeError(f"solution must define `{attr}(...)`")
    surface = getattr(mod, attr)
    if not callable(surface):
        raise TypeError(f"solution attribute `{attr}` must be callable")
    return surface


def _require_finite_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise RuntimeError(f"{label} must be a non-empty tensor")
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{label} contains non-finite values")
    return value


def validate_codec_output(out, x: torch.Tensor) -> None:
    if not isinstance(out, dict):
        raise RuntimeError("codec forward must return a dict")
    x_hat = _require_finite_tensor(out.get("x_hat"), "codec x_hat")
    if x_hat.shape != x.shape:
        raise RuntimeError(
            f"codec x_hat shape {tuple(x_hat.shape)} does not match input {tuple(x.shape)}"
        )
    likelihoods = out.get("likelihoods")
    if not isinstance(likelihoods, dict) or not likelihoods:
        raise RuntimeError("codec likelihoods must be a non-empty dict")
    for name, likelihood in likelihoods.items():
        likelihood = _require_finite_tensor(likelihood, f"likelihood {name!r}")
        if not torch.all(likelihood > 0).item():
            raise RuntimeError(f"likelihood {name!r} must be strictly positive")


def _require_finite_gradients(model) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"gradient for {name!r} contains non-finite values")


# --------------------------------------------------------------------------- #
# Data: fixed training patches + real Kodak eval images                       #
# --------------------------------------------------------------------------- #

def _load_png(path: Path) -> torch.Tensor:
    """Load a PNG as a float32 CHW tensor in [0,1] without torchvision.io codecs."""
    from PIL import Image

    img = Image.open(str(path)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_eval_images(data_dir: Path, max_side: int = 512) -> list[torch.Tensor]:
    """Load the FIXED Kodak eval set as CHW float tensors, cropped to a multiple
    of 64 (the max downsampling factor) and capped in size for a fast eval."""
    man = json.loads((data_dir / "manifest.json").read_text())
    imgs = []
    for rec in man["eval"]:
        t = _load_png(data_dir / rec["path"])
        # center-crop to <= max_side and a multiple of 64
        _, h, w = t.shape
        h = min(h, max_side)
        w = min(w, max_side)
        h -= h % 64
        w -= w % 64
        _, H, W = t.shape
        top = (H - h) // 2
        left = (W - w) // 2
        imgs.append(t[:, top : top + h, left : left + w].contiguous())
    return imgs


def build_train_patches(
    data_dir: Path, n_patches: int, patch: int, seed: int
) -> torch.Tensor:
    """Deterministically crop `n_patches` fixed patches from the training images.

    Returns an (n_patches, 3, patch, patch) float tensor in [0,1]. The crops are
    fixed by `seed` so every run trains on identical data.
    """
    man = json.loads((data_dir / "manifest.json").read_text())
    train_imgs = [_load_png(data_dir / rec["path"]) for rec in man["train"]]
    rng = random.Random(seed)
    out = []
    for _ in range(n_patches):
        img = train_imgs[rng.randrange(len(train_imgs))]
        _, h, w = img.shape
        top = rng.randrange(0, h - patch + 1)
        left = rng.randrange(0, w - patch + 1)
        out.append(img[:, top : top + patch, left : left + patch])
    return torch.stack(out, 0).contiguous()


# --------------------------------------------------------------------------- #
# Metrics (FIXED)                                                             #
# --------------------------------------------------------------------------- #

def psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """PSNR in dB over [0,1] images (higher is better)."""
    _require_finite_tensor(x, "PSNR input")
    _require_finite_tensor(x_hat, "PSNR reconstruction")
    if x.shape != x_hat.shape:
        raise RuntimeError("PSNR tensors must have identical shapes")
    x_hat = x_hat.clamp(0, 1)
    mse = F.mse_loss(x_hat, x).item()
    if not math.isfinite(mse) or mse < 0:
        raise RuntimeError("PSNR MSE is non-finite or negative")
    if mse <= 0:
        return 99.0
    value = float(-10.0 * math.log10(mse))
    if not math.isfinite(value):
        raise RuntimeError("PSNR is non-finite")
    return value


def real_bpp(likelihoods: dict, num_pixels: int) -> float:
    """Estimated bits-per-pixel from the entropy-model likelihoods (FIXED formula:
    the standard differential-entropy estimate used across the learned-compression
    literature). Lower is better."""
    if not isinstance(likelihoods, dict) or not likelihoods:
        raise RuntimeError("likelihoods must be a non-empty dict")
    if not isinstance(num_pixels, int) or num_pixels <= 0:
        raise RuntimeError("num_pixels must be a positive integer")
    total_bits = 0.0
    for name, lk in likelihoods.items():
        lk = _require_finite_tensor(lk, f"likelihood {name!r}")
        if not torch.all(lk > 0).item():
            raise RuntimeError(f"likelihood {name!r} must be strictly positive")
        total_bits += float(torch.log(lk).sum().item()) / (-math.log(2))
    value = total_bits / num_pixels
    if not math.isfinite(value):
        raise RuntimeError("bits-per-pixel is non-finite")
    return value


def evaluate(model, eval_imgs: list[torch.Tensor], device) -> tuple[float, float]:
    """Return (mean_psnr_dB, mean_bpp) over the fixed Kodak eval images.

    Uses the forward (soft) path so it is robust and fast; bpp is the standard
    entropy estimate. This path is FIXED and identical for every submission.
    """
    if not eval_imgs:
        raise RuntimeError("evaluation image set is empty")
    model.eval()
    psnrs, bpps = [], []
    with torch.no_grad():
        for img in eval_imgs:
            x = img.unsqueeze(0).to(device)
            out = model(x)
            validate_codec_output(out, x)
            _, _, H, W = x.shape
            num_pixels = H * W
            psnrs.append(psnr(x, out["x_hat"]))
            bpps.append(real_bpp(out["likelihoods"], num_pixels))
    mean_psnr = float(np.mean(psnrs))
    mean_bpp = float(np.mean(bpps))
    if not math.isfinite(mean_psnr) or not math.isfinite(mean_bpp):
        raise RuntimeError("evaluation produced non-finite aggregate metrics")
    return mean_psnr, mean_bpp


# --------------------------------------------------------------------------- #
# Fixed training loop                                                         #
# --------------------------------------------------------------------------- #

def train_model(
    model,
    patches: torch.Tensor,
    lmbda: float,
    steps: int,
    batch: int,
    lr: float,
    device,
    seed: int,
    distortion_fn=None,
):
    """FIXED rate-distortion training loop.

    loss = lmbda * 255^2 * MSE(x, x_hat) + bpp    (unless distortion_fn overrides,
    used only by the R-D-loss task which supplies its own objective).

    Two optimizers (net + aux) exactly like CompressAI's reference train.py.
    """
    from compressai.optimizers import net_aux_optimizer

    if not torch.is_tensor(patches) or patches.ndim != 4 or patches.size(0) == 0:
        raise RuntimeError("training patches must be a non-empty NCHW tensor")
    _require_finite_tensor(patches, "training patches")
    if not isinstance(steps, int) or steps <= 0:
        raise RuntimeError("steps must be a positive integer")
    if not isinstance(batch, int) or batch <= 0:
        raise RuntimeError("batch must be a positive integer")
    if not math.isfinite(float(lmbda)) or float(lmbda) <= 0:
        raise RuntimeError("lmbda must be finite and positive")
    if not math.isfinite(float(lr)) or float(lr) <= 0:
        raise RuntimeError("learning rate must be finite and positive")

    model.to(device).train()
    conf = {
        "net": {"type": "Adam", "lr": lr},
        "aux": {"type": "Adam", "lr": 1e-3},
    }
    opt = net_aux_optimizer(model, conf)
    net_opt, aux_opt = opt["net"], opt["aux"]

    g = torch.Generator().manual_seed(seed)
    N = patches.size(0)
    patches = patches.to(device)

    for step in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g).to(device)
        x = patches[idx]
        net_opt.zero_grad()
        out = model(x)
        validate_codec_output(out, x)
        _, _, H, W = x.shape
        num_pixels = x.size(0) * H * W
        bpp = sum(
            torch.log(lk).sum() / (-math.log(2) * num_pixels)
            for lk in out["likelihoods"].values()
        )
        if distortion_fn is not None:
            loss = distortion_fn(x, out["x_hat"], bpp, lmbda)
        else:
            mse = F.mse_loss(out["x_hat"], x)
            loss = lmbda * (255.0 ** 2) * mse + bpp
        if not torch.is_tensor(loss) or loss.numel() != 1 or not torch.isfinite(loss).item():
            raise RuntimeError(f"training loss is invalid at step {step}")
        loss.backward()
        _require_finite_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(f"gradient norm is non-finite at step {step}")
        net_opt.step()
        aux_loss = model.aux_loss()
        if not torch.is_tensor(aux_loss) or aux_loss.numel() != 1 or not torch.isfinite(aux_loss).item():
            raise RuntimeError(f"auxiliary loss is invalid at step {step}")
        aux_opt.zero_grad()
        aux_loss.backward()
        _require_finite_gradients(model)
        aux_opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"TRAIN step={step} loss={loss.item():.4f} "
                  f"bpp={bpp.detach().item():.4f}", flush=True)
    return model
