"""INR verification runtime component.












































"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- utils
def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inr-signal-fitting verification")
    return torch.device("cuda")


def load_surface_config(path: str):
    """Parse a JSON-compatible literal configuration without executing agent code."""
    solution_path = Path(path).resolve()
    if not solution_path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {solution_path}")
    try:
        source = solution_path.read_text()
        if len(source.encode()) > 65536:
            raise ValueError("solution configuration exceeds the 64 KiB source limit")
        tree = ast.parse(source, filename=str(solution_path))
        if sum(1 for _node in ast.walk(tree)) > 512:
            raise ValueError("solution configuration exceeds the AST complexity limit")
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError(f"cannot parse solution configuration: {exc}") from exc

    surfaces = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "surface_config"
    ]
    if len(surfaces) != 1 or isinstance(surfaces[0], ast.AsyncFunctionDef):
        raise ValueError("solution must define exactly one synchronous surface_config")
    surface = surfaces[0]
    if (
        surface.decorator_list
        or surface.args.posonlyargs
        or surface.args.args
        or surface.args.kwonlyargs
        or surface.args.vararg is not None
        or surface.args.kwarg is not None
    ):
        raise ValueError("surface_config must be undecorated and accept no arguments")
    if len(surface.body) != 1 or not isinstance(surface.body[0], ast.Return):
        raise ValueError("surface_config body must contain exactly one return statement")

    for node in tree.body:
        if node is surface:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
            if not node.orelse and node.test.value is False and all(
                isinstance(child, (ast.Import, ast.ImportFrom)) for child in node.body
            ):
                continue
        raise ValueError("solution contains executable top-level statements outside the surface")

    try:
        value = ast.literal_eval(surface.body[0].value)
        payload = json.dumps(value, allow_nan=False, separators=(",", ":"))
        return json.loads(payload)
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError(f"surface_config must return a finite JSON literal: {exc}") from exc


def require_rgb_prediction(value, n_rows: int, stage: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{stage} must return a torch.Tensor")
    if value.shape != (n_rows, 3):
        raise ValueError(
            f"{stage} must have shape {(n_rows, 3)}, got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{stage} must use a floating-point dtype")
    if value.numel() == 0 or not torch.isfinite(value).all().item():
        raise ValueError(f"{stage} contains non-finite or empty output")
    return value


# ------------------------------------------------------------------- real target signals
# Three FIXED REAL photos from the classic 24-image Kodak Lossless True Color Image
# Suite, center-cropped to _RES x _RES, spanning increasing frequency content:
#   low    = kodim10 (calm dockside scene)               laplacian_var ~ 228
#   medium = kodim07 (flowers + window lattice)           laplacian_var ~ 943
#   high   = kodim13 (forest / rock / mountain texture)   laplacian_var ~ 4740
# The package data preparation script records the source-image selection and bake
# provenance. The runtime consumes only the fixed arrays staged for verification.
#
# These are baked ONCE (offline, deterministic) into {INR_DATA}/{name}.npz by that prep
# script (coords [R*R,2] in [-1,1], target [R*R,3] in [0,1], row-major matching
# _coord_grid's ravel order) — the harness never re-downloads or regenerates at run time.
_RES = 256                    # image resolution (square); 256^2 = 65536 coords
                              # (enough pixels that a 256-wide 4-layer MLP cannot
                              # memorize to machine precision, so PSNRs land in the
                              # realistic 15-40 dB band and the method gap is exposed)

_SIGNAL_NAMES = ("low", "medium", "high")
_COORDS_SHA256 = "2ec4e6d2329db8d380428d6a411ff722a15715a05fbd067bf56540fedfd8996c"
_TARGET_SHA256 = {
    "low": "7a96bbe1ecfc6822cd92bfeec76690fbdee039905edbfeec89fee04726e2f59d",
    "medium": "e486c21f5e19429308bef5435f0234c6332aee0975d9e005d01d0f07580187be",
    "high": "a63555e868a232e07027295f99fd5f4aab00fda840219de9890c022eb03d2c79",
}


def _coord_grid(res: int) -> np.ndarray:
    """INR verification runtime component."""
    lin = np.linspace(-1.0, 1.0, res, dtype=np.float64)
    yy, xx = np.meshgrid(lin, lin, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)  # [res*res, 2]


def _data_root() -> Path:
    return Path(os.environ.get("INR_DATA", "/data/inr-signal-fitting"))


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_signal(name: str):
    """INR verification runtime component.





    """
    if name not in _SIGNAL_NAMES:
        raise ValueError(f"unknown signal {name!r}; choose from {list(_SIGNAL_NAMES)}")
    dev = device()
    path = _data_root() / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing staged Kodak signal {path}; run "
            "vendor/data_scripts/inr-signal-fitting/prepare_data.py "
            "--data-root <root> (or `mlsbench data prepare inr-signal-fitting`) first"
        )
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"coords", "target"}:
            raise ValueError(f"{path} must contain exactly coords and target arrays")
        coords = np.asarray(z["coords"])
        target = np.asarray(z["target"])
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError(f"invalid coordinate array shape {coords.shape} in {path}")
    if target.shape != (coords.shape[0], 3):
        raise ValueError(f"invalid target array shape {target.shape} in {path}")
    if coords.dtype != np.float64:
        raise TypeError(f"coordinates in {path} must use the anchored float64 dtype")
    if target.dtype != np.float64:
        raise TypeError(f"targets in {path} must use the anchored float64 dtype")
    if not np.isfinite(coords).all() or not np.isfinite(target).all():
        raise ValueError(f"non-finite signal data in {path}")
    if np.min(coords) < -1.000001 or np.max(coords) > 1.000001:
        raise ValueError(f"coordinates in {path} must lie in [-1, 1]")
    if np.min(target) < 0.0 or np.max(target) > 1.0:
        raise ValueError(f"targets in {path} must lie in [0, 1]")
    res = int(round(math.sqrt(coords.shape[0])))
    if res != _RES or res * res != coords.shape[0]:
        raise ValueError(
            f"signal {path} must contain a {_RES}x{_RES} grid, got {coords.shape[0]} rows"
        )
    expected_coords = _coord_grid(_RES)
    if not np.array_equal(coords, expected_coords):
        raise ValueError(f"coordinates in {path} do not match the anchored row-major grid")
    if _array_sha256(coords) != _COORDS_SHA256:
        raise ValueError(f"coordinate digest mismatch in {path}")
    if _array_sha256(target) != _TARGET_SHA256[name]:
        raise ValueError(f"target digest mismatch in {path}")
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=dev)
    target_t = torch.as_tensor(target, dtype=torch.float32, device=dev)
    return coords_t, target_t, res


# ------------------------------------------------------------- coordinate-MLP building blocks
class SineLayer(nn.Module):
    """INR verification runtime component.




    """

    def __init__(self, in_f, out_f, is_first=False, w0=30.0):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.in_f = in_f
        self.linear = nn.Linear(in_f, out_f)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                b = 1.0 / self.in_f
            else:
                b = math.sqrt(6.0 / self.in_f) / self.w0
            self.linear.weight.uniform_(-b, b)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class FourierFeatures(nn.Module):
    """INR verification runtime component.





    """

    def __init__(self, in_dim=2, num_freqs=128, sigma=10.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(num_freqs, in_dim, generator=g) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * num_freqs

    def forward(self, x):
        proj = 2.0 * math.pi * x @ self.B.t()  # [N, num_freqs]
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


# --- Fixed capacity budget shared by every method so comparisons are size-matched. ---
HIDDEN = 256                  # MLP width
N_LAYERS = 4                  # number of hidden layers (depth)
FOURIER_FREQS = 128           # number of random Fourier features (encoding dim = 256)


def build_relu_mlp(in_dim: int, hidden: int = HIDDEN, n_layers: int = N_LAYERS):
    """INR verification runtime component."""
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, 3)]
    return nn.Sequential(*layers)


def build_relu_mlp_head(in_dim: int, hidden: int = HIDDEN, n_layers: int = N_LAYERS):
    """INR verification runtime component."""
    return build_relu_mlp(in_dim, hidden, n_layers)


class SirenMLP(nn.Module):
    """INR verification runtime component.





    """

    def __init__(self, in_dim=2, hidden=HIDDEN, n_layers=N_LAYERS, w0=30.0, w0_hidden=30.0):
        super().__init__()
        layers = [SineLayer(in_dim, hidden, is_first=True, w0=w0)]
        for _ in range(n_layers - 1):
            layers.append(SineLayer(hidden, hidden, is_first=False, w0=w0_hidden))
        self.body = nn.Sequential(*layers)
        final = nn.Linear(hidden, 3)
        with torch.no_grad():
            b = math.sqrt(6.0 / hidden) / w0_hidden
            final.weight.uniform_(-b, b)
            final.bias.uniform_(-b, b)
        self.final = final

    def forward(self, x):
        return self.final(self.body(x))


# ------------------------------------------------------- additional coordinate-MLP blocks
# The blocks below give the extra inr-* research questions (depth, width, skip connections,
# encoding dimension, hash-grid encoding, normalization, learning-rate/schedule, coordinate
# transform, per-layer w0) a shared, size-matched implementation so every agent surface for
# a given RQ is scored identically. They all plug into the SAME fixed harness + PSNR scoring.

class SkipMLP(nn.Module):
    """INR verification runtime component.






    """

    def __init__(self, in_dim, hidden=None, n_layers=8, skip_at=4):
        super().__init__()
        hidden = HIDDEN if hidden is None else hidden
        self.n_layers = n_layers
        self.skip_at = skip_at
        self.in_dim = in_dim
        self.blocks = nn.ModuleList()
        d = in_dim
        for i in range(n_layers):
            d_in = d + in_dim if (skip_at is not None and i == skip_at) else d
            self.blocks.append(nn.Linear(d_in, hidden))
            d = hidden
        self.out = nn.Linear(hidden, 3)

    def forward(self, x):
        h = x
        for i, lin in enumerate(self.blocks):
            if self.skip_at is not None and i == self.skip_at:
                h = torch.cat([h, x], dim=-1)
            h = torch.relu(lin(h))
        return self.out(h)


class NormMLP(nn.Module):
    """INR verification runtime component.





    """

    def __init__(self, in_dim, hidden=None, n_layers=None, norm="none"):
        super().__init__()
        hidden = HIDDEN if hidden is None else hidden
        n_layers = N_LAYERS if n_layers is None else n_layers
        self.norm = norm
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            if norm == "layer":
                layers.append(nn.LayerNorm(hidden))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HashGridEncoder(nn.Module):
    """INR verification runtime component.









    """

    _PRIMES = (1, 2654435761, 805459861)

    def __init__(self, in_dim=2, n_levels=8, n_features=2, log2_hashmap=14,
                 base_res=16, finest_res=256, seed=0):
        super().__init__()
        if in_dim != 2:
            raise ValueError("HashGridEncoder requires two-dimensional coordinates")
        self.in_dim = in_dim
        self.n_levels = n_levels
        self.n_features = n_features
        self.table_size = 2 ** log2_hashmap
        b = 1.0 if n_levels <= 1 else math.exp(
            (math.log(finest_res) - math.log(base_res)) / (n_levels - 1))
        res = [int(round(base_res * (b ** l))) for l in range(n_levels)]
        self.register_buffer("res", torch.tensor(res, dtype=torch.long))
        g = torch.Generator().manual_seed(seed)
        # one embedding table per level, small init (Instant-NGP uses ~U(-1e-4, 1e-4))
        self.tables = nn.ParameterList([
            nn.Parameter((torch.rand(self.table_size, n_features, generator=g) * 2 - 1) * 1e-4)
            for _ in range(n_levels)
        ])
        self.out_dim = n_levels * n_features

    def _hash(self, ix, iy):
        # spatial hash of integer grid coords -> table index
        h = (ix * self._PRIMES[1]) ^ (iy * self._PRIMES[2])
        return h % self.table_size

    def forward(self, x):
        # x in [-1,1] -> [0,1]
        xs = (x * 0.5 + 0.5).clamp(0.0, 1.0)
        feats = []
        for l in range(self.n_levels):
            r = int(self.res[l].item())
            p = xs * (r - 1)
            x0 = torch.floor(p[:, 0]).long().clamp(0, r - 1)
            y0 = torch.floor(p[:, 1]).long().clamp(0, r - 1)
            x1 = (x0 + 1).clamp(0, r - 1)
            y1 = (y0 + 1).clamp(0, r - 1)
            wx = (p[:, 0] - x0.float()).unsqueeze(-1)
            wy = (p[:, 1] - y0.float()).unsqueeze(-1)
            t = self.tables[l]
            f00 = t[self._hash(x0, y0)]
            f10 = t[self._hash(x1, y0)]
            f01 = t[self._hash(x0, y1)]
            f11 = t[self._hash(x1, y1)]
            f0 = f00 * (1 - wx) + f10 * wx
            f1 = f01 * (1 - wx) + f11 * wx
            feats.append(f0 * (1 - wy) + f1 * wy)
        return torch.cat(feats, dim=-1)


def build_relu_mlp_head_small(in_dim, hidden=64, n_layers=2):
    """INR verification runtime component."""
    return build_relu_mlp(in_dim, hidden=hidden, n_layers=n_layers)


class CoordTransform(nn.Module):
    """INR verification runtime component.

















    """

    def __init__(self, mode: str = "identity", scale: float = 100.0):
        super().__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == "identity":
            return x
        if self.mode == "unit":
            return x * 0.5 + 0.5
        if self.mode == "inflate":
            return x * self.scale
        raise ValueError(f"unknown CoordTransform mode {self.mode!r}")


class EncodedMLP(nn.Module):
    """INR verification runtime component.




    """

    def __init__(self, transform: CoordTransform, sigma: float = 10.0,
                 num_freqs: int = None, hidden: int = None, n_layers: int = None):
        super().__init__()
        num_freqs = FOURIER_FREQS if num_freqs is None else num_freqs
        self.transform = transform
        self.encoder = FourierFeatures(in_dim=2, num_freqs=num_freqs, sigma=sigma)
        self.head = build_relu_mlp_head(
            self.encoder.out_dim,
            hidden=HIDDEN if hidden is None else hidden,
            n_layers=N_LAYERS if n_layers is None else n_layers,
        )

    def forward(self, x):
        return self.head(self.encoder(self.transform(x)))


_OUT_ACTS = {
    "identity": lambda z: z,                          # correct: target is in [0,1], no squash
    "sigmoid": torch.sigmoid,                         # correct: bounds output to (0,1)
    "relu": torch.relu,
    "tanh": torch.tanh,
}


class OutActMLP(nn.Module):
    """INR verification runtime component.














    """

    def __init__(self, out_act: str = "relu", sigma: float = 10.0, num_freqs: int = None):
        super().__init__()
        num_freqs = FOURIER_FREQS if num_freqs is None else num_freqs
        self.encoder = FourierFeatures(in_dim=2, num_freqs=num_freqs, sigma=sigma)
        self.body = build_relu_mlp_head(self.encoder.out_dim)
        if out_act not in _OUT_ACTS:
            raise ValueError(f"unknown output activation {out_act!r}")
        self.act = _OUT_ACTS[out_act]

    def forward(self, x):
        return self.act(self.body(self.encoder(x)))


# ------------------------------------------------------------------------- fixed trainer
STEPS = 2000                  # fixed number of full-batch Adam steps
LR = 5e-4                     # fixed learning rate (stable for both ReLU and SIREN here)


def _validate_train_inputs(model, coords, target, steps, lr, log_every, encoder=None):
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if encoder is not None and not isinstance(encoder, nn.Module):
        raise TypeError("encoder must be a torch.nn.Module")
    if not isinstance(coords, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("coords and target must be torch.Tensor instances")
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError(f"coords must have non-empty shape [N, 2], got {tuple(coords.shape)}")
    if target.shape != (coords.shape[0], 3):
        raise ValueError(f"target must have shape {(coords.shape[0], 3)}, got {tuple(target.shape)}")
    if not coords.is_floating_point() or not target.is_floating_point():
        raise TypeError("coords and target must use floating-point dtypes")
    if not torch.isfinite(coords).all().item() or not torch.isfinite(target).all().item():
        raise ValueError("coords and target must contain only finite values")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every <= 0:
        raise ValueError("log_every must be a positive integer")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise TypeError("learning rate must be numeric")
    if not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ValueError("learning rate must be finite and positive")


def _trainable_parameters(*modules):
    params = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
    if not params:
        raise ValueError("training surface exposes no trainable parameters")
    return params


def _checked_loss(pred, target, stage):
    pred = require_rgb_prediction(pred, target.shape[0], f"{stage} prediction")
    loss = torch.mean((pred - target) ** 2)
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise RuntimeError(f"{stage} loss is non-finite or non-scalar")
    return pred, loss


def _check_gradients(params, stage):
    missing = [param for param in params if param.grad is None]
    if missing:
        raise RuntimeError(f"{stage} left {len(missing)} trainable parameters without gradients")
    grads = [param.grad for param in params]
    finite = torch.stack([torch.isfinite(grad).all() for grad in grads]).all()
    if not finite.item():
        raise RuntimeError(f"{stage} produced non-finite gradients")


def _check_parameters(params, stage):
    finite = torch.stack([torch.isfinite(param).all() for param in params]).all()
    if not finite.item():
        raise RuntimeError(f"{stage} produced non-finite parameters")


def train_inr(model, coords, target, dev, steps: int = STEPS, lr: float = LR,
              log_every: int = 750, encoder=None, label: str = "eval"):
    """INR verification runtime component.








    """
    _validate_train_inputs(model, coords, target, steps, lr, log_every, encoder)
    model = model.to(dev)
    if encoder is not None:
        encoder = encoder.to(dev)
    coords = coords.to(dev)
    target = target.to(dev)
    params = _trainable_parameters(model)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    for step in range(steps):
        opt.zero_grad()
        inp = encoder(coords) if encoder is not None else coords
        pred, loss = _checked_loss(model(inp), target, f"train step {step + 1}")
        loss.backward()
        _check_gradients(params, f"train step {step + 1}")
        opt.step()
        _check_parameters(params, f"train step {step + 1}")
        sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            with torch.no_grad():
                p = torch.clamp(pred, 0.0, 1.0)
                mse = torch.mean((p - target) ** 2).item()
                psnr = -10.0 * math.log10(mse + 1e-12)
            print(f"STEP_METRICS label={label} step={step+1}/{steps} "
                  f"loss={loss.item():.6f} psnr={psnr:.3f}", flush=True)
    model.eval()
    return model


def train_inr_joint(model, coords, target, dev, steps: int = STEPS, lr: float = LR,
                    log_every: int = 750, encoder=None, label: str = "eval"):
    """INR verification runtime component.




    """
    _validate_train_inputs(model, coords, target, steps, lr, log_every, encoder)
    model = model.to(dev)
    if encoder is not None:
        encoder = encoder.to(dev)
    coords = coords.to(dev)
    target = target.to(dev)
    params = _trainable_parameters(model, encoder)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    for step in range(steps):
        opt.zero_grad()
        inp = encoder(coords) if encoder is not None else coords
        pred, loss = _checked_loss(model(inp), target, f"joint step {step + 1}")
        loss.backward()
        _check_gradients(params, f"joint step {step + 1}")
        opt.step()
        _check_parameters(params, f"joint step {step + 1}")
        sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            with torch.no_grad():
                p = torch.clamp(pred, 0.0, 1.0)
                mse = torch.mean((p - target) ** 2).item()
                psnr = -10.0 * math.log10(mse + 1e-12)
            print(f"STEP_METRICS label={label} step={step+1}/{steps} "
                  f"loss={loss.item():.6f} psnr={psnr:.3f}", flush=True)
    model.eval()
    return model


def train_inr_sched(model, coords, target, dev, lr: float = LR, schedule: str = "cosine",
                    steps: int = STEPS, log_every: int = 750, encoder=None,
                    label: str = "eval"):
    """INR verification runtime component.





    """
    _validate_train_inputs(model, coords, target, steps, lr, log_every, encoder)
    if schedule not in {"cosine", "constant"}:
        raise ValueError(f"unknown learning-rate schedule {schedule!r}")
    if encoder is not None and any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("train_inr_sched does not optimize trainable encoder parameters")
    model = model.to(dev)
    if encoder is not None:
        encoder = encoder.to(dev)
    coords = coords.to(dev)
    target = target.to(dev)
    params = _trainable_parameters(model)
    opt = torch.optim.Adam(params, lr=lr)
    if schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    else:
        sched = None
    model.train()
    for step in range(steps):
        opt.zero_grad()
        inp = encoder(coords) if encoder is not None else coords
        pred, loss = _checked_loss(model(inp), target, f"scheduled step {step + 1}")
        loss.backward()
        _check_gradients(params, f"scheduled step {step + 1}")
        opt.step()
        _check_parameters(params, f"scheduled step {step + 1}")
        if sched is not None:
            sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            with torch.no_grad():
                p = torch.clamp(pred, 0.0, 1.0)
                mse = torch.mean((p - target) ** 2).item()
                psnr = -10.0 * math.log10(mse + 1e-12)
            print(f"STEP_METRICS label={label} step={step+1}/{steps} "
                  f"loss={loss.item():.6f} psnr={psnr:.3f}", flush=True)
    model.eval()
    return model


def train_inr_jacobian(
    model,
    coords,
    target,
    dev,
    jacobian_weight: float = 0.0,
    steps: int = STEPS,
    lr: float = LR,
    log_every: int = 750,
    encoder=None,
    label: str = "eval",
):
    """INR verification runtime component.














    """
    _validate_train_inputs(model, coords, target, steps, lr, log_every, encoder)
    if isinstance(jacobian_weight, bool) or not isinstance(
        jacobian_weight, (int, float)
    ):
        raise TypeError("jacobian_weight must be numeric")
    if not math.isfinite(float(jacobian_weight)) or float(jacobian_weight) < 0.0:
        raise ValueError("jacobian_weight must be finite and non-negative")
    if encoder is not None and any(p.requires_grad for p in encoder.parameters()):
        raise ValueError(
            "train_inr_jacobian does not optimize trainable encoder parameters"
        )
    model = model.to(dev)
    if encoder is not None:
        encoder = encoder.to(dev)
    coords = coords.to(dev)
    target = target.to(dev)
    params = _trainable_parameters(model)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    for step in range(steps):
        opt.zero_grad()
        if jacobian_weight > 0.0:
            c = coords.detach().clone().requires_grad_(True)
            inp = encoder(c) if encoder is not None else c
            pred = model(inp)
            pred, data_loss = _checked_loss(
                pred, target, f"regularized step {step + 1}"
            )
            channel_grads = [
                torch.autograd.grad(
                    pred[:, channel].sum(),
                    c,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                for channel in range(pred.shape[1])
            ]
            jacobian = torch.stack(channel_grads, dim=1)
            if not torch.isfinite(jacobian).all().item():
                raise RuntimeError(
                    f"regularized step {step + 1} produced a non-finite RGB Jacobian"
                )
            jacobian_penalty = torch.mean(jacobian ** 2)
            loss = data_loss + jacobian_weight * jacobian_penalty
        else:
            inp = encoder(coords) if encoder is not None else coords
            pred, loss = _checked_loss(
                model(inp), target, f"regularized step {step + 1}"
            )
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise RuntimeError(f"regularized step {step + 1} loss is non-finite")
        loss.backward()
        _check_gradients(params, f"regularized step {step + 1}")
        opt.step()
        _check_parameters(params, f"regularized step {step + 1}")
        sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            with torch.no_grad():
                p = torch.clamp(pred.detach(), 0.0, 1.0)
                mse = torch.mean((p - target) ** 2).item()
                psnr = -10.0 * math.log10(mse + 1e-12)
            print(f"STEP_METRICS label={label} step={step+1}/{steps} "
                  f"loss={loss.item():.6f} psnr={psnr:.3f}", flush=True)
    model.eval()
    return model


def _require_plan(plan, keys, surface):
    if not isinstance(plan, dict):
        raise TypeError(f"{surface} surface_config() must return a JSON object")
    expected = set(keys)
    actual = set(plan)
    if actual != expected:
        raise ValueError(
            f"{surface} config keys must be {sorted(expected)}, got {sorted(actual)}"
        )
    return plan


def _plan_float(plan, key, low, high, surface, *, allow_zero=False):
    value = plan[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{surface}.{key} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{surface}.{key} must be finite")
    lower_ok = value >= low if allow_zero else value > low
    if not lower_ok or value > high:
        bracket = "[" if allow_zero else "("
        raise ValueError(
            f"{surface}.{key} must lie in {bracket}{low}, {high}]"
        )
    return value


def _plan_int(plan, key, low, high, surface):
    value = plan[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{surface}.{key} must be an integer")
    if value < low or value > high:
        raise ValueError(f"{surface}.{key} must lie in [{low}, {high}]")
    return value


def _predictor(model, dev, encoder=None):
    if encoder is not None:
        encoder = encoder.to(dev)

    @torch.no_grad()
    def predict(coords):
        model.eval()
        inp = encoder(coords.to(dev)) if encoder is not None else coords.to(dev)
        return model(inp).detach()

    return predict


def fit_surface(surface, plan, coords, target, dev):
    """Build and train the fixed pipeline for one validated single-axis config."""
    if surface == "activation":
        plan = _require_plan(plan, {"family"}, surface)
        family = plan["family"]
        if family == "relu":
            model = build_relu_mlp(in_dim=2)
            model = train_inr(model, coords, target, dev, label="activation_relu")
            return _predictor(model, dev)
        if family == "fourier":
            encoder = FourierFeatures(
                in_dim=2, num_freqs=FOURIER_FREQS, sigma=10.0
            )
            model = build_relu_mlp_head(encoder.out_dim)
            model = train_inr(
                model, coords, target, dev, encoder=encoder, label="activation_fourier"
            )
            return _predictor(model, dev, encoder)
        if family == "siren":
            model = SirenMLP(in_dim=2, w0=30.0, w0_hidden=30.0)
            model = train_inr(model, coords, target, dev, label="activation_siren")
            return _predictor(model, dev)
        raise ValueError("activation.family must be relu, fourier, or siren")

    if surface == "coord_transform":
        plan = _require_plan(plan, {"mode", "scale"}, surface)
        if plan["mode"] not in {"identity", "unit", "inflate"}:
            raise ValueError("coord_transform.mode must be identity, unit, or inflate")
        scale = _plan_float(plan, "scale", 0.0, 1000.0, surface)
        model = EncodedMLP(CoordTransform(mode=plan["mode"], scale=scale), sigma=10.0)
        model = train_inr(model, coords, target, dev, label="coord_transform")
        return _predictor(model, dev)

    if surface == "jacobian_reg":
        plan = _require_plan(plan, {"weight"}, surface)
        weight = _plan_float(plan, "weight", 0.0, 10.0, surface, allow_zero=True)
        encoder = FourierFeatures(in_dim=2, num_freqs=FOURIER_FREQS, sigma=10.0)
        model = build_relu_mlp_head(encoder.out_dim)
        model = train_inr_jacobian(
            model,
            coords,
            target,
            dev,
            jacobian_weight=weight,
            encoder=encoder,
            label="rgb_jacobian_reg",
        )
        return _predictor(model, dev, encoder)

    if surface == "encoding_dim":
        plan = _require_plan(plan, {"num_freqs"}, surface)
        num_freqs = _plan_int(plan, "num_freqs", 1, 512, surface)
        encoder = FourierFeatures(in_dim=2, num_freqs=num_freqs, sigma=10.0)
        model = build_relu_mlp_head(encoder.out_dim)
        model = train_inr(
            model, coords, target, dev, encoder=encoder, label="encoding_dim"
        )
        return _predictor(model, dev, encoder)

    if surface == "frequency":
        plan = _require_plan(plan, {"sigma"}, surface)
        sigma = _plan_float(plan, "sigma", 0.0, 100.0, surface)
        encoder = FourierFeatures(
            in_dim=2, num_freqs=FOURIER_FREQS, sigma=sigma
        )
        model = build_relu_mlp_head(encoder.out_dim)
        model = train_inr(model, coords, target, dev, encoder=encoder, label="frequency")
        return _predictor(model, dev, encoder)

    if surface == "hash_grid":
        plan = _require_plan(
            plan, {"n_levels", "base_res", "finest_res"}, surface
        )
        n_levels = _plan_int(plan, "n_levels", 1, 16, surface)
        base_res = _plan_int(plan, "base_res", 2, 256, surface)
        finest_res = _plan_int(plan, "finest_res", base_res, 512, surface)
        if n_levels == 1 and base_res != finest_res:
            raise ValueError("one-level hash grids require base_res == finest_res")
        encoder = HashGridEncoder(
            in_dim=2,
            n_levels=n_levels,
            base_res=base_res,
            finest_res=finest_res,
        )
        model = build_relu_mlp_head_small(encoder.out_dim)
        model = train_inr_joint(
            model, coords, target, dev, encoder=encoder, label="hash_grid"
        )
        return _predictor(model, dev, encoder)

    if surface == "lr_schedule":
        plan = _require_plan(plan, {"lr", "schedule"}, surface)
        lr = _plan_float(plan, "lr", 0.0, 0.5, surface)
        if plan["schedule"] not in {"constant", "cosine"}:
            raise ValueError("lr_schedule.schedule must be constant or cosine")
        encoder = FourierFeatures(in_dim=2, num_freqs=FOURIER_FREQS, sigma=10.0)
        model = build_relu_mlp_head(encoder.out_dim)
        model = train_inr_sched(
            model,
            coords,
            target,
            dev,
            lr=lr,
            schedule=plan["schedule"],
            encoder=encoder,
            label="lr_schedule",
        )
        return _predictor(model, dev, encoder)

    if surface == "depth":
        plan = _require_plan(plan, {"n_layers"}, surface)
        n_layers = _plan_int(plan, "n_layers", 1, 12, surface)
        encoder = FourierFeatures(in_dim=2, num_freqs=FOURIER_FREQS, sigma=10.0)
        model = build_relu_mlp_head(encoder.out_dim, n_layers=n_layers)
        model = train_inr(model, coords, target, dev, encoder=encoder, label="depth")
        return _predictor(model, dev, encoder)

    if surface == "width":
        plan = _require_plan(plan, {"hidden"}, surface)
        hidden = _plan_int(plan, "hidden", 1, 512, surface)
        encoder = FourierFeatures(in_dim=2, num_freqs=FOURIER_FREQS, sigma=10.0)
        model = build_relu_mlp_head(encoder.out_dim, hidden=hidden)
        model = train_inr(model, coords, target, dev, encoder=encoder, label="width")
        return _predictor(model, dev, encoder)

    if surface == "per_layer_w0":
        plan = _require_plan(plan, {"first", "hidden"}, surface)
        first = _plan_float(plan, "first", 0.0, 200.0, surface)
        hidden = _plan_float(plan, "hidden", 0.0, 200.0, surface)
        model = SirenMLP(in_dim=2, w0=first, w0_hidden=hidden)
        model = train_inr(model, coords, target, dev, label="sine_frequency")
        return _predictor(model, dev)

    if surface == "skip":
        plan = _require_plan(plan, {"skip_at"}, surface)
        skip_at = plan["skip_at"]
        if skip_at is not None:
            if isinstance(skip_at, bool) or not isinstance(skip_at, int):
                raise TypeError("skip.skip_at must be null or an integer")
            if skip_at < 1 or skip_at > 7:
                raise ValueError("skip.skip_at must lie in [1, 7]")
        encoder = FourierFeatures(in_dim=2, num_freqs=FOURIER_FREQS, sigma=10.0)
        model = SkipMLP(in_dim=encoder.out_dim, n_layers=8, skip_at=skip_at)
        model = train_inr(model, coords, target, dev, encoder=encoder, label="skip")
        return _predictor(model, dev, encoder)

    raise ValueError(f"unsupported INR surface {surface!r}")


# ----------------------------------------------------------------------------- scoring
def psnr_db(pred, target) -> float:
    """INR verification runtime component.




    """
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("pred and target must be torch.Tensor instances")
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(
            f"pred and target must share shape [N, 3], got {tuple(pred.shape)} and "
            f"{tuple(target.shape)}"
        )
    if pred.numel() == 0:
        raise ValueError("pred and target must be non-empty")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred and target must use floating-point dtypes")
    if not torch.isfinite(pred).all().item() or not torch.isfinite(target).all().item():
        raise ValueError("pred and target must contain only finite values")
    if torch.min(target).item() < 0.0 or torch.max(target).item() > 1.0:
        raise ValueError("target must lie in [0, 1]")
    p = torch.clamp(pred, 0.0, 1.0).reshape(-1).double()
    t = target.reshape(-1).double()
    mse = torch.mean((p - t) ** 2).item()
    if mse <= 1e-12:
        return 100.0
    return float(-10.0 * math.log10(mse))
