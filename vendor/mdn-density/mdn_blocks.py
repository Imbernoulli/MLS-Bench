"""Reusable Mixture-Density-Network head constructors for the mdn-* tasks.

Pure-torch building blocks used by the frozen MDN task builders. Every builder
returns an `nn.Module` whose `forward(x)` yields the
mixture parameters `(pi_logits, mu, log_sigma)`, each of shape `(B, K)`, on top
of a configurable MLP trunk. (For the 2-D covariance surface a separate builder
emits a full/diagonal 2-D Gaussian mixture — see `mdn2d`.)

The design axes exposed here are exactly the classic MDN levers (Bishop 1994;
Kruse 2020) plus the standard neural-net knobs, so each mdn-* research question
is one editable surface plugging into the SAME frozen harness:
  * K            — number of mixture components.
  * var_mode     — how the positive component std-dev is produced from the raw
                   head output:
                       "exp"      : sigma = exp(z)          (Bishop's original)
                       "softplus" : sigma = softplus(z)+eps
                       "free"     : sigma = exp(z) without a head-side floor
  * hidden       — trunk width (capacity of the shared feature extractor).
  * depth        — number of hidden layers in the trunk.
  * act          — trunk activation ("tanh"/"relu"/"gelu"/"sigmoid").
  * sigma_eps    — head-side variance floor added to sigma.
  * sigma_init   — starting sigma scale (initialisation of the variance head).
  * init_gain    — weight-init gain for the mixture head (Xavier scaling).
  * feature      — input feature map ("raw" identity vs "fourier" random
                   Fourier features that make the multimodal boundary learnable
                   faster).
  * lr           — the module may REQUEST an Adam learning rate the frozen
                   harness will honour (the only frozen-optimiser knob exposed).

Nothing here touches the frozen metric: the log-sum-exp NLL and the SIGMA_FLOOR
anti-collapse guard live in `common.py` and are re-applied inside the evaluator.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import common


# ---------------------------------------------------------------------------
# Configurable trunk (width / depth / activation / input features)
# ---------------------------------------------------------------------------

_ACTS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
}


class FourierFeatures(nn.Module):
    """Fixed random Fourier feature map R^1 -> R^{2m}: x -> [sin(Bx),cos(Bx)].

    Random-but-seeded projection frequencies (deterministic given the global
    seed the harness sets).
    """

    def __init__(self, in_dim: int = 1, num: int = 16, scale: float = 4.0):
        super().__init__()
        B = torch.randn(in_dim, num) * scale
        self.register_buffer("B", B)
        self.out_dim = 2 * num

    def forward(self, x):
        proj = x @ self.B  # (batch, num)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ConfigTrunk(nn.Module):
    """MLP trunk with configurable width / depth / activation / input features.

    The DEFAULT (hidden=64, depth=2, act='tanh', feature='raw') reproduces the
    frozen `common.Trunk` exactly, so surfaces that do not touch these knobs are
    identical to the original harness."""

    def __init__(self, hidden: int = 64, depth: int = 2, act: str = "tanh",
                 feature: str = "raw", fourier_num: int = 16,
                 fourier_scale: float = 4.0):
        super().__init__()
        self.hidden = hidden
        if act not in _ACTS:
            raise ValueError(f"unknown act {act!r}; choose from {list(_ACTS)}")
        if feature == "fourier":
            self.feat = FourierFeatures(1, fourier_num, fourier_scale)
            in_dim = self.feat.out_dim
        elif feature == "raw":
            self.feat = None
            in_dim = 1
        else:
            raise ValueError(f"unknown feature {feature!r}; choose raw|fourier")
        act_cls = _ACTS[act]
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(1, depth)):
            layers += [nn.Linear(d, hidden), act_cls()]
            d = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        return self.net(x)


# ---------------------------------------------------------------------------
# 1-D Gaussian-mixture head (the standard MDN)
# ---------------------------------------------------------------------------

class MDNHead(nn.Module):
    """MLP-MDN: shared trunk + a linear head emitting 3*K mixture params.

    forward(x) -> (pi_logits, mu, log_sigma), each (B, K).

    Trusted builders may set width, depth, activation, features, sigma floor, or
    initialization. Leaving them at their defaults reproduces the original head
    exactly, so the component-count and variance-floor surfaces share the same
    base head.

    A small sigma_init keeps the initial sigmas at a sensible scale so
    optimisation starts in a well-conditioned region. `temperature` is retained
    at exactly 1.0 only to preserve the measured representative numerical path;
    it is not an editable surface. `sigma_eps` is a head-side variance floor.
    """

    def __init__(self, k: int, hidden: int = 64, var_mode: str = "exp",
                 sigma_init: float = 0.3, sigma_eps: float = 0.0,
                 depth: int = 2, act: str = "tanh", feature: str = "raw",
                 temperature: float = 1.0, init_gain: float = 1.0,
                 fourier_num: int = 16, fourier_scale: float = 4.0):
        super().__init__()
        if var_mode not in ("exp", "softplus", "free"):
            raise ValueError(f"unknown var_mode {var_mode!r}")
        self.k = int(k)
        self.var_mode = var_mode
        self.sigma_eps = float(sigma_eps)
        self.temperature = float(temperature)
        if self.temperature != 1.0:
            raise ValueError("mixture temperature is fixed at 1.0")
        self.trunk = ConfigTrunk(hidden=hidden, depth=depth, act=act,
                                 feature=feature, fourier_num=fourier_num,
                                 fourier_scale=fourier_scale)
        self.pi = nn.Linear(hidden, self.k)
        self.mu = nn.Linear(hidden, self.k)
        self.sig = nn.Linear(hidden, self.k)
        # Initialise sigma output so the starting sigma ~ sigma_init.
        with torch.no_grad():
            if var_mode == "softplus":
                # softplus(b) ~ sigma_init  =>  b = log(exp(sigma_init)-1)
                b = float(torch.log(torch.expm1(torch.tensor(sigma_init))))
            else:  # exp / free
                b = float(torch.log(torch.tensor(sigma_init)))
            self.sig.bias.fill_(b)
            self.sig.weight.mul_(0.1)
            # Optional weight-init gain (initialisation surface): rescale the
            # mixture-mean/weight heads. gain != 1 changes the starting basin.
            if init_gain != 1.0:
                self.mu.weight.mul_(init_gain)
                self.pi.weight.mul_(init_gain)

    def forward(self, x):
        h = self.trunk(x)
        pi_logits = self.pi(h) / self.temperature
        mu = self.mu(h)
        z = self.sig(h)
        if self.var_mode == "softplus":
            # softplus keeps its historic +1e-3 eps unless overridden higher.
            sigma = F.softplus(z) + max(self.sigma_eps, 1e-3)
            log_sigma = torch.log(sigma)
        else:  # "exp" or "free" both use exp; "free" simply skips any clamping
            log_sigma = z
            if self.sigma_eps > 0.0 and self.var_mode == "exp":
                # OPT-IN head-side variance floor (the variance-floor surface):
                # default sigma_eps for exp is 0.0 -> identical to original head.
                log_sigma = torch.log(torch.exp(z) + self.sigma_eps)
        return pi_logits, mu, log_sigma


def mdn(k: int, hidden: int = 64, var_mode: str = "exp",
        sigma_init: float = 0.3, lr: float | None = None, **kw) -> nn.Module:
    """Build a K-component MLP-MDN head (the standard constructor).

    Trusted keyword args select the declared task surface. The compatibility
    `temperature` argument is fixed at 1.0. Omitting all keyword args reproduces
    the original frozen head. If `lr` is given it is attached as a `.lr`
    attribute the frozen harness will honour (the learning-rate surface).
    """
    m = MDNHead(k=k, hidden=hidden, var_mode=var_mode, sigma_init=sigma_init,
                **kw)
    if lr is not None:
        m.lr = float(lr)
    return m


def single_gaussian(hidden: int = 64, var_mode: str = "exp", **kw) -> nn.Module:
    """Convenience constructor for a heteroscedastic K=1 Gaussian head."""
    return MDNHead(k=1, hidden=hidden, var_mode=var_mode, **kw)


class PointRegressor(nn.Module):
    """MSE point regressor read out as a fixed-width Gaussian (K=1, sigma FIXED).

    Trained through the same NLL loss with a non-trainable scale, so gradients
    update only the conditional mean."""

    def __init__(self, hidden: int = 64, sigma: float = 0.3):
        super().__init__()
        self.trunk = ConfigTrunk(hidden=hidden)
        self.k = 1
        self.mu = nn.Linear(hidden, 1)
        self.register_buffer("_log_sigma", torch.tensor([[float(torch.log(torch.tensor(sigma)))]]))

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu(h)                                   # (B,1)
        pi_logits = torch.zeros_like(mu)                  # single component
        log_sigma = self._log_sigma.expand_as(mu)         # fixed width
        return pi_logits, mu, log_sigma


# ---------------------------------------------------------------------------
# 2-D Gaussian-mixture head (diagonal vs FULL covariance surface)
# ---------------------------------------------------------------------------

class MDN2DHead(nn.Module):
    """2-D MDN emitting a K-component bivariate Gaussian mixture over (y1, y2).

    forward(x) -> (pi_logits, mu, tril) where mu is (B,K,2) and tril is
    (B,K,3) holding the lower-Cholesky entries [L11, L21, L22] of each
    component's precision-root. With `cov='diag'` the off-diagonal L21 is forced
    to zero (axis-aligned Gaussians); with `cov='full'` L21 is free, so each
    component can rotate to align with a curved / correlated branch of the
    target -> a lower held-out NLL on a target with correlated conditional modes.

    The diagonal of L is produced through softplus + a floor (mirrors the 1-D
    variance guard), so 2-D variance collapse is likewise neutralised on
    held-out data.
    """

    def __init__(self, k: int, cov: str = "full", hidden: int = 64,
                 depth: int = 2, act: str = "tanh", diag_floor: float = 1e-2):
        super().__init__()
        if cov not in ("diag", "full"):
            raise ValueError(f"unknown cov {cov!r}; choose diag|full")
        self.k = int(k)
        self.cov = cov
        self.diag_floor = float(diag_floor)
        self.trunk = ConfigTrunk(hidden=hidden, depth=depth, act=act)
        self.pi = nn.Linear(hidden, self.k)
        self.mu = nn.Linear(hidden, self.k * 2)
        # 3 tril params per component: diag1, diag2 (via softplus), offdiag
        self.tril = nn.Linear(hidden, self.k * 3)
        with torch.no_grad():
            # start diag ~ 3 (precision-root; sigma ~ 1/3) — a moderate width
            self.tril.bias.zero_()
            self.tril.weight.mul_(0.1)

    def forward(self, x):
        h = self.trunk(x)
        B = h.shape[0]
        pi_logits = self.pi(h)
        mu = self.mu(h).view(B, self.k, 2)
        raw = self.tril(h).view(B, self.k, 3)
        d1 = F.softplus(raw[..., 0]) + self.diag_floor
        d2 = F.softplus(raw[..., 1]) + self.diag_floor
        off = raw[..., 2] if self.cov == "full" else torch.zeros_like(raw[..., 2])
        tril = torch.stack([d1, off, d2], dim=-1)         # (B,K,3)
        return pi_logits, mu, tril


def mdn2d(k: int, cov: str = "full", hidden: int = 64, **kw) -> nn.Module:
    """Build a 2-D (bivariate) K-component MDN with diagonal or full covariance."""
    return MDN2DHead(k=k, cov=cov, hidden=hidden, **kw)
