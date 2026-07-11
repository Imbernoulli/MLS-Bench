"""Shared utilities for the Mamba (selective state-space model) tasks.

Everything here is fixed. The agent supplies one finite JSON-literal plan; trusted
builders construct the selected parameterization, initialization, or module. Agent
Python is never imported or executed. The current model is a tiny selective SSM
(Mamba-1 style block: in_proj -> causal conv1d -> selective scan -> gated out_proj)
built on the real `mamba-ssm` CUDA `selective_scan_fn` kernel.

Formal evaluation follows the selective-copying scale in Gu and Dao, Mamba,
Appendix E.1: total sequence length 4096, 16 memorized tokens, vocabulary 16,
two D=64 layers, batch 64, learning rate 1e-4, and 400,000 optimizer steps.
Historical L256/L384/L512 smoke anchors are not eligible for scoring.

Task: SELECTIVE COPYING (Gu & Dao 2023, "Mamba", sec 4.1.1). A few data tokens are
scattered at random positions in a fixed-length sequence; the model must emit them,
in order, at a run of marker positions appended to the end. Because the data
tokens are at *content-dependent, random* positions, a time-invariant (LTI / S4)
recurrence cannot solve it — only an input-dependent (selective / S6) SSM can gate
the right tokens into state. This makes the task a clean discriminator of the SSM
design choices the agent makes.

The same TinyMamba harness hosts ten research questions. Each task freezes every
other axis and exposes one literal choice covering scan selectivity, A stability,
B/C coupling, convolution activation, Delta finalization, Delta initialization,
output gating, normalization, block residuals, or A initialization.
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


_SURFACE_SOURCE_BYTES = 64 * 1024
_SURFACE_AST_NODES = 256
_METRIC_PROTOCOL = "mamba_selective_copy_paper_e1_v1"
PROTOCOL_SHIP_ELIGIBLE = True
PROTOCOL_BLOCK_REASON = ""
_SURFACE_SPECS = {
    "parameterize": ("mode", {"lti", "bc_only", "selective"}),
    "compute_A": ("transform", {"identity", "neg_abs", "neg_exp"}),
    "couple_bc": ("coupling", {"tied", "constant", "independent"}),
    "conv_act": ("activation", {"identity", "relu", "silu"}),
    "finalize_dt": ("activation", {"identity", "relu", "softplus"}),
    "init_delta": ("scheme", {"too_large", "too_small", "log_uniform_s4d"}),
    "gate": ("activation", {"none", "sigmoid", "silu"}),
    "make_norm": ("normalization", {"none", "layer", "rms"}),
    "residual_step": ("residual", {"none", "scaled_add", "add"}),
    "init_state": ("scheme", {"constant_rate", "s4d_spectrum"}),
}


def load_surface_config(sol_path: Path | str) -> dict:
    """Read one finite JSON literal without importing agent-authored Python."""
    path = Path(sol_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {path}")
    try:
        source = path.read_text()
        if len(source.encode()) > _SURFACE_SOURCE_BYTES:
            raise ValueError("solution configuration exceeds 64 KiB")
        tree = ast.parse(source, filename=str(path))
        if sum(1 for _ in ast.walk(tree)) > _SURFACE_AST_NODES:
            raise ValueError("solution configuration exceeds the AST complexity limit")
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError(f"cannot parse solution configuration: {exc}") from exc

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "surface_config"
    ]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise ValueError("solution must define exactly one synchronous surface_config")
    function = functions[0]
    if (
        function.decorator_list
        or function.args.posonlyargs
        or function.args.args
        or function.args.kwonlyargs
        or function.args.vararg is not None
        or function.args.kwarg is not None
    ):
        raise ValueError("surface_config must be undecorated and accept no arguments")
    if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
        raise ValueError("surface_config must contain exactly one return statement")

    for node in tree.body:
        if node is function:
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        raise ValueError("solution contains executable top-level statements")

    try:
        value = ast.literal_eval(function.body[0].value)
        value = json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError(f"surface_config must return a finite JSON literal: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("surface_config must return a JSON object")
    return value


def validate_surface_config(surface: str, config: dict) -> dict:
    """Restrict each task to one enumerated, trusted Mamba design axis."""
    if surface not in _SURFACE_SPECS:
        raise ValueError(f"unknown Mamba surface {surface!r}")
    if not isinstance(config, dict):
        raise TypeError("surface configuration must be a JSON object")
    key, choices = _SURFACE_SPECS[surface]
    if set(config) != {key}:
        raise ValueError(f"{surface} requires exactly the {key!r} field")
    if config[key] not in choices:
        raise ValueError(f"invalid {surface} choice {config[key]!r}")
    return config


class _RMSNorm(nn.Module):
    """Dependency-free RMSNorm matching the standard Mamba normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, value):
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.eps).to(value.dtype)
        return normalized * self.weight


def build_surface_hook(surface: str, config: dict):
    """Construct a trusted hook for a validated literal surface plan."""
    config = validate_surface_config(surface, config)

    if surface == "parameterize":
        mode = config["mode"]
        if mode == "selective":
            return default_parameterize

        def parameterize(block, x, b, length):
            dt = repeat(
                torch.zeros_like(block.dt_const),
                "d -> b d l",
                b=b,
                l=length,
            ).contiguous()
            if mode == "lti":
                # Constant SSM parameters still need one state vector per
                # inner channel. Sharing a single N-vector across all channels
                # confounds selectivity with a large capacity reduction.
                B = block.B_const.float().contiguous()
                C = block.C_const.float().contiguous()
            else:
                x_dbl = block.x_proj(rearrange(x, "b d l -> (b l) d"))
                _dt, B, C = torch.split(
                    x_dbl,
                    [block.dt_rank, block.d_state, block.d_state],
                    dim=-1,
                )
                B = rearrange(B, "(b l) n -> b n l", l=length).contiguous()
                C = rearrange(C, "(b l) n -> b n l", l=length).contiguous()
            return dt, B, C, block.dt_const.float()

        return parameterize

    if surface == "compute_A":
        transform = config["transform"]

        def compute_A(A_log):
            if transform == "identity":
                return A_log.float()
            if transform == "neg_abs":
                return -torch.abs(A_log.float())
            return -torch.exp(A_log.float())

        return compute_A

    if surface == "couple_bc":
        coupling = config["coupling"]

        def couple_bc(block, B, C_lowrank, b, length):
            if coupling == "tied":
                return B.contiguous()
            if coupling == "constant":
                return block.C_const.float().contiguous()
            return rearrange(
                C_lowrank, "(b l) n -> b n l", l=length
            ).contiguous()

        return couple_bc

    if surface == "conv_act":
        activation = config["activation"]

        def conv_act(value):
            if activation == "identity":
                return value
            if activation == "relu":
                return torch.relu(value)
            return F.silu(value)

        return conv_act

    if surface == "finalize_dt":
        activation = config["activation"]

        def finalize_dt(block, dt):
            biased = dt + rearrange(
                block.dt_proj.bias.float(), "d -> 1 d 1"
            )
            if activation == "identity":
                return biased
            if activation == "relu":
                return torch.relu(biased)
            return F.softplus(biased)

        return finalize_dt

    if surface == "init_delta":
        scheme = config["scheme"]

        def init_delta(block):
            with torch.no_grad():
                if scheme == "log_uniform_s4d":
                    dt = torch.exp(
                        torch.rand(block.d_inner)
                        * (math.log(1e-1) - math.log(1e-3))
                        + math.log(1e-3)
                    ).clamp(min=1e-4)
                    value = dt + torch.log(-torch.expm1(-dt))
                else:
                    fill = 3.0 if scheme == "too_large" else -12.0
                    value = torch.full((block.d_inner,), fill)
                block.dt_proj.bias.copy_(value)
                block.dt_const.copy_(value)

        return init_delta

    if surface == "gate":
        activation = config["activation"]

        def gate(y, z):
            if activation == "none":
                return y
            if activation == "sigmoid":
                return y * torch.sigmoid(z)
            return y * F.silu(z)

        return gate

    if surface == "make_norm":
        normalization = config["normalization"]

        def make_norm(d_model):
            if normalization == "none":
                return nn.Identity()
            if normalization == "rms":
                return _RMSNorm(d_model)
            return nn.LayerNorm(d_model)

        return make_norm

    if surface == "residual_step":
        residual = config["residual"]

        def residual_step(hidden, block_out):
            if residual == "none":
                return block_out
            if residual == "scaled_add":
                return (hidden + block_out) * math.sqrt(0.5)
            return hidden + block_out

        return residual_step

    scheme = config["scheme"]

    def init_state(block):
        with torch.no_grad():
            if scheme == "constant_rate":
                # compute_A=-exp(A_log), so this is A=-1, not a zero matrix.
                block.A_log.zero_()
            else:
                value = repeat(
                    torch.arange(1, block.d_state + 1, dtype=torch.float32),
                    "n -> d n",
                    d=block.d_inner,
                ).contiguous()
                block.A_log.copy_(torch.log(value))

    return init_state


def load_surface_hook(sol_path: Path | str, surface: str):
    return build_surface_hook(surface, load_surface_config(sol_path))


def load_surface_hook_and_choice(sol_path: Path | str, surface: str):
    """Return the trusted hook and a proof-safe token for its literal choice."""
    config = load_surface_config(sol_path)
    hook = build_surface_hook(surface, config)
    key, _choices = _SURFACE_SPECS[surface]
    return hook, f"{key}.{config[key]}"


def require_finite_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise RuntimeError(f"{label} must be a non-empty tensor")
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"MAMBA_NONFINITE {label}")
    return value


def require_finite_module(module, label: str) -> None:
    for name, parameter in module.named_parameters():
        require_finite_tensor(parameter, f"{label} parameter {name!r}")


def require_finite_gradients(module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            require_finite_tensor(parameter.grad, f"gradient {name!r}")


# ----------------------------------------------------------------- data
def torch_copying_data(L, M, A, batch_size, variable=True, device="cpu", gen=None):
    """Selective-copying batch.

    tokens 1..A-2 = data; 0 = blank/filler; A-1 = output marker.
    ``L`` is the total sequence length, including the final ``M`` output
    markers. The prefix therefore has length ``L-M`` and contains ``M`` data
    tokens plus ``L-2M`` filler tokens.

    Returns (x:(B, L) inputs, y:(B, M) target data tokens at the tail).
    """
    if L <= 2 * M:
        raise ValueError("selective-copy sequence length must exceed 2*M")
    tokens = torch.randint(low=1, high=A - 1, size=(batch_size, M),
                           generator=gen, device=device)
    body_length = L - M
    if variable:
        inds = torch.stack([
            torch.randperm(body_length, generator=gen, device=device)[:M]
            for _ in range(batch_size)
        ], dim=0).sort(dim=1).values
    else:
        inds = torch.arange(M, device=device).repeat((batch_size, 1))
    body = torch.zeros(batch_size, body_length, dtype=torch.long, device=device)
    body.scatter_(1, inds, tokens)
    markers = (A - 1) * torch.ones(batch_size, M, dtype=torch.long, device=device)
    x = torch.cat([body, markers], dim=1)
    return x, tokens


# ----------------------------------------------------------------- default hooks
# Every editable design surface has a default here that reproduces the standard
# Mamba-1 block. A task overrides exactly ONE of these; the rest stay default.

def default_parameterize(block, x, b, l):
    """Full selective (S6) Δ/B/C from the post-conv features (standard Mamba)."""
    x_dbl = block.x_proj(rearrange(x, "b d l -> (b l) d"))
    dt, B, C = torch.split(x_dbl, [block.dt_rank, block.d_state, block.d_state], dim=-1)
    dt = block.project_dt(block, dt, b, l)
    dt = block.finalize_dt(block, dt)
    B = block.make_B(block, x, B, b, l)
    C = block.couple_bc(block, B, C, b, l)
    return dt, B, C, block.dt_proj.bias.float()


def default_project_dt(block, dt_lowrank, b, l):
    """Map the low-rank Δ (shape ((b l), dt_rank)) up to (b, d_inner, l)."""
    dt = block.dt_proj.weight @ dt_lowrank.t()
    return rearrange(dt, "d (b l) -> b d l", l=l)


def default_couple_bc(block, B, C_lowrank, b, l):
    """Read matrix C is an INDEPENDENT input-dependent projection (standard S6)."""
    return rearrange(C_lowrank, "(b l) n -> b n l", l=l).contiguous()


def default_make_B(block, x, B_lowrank, b, l):
    """Write matrix B is an INPUT-DEPENDENT projection (standard S6 selectivity)."""
    return rearrange(B_lowrank, "(b l) n -> b n l", l=l).contiguous()


def default_finalize_dt(block, dt):
    """Δ passthrough: the CUDA scan applies softplus (delta_softplus=True) itself."""
    return dt


def default_conv_act(x):
    """Post-conv nonlinearity (SiLU in the standard Mamba block)."""
    return F.silu(x)


def default_build_conv(d_inner, d_conv):
    """Depthwise causal conv1d of width d_conv over the inner features (Mamba)."""
    return nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner,
                     padding=d_conv - 1, bias=True)


def default_gate(y, z):
    """Gated output branch: multiply the SSM output by SiLU(z) (Mamba block)."""
    return y * F.silu(z)


def default_compute_A(A_log):
    """Stable diagonal A = -exp(A_log): strictly negative real eigenvalues."""
    return -torch.exp(A_log.float())


def default_apply_skip(y, x, D):
    """Add the D skip (residual) term: y = scan_out + D * x  (standard Mamba)."""
    return y + x * rearrange(D.float(), "d -> 1 d 1")


def default_make_norm(d_model):
    """Pre-block normalization (LayerNorm in this reference stack)."""
    return nn.LayerNorm(d_model)


def default_residual_step(h, block_out):
    """Residual (skip) connection around each block (pre-norm residual stack)."""
    return h + block_out


def default_init_ssm(block: "SSMBlock"):
    """Standard Mamba init: S4D-real A=-(1..N), softplus(dt)∈[dt_min,dt_max]."""
    d_state = block.d_state
    A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),
               "n -> d n", d=block.d_inner).contiguous()
    with torch.no_grad():
        block.A_log.copy_(torch.log(A))
    dt_min, dt_max = 1e-3, 1e-1
    dt = torch.exp(torch.rand(block.d_inner) * (math.log(dt_max) - math.log(dt_min))
                   + math.log(dt_min)).clamp(min=1e-4)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        block.dt_proj.bias.copy_(inv_dt)
        block.dt_const.copy_(inv_dt)


# ----------------------------------------------------------------- model
class SSMBlock(nn.Module):
    """Minimal Mamba-1 style selective-SSM block.

    All of its design surfaces are pluggable hooks (see the default_* functions).
    A task overrides one hook via the corresponding solution function; all unset
    hooks use the frozen implementation so tasks remain independent.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 parameterize=None, init_fn=None, project_dt=None, couple_bc=None,
                 build_conv=None, gate=None, compute_A=None, apply_skip=None,
                 make_B=None, finalize_dt=None, conv_act=None, init_allowed=None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)

        # ---- pluggable design hooks (default -> standard Mamba behaviour) ----
        self.parameterize = parameterize or default_parameterize
        self.project_dt = project_dt or default_project_dt
        self.couple_bc = couple_bc or default_couple_bc
        self.make_B = make_B or default_make_B
        self.finalize_dt = finalize_dt or default_finalize_dt
        self.conv_act = conv_act or default_conv_act
        self.gate = gate or default_gate
        self.compute_A = compute_A or default_compute_A
        self.apply_skip = apply_skip or default_apply_skip
        # The z-gate and D-skip are numerically FUSED inside the CUDA kernel by
        # default (bit-for-bit the validated standard-Mamba path). We only pull them
        # OUT of the kernel when the task under test IS the gate or skip surface, so
        # that hook is actually exercised. This keeps every OTHER task on the proven
        # fused path (the out-of-kernel path is slightly less stable at long L).
        self._ext_gate = gate is not None
        self._ext_skip = apply_skip is not None
        # Δ softplus: by default the kernel applies it (delta_softplus=True). When the
        # task IS the Δ-nonlinearity (finalize_dt) surface, the kernel does NOT apply
        # softplus and the agent's finalize_dt must supply the positivity nonlinearity.
        self._ext_dt = finalize_dt is not None

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        _build_conv = build_conv or default_build_conv
        self.conv1d = _build_conv(self.d_inner, d_conv)
        self.act = nn.SiLU()
        # Full input-dependent projection (selective path uses it; LTI path ignores it)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # Input-independent alternatives (for a time-invariant parameterization)
        self.B_const = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.1)
        self.C_const = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.1)
        self.dt_const = nn.Parameter(torch.zeros(self.d_inner))

        self.A_log = nn.Parameter(torch.empty(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Default initialization, optionally followed by the one init surface.
        # The init task may change only A/Delta initialization tensors; changing
        # another parameter, module, or hook would turn it into a multi-axis task.
        default_init_ssm(self)
        if init_fn is not None:
            allowed_by_surface = {
                "delta": {"dt_proj.bias", "dt_const"},
                "state": {"A_log"},
            }
            if init_allowed not in allowed_by_surface:
                raise ValueError("init_fn requires init_allowed='delta' or 'state'")
            allowed = allowed_by_surface[init_allowed]
            before_state = {name: value.detach().clone() for name, value in self.state_dict().items()}
            before_parameters = {name: id(value) for name, value in self.named_parameters()}
            before_buffers = {name: id(value) for name, value in self.named_buffers()}
            before_modules = {name: id(value) for name, value in self.named_modules()}
            hook_names = (
                "parameterize", "project_dt", "couple_bc", "make_B",
                "finalize_dt", "conv_act", "gate", "compute_A", "apply_skip",
            )
            before_hooks = {name: getattr(self, name) for name in hook_names}
            init_fn(self)
            if before_parameters != {name: id(value) for name, value in self.named_parameters()}:
                raise RuntimeError("init_ssm must not replace or add model parameters")
            if before_buffers != {name: id(value) for name, value in self.named_buffers()}:
                raise RuntimeError("init_ssm must not replace or add model buffers")
            if before_modules != {name: id(value) for name, value in self.named_modules()}:
                raise RuntimeError("init_ssm must not replace or add model modules")
            if any(getattr(self, name) is not value for name, value in before_hooks.items()):
                raise RuntimeError("init_ssm must not modify non-initialization hooks")
            after_state = self.state_dict()
            if set(after_state) != set(before_state):
                raise RuntimeError("init_ssm changed the model state schema")
            for name, previous in before_state.items():
                if name not in allowed and not torch.equal(after_state[name], previous):
                    raise RuntimeError(f"init_ssm modified non-active state {name!r}")
        self._frozen_hook_refs = {
            name: getattr(self, name)
            for name in (
                "parameterize", "project_dt", "couple_bc", "make_B",
                "finalize_dt", "conv_act", "gate", "compute_A", "apply_skip",
            )
        }

    def forward(self, hidden_states):
        parameter_versions = {
            name: parameter._version for name, parameter in self.named_parameters()
        }
        b, l, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l").contiguous()
        x = self.conv_act(self.conv1d(x)[..., :l])
        require_finite_tensor(x, "post-convolution features")
        if x.shape != (b, self.d_inner, l):
            raise RuntimeError("post-convolution hook returned the wrong shape")
        A = self.compute_A(self.A_log)
        require_finite_tensor(A, "state matrix A")
        if A.shape != (self.d_inner, self.d_state):
            raise RuntimeError("compute_A returned the wrong shape")
        parameters = self.parameterize(self, x, b, l)
        if not isinstance(parameters, (tuple, list)) or len(parameters) != 4:
            raise RuntimeError("parameterize must return (dt, B, C, delta_bias)")
        dt, B, C, delta_bias = parameters
        for label, value, shapes in (
            ("Delta", dt, {(b, self.d_inner, l)}),
            ("B", B, {(b, self.d_state, l), (self.d_inner, self.d_state)}),
            ("C", C, {(b, self.d_state, l), (self.d_inner, self.d_state)}),
            ("Delta bias", delta_bias, {(self.d_inner,)}),
        ):
            require_finite_tensor(value, label)
            if tuple(value.shape) not in shapes:
                raise RuntimeError(
                    f"{label} has shape {tuple(value.shape)}, expected one of "
                    f"{sorted(shapes)}"
                )
        dt = dt.contiguous()
        # Fused path (default): the kernel adds the D skip and applies the SiLU(z)
        # gate internally — the validated, most-stable standard-Mamba numerics.
        kern_D = None if self._ext_skip else self.D.float()
        kern_z = None if self._ext_gate else z
        # Softplus: kernel applies it unless the task IS the Δ-nonlinearity surface,
        # in which case the agent's finalize_dt already supplied the nonlinearity (and
        # owns the whole Δ, so the kernel adds no delta_bias and does no softplus).
        kern_bias = None if self._ext_dt else delta_bias
        y = selective_scan_fn(x, dt, A, B, C, D=kern_D, z=kern_z,
                              delta_bias=kern_bias, delta_softplus=not self._ext_dt)
        require_finite_tensor(y, "selective scan output")
        if y.shape != (b, self.d_inner, l):
            raise RuntimeError("selective scan returned the wrong shape")
        # When the task under test IS the skip/gate surface, apply it OUTSIDE the
        # kernel (the kernel got D=None / z=None above) so the agent hook is graded.
        if self._ext_skip:
            y = self.apply_skip(y, x, self.D)
        if self._ext_gate:
            y = self.gate(y, z)
        require_finite_tensor(y, "post-hook scan output")
        if y.shape != (b, self.d_inner, l):
            raise RuntimeError("gate or skip hook returned the wrong shape")
        if any(getattr(self, name) is not value for name, value in self._frozen_hook_refs.items()):
            raise RuntimeError("an active hook modified a non-active model hook")
        current_versions = {
            name: parameter._version for name, parameter in self.named_parameters()
        }
        if current_versions != parameter_versions:
            raise RuntimeError("an active hook modified model parameters during forward")
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


class TinyMamba(nn.Module):
    def __init__(self, vocab, d_model=64, n_layer=2, d_state=16, expand=2, d_conv=4,
                 parameterize=None, init_fn=None, project_dt=None, couple_bc=None,
                 build_conv=None, gate=None, compute_A=None, apply_skip=None,
                 make_norm=None, residual_step=None, make_B=None, finalize_dt=None,
                 conv_act=None, init_allowed=None):
        super().__init__()
        _make_norm = make_norm or default_make_norm
        self.residual_step = residual_step or default_residual_step
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            SSMBlock(d_model, d_state=d_state, expand=expand, d_conv=d_conv,
                     parameterize=parameterize, init_fn=init_fn, project_dt=project_dt,
                     couple_bc=couple_bc, build_conv=build_conv, gate=gate,
                     compute_A=compute_A, apply_skip=apply_skip, make_B=make_B,
                     finalize_dt=finalize_dt, conv_act=conv_act,
                     init_allowed=init_allowed)
            for _ in range(n_layer)])
        self.norms = nn.ModuleList([_make_norm(d_model) for _ in range(n_layer)])
        self.norm_f = _make_norm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for blk, nrm in zip(self.blocks, self.norms):
            h = self.residual_step(h, blk(nrm(h)))
        return self.lm_head(self.norm_f(h))


def _format_proof_float(value: float) -> str:
    return format(float(value), ".12g")


def _proof_fields(*, protocol, task, label, surface_choice, L, M, A, d_model, d_state,
                  n_layer, steps, batch, lr, optimizer, weight_decay,
                  grad_clip, eval_batches, seed, n_params) -> str:
    for field, value in (
        ("protocol", protocol), ("task", task), ("label", label),
        ("optimizer", optimizer), ("surface_choice", surface_choice),
    ):
        if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
            raise ValueError(f"Mamba proof {field} must be a non-empty token")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Mamba proof seed must be an integer")
    if isinstance(n_params, bool) or not isinstance(n_params, int) or n_params <= 0:
        raise ValueError("Mamba proof n_params must be a positive integer")
    train_examples = steps * batch
    eval_examples = eval_batches * batch
    return (
        f"protocol={protocol} task={task} label={label} surface={surface_choice} "
        f"L={L} M={M} A={A} "
        f"d_model={d_model} d_state={d_state} n_layer={n_layer} steps={steps} "
        f"batch={batch} lr={_format_proof_float(lr)} optimizer={optimizer} "
        f"weight_decay={_format_proof_float(weight_decay)} "
        f"grad_clip={_format_proof_float(grad_clip)} eval_batches={eval_batches} "
        f"seed={seed} n_params={n_params} train_examples={train_examples} "
        f"train_tokens={train_examples * L} eval_examples={eval_examples} "
        f"eval_tokens={eval_examples * L}"
    )


def format_metric_line(kind: str, *, protocol: str, task: str, label: str,
                       surface_choice: str,
                       L: int, M: int, A: int, d_model: int, d_state: int,
                       n_layer: int, steps: int, batch: int, lr: float,
                       optimizer: str, weight_decay: float, grad_clip: float,
                       eval_batches: int, seed: int, result: dict) -> str:
    if kind not in {"MAMBA_COPY_METRICS", "MAMBA_INIT_METRICS"}:
        raise ValueError("unknown Mamba metric kind")
    required = {
        "acc", "n_params", "final_loss", "wall_s", "eval_examples",
        "eval_correct",
    }
    if not isinstance(result, dict) or set(result) != required:
        raise RuntimeError("Mamba result is incomplete")
    acc = result["acc"]
    final_loss = result["final_loss"]
    wall_s = result["wall_s"]
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in (acc, final_loss, wall_s)
    ):
        raise RuntimeError("Mamba metric values must be finite")
    if not 0.0 <= float(acc) <= 1.0 or float(final_loss) < 0 or float(wall_s) < 0:
        raise RuntimeError("Mamba metric values are outside their valid ranges")
    expected_eval_examples = eval_batches * batch
    eval_examples = result["eval_examples"]
    eval_correct = result["eval_correct"]
    if (
        isinstance(eval_examples, bool) or not isinstance(eval_examples, int)
        or eval_examples != expected_eval_examples
        or isinstance(eval_correct, bool) or not isinstance(eval_correct, int)
        or not 0 <= eval_correct <= eval_examples * M
        or abs(float(acc) - eval_correct / (eval_examples * M)) > 0.5e-6
    ):
        raise RuntimeError("Mamba evaluation cardinality is inconsistent")
    fields = _proof_fields(
        protocol=protocol, task=task, label=label, surface_choice=surface_choice,
        L=L, M=M, A=A,
        d_model=d_model, d_state=d_state, n_layer=n_layer, steps=steps,
        batch=batch, lr=lr, optimizer=optimizer, weight_decay=weight_decay,
        grad_clip=grad_clip, eval_batches=eval_batches, seed=seed,
        n_params=result["n_params"],
    )
    return (
        f"{kind} {fields} copy_acc={float(acc):.6f} "
        f"final_loss={float(final_loss):.6f} wall_s={float(wall_s):.1f} "
        f"eval_correct={eval_correct}"
    )


def train_and_eval(parameterize, init_fn, *, steps, L, M, A, d_model, d_state,
                   n_layer, batch, lr, seed, device, expand=2, d_conv=4,
                   project_dt=None, couple_bc=None, build_conv=None, gate=None,
                   compute_A=None, apply_skip=None, make_norm=None,
                   residual_step=None, make_B=None, finalize_dt=None, conv_act=None,
                   init_allowed=None, proof_task, proof_label, surface_choice,
                   protocol=_METRIC_PROTOCOL, optimizer_name="adam",
                   weight_decay=0.0, grad_clip=1.0, eval_batches=16):
    integer_args = {
        "steps": steps, "L": L, "M": M, "A": A, "d_model": d_model,
        "d_state": d_state, "n_layer": n_layer, "batch": batch,
        "expand": expand, "d_conv": d_conv,
    }
    for label, value in integer_args.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if A < 3 or L <= 2 * M:
        raise ValueError("selective-copy dimensions are invalid")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise TypeError("learning rate must be numeric")
    if not math.isfinite(float(lr)) or not 0.0 < float(lr) <= 1.0:
        raise ValueError("learning rate must be finite and in (0, 1]")
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("optimizer_name must be 'adam' or 'adamw'")
    for label, value in (("weight_decay", weight_decay), ("grad_clip", grad_clip)):
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0
        ):
            raise ValueError(f"{label} must be a finite non-negative number")
    if optimizer_name == "adam" and float(weight_decay) != 0.0:
        raise ValueError("Adam protocol requires weight_decay=0")
    if isinstance(eval_batches, bool) or not isinstance(eval_batches, int) or eval_batches <= 0:
        raise ValueError("eval_batches must be a positive integer")

    set_all_seeds(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = TinyMamba(vocab=A, d_model=d_model, n_layer=n_layer, d_state=d_state,
                      expand=expand, d_conv=d_conv,
                      parameterize=parameterize, init_fn=init_fn, project_dt=project_dt,
                      couple_bc=couple_bc, build_conv=build_conv, gate=gate,
                      compute_A=compute_A, apply_skip=apply_skip, make_norm=make_norm,
                      residual_step=residual_step, make_B=make_B, finalize_dt=finalize_dt,
                      conv_act=conv_act, init_allowed=init_allowed).to(device)
    require_finite_module(model, "Mamba model")
    n_params = sum(p.numel() for p in model.parameters())
    proof_fields = _proof_fields(
        protocol=protocol, task=proof_task, label=proof_label,
        surface_choice=surface_choice, L=L, M=M, A=A,
        d_model=d_model, d_state=d_state, n_layer=n_layer, steps=steps,
        batch=batch, lr=lr, optimizer=optimizer_name, weight_decay=weight_decay,
        grad_clip=grad_clip, eval_batches=eval_batches, seed=seed,
        n_params=n_params,
    )
    if device not in {"cpu", "cuda"}:
        raise ValueError("Mamba device proof must be cpu or cuda")
    print(f"POOL_LOADED {proof_fields} device={device}", flush=True)
    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=float(weight_decay)
        )
    import time
    t0 = time.time()
    model.train()
    last = None
    for step in range(steps):
        x, y = torch_copying_data(L, M, A, batch, variable=True, device=device, gen=gen)
        logits = model(x)[:, -M:, :]
        require_finite_tensor(logits, "training logits")
        if logits.shape != (batch, M, A):
            raise RuntimeError("Mamba training logits have the wrong shape")
        loss = F.cross_entropy(logits.reshape(-1, A), y.reshape(-1))
        if loss.numel() != 1 or not torch.isfinite(loss).item():
            raise RuntimeError(f"MAMBA_NONFINITE training loss at step {step}")
        opt.zero_grad()
        loss.backward()
        require_finite_gradients(model)
        if float(grad_clip) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(grad_clip)
            )
            if not torch.isfinite(grad_norm).item():
                raise RuntimeError(f"MAMBA_NONFINITE gradient norm at step {step}")
        opt.step()
        require_finite_module(model, "Mamba model")
        last = loss.item()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"TRAIN step={step} loss={last:.4f}", flush=True)
    if last is None or not math.isfinite(last):
        raise RuntimeError("MAMBA_NONFINITE missing or invalid final training loss")
    print(
        f"MAMBA_TRAIN_COMPLETE {proof_fields} final_loss={last:.6f}",
        flush=True,
    )
    model.eval()
    with torch.no_grad():
        ev = torch.Generator(device=device).manual_seed(seed + 777)
        correct = tot = 0
        for _ in range(eval_batches):
            x, y = torch_copying_data(L, M, A, batch, variable=True, device=device, gen=ev)
            eval_logits = model(x)[:, -M:, :]
            require_finite_tensor(eval_logits, "evaluation logits")
            if eval_logits.shape != (batch, M, A):
                raise RuntimeError("Mamba evaluation logits have the wrong shape")
            pred = eval_logits.argmax(-1)
            correct += (pred == y).sum().item(); tot += y.numel()
        if tot <= 0:
            raise RuntimeError("Mamba evaluation produced no predictions")
        acc = correct / tot
    if not math.isfinite(acc) or not 0.0 <= acc <= 1.0:
        raise RuntimeError("MAMBA_NONFINITE invalid evaluation accuracy")
    eval_examples = eval_batches * batch
    if tot != eval_examples * M:
        raise RuntimeError("Mamba evaluation cardinality proof is inconsistent")
    print(
        f"MAMBA_EVAL_COMPLETE {proof_fields} eval_correct={correct} "
        f"copy_acc={acc:.6f}",
        flush=True,
    )
    wall_s = time.time() - t0
    if not math.isfinite(wall_s) or wall_s < 0:
        raise RuntimeError("invalid Mamba wall time")
    return {
        "acc": acc,
        "n_params": n_params,
        "final_loss": last,
        "wall_s": wall_s,
        "eval_examples": eval_examples,
        "eval_correct": correct,
    }
