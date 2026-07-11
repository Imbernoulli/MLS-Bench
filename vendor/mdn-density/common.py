"""Shared utilities for the mdn-* (mixture density network) MLS-Bench tasks.

Fixed, un-gameable evaluation path: every task builds a Mixture Density Network
(Bishop 1994) — an MLP that maps an input x to the parameters of a K-component
1-D Gaussian mixture p(y|x) = sum_k pi_k(x) N(y | mu_k(x), sigma_k(x)^2) — from
the agent-editable surface, trains it for a FIXED optimisation budget with a
FIXED optimiser on the exact mixture NEGATIVE-LOG-LIKELIHOOD, then reports test
NLL (nats, lower is better) on a FIXED held-out sample of a MULTIMODAL 1-D->1-D
regression problem.

Conditional-density setting
---------------------------
The fixed datasets are inverse problems in which the conditional target may be
multimodal. Every candidate is evaluated with the same exact held-out mixture
log likelihood.

The NLL is the exact log-sum-exp mixture log likelihood on unseen points.
Component scales are bounded inside the frozen evaluator before this quantity is
computed.

The agent controls one bounded JSON-literal design axis. Trusted verifier code
constructs the model; agent-authored Python is never imported or executed. The
target data, seed, optimiser, budget, log-sum-exp NLL, variance guard, and
evaluator are frozen here. One sibling exposes a bounded component-balance
regularization coefficient in the trusted training objective; its held-out
metric remains the same unregularized NLL. Pure torch runs on one GPU in
seconds-to-minutes.

Data provenance (held-out generator + held-out TEST split)
------------------------------------------------------------
The train/test samples are FIXED and pre-generated; this module does NOT
contain (and never imports) the data-generating process. The exact forward-map
formulas and their hardcoded noise-scale / rotation / covariance constants
that produced the raw samples live host-side only, in
``holdout/mdn-density/dgp.py`` (never copied into the agent's workspace) -- so
a model cannot shortcut training by reading off the true closed-form
conditional density and hand-coding it; it must actually fit the data. See
that module's docstring for the full threat model. The setting NAME
(inverse_sine/two_branch/spiral/rot_bimodal) is legitimate, disclosed framing;
only the exact generator code/constants are held out.

In addition, the TEST split -- the literal held-out answer key the scored NLL
is computed against -- is not shipped alongside this file at all.
``make_dataset``/``make_dataset_2d`` load agent-visible ``train_raw`` from the
frozen package archive and verifier-only ``test_raw`` from the task data
archive. Both archives are checked against immutable byte-level SHA-256
contracts. The test archive is staged under
``tests/meta/data`` and reachable only via the ``TASK_DIR``/``_task`` symlink
that Harbor's ``test.sh`` (and ``score_task.py``'s
``_install_task_meta_legacy_links``) create ONLY during verification --
see ``harbor_adapter/src/mls_bench/adapter.py::_stage_verifier_assets`` and
``harbor_adapter/src/mls_bench/task-template/tests/test.sh``. During an
agent's action session ``$TASK_DIR/data/`` does not exist, so the test split
is genuinely absent from the agent's view, not merely present-but-unlabeled.
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
import torch.nn.functional as F

DEVICE = "cuda"


def require_cuda_device() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the full MDN protocol requires one visible CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "the full MDN protocol requires exactly one visible CUDA GPU"
        )

# Numerical floor on every component std-dev, applied INSIDE the frozen NLL
# evaluator. This is the anti-variance-collapse guard: it makes a sigma->0
# "spike on a memorised train point" strategy useless on held-out data. Chosen
# configured independently of the editable head.
SIGMA_FLOOR = 1e-3
# Upper clamp keeps a pathological exp() blow-up from producing inf/nan.
SIGMA_CEIL = 1e3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Agent-editable surface loader
# ---------------------------------------------------------------------------

_SURFACE_SOURCE_BYTES = 16 * 1024
_SURFACE_AST_NODES = 128


def load_surface_config(sol_path: str) -> dict:
    """Parse one finite JSON literal without executing agent-authored Python."""
    path = Path(sol_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {path}")
    try:
        source = path.read_text()
        if len(source.encode()) > _SURFACE_SOURCE_BYTES:
            raise ValueError("solution configuration exceeds 16 KiB")
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
        payload = json.dumps(value, allow_nan=False, separators=(",", ":"))
        value = json.loads(payload)
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError(f"surface_config must return a finite JSON literal: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("surface_config must return a JSON object")
    return value


def require_finite_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise RuntimeError(f"{label} must be a non-empty tensor")
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"MDN_NONFINITE {label}")
    return value


def require_finite_module(module, label: str) -> None:
    for name, parameter in module.named_parameters():
        require_finite_tensor(parameter, f"{label} parameter {name!r}")


def require_finite_gradients(module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            require_finite_tensor(parameter.grad, f"gradient {name!r}")


def _validate_training_inputs(model, x_tr, y_tr, x_te, y_te, *, steps, batch_size, lr):
    if not isinstance(model, nn.Module):
        raise TypeError("build_mdn must return a torch.nn.Module")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or not 1 <= batch_size <= 8192
    ):
        raise ValueError("batch_size must be an integer in [1, 8192]")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise TypeError("learning rate must be numeric")
    if not math.isfinite(float(lr)) or not 1e-5 <= float(lr) <= 1e-1:
        raise ValueError("learning rate must be finite and in [1e-5, 1e-1]")
    for label, value in (
        ("training inputs", x_tr), ("training targets", y_tr),
        ("test inputs", x_te), ("test targets", y_te),
    ):
        require_finite_tensor(value, label)
    if x_tr.ndim != 2 or x_te.ndim != 2 or x_tr.shape[1] != 1 or x_te.shape[1] != 1:
        raise RuntimeError("MDN inputs must have shape [N, 1]")
    if y_tr.ndim != 2 or y_te.ndim != 2:
        raise RuntimeError("MDN targets must be rank-2 tensors")
    if x_tr.shape[0] != y_tr.shape[0] or x_te.shape[0] != y_te.shape[0]:
        raise RuntimeError("MDN input and target row counts do not match")


def _validate_1d_outputs(pi_logits, mu, log_sigma, y) -> None:
    for label, value in (
        ("mixture logits", pi_logits), ("mixture means", mu),
        ("mixture log scales", log_sigma),
    ):
        require_finite_tensor(value, label)
    if pi_logits.ndim != 2 or mu.shape != pi_logits.shape or log_sigma.shape != pi_logits.shape:
        raise RuntimeError("1-D MDN outputs must have matching [B, K] shapes")
    if y.ndim != 2 or y.shape != (pi_logits.shape[0], 1):
        raise RuntimeError("1-D MDN targets must have shape [B, 1]")


def _validate_2d_outputs(pi_logits, mu, tril, y) -> None:
    for label, value in (
        ("mixture logits", pi_logits), ("mixture means", mu),
        ("mixture precision factors", tril),
    ):
        require_finite_tensor(value, label)
    if pi_logits.ndim != 2:
        raise RuntimeError("2-D mixture logits must have shape [B, K]")
    batch, components = pi_logits.shape
    if mu.shape != (batch, components, 2) or tril.shape != (batch, components, 3):
        raise RuntimeError("2-D MDN outputs have invalid shapes")
    if y.shape != (batch, 2):
        raise RuntimeError("2-D MDN targets must have shape [B, 2]")


# ---------------------------------------------------------------------------
# Fixed multimodal inverse targets: one Bishop-derived and three local extensions.
# loaded from pre-generated data, no generator here.
# ---------------------------------------------------------------------------
# The exact forward-map formulas (and their hardcoded noise-scale / rotation /
# covariance constants) live host-side only in holdout/mdn-density/dgp.py,
# which is NEVER staged into the agent's workspace/image. This module only
# knows the PUBLIC target names.
#
# Train samples load from agent-visible _mdn_data/*.npz. Test samples load from
# the task's verifier-only data/ directory via TASK_DIR and are absent during the
# agent action session. x := column 0; y := remaining columns.

TARGETS = {"inverse_sine", "two_branch", "spiral"}
TARGETS_2D = {"rot_bimodal"}

_DATA_DIR = Path(__file__).resolve().parent / "_mdn_data"
_DATA_SHA256 = {
    "inverse_sine": {
        "train": "9bc8821db766ac288db6cb1d5235d8288c59da1c217d885c66221659781b1c41",
        "test": "0650c8fa5d6b8ee7b2632def88901f1b281ebc2c49ae147207bf7087b3e82d39",
    },
    "two_branch": {
        "train": "e48a4d98064ab407bc36e8a345572a2004414f1e60645724330c9e514ffc2aa9",
        "test": "7ebd1e2943273ad9a8ea989bfc07305dff6722a2bf7b718b2666bc99bf19815f",
    },
    "spiral": {
        "train": "5ce99ae78507fb6d06fa6a7cf7c90863665a21c8ab8ab9e72a0bf4276a7ab3cb",
        "test": "e365e41b2db7566bd62ed36c668e93d10f71ea5780db198fb1b7cb7e2b4993f0",
    },
    "rot_bimodal": {
        "train": "7fcd661720ea9c6cdaa308315fbdc4d72ab7c0ac588f9fe1d9306eb3d3db86a1",
        "test": "902b60a9b4bd22129e5142e52f656b03b0fd72b33e93f9ae19affef44b129897",
    },
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _test_data_path(target: str, seed: int) -> Path:
    """Resolve the verifier-only held-out test npz for `target`/`seed`.

    Only exists under $TASK_DIR/data (set by test.sh / score_task.py's
    _install_task_meta_legacy_links to the per-task verifier meta dir) during
    scoring; absent during the agent's action session.
    """
    task_dir = Path(os.environ.get("TASK_DIR", "/workspace/_task"))
    return task_dir / "data" / f"{target}_seed{seed}_test.npz"


def data_proof(target: str, seed: int) -> tuple[str, str]:
    """Return verified train/test archive hashes for the one supported seed."""
    if seed != 42 or target not in _DATA_SHA256:
        raise RuntimeError(f"no immutable MDN data contract for {target!r}/seed={seed}")
    train_path = _DATA_DIR / f"{target}_seed{seed}.npz"
    test_path = _test_data_path(target, seed)
    expected = _DATA_SHA256[target]
    observed_train = _file_sha256(train_path)
    observed_test = _file_sha256(test_path)
    if observed_train != expected["train"] or observed_test != expected["test"]:
        raise RuntimeError(
            f"MDN data checksum mismatch for {target}: "
            f"train={observed_train}, test={observed_test}"
        )
    return observed_train, observed_test


def _load_raw(target: str, seed: int, n_train: int, n_test: int) -> tuple:
    """Load the (train_raw, test_raw) numpy arrays for `target`/`seed`.

    train_raw comes from the agent-visible _mdn_data/ archive; test_raw comes
    from the verifier-only $TASK_DIR/data/ location (see module docstring).
    Validates `n_train`/`n_test` against the actual row counts so a stale or
    mismatched request fails loudly instead of silently substituting data of
    a different size.
    """
    train_npz_path = _DATA_DIR / f"{target}_seed{seed}.npz"
    if not train_npz_path.exists():
        raise SystemExit(
            f"no frozen train data for target={target!r} seed={seed}; "
            f"expected {train_npz_path}. Regenerate with "
            f"holdout/mdn-density/generate_data.py (host-side only)."
        )
    data_proof(target, seed)
    with np.load(train_npz_path) as npz:
        tr = np.asarray(npz["train_raw"])

    test_npz_path = _test_data_path(target, seed)
    if not test_npz_path.exists():
        raise SystemExit(
            f"no held-out test data for target={target!r} seed={seed}; "
            f"expected {test_npz_path}. This file is staged verifier-side "
            f"only (tests/meta/data, mounted at $TASK_DIR/data) and is not "
            f"available during an agent action session -- the held-out NLL "
            f"can only be computed at scoring time."
        )
    with np.load(test_npz_path) as npz:
        te = np.asarray(npz["test_raw"])

    if tr.shape[0] != n_train or te.shape[0] != n_test:
        raise SystemExit(
            f"frozen data for target={target!r} seed={seed} has "
            f"{tr.shape[0]}/{te.shape[0]} train/test rows, requested "
            f"{n_train}/{n_test}"
        )
    if tr.ndim != 2 or te.ndim != 2 or tr.shape[1] != te.shape[1]:
        raise RuntimeError("MDN raw train/test arrays have incompatible shapes")
    if not np.isfinite(tr).all() or not np.isfinite(te).all():
        raise RuntimeError("MDN raw train/test arrays contain non-finite values")
    return tr, te


def make_dataset_2d(target: str, n_train: int, n_test: int, seed: int):
    """FIXED train/test samples for a 2-D target (x standardized by TRAIN stats).

    Loads the frozen ``_mdn_data/<target>_seed<seed>.npz`` archive (produced
    host-side by ``holdout/mdn-density/generate_data.py`` from the held-out
    DGP) rather than sampling anything itself, then applies the (public)
    x-standardization step.

    Returns (x_tr, y_tr, x_te, y_te): x is (N,1), y is (N,2) float32 tensors."""
    if target not in TARGETS_2D:
        raise SystemExit(f"unknown 2-D target {target!r}; choose from {list(TARGETS_2D)}")
    tr, te = _load_raw(target, seed, n_train, n_test)

    x_mean = tr[:, 0].mean()
    x_std = tr[:, 0].std()
    if not math.isfinite(float(x_std)) or x_std < 1e-8:
        raise RuntimeError("MDN training input standard deviation is invalid")

    def _split(a):
        x = ((a[:, 0] - x_mean) / x_std).astype(np.float32)
        y = a[:, 1:3].astype(np.float32)
        return (torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y))

    x_tr, y_tr = _split(tr)
    x_te, y_te = _split(te)
    for label, value in (("x_tr", x_tr), ("y_tr", y_tr), ("x_te", x_te), ("y_te", y_te)):
        require_finite_tensor(value, label)
    return x_tr, y_tr, x_te, y_te


def make_dataset(target: str, n_train: int, n_test: int, seed: int):
    """FIXED train/test samples for a target. Test uses a disjoint RNG stream.

    Loads the frozen ``_mdn_data/<target>_seed<seed>.npz`` archive (produced
    host-side by ``holdout/mdn-density/generate_data.py`` from the held-out
    DGP) rather than sampling anything itself, then applies the (public)
    x-standardization step.

    Returns (x_tr, y_tr, x_te, y_te) as (N,1) float32 tensors, with x
    standardized by TRAIN statistics (y left on its native scale so NLL is
    directly comparable across models)."""
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target!r}; choose from {list(TARGETS)}")
    tr, te = _load_raw(target, seed, n_train, n_test)

    x_mean = tr[:, 0].mean()
    x_std = tr[:, 0].std()
    if not math.isfinite(float(x_std)) or x_std < 1e-8:
        raise RuntimeError("MDN training input standard deviation is invalid")

    def _split(a):
        x = ((a[:, 0] - x_mean) / x_std).astype(np.float32)
        y = a[:, 1].astype(np.float32)
        return (torch.from_numpy(x).unsqueeze(1),
                torch.from_numpy(y).unsqueeze(1))

    x_tr, y_tr = _split(tr)
    x_te, y_te = _split(te)
    for label, value in (("x_tr", x_tr), ("y_tr", y_tr), ("x_te", x_te), ("y_te", y_te)):
        require_finite_tensor(value, label)
    return x_tr, y_tr, x_te, y_te


# ---------------------------------------------------------------------------
# Frozen mixture NLL (log-sum-exp) — the metric core
# ---------------------------------------------------------------------------

def mixture_nll(pi_logits: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
    """Exact per-point 1-D Gaussian-mixture NLL, computed in log-space.

    Args (all for a batch of B points, K components):
      pi_logits : (B, K) unnormalised mixture-weight logits (softmax'd here)
      mu        : (B, K) component means
      log_sigma : (B, K) component log std-devs (RAW head output; the head may
                  use exp/softplus internally, but the FROZEN evaluator re-floors
                  sigma so a collapsed component cannot game the metric)
      y         : (B, 1) targets

    Returns (B,) per-point NLL = -log sum_k pi_k N(y | mu_k, sigma_k^2).
    """
    _validate_1d_outputs(pi_logits, mu, log_sigma, y)
    # Bound in log-space so a finite but extreme head output cannot overflow exp().
    bounded_log_sigma = log_sigma.clamp(math.log(SIGMA_FLOOR), math.log(SIGMA_CEIL))
    sigma = torch.exp(bounded_log_sigma)
    log_sigma_c = torch.log(sigma)                       # floored log-sigma
    log_pi = F.log_softmax(pi_logits, dim=1)             # (B,K)

    y = y.expand_as(mu)                                  # (B,K)
    # log N(y|mu,sigma) = -0.5*log(2pi) - log(sigma) - 0.5*((y-mu)/sigma)^2
    log_comp = (
        -0.5 * math.log(2.0 * math.pi)
        - log_sigma_c
        - 0.5 * ((y - mu) / sigma) ** 2
    )                                                    # (B,K)
    log_prob = torch.logsumexp(log_pi + log_comp, dim=1)  # (B,)
    result = -log_prob
    require_finite_tensor(result, "per-sample mixture NLL")
    return result


# ---------------------------------------------------------------------------
# Frozen training loop + held-out NLL evaluation
# ---------------------------------------------------------------------------

def train_and_eval(model, x_tr, y_tr, x_te, y_te, *,
                   steps: int, batch_size: int, lr: float, seed: int,
                   weight_decay: float = 0.0,
                   component_balance_weight: float = 0.0,
                   log_every: int = 500):
    """FROZEN optimisation budget shared by all mdn tasks.

    The `model` is an nn.Module whose forward(x) returns
    (pi_logits, mu, log_sigma) each (B, K). Trained with Adam on the exact
    mixture NLL for a FIXED number of steps, optionally with the one declared
    component-balance penalty, then returns mean unregularized held-out NLL.
    """
    _validate_training_inputs(
        model, x_tr, y_tr, x_te, y_te,
        steps=steps, batch_size=batch_size, lr=lr,
    )
    if y_tr.shape[1] != 1 or y_te.shape[1] != 1:
        raise RuntimeError("1-D MDN training requires one target dimension")
    if (
        isinstance(component_balance_weight, bool)
        or not isinstance(component_balance_weight, (int, float))
        or not math.isfinite(float(component_balance_weight))
        or not 0.0 <= float(component_balance_weight) <= 1.0
    ):
        raise ValueError("component balance weight must be finite and in [0, 1]")
    component_balance_weight = float(component_balance_weight)
    model = model.to(DEVICE)
    require_finite_module(model, "MDN model")
    x_tr, y_tr = x_tr.to(DEVICE), y_tr.to(DEVICE)
    x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    n = x_tr.shape[0]
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        xb, yb = x_tr[idx], y_tr[idx]
        opt.zero_grad(set_to_none=True)
        pi_logits, mu, log_sigma = model(xb)
        batch_nll = mixture_nll(pi_logits, mu, log_sigma, yb).mean()
        loss = batch_nll
        if component_balance_weight > 0.0:
            mean_usage = torch.softmax(pi_logits, dim=1).mean(dim=0)
            balance_kl = torch.sum(
                mean_usage
                * (
                    torch.log(mean_usage.clamp_min(1e-12))
                    + math.log(mean_usage.numel())
                )
            )
            require_finite_tensor(balance_kl, "component-balance KL")
            loss = batch_nll + component_balance_weight * balance_kl
        if loss.numel() != 1 or not torch.isfinite(loss).item():
            raise RuntimeError(f"MDN_NONFINITE training loss at step {step}")
        loss.backward()
        require_finite_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(f"MDN_NONFINITE gradient norm at step {step}")
        opt.step()
        require_finite_module(model, "MDN model")
        if step % log_every == 0 or step == steps - 1:
            print(
                f"MDN_TRAIN step={step} train_nll={batch_nll.item():.4f}",
                flush=True,
            )

    # Exact held-out mixture NLL (mean over disjoint test points).
    model.eval()
    with torch.no_grad():
        nlls = []
        for i in range(0, x_te.shape[0], 8192):
            xb, yb = x_te[i:i + 8192], y_te[i:i + 8192]
            pi_logits, mu, log_sigma = model(xb)
            nlls.append(mixture_nll(pi_logits, mu, log_sigma, yb).double())
        test_nll = torch.cat(nlls).mean().item()
    if not math.isfinite(test_nll):
        raise RuntimeError("MDN_NONFINITE held-out NLL")
    return float(test_nll)


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))

class Trunk(nn.Module):
    """Fixed 2-hidden-layer Tanh MLP trunk: 1 -> H -> H, exposing a feature
    vector of width H. Frozen across all tasks so only the HEAD design varies."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Frozen 2-D mixture NLL + train/eval (for the covariance / diag-vs-full surface)
# ---------------------------------------------------------------------------
# A 2-D Gaussian component is parameterised by its mean mu (2,) and a lower
# Cholesky L of the PRECISION matrix (so Sigma^{-1} = L L^T). The head emits
# tril=[L11, L21, L22]; the diagonal entries L11,L22 are POSITIVE (produced via
# softplus+floor in the head) so log|Sigma^{-1}|^{1/2} = log(L11)+log(L22). This
# is the standard numerically-stable full-covariance MDN parameterisation.

# Anti-collapse: floor the precision-root diagonal to a CEIL (equivalently floor
# sigma). A component cannot shrink its 2-D variance to spike on a train point
# and cheat on held-out data.
PREC_ROOT_CEIL = 1.0 / SIGMA_FLOOR   # matches the 1-D SIGMA_FLOOR guard


def mixture_nll_2d(pi_logits: torch.Tensor, mu: torch.Tensor,
                   tril: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact per-point 2-D Gaussian-mixture NLL (precision-Cholesky form).

    Args (B points, K components):
      pi_logits : (B, K)   mixture-weight logits
      mu        : (B, K, 2) component means
      tril      : (B, K, 3) precision-root lower-Cholesky [L11, L21, L22], with
                  L11,L22 > 0 (the head guarantees positivity)
      y         : (B, 2)    targets
    Returns (B,) per-point NLL.
    """
    _validate_2d_outputs(pi_logits, mu, tril, y)
    if not torch.all(tril[..., (0, 2)] > 0).item():
        raise RuntimeError("2-D MDN precision diagonals must be strictly positive")
    L11 = tril[..., 0].clamp(max=PREC_ROOT_CEIL)          # (B,K) anti-collapse
    L21 = tril[..., 1]
    L22 = tril[..., 2].clamp(max=PREC_ROOT_CEIL)
    d = y.unsqueeze(1) - mu                               # (B,K,2)
    d1, d2 = d[..., 0], d[..., 1]
    # z = L^T d  (L lower-triangular precision root): quadratic form = |z|^2
    z1 = L11 * d1 + L21 * d2
    z2 = L22 * d2
    quad = z1 * z1 + z2 * z2                              # (B,K)
    log_det_half = torch.log(L11) + torch.log(L22)        # log|Sigma^{-1}|^{1/2}
    log_comp = -math.log(2.0 * math.pi) + log_det_half - 0.5 * quad
    log_pi = F.log_softmax(pi_logits, dim=1)              # (B,K)
    log_prob = torch.logsumexp(log_pi + log_comp, dim=1)  # (B,)
    result = -log_prob
    require_finite_tensor(result, "per-sample 2-D mixture NLL")
    return result


def train_and_eval_2d(model, x_tr, y_tr, x_te, y_te, *,
                      steps: int, batch_size: int, lr: float, seed: int,
                      weight_decay: float = 0.0, log_every: int = 500):
    """FROZEN 2-D optimisation budget (mirrors train_and_eval for 1-D).

    model.forward(x) returns (pi_logits, mu, tril) with mu (B,K,2), tril
    (B,K,3). Returns mean held-out 2-D mixture NLL."""
    _validate_training_inputs(
        model, x_tr, y_tr, x_te, y_te,
        steps=steps, batch_size=batch_size, lr=lr,
    )
    if y_tr.shape[1] != 2 or y_te.shape[1] != 2:
        raise RuntimeError("2-D MDN training requires two target dimensions")
    model = model.to(DEVICE)
    require_finite_module(model, "2-D MDN model")
    x_tr, y_tr = x_tr.to(DEVICE), y_tr.to(DEVICE)
    x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    n = x_tr.shape[0]
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        xb, yb = x_tr[idx], y_tr[idx]
        opt.zero_grad(set_to_none=True)
        pi_logits, mu, tril = model(xb)
        loss = mixture_nll_2d(pi_logits, mu, tril, yb).mean()
        if loss.numel() != 1 or not torch.isfinite(loss).item():
            raise RuntimeError(f"MDN_NONFINITE training loss at step {step}")
        loss.backward()
        require_finite_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(f"MDN_NONFINITE gradient norm at step {step}")
        opt.step()
        require_finite_module(model, "2-D MDN model")
        if step % log_every == 0 or step == steps - 1:
            print(f"MDN_TRAIN step={step} train_nll={loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        nlls = []
        for i in range(0, x_te.shape[0], 8192):
            xb, yb = x_te[i:i + 8192], y_te[i:i + 8192]
            pi_logits, mu, tril = model(xb)
            nlls.append(mixture_nll_2d(pi_logits, mu, tril, yb).double())
        test_nll = torch.cat(nlls).mean().item()
    if not math.isfinite(test_nll):
        raise RuntimeError("MDN_NONFINITE held-out 2-D NLL")
    return float(test_nll)
