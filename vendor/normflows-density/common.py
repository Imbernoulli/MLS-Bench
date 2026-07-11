"""Shared utilities for the normflows (flow-*) MLS-Bench tasks.

Fixed evaluation path: every task loads a deterministic 2-D density dataset,
builds a normalizing flow from one literal agent-selected value, trains it for a
fixed optimization budget with a fixed optimizer on the exact log-likelihood
objective, then reports total test negative log-likelihood (nats, lower is
better) on a fixed held-out sample.

The metric is the exact change-of-variables NLL of the learned flow on unseen
data and is computed only after the fixed training loop completes.

The agent controls only one scalar, enum, or literal mask sequence. The frozen
harness maps it to a complete layer stack. The target density, train/test
samples, seed, optimizer, 20,000-step budget, and NLL evaluator are fixed. The
pinned repository image already contains the complete runtime; verification does
not install, download, extract, or compile dependencies.

Data provenance and verifier-only evaluation assets
----------------------------------------------------
The train/test samples are fixed and pre-generated; this module does not
contain or import the sampling process. The exact closed-form sampler lives
host-side in ``holdout/normflows-density/dgp.py``. Only sampled arrays enter
verification.

The final render lists this module, the frozen harness, flow builders, and all
training archives in ``verifier_only_package_files``. They are restored only
during verification, so the evaluated agent sees the literal solution surface
but not the scoring implementation or data arrays. ``make_dataset`` then loads
``train_x`` from ``_flow_data/<target>_seed<seed>.npz`` and loads ``test_x``
from the task's separate ``data/<target>_seed<seed>_test.npz`` under
``$TASK_DIR/data``. The adapter authenticates and stages both classes of assets
under the verifier mount; neither the DGP nor held-out arrays enter the agent's
action workspace.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 2  # all toy targets are 2-D


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Agent-editable surface loader
# ---------------------------------------------------------------------------

_SURFACE_SOURCE_BYTES = 32 * 1024
_SURFACE_AST_NODES = 256


def _literal_surface_value(sol_path: str, attr: str):
    """Parse one no-argument literal-return surface without executing Python."""
    path = Path(sol_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {path}")
    source = path.read_text()
    if len(source.encode()) > _SURFACE_SOURCE_BYTES:
        raise ValueError("flow solution surface exceeds 32 KiB")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"flow solution surface does not parse: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > _SURFACE_AST_NODES:
        raise ValueError("flow solution surface exceeds 256 AST nodes")

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    matches = [
        node for node in functions
        if isinstance(node, ast.FunctionDef) and node.name == attr
    ]
    if len(matches) != 1 or len(functions) != 1:
        raise ValueError(f"solution must define exactly one `{attr}()` function")
    function = matches[0]
    if (function.decorator_list or function.args.posonlyargs or function.args.args
            or function.args.vararg is not None or function.args.kwarg is not None
            or function.args.kwonlyargs or function.args.defaults
            or function.args.kw_defaults):
        raise ValueError(f"`{attr}` must be undecorated and take no arguments")

    for node in tree.body:
        if node is function:
            continue
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        raise ValueError("flow solution may not contain imports or top-level execution")

    body = list(function.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        raise ValueError(f"`{attr}` must contain only one literal return")
    try:
        return ast.literal_eval(body[0].value)
    except (ValueError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(f"`{attr}` must return a scalar or literal mask list") from exc


def load_surface(sol_path: str, attr: str):
    """Return a trusted callable backed by a parsed literal value."""
    value = _literal_surface_value(sol_path, attr)

    def surface():
        return copy.deepcopy(value)

    return surface


def _require_finite_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise RuntimeError(f"{label} must be a non-empty tensor")
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{label} contains non-finite values")
    return value


def _require_finite_module(module, label: str) -> None:
    for name, parameter in module.named_parameters():
        _require_finite_tensor(parameter, f"{label} parameter {name!r}")


def _require_finite_gradients(module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            _require_finite_tensor(parameter.grad, f"gradient {name!r}")


# ---------------------------------------------------------------------------
# FIXED 2-D toy-target samples — loaded from pre-generated data, no sampler here.
# ---------------------------------------------------------------------------
# The exact closed-form sampler (cell sizes, radii, noise scales, rotation
# rates, means/covariances) lives host-side only in
# holdout/normflows-density/dgp.py, which is NEVER staged into the agent's
# workspace/image. This module only knows the PUBLIC target names and loads
# their frozen sample arrays from _flow_data/*.npz (shipped alongside this
# file, containing only sampled (x1, x2) points -- no sampler code, no
# parameters). We DO NOT need the analytic log-density of the target: NLL is
# computed under the *learned flow*, which is exact via change-of-variables.
# The reported held-out quantity is mean negative log likelihood.

# Valid target names with complete pinned train/test archives.
TARGETS = ("checkerboard", "moons", "pinwheel", "8gaussians")

# Training samples load from verifier-staged _flow_data/*.npz beside this file.
# Test samples load from the task's separate verifier-only data/ directory via
# TASK_DIR. Neither class of archive is present during the agent action phase.
_DATA_DIR = Path(__file__).resolve().parent / "_flow_data"

EXPECTED_TRAIN_FILE_SHA256 = {
    "checkerboard": "61a03c2d24a4a44eebbf61b7acd397b6df5834850889815aeaa4d4a3f1290ad4",
    "moons": "48006975d1b5b065a9492f865fcfef808c2370312c6cf822cc7f0a94efa8b87e",
    "pinwheel": "aec14ab3ed4d82f2976c369fdf53b51a1b600bbb9d32317702e0fc5670a88c84",
    "8gaussians": "54ec6fc49522ccea8526bd390037403034443fad7d484947ba3863538a51f332",
}
EXPECTED_TEST_FILE_SHA256 = {
    "checkerboard": "32137a42fad38a1363a3d0934ee3e46e6fbec3eca9ceca4d258b3b7e3d5b4f7f",
    "moons": "1e0bb490483a9992fc6f6c5321cdef6feb74c3e65a1e03a6099deac028024aac",
    "pinwheel": "adeca11290413579bdf541647debfbfdfb82b725d2e98e0fdb267dab4a1606db",
    "8gaussians": "78f79e5b8e87cf865e38c46d303cf6ee99d4dddb7608d575e3abb57f514bf4ed",
}


def _require_file_sha256(path: Path, expected: str, label: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"{label} digest mismatch: expected {expected}, got {digest}"
        )


def _test_data_path(target: str, seed: int) -> Path:
    """Resolve the verifier-only held-out test npz for `target`/`seed`.

    Only exists under $TASK_DIR/data (set by test.sh / score_task.py's
    _install_task_meta_legacy_links to the per-task verifier meta dir) during
    scoring; absent during the agent's action session.
    """
    task_dir = Path(os.environ.get("TASK_DIR", "/workspace/_task"))
    return task_dir / "data" / f"{target}_seed{seed}_test.npz"


def make_dataset(target: str, n_train: int, n_test: int, seed: int):
    """FIXED train/test samples for a target.

    Loads `train_x` from the frozen, verifier-staged
    ``_flow_data/<target>_seed<seed>.npz`` archive (produced host-side by
    ``holdout/normflows-density/generate_data.py`` from the held-out DGP)
    rather than sampling anything itself, and loads `test_x` from the
    verifier-only ``$TASK_DIR/data/<target>_seed<seed>_test.npz`` (see module
    docstring). `n_train`/`n_test` are validated against the actual sample
    counts so a stale or mismatched request fails loudly instead of silently
    substituting data of a different size.
    """
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target!r}; choose from {list(TARGETS)}")

    train_npz_path = _DATA_DIR / f"{target}_seed{seed}.npz"
    if not train_npz_path.exists():
        raise SystemExit(
            f"no frozen train data for target={target!r} seed={seed}; "
            f"expected {train_npz_path}. Regenerate with "
            f"holdout/normflows-density/generate_data.py (host-side only)."
        )
    if seed != 42 or target not in EXPECTED_TRAIN_FILE_SHA256:
        raise SystemExit(
            f"no pinned train digest for target={target!r} seed={seed}"
        )
    _require_file_sha256(
        train_npz_path, EXPECTED_TRAIN_FILE_SHA256[target], "training archive"
    )
    with np.load(train_npz_path, allow_pickle=False) as npz:
        if tuple(npz.files) != ("train_x",):
            raise RuntimeError("training archive must contain only train_x")
        train_array = np.array(npz["train_x"], copy=True)

    test_npz_path = _test_data_path(target, seed)
    if not test_npz_path.exists():
        raise SystemExit(
            f"no held-out test data for target={target!r} seed={seed}; "
            f"expected {test_npz_path}. This file is staged verifier-side "
            f"only (tests/meta/data, mounted at $TASK_DIR/data) and is not "
            f"available during an agent action session -- the held-out NLL "
            f"can only be computed at scoring time."
        )
    if seed != 42 or target not in EXPECTED_TEST_FILE_SHA256:
        raise SystemExit(
            f"no pinned test digest for target={target!r} seed={seed}"
        )
    _require_file_sha256(
        test_npz_path, EXPECTED_TEST_FILE_SHA256[target], "test archive"
    )
    with np.load(test_npz_path, allow_pickle=False) as npz:
        if tuple(npz.files) != ("test_x",):
            raise RuntimeError("test archive must contain only test_x")
        test_array = np.array(npz["test_x"], copy=True)

    if train_array.dtype != np.float32 or test_array.dtype != np.float32:
        raise RuntimeError("flow train/test arrays must use float32")
    x_tr = torch.from_numpy(train_array)
    x_te = torch.from_numpy(test_array)

    if x_tr.shape[0] != n_train or x_te.shape[0] != n_test:
        raise SystemExit(
            f"frozen data for target={target!r} seed={seed} has "
            f"{x_tr.shape[0]}/{x_te.shape[0]} train/test samples, "
            f"requested {n_train}/{n_test}"
        )
    for label, value in (("training data", x_tr), ("test data", x_te)):
        _require_finite_tensor(value, label)
        if value.ndim != 2 or value.shape[1] != DIM:
            raise RuntimeError(f"{label} must have shape [N, {DIM}]")
    return x_tr, x_te


# ---------------------------------------------------------------------------
# Fixed training loop + exact NLL evaluation
# ---------------------------------------------------------------------------

def train_and_eval(model, x_tr: torch.Tensor, x_te: torch.Tensor, *,
                   steps: int, batch_size: int, lr: float, seed: int,
                   weight_decay: float = 1e-5, log_every: int = 200):
    """FROZEN optimisation budget shared by all flow tasks.

    Minimises the exact forward KL (== mean NLL) of the flow on train samples
    with Adam for a fixed number of steps, then returns total mean test NLL in
    nats (bits/dim is reported separately). Uses `model.forward_kld` from
    normflows, which is the negative mean log-likelihood of the batch under the
    flow.
    """
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
    if not math.isfinite(float(lr)) or not 0.0 < float(lr) <= 1.0:
        raise ValueError("learning rate must be finite and in (0, 1]")
    _require_finite_tensor(x_tr, "training data")
    _require_finite_tensor(x_te, "test data")

    model = model.to(DEVICE)
    _require_finite_module(model, "flow model")
    x_tr = x_tr.to(DEVICE)
    x_te = x_te.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    n = x_tr.shape[0]
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        xb = x_tr[idx]
        opt.zero_grad(set_to_none=True)
        loss = model.forward_kld(xb)  # mean NLL over the batch
        if not torch.is_tensor(loss) or loss.numel() != 1 or not torch.isfinite(loss).item():
            raise RuntimeError(f"FLOW_NONFINITE training loss at step {step}")
        loss.backward()
        _require_finite_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(f"FLOW_NONFINITE gradient norm at step {step}")
        opt.step()
        _require_finite_module(model, "flow model")
        if step % log_every == 0 or step == steps - 1:
            print(f"FLOW_TRAIN step={step} train_nll={loss.item():.4f}", flush=True)

    # Exact held-out NLL via change-of-variables (mean over test points).
    model.eval()
    with torch.no_grad():
        nlls = []
        for i in range(0, x_te.shape[0], 4096):
            xb = x_te[i:i + 4096]
            lp = model.log_prob(xb)          # exact log p(x) under the flow
            _require_finite_tensor(lp, "flow test log probability")
            if lp.ndim != 1 or lp.shape[0] != xb.shape[0]:
                raise RuntimeError("flow log_prob must return one value per test sample")
            nlls.append((-lp).double())
        test_nll = torch.cat(nlls).mean().item()   # total NLL (nats, D=2)
    test_bpd = test_nll / (DIM * math.log(2.0))     # bits/dim, for reference
    if not math.isfinite(test_nll) or not math.isfinite(test_bpd):
        raise RuntimeError("FLOW_NONFINITE held-out metric")
    return float(test_nll), float(test_bpd)


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))
