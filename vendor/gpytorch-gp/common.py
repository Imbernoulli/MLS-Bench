"""Shared helpers for the gpytorch-gp (gp-*) MLS-Bench tasks.

Fixed data loading, seeding, standardization, and test-NLL / RMSE scoring so every
task's harness scores identical held-out points regardless of the agent's design.

Datasets are pre-staged as checksum-bound, versioned .npz splits under $GP_DATA
(see vendor/data_scripts/gpytorch-gp/prepare_data.py). The harness rejects stale,
partial, wrong-provenance, or non-finite caches. It standardizes x and y using
training statistics only; NLL and RMSE are reported on the original y scale.

Agent solution files contain one finite JSON literal. They are parsed with a
bounded AST and are never imported or executed. Trusted builders below construct
every GPyTorch module used by the metric-producing process.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch


def install_strict_numerical_warnings() -> None:
    """Turn linear-solver non-convergence into a verifier failure.

    GPyTorch otherwise emits ``NumericalWarning`` and continues with an
    unconverged CG result.  Such a result is not an authoritative metric, so
    every metric-producing harness installs this filter before doing any work.
    """
    categories = []
    try:
        from linear_operator.utils.warnings import NumericalWarning as LinearOperatorWarning

        categories.append(LinearOperatorWarning)
    except ImportError:
        pass
    try:
        from gpytorch.utils.warnings import NumericalWarning as GPyTorchWarning

        categories.append(GPyTorchWarning)
    except ImportError:
        pass

    if not categories:
        raise RuntimeError("GPyTorch exposes no NumericalWarning class")
    for category in dict.fromkeys(categories):
        warnings.filterwarnings("error", category=category)


@contextmanager
def exact_gp_numerics(dataset: str):
    """Use reproducible, converged linear algebra for scalable ExactGPs.

    GPyTorch's large-matrix defaults use a training CG tolerance of 1 and only
    1,000 evaluation iterations. Those defaults are suitable for quick
    exploration but produced hardware-dependent anchors here. The fixed
    verifier protocol uses deterministic log-determinant probes, explicit
    train/eval tolerances, and a measured per-dataset pivoted-Cholesky rank.
    Kin8nm requires rank 500 to converge at the fixed tolerance; Concrete and
    Elevators remain stable at rank 100, while larger ranks destabilize the
    Elevators solve under the same tolerance.
    """
    import gpytorch

    preconditioner_sizes = {
        "concrete": 100,
        "kin8nm": 500,
        "elevators": 100,
    }
    if dataset not in preconditioner_sizes:
        raise ValueError(f"unknown ExactGP numerical protocol for {dataset!r}")

    with ExitStack() as stack:
        stack.enter_context(gpytorch.settings.cg_tolerance(0.01))
        stack.enter_context(gpytorch.settings.eval_cg_tolerance(0.001))
        stack.enter_context(gpytorch.settings.max_cg_iterations(10_000))
        stack.enter_context(
            gpytorch.settings.max_preconditioner_size(
                preconditioner_sizes[dataset]
            )
        )
        stack.enter_context(gpytorch.settings.num_trace_samples(32))
        stack.enter_context(gpytorch.settings.deterministic_probes(True))
        yield


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("the full GP protocol requires one visible CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "the full GP protocol requires exactly one visible CUDA GPU"
        )
    return torch.device("cuda")


_SURFACE_SOURCE_BYTES = 64 * 1024
_SURFACE_AST_NODES = 512
_DATA_FORMAT_VERSION = 2
_SPLIT_PROTOCOL = "openml-full-random90-10-seed42-v1"
_METRIC_PROTOCOL = "openml_full_v2"
_DATASET_CONTRACTS = {
    "concrete": {
        "data_id": 4353,
        "rows": 1030,
        "dim": 8,
        "source_sha256": "092c87480aff2080e039c1c1ef9d87e6f5748352549682875a5834eb52a8bfde",
        "split_sha256": "c6cb20776e4eebaee665fea5d9a5a688db19c6699219dfcac2a258e7977f1773",
    },
    "kin8nm": {
        "data_id": 189,
        "rows": 8192,
        "dim": 8,
        "source_sha256": "970265a06441ea9ed0524a75c28a60880c3e76fbac1a5a374d090b848b6156bd",
        "split_sha256": "7b0680527b8b8835c300c183a81fa92c6b7f64ce047db6b574d9f63aa02d9fd0",
    },
    "elevators": {
        "data_id": 216,
        "rows": 16599,
        "dim": 18,
        "source_sha256": "9e68b4c7838d3c2c998d5ae96f5aa033b722a7fad2a581b7a9511a73f05d0a39",
        "split_sha256": "2d852b51f9424cf235786e4402aa7a80cc0676cee68b0eafcfaf8e9c71e46c2c",
    },
}

_RUN_CONTRACTS = {
    "gp-ard-lengthscale": ("ard", "iterations", 200),
    "gp-deep-kernel": ("deep_kernel", "iterations", 200),
    "gp-deep-kernel-width": ("deep_kernel_width", "iterations", 200),
    "gp-exact-lr": ("exact_lr", "iterations", 200),
    "gp-kernel-design": ("kernel_design", "iterations", 200),
    "gp-kernel-smoothness": ("smoothness", "iterations", 200),
    "gp-likelihood-noise": ("likelihood_noise", "iterations", 200),
    "gp-mean-function": ("mean_function", "iterations", 200),
    "gp-sparse-inducing": ("inducing", "epochs", 20),
    "gp-svgp-lr": ("svgp_lr", "epochs", 20),
}


def load_surface_config(path: str) -> dict:
    """Parse one finite JSON literal without executing agent-authored Python."""
    solution_path = Path(path).resolve()
    if not solution_path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {solution_path}")
    try:
        source = solution_path.read_text()
        if len(source.encode()) > _SURFACE_SOURCE_BYTES:
            raise ValueError("solution configuration exceeds the 64 KiB source limit")
        tree = ast.parse(source, filename=str(solution_path))
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
        raise ValueError("surface_config body must contain exactly one return statement")

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


def _require_keys(config: dict, required: set[str]) -> None:
    if not isinstance(config, dict):
        raise TypeError("surface configuration must be a JSON object")
    if set(config) != required:
        raise ValueError(
            f"surface configuration requires exactly {sorted(required)}, got {sorted(config)}"
        )


def _finite_number(value, label: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{label} must be finite and in [{lower}, {upper}]")
    return value


def _bounded_int(value, label: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{label} must be in [{lower}, {upper}]")
    return value


def validate_surface_config(surface: str, config: dict) -> dict:
    """Validate that a literal plan controls only the task's declared GP axis."""
    if surface == "ard":
        _require_keys(config, {"ard"})
        if not isinstance(config["ard"], bool):
            raise TypeError("ard must be a boolean")
    elif surface == "smoothness":
        _require_keys(config, {"kernel"})
        if config["kernel"] not in {"rbf", "matern12", "matern52"}:
            raise ValueError("kernel must be rbf, matern12, or matern52")
    elif surface == "kernel_design":
        kernel = config.get("kernel")
        required = {
            "rbf": {"kernel", "ard", "mean"},
            "matern": {"kernel", "ard", "nu", "mean"},
            "spectral_mixture": {
                "kernel", "ard", "num_mixtures", "mean",
            },
        }.get(kernel)
        if required is None:
            raise ValueError("kernel must be rbf, matern, or spectral_mixture")
        _require_keys(config, required)
        if not isinstance(config["ard"], bool):
            raise TypeError("ard must be a boolean")
        if config["mean"] not in {"zero", "constant", "linear"}:
            raise ValueError("mean must be zero, constant, or linear")
        if kernel == "matern":
            if config["nu"] not in {0.5, 1.5, 2.5}:
                raise ValueError("Matern nu must be 0.5, 1.5, or 2.5")
        elif kernel == "spectral_mixture":
            _bounded_int(config["num_mixtures"], "num_mixtures", 1, 8)
            if not config["ard"]:
                raise ValueError(
                    "spectral_mixture requires ard=True for multidimensional inputs"
                )
    elif surface == "mean_function":
        _require_keys(config, {"mean"})
        if config["mean"] not in {"zero", "constant", "linear"}:
            raise ValueError("mean must be zero, constant, or linear")
    elif surface == "likelihood_noise":
        mode = config.get("mode")
        required = {"mode"} if mode == "learned" else {"mode", "noise"}
        _require_keys(config, required)
        if mode not in {"learned", "fixed"}:
            raise ValueError("mode must be learned or fixed")
        if mode == "fixed":
            _finite_number(config["noise"], "fixed noise", 1e-6, 1.0)
    elif surface in {"exact_lr", "svgp_lr"}:
        _require_keys(config, {"learning_rate"})
        validate_learning_rate(config["learning_rate"])
    elif surface == "deep_kernel_width":
        _require_keys(config, {"width"})
        _bounded_int(config["width"], "feature width", 1, 32)
    elif surface == "deep_kernel":
        _require_keys(config, {"extractor"})
        if config["extractor"] not in {"identity", "mlp"}:
            raise ValueError("extractor must be identity or mlp")
    elif surface == "inducing":
        _require_keys(config, {"method", "count"})
        if config["method"] not in {"random", "kmeans"}:
            raise ValueError("inducing method must be random or kmeans")
        _bounded_int(config["count"], "inducing count", 1, 2048)
    else:
        raise ValueError(f"unknown GP surface {surface!r}")
    return config


def build_mean_from_config(config: dict, d: int):
    import gpytorch

    choice = config["mean"]
    if choice == "zero":
        return gpytorch.means.ZeroMean()
    if choice == "constant":
        return gpytorch.means.ConstantMean()
    return gpytorch.means.LinearMean(input_size=d)


def build_kernel_design(config: dict, train_x: torch.Tensor, train_y: torch.Tensor, d: int):
    import gpytorch

    validate_surface_config("kernel_design", config)
    ard_dims = d if config["ard"] else None
    if config["kernel"] == "rbf":
        covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        )
    elif config["kernel"] == "matern":
        covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=config["nu"], ard_num_dims=ard_dims)
        )
    else:
        covar = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=config["num_mixtures"],
            ard_num_dims=d,
        ).to(device=train_x.device, dtype=train_x.dtype)
        covar.initialize_from_data(train_x, train_y)
    mean = build_mean_from_config(config, d)
    require_finite_module(covar, "covariance")
    require_finite_module(mean, "mean")
    return covar, mean


def build_likelihood_from_config(config: dict):
    import gpytorch

    validate_surface_config("likelihood_noise", config)
    if config["mode"] == "fixed":
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
        )
        likelihood.noise = float(config["noise"])
        likelihood.raw_noise.requires_grad_(False)
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    require_finite_module(likelihood, "likelihood")
    return likelihood


def build_feature_extractor(config: dict, d: int):
    import torch.nn as nn

    validate_surface_config("deep_kernel", config)
    if config["extractor"] == "identity":
        class Identity(nn.Module):
            out_features = d

            def forward(self, value):
                return value

        return Identity(), d

    module = nn.Sequential(
        nn.Linear(d, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    module.out_features = 4
    require_finite_module(module, "feature extractor")
    return module, module.out_features


def select_inducing_from_config(config: dict, train_x: torch.Tensor) -> torch.Tensor:
    validate_surface_config("inducing", config)
    count = min(int(config["count"]), train_x.size(0))
    if config["method"] == "random":
        indices = torch.randperm(train_x.size(0), device=train_x.device)[:count]
        return train_x[indices].clone()

    from sklearn.cluster import KMeans

    clusters = KMeans(n_clusters=count, n_init=3, random_state=42).fit(
        train_x.detach().cpu().numpy()
    )
    return torch.as_tensor(
        clusters.cluster_centers_, dtype=train_x.dtype, device=train_x.device
    )


def validate_run_contract(
    task: str,
    surface: str,
    dataset: str,
    budget_kind: str,
    budget: int,
    seed: int,
) -> None:
    """Reject any task/surface/data/budget combination outside the fixed protocol."""
    expected = _RUN_CONTRACTS.get(task)
    if expected is None:
        raise ValueError(f"unknown GP task {task!r}")
    if (surface, budget_kind, budget) != expected:
        raise ValueError(
            f"GP task {task!r} requires surface/budget {expected!r}, got "
            f"{(surface, budget_kind, budget)!r}"
        )
    if dataset not in _DATASET_CONTRACTS:
        raise ValueError(f"unknown GP dataset {dataset!r}")
    if seed != 42:
        raise ValueError("the full GP protocol requires seed 42")
    device()


def _validate_terminal_inventory(
    dataset: str,
    n_train: int,
    n_test: int,
    batch_size: int,
    updates: int,
) -> dict:
    contract = _DATASET_CONTRACTS[dataset]
    expected_test = int(round(contract["rows"] * 0.1))
    expected_train = contract["rows"] - expected_test
    if (n_train, n_test) != (expected_train, expected_test):
        raise RuntimeError("GP terminal proof has an incomplete dataset split")
    _bounded_int(batch_size, "training batch size", 1, 1_000_000)
    _bounded_int(updates, "completed optimizer updates", 1, 10_000_000)
    return contract


def format_completion_line(
    task: str,
    surface: str,
    dataset: str,
    n_train: int,
    n_test: int,
    budget_kind: str,
    budget: int,
    batch_size: int,
    updates: int,
    seed: int,
) -> str:
    """Bind terminal completion to task identity, split, GPU, and fixed budget."""
    validate_run_contract(task, surface, dataset, budget_kind, budget, seed)
    contract = _validate_terminal_inventory(
        dataset, n_train, n_test, batch_size, updates
    )
    return (
        f"GP_COMPLETE protocol={_METRIC_PROTOCOL} task={task} surface={surface} "
        f"dataset={dataset} seed={seed} device=cuda "
        f"split_sha256={contract['split_sha256']} n_train={n_train} n_test={n_test} "
        f"budget_kind={budget_kind} budget={budget} batch_size={batch_size} "
        f"updates={updates}"
    )


def format_metric_line(
    task: str,
    surface: str,
    dataset: str,
    n_train: int,
    n_test: int,
    budget_kind: str,
    budget: int,
    batch_size: int,
    updates: int,
    seed: int,
    nll: float,
    rmse: float,
    elapsed: float,
) -> str:
    """Bind metrics to task identity, split, GPU, and completed fixed budget."""
    validate_run_contract(task, surface, dataset, budget_kind, budget, seed)
    contract = _validate_terminal_inventory(
        dataset, n_train, n_test, batch_size, updates
    )
    for label, value in (("nll", nll), ("rmse", rmse), ("elapsed", elapsed)):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"cannot emit non-finite GP {label}")
    if rmse < 0 or elapsed <= 0:
        raise RuntimeError("GP RMSE must be non-negative and elapsed time positive")
    return (
        f"GP_METRICS protocol={_METRIC_PROTOCOL} task={task} surface={surface} "
        f"dataset={dataset} seed={seed} device=cuda "
        f"split_sha256={contract['split_sha256']} n_train={n_train} n_test={n_test} "
        f"budget_kind={budget_kind} budget={budget} batch_size={batch_size} "
        f"updates={updates} nll={nll:.6f} rmse={rmse:.6f} elapsed={elapsed:.1f}"
    )


def require_finite_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise RuntimeError(f"{label} must be a non-empty tensor")
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{label} contains non-finite values")
    return value


def require_finite_module(module, label: str) -> None:
    for name, parameter in module.named_parameters():
        require_finite_tensor(parameter, f"{label} parameter {name!r}")


def require_finite_gradients(module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            require_finite_tensor(parameter.grad, f"gradient {name!r}")


def validate_learning_rate(value, label: str = "learning rate") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{label} must be finite and in (0, 1]")
    return value


class Standardizer:
    """Zero-mean / unit-var standardization fit on train, applied to train+test."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_mean = x.mean(0, keepdims=True)
        self.x_std = x.std(0, keepdims=True)
        self.x_std[self.x_std < 1e-8] = 1.0
        self.y_mean = float(y.mean())
        self.y_std = float(y.std())
        if self.y_std < 1e-8:
            self.y_std = 1.0

    def tx(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def ty(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std


def _split_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for label in ("train_x", "train_y", "test_x", "test_y"):
        value = np.ascontiguousarray(arrays[label], dtype="<f4")
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(str(value.shape).encode())
        digest.update(b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_split(name: str, data_root: str | None = None):
    """Load a fixed regression split, standardize, return tensors + standardizer.

    Returns (train_x, train_y, test_x, test_y, std, d) as float32 tensors on the
    active device. x standardized by train stats; y standardized by train stats.
    """
    if name not in _DATASET_CONTRACTS:
        raise ValueError(f"unknown GP dataset {name!r}")
    root = Path(data_root or os.environ.get("GP_DATA", "/data/gpytorch-gp"))
    path = root / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"dataset split not found: {path}")
    required = {
        "train_x", "train_y", "test_x", "test_y", "format_version",
        "split_protocol", "dataset_name", "openml_data_id", "source_sha256",
        "split_sha256", "split_seed", "total_rows", "feature_dim",
    }
    try:
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != required:
                raise ValueError(
                    f"dataset cache keys mismatch: {sorted(cache.files)} != {sorted(required)}"
                )
            metadata = {
                key: cache[key].item()
                for key in required - {"train_x", "train_y", "test_x", "test_y"}
            }
            train_x = cache["train_x"].astype(np.float64)
            train_y = cache["train_y"].astype(np.float64)
            test_x = cache["test_x"].astype(np.float64)
            test_y = cache["test_y"].astype(np.float64)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot load a current GP dataset cache from {path}: {exc}") from exc

    contract = _DATASET_CONTRACTS[name]
    expected_metadata = {
        "format_version": _DATA_FORMAT_VERSION,
        "split_protocol": _SPLIT_PROTOCOL,
        "dataset_name": name,
        "openml_data_id": contract["data_id"],
        "source_sha256": contract["source_sha256"],
        "split_sha256": contract["split_sha256"],
        "split_seed": 42,
        "total_rows": contract["rows"],
        "feature_dim": contract["dim"],
    }
    if metadata != expected_metadata:
        raise ValueError(
            f"GP dataset provenance mismatch for {name}: {metadata} != {expected_metadata}"
        )
    arrays = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }
    for label, value in arrays.items():
        if value.size == 0 or not np.isfinite(value).all():
            raise RuntimeError(f"dataset array {label} is empty or non-finite")
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise RuntimeError("train/test features must be compatible rank-2 arrays")
    if train_y.ndim != 1 or test_y.ndim != 1:
        raise RuntimeError("train/test targets must be rank-1 arrays")
    if train_x.shape[0] != train_y.shape[0] or test_x.shape[0] != test_y.shape[0]:
        raise RuntimeError("feature and target row counts do not match")
    expected_test = int(round(contract["rows"] * 0.1))
    expected_train = contract["rows"] - expected_test
    if train_x.shape != (expected_train, contract["dim"]):
        raise RuntimeError(f"{name}: training split shape violates the full-data contract")
    if test_x.shape != (expected_test, contract["dim"]):
        raise RuntimeError(f"{name}: test split shape violates the full-data contract")
    observed_split_sha256 = _split_sha256(arrays)
    if observed_split_sha256 != contract["split_sha256"]:
        raise RuntimeError(
            f"{name}: split checksum mismatch: {observed_split_sha256} != "
            f"{contract['split_sha256']}"
        )

    std = Standardizer(train_x, train_y)
    dev = device()

    def _t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=dev)

    tr_x = _t(std.tx(train_x))
    tr_y = _t(std.ty(train_y))
    te_x = _t(std.tx(test_x))
    te_y = _t(std.ty(test_y))
    return tr_x, tr_y, te_x, te_y, std, train_x.shape[1]


def score(pred_dist, test_y_std: torch.Tensor, std: Standardizer):
    """Compute test NLL (per point, ORIGINAL y scale) and RMSE (original scale).

    ``pred_dist`` is the predictive distribution INCLUDING observation noise
    (i.e. ``likelihood(model(test_x))``), evaluated on standardized test x, whose
    mean/variance are in standardized-y units.
    """
    mean_std = require_finite_tensor(pred_dist.mean, "predictive mean")
    var_std = require_finite_tensor(pred_dist.variance, "predictive variance")
    require_finite_tensor(test_y_std, "test targets")
    if mean_std.shape != test_y_std.shape or var_std.shape != test_y_std.shape:
        raise RuntimeError("predictive mean, variance, and targets must have identical shapes")
    if not torch.all(var_std > 0).item():
        raise RuntimeError("predictive variance must be strictly positive")

    y_std_scale = std.y_std
    # de-standardize
    mean = mean_std * y_std_scale + std.y_mean
    var = var_std * (y_std_scale ** 2)
    y = test_y_std * y_std_scale + std.y_mean

    # Gaussian NLL per point on the original scale (lower is better)
    nll = 0.5 * (torch.log(2 * np.pi * var) + (y - mean) ** 2 / var)
    nll = float(nll.mean().item())
    rmse = float(torch.sqrt(((mean - y) ** 2).mean()).item())
    if not math.isfinite(nll) or not math.isfinite(rmse):
        raise RuntimeError("GP evaluation produced non-finite metrics")
    return nll, rmse
