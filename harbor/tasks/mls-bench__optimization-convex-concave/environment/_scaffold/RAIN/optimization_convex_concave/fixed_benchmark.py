"""Fixed benchmark harness matching the official RAIN convex-concave scripts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    dim: int
    tau: float
    sigma: float
    num_iterations: int
    delta: float = 0.0
    nu: float | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    official_seed: int = 42
    bilinear_iterations: int = 900
    delta_nu_iterations: int = 6000


@dataclass
class StepOutput:
    state: dict[str, Any]
    iterate: np.ndarray
    sfo_calls: int


DEFAULT_CONFIG = BenchmarkConfig()


def as_vector(value: Any, expected_dim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if expected_dim is not None and array.shape[0] != expected_dim:
        raise ValueError(f"Expected vector of length {expected_dim}, got {array.shape[0]}.")
    return array.copy()


def make_step_output(state: dict[str, Any], iterate: np.ndarray, sfo_calls: int) -> StepOutput:
    return StepOutput(state=state, iterate=as_vector(iterate), sfo_calls=int(sfo_calls))


def split_xy(z: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    z = as_vector(z, expected_dim=2 * dim)
    return z[:dim], z[dim:]


def concat_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.concatenate([as_vector(x), as_vector(y)])


def _problem_specs(config: BenchmarkConfig, sigma_scale: float = 1.0) -> tuple[ProblemSpec, ...]:
    return (
        ProblemSpec(name="bilinear", dim=1, tau=0.1, sigma=0.001 * sigma_scale, num_iterations=config.bilinear_iterations),
        ProblemSpec(
            name="delta_nu",
            dim=100,
            tau=1.0,
            sigma=0.02 * sigma_scale,
            num_iterations=config.delta_nu_iterations,
            delta=1e-2,
            nu=5e-5,
        ),
    )


def _sub_grad(eps: float, u: np.ndarray) -> np.ndarray:
    """Mirror the repository's MATLAB sub_grad implementation exactly."""
    g = as_vector(u)
    g[g > eps] = eps
    g[g < eps] = -eps
    return g


def full_grad(problem: ProblemSpec, z: np.ndarray) -> np.ndarray:
    x, y = split_xy(z, problem.dim)
    if problem.name == "bilinear":
        return concat_xy(y, -x)
    if problem.name == "delta_nu":
        if problem.nu is None:
            raise ValueError("delta_nu requires nu.")
        gx = (1.0 - problem.delta) * _sub_grad(problem.nu, x) + problem.delta * y
        gy = (1.0 - problem.delta) * _sub_grad(problem.nu, y) - problem.delta * x
        return concat_xy(gx, gy)
    raise KeyError(f"Unknown problem: {problem.name}")


def gradient_norm(problem: ProblemSpec, z: np.ndarray) -> float:
    return float(np.linalg.norm(full_grad(problem, z)))


def initial_point(problem: ProblemSpec, rng: np.random.RandomState) -> np.ndarray:
    if problem.name == "bilinear":
        return np.array([10.0, 10.0], dtype=np.float64)
    if problem.name == "delta_nu":
        return rng.randn(2 * problem.dim).astype(np.float64)
    raise KeyError(f"Unknown problem: {problem.name}")


class StochasticOracle:
    """Official-script oracle: deterministic gradient + additive Gaussian update noise."""

    def __init__(self, problem: ProblemSpec, rng: np.random.RandomState) -> None:
        self.problem = problem
        self.rng = rng
        self.grad_calls = 0
        self.noise_calls = 0

    def grad(self, z: np.ndarray) -> np.ndarray:
        self.grad_calls += 1
        return full_grad(self.problem, z)

    def noise(self) -> np.ndarray:
        self.noise_calls += 1
        return self.rng.normal(loc=0.0, scale=self.problem.sigma, size=2 * self.problem.dim)


def _validate_initial_state(state: Any, initial_z: np.ndarray, problem: ProblemSpec) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise TypeError("init_state(...) must return a dict.")
    if "z" not in state:
        raise KeyError('init_state(...) must include state["z"].')
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    if not np.allclose(z, initial_z):
        raise ValueError('init_state(...) must preserve the provided starting point in state["z"].')
    normalized = dict(state)
    normalized["z"] = z
    return normalized


def _validate_step_output(
    output: Any,
    before_grad_calls: int,
    oracle: StochasticOracle,
    problem: ProblemSpec,
) -> StepOutput:
    if not isinstance(output, StepOutput):
        raise TypeError("step(...) must return StepOutput.")
    if not isinstance(output.state, dict):
        raise TypeError("StepOutput.state must be a dict.")
    if "z" not in output.state:
        raise KeyError('StepOutput.state must include state["z"].')
    state = dict(output.state)
    state["z"] = as_vector(state["z"], expected_dim=2 * problem.dim)
    iterate = as_vector(output.iterate, expected_dim=2 * problem.dim)
    actual_calls = oracle.grad_calls - before_grad_calls
    if actual_calls <= 0:
        raise ValueError("step(...) must consume at least one gradient call.")
    if output.sfo_calls != actual_calls:
        raise ValueError(
            f"step(...) reported sfo_calls={output.sfo_calls}, but consumed {actual_calls} gradient calls."
        )
    return StepOutput(state=state, iterate=iterate, sfo_calls=actual_calls)


def _log_auc_log_iteration_log_grad(iterations: np.ndarray, gradients: np.ndarray) -> float:
    x = np.log10(iterations.astype(np.float64))
    y = np.log10(np.maximum(gradients.astype(np.float64), 1e-12))
    return float(np.trapezoid(y, x))


def _selected_log_iterations(problem: ProblemSpec) -> set[int]:
    if problem.name == "bilinear":
        return {1, 10, 50, 100, 300, 600, problem.num_iterations}
    return {1, 10, 100, 500, 1000, 3000, problem.num_iterations}


def _maybe_apply_smoke_mode(config: BenchmarkConfig, enabled: bool) -> BenchmarkConfig:
    if not enabled:
        return config
    return replace(config, bilinear_iterations=120, delta_nu_iterations=800)


def _run_problem(
    problem: ProblemSpec,
    config: BenchmarkConfig,
    init_state_fn: Callable[[ProblemSpec, np.ndarray, int, dict[str, Any]], dict[str, Any]],
    step_fn: Callable[[dict[str, Any], StochasticOracle, ProblemSpec, dict[str, Any], int], StepOutput],
    get_hparams_fn: Callable[[str, float], dict[str, Any]],
) -> dict[str, Any]:
    hyperparameters = dict(get_hparams_fn(problem.name, problem.sigma))
    rng = np.random.RandomState(config.official_seed)
    initial_z = initial_point(problem, rng)
    oracle = StochasticOracle(problem=problem, rng=rng)

    state = _validate_initial_state(
        init_state_fn(problem, initial_z.copy(), config.official_seed, hyperparameters),
        initial_z=initial_z,
        problem=problem,
    )

    gradient_values: list[float] = []
    records: list[dict[str, Any]] = []
    log_iterations = _selected_log_iterations(problem)

    for iteration in range(1, problem.num_iterations + 1):
        before_grad_calls = oracle.grad_calls
        output = _validate_step_output(
            step_fn(
                state=state,
                oracle=oracle,
                problem=problem,
                hyperparameters=hyperparameters,
                max_sfo_calls=2,
            ),
            before_grad_calls=before_grad_calls,
            oracle=oracle,
            problem=problem,
        )
        state = output.state
        gnorm = gradient_norm(problem, output.iterate)
        gradient_values.append(gnorm)
        record = {
            "problem": problem.name,
            "iteration": iteration,
            "sfo_calls": int(oracle.grad_calls),
            "gradient_norm": gnorm,
        }
        records.append(record)
        if iteration in log_iterations:
            print(
                "STEP_METRICS "
                f"problem={problem.name} iteration={iteration} sfo_calls={oracle.grad_calls} "
                f"gradient_norm={gnorm:.6f}",
                flush=True,
            )

    iterations = np.arange(1, problem.num_iterations + 1, dtype=np.float64)
    gradients = np.asarray(gradient_values, dtype=np.float64)
    final_gradient = float(gradients[-1])
    auc = _log_auc_log_iteration_log_grad(iterations, gradients)
    print(
        "RUN_METRICS "
        f"problem={problem.name} final_gradient_norm={final_gradient:.6f} "
        f"auc_log_iteration_log_grad={auc:.6f} total_sfo_calls={oracle.grad_calls}",
        flush=True,
    )
    return {
        "problem": problem.name,
        "hyperparameters": hyperparameters,
        "records": records,
        "final_gradient_norm": final_gradient,
        "auc_log_iteration_log_grad": auc,
        "total_sfo_calls": int(oracle.grad_calls),
    }


def _run_benchmark(
    config: BenchmarkConfig,
    init_state_fn: Callable[[ProblemSpec, np.ndarray, int, dict[str, Any]], dict[str, Any]],
    step_fn: Callable[[dict[str, Any], StochasticOracle, ProblemSpec, dict[str, Any], int], StepOutput],
    get_hparams_fn: Callable[[str, float], dict[str, Any]],
    sigma_scale: float = 1.0,
) -> dict[str, Any]:
    print(
        "TASK_CONFIG "
        f"official_seed={config.official_seed} bilinear_iterations={config.bilinear_iterations} "
        f"delta_nu_iterations={config.delta_nu_iterations} sigma_scale={sigma_scale}",
        flush=True,
    )

    runs = [
        _run_problem(problem, config, init_state_fn, step_fn, get_hparams_fn)
        for problem in _problem_specs(config, sigma_scale=sigma_scale)
    ]
    metrics = {
        "final_gradient_norm": float(np.mean([run["final_gradient_norm"] for run in runs])),
        "score": float(-np.mean([run["final_gradient_norm"] for run in runs])),
        "auc_log_iteration_log_grad": float(np.mean([run["auc_log_iteration_log_grad"] for run in runs])),
        "bilinear_final_gradient_norm": float(next(run["final_gradient_norm"] for run in runs if run["problem"] == "bilinear")),
        "delta_nu_final_gradient_norm": float(next(run["final_gradient_norm"] for run in runs if run["problem"] == "delta_nu")),
        "num_runs": int(len(runs)),
    }
    print(
        "FINAL_METRICS "
        + " ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in metrics.items()
        ),
        flush=True,
    )
    return {
        "config": asdict(config),
        "metrics": metrics,
        "runs": runs,
    }


def run_cli(
    init_state: Callable[[ProblemSpec, np.ndarray, int, dict[str, Any]], dict[str, Any]],
    step: Callable[[dict[str, Any], StochasticOracle, ProblemSpec, dict[str, Any], int], StepOutput],
    get_hyperparameters: Callable[[str, float], dict[str, Any]],
) -> None:
    parser = argparse.ArgumentParser(description="Run the MLS-Bench optimization-convex-concave task.")
    parser.add_argument("--seed", type=int, default=42, help="Ignored for the strict official benchmark; kept for CLI compatibility.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for a JSON summary.")
    parser.add_argument("--label", type=str, default="eval", help="Optional label stored in the JSON summary.")
    parser.add_argument("--smoke", action="store_true", help="Run a shorter local sanity check.")
    parser.add_argument("--sigma-scale", type=float, default=1.0, help="Multiplicative factor applied to each problem's base sigma.")
    args = parser.parse_args()

    config = _maybe_apply_smoke_mode(DEFAULT_CONFIG, args.smoke)
    summary = _run_benchmark(
        config=config,
        init_state_fn=init_state,
        step_fn=step,
        get_hparams_fn=get_hyperparameters,
        sigma_scale=args.sigma_scale,
    )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{args.label}_seed{args.seed}.json"
        output_path.write_text(json.dumps(summary, indent=2))
