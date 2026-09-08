"""
Hyperparameter Optimization — Custom Strategy Template

This script runs a complete HPO loop on real ML model tuning benchmarks.
The agent should implement CustomHPOStrategy which proposes hyperparameter
configurations to evaluate, given a search space and history of past trials.

Usage:
    python scikit-learn/custom_hpo.py --benchmark xgboost --seed 42 \
        --budget 50 --output-dir ./out
"""

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm as scipy_norm

from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

# ================================================================
# FIXED — Data types and search space definitions (do not modify)
# ================================================================


@dataclass
class HParam:
    """A single hyperparameter specification."""
    name: str
    type: str  # "float", "int", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    log_scale: bool = False
    choices: Optional[list] = None


@dataclass
class Trial:
    """Record of one evaluated configuration."""
    config: Dict[str, Any]
    score: float  # validation score (higher is better)
    budget: float = 1.0  # fidelity/budget fraction (1.0 = full)


@dataclass
class SearchSpace:
    """Hyperparameter search space."""
    params: List[HParam] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.params)

    def sample_uniform(self, rng: np.random.RandomState) -> Dict[str, Any]:
        """Sample a random configuration uniformly from the space."""
        config = {}
        for p in self.params:
            if p.type == "categorical":
                config[p.name] = rng.choice(p.choices)
            elif p.type == "float":
                if p.log_scale:
                    log_val = rng.uniform(np.log(p.low), np.log(p.high))
                    config[p.name] = float(np.exp(log_val))
                else:
                    config[p.name] = float(rng.uniform(p.low, p.high))
            elif p.type == "int":
                if p.log_scale:
                    log_val = rng.uniform(np.log(p.low), np.log(p.high))
                    config[p.name] = int(round(np.exp(log_val)))
                else:
                    config[p.name] = int(rng.randint(p.low, p.high + 1))
        return config

    def clip(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Clip configuration values to valid ranges."""
        clipped = {}
        for p in self.params:
            val = config.get(p.name)
            if val is None:
                raise ValueError(f"Missing hyperparameter: {p.name}")
            if p.type == "categorical":
                if val not in p.choices:
                    raise ValueError(f"{p.name}={val} not in {p.choices}")
                clipped[p.name] = val
            elif p.type == "float":
                clipped[p.name] = float(np.clip(val, p.low, p.high))
            elif p.type == "int":
                clipped[p.name] = int(np.clip(round(val), p.low, p.high))
        return clipped


# ================================================================
# FIXED — Benchmark problems (do not modify)
# ================================================================


def _make_xgboost_benchmark():
    """XGBoost hyperparameter tuning on California Housing (regression).

    Search space: n_estimators, max_depth, learning_rate, subsample,
                  min_samples_split, min_samples_leaf.
    Metric: neg_mean_squared_error (converted to positive = higher is better).
    """
    data = fetch_california_housing(data_home=os.environ.get("SKLEARN_DATA_HOME"))
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    space = SearchSpace(params=[
        HParam("n_estimators", "int", low=50, high=500),
        HParam("max_depth", "int", low=2, high=10),
        HParam("learning_rate", "float", low=0.001, high=0.5, log_scale=True),
        HParam("subsample", "float", low=0.5, high=1.0),
        HParam("min_samples_split", "int", low=2, high=20),
        HParam("min_samples_leaf", "int", low=1, high=10),
    ])

    def objective(config: Dict[str, Any], budget: float = 1.0) -> float:
        n_est = config["n_estimators"]
        if budget < 1.0:
            n_est = max(10, int(n_est * budget))
        model = GradientBoostingRegressor(
            n_estimators=n_est,
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=0,
        )
        scores = cross_val_score(model, X, y, cv=3,
                                 scoring="neg_mean_squared_error")
        return float(scores.mean())  # negative MSE, higher is better

    return space, objective


def _make_svm_benchmark():
    """SVM hyperparameter tuning on Breast Cancer (classification).

    Search space: C, gamma, kernel.
    Metric: accuracy (higher is better).
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    space = SearchSpace(params=[
        HParam("C", "float", low=0.001, high=100.0, log_scale=True),
        HParam("gamma", "float", low=1e-5, high=10.0, log_scale=True),
        HParam("kernel", "categorical", choices=["rbf", "poly", "sigmoid"]),
    ])

    def objective(config: Dict[str, Any], budget: float = 1.0) -> float:
        cv_folds = max(2, int(5 * budget))
        model = SVC(
            C=config["C"],
            gamma=config["gamma"],
            kernel=config["kernel"],
            random_state=0,
        )
        scores = cross_val_score(model, X, y, cv=cv_folds,
                                 scoring="accuracy")
        return float(scores.mean())

    return space, objective


def _make_nn_benchmark():
    """Small neural network hyperparameter tuning on Diabetes (regression).

    Search space: hidden_layer_1, hidden_layer_2, learning_rate_init, alpha,
                  batch_size, activation.
    Metric: neg_mean_squared_error (converted to positive = higher is better).
    """
    data = load_diabetes()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    space = SearchSpace(params=[
        HParam("hidden_layer_1", "int", low=16, high=256, log_scale=True),
        HParam("hidden_layer_2", "int", low=8, high=128, log_scale=True),
        HParam("learning_rate_init", "float", low=1e-4, high=0.1,
               log_scale=True),
        HParam("alpha", "float", low=1e-6, high=0.1, log_scale=True),
        HParam("batch_size", "int", low=16, high=128),
        HParam("activation", "categorical", choices=["relu", "tanh"]),
    ])

    def objective(config: Dict[str, Any], budget: float = 1.0) -> float:
        max_iter = max(50, int(500 * budget))
        model = MLPRegressor(
            hidden_layer_sizes=(config["hidden_layer_1"],
                                config["hidden_layer_2"]),
            learning_rate_init=config["learning_rate_init"],
            alpha=config["alpha"],
            batch_size=config["batch_size"],
            activation=config["activation"],
            max_iter=max_iter,
            random_state=0,
            early_stopping=True,
            validation_fraction=0.15,
        )
        scores = cross_val_score(model, X, y, cv=3,
                                 scoring="neg_mean_squared_error")
        return float(scores.mean())

    return space, objective


BENCHMARKS = {
    "xgboost": {
        "make_fn": _make_xgboost_benchmark,
        "budget": 50,
        "description": "Gradient Boosting Regressor on California Housing",
    },
    "svm": {
        "make_fn": _make_svm_benchmark,
        "budget": 40,
        "description": "SVM Classifier on Breast Cancer",
    },
    "nn": {
        "make_fn": _make_nn_benchmark,
        "budget": 40,
        "description": "MLP Regressor on Diabetes",
    },
}


# ================================================================
# EDITABLE — Custom HPO strategy (lines 255 to 326)
# The agent modifies ONLY this section.
# ================================================================


class CustomHPOStrategy:
    """Custom hyperparameter optimization strategy.

    The agent should implement suggest() which proposes the next
    hyperparameter configuration to evaluate, given the search space
    and history of previous trials.

    The strategy is called repeatedly in a loop:
        1. strategy.suggest(space, history, budget_left) -> (config, fidelity)
        2. config is evaluated -> score
        3. Trial(config, score, fidelity) is added to history
        4. Repeat until budget exhausted

    Available utilities:
        space.params        — list of HParam objects with name, type, range
        space.dim           — number of hyperparameters
        space.sample_uniform(rng) — sample random config
        space.clip(config)  — clip values to valid ranges

        trial.config        — dict of hyperparameter values
        trial.score         — observed validation score (higher is better)
        trial.budget        — fidelity fraction used (1.0 = full evaluation)

    Useful scipy:
        from scipy.stats import norm
        norm.cdf(x), norm.pdf(x)

    Useful numpy:
        np.random.RandomState for reproducibility

    Args:
        seed: random seed for reproducibility

    Returns from suggest():
        config: dict mapping param names to values
        fidelity: float in (0, 1] — fraction of full evaluation budget.
                  Use 1.0 for full-fidelity evaluation.
                  Lower values = cheaper evaluation (e.g., fewer epochs/trees).
    """

    def __init__(self, seed: int = 42):
        """Initialize the strategy.

        Default: stores seed and creates RNG.
        The agent may add any internal state needed.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Propose the next configuration to evaluate.

        Default: uniform random search (poor — replace with a better
        strategy).

        Args:
            space: search space definition
            history: list of previously evaluated trials
            budget_left: number of full-fidelity evaluations remaining

        Returns:
            config: dict of hyperparameter name -> value
            fidelity: float in (0, 1], fraction of full evaluation
        """
        config = space.sample_uniform(self.rng)
        return config, 1.0


# ================================================================
# FIXED — HPO loop and evaluation (do not modify below)
# ================================================================


def run_hpo_loop(benchmark_name: str, seed: int, budget: int,
                 output_dir: str):
    """Run full HPO loop and report metrics."""
    cfg = BENCHMARKS[benchmark_name]
    space, objective = cfg["make_fn"]()

    strategy = CustomHPOStrategy(seed=seed)
    history: List[Trial] = []
    best_score = -np.inf
    best_config = None
    total_cost = 0.0
    convergence_curve = []
    convergence_threshold_reached = budget  # default: never reached early

    # Determine convergence threshold (90% of budget's potential)
    # We'll compute this after seeing some results

    start_time = time.time()

    eval_count = 0
    while total_cost < budget:
        budget_left = budget - total_cost
        if budget_left < 0.1:
            break

        config, fidelity = strategy.suggest(space, history, int(budget_left))
        fidelity = float(np.clip(fidelity, 0.1, 1.0))
        config = space.clip(config)

        score = objective(config, budget=fidelity)
        trial = Trial(config=config, score=score, budget=fidelity)
        history.append(trial)

        total_cost += fidelity
        eval_count += 1

        if score > best_score:
            best_score = score
            best_config = config.copy()

        convergence_curve.append({
            "eval": eval_count,
            "cost": total_cost,
            "best_score": best_score,
        })

        if eval_count % 5 == 0 or total_cost >= budget - 0.1:
            elapsed = time.time() - start_time
            print(
                f"TRAIN_METRICS eval={eval_count} cost={total_cost:.1f}/{budget} "
                f"best_score={best_score:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - start_time

    # Compute convergence speed: area under the normalized curve (AUC)
    # Higher AUC = faster convergence (found good configs earlier)
    if len(convergence_curve) > 1:
        costs = [c["cost"] / budget for c in convergence_curve]
        scores = [c["best_score"] for c in convergence_curve]
        # Normalize scores to [0, 1] range
        s_min, s_max = min(scores), max(scores)
        if s_max > s_min:
            norm_scores = [(s - s_min) / (s_max - s_min) for s in scores]
        else:
            norm_scores = [1.0] * len(scores)
        # Trapezoidal AUC
        auc = float(np.trapezoid(norm_scores, costs)) if hasattr(np, 'trapezoid') else float(np.trapz(norm_scores, costs))
    else:
        auc = 0.0

    # Print final metrics
    print(f"TEST_METRICS best_val_score={best_score:.6f}", flush=True)
    print(f"TEST_METRICS convergence_auc={auc:.6f}", flush=True)
    print(f"TEST_METRICS total_evals={eval_count}", flush=True)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "benchmark": benchmark_name,
        "seed": seed,
        "budget": budget,
        "total_cost": total_cost,
        "total_evals": eval_count,
        "best_score": best_score,
        "best_config": best_config,
        "convergence_auc": auc,
        "elapsed_seconds": elapsed,
        "convergence_curve": convergence_curve,
    }
    with open(os.path.join(output_dir,
                           f"{benchmark_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return best_score, auc


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization Strategy Benchmark")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("SEED", 42)))
    parser.add_argument("--budget", type=int, default=None,
                        help="Override default budget for this benchmark")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("OUTPUT_DIR", "./output"))
    args = parser.parse_args()

    benchmark_budget = args.budget or BENCHMARKS[args.benchmark]["budget"]

    print(f"Running HPO benchmark: {args.benchmark} "
          f"(seed={args.seed}, budget={benchmark_budget})", flush=True)
    best_score, auc = run_hpo_loop(
        args.benchmark, args.seed, benchmark_budget, args.output_dir)
    print(f"Final best score on {args.benchmark}: {best_score:.6f} "
          f"(convergence AUC: {auc:.4f})", flush=True)


if __name__ == "__main__":
    main()
