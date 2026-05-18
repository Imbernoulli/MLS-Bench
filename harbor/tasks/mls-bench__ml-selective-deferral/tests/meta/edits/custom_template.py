"""Selective prediction / deferral benchmark.

Fixed:
- offline AIF360 high-stakes dataset loading
- train / calibration / test splits
- base classifier training
- metric computation

Editable:
- SelectivePolicy, which decides whether to accept or defer predictions
  based on calibration outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGET_COVERAGE_DEFAULT = 0.80
DATA_HOME = os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn")


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    load_raw: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, bool]]
    group_name: str


def _dataset_parts(bundle):
    if hasattr(bundle, "X") and hasattr(bundle, "y"):
        return bundle.X, bundle.y
    return bundle[0], bundle[1]


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        if any(name is not None for name in df.index.names):
            idx_df = df.index.to_frame(index=False)
            for col in reversed(list(idx_df.columns)):
                if col not in df.columns:
                    df.insert(0, col, idx_df[col].to_numpy())
        return df.reset_index(drop=True)
    return pd.DataFrame(np.asarray(X)).reset_index(drop=True)


def _encode_features(df: pd.DataFrame) -> np.ndarray:
    encoded = pd.get_dummies(df, dummy_na=False)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    return encoded.astype(np.float32).to_numpy()


def _column_values(df: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        col = lower_map.get(str(name).lower())
        if col is not None:
            values = df[col]
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().mean() >= 0.8:
                arr = numeric.to_numpy(dtype=float)
                med = np.nanmedian(arr)
                return np.nan_to_num(arr, nan=med)
            return values.astype("category").cat.codes.to_numpy(dtype=float)
    first = df.iloc[:, 0]
    numeric = pd.to_numeric(first, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        arr = numeric.to_numpy(dtype=float)
        med = np.nanmedian(arr)
        return np.nan_to_num(arr, nan=med)
    return first.astype("category").cat.codes.to_numpy(dtype=float)


def _protected_groups(df: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    lower_map = {str(c).lower(): c for c in df.columns}
    code_columns = []
    for name in candidates:
        col = lower_map.get(str(name).lower())
        if col is None:
            continue
        codes = pd.Series(df[col]).astype("category").cat.codes.to_numpy(dtype=int)
        code_columns.append(codes)
    if not code_columns:
        return _quantile_bins(_column_values(df, [df.columns[0]]), n_bins=2)

    combined = np.zeros(len(df), dtype=int)
    factor = 1
    for codes in code_columns:
        codes = codes - int(codes.min())
        combined += factor * codes
        factor *= int(codes.max()) + 1
    unique = {value: idx for idx, value in enumerate(sorted(np.unique(combined)))}
    return np.asarray([unique[value] for value in combined], dtype=int)


def _binary_labels(y, positive_tokens: list[str] | None = None) -> tuple[np.ndarray, bool]:
    series = pd.Series(y).reset_index(drop=True)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        arr = numeric.to_numpy(dtype=float)
        uniq = np.unique(arr[~np.isnan(arr)])
        if len(uniq) <= 2:
            return (arr == np.max(uniq)).astype(int), False
        return arr.astype(np.float32), True

    text = series.astype(str).str.lower().str.strip()
    if positive_tokens:
        tokens = tuple(tok.lower() for tok in positive_tokens)
        return text.apply(lambda value: any(tok in value for tok in tokens)).to_numpy(dtype=int), False
    values = sorted(text.unique())
    return (text == values[-1]).to_numpy(dtype=int), False


def _load_adult() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    from aif360.sklearn.datasets import fetch_adult

    X_raw, y_raw = _dataset_parts(
        fetch_adult(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
    )
    df = _as_frame(X_raw)
    y, is_regression = _binary_labels(y_raw, positive_tokens=[">50k"])
    groups = _protected_groups(df, ["sex", "race"])
    return _encode_features(df), y, groups, is_regression


def _load_compas() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    from aif360.sklearn.datasets import fetch_compas

    X_raw, y_raw = _dataset_parts(
        fetch_compas(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
    )
    df = _as_frame(X_raw)
    y, is_regression = _binary_labels(y_raw, positive_tokens=["survived", "no recid", "0"])
    groups = _protected_groups(df, ["race", "sex"])
    return _encode_features(df), y, groups, is_regression


def _load_law_school() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    from aif360.sklearn.datasets import fetch_lawschool_gpa

    X_raw, y_raw = _dataset_parts(
        fetch_lawschool_gpa(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
    )
    df = _as_frame(X_raw)
    y, is_regression = _binary_labels(y_raw)
    groups = _protected_groups(df, ["race", "gender"])
    return _encode_features(df), y, groups, is_regression


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "adult": BenchmarkSpec("adult", _load_adult, "sex/race"),
    "compas": BenchmarkSpec("compas", _load_compas, "race/sex"),
    "law_school": BenchmarkSpec("law_school", _load_law_school, "race/gender"),
}


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    if np.allclose(scores, scores[0]):
        return 0.5
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def _quantile_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) <= 2:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, quantiles[1:-1], right=True)


def _make_binary_targets(raw_y: np.ndarray, train_idx: np.ndarray, is_regression: bool) -> tuple[np.ndarray, float]:
    if not is_regression:
        return raw_y.astype(int), float("nan")
    threshold = float(np.median(raw_y[train_idx]))
    y = (raw_y > threshold).astype(int)
    return y, threshold


def _split_dataset(train_idx: np.ndarray, test_idx: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    n_groups = int(np.max(groups)) + 1
    strata_train = y[train_idx] * n_groups + groups[train_idx]
    counts = np.bincount(strata_train.astype(int))
    stratify = strata_train if np.all(counts[counts > 0] >= 2) else None
    fit_idx, cal_idx = train_test_split(train_idx, test_size=0.25, random_state=seed, stratify=stratify)
    return {
        "fit_idx": np.sort(fit_idx),
        "cal_idx": np.sort(cal_idx),
        "test_idx": np.sort(test_idx),
    }


def _build_base_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=seed,
                ),
            ),
        ]
    )


def _confidence_features(probs: np.ndarray, groups: np.ndarray | None = None, X: np.ndarray | None = None) -> np.ndarray:
    p1 = probs[:, 1]
    p0 = probs[:, 0]
    max_prob = np.maximum(p0, p1)
    margin = np.abs(p1 - p0)
    entropy = -(p0 * np.log(np.clip(p0, 1e-12, 1.0)) + p1 * np.log(np.clip(p1, 1e-12, 1.0)))
    feats = [p1, max_prob, margin, entropy]
    if groups is not None:
        feats.append(groups.astype(float))
    if X is not None and X.ndim == 2 and X.shape[1] > 0:
        feats.append(X[:, 0].astype(float))
    return np.column_stack(feats)


# =============================================================================
# EDITABLE REGION START
# =============================================================================


class SelectivePolicy:
    """Policy that maps calibration outputs to accept / defer decisions.

    The default implementation is intentionally conservative:
    it accepts the top-confidence examples needed to reach the target coverage.
    Baselines replace this class with more specialized policies.
    """

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "global_threshold"

    def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        self.group_thresholds_ = {}
        self.meta_model_ = None
        return self

    def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        return np.max(probs, axis=1)

    def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        scores = self.acceptance_score(probs, groups, X)
        return scores >= self.threshold_

    def calibration_summary(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold_),
        }


# =============================================================================
# EDITABLE REGION END
# =============================================================================


def _predict_labels(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1)


def _selective_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accept: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    accept = accept.astype(bool)
    coverage = float(accept.mean())
    if accept.any():
        selective_risk = float(np.mean(y_pred[accept] != y_true[accept]))
    else:
        selective_risk = 1.0

    group_risks = []
    group_deferrals = []
    for group_id in np.unique(groups):
        group_mask = groups == group_id
        group_accept = accept[group_mask]
        group_y = y_true[group_mask]
        group_pred = y_pred[group_mask]
        if group_mask.sum() == 0:
            continue
        if group_accept.any():
            group_risk = float(np.mean(group_pred[group_accept] != group_y[group_accept]))
        else:
            group_risk = 1.0
        group_risks.append(group_risk)
        group_deferrals.append(float(1.0 - group_accept.mean()))

    worst_group_risk = float(max(group_risks)) if group_risks else selective_risk
    deferral_gap = float(max(group_deferrals) - min(group_deferrals)) if group_deferrals else 0.0
    correctness = (y_pred == y_true).astype(int)
    auroc = _safe_roc_auc(correctness, scores)
    return {
        "selective_risk_at80": selective_risk,
        "coverage_at80": coverage,
        "worst_group_selective_risk": worst_group_risk,
        "deferral_rate_gap": deferral_gap,
        "auroc": auroc,
    }


def _print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    parts = [f"{key}={value:.6f}" for key, value in metrics.items()]
    print(f"{prefix}: " + " ".join(parts), flush=True)


def run_benchmark(dataset: str, seed: int, target_coverage: float, output_dir: str | None = None) -> dict[str, float]:
    if dataset not in BENCHMARKS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {sorted(BENCHMARKS)}")

    spec = BENCHMARKS[dataset]
    X, raw_y, raw_groups, is_regression = spec.load_raw()

    indices = np.arange(len(X))
    if is_regression:
        stratify_for_split = _quantile_bins(raw_y, n_bins=5)
    else:
        stratify_for_split = raw_y.astype(int)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_for_split,
    )
    y, label_threshold = _make_binary_targets(raw_y, train_idx, is_regression=is_regression)
    groups = np.asarray(raw_groups, dtype=int)
    group_threshold = -1.0

    split = _split_dataset(train_idx, test_idx, y, groups, seed)
    fit_idx = split["fit_idx"]
    cal_idx = split["cal_idx"]
    test_idx = split["test_idx"]

    model = _build_base_model(seed)
    model.fit(X[fit_idx], y[fit_idx])

    cal_probs = model.predict_proba(X[cal_idx])
    test_probs = model.predict_proba(X[test_idx])
    cal_pred = _predict_labels(cal_probs)
    test_pred = _predict_labels(test_probs)

    policy = SelectivePolicy(target_coverage=target_coverage, random_state=seed)
    policy.fit(cal_probs, y[cal_idx], groups[cal_idx], X=X[cal_idx])
    cal_accept = policy.predict_accept(cal_probs, groups[cal_idx], X=X[cal_idx])
    test_accept = policy.predict_accept(test_probs, groups[test_idx], X=X[test_idx])
    test_scores = policy.acceptance_score(test_probs, groups[test_idx], X=X[test_idx])

    train_acc = float(np.mean(model.predict(X[fit_idx]) == y[fit_idx]))
    cal_acc = float(np.mean(cal_pred == y[cal_idx]))
    train_summary = {
        "train_accuracy": train_acc,
        "cal_accuracy": cal_acc,
        "cal_coverage": float(cal_accept.mean()),
        "policy_threshold": float(getattr(policy, "threshold_", 0.0)),
    }
    _print_metrics("TRAIN_METRICS", train_summary)

    test_metrics = _selective_metrics(y[test_idx], test_pred, test_accept, test_scores, groups[test_idx])
    test_metrics["target_coverage"] = float(target_coverage)
    test_metrics["actual_coverage"] = float(test_accept.mean())
    test_metrics["label_threshold"] = float(label_threshold) if np.isfinite(label_threshold) else -1.0
    test_metrics["group_threshold"] = float(group_threshold)
    _print_metrics("TEST_METRICS", test_metrics)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(output_dir) / f"{dataset}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({"train": train_summary, "test": test_metrics}, f, indent=2, sort_keys=True)

    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective prediction / deferral benchmark.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(BENCHMARKS),
        help="Benchmark dataset name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-coverage", type=float, default=TARGET_COVERAGE_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_benchmark(args.dataset, args.seed, args.target_coverage, args.output_dir)


if __name__ == "__main__":
    main()
