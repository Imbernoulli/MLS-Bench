#!/usr/bin/env python3
"""Pure SLDBench scaling-law discovery benchmark."""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


DATA_DIR = Path(os.environ.get("SCALING_LAW_DATA_DIR", "/data/scaling_law"))
EPS = 1e-8


@dataclass
class BenchmarkData:
    name: str
    X_num_train: np.ndarray
    X_cat_train: np.ndarray
    y_train: np.ndarray
    X_num_test: np.ndarray
    X_cat_test: np.ndarray
    y_test: np.ndarray
    numeric_names: list[str]
    categorical_names: list[str]
    target_name: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick(mapping: dict, keys: list[str], default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def safe_float(value, default=np.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_sld_vocab() -> BenchmarkData:
    train_rows = load_jsonl(DATA_DIR / "sldbench__vocab_scaling_law__train.jsonl")
    test_rows = load_jsonl(DATA_DIR / "sldbench__vocab_scaling_law__test.jsonl")

    def convert(rows: list[dict]):
        X_num, X_cat, y = [], [], []
        for row in rows:
            X_num.append([
                safe_float(pick(row, ["non_vocab_parameters", "N"])),
                safe_float(pick(row, ["vocab_size", "V"])),
                safe_float(pick(row, ["num_characters", "D"])),
            ])
            X_cat.append([str(pick(row, ["group"], "all_data"))])
            y.append(safe_float(pick(row, ["unigram_normalized_loss", "loss"])))
        return np.asarray(X_num, dtype=float), np.asarray(X_cat, dtype=object), np.asarray(y, dtype=float)

    X_num_train, X_cat_train, y_train = convert(train_rows)
    X_num_test, X_cat_test, y_test = convert(test_rows)
    return BenchmarkData(
        name="sld-vocab",
        X_num_train=X_num_train,
        X_cat_train=X_cat_train,
        y_train=y_train,
        X_num_test=X_num_test,
        X_cat_test=X_cat_test,
        y_test=y_test,
        numeric_names=["non_vocab_parameters", "vocab_size", "num_characters"],
        categorical_names=["group"],
        target_name="unigram_normalized_loss",
    )


def _build_sld_lrbsz() -> BenchmarkData:
    train_rows = load_jsonl(DATA_DIR / "sldbench__lr_bsz_scaling_law__train.jsonl")
    test_rows = load_jsonl(DATA_DIR / "sldbench__lr_bsz_scaling_law__test.jsonl")

    def convert(rows: list[dict]):
        X_num, X_cat, y = [], [], []
        for row in rows:
            X_num.append([
                safe_float(pick(row, ["lr"])),
                safe_float(pick(row, ["bsz"])),
                safe_float(pick(row, ["data_size", "D"])),
                safe_float(pick(row, ["non_embedding_param_size", "N"])),
            ])
            X_cat.append([str(pick(row, ["group"], "all_data"))])
            y.append(safe_float(pick(row, ["lm_loss", "loss"])))
        return np.asarray(X_num, dtype=float), np.asarray(X_cat, dtype=object), np.asarray(y, dtype=float)

    X_num_train, X_cat_train, y_train = convert(train_rows)
    X_num_test, X_cat_test, y_test = convert(test_rows)
    return BenchmarkData(
        name="sld-lrbsz",
        X_num_train=X_num_train,
        X_cat_train=X_cat_train,
        y_train=y_train,
        X_num_test=X_num_test,
        X_cat_test=X_cat_test,
        y_test=y_test,
        numeric_names=["lr", "bsz", "data_size", "non_embedding_param_size"],
        categorical_names=["group"],
        target_name="lm_loss",
    )


def _build_sld_dataconstrained() -> BenchmarkData:
    train_rows = load_jsonl(DATA_DIR / "sldbench__data_constrained_scaling_law__train.jsonl")
    test_rows = load_jsonl(DATA_DIR / "sldbench__data_constrained_scaling_law__test.jsonl")

    def convert(rows: list[dict]):
        X_num, X_cat, y = [], [], []
        for row in rows:
            X_num.append([
                safe_float(pick(row, ["unique_tokens", "U"])),
                safe_float(pick(row, ["params", "N"])),
                safe_float(pick(row, ["tokens", "D"])),
            ])
            X_cat.append([str(pick(row, ["group"], "all_data"))])
            y.append(safe_float(pick(row, ["loss"])))
        return np.asarray(X_num, dtype=float), np.asarray(X_cat, dtype=object), np.asarray(y, dtype=float)

    X_num_train, X_cat_train, y_train = convert(train_rows)
    X_num_test, X_cat_test, y_test = convert(test_rows)
    return BenchmarkData(
        name="sld-dataconstrained",
        X_num_train=X_num_train,
        X_cat_train=X_cat_train,
        y_train=y_train,
        X_num_test=X_num_test,
        X_cat_test=X_cat_test,
        y_test=y_test,
        numeric_names=["unique_tokens", "params", "tokens"],
        categorical_names=["group"],
        target_name="loss",
    )


def load_benchmark(name: str) -> BenchmarkData:
    if name == "sld-vocab":
        return _build_sld_vocab()
    if name == "sld-lrbsz":
        return _build_sld_lrbsz()
    if name == "sld-dataconstrained":
        return _build_sld_dataconstrained()
    raise ValueError(f"Unknown benchmark: {name}")


def group_labels(X_cat: np.ndarray) -> np.ndarray:
    X_cat = np.asarray(X_cat, dtype=object)
    if X_cat.ndim == 1:
        X_cat = X_cat[:, None]
    if X_cat.size == 0 or X_cat.shape[1] == 0:
        return np.asarray(["__all__"] * len(X_cat), dtype=object)
    return np.asarray(
        [str(v) if v is not None else "__MISSING__" for v in X_cat[:, 0]],
        dtype=object,
    )


# ============================================================
# Scaling Law Model (EDITABLE)
# ============================================================

class ScalingLawModel:
    """Editable benchmark-specific symbolic law scaffold.

    You may implement different symbolic forms for:
    - sld-vocab
    - sld-lrbsz
    - sld-dataconstrained

    The raw observed training trials are mirrored in:
    - observed_trials/sld_vocab_train.jsonl
    - observed_trials/sld_lrbsz_train.jsonl
    - observed_trials/sld_dataconstrained_train.jsonl
    """

    def __init__(self, benchmark_name: str, numeric_names=None, categorical_names=None):
        self.benchmark_name = benchmark_name
        self.numeric_names = list(numeric_names or [])
        self.categorical_names = list(categorical_names or [])

    def fit(self, X_num, X_cat, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X_num, X_cat):
        return np.full(len(X_num), self.mean_)


# ============================================================
# Evaluation
# ============================================================

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom < EPS:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff ** 2))


def run_benchmark(benchmark: str, seed: int, output_dir: str) -> None:
    set_seed(seed)
    data = load_benchmark(benchmark)
    model = ScalingLawModel(data.name, data.numeric_names, data.categorical_names)
    model.fit(data.X_num_train, data.X_cat_train, data.y_train)

    train_pred = model.predict(data.X_num_train, data.X_cat_train)
    test_pred = model.predict(data.X_num_test, data.X_cat_test)

    train_r2 = r2_score(data.y_train, train_pred)
    train_mae = mean_absolute_error(data.y_train, train_pred)
    test_r2 = r2_score(data.y_test, test_pred)
    test_mae = mean_absolute_error(data.y_test, test_pred)
    test_rmse = float(np.sqrt(mean_squared_error(data.y_test, test_pred)))
    test_nmae = float(test_mae / (np.std(data.y_test) + EPS))

    n_features = int(getattr(model, "num_features_", data.X_num_train.shape[1]))
    print(
        "TRAIN_METRICS "
        f"n_train={len(data.y_train)} n_test={len(data.y_test)} "
        f"n_features={n_features} train_r2={train_r2:.6f} train_mae={train_mae:.6f}",
        flush=True,
    )
    print(
        "TEST_METRICS "
        f"r2={test_r2:.6f} mae={test_mae:.6f} rmse={test_rmse:.6f} nmae={test_nmae:.6f}",
        flush=True,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / f"{benchmark}_predictions.json").open("w") as f:
        json.dump(
            {
                "benchmark": benchmark,
                "target": data.target_name,
                "y_true": data.y_test.tolist(),
                "y_pred": np.asarray(test_pred).tolist(),
                "metrics": {
                    "r2": test_r2,
                    "mae": float(test_mae),
                    "rmse": test_rmse,
                    "nmae": test_nmae,
                },
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output")
    args = parser.parse_args()
    run_benchmark(args.benchmark, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
