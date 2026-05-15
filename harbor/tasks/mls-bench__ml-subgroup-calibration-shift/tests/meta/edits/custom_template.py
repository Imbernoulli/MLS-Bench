"""Subgroup calibration under distribution shift.

The benchmark uses cached high-stakes tabular datasets from AIF360 rather than
task-local sklearn proxies. Package-level data preparation downloads the needed
Adult, COMPAS, and Law School assets before compute-node execution.

Fixed:
- dataset loading
- shifted train/calibration/test split
- base classifier training
- metric computation

Editable:
- CalibrationMethod
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_HOME = os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn")


def expected_calibration_error(probs, labels, n_bins=15):
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        prop = float(mask.mean())
        if prop > 0:
            ece += prop * abs(float(accuracies[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def _safe_auc(labels, probs):
    labels = np.asarray(labels).reshape(-1).astype(int)
    probs = np.asarray(probs).reshape(-1)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def _quantile_groups(score, n_groups=4):
    score = np.asarray(score).reshape(-1)
    edges = np.quantile(score, np.linspace(0.0, 1.0, n_groups + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(score, edges[1:-1], right=True).astype(int)


def _binary_threshold(target):
    target = np.asarray(target).reshape(-1)
    return (target > np.median(target)).astype(int)


def _dataset_parts(bundle):
    if hasattr(bundle, "X") and hasattr(bundle, "y"):
        return bundle.X, bundle.y
    return bundle[0], bundle[1]


def _as_frame(X):
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        if any(name is not None for name in df.index.names):
            idx_df = df.index.to_frame(index=False)
            for col in reversed(list(idx_df.columns)):
                if col not in df.columns:
                    df.insert(0, col, idx_df[col].to_numpy())
        return df.reset_index(drop=True)
    return pd.DataFrame(np.asarray(X)).reset_index(drop=True)


def _encode_features(df):
    encoded = pd.get_dummies(df, dummy_na=False)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    return encoded.astype(np.float32).to_numpy()


def _column_values(df, candidates):
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


def _protected_groups(df, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}
    code_columns = []
    for name in candidates:
        col = lower_map.get(str(name).lower())
        if col is None:
            continue
        codes = pd.Series(df[col]).astype("category").cat.codes.to_numpy(dtype=int)
        code_columns.append(codes)
    if not code_columns:
        return _quantile_groups(_column_values(df, [df.columns[0]]), n_groups=2)

    combined = np.zeros(len(df), dtype=int)
    factor = 1
    for codes in code_columns:
        codes = codes - int(codes.min())
        combined += factor * codes
        factor *= int(codes.max()) + 1
    unique = {value: idx for idx, value in enumerate(sorted(np.unique(combined)))}
    return np.asarray([unique[value] for value in combined], dtype=int)


def _binary_labels(y, positive_tokens=None):
    series = pd.Series(y).reset_index(drop=True)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        arr = numeric.to_numpy(dtype=float)
        uniq = np.unique(arr[~np.isnan(arr)])
        if len(uniq) <= 2:
            return (arr == np.max(uniq)).astype(int)
        return _binary_threshold(arr)

    text = series.astype(str).str.lower().str.strip()
    if positive_tokens:
        tokens = tuple(tok.lower() for tok in positive_tokens)
        return text.apply(lambda value: any(tok in value for tok in tokens)).to_numpy(dtype=int)
    values = sorted(text.unique())
    positive = values[-1]
    return (text == positive).to_numpy(dtype=int)


def _load_adult():
    from aif360.sklearn.datasets import fetch_adult

    X_raw, y_raw = _dataset_parts(
        fetch_adult(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
    )
    df = _as_frame(X_raw)
    y = _binary_labels(y_raw, positive_tokens=[">50k"])
    domain_score = _column_values(df, ["age", "education-num", "hours-per-week"])
    groups = _protected_groups(df, ["sex", "race"])
    return _encode_features(df), y, domain_score, groups


def _load_compas():
    from aif360.sklearn.datasets import fetch_compas

    X_raw, y_raw = _dataset_parts(
        fetch_compas(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
    )
    df = _as_frame(X_raw)
    y = _binary_labels(y_raw, positive_tokens=["survived", "no recid", "0"])
    domain_score = _column_values(df, ["priors_count", "age"])
    groups = _protected_groups(df, ["race", "sex"])
    return _encode_features(df), y, domain_score, groups


def _load_law_school():
    from aif360.sklearn.datasets import fetch_lawschool_gpa

    X_raw, y_raw = _dataset_parts(
        fetch_lawschool_gpa(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
    )
    df = _as_frame(X_raw)
    y = _binary_labels(y_raw)
    domain_score = _column_values(df, ["lsat", "ugpa"])
    groups = _protected_groups(df, ["race", "gender"])
    return _encode_features(df), y, domain_score, groups


class CalibrationMethod:
    """Editable calibration method.

    Implement fit() and predict_proba() to map raw positive-class probabilities
    to calibrated positive-class probabilities.
    """

    def __init__(self):
        self.eps = 1e-6
        self._identity = True

    def fit(self, probs, labels, groups=None):
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        self._base_rate = float(np.clip(labels.mean(), self.eps, 1.0 - self.eps))
        return self

    def predict_proba(self, probs, groups=None):
        probs = np.asarray(probs).reshape(-1)
        return np.clip(probs, self.eps, 1.0 - self.eps)


def _load_dataset(name):
    loaders = {
        "adult": _load_adult,
        "compas": _load_compas,
        "law_school": _load_law_school,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    return loaders[name]()


def _shifted_split(y, domain_score, seed, test_frac=0.30, calib_frac=0.25):
    rng = np.random.RandomState(seed)
    train_idx = []
    calib_idx = []
    test_idx = []

    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        order = cls_idx[np.argsort(domain_score[cls_idx])]
        n_test = max(1, int(round(test_frac * len(order))))
        source_idx = order[:-n_test]
        test_cls = order[-n_test:]
        n_cal = max(1, int(round(calib_frac * len(source_idx))))
        calib_cls = rng.choice(source_idx, size=n_cal, replace=False)
        train_cls = np.setdiff1d(source_idx, calib_cls, assume_unique=False)

        train_idx.append(train_cls)
        calib_idx.append(calib_cls)
        test_idx.append(test_cls)

    train_idx = np.sort(np.concatenate(train_idx))
    calib_idx = np.sort(np.concatenate(calib_idx))
    test_idx = np.sort(np.concatenate(test_idx))
    return train_idx, calib_idx, test_idx


def _fit_base_classifier(X_train, y_train, seed):
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1200,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def _evaluate(probs, labels, groups):
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    groups = np.asarray(groups).reshape(-1).astype(int)

    group_ece = []
    group_auc = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 5:
            continue
        group_ece.append(expected_calibration_error(probs[mask], labels[mask]))
        group_auc.append(_safe_auc(labels[mask], probs[mask]))

    worst_group_ece = float(np.max(group_ece)) if group_ece else float("nan")
    subgroup_auroc = float(np.nanmean(group_auc)) if group_auc else float("nan")
    max_subgroup_gap = float(np.max(group_ece) - np.min(group_ece)) if len(group_ece) > 1 else float("nan")
    brier = float(brier_score_loss(labels, probs))
    return {
        "worst_group_ece": worst_group_ece,
        "brier": brier,
        "subgroup_auroc": subgroup_auroc,
        "max_subgroup_gap": max_subgroup_gap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["adult", "compas", "law_school"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X, y, domain_score, groups = _load_dataset(args.dataset)
    train_idx, calib_idx, test_idx = _shifted_split(y, domain_score, seed=args.seed)

    model = _fit_base_classifier(X[train_idx], y[train_idx], seed=args.seed)
    cal_probs = model.predict_proba(X[calib_idx])[:, 1]
    test_probs = model.predict_proba(X[test_idx])[:, 1]

    method = CalibrationMethod().fit(cal_probs, y[calib_idx], groups=groups[calib_idx])
    cal_probs_hat = method.predict_proba(cal_probs, groups=groups[calib_idx])
    test_probs_hat = method.predict_proba(test_probs, groups=groups[test_idx])

    print(
        "TRAIN_METRICS: "
        f"dataset={args.dataset} "
        f"cal_ece_before={expected_calibration_error(cal_probs, y[calib_idx]):.6f} "
        f"cal_ece_after={expected_calibration_error(cal_probs_hat, y[calib_idx]):.6f} "
        f"cal_brier_before={brier_score_loss(y[calib_idx], cal_probs):.6f} "
        f"cal_brier_after={brier_score_loss(y[calib_idx], cal_probs_hat):.6f}",
        flush=True,
    )

    test_metrics = _evaluate(test_probs_hat, y[test_idx], groups[test_idx])
    print(
        "TEST_METRICS: "
        + " ".join(f"{k}={v:.6f}" for k, v in test_metrics.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
