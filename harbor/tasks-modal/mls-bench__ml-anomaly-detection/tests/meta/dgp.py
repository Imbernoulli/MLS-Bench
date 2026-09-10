"""Held-out scoring module for ml-anomaly-detection (NOT agent-visible).

Imported only by the host-side parser.py (native) / Harbor verifier. Holds the
dataset identity (the adbench file mapping), the loader, the train/test split,
the test labels, and the metrics — so the editable detector can neither name the
eval datasets nor recover y_test to return it as scores.

Loader/split/metric bodies are byte-identical to the originals, so honest
results are reproduced exactly.
"""

import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILES = {
    "cardio": "6_cardio.npz",
    "thyroid": "38_thyroid.npz",
    "satellite": "30_satellite.npz",
    "shuttle": "32_shuttle.npz",
}
TRAIN_RATIO = 0.6  # Task-local 60/40 stratified train/test split.


def _adbench_dir():
    # The module runs host-side in two contexts: the eval container (DATA_ROOT
    # is exported, data lives under $DATA_ROOT/adbench) and the verifier/scorer
    # (env is stripped, _PROJECT_ROOT resolves to a temp dir). Try, in order:
    # DATA_ROOT, the committed native data dir, then the Harbor image's baked
    # data dir (/data/adbench). Return the first that exists.
    env_root = os.environ.get("DATA_ROOT")
    cands = []
    if env_root:
        cands.append(Path(env_root) / "adbench")
    cands.append(_PROJECT_ROOT / "vendor" / "data" / "adbench")
    cands.append(Path("/data") / "adbench")
    for c in cands:
        if c.is_dir():
            return c
    return cands[-1]


def load_dataset(name):
    """Load an anomaly detection dataset. Returns (X float64, y int 0/1)."""
    filepath = _adbench_dir() / DATASET_FILES[name]
    data = np.load(filepath, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int32).ravel()
    y = (y > 0).astype(np.int32)
    return X, y


def _split_scaled(env_name, seed):
    X, y = load_dataset(env_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1.0 - TRAIN_RATIO, stratify=y, random_state=seed,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_test


def score_predictions(scores, y_test):
    """AUROC + F1 (at the contamination-ratio threshold) — same as before."""
    scores = np.asarray(scores)
    try:
        auroc = roc_auc_score(y_test, scores)
    except ValueError:
        auroc = 0.5
    contamination = y_test.mean()
    if 0 < contamination < 1:
        threshold = np.percentile(scores, 100 * (1 - contamination))
        y_pred = (scores >= threshold).astype(int)
    else:
        y_pred = np.zeros_like(y_test)
    f1 = f1_score(y_test, y_pred, zero_division=0.0)
    return {"auroc": float(auroc), "f1": float(f1)}


# ---- helpers for the input pre-generator and the host-side scorer ----

def gen_input(env_name, seed=42):
    """Agent-visible input: standardized (X_train, X_test). Labels withheld."""
    X_train_s, X_test_s, _y_test = _split_scaled(env_name, seed)
    return X_train_s, X_test_s


def truth(env_name, seed=42):
    """Held-out ground truth: y_test for the same split."""
    _Xtr, _Xte, y_test = _split_scaled(env_name, seed)
    return y_test


def opaque_label(label):
    """Host-only opaque token for a dataset label (the agent must not learn which
    dataset is evaluated). Used to name the pre-generated input files and as the
    identity the run scripts pass, so listing the workspace reveals only tokens;
    the real label stays host-side (the scorer uses the test-command label)."""
    import hashlib
    return hashlib.sha1(
        ("ml-anomaly-detection::dataset-token::v1::" + str(label)).encode("utf-8")
    ).hexdigest()[:12]
