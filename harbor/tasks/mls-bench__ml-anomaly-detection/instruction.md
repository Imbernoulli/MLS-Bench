# MLS-Bench: ml-anomaly-detection

# Unsupervised Anomaly Detection Algorithm Design

## Research Question
Design a novel unsupervised anomaly detection algorithm for tabular data that generalizes across datasets with different sample counts, dimensionality, and anomaly rates. The contribution is the *scoring rule* — how to model normal structure on standardized tabular features and assign higher scores to deviating points — using only unlabeled features at fit time.

## Background
Unsupervised anomaly detection identifies rare or unusual samples without labels during training. No single method dominates across dataset characteristics; promising designs combine density, isolation, distance, projection, ensemble, or robust-statistics ideas.

Reference baselines:
- **Isolation Forest (iForest)** — Liu, Ting, Zhou, ICDM 2008 ([paper](https://ieeexplore.ieee.org/document/4781136)). Tree-based isolation: anomalies are isolated with shorter random-partition path lengths. Default hyperparameters: 100 trees, sub-sample size 256.
- **Local Outlier Factor (LOF)** — Breunig, Kriegel, Ng, Sander, SIGMOD 2000. Density-based: ratio of a point's local reachability density to that of its k-nearest neighbors. Default `n_neighbors=20`.
- **One-Class SVM (OCSVM)** — Schölkopf, Platt, Shawe-Taylor, Smola, Williamson, 2001. Boundary-based: RBF kernel with `nu` controlling outlier fraction.
- **ECOD (Empirical Cumulative-distribution Outlier Detection)** — Li, Zhao, Hu, Botta, Ionescu, Chen, TKDE 2022 ([arXiv:2201.00382](https://arxiv.org/abs/2201.00382)). Per-dimension empirical CDFs; aggregate (negative) log tail probabilities across dimensions. Parameter-free.
- **COPOD (Copula-Based Outlier Detection)** — Li, Zhao, Botta, Ionescu, Hu, ICDM 2020 ([arXiv:2009.09463](https://arxiv.org/abs/2009.09463)). Empirical copula on per-dimension marginals; uses left/right/skewness-corrected tail probabilities. Parameter-free.

## Implementation Contract
Implement `CustomAnomalyDetector` in `custom_anomaly.py`:

```python
class CustomAnomalyDetector:
    def __init__(self):
        # Initialize hyperparameters and internal state
        ...

    def fit(self, X):
        # X: numpy array (n_samples, n_features), already standardized
        # (zero mean, unit variance). No labels used.
        return self

    def decision_function(self, X):
        # Return anomaly scores: numpy array (n_samples,)
        # Higher = more anomalous.
        return scores
```

Available libraries: `numpy`, `scipy` (linear algebra, statistics, spatial, optimization), `scikit-learn` (PCA, KDE, NearestNeighbors, GaussianMixture, ...), `pyod` (IForest, LOF, OCSVM, ECOD, COPOD, KNN, HBOS, PCA, LODA, SUOD, ...).

## Fixed Pipeline & Evaluation
Datasets (from ADBench / ODDS):
- **Cardio** — 1,831 samples, 21 features, ~9.6% anomalies (cardiotocography).
- **Thyroid** — 3,772 samples, 6 features, ~2.5% anomalies.
- **Satellite** — 6,435 samples, 36 features, ~31.6% anomalies (Landsat).
- **Shuttle** — 49,097 samples, 9 features, ~7.2% anomalies (NASA shuttle).

Protocol: 60/40 stratified train/test split (standard ADBench/ECOD protocol). Detector fits on train features without labels; scores are computed for test features.

Metrics (higher is better):
- **AUROC** — area under ROC curve (ranking quality).
- **F1** — F1 score at the optimal contamination threshold (decision quality after thresholding).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `scikit-learn/custom_anomaly.py`
- editable lines **160–212**




## Readable Context


### `scikit-learn/custom_anomaly.py`  [EDITABLE — lines 160–212 only]

```python
     1: """Unsupervised Anomaly Detection Benchmark for MLS-Bench.
     2: 
     3: FIXED: Data loading, evaluation pipeline, metrics computation.
     4: EDITABLE: CustomAnomalyDetector class — the agent's anomaly detection algorithm.
     5: 
     6: Usage:
     7:     ENV=cardio SEED=42 OUTPUT_DIR=./output python custom_anomaly.py
     8: """
     9: 
    10: import os
    11: import sys
    12: import json
    13: import time
    14: import warnings
    15: from pathlib import Path
    16: 
    17: import numpy as np
    18: from scipy.io import loadmat
    19: from sklearn.preprocessing import StandardScaler
    20: from sklearn.model_selection import train_test_split
    21: from sklearn.metrics import roc_auc_score, f1_score
    22: from sklearn.base import BaseEstimator
    23: 
    24: 
    25: # =====================================================================
    26: # FIXED: Configuration
    27: # =====================================================================
    28: SEED = int(os.environ.get("SEED", "42"))
    29: OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
    30: DATASET_NAME = os.environ.get("ENV", "cardio")
    31: 
    32: DATA_DIR = os.environ.get("DATA_ROOT", "/data") + "/adbench"
    33: 
    34: # Dataset file mapping
    35: DATASET_FILES = {
    36:     "cardio": "6_cardio.npz",
    37:     "thyroid": "38_thyroid.npz",
    38:     "satellite": "30_satellite.npz",
    39:     "shuttle": "32_shuttle.npz",
    40: }
    41: 
    42: TRAIN_RATIO = 0.6  # Task-local 60/40 stratified train/test split.
    43: 
    44: warnings.filterwarnings("ignore")
    45: np.random.seed(SEED)
    46: 
    47: 
    48: # =====================================================================
    49: # FIXED: Data loading
    50: # =====================================================================
    51: def load_dataset(name: str):
    52:     """Load an anomaly detection dataset.
    53: 
    54:     Returns:
    55:         X: feature matrix of shape (n_samples, n_features), float64
    56:         y: binary labels of shape (n_samples,), 0=normal 1=anomaly
    57:     """
    58:     filename = DATASET_FILES[name]
    59:     filepath = os.path.join(DATA_DIR, filename)
    60:     data = np.load(filepath, allow_pickle=True)
    61:     X = data["X"].astype(np.float64)
    62:     y = data["y"].astype(np.int32).ravel()
    63:     # Ensure binary: 0=normal, 1=anomaly
    64:     y = (y > 0).astype(np.int32)
    65:     return X, y
    66: 
    67: 
    68: # =====================================================================
    69: # FIXED: Evaluation utilities
    70: # =====================================================================
    71: def evaluate_detector(detector, X_train, X_test, y_test):
    72:     """Fit detector on training data and evaluate on test data.
    73: 
    74:     Args:
    75:         detector: an object with fit(X) and decision_function(X) methods.
    76:                   fit(X) trains on UNLABELED data (no y).
    77:                   decision_function(X) returns anomaly scores (higher = more anomalous).
    78:         X_train: training features (n_train, n_features)
    79:         X_test: test features (n_test, n_features)
    80:         y_test: test labels (n_test,), 0=normal, 1=anomaly
    81: 
    82:     Returns:
    83:         dict with 'auroc' and 'f1' metrics
    84:     """
    85:     # Fit on training data (unsupervised — no labels)
    86:     detector.fit(X_train)
    87: 
    88:     # Get anomaly scores on test data
    89:     scores = detector.decision_function(X_test)
    90: 
    91:     # AUROC
    92:     try:
    93:         auroc = roc_auc_score(y_test, scores)
    94:     except ValueError:
    95:         auroc = 0.5  # fallback if only one class present
    96: 
    97:     # F1 at optimal threshold (using test set threshold for fair comparison)
    98:     # Threshold at the contamination ratio percentile
    99:     contamination = y_test.mean()
   100:     if contamination > 0 and contamination < 1:
   101:         threshold = np.percentile(scores, 100 * (1 - contamination))
   102:         y_pred = (scores >= threshold).astype(int)
   103:     else:
   104:         y_pred = np.zeros_like(y_test)
   105: 
   106:     f1 = f1_score(y_test, y_pred, zero_division=0.0)
   107: 
   108:     return {"auroc": auroc, "f1": f1}
   109: 
   110: 
   111: def run_evaluation(detector_cls, X, y, seed):
   112:     """Run evaluation with a 60/40 stratified train/test split.
   113: 
   114:     This task uses a fixed 60/40 stratified split. It is inspired by
   115:     ADBench-style held-out evaluation, but is not the ADBench 70/30 protocol.
   116: 
   117:     Args:
   118:         detector_cls: callable that returns a fresh detector instance
   119:         X: full feature matrix
   120:         y: full label vector
   121:         seed: random seed
   122: 
   123:     Returns:
   124:         dict with auroc and f1 metrics
   125:     """
   126:     X_train, X_test, y_train, y_test = train_test_split(
   127:         X, y, test_size=1.0 - TRAIN_RATIO, stratify=y, random_state=seed,
   128:     )
   129: 
   130:     # Standardize features
   131:     scaler = StandardScaler()
   132:     X_train_scaled = scaler.fit_transform(X_train)
   133:     X_test_scaled = scaler.transform(X_test)
   134: 
   135:     # Create fresh detector and evaluate
   136:     detector = detector_cls()
   137: 
   138:     try:
   139:         metrics = evaluate_detector(detector, X_train_scaled, X_test_scaled, y_test)
   140:         print(
   141:             f"TRAIN_METRICS split=60/40 "
   142:             f"auroc={metrics['auroc']:.4f} f1={metrics['f1']:.4f}",
   143:             flush=True,
   144:         )
   145:     except Exception as e:
   146:         print(f"TRAIN_METRICS split=60/40 error={str(e)}", flush=True)
   147:         metrics = {"auroc": 0.5, "f1": 0.0}
   148: 
   149:     return {
   150:         "auroc_mean": float(metrics["auroc"]),
   151:         "auroc_std": 0.0,
   152:         "f1_mean": float(metrics["f1"]),
   153:         "f1_std": 0.0,
   154:     }
   155: 
   156: 
   157: # =====================================================================
   158: # EDITABLE: Custom Anomaly Detector (lines 160-212)
   159: # =====================================================================
   160: class CustomAnomalyDetector:
   161:     """Custom unsupervised anomaly detection algorithm.
   162: 
   163:     You MUST implement:
   164:         - __init__(self): initialize any hyperparameters and internal state
   165:         - fit(self, X): train the detector on unlabeled data X (n_samples, n_features).
   166:                         This is UNSUPERVISED — you do not receive labels.
   167:         - decision_function(self, X): return anomaly scores for X.
   168:                         Shape: (n_samples,). Higher scores = more anomalous.
   169: 
   170:     Available libraries (pre-installed):
   171:         - numpy, scipy, scikit-learn (StandardScaler, PCA, KernelDensity, etc.)
   172:         - pyod (IForest, LOF, OCSVM, ECOD, COPOD, KNN, HBOS, PCA, LODA, etc.)
   173: 
   174:     The detector will be evaluated on tabular anomaly detection benchmarks via
   175:     a 60/40 stratified train/test split, measuring AUROC and F1.
   176: 
   177:     Design considerations:
   178:         - Anomalies are rare (typically 2-30% of data)
   179:         - Feature dimensions vary (6 to 36 features)
   180:         - Dataset sizes vary (1,800 to 49,000 samples)
   181:         - Data is pre-standardized before being passed to fit/decision_function
   182:         - Your algorithm should work WITHOUT labels (unsupervised)
   183:         - Consider: density estimation, distance-based, projection-based,
   184:           ensemble methods, or hybrid approaches
   185:     """
   186: 
   187:     def __init__(self):
   188:         """Initialize the anomaly detector."""
   189:         # Default: simple Isolation Forest wrapper
   190:         from pyod.models.iforest import IForest
   191: 
   192:         self.model = IForest(random_state=SEED)
   193: 
   194:     def fit(self, X):
   195:         """Fit the detector on unlabeled training data.
   196: 
   197:         Args:
   198:             X: numpy array of shape (n_samples, n_features), standardized
   199:         """
   200:         self.model.fit(X)
   201:         return self
   202: 
   203:     def decision_function(self, X):
   204:         """Compute anomaly scores for input data.
   205: 
   206:         Args:
   207:             X: numpy array of shape (n_samples, n_features), standardized
   208: 
   209:         Returns:
   210:             scores: numpy array of shape (n_samples,), higher = more anomalous
   211:         """
   212:         return self.model.decision_function(X)
   213: 
   214: 
   215: # =====================================================================
   216: # FIXED: Main evaluation script
   217: # =====================================================================
   218: if __name__ == "__main__":
   219:     os.makedirs(OUTPUT_DIR, exist_ok=True)
   220: 
   221:     print(f"Dataset: {DATASET_NAME}, Seed: {SEED}", flush=True)
   222: 
   223:     # Load data
   224:     X, y = load_dataset(DATASET_NAME)
   225:     print(
   226:         f"Loaded {DATASET_NAME}: {X.shape[0]} samples, {X.shape[1]} features, "
   227:         f"{y.mean()*100:.1f}% anomalies",
   228:         flush=True,
   229:     )
   230: 
   231:     # Run evaluation
   232:     start_time = time.time()
   233:     results = run_evaluation(
   234:         detector_cls=CustomAnomalyDetector,
   235:         X=X,
   236:         y=y,
   237:         seed=SEED,
   238:     )
   239:     elapsed = time.time() - start_time
   240: 
   241:     print(f"\nResults on {DATASET_NAME} (seed={SEED}):", flush=True)
   242:     print(
   243:         f"  AUROC: {results['auroc_mean']:.4f} +/- {results['auroc_std']:.4f}",
   244:         flush=True,
   245:     )
   246:     print(
   247:         f"  F1:    {results['f1_mean']:.4f} +/- {results['f1_std']:.4f}",
   248:         flush=True,
   249:     )
   250:     print(f"  Time:  {elapsed:.1f}s", flush=True)
   251: 
   252:     # Output final metrics for parser
   253:     print(
   254:         f"TEST_METRICS auroc={results['auroc_mean']:.6f} f1={results['f1_mean']:.6f}",
   255:         flush=True,
   256:     )
```

## Parameter Budget

This task enforces a parameter-count cap. Your edits will be rejected if
the resulting model exceeds **1.05×** the strongest
baseline's parameter count. The check runs automatically inside the eval
scripts — you don't need to invoke it.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `isolation_forest` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 160–183:
   157: # =====================================================================
   158: # EDITABLE: Custom Anomaly Detector (lines 160-212)
   159: # =====================================================================
   160: class CustomAnomalyDetector:
   161:     """Isolation Forest anomaly detector.
   162: 
   163:     Ensemble of random isolation trees. Anomaly score is based on the
   164:     average path length to isolate each sample.
   165:     """
   166: 
   167:     def __init__(self):
   168:         from pyod.models.iforest import IForest
   169: 
   170:         self.model = IForest(
   171:             n_estimators=100,
   172:             max_samples="auto",
   173:             contamination=0.1,
   174:             random_state=SEED,
   175:         )
   176: 
   177:     def fit(self, X):
   178:         self.model.fit(X)
   179:         return self
   180: 
   181:     def decision_function(self, X):
   182:         return self.model.decision_function(X)
   183: 
   184: 
   185: 
   186: # =====================================================================
```

### `lof` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 160–188:
   157: # =====================================================================
   158: # EDITABLE: Custom Anomaly Detector (lines 160-212)
   159: # =====================================================================
   160: class CustomAnomalyDetector:
   161:     """Local Outlier Factor anomaly detector (ADBench protocol).
   162: 
   163:     Applies MinMax normalization internally to match the preprocessing
   164:     used by ADBench (data_generator.py: MinMaxScaler().fit(X_train)).
   165:     LOF is density-based and extremely sensitive to feature scaling,
   166:     so this is required to reproduce the Table D4 numbers.
   167:     """
   168: 
   169:     def __init__(self):
   170:         from pyod.models.lof import LOF
   171: 
   172:         # PyOD defaults (matches ADBench with no hyperparameter tuning):
   173:         # n_neighbors=20, algorithm='auto', metric='minkowski', p=2,
   174:         # contamination=0.1.
   175:         self.model = LOF()
   176:         self._scaler = None
   177: 
   178:     def fit(self, X):
   179:         from sklearn.preprocessing import MinMaxScaler
   180:         self._scaler = MinMaxScaler()
   181:         Xs = self._scaler.fit_transform(X)
   182:         self.model.fit(Xs)
   183:         return self
   184: 
   185:     def decision_function(self, X):
   186:         Xs = self._scaler.transform(X)
   187:         return self.model.decision_function(Xs)
   188: 
   189: 
   190: 
   191: # =====================================================================
```

### `ocsvm` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 160–185:
   157: # =====================================================================
   158: # EDITABLE: Custom Anomaly Detector (lines 160-212)
   159: # =====================================================================
   160: class CustomAnomalyDetector:
   161:     """One-Class SVM anomaly detector (ADBench protocol).
   162: 
   163:     Applies MinMax normalization internally to match ADBench's
   164:     preprocessing. Uses PyOD defaults: kernel='rbf', nu=0.5,
   165:     gamma='auto' (= 1/n_features).
   166:     """
   167: 
   168:     def __init__(self):
   169:         from pyod.models.ocsvm import OCSVM
   170: 
   171:         # PyOD default: kernel='rbf', nu=0.5, gamma='auto'.
   172:         self.model = OCSVM()
   173:         self._scaler = None
   174: 
   175:     def fit(self, X):
   176:         from sklearn.preprocessing import MinMaxScaler
   177:         self._scaler = MinMaxScaler()
   178:         Xs = self._scaler.fit_transform(X)
   179:         self.model.fit(Xs)
   180:         return self
   181: 
   182:     def decision_function(self, X):
   183:         Xs = self._scaler.transform(X)
   184:         return self.model.decision_function(Xs)
   185: 
   186: 
   187: 
   188: # =====================================================================
```

### `ecod` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 160–174:
   157: # =====================================================================
   158: # EDITABLE: Custom Anomaly Detector (lines 160-212)
   159: # =====================================================================
   160: class CustomAnomalyDetector:
   161:     """ECOD anomaly detector (PyOD default, matches ADBench)."""
   162: 
   163:     def __init__(self):
   164:         from pyod.models.ecod import ECOD
   165: 
   166:         self.model = ECOD()
   167: 
   168:     def fit(self, X):
   169:         self.model.fit(X)
   170:         return self
   171: 
   172:     def decision_function(self, X):
   173:         return self.model.decision_function(X)
   174: 
   175: 
   176: 
   177: # =====================================================================
```

### `copod` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 160–178:
   157: # =====================================================================
   158: # EDITABLE: Custom Anomaly Detector (lines 160-212)
   159: # =====================================================================
   160: class CustomAnomalyDetector:
   161:     """COPOD: Copula-Based Outlier Detection.
   162: 
   163:     Parameter-free method using empirical copula functions to model
   164:     the joint tail probability of observations across features.
   165:     """
   166: 
   167:     def __init__(self):
   168:         from pyod.models.copod import COPOD
   169: 
   170:         self.model = COPOD(contamination=0.1)
   171: 
   172:     def fit(self, X):
   173:         self.model.fit(X)
   174:         return self
   175: 
   176:     def decision_function(self, X):
   177:         return self.model.decision_function(X)
   178: 
   179: 
   180: 
   181: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
