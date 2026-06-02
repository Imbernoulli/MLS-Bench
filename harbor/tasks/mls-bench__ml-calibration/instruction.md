# MLS-Bench: ml-calibration

# Probability Calibration Method Design

## Research Question
Design a novel post-hoc probability calibration method that maps a classifier's raw confidence estimates into well-calibrated probabilities. The base classifier and train/calibration/test splits are fixed; the contribution is the *calibration mapping itself*, learned only from a held-out calibration set.

## Background
A well-calibrated model satisfies: among all predictions where the model outputs probability p for a class, the empirical fraction that are correct is approximately p. Modern neural networks, GBMs, RFs, and SVMs are routinely miscalibrated.

Reference baselines:
- **Platt scaling** — Platt, 1999. Fit a sigmoid `1 / (1 + exp(a*x + b))` on classifier scores via maximum likelihood. Designed for SVM margins.
- **Isotonic regression** — Zadrozny & Elkan, 2002. Non-parametric monotonic mapping; can overfit on small calibration sets.
- **Temperature scaling** — Guo, Pleiss, Sun, Weinberger, ICML 2017 ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Single scalar temperature `T` divides logits before softmax; fit by minimizing NLL on calibration set. Preserves accuracy (argmax invariant).

## Implementation Contract
Implement `CalibrationMethod` in `custom_calibration.py`:

```python
class CalibrationMethod(BaseEstimator):
    def fit(self, probs, labels):
        # probs: (n,) for binary (positive-class probability)
        #        or (n, C) for multiclass (rows sum to 1)
        # labels: (n,) integer class labels
        return self

    def predict_proba(self, probs):
        # Returns calibrated probabilities of the same shape as input.
        return calibrated_probs
```

Available imports: `numpy`, `scipy` (`optimize`, `interpolate`, `special`), `sklearn`. The output must remain a valid probability distribution (non-negative, sums to 1 for multiclass).

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `scikit-learn/custom_calibration.py`
- editable lines **45–102**




## Readable Context


### `scikit-learn/custom_calibration.py`  [EDITABLE — lines 45–102 only]

```python
     1: """ML Calibration Benchmark.
     2: 
     3: Evaluate post-hoc probability calibration methods across different classifiers
     4: and datasets.
     5: 
     6: FIXED: Classifier training, data loading, evaluation metrics, train/calibrate/test split.
     7: EDITABLE: CalibrationMethod class (fit + predict_proba).
     8: 
     9: Usage:
    10:     python scikit-learn/custom_calibration.py \
    11:         --classifier rf --dataset mnist --seed 42
    12: """
    13: 
    14: import argparse
    15: import math
    16: import os
    17: import warnings
    18: 
    19: import numpy as np
    20: from scipy import optimize, interpolate, special
    21: from sklearn.base import BaseEstimator
    22: from sklearn.datasets import (
    23:     fetch_openml,
    24:     load_breast_cancer,
    25: )
    26: from sklearn.ensemble import (
    27:     GradientBoostingClassifier,
    28:     RandomForestClassifier,
    29: )
    30: from sklearn.linear_model import LogisticRegression
    31: from sklearn.metrics import brier_score_loss, log_loss
    32: from sklearn.model_selection import train_test_split
    33: from sklearn.neural_network import MLPClassifier
    34: from sklearn.preprocessing import LabelBinarizer, label_binarize
    35: from sklearn.svm import SVC
    36: 
    37: warnings.filterwarnings("ignore")
    38: 
    39: 
    40: # ============================================================================
    41: # Calibration Method (EDITABLE)
    42: # ============================================================================
    43: 
    44: # -- EDITABLE REGION START (lines 45-102) ------------------------------------
    45: class CalibrationMethod(BaseEstimator):
    46:     """Post-hoc probability calibration method.
    47: 
    48:     Given a trained classifier's uncalibrated probability outputs, learn a
    49:     calibration mapping that produces well-calibrated probabilities.
    50: 
    51:     For binary classification, fit() receives probabilities for the positive
    52:     class. For multiclass, it receives the full probability matrix.
    53: 
    54:     Interface:
    55:         fit(probs, labels):
    56:             probs: np.ndarray, shape (n_samples,) for binary or
    57:                    (n_samples, n_classes) for multiclass.
    58:                    Uncalibrated probability outputs from a classifier
    59:                    on the calibration set.
    60:             labels: np.ndarray, shape (n_samples,) integer class labels.
    61: 
    62:         predict_proba(probs) -> np.ndarray:
    63:             probs: same format as fit().
    64:             Returns calibrated probabilities, same shape as input.
    65:             For binary: 1-D array of positive-class probabilities in [0, 1].
    66:             For multiclass: 2-D array (n_samples, n_classes), rows sum to 1.
    67: 
    68:     Design considerations:
    69:         - Parametric vs non-parametric calibration mappings
    70:         - Monotonicity preservation (calibration should not reorder predictions)
    71:         - Overfitting on small calibration sets
    72:         - Multiclass extension strategy (per-class, matrix, or joint)
    73:         - Binning vs continuous calibration functions
    74:         - Regularization to prevent extreme probability outputs
    75:     """
    76: 
    77:     def __init__(self):
    78:         self.is_binary = None
    79: 
    80:     def fit(self, probs, labels):
    81:         """Fit calibration mapping on held-out calibration data.
    82: 
    83:         Default: identity (no calibration).
    84:         """
    85:         if probs.ndim == 1:
    86:             self.is_binary = True
    87:         else:
    88:             self.is_binary = False
    89:         return self
    90: 
    91:     def predict_proba(self, probs):
    92:         """Apply calibration mapping to produce calibrated probabilities.
    93: 
    94:         Default: return uncalibrated probabilities unchanged.
    95:         """
    96:         if self.is_binary:
    97:             return np.clip(probs, 0, 1)
    98:         else:
    99:             # Ensure rows sum to 1
   100:             probs = np.clip(probs, 1e-15, 1.0)
   101:             probs = probs / probs.sum(axis=1, keepdims=True)
   102:             return probs
   103: # -- EDITABLE REGION END (lines 45-102) --------------------------------------
   104: 
   105: 
   106: # ============================================================================
   107: # Evaluation Metrics (FIXED)
   108: # ============================================================================
   109: 
   110: def expected_calibration_error(probs, labels, n_bins=15):
   111:     """Compute Expected Calibration Error (ECE).
   112: 
   113:     For binary: probs is 1-D (positive class probability).
   114:     For multiclass: probs is 2-D, we compute ECE on the max-class probability
   115:     and whether the argmax prediction is correct.
   116: 
   117:     Lower is better.
   118:     """
   119:     if probs.ndim == 2:
   120:         confidences = np.max(probs, axis=1)
   121:         predictions = np.argmax(probs, axis=1)
   122:         accuracies = (predictions == labels).astype(float)
   123:     else:
   124:         confidences = np.where(probs >= 0.5, probs, 1 - probs)
   125:         predictions = (probs >= 0.5).astype(int)
   126:         accuracies = (predictions == labels).astype(float)
   127: 
   128:     bin_boundaries = np.linspace(0, 1, n_bins + 1)
   129:     ece = 0.0
   130:     for i in range(n_bins):
   131:         lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
   132:         in_bin = (confidences > lo) & (confidences <= hi)
   133:         prop = in_bin.mean()
   134:         if prop > 0:
   135:             avg_conf = confidences[in_bin].mean()
   136:             avg_acc = accuracies[in_bin].mean()
   137:             ece += prop * abs(avg_acc - avg_conf)
   138:     return ece
   139: 
   140: 
   141: def compute_brier_score(probs, labels, n_classes):
   142:     """Compute Brier score (multi-class generalization).
   143: 
   144:     Lower is better. Range [0, 2] for multiclass, [0, 1] for binary.
   145:     """
   146:     if n_classes == 2 and probs.ndim == 1:
   147:         return brier_score_loss(labels, probs)
   148:     else:
   149:         lb = LabelBinarizer()
   150:         lb.classes_ = np.arange(n_classes)
   151:         labels_onehot = lb.transform(labels)
   152:         if n_classes == 2:
   153:             labels_onehot = np.column_stack([1 - labels_onehot, labels_onehot])
   154:         if probs.ndim == 1:
   155:             probs = np.column_stack([1 - probs, probs])
   156:         return np.mean(np.sum((probs - labels_onehot) ** 2, axis=1))
   157: 
   158: 
   159: def compute_nll(probs, labels, n_classes):
   160:     """Compute negative log-likelihood (cross-entropy).
   161: 
   162:     Lower is better.
   163:     """
   164:     if probs.ndim == 1:
   165:         probs_2d = np.column_stack([1 - probs, probs])
   166:     else:
   167:         probs_2d = probs
   168:     probs_2d = np.clip(probs_2d, 1e-15, 1 - 1e-15)
   169:     return log_loss(labels, probs_2d, labels=np.arange(n_classes))
   170: 
   171: 
   172: # ============================================================================
   173: # Data Loading (FIXED)
   174: # ============================================================================
   175: 
   176: def load_dataset(name, seed=42):
   177:     """Load and split dataset into train/calibrate/test.
   178: 
   179:     Split ratios: 60% train, 20% calibrate, 20% test.
   180:     Returns: X_train, y_train, X_cal, y_cal, X_test, y_test, n_classes
   181:     """
   182:     if name == "mnist":
   183:         data = fetch_openml("mnist_784", version=1, data_home=os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn"),
   184:                             parser="auto", as_frame=False)
   185:         X, y = data["data"].astype(np.float32), data["target"].astype(int)
   186:         # Subsample to 20000 for speed
   187:         rng = np.random.RandomState(seed)
   188:         idx = rng.choice(len(X), 20000, replace=False)
   189:         X, y = X[idx], y[idx]
   190:         X = X / 255.0
   191:         n_classes = 10
   192:     elif name == "fashion_mnist":
   193:         data = fetch_openml("Fashion-MNIST", version=1, data_home=os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn"),
   194:                             parser="auto", as_frame=False)
   195:         X, y = data["data"].astype(np.float32), data["target"].astype(int)
   196:         rng = np.random.RandomState(seed)
   197:         idx = rng.choice(len(X), 20000, replace=False)
   198:         X, y = X[idx], y[idx]
   199:         X = X / 255.0
   200:         n_classes = 10
   201:     elif name == "breast_cancer":
   202:         data = load_breast_cancer()
   203:         X, y = data["data"].astype(np.float32), data["target"].astype(int)
   204:         # Standardize features
   205:         mean = X.mean(axis=0)
   206:         std = X.std(axis=0) + 1e-8
   207:         X = (X - mean) / std
   208:         n_classes = 2
   209:     elif name == "madelon":
   210:         data = fetch_openml("madelon", version=1, data_home=os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn"),
   211:                             parser="auto", as_frame=False)
   212:         X = data["data"].astype(np.float32)
   213:         y_raw = data["target"]
   214:         # madelon labels can be {1, 2} or {-1, 1}; normalize to {0, 1}
   215:         unique_labels = np.unique(y_raw)
   216:         label_map = {lab: i for i, lab in enumerate(sorted(unique_labels))}
   217:         y = np.array([label_map[lab] for lab in y_raw])
   218:         mean = X.mean(axis=0)
   219:         std = X.std(axis=0) + 1e-8
   220:         X = (X - mean) / std
   221:         n_classes = 2
   222:     else:
   223:         raise ValueError(f"Unknown dataset: {name}")
   224: 
   225:     # Split: train 60%, calibrate 20%, test 20%
   226:     X_trainval, X_test, y_trainval, y_test = train_test_split(
   227:         X, y, test_size=0.2, random_state=seed, stratify=y
   228:     )
   229:     X_train, X_cal, y_train, y_cal = train_test_split(
   230:         X_trainval, y_trainval, test_size=0.25, random_state=seed, stratify=y_trainval
   231:     )
   232:     return X_train, y_train, X_cal, y_cal, X_test, y_test, n_classes
   233: 
   234: 
   235: # ============================================================================
   236: # Classifier Training (FIXED)
   237: # ============================================================================
   238: 
   239: def build_classifier(name, seed=42):
   240:     """Build an uncalibrated classifier."""
   241:     if name == "rf":
   242:         return RandomForestClassifier(
   243:             n_estimators=200, max_depth=None, min_samples_leaf=2,
   244:             random_state=seed, n_jobs=-1,
   245:         )
   246:     elif name == "svm":
   247:         return SVC(
   248:             kernel="rbf", C=10.0, gamma="scale",
   249:             probability=True, random_state=seed,
   250:         )
   251:     elif name == "mlp":
   252:         return MLPClassifier(
   253:             hidden_layer_sizes=(256, 128), activation="relu",
   254:             max_iter=200, early_stopping=True, validation_fraction=0.1,
   255:             random_state=seed,
   256:         )
   257:     elif name == "gbm":
   258:         return GradientBoostingClassifier(
   259:             n_estimators=200, max_depth=5, learning_rate=0.1,
   260:             random_state=seed,
   261:         )
   262:     elif name == "lr":
   263:         return LogisticRegression(
   264:             C=1.0, max_iter=1000, random_state=seed,
   265:         )
   266:     else:
   267:         raise ValueError(f"Unknown classifier: {name}")
   268: 
   269: 
   270: # ============================================================================
   271: # Main Pipeline (FIXED)
   272: # ============================================================================
   273: 
   274: def main():
   275:     parser = argparse.ArgumentParser(description="ML Calibration Benchmark")
   276:     parser.add_argument("--classifier", type=str, required=True,
   277:                         choices=["rf", "svm", "mlp", "gbm", "lr"])
   278:     parser.add_argument("--dataset", type=str, required=True,
   279:                         choices=["mnist", "fashion_mnist", "breast_cancer", "madelon"])
   280:     parser.add_argument("--seed", type=int, default=42)
   281:     parser.add_argument("--output-dir", type=str, default=".")
   282:     args = parser.parse_args()
   283: 
   284:     np.random.seed(args.seed)
   285: 
   286:     # Load data
   287:     print(f"Loading dataset: {args.dataset}", flush=True)
   288:     X_train, y_train, X_cal, y_cal, X_test, y_test, n_classes = load_dataset(
   289:         args.dataset, seed=args.seed
   290:     )
   291:     print(f"  Train: {X_train.shape}, Cal: {X_cal.shape}, Test: {X_test.shape}, "
   292:           f"Classes: {n_classes}", flush=True)
   293: 
   294:     # Train classifier
   295:     print(f"Training classifier: {args.classifier}", flush=True)
   296:     clf = build_classifier(args.classifier, seed=args.seed)
   297:     clf.fit(X_train, y_train)
   298: 
   299:     train_acc = clf.score(X_train, y_train)
   300:     test_acc = clf.score(X_test, y_test)
   301:     print(f"TRAIN_METRICS: train_acc={train_acc:.4f} test_acc={test_acc:.4f}", flush=True)
   302: 
   303:     # Get uncalibrated probabilities
   304:     if n_classes == 2:
   305:         cal_probs_uncal = clf.predict_proba(X_cal)[:, 1]
   306:         test_probs_uncal = clf.predict_proba(X_test)[:, 1]
   307:     else:
   308:         cal_probs_uncal = clf.predict_proba(X_cal)
   309:         test_probs_uncal = clf.predict_proba(X_test)
   310: 
   311:     # Evaluate BEFORE calibration
   312:     ece_before = expected_calibration_error(test_probs_uncal, y_test)
   313:     brier_before = compute_brier_score(test_probs_uncal, y_test, n_classes)
   314:     nll_before = compute_nll(test_probs_uncal, y_test, n_classes)
   315:     print(f"TRAIN_METRICS: before_calibration ECE={ece_before:.6f} "
   316:           f"Brier={brier_before:.6f} NLL={nll_before:.6f}", flush=True)
   317: 
   318:     # Fit calibration method on calibration set
   319:     print("Fitting calibration method...", flush=True)
   320:     calibrator = CalibrationMethod()
   321:     calibrator.fit(cal_probs_uncal, y_cal)
   322: 
   323:     # Apply calibration to test set
   324:     test_probs_cal = calibrator.predict_proba(test_probs_uncal)
   325: 
   326:     # Validate output shape and values
   327:     assert test_probs_cal.shape == test_probs_uncal.shape, \
   328:         f"Shape mismatch: {test_probs_cal.shape} vs {test_probs_uncal.shape}"
   329:     if test_probs_cal.ndim == 2:
   330:         row_sums = test_probs_cal.sum(axis=1)
   331:         assert np.allclose(row_sums, 1.0, atol=1e-3), \
   332:             f"Rows do not sum to 1: min={row_sums.min():.4f}, max={row_sums.max():.4f}"
   333: 
   334:     # Evaluate AFTER calibration
   335:     ece_after = expected_calibration_error(test_probs_cal, y_test)
   336:     brier_after = compute_brier_score(test_probs_cal, y_test, n_classes)
   337:     nll_after = compute_nll(test_probs_cal, y_test, n_classes)
   338: 
   339:     print(f"TRAIN_METRICS: after_calibration ECE={ece_after:.6f} "
   340:           f"Brier={brier_after:.6f} NLL={nll_after:.6f}", flush=True)
   341: 
   342:     # Final metrics (improvement = reduction in error)
   343:     print(f"TEST_METRICS: ECE={ece_after:.6f} Brier={brier_after:.6f} "
   344:           f"NLL={nll_after:.6f}", flush=True)
   345: 
   346: 
   347: if __name__ == "__main__":
   348:     main()
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


### `platt_scaling` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_calibration.py`:

```python
Lines 45–108:
    42: # ============================================================================
    43: 
    44: # -- EDITABLE REGION START (lines 45-102) ------------------------------------
    45: class CalibrationMethod(BaseEstimator):
    46:     """Platt Scaling (logistic/sigmoid calibration).
    47: 
    48:     Fits A*f + B through a sigmoid for each class, where f is the
    49:     uncalibrated probability (log-odds transformed).
    50:     """
    51: 
    52:     def __init__(self):
    53:         self.is_binary = None
    54:         self.a_ = None
    55:         self.b_ = None
    56: 
    57:     def fit(self, probs, labels):
    58:         if probs.ndim == 1:
    59:             self.is_binary = True
    60:             self.a_, self.b_ = self._fit_sigmoid(probs, labels)
    61:         else:
    62:             self.is_binary = False
    63:             n_classes = probs.shape[1]
    64:             self.a_ = np.zeros(n_classes)
    65:             self.b_ = np.zeros(n_classes)
    66:             for c in range(n_classes):
    67:                 binary_labels = (labels == c).astype(float)
    68:                 self.a_[c], self.b_[c] = self._fit_sigmoid(probs[:, c], binary_labels)
    69:         return self
    70: 
    71:     def _fit_sigmoid(self, probs, labels):
    72:         """Fit sigmoid parameters A, B: calibrated = 1 / (1 + exp(A*f + B))."""
    73:         # Transform to log-odds space, clip to avoid inf
    74:         eps = 1e-12
    75:         f = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
    76: 
    77:         # Target probabilities (Platt's target encoding)
    78:         n_pos = labels.sum()
    79:         n_neg = len(labels) - n_pos
    80:         t_pos = (n_pos + 1) / (n_pos + 2) if n_pos > 0 else 0.5
    81:         t_neg = 1 / (n_neg + 2) if n_neg > 0 else 0.5
    82:         target = np.where(labels > 0.5, t_pos, t_neg)
    83: 
    84:         def objective(params):
    85:             a, b = params
    86:             p = 1.0 / (1.0 + np.exp(a * f + b))
    87:             p = np.clip(p, eps, 1 - eps)
    88:             loss = -(target * np.log(p) + (1 - target) * np.log(1 - p)).mean()
    89:             return loss
    90: 
    91:         result = optimize.minimize(objective, x0=[1.0, 0.0], method="L-BFGS-B")
    92:         return result.x[0], result.x[1]
    93: 
    94:     def predict_proba(self, probs):
    95:         eps = 1e-12
    96:         if self.is_binary:
    97:             f = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
    98:             calibrated = 1.0 / (1.0 + np.exp(self.a_ * f + self.b_))
    99:             return np.clip(calibrated, 0, 1)
   100:         else:
   101:             n_classes = probs.shape[1]
   102:             calibrated = np.zeros_like(probs)
   103:             for c in range(n_classes):
   104:                 f = np.log(np.clip(probs[:, c], eps, 1 - eps) /
   105:                            np.clip(1 - probs[:, c], eps, 1 - eps))
   106:                 calibrated[:, c] = 1.0 / (1.0 + np.exp(self.a_[c] * f + self.b_[c]))
   107:             calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
   108:             return calibrated
   109: # -- EDITABLE REGION END (lines 45-102) --------------------------------------
   110: 
   111: 
```

### `temperature_scaling` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_calibration.py`:

```python
Lines 45–97:
    42: # ============================================================================
    43: 
    44: # -- EDITABLE REGION START (lines 45-102) ------------------------------------
    45: class CalibrationMethod(BaseEstimator):
    46:     """Temperature Scaling calibration.
    47: 
    48:     Learns a single temperature T that scales all logits: softmax(z/T).
    49:     Optimized by minimizing NLL on the calibration set.
    50:     """
    51: 
    52:     def __init__(self):
    53:         self.is_binary = None
    54:         self.temperature_ = 1.0
    55: 
    56:     def fit(self, probs, labels):
    57:         if probs.ndim == 1:
    58:             self.is_binary = True
    59:             # Convert to 2-class logits
    60:             eps = 1e-15
    61:             p = np.clip(probs, eps, 1 - eps)
    62:             logits = np.column_stack([np.log(1 - p), np.log(p)])
    63:         else:
    64:             self.is_binary = False
    65:             eps = 1e-15
    66:             logits = np.log(np.clip(probs, eps, 1.0))
    67: 
    68:         def nll(T):
    69:             T_val = max(T[0], 0.01)
    70:             scaled = logits / T_val
    71:             # Numerically stable softmax
    72:             scaled = scaled - scaled.max(axis=1, keepdims=True)
    73:             exp_scaled = np.exp(scaled)
    74:             log_probs = scaled - np.log(exp_scaled.sum(axis=1, keepdims=True))
    75:             return -log_probs[np.arange(len(labels)), labels.astype(int)].mean()
    76: 
    77:         result = optimize.minimize(nll, x0=[1.5], bounds=[(0.01, 20.0)],
    78:                                    method="L-BFGS-B")
    79:         self.temperature_ = max(result.x[0], 0.01)
    80:         return self
    81: 
    82:     def predict_proba(self, probs):
    83:         eps = 1e-15
    84:         if self.is_binary:
    85:             p = np.clip(probs, eps, 1 - eps)
    86:             logits = np.column_stack([np.log(1 - p), np.log(p)])
    87:         else:
    88:             logits = np.log(np.clip(probs, eps, 1.0))
    89: 
    90:         scaled = logits / self.temperature_
    91:         scaled = scaled - scaled.max(axis=1, keepdims=True)
    92:         exp_scaled = np.exp(scaled)
    93:         calibrated = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    94: 
    95:         if self.is_binary:
    96:             return calibrated[:, 1]
    97:         return calibrated
    98: # -- EDITABLE REGION END (lines 45-102) --------------------------------------
    99: 
   100: 
```

### `isotonic_regression` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_calibration.py`:

```python
Lines 45–86:
    42: # ============================================================================
    43: 
    44: # -- EDITABLE REGION START (lines 45-102) ------------------------------------
    45: class CalibrationMethod(BaseEstimator):
    46:     """Isotonic Regression calibration.
    47: 
    48:     Fits a non-parametric, monotonically non-decreasing function
    49:     from uncalibrated probabilities to calibrated ones.
    50:     """
    51: 
    52:     def __init__(self):
    53:         self.is_binary = None
    54:         self.calibrators_ = None
    55: 
    56:     def fit(self, probs, labels):
    57:         from sklearn.isotonic import IsotonicRegression as IR
    58: 
    59:         if probs.ndim == 1:
    60:             self.is_binary = True
    61:             iso = IR(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    62:             iso.fit(probs, labels)
    63:             self.calibrators_ = [iso]
    64:         else:
    65:             self.is_binary = False
    66:             n_classes = probs.shape[1]
    67:             self.calibrators_ = []
    68:             for c in range(n_classes):
    69:                 binary_labels = (labels == c).astype(float)
    70:                 iso = IR(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    71:                 iso.fit(probs[:, c], binary_labels)
    72:                 self.calibrators_.append(iso)
    73:         return self
    74: 
    75:     def predict_proba(self, probs):
    76:         if self.is_binary:
    77:             calibrated = self.calibrators_[0].predict(probs)
    78:             return np.clip(calibrated, 0, 1)
    79:         else:
    80:             n_classes = probs.shape[1]
    81:             calibrated = np.zeros_like(probs)
    82:             for c in range(n_classes):
    83:                 calibrated[:, c] = self.calibrators_[c].predict(probs[:, c])
    84:             calibrated = np.clip(calibrated, 1e-15, None)
    85:             calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    86:             return calibrated
    87: # -- EDITABLE REGION END (lines 45-102) --------------------------------------
    88: 
    89: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
