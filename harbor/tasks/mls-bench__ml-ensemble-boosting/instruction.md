# MLS-Bench: ml-ensemble-boosting

# Ensemble Boosting Strategy Design

## Research Question
Design a novel sample-weighting and update strategy for boosting that improves over standard methods (AdaBoost, gradient boosting, XGBoost-style Newton update) across both classification and regression tasks. The contribution is the *strategy itself* (how sample weights are initialized and updated, what pseudo-targets each weak learner fits, how each learner is weighted), with shallow decision trees as the fixed weak learner.

## Background
Boosting builds an ensemble of weak learners sequentially, each round trying to correct errors left by previous rounds. Key design axes:
- **Pseudo-target computation**: original labels (AdaBoost), negative gradients (gradient boosting), Newton-step targets using second-order information (XGBoost).
- **Learner weighting**: from weighted error (AdaBoost), fixed at 1.0 with learning rate shrinkage (gradient boosting), via line search / Newton optimization (XGBoost).
- **Sample reweighting**: exponential reweighting of misclassified samples (AdaBoost) vs. uniform weights with pseudo-residual fitting (gradient methods).

Reference baselines:
- **AdaBoost** — Freund & Schapire, JCSS 1997 ([paper](https://www.sciencedirect.com/science/article/pii/S002200009791504X)). Exponential loss; alpha = `0.5 * log((1-err)/err)`; multiplicative reweighting `w_i *= exp(alpha * 1[y_i ≠ h(x_i)])` (binary classification).
- **Gradient boosting** — Friedman, Annals of Statistics 2001. Fit each new tree to the negative gradient of the loss at current predictions; constant learner weight 1.0 with global learning-rate shrinkage (here `lr=0.1`).
- **XGBoost-style (second-order)** — Chen & Guestrin, KDD 2016 ([arXiv:1603.02754](https://arxiv.org/abs/1603.02754)). Use both gradient `g` and Hessian `h` of the loss; pseudo-targets and leaf values follow the Newton step `-g/h`.

## Implementation Contract
Modify `BoostingStrategy` in `scikit-learn/custom_boosting.py`:

```python
class BoostingStrategy:
    def init_weights(self, n_samples):
        # Initialize sample weights (should sum to 1).
        ...

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        # Pseudo-targets the next weak learner will fit.
        ...

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        # Alpha for the just-fitted learner.
        ...

    def update_weights(self, sample_weights, learner, X, y,
                       pseudo_targets, alpha, round_idx):
        # Sample weights for the next round.
        ...
```

Available context: true labels, current ensemble predictions, sample weights, fitted learner (`learner.predict(X)`), round index, config dict with dataset metadata. Available imports in the FIXED section: `numpy`, `sklearn.tree`, `sklearn.metrics`, `sklearn.datasets`, `sklearn.model_selection`.

## Fixed Pipeline
- 200 boosting rounds, base learner = `DecisionTree(max_depth=3)`, learning rate `0.1`.
- Your strategy is evaluated on both classification and regression tabular datasets.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `scikit-learn/custom_boosting.py`
- editable lines **147–256**




## Readable Context


### `scikit-learn/custom_boosting.py`  [EDITABLE — lines 147–256 only]

```python
     1: """ML Ensemble Boosting Benchmark.
     2: 
     3: Train gradient-boosted ensembles of decision stumps/trees on tabular datasets
     4: to evaluate novel sample weighting / boosting update strategies.
     5: 
     6: FIXED: Data loading, base learner (decision trees), prediction aggregation,
     7:        evaluation loop, CLI.
     8: EDITABLE: BoostingStrategy class — compute_sample_weights() and update_weights().
     9: 
    10: Usage:
    11:     python custom_boosting.py --dataset breast_cancer --task classification --seed 42
    12:     python custom_boosting.py --dataset diabetes --task regression --seed 42
    13: """
    14: 
    15: import argparse
    16: import math
    17: import os
    18: import time
    19: from abc import ABC, abstractmethod
    20: 
    21: import numpy as np
    22: from sklearn.datasets import (
    23:     fetch_california_housing,
    24:     load_breast_cancer,
    25:     load_diabetes,
    26: )
    27: from sklearn.model_selection import train_test_split
    28: from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    29: from sklearn.metrics import accuracy_score, mean_squared_error
    30: 
    31: 
    32: # ============================================================================
    33: # FIXED — Data loading and preprocessing
    34: # ============================================================================
    35: 
    36: def load_dataset(name):
    37:     """Load a dataset by name. Returns X, y, task_type."""
    38:     if name == "breast_cancer":
    39:         data = load_breast_cancer()
    40:         return data.data, data.target, "classification"
    41:     elif name == "diabetes":
    42:         data = load_diabetes()
    43:         return data.data, data.target, "regression"
    44:     elif name == "california_housing":
    45:         data = fetch_california_housing(data_home=os.environ.get("SKLEARN_DATA_HOME"))
    46:         return data.data, data.target, "regression"
    47:     else:
    48:         raise ValueError(f"Unknown dataset: {name}")
    49: 
    50: 
    51: def normalize_features(X_train, X_test):
    52:     """Standardize features to zero mean and unit variance."""
    53:     mean = X_train.mean(axis=0)
    54:     std = X_train.std(axis=0) + 1e-8
    55:     return (X_train - mean) / std, (X_test - mean) / std
    56: 
    57: 
    58: # ============================================================================
    59: # FIXED — Base learner interface
    60: # ============================================================================
    61: 
    62: class BaseLearner:
    63:     """Wrapper around sklearn decision tree as weak learner."""
    64: 
    65:     def __init__(self, task_type, max_depth=1, random_state=None):
    66:         self.task_type = task_type
    67:         if task_type == "classification":
    68:             self.tree = DecisionTreeClassifier(
    69:                 max_depth=max_depth, random_state=random_state,
    70:             )
    71:         else:
    72:             self.tree = DecisionTreeRegressor(
    73:                 max_depth=max_depth, random_state=random_state,
    74:             )
    75: 
    76:     def fit(self, X, y, sample_weight=None):
    77:         self.tree.fit(X, y, sample_weight=sample_weight)
    78:         return self
    79: 
    80:     def predict(self, X):
    81:         return self.tree.predict(X)
    82: 
    83: 
    84: # ============================================================================
    85: # FIXED — Ensemble prediction and evaluation
    86: # ============================================================================
    87: 
    88: def ensemble_predict(learners, alphas, learner_modes, X, task_type,
    89:                      learning_rate=0.1):
    90:     """Predict using the ensemble.
    91: 
    92:     For classification:
    93:       - Discrete learners (AdaBoost-style): weighted majority vote with {-1,+1} coding
    94:       - Continuous learners (gradient-based): accumulate raw scores, threshold at 0.5
    95:     For regression:
    96:       - First learner is the initial constant predictor
    97:       - Subsequent learners predict residuals, scaled by alpha * learning_rate
    98: 
    99:     Args:
   100:         learners: list of fitted BaseLearner / MeanPredictor.
   101:         alphas: list of float learner weights.
   102:         learner_modes: list of str, "discrete" or "continuous" per learner.
   103:         X: np.ndarray [n_samples, n_features].
   104:         task_type: "classification" or "regression".
   105:         learning_rate: shrinkage for regression / gradient methods.
   106:     """
   107:     n_samples = X.shape[0]
   108:     raw_scores = np.zeros(n_samples)
   109: 
   110:     for i, (learner, alpha, mode) in enumerate(zip(learners, alphas, learner_modes)):
   111:         preds = learner.predict(X)
   112:         if task_type == "regression":
   113:             if i == 0:
   114:                 raw_scores += preds  # initial mean predictor
   115:             else:
   116:                 raw_scores += alpha * learning_rate * preds
   117:         elif mode == "discrete":
   118:             # AdaBoost-style: convert {0,1} -> {-1,+1}
   119:             raw_scores += alpha * (2 * preds - 1)
   120:         else:
   121:             # Gradient-based: accumulate continuous predictions
   122:             raw_scores += alpha * learning_rate * preds
   123: 
   124:     if task_type == "classification":
   125:         return (raw_scores >= 0).astype(int)
   126:     else:
   127:         return raw_scores
   128: 
   129: 
   130: def evaluate_ensemble(learners, alphas, learner_modes, X, y, task_type,
   131:                       learning_rate=0.1):
   132:     """Evaluate the ensemble on given data."""
   133:     preds = ensemble_predict(learners, alphas, learner_modes, X, task_type,
   134:                              learning_rate)
   135:     if task_type == "classification":
   136:         acc = accuracy_score(y, preds)
   137:         return {"accuracy": acc}
   138:     else:
   139:         rmse = np.sqrt(mean_squared_error(y, preds))
   140:         return {"rmse": rmse}
   141: 
   142: 
   143: # ============================================================================
   144: # EDITABLE — Boosting strategy (lines 147-256)
   145: # ============================================================================
   146: 
   147: class BoostingStrategy:
   148:     """Sample weighting and update strategy for gradient boosting.
   149: 
   150:     This class controls how sample weights are initialized, how pseudo-targets
   151:     (residuals or transformed targets) are computed for the next weak learner,
   152:     how learner weights (alphas) are determined, and how sample weights are
   153:     updated after each boosting round.
   154: 
   155:     The strategy is used by the fixed training loop (below) which:
   156:     1. Calls init_weights() once at the start
   157:     2. For each round t = 0..T-1:
   158:        a. Calls compute_targets() to get pseudo-targets for fitting the learner
   159:        b. Fits a base learner on (X, pseudo_targets, sample_weights)
   160:        c. Calls compute_learner_weight() to get alpha_t
   161:        d. Calls update_weights() to adjust sample weights
   162: 
   163:     Args (available via self.config set in __init__):
   164:         n_samples: int — number of training samples
   165:         n_features: int — number of input features
   166:         n_rounds: int — total boosting rounds
   167:         task_type: str — 'classification' or 'regression'
   168:         learning_rate: float — shrinkage factor (default 0.1)
   169:         dataset: str — dataset name
   170: 
   171:     For classification: y in {0, 1}, use signed labels y_signed = 2*y - 1
   172:     For regression: y is continuous, use residual-based approaches
   173:     """
   174: 
   175:     def __init__(self, config):
   176:         """Initialize the boosting strategy.
   177: 
   178:         Args:
   179:             config: dict with keys n_samples, n_features, n_rounds,
   180:                     task_type, learning_rate, dataset.
   181:         """
   182:         self.config = config
   183:         self.task_type = config["task_type"]
   184:         self.n_rounds = config["n_rounds"]
   185:         self.learning_rate = config["learning_rate"]
   186: 
   187:     def init_weights(self, n_samples):
   188:         """Initialize sample weights.
   189: 
   190:         Args:
   191:             n_samples: int — number of training samples.
   192: 
   193:         Returns:
   194:             np.ndarray of shape [n_samples] — initial sample weights (should sum to 1).
   195:         """
   196:         return np.ones(n_samples) / n_samples
   197: 
   198:     def compute_targets(self, y, current_predictions, sample_weights, round_idx):
   199:         """Compute pseudo-targets for the next weak learner to fit.
   200: 
   201:         This determines WHAT the weak learner tries to predict at each round.
   202: 
   203:         Args:
   204:             y: np.ndarray [n_samples] — true labels/targets.
   205:             current_predictions: np.ndarray [n_samples] — ensemble prediction so far
   206:                 (raw scores for classification, values for regression).
   207:             sample_weights: np.ndarray [n_samples] — current sample weights.
   208:             round_idx: int — current boosting round (0-indexed).
   209: 
   210:         Returns:
   211:             np.ndarray [n_samples] — pseudo-targets to fit the weak learner on.
   212:         """
   213:         # Default: fit on original labels (basic boosting)
   214:         return y
   215: 
   216:     def compute_learner_weight(self, learner, X, y, pseudo_targets,
   217:                                 sample_weights, round_idx):
   218:         """Compute the weight (alpha) for the newly fitted learner.
   219: 
   220:         Args:
   221:             learner: BaseLearner — the just-fitted weak learner.
   222:             X: np.ndarray [n_samples, n_features] — training features.
   223:             y: np.ndarray [n_samples] — true labels/targets.
   224:             pseudo_targets: np.ndarray [n_samples] — what the learner was fit on.
   225:             sample_weights: np.ndarray [n_samples] — current sample weights.
   226:             round_idx: int — current boosting round.
   227: 
   228:         Returns:
   229:             float — learner weight alpha_t. For classification, higher alpha
   230:                 means more influence in the vote. For regression, alpha scales
   231:                 the contribution (multiplied by learning_rate).
   232:         """
   233:         return 1.0
   234: 
   235:     def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
   236:                        alpha, round_idx):
   237:         """Update sample weights after fitting a learner.
   238: 
   239:         This determines how the distribution over training samples shifts
   240:         to focus on harder examples in subsequent rounds.
   241: 
   242:         Args:
   243:             sample_weights: np.ndarray [n_samples] — current sample weights.
   244:             learner: BaseLearner — the just-fitted weak learner.
   245:             X: np.ndarray [n_samples, n_features] — training features.
   246:             y: np.ndarray [n_samples] — true labels/targets.
   247:             pseudo_targets: np.ndarray [n_samples] — what the learner was fit on.
   248:             alpha: float — the learner's weight.
   249:             round_idx: int — current boosting round.
   250: 
   251:         Returns:
   252:             np.ndarray [n_samples] — updated sample weights (should sum to 1).
   253:         """
   254:         # Default: uniform weights (no reweighting)
   255:         return sample_weights
   256: 
   257: 
   258: # ============================================================================
   259: # FIXED — Training loop
   260: # ============================================================================
   261: 
   262: def train_boosting(X_train, y_train, X_test, y_test, strategy, config):
   263:     """Train a boosted ensemble using the given strategy.
   264: 
   265:     Args:
   266:         X_train, y_train: training data.
   267:         X_test, y_test: test data.
   268:         strategy: BoostingStrategy instance.
   269:         config: dict with n_rounds, task_type, learning_rate, max_depth, seed.
   270: 
   271:     Returns:
   272:         learners: list of fitted BaseLearner.
   273:         alphas: list of float learner weights.
   274:         metrics: dict of final test metrics.
   275:     """
   276:     n_rounds = config["n_rounds"]
   277:     task_type = config["task_type"]
   278:     lr = config["learning_rate"]
   279:     max_depth = config["max_depth"]
   280:     seed = config["seed"]
   281: 
   282:     learners = []
   283:     alphas = []
   284:     learner_modes = []  # "discrete" or "continuous" per learner
   285: 
   286:     # Initialize sample weights
   287:     n_samples = X_train.shape[0]
   288:     sample_weights = strategy.init_weights(n_samples)
   289: 
   290:     # For regression: track cumulative predictions for residual computation
   291:     # Use a simple mean predictor as the initial model
   292:     if task_type == "regression":
   293:         class MeanPredictor:
   294:             def __init__(self, mean_val):
   295:                 self._mean = mean_val
   296:             def predict(self, X):
   297:                 return np.full(X.shape[0], self._mean)
   298:         init_learner = MeanPredictor(y_train.mean())
   299:         learners.append(init_learner)
   300:         alphas.append(1.0)
   301:         learner_modes.append("continuous")
   302:         current_preds_train = init_learner.predict(X_train)
   303:     else:
   304:         current_preds_train = np.zeros(n_samples)
   305: 
   306:     for t in range(n_rounds):
   307:         # 1. Compute pseudo-targets
   308:         pseudo_targets = strategy.compute_targets(
   309:             y_train, current_preds_train, sample_weights, t,
   310:         )
   311: 
   312:         # 2. Fit weak learner
   313:         # Use regressor if pseudo-targets are continuous (e.g. gradient boosting
   314:         # fits residuals even for classification tasks).
   315:         is_continuous = not np.array_equal(pseudo_targets, pseudo_targets.astype(int))
   316:         learner_type = "regression" if is_continuous else task_type
   317:         learner = BaseLearner(learner_type, max_depth=max_depth,
   318:                               random_state=seed + t + 1)
   319:         learner.fit(X_train, pseudo_targets, sample_weight=sample_weights)
   320:         mode = "continuous" if is_continuous else "discrete"
   321: 
   322:         # 3. Compute learner weight
   323:         alpha = strategy.compute_learner_weight(
   324:             learner, X_train, y_train, pseudo_targets, sample_weights, t,
   325:         )
   326: 
   327:         # 4. Update sample weights
   328:         sample_weights = strategy.update_weights(
   329:             sample_weights, learner, X_train, y_train, pseudo_targets, alpha, t,
   330:         )
   331: 
   332:         # Ensure weights are valid
   333:         sample_weights = np.clip(sample_weights, 1e-10, None)
   334:         sample_weights = sample_weights / sample_weights.sum()
   335: 
   336:         # 5. Update cumulative predictions
   337:         preds_t = learner.predict(X_train)
   338:         if task_type == "classification" and mode == "discrete":
   339:             # AdaBoost-style: discrete predictions, signed vote
   340:             current_preds_train += alpha * (2 * preds_t - 1)
   341:         else:
   342:             # Gradient-based or regression: accumulate scaled predictions
   343:             current_preds_train += alpha * lr * preds_t
   344: 
   345:         learners.append(learner)
   346:         alphas.append(alpha)
   347:         learner_modes.append(mode)
   348: 
   349:         # Log progress
   350:         if (t + 1) % max(1, n_rounds // 10) == 0 or t == 0:
   351:             test_metrics = evaluate_ensemble(
   352:                 learners, alphas, learner_modes,
   353:                 X_test, y_test, task_type, lr,
   354:             )
   355:             train_metrics = evaluate_ensemble(
   356:                 learners, alphas, learner_modes,
   357:                 X_train, y_train, task_type, lr,
   358:             )
   359:             if task_type == "classification":
   360:                 print(
   361:                     f"TRAIN_METRICS: round={t+1}/{n_rounds} "
   362:                     f"train_acc={train_metrics['accuracy']:.4f} "
   363:                     f"test_acc={test_metrics['accuracy']:.4f}",
   364:                     flush=True,
   365:                 )
   366:             else:
   367:                 print(
   368:                     f"TRAIN_METRICS: round={t+1}/{n_rounds} "
   369:                     f"train_rmse={train_metrics['rmse']:.4f} "
   370:                     f"test_rmse={test_metrics['rmse']:.4f}",
   371:                     flush=True,
   372:                 )
   373: 
   374:     # Final evaluation
   375:     final_metrics = evaluate_ensemble(
   376:         learners, alphas, learner_modes, X_test, y_test, task_type, lr,
   377:     )
   378:     return learners, alphas, final_metrics
   379: 
   380: 
   381: # ============================================================================
   382: # FIXED — Main
   383: # ============================================================================
   384: 
   385: def main():
   386:     parser = argparse.ArgumentParser(description="ML Ensemble Boosting Benchmark")
   387:     parser.add_argument("--dataset", type=str, required=True,
   388:                         choices=["breast_cancer", "diabetes", "california_housing"])
   389:     parser.add_argument("--task", type=str, required=True,
   390:                         choices=["classification", "regression"])
   391:     parser.add_argument("--n-rounds", type=int, default=200,
   392:                         help="Number of boosting rounds")
   393:     parser.add_argument("--max-depth", type=int, default=3,
   394:                         help="Max depth of base decision trees")
   395:     parser.add_argument("--learning-rate", type=float, default=0.1,
   396:                         help="Shrinkage / learning rate")
   397:     parser.add_argument("--test-size", type=float, default=0.2,
   398:                         help="Fraction of data for testing")
   399:     parser.add_argument("--seed", type=int, default=42)
   400:     parser.add_argument("--output-dir", type=str, default=".")
   401:     args = parser.parse_args()
   402: 
   403:     np.random.seed(args.seed)
   404: 
   405:     # Load data
   406:     X, y, detected_task = load_dataset(args.dataset)
   407:     task_type = args.task
   408:     print(f"Dataset: {args.dataset} ({task_type})", flush=True)
   409:     print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}", flush=True)
   410: 
   411:     # Split
   412:     X_train, X_test, y_train, y_test = train_test_split(
   413:         X, y, test_size=args.test_size, random_state=args.seed,
   414:     )
   415: 
   416:     # Normalize
   417:     X_train, X_test = normalize_features(X_train, X_test)
   418: 
   419:     print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}", flush=True)
   420:     print(f"Boosting rounds: {args.n_rounds}, Max depth: {args.max_depth}, "
   421:           f"LR: {args.learning_rate}", flush=True)
   422: 
   423:     # Build strategy config
   424:     config = {
   425:         "n_samples": X_train.shape[0],
   426:         "n_features": X_train.shape[1],
   427:         "n_rounds": args.n_rounds,
   428:         "task_type": task_type,
   429:         "learning_rate": args.learning_rate,
   430:         "max_depth": args.max_depth,
   431:         "dataset": args.dataset,
   432:         "seed": args.seed,
   433:     }
   434: 
   435:     # Create strategy and train
   436:     strategy = BoostingStrategy(config)
   437:     learners, alphas, final_metrics = train_boosting(
   438:         X_train, y_train, X_test, y_test, strategy, config,
   439:     )
   440: 
   441:     # Report final metrics
   442:     if task_type == "classification":
   443:         print(f"TEST_METRICS: test_accuracy={final_metrics['accuracy']:.4f}", flush=True)
   444:     else:
   445:         print(f"TEST_METRICS: test_rmse={final_metrics['rmse']:.4f}", flush=True)
   446: 
   447: 
   448: if __name__ == "__main__":
   449:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `adaboost` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_boosting.py`:

```python
Lines 147–205:
   144: # EDITABLE — Boosting strategy (lines 147-256)
   145: # ============================================================================
   146: 
   147: class BoostingStrategy:
   148:     """AdaBoost: exponential loss reweighting (classification) / AdaBoost.R2 (regression)."""
   149: 
   150:     def __init__(self, config):
   151:         self.config = config
   152:         self.task_type = config["task_type"]
   153:         self.n_rounds = config["n_rounds"]
   154:         self.learning_rate = config["learning_rate"]
   155: 
   156:     def init_weights(self, n_samples):
   157:         return np.ones(n_samples) / n_samples
   158: 
   159:     def compute_targets(self, y, current_predictions, sample_weights, round_idx):
   160:         if self.task_type == "classification":
   161:             # AdaBoost fits on original labels (not residuals)
   162:             return y
   163:         else:
   164:             # Regression: fit on negative gradient (residuals) so that the
   165:             # fixed ensemble_predict accumulation (mean + sum alpha*lr*pred)
   166:             # works correctly.
   167:             return y - current_predictions
   168: 
   169:     def compute_learner_weight(self, learner, X, y, pseudo_targets,
   170:                                 sample_weights, round_idx):
   171:         if self.task_type == "classification":
   172:             preds = learner.predict(X)
   173:             incorrect = (preds != y).astype(float)
   174:             weighted_err = np.dot(sample_weights, incorrect) / sample_weights.sum()
   175:             weighted_err = np.clip(weighted_err, 1e-10, 1.0 - 1e-10)
   176:             alpha = self.learning_rate * 0.5 * np.log((1.0 - weighted_err) / weighted_err)
   177:             return alpha
   178:         else:
   179:             # Regression: use alpha=1.0; shrinkage is applied by the fixed
   180:             # ensemble_predict via learning_rate.  Sample reweighting in
   181:             # update_weights handles the AdaBoost.R2 emphasis on hard examples.
   182:             return 1.0
   183: 
   184:     def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
   185:                        alpha, round_idx):
   186:         preds = learner.predict(X)
   187:         if self.task_type == "classification":
   188:             incorrect = (preds != y).astype(float)
   189:             # w_i *= exp(alpha * I(wrong))
   190:             sample_weights = sample_weights * np.exp(alpha * incorrect)
   191:         else:
   192:             # AdaBoost.R2-style: reduce weight on well-predicted samples
   193:             # pseudo_targets are residuals; compare learner predictions to them
   194:             errors = np.abs(preds - pseudo_targets)
   195:             max_err = errors.max()
   196:             if max_err > 0:
   197:                 errors = errors / max_err  # normalize to [0, 1]
   198:             avg_loss = np.dot(sample_weights, errors)
   199:             avg_loss = np.clip(avg_loss, 1e-10, 1.0 - 1e-10)
   200:             beta = avg_loss / (1.0 - avg_loss)
   201:             # Decrease weight for well-predicted samples
   202:             sample_weights = sample_weights * np.power(beta, 1.0 - errors)
   203:         # Normalize
   204:         sample_weights = sample_weights / sample_weights.sum()
   205:         return sample_weights
   206: 
   207: # ============================================================================
   208: # FIXED — Training loop
```

### `gradient_boosting` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_boosting.py`:

```python
Lines 147–200:
   144: # EDITABLE — Boosting strategy (lines 147-256)
   145: # ============================================================================
   146: 
   147: class BoostingStrategy:
   148:     """Gradient Boosting: negative gradient (pseudo-residual) fitting."""
   149: 
   150:     def __init__(self, config):
   151:         self.config = config
   152:         self.task_type = config["task_type"]
   153:         self.n_rounds = config["n_rounds"]
   154:         self.learning_rate = config["learning_rate"]
   155:         # Track raw scores for logistic gradient computation
   156:         self._raw_scores = None
   157: 
   158:     def init_weights(self, n_samples):
   159:         # Gradient boosting uses uniform weights (no reweighting);
   160:         # the key insight is fitting to pseudo-residuals instead.
   161:         self._raw_scores = np.zeros(n_samples)
   162:         return np.ones(n_samples) / n_samples
   163: 
   164:     def _sigmoid(self, x):
   165:         return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
   166: 
   167:     def compute_targets(self, y, current_predictions, sample_weights, round_idx):
   168:         if self.task_type == "regression":
   169:             # Negative gradient of squared error = residuals
   170:             return y - current_predictions
   171:         else:
   172:             # Negative gradient of log-loss (logistic)
   173:             # For log-loss: -dL/dF = y - sigmoid(F)
   174:             probs = self._sigmoid(self._raw_scores)
   175:             return y - probs
   176: 
   177:     def compute_learner_weight(self, learner, X, y, pseudo_targets,
   178:                                 sample_weights, round_idx):
   179:         if self.task_type == "regression":
   180:             # Standard gradient boosting: alpha=1, shrinkage via learning_rate in ensemble
   181:             return 1.0
   182:         else:
   183:             # For classification: use line search on log-loss
   184:             preds = learner.predict(X)
   185:             # Approximate optimal step size via Newton step
   186:             probs = self._sigmoid(self._raw_scores)
   187:             numerator = np.sum(pseudo_targets * preds)
   188:             denominator = np.sum(probs * (1 - probs) * preds ** 2) + 1e-10
   189:             alpha = numerator / denominator
   190:             return max(alpha, 0.0)
   191: 
   192:     def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
   193:                        alpha, round_idx):
   194:         # Gradient boosting doesn't reweight samples; it fits to pseudo-residuals.
   195:         # But we update raw scores for classification gradient computation.
   196:         if self.task_type == "classification":
   197:             preds = learner.predict(X)
   198:             self._raw_scores += self.learning_rate * alpha * preds
   199:         # Weights stay uniform
   200:         return sample_weights
   201: 
   202: # ============================================================================
   203: # FIXED — Training loop
```

### `xgboost_style` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_boosting.py`:

```python
Lines 147–202:
   144: # EDITABLE — Boosting strategy (lines 147-256)
   145: # ============================================================================
   146: 
   147: class BoostingStrategy:
   148:     """XGBoost-style: second-order Newton boosting with regularization."""
   149: 
   150:     def __init__(self, config):
   151:         self.config = config
   152:         self.task_type = config["task_type"]
   153:         self.n_rounds = config["n_rounds"]
   154:         self.learning_rate = config["learning_rate"]
   155:         # L2 regularization on leaf weights (lambda in XGBoost)
   156:         self.reg_lambda = 1.0
   157:         # Track raw scores for gradient/Hessian computation
   158:         self._raw_scores = None
   159: 
   160:     def init_weights(self, n_samples):
   161:         self._raw_scores = np.zeros(n_samples)
   162:         return np.ones(n_samples) / n_samples
   163: 
   164:     def _sigmoid(self, x):
   165:         return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
   166: 
   167:     def compute_targets(self, y, current_predictions, sample_weights, round_idx):
   168:         if self.task_type == "regression":
   169:             # Negative gradient of squared error = residuals
   170:             return y - current_predictions
   171:         else:
   172:             # Negative gradient of log-loss
   173:             probs = self._sigmoid(self._raw_scores)
   174:             return y - probs
   175: 
   176:     def compute_learner_weight(self, learner, X, y, pseudo_targets,
   177:                                 sample_weights, round_idx):
   178:         preds = learner.predict(X)
   179:         if self.task_type == "regression":
   180:             # Newton step: sum(gradient * pred) / (sum(hessian * pred^2) + lambda)
   181:             # For squared error: gradient = residual, hessian = 1
   182:             numerator = np.sum(pseudo_targets * preds)
   183:             denominator = np.sum(preds ** 2) + self.reg_lambda
   184:             alpha = numerator / denominator
   185:             return max(alpha, 0.0)
   186:         else:
   187:             # For log-loss: hessian = p*(1-p)
   188:             probs = self._sigmoid(self._raw_scores)
   189:             hessians = probs * (1.0 - probs)
   190:             numerator = np.sum(pseudo_targets * preds)
   191:             denominator = np.sum(hessians * preds ** 2) + self.reg_lambda
   192:             alpha = numerator / denominator
   193:             return max(alpha, 0.0)
   194: 
   195:     def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
   196:                        alpha, round_idx):
   197:         # XGBoost uses second-order info, not sample reweighting.
   198:         # Update raw scores for next round's gradient computation.
   199:         preds = learner.predict(X)
   200:         self._raw_scores += self.learning_rate * alpha * preds
   201:         # Weights stay uniform — boosting signal is in the pseudo-residuals
   202:         return sample_weights
   203: 
   204: # ============================================================================
   205: # FIXED — Training loop
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
