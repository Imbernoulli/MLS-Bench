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
The training and evaluation pipeline (number of boosting rounds, the shallow decision-tree weak learner, learning-rate shrinkage, datasets, and metrics) is fixed by the harness and not editable. Your strategy is evaluated on both classification and regression tabular tasks. Dataset metadata (including `learning_rate`) is provided to your class via the `config` dict.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `scikit-learn/custom_boosting.py`
- editable lines **159–266**




## Readable Context


### `scikit-learn/custom_boosting.py`  [EDITABLE — lines 159–266 only]

```python
     1: """ML Ensemble Boosting Benchmark.
     2: 
     3: Train gradient-boosted ensembles of shallow decision trees on standardized
     4: tabular data to evaluate novel sample weighting / boosting update strategies.
     5: 
     6: EDITABLE: BoostingStrategy class -- the agent's boosting strategy.
     7: FIXED: input loading + base learner + ensemble accumulation + prediction emit.
     8:        The dataset identity, the train/test split, the test labels, and the
     9:        metric live in a host-only module the agent's process cannot import.
    10:        The pre-generated standardized (X_train, y_train, X_test) triple is
    11:        loaded -- and ENV/SEED scrubbed -- in the FIXED header BELOW, *before*
    12:        the editable strategy class is defined, so editable code that runs at
    13:        import time cannot read the dataset identity, reconstruct the public
    14:        loader + split, and recover the held-out test labels. The host-side
    15:        parser regenerates the labels and scores the same metric. Inputs are
    16:        pre-standardized, exactly as before; the split is identical.
    17: """
    18: 
    19: import io
    20: import os
    21: import base64
    22: import warnings
    23: from abc import ABC, abstractmethod
    24: 
    25: import numpy as np
    26: from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    27: 
    28: warnings.filterwarnings("ignore")
    29: 
    30: 
    31: # ============================================================================
    32: # FIXED -- Input loading + run bootstrap (do not modify)
    33: # ============================================================================
    34: # The dataset generator (incl. identity), the train/test split, the test
    35: # labels, and the metric live in a host-only module the agent's process cannot
    36: # import. We load the pre-generated standardized (X_train, y_train, X_test)
    37: # triple HERE -- before the editable strategy class below is defined -- and
    38: # immediately scrub ENV/SEED from the environment, so editable class-body code
    39: # (which executes at import time) cannot read the dataset identity, reconstruct
    40: # the public loader + split, and recover the held-out test labels.
    41: 
    42: def _inputs_dir():
    43:     d = os.environ.get("BOOST_INPUTS_DIR")
    44:     if d:
    45:         return d
    46:     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_boost_inputs")
    47: 
    48: 
    49: def _load_input(env_name, seed):
    50:     path = os.path.join(_inputs_dir(), f"{env_name}_seed{seed}.npz.b64")
    51:     with open(path, "r") as f:
    52:         raw = base64.b64decode(f.read())
    53:     d = np.load(io.BytesIO(raw))
    54:     return (d["X_train"], d["y_train"], d["X_test"],
    55:             str(d["task_type"]), int(d["n_rounds"]),
    56:             int(d["max_depth"]), float(d["learning_rate"]))
    57: 
    58: 
    59: def _bootstrap():
    60:     """Load this run's inputs, then scrub the dataset identity + seed.
    61: 
    62:     Runs once at import, BEFORE the editable ``BoostingStrategy`` class below is
    63:     defined, so editable class-body code cannot cache ENV/SEED and replay the
    64:     public split. The dataset identity (``env``) is used only inside this
    65:     function and is never stored as a module global; only label-free arrays and
    66:     pipeline hyperparameters survive.
    67:     """
    68:     env = os.environ.get("ENV", "")
    69:     if not env:
    70:         raise SystemExit("ENV not set")
    71:     seed = int(os.environ.get("SEED", "42"))
    72:     print(f"=== Boosting benchmark (seed={seed}) ===", flush=True)
    73:     bundle = _load_input(env, seed)
    74:     for _k in ("ENV", "SEED", "BOOST_INPUTS_DIR"):
    75:         os.environ.pop(_k, None)
    76:     return seed, bundle
    77: 
    78: 
    79: _SEED, _INPUT_BUNDLE = _bootstrap()
    80: (_X_TRAIN, _Y_TRAIN, _X_TEST, _TASK_TYPE, _N_ROUNDS, _MAX_DEPTH, _LR) = _INPUT_BUNDLE
    81: 
    82: 
    83: # ============================================================================
    84: # FIXED -- Base learner interface (do not modify)
    85: # ============================================================================
    86: 
    87: class BaseLearner:
    88:     """Wrapper around sklearn decision tree as weak learner."""
    89: 
    90:     def __init__(self, task_type, max_depth=1, random_state=None):
    91:         self.task_type = task_type
    92:         if task_type == "classification":
    93:             self.tree = DecisionTreeClassifier(
    94:                 max_depth=max_depth, random_state=random_state,
    95:             )
    96:         else:
    97:             self.tree = DecisionTreeRegressor(
    98:                 max_depth=max_depth, random_state=random_state,
    99:             )
   100: 
   101:     def fit(self, X, y, sample_weight=None):
   102:         self.tree.fit(X, y, sample_weight=sample_weight)
   103:         return self
   104: 
   105:     def predict(self, X):
   106:         return self.tree.predict(X)
   107: 
   108: 
   109: # ============================================================================
   110: # FIXED -- Ensemble prediction (do not modify)
   111: # ============================================================================
   112: 
   113: def ensemble_predict(learners, alphas, learner_modes, X, task_type,
   114:                      learning_rate=0.1):
   115:     """Predict using the ensemble.
   116: 
   117:     For classification:
   118:       - Discrete learners (AdaBoost-style): weighted majority vote with {-1,+1} coding
   119:       - Continuous learners (gradient-based): accumulate raw scores, threshold at 0.5
   120:     For regression:
   121:       - First learner is the initial constant predictor
   122:       - Subsequent learners predict residuals, scaled by alpha * learning_rate
   123: 
   124:     Args:
   125:         learners: list of fitted BaseLearner / MeanPredictor.
   126:         alphas: list of float learner weights.
   127:         learner_modes: list of str, "discrete" or "continuous" per learner.
   128:         X: np.ndarray [n_samples, n_features].
   129:         task_type: "classification" or "regression".
   130:         learning_rate: shrinkage for regression / gradient methods.
   131:     """
   132:     n_samples = X.shape[0]
   133:     raw_scores = np.zeros(n_samples)
   134: 
   135:     for i, (learner, alpha, mode) in enumerate(zip(learners, alphas, learner_modes)):
   136:         preds = learner.predict(X)
   137:         if task_type == "regression":
   138:             if i == 0:
   139:                 raw_scores += preds  # initial mean predictor
   140:             else:
   141:                 raw_scores += alpha * learning_rate * preds
   142:         elif mode == "discrete":
   143:             # AdaBoost-style: convert {0,1} -> {-1,+1}
   144:             raw_scores += alpha * (2 * preds - 1)
   145:         else:
   146:             # Gradient-based: accumulate continuous predictions
   147:             raw_scores += alpha * learning_rate * preds
   148: 
   149:     if task_type == "classification":
   150:         return (raw_scores >= 0).astype(int)
   151:     else:
   152:         return raw_scores
   153: 
   154: 
   155: # ============================================================================
   156: # EDITABLE -- Boosting strategy (lines 159 to 266)
   157: # ============================================================================
   158: 
   159: class BoostingStrategy:
   160:     """Sample weighting and update strategy for gradient boosting.
   161: 
   162:     This class controls how sample weights are initialized, how pseudo-targets
   163:     (residuals or transformed targets) are computed for the next weak learner,
   164:     how learner weights (alphas) are determined, and how sample weights are
   165:     updated after each boosting round.
   166: 
   167:     The strategy is used by the fixed training loop (below) which:
   168:     1. Calls init_weights() once at the start
   169:     2. For each round t = 0..T-1:
   170:        a. Calls compute_targets() to get pseudo-targets for fitting the learner
   171:        b. Fits a base learner on (X, pseudo_targets, sample_weights)
   172:        c. Calls compute_learner_weight() to get alpha_t
   173:        d. Calls update_weights() to adjust sample weights
   174: 
   175:     Args (available via self.config set in __init__):
   176:         n_samples: int -- number of training samples
   177:         n_features: int -- number of input features
   178:         n_rounds: int -- total boosting rounds
   179:         task_type: str -- 'classification' or 'regression'
   180:         learning_rate: float -- shrinkage factor (default 0.1)
   181: 
   182:     For classification: y in {0, 1}, use signed labels y_signed = 2*y - 1
   183:     For regression: y is continuous, use residual-based approaches
   184:     """
   185: 
   186:     def __init__(self, config):
   187:         """Initialize the boosting strategy.
   188: 
   189:         Args:
   190:             config: dict with keys n_samples, n_features, n_rounds,
   191:                     task_type, learning_rate.
   192:         """
   193:         self.config = config
   194:         self.task_type = config["task_type"]
   195:         self.n_rounds = config["n_rounds"]
   196:         self.learning_rate = config["learning_rate"]
   197: 
   198:     def init_weights(self, n_samples):
   199:         """Initialize sample weights.
   200: 
   201:         Args:
   202:             n_samples: int -- number of training samples.
   203: 
   204:         Returns:
   205:             np.ndarray of shape [n_samples] -- initial sample weights (should sum to 1).
   206:         """
   207:         return np.ones(n_samples) / n_samples
   208: 
   209:     def compute_targets(self, y, current_predictions, sample_weights, round_idx):
   210:         """Compute pseudo-targets for the next weak learner to fit.
   211: 
   212:         This determines WHAT the weak learner tries to predict at each round.
   213: 
   214:         Args:
   215:             y: np.ndarray [n_samples] -- true labels/targets.
   216:             current_predictions: np.ndarray [n_samples] -- ensemble prediction so far
   217:                 (raw scores for classification, values for regression).
   218:             sample_weights: np.ndarray [n_samples] -- current sample weights.
   219:             round_idx: int -- current boosting round (0-indexed).
   220: 
   221:         Returns:
   222:             np.ndarray [n_samples] -- pseudo-targets to fit the weak learner on.
   223:         """
   224:         # Default: fit on original labels (basic boosting)
   225:         return y
   226: 
   227:     def compute_learner_weight(self, learner, X, y, pseudo_targets,
   228:                                 sample_weights, round_idx):
   229:         """Compute the weight (alpha) for the newly fitted learner.
   230: 
   231:         Args:
   232:             learner: BaseLearner -- the just-fitted weak learner.
   233:             X: np.ndarray [n_samples, n_features] -- training features.
   234:             y: np.ndarray [n_samples] -- true labels/targets.
   235:             pseudo_targets: np.ndarray [n_samples] -- what the learner was fit on.
   236:             sample_weights: np.ndarray [n_samples] -- current sample weights.
   237:             round_idx: int -- current boosting round.
   238: 
   239:         Returns:
   240:             float -- learner weight alpha_t. For classification, higher alpha
   241:                 means more influence in the vote. For regression, alpha scales
   242:                 the contribution (multiplied by learning_rate).
   243:         """
   244:         return 1.0
   245: 
   246:     def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
   247:                        alpha, round_idx):
   248:         """Update sample weights after fitting a learner.
   249: 
   250:         This determines how the distribution over training samples shifts
   251:         to focus on harder examples in subsequent rounds.
   252: 
   253:         Args:
   254:             sample_weights: np.ndarray [n_samples] -- current sample weights.
   255:             learner: BaseLearner -- the just-fitted weak learner.
   256:             X: np.ndarray [n_samples, n_features] -- training features.
   257:             y: np.ndarray [n_samples] -- true labels/targets.
   258:             pseudo_targets: np.ndarray [n_samples] -- what the learner was fit on.
   259:             alpha: float -- the learner's weight.
   260:             round_idx: int -- current boosting round.
   261: 
   262:         Returns:
   263:             np.ndarray [n_samples] -- updated sample weights (should sum to 1).
   264:         """
   265:         # Default: uniform weights (no reweighting)
   266:         return sample_weights
   267: 
   268: 
   269: # ============================================================================
   270: # FIXED -- Training loop + prediction emit (do not modify below)
   271: # ============================================================================
   272: # The dataset generator (incl. identity), the train/test split, the test labels,
   273: # and the metric live in a host-only module the agent's process cannot import.
   274: # The pre-generated standardized (X_train, y_train, X_test) triple was loaded in
   275: # the header above (with ENV/SEED scrubbed before this editable class was even
   276: # defined). This program builds the boosting ensemble using the agent's strategy
   277: # on the training split, predicts on the held-out test split, and emits those
   278: # predictions. The host-side parser regenerates the truth and scores it.
   279: 
   280: def train_boosting(X_train, y_train, strategy, config):
   281:     """Train a boosted ensemble using the given strategy on the training split.
   282: 
   283:     Args:
   284:         X_train, y_train: training data.
   285:         strategy: BoostingStrategy instance.
   286:         config: dict with n_rounds, task_type, learning_rate, max_depth, seed.
   287: 
   288:     Returns:
   289:         learners: list of fitted BaseLearner.
   290:         alphas: list of float learner weights.
   291:         learner_modes: list of "discrete"/"continuous" per learner.
   292:     """
   293:     n_rounds = config["n_rounds"]
   294:     task_type = config["task_type"]
   295:     lr = config["learning_rate"]
   296:     max_depth = config["max_depth"]
   297:     seed = config["seed"]
   298: 
   299:     learners = []
   300:     alphas = []
   301:     learner_modes = []  # "discrete" or "continuous" per learner
   302: 
   303:     # Initialize sample weights
   304:     n_samples = X_train.shape[0]
   305:     sample_weights = strategy.init_weights(n_samples)
   306: 
   307:     # For regression: track cumulative predictions for residual computation
   308:     # Use a simple mean predictor as the initial model
   309:     if task_type == "regression":
   310:         class MeanPredictor:
   311:             def __init__(self, mean_val):
   312:                 self._mean = mean_val
   313:             def predict(self, X):
   314:                 return np.full(X.shape[0], self._mean)
   315:         init_learner = MeanPredictor(y_train.mean())
   316:         learners.append(init_learner)
   317:         alphas.append(1.0)
   318:         learner_modes.append("continuous")
   319:         current_preds_train = init_learner.predict(X_train)
   320:     else:
   321:         current_preds_train = np.zeros(n_samples)
   322: 
   323:     for t in range(n_rounds):
   324:         # 1. Compute pseudo-targets
   325:         pseudo_targets = strategy.compute_targets(
   326:             y_train, current_preds_train, sample_weights, t,
   327:         )
   328: 
   329:         # 2. Fit weak learner
   330:         # Use regressor if pseudo-targets are continuous (e.g. gradient boosting
   331:         # fits residuals even for classification tasks).
   332:         is_continuous = not np.array_equal(pseudo_targets, pseudo_targets.astype(int))
   333:         learner_type = "regression" if is_continuous else task_type
   334:         learner = BaseLearner(learner_type, max_depth=max_depth,
   335:                               random_state=seed + t + 1)
   336:         learner.fit(X_train, pseudo_targets, sample_weight=sample_weights)
   337:         mode = "continuous" if is_continuous else "discrete"
   338: 
   339:         # 3. Compute learner weight
   340:         alpha = strategy.compute_learner_weight(
   341:             learner, X_train, y_train, pseudo_targets, sample_weights, t,
   342:         )
   343: 
   344:         # 4. Update sample weights
   345:         sample_weights = strategy.update_weights(
   346:             sample_weights, learner, X_train, y_train, pseudo_targets, alpha, t,
   347:         )
   348: 
   349:         # Ensure weights are valid
   350:         sample_weights = np.clip(sample_weights, 1e-10, None)
   351:         sample_weights = sample_weights / sample_weights.sum()
   352: 
   353:         # 5. Update cumulative predictions
   354:         preds_t = learner.predict(X_train)
   355:         if task_type == "classification" and mode == "discrete":
   356:             # AdaBoost-style: discrete predictions, signed vote
   357:             current_preds_train += alpha * (2 * preds_t - 1)
   358:         else:
   359:             # Gradient-based or regression: accumulate scaled predictions
   360:             current_preds_train += alpha * lr * preds_t
   361: 
   362:         learners.append(learner)
   363:         alphas.append(alpha)
   364:         learner_modes.append(mode)
   365: 
   366:         # Log training progress (train-split only; test split is held out)
   367:         if (t + 1) % max(1, n_rounds // 10) == 0 or t == 0:
   368:             train_preds = ensemble_predict(
   369:                 learners, alphas, learner_modes, X_train, task_type, lr,
   370:             )
   371:             if task_type == "classification":
   372:                 train_acc = float(np.mean(train_preds == y_train))
   373:                 print(
   374:                     f"TRAIN_METRICS: round={t+1}/{n_rounds} "
   375:                     f"train_acc={train_acc:.4f}",
   376:                     flush=True,
   377:                 )
   378:             else:
   379:                 train_rmse = float(np.sqrt(np.mean((train_preds - y_train) ** 2)))
   380:                 print(
   381:                     f"TRAIN_METRICS: round={t+1}/{n_rounds} "
   382:                     f"train_rmse={train_rmse:.4f}",
   383:                     flush=True,
   384:                 )
   385: 
   386:     return learners, alphas, learner_modes
   387: 
   388: 
   389: def main():
   390:     # Inputs were loaded -- and ENV/SEED scrubbed -- in the header bootstrap,
   391:     # before the editable strategy class above was defined.
   392:     np.random.seed(_SEED)
   393:     X_train, y_train, X_test = _X_TRAIN, _Y_TRAIN, _X_TEST
   394:     task_type, n_rounds, max_depth, lr = _TASK_TYPE, _N_ROUNDS, _MAX_DEPTH, _LR
   395:     print(f"Input: train={X_train.shape}, test={X_test.shape}, "
   396:           f"task={task_type}", flush=True)
   397:     print(f"Boosting rounds: {n_rounds}, Max depth: {max_depth}, "
   398:           f"LR: {lr}", flush=True)
   399: 
   400:     # config for the strategy: NO dataset identity, NO seed (cannot replay split)
   401:     config = {
   402:         "n_samples": X_train.shape[0],
   403:         "n_features": X_train.shape[1],
   404:         "n_rounds": n_rounds,
   405:         "task_type": task_type,
   406:         "learning_rate": lr,
   407:     }
   408:     # runner config carries seed/max_depth for reproducible tree fitting
   409:     run_config = dict(config)
   410:     run_config["max_depth"] = max_depth
   411:     run_config["seed"] = _SEED
   412: 
   413:     strategy = BoostingStrategy(config)
   414:     learners, alphas, learner_modes = train_boosting(
   415:         X_train, y_train, strategy, run_config,
   416:     )
   417: 
   418:     # Predict on the held-out test split and emit predictions for host scoring.
   419:     # The dataset identity is intentionally NOT echoed (the host-side parser
   420:     # already knows which environment it scores, by command label).
   421:     test_preds = ensemble_predict(
   422:         learners, alphas, learner_modes, X_test, task_type, lr,
   423:     )
   424:     test_preds = np.asarray(test_preds, dtype=np.float64).ravel()
   425:     payload = base64.b64encode(
   426:         np.ascontiguousarray(test_preds, dtype=np.float64).tobytes()
   427:     ).decode("ascii")
   428:     print(
   429:         f"BOOST_PRED seed={_SEED} n={test_preds.shape[0]} preds={payload}",
   430:         flush=True,
   431:     )
   432:     print("Done.", flush=True)
   433: 
   434: 
   435: if __name__ == "__main__":
   436:     main()
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
Lines 159–217:
   156: # EDITABLE -- Boosting strategy (lines 159 to 266)
   157: # ============================================================================
   158: 
   159: class BoostingStrategy:
   160:     """AdaBoost: exponential loss reweighting (classification) / AdaBoost.R2 (regression)."""
   161: 
   162:     def __init__(self, config):
   163:         self.config = config
   164:         self.task_type = config["task_type"]
   165:         self.n_rounds = config["n_rounds"]
   166:         self.learning_rate = config["learning_rate"]
   167: 
   168:     def init_weights(self, n_samples):
   169:         return np.ones(n_samples) / n_samples
   170: 
   171:     def compute_targets(self, y, current_predictions, sample_weights, round_idx):
   172:         if self.task_type == "classification":
   173:             # AdaBoost fits on original labels (not residuals)
   174:             return y
   175:         else:
   176:             # Regression: fit on negative gradient (residuals) so that the
   177:             # fixed ensemble_predict accumulation (mean + sum alpha*lr*pred)
   178:             # works correctly.
   179:             return y - current_predictions
   180: 
   181:     def compute_learner_weight(self, learner, X, y, pseudo_targets,
   182:                                 sample_weights, round_idx):
   183:         if self.task_type == "classification":
   184:             preds = learner.predict(X)
   185:             incorrect = (preds != y).astype(float)
   186:             weighted_err = np.dot(sample_weights, incorrect) / sample_weights.sum()
   187:             weighted_err = np.clip(weighted_err, 1e-10, 1.0 - 1e-10)
   188:             alpha = self.learning_rate * 0.5 * np.log((1.0 - weighted_err) / weighted_err)
   189:             return alpha
   190:         else:
   191:             # Regression: use alpha=1.0; shrinkage is applied by the fixed
   192:             # ensemble_predict via learning_rate.  Sample reweighting in
   193:             # update_weights handles the AdaBoost.R2 emphasis on hard examples.
   194:             return 1.0
   195: 
   196:     def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
   197:                        alpha, round_idx):
   198:         preds = learner.predict(X)
   199:         if self.task_type == "classification":
   200:             incorrect = (preds != y).astype(float)
   201:             # w_i *= exp(alpha * I(wrong))
   202:             sample_weights = sample_weights * np.exp(alpha * incorrect)
   203:         else:
   204:             # AdaBoost.R2-style: reduce weight on well-predicted samples
   205:             # pseudo_targets are residuals; compare learner predictions to them
   206:             errors = np.abs(preds - pseudo_targets)
   207:             max_err = errors.max()
   208:             if max_err > 0:
   209:                 errors = errors / max_err  # normalize to [0, 1]
   210:             avg_loss = np.dot(sample_weights, errors)
   211:             avg_loss = np.clip(avg_loss, 1e-10, 1.0 - 1e-10)
   212:             beta = avg_loss / (1.0 - avg_loss)
   213:             # Decrease weight for well-predicted samples
   214:             sample_weights = sample_weights * np.power(beta, 1.0 - errors)
   215:         # Normalize
   216:         sample_weights = sample_weights / sample_weights.sum()
   217:         return sample_weights
   218: 
   219: 
   220: # ============================================================================
```

### `gradient_boosting` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_boosting.py`:

```python
Lines 159–212:
   156: # EDITABLE -- Boosting strategy (lines 159 to 266)
   157: # ============================================================================
   158: 
   159: class BoostingStrategy:
   160:     """Gradient Boosting: negative gradient (pseudo-residual) fitting."""
   161: 
   162:     def __init__(self, config):
   163:         self.config = config
   164:         self.task_type = config["task_type"]
   165:         self.n_rounds = config["n_rounds"]
   166:         self.learning_rate = config["learning_rate"]
   167:         # Track raw scores for logistic gradient computation
   168:         self._raw_scores = None
   169: 
   170:     def init_weights(self, n_samples):
   171:         # Gradient boosting uses uniform weights (no reweighting);
   172:         # the key insight is fitting to pseudo-residuals instead.
   173:         self._raw_scores = np.zeros(n_samples)
   174:         return np.ones(n_samples) / n_samples
   175: 
   176:     def _sigmoid(self, x):
   177:         return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
   178: 
   179:     def compute_targets(self, y, current_predictions, sample_weights, round_idx):
   180:         if self.task_type == "regression":
   181:             # Negative gradient of squared error = residuals
   182:             return y - current_predictions
   183:         else:
   184:             # Negative gradient of log-loss (logistic)
   185:             # For log-loss: -dL/dF = y - sigmoid(F)
   186:             probs = self._sigmoid(self._raw_scores)
   187:             return y - probs
   188: 
   189:     def compute_learner_weight(self, learner, X, y, pseudo_targets,
   190:                                 sample_weights, round_idx):
   191:         if self.task_type == "regression":
   192:             # Standard gradient boosting: alpha=1, shrinkage via learning_rate in ensemble
   193:             return 1.0
   194:         else:
   195:             # For classification: use line search on log-loss
   196:             preds = learner.predict(X)
   197:             # Approximate optimal step size via Newton step
   198:             probs = self._sigmoid(self._raw_scores)
   199:             numerator = np.sum(pseudo_targets * preds)
   200:             denominator = np.sum(probs * (1 - probs) * preds ** 2) + 1e-10
   201:             alpha = numerator / denominator
   202:             return max(alpha, 0.0)
   203: 
   204:     def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
   205:                        alpha, round_idx):
   206:         # Gradient boosting doesn't reweight samples; it fits to pseudo-residuals.
   207:         # But we update raw scores for classification gradient computation.
   208:         if self.task_type == "classification":
   209:             preds = learner.predict(X)
   210:             self._raw_scores += self.learning_rate * alpha * preds
   211:         # Weights stay uniform
   212:         return sample_weights
   213: 
   214: 
   215: # ============================================================================
```

### `xgboost_style` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_boosting.py`:

```python
Lines 159–214:
   156: # EDITABLE -- Boosting strategy (lines 159 to 266)
   157: # ============================================================================
   158: 
   159: class BoostingStrategy:
   160:     """XGBoost-style: second-order Newton boosting with regularization."""
   161: 
   162:     def __init__(self, config):
   163:         self.config = config
   164:         self.task_type = config["task_type"]
   165:         self.n_rounds = config["n_rounds"]
   166:         self.learning_rate = config["learning_rate"]
   167:         # L2 regularization on leaf weights (lambda in XGBoost)
   168:         self.reg_lambda = 1.0
   169:         # Track raw scores for gradient/Hessian computation
   170:         self._raw_scores = None
   171: 
   172:     def init_weights(self, n_samples):
   173:         self._raw_scores = np.zeros(n_samples)
   174:         return np.ones(n_samples) / n_samples
   175: 
   176:     def _sigmoid(self, x):
   177:         return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
   178: 
   179:     def compute_targets(self, y, current_predictions, sample_weights, round_idx):
   180:         if self.task_type == "regression":
   181:             # Negative gradient of squared error = residuals
   182:             return y - current_predictions
   183:         else:
   184:             # Negative gradient of log-loss
   185:             probs = self._sigmoid(self._raw_scores)
   186:             return y - probs
   187: 
   188:     def compute_learner_weight(self, learner, X, y, pseudo_targets,
   189:                                 sample_weights, round_idx):
   190:         preds = learner.predict(X)
   191:         if self.task_type == "regression":
   192:             # Newton step: sum(gradient * pred) / (sum(hessian * pred^2) + lambda)
   193:             # For squared error: gradient = residual, hessian = 1
   194:             numerator = np.sum(pseudo_targets * preds)
   195:             denominator = np.sum(preds ** 2) + self.reg_lambda
   196:             alpha = numerator / denominator
   197:             return max(alpha, 0.0)
   198:         else:
   199:             # For log-loss: hessian = p*(1-p)
   200:             probs = self._sigmoid(self._raw_scores)
   201:             hessians = probs * (1.0 - probs)
   202:             numerator = np.sum(pseudo_targets * preds)
   203:             denominator = np.sum(hessians * preds ** 2) + self.reg_lambda
   204:             alpha = numerator / denominator
   205:             return max(alpha, 0.0)
   206: 
   207:     def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
   208:                        alpha, round_idx):
   209:         # XGBoost uses second-order info, not sample reweighting.
   210:         # Update raw scores for next round's gradient computation.
   211:         preds = learner.predict(X)
   212:         self._raw_scores += self.learning_rate * alpha * preds
   213:         # Weights stay uniform — boosting signal is in the pseudo-residuals
   214:         return sample_weights
   215: 
   216: 
   217: # ============================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
