# MLS-Bench: optimization-hyperparameter-search

# Hyperparameter Optimization: Custom Search Strategy Design

## Research Question
Design a novel hyperparameter optimization (HPO) strategy that achieves better final validation scores and faster convergence than standard approaches like Random Search, TPE, Hyperband, and their combinations (BOHB, DEHB).

## Background
Hyperparameter optimization is a fundamental problem in machine learning: given a model and dataset, find the hyperparameter configuration that maximizes validation performance within a limited evaluation budget. This is a black-box optimization problem where each function evaluation (training + validation) is expensive.

Classic strategies include:
- **Random Search** — samples configurations uniformly. Simple but surprisingly effective when some hyperparameters dominate (Bergstra and Bengio, "Random Search for Hyper-Parameter Optimization", JMLR 2012).
- **TPE (Tree-structured Parzen Estimator)** — models `p(x | y < y*)` and `p(x | y >= y*)` with kernel density estimation and maximizes their ratio (Bergstra, Bardenet, Bengio, and Kégl, "Algorithms for Hyper-Parameter Optimization", NIPS 2011).
- **Hyperband** — multi-fidelity evaluation via successive halving to allocate resources to promising configurations (Li, Jamieson, DeSalvo, Rostamizadeh, and Talwalkar, JMLR 2017; arXiv:1603.06560).

State-of-the-art combinations:
- **BOHB** — replaces random sampling in Hyperband with TPE-guided suggestions (Falkner, Klein, and Hutter, ICML 2018; arXiv:1807.01774).
- **DEHB** — uses Differential Evolution within Hyperband's multi-fidelity framework (Awad, Mallik, and Hutter, IJCAI 2021; arXiv:2105.09821).
- **CMA-ES** — adapts a full covariance matrix of a Gaussian distribution for efficient continuous optimization (Hansen and Ostermeier, "Completely Derandomized Self-Adaptation in Evolution Strategies", *Evolutionary Computation* 9(2), 2001).

## Task
Implement a custom HPO strategy by modifying the `CustomHPOStrategy` class in `scikit-learn/custom_hpo.py`. You should implement both `__init__` and `suggest` methods. The class is called repeatedly in a sequential loop where each call proposes one configuration to evaluate.

## Interface
```python
class CustomHPOStrategy:
    def __init__(self, seed: int = 42):
        """Initialize the strategy with a random seed."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Propose the next configuration to evaluate.

        Args:
            space: SearchSpace with .params (list of HParam), .dim,
                   .sample_uniform(rng), .clip(config)
            history: list of Trial(config, score, budget) from past evals
            budget_left: remaining budget in full-fidelity units

        Returns:
            config: dict mapping hyperparameter names to values
            fidelity: float in (0, 1] for multi-fidelity evaluation
        """
```

The search space provides:
- `space.params` — list of `HParam` objects with name, type (`"float"`/`"int"`/`"categorical"`), low, high, log_scale, choices.
- `space.sample_uniform(rng)` — sample a random valid configuration.
- `space.clip(config)` — clip values to valid ranges.

Each `Trial` records:
- `trial.config` — the hyperparameter configuration dict.
- `trial.score` — observed validation score (higher is better).
- `trial.budget` — fidelity fraction used (`1.0` = full evaluation).

The fidelity parameter controls evaluation cost: lower fidelity means cheaper but noisier evaluation (e.g., fewer boosting rounds, fewer CV folds, fewer MLP epochs).

## Evaluation
Evaluated on three ML model tuning benchmarks (**higher `best_val_score` is better, higher `convergence_auc` is better**):

- **XGBoost** (6D: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `min_samples_split`, `min_samples_leaf`; `GradientBoostingRegressor` on California Housing; budget = 50).
- **SVM** (3D: `C`, `gamma`, `kernel`; `SVC` on Breast Cancer; budget = 40).
- **Neural Net** (6D: hidden layers, learning rate, alpha, batch_size, activation; MLP on Diabetes; budget = 40).

Metrics:
- **best_val_score**: best validation score found within the budget (primary metric).
- **convergence_auc**: area under the normalized convergence curve (higher = found good configs earlier).

Each benchmark runs with multiple seeds; mean metrics across seeds are reported.

## Baselines (paper-cited reference implementations)
- **random_search** — Bergstra and Bengio (JMLR 2012).
- **tpe** — Bergstra et al. (NIPS 2011); paper-default `gamma = 0.25`, 24 candidate configurations per suggestion.
- **hyperband** — Li et al. (JMLR 2017; arXiv:1603.06560); paper-default `eta = 3`.
- **bohb** — Falkner et al. (ICML 2018; arXiv:1807.01774); same `eta = 3` and TPE-style model on the highest budget with enough observations.
- **dehb** — Awad et al. (IJCAI 2021; arXiv:2105.09821); paper-default `eta = 3`, mutation factor `F = 0.5`, crossover `Cr = 0.5`.
- **optuna_cma** — Optuna's CMA-ES sampler wrapping Hansen and Ostermeier (2001).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `scikit-learn/custom_hpo.py`
- editable lines **255–326**




## Readable Context


### `scikit-learn/custom_hpo.py`  [EDITABLE — lines 255–326 only]

```python
     1: """
     2: Hyperparameter Optimization — Custom Strategy Template
     3: 
     4: This script runs a complete HPO loop on real ML model tuning benchmarks.
     5: The agent should implement CustomHPOStrategy which proposes hyperparameter
     6: configurations to evaluate, given a search space and history of past trials.
     7: 
     8: Usage:
     9:     python scikit-learn/custom_hpo.py --benchmark xgboost --seed 42 \
    10:         --budget 50 --output-dir ./out
    11: """
    12: 
    13: import argparse
    14: import json
    15: import math
    16: import os
    17: import time
    18: import warnings
    19: from dataclasses import dataclass, field
    20: from typing import Any, Dict, List, Optional, Tuple
    21: 
    22: import numpy as np
    23: from scipy.stats import norm as scipy_norm
    24: 
    25: from sklearn.datasets import (
    26:     fetch_california_housing,
    27:     load_breast_cancer,
    28:     load_diabetes,
    29: )
    30: from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    31: from sklearn.model_selection import cross_val_score
    32: from sklearn.neural_network import MLPClassifier, MLPRegressor
    33: from sklearn.preprocessing import StandardScaler
    34: from sklearn.svm import SVC, SVR
    35: 
    36: warnings.filterwarnings("ignore")
    37: 
    38: # ================================================================
    39: # FIXED — Data types and search space definitions (do not modify)
    40: # ================================================================
    41: 
    42: 
    43: @dataclass
    44: class HParam:
    45:     """A single hyperparameter specification."""
    46:     name: str
    47:     type: str  # "float", "int", "categorical"
    48:     low: Optional[float] = None
    49:     high: Optional[float] = None
    50:     log_scale: bool = False
    51:     choices: Optional[list] = None
    52: 
    53: 
    54: @dataclass
    55: class Trial:
    56:     """Record of one evaluated configuration."""
    57:     config: Dict[str, Any]
    58:     score: float  # validation score (higher is better)
    59:     budget: float = 1.0  # fidelity/budget fraction (1.0 = full)
    60: 
    61: 
    62: @dataclass
    63: class SearchSpace:
    64:     """Hyperparameter search space."""
    65:     params: List[HParam] = field(default_factory=list)
    66: 
    67:     @property
    68:     def dim(self) -> int:
    69:         return len(self.params)
    70: 
    71:     def sample_uniform(self, rng: np.random.RandomState) -> Dict[str, Any]:
    72:         """Sample a random configuration uniformly from the space."""
    73:         config = {}
    74:         for p in self.params:
    75:             if p.type == "categorical":
    76:                 config[p.name] = rng.choice(p.choices)
    77:             elif p.type == "float":
    78:                 if p.log_scale:
    79:                     log_val = rng.uniform(np.log(p.low), np.log(p.high))
    80:                     config[p.name] = float(np.exp(log_val))
    81:                 else:
    82:                     config[p.name] = float(rng.uniform(p.low, p.high))
    83:             elif p.type == "int":
    84:                 if p.log_scale:
    85:                     log_val = rng.uniform(np.log(p.low), np.log(p.high))
    86:                     config[p.name] = int(round(np.exp(log_val)))
    87:                 else:
    88:                     config[p.name] = int(rng.randint(p.low, p.high + 1))
    89:         return config
    90: 
    91:     def clip(self, config: Dict[str, Any]) -> Dict[str, Any]:
    92:         """Clip configuration values to valid ranges."""
    93:         clipped = {}
    94:         for p in self.params:
    95:             val = config.get(p.name)
    96:             if val is None:
    97:                 raise ValueError(f"Missing hyperparameter: {p.name}")
    98:             if p.type == "categorical":
    99:                 if val not in p.choices:
   100:                     raise ValueError(f"{p.name}={val} not in {p.choices}")
   101:                 clipped[p.name] = val
   102:             elif p.type == "float":
   103:                 clipped[p.name] = float(np.clip(val, p.low, p.high))
   104:             elif p.type == "int":
   105:                 clipped[p.name] = int(np.clip(round(val), p.low, p.high))
   106:         return clipped
   107: 
   108: 
   109: # ================================================================
   110: # FIXED — Benchmark problems (do not modify)
   111: # ================================================================
   112: 
   113: 
   114: def _make_xgboost_benchmark():
   115:     """XGBoost hyperparameter tuning on California Housing (regression).
   116: 
   117:     Search space: n_estimators, max_depth, learning_rate, subsample,
   118:                   min_samples_split, min_samples_leaf.
   119:     Metric: neg_mean_squared_error (converted to positive = higher is better).
   120:     """
   121:     data = fetch_california_housing(data_home=os.environ.get("SKLEARN_DATA_HOME"))
   122:     X, y = data.data, data.target
   123:     scaler = StandardScaler()
   124:     X = scaler.fit_transform(X)
   125: 
   126:     space = SearchSpace(params=[
   127:         HParam("n_estimators", "int", low=50, high=500),
   128:         HParam("max_depth", "int", low=2, high=10),
   129:         HParam("learning_rate", "float", low=0.001, high=0.5, log_scale=True),
   130:         HParam("subsample", "float", low=0.5, high=1.0),
   131:         HParam("min_samples_split", "int", low=2, high=20),
   132:         HParam("min_samples_leaf", "int", low=1, high=10),
   133:     ])
   134: 
   135:     def objective(config: Dict[str, Any], budget: float = 1.0) -> float:
   136:         n_est = config["n_estimators"]
   137:         if budget < 1.0:
   138:             n_est = max(10, int(n_est * budget))
   139:         model = GradientBoostingRegressor(
   140:             n_estimators=n_est,
   141:             max_depth=config["max_depth"],
   142:             learning_rate=config["learning_rate"],
   143:             subsample=config["subsample"],
   144:             min_samples_split=config["min_samples_split"],
   145:             min_samples_leaf=config["min_samples_leaf"],
   146:             random_state=0,
   147:         )
   148:         scores = cross_val_score(model, X, y, cv=3,
   149:                                  scoring="neg_mean_squared_error")
   150:         return float(scores.mean())  # negative MSE, higher is better
   151: 
   152:     return space, objective
   153: 
   154: 
   155: def _make_svm_benchmark():
   156:     """SVM hyperparameter tuning on Breast Cancer (classification).
   157: 
   158:     Search space: C, gamma, kernel.
   159:     Metric: accuracy (higher is better).
   160:     """
   161:     data = load_breast_cancer()
   162:     X, y = data.data, data.target
   163:     scaler = StandardScaler()
   164:     X = scaler.fit_transform(X)
   165: 
   166:     space = SearchSpace(params=[
   167:         HParam("C", "float", low=0.001, high=100.0, log_scale=True),
   168:         HParam("gamma", "float", low=1e-5, high=10.0, log_scale=True),
   169:         HParam("kernel", "categorical", choices=["rbf", "poly", "sigmoid"]),
   170:     ])
   171: 
   172:     def objective(config: Dict[str, Any], budget: float = 1.0) -> float:
   173:         cv_folds = max(2, int(5 * budget))
   174:         model = SVC(
   175:             C=config["C"],
   176:             gamma=config["gamma"],
   177:             kernel=config["kernel"],
   178:             random_state=0,
   179:         )
   180:         scores = cross_val_score(model, X, y, cv=cv_folds,
   181:                                  scoring="accuracy")
   182:         return float(scores.mean())
   183: 
   184:     return space, objective
   185: 
   186: 
   187: def _make_nn_benchmark():
   188:     """Small neural network hyperparameter tuning on Diabetes (regression).
   189: 
   190:     Search space: hidden_layer_1, hidden_layer_2, learning_rate_init, alpha,
   191:                   batch_size, activation.
   192:     Metric: neg_mean_squared_error (converted to positive = higher is better).
   193:     """
   194:     data = load_diabetes()
   195:     X, y = data.data, data.target
   196:     scaler = StandardScaler()
   197:     X = scaler.fit_transform(X)
   198: 
   199:     space = SearchSpace(params=[
   200:         HParam("hidden_layer_1", "int", low=16, high=256, log_scale=True),
   201:         HParam("hidden_layer_2", "int", low=8, high=128, log_scale=True),
   202:         HParam("learning_rate_init", "float", low=1e-4, high=0.1,
   203:                log_scale=True),
   204:         HParam("alpha", "float", low=1e-6, high=0.1, log_scale=True),
   205:         HParam("batch_size", "int", low=16, high=128),
   206:         HParam("activation", "categorical", choices=["relu", "tanh"]),
   207:     ])
   208: 
   209:     def objective(config: Dict[str, Any], budget: float = 1.0) -> float:
   210:         max_iter = max(50, int(500 * budget))
   211:         model = MLPRegressor(
   212:             hidden_layer_sizes=(config["hidden_layer_1"],
   213:                                 config["hidden_layer_2"]),
   214:             learning_rate_init=config["learning_rate_init"],
   215:             alpha=config["alpha"],
   216:             batch_size=config["batch_size"],
   217:             activation=config["activation"],
   218:             max_iter=max_iter,
   219:             random_state=0,
   220:             early_stopping=True,
   221:             validation_fraction=0.15,
   222:         )
   223:         scores = cross_val_score(model, X, y, cv=3,
   224:                                  scoring="neg_mean_squared_error")
   225:         return float(scores.mean())
   226: 
   227:     return space, objective
   228: 
   229: 
   230: BENCHMARKS = {
   231:     "xgboost": {
   232:         "make_fn": _make_xgboost_benchmark,
   233:         "budget": 50,
   234:         "description": "Gradient Boosting Regressor on California Housing",
   235:     },
   236:     "svm": {
   237:         "make_fn": _make_svm_benchmark,
   238:         "budget": 40,
   239:         "description": "SVM Classifier on Breast Cancer",
   240:     },
   241:     "nn": {
   242:         "make_fn": _make_nn_benchmark,
   243:         "budget": 40,
   244:         "description": "MLP Regressor on Diabetes",
   245:     },
   246: }
   247: 
   248: 
   249: # ================================================================
   250: # EDITABLE — Custom HPO strategy (lines 255 to 326)
   251: # The agent modifies ONLY this section.
   252: # ================================================================
   253: 
   254: 
   255: class CustomHPOStrategy:
   256:     """Custom hyperparameter optimization strategy.
   257: 
   258:     The agent should implement suggest() which proposes the next
   259:     hyperparameter configuration to evaluate, given the search space
   260:     and history of previous trials.
   261: 
   262:     The strategy is called repeatedly in a loop:
   263:         1. strategy.suggest(space, history, budget_left) -> (config, fidelity)
   264:         2. config is evaluated -> score
   265:         3. Trial(config, score, fidelity) is added to history
   266:         4. Repeat until budget exhausted
   267: 
   268:     Available utilities:
   269:         space.params        — list of HParam objects with name, type, range
   270:         space.dim           — number of hyperparameters
   271:         space.sample_uniform(rng) — sample random config
   272:         space.clip(config)  — clip values to valid ranges
   273: 
   274:         trial.config        — dict of hyperparameter values
   275:         trial.score         — observed validation score (higher is better)
   276:         trial.budget        — fidelity fraction used (1.0 = full evaluation)
   277: 
   278:     Useful scipy:
   279:         from scipy.stats import norm
   280:         norm.cdf(x), norm.pdf(x)
   281: 
   282:     Useful numpy:
   283:         np.random.RandomState for reproducibility
   284: 
   285:     Args:
   286:         seed: random seed for reproducibility
   287: 
   288:     Returns from suggest():
   289:         config: dict mapping param names to values
   290:         fidelity: float in (0, 1] — fraction of full evaluation budget.
   291:                   Use 1.0 for full-fidelity evaluation.
   292:                   Lower values = cheaper evaluation (e.g., fewer epochs/trees).
   293:     """
   294: 
   295:     def __init__(self, seed: int = 42):
   296:         """Initialize the strategy.
   297: 
   298:         Default: stores seed and creates RNG.
   299:         The agent may add any internal state needed.
   300:         """
   301:         self.seed = seed
   302:         self.rng = np.random.RandomState(seed)
   303: 
   304:     def suggest(
   305:         self,
   306:         space: SearchSpace,
   307:         history: List[Trial],
   308:         budget_left: int,
   309:     ) -> Tuple[Dict[str, Any], float]:
   310:         """Propose the next configuration to evaluate.
   311: 
   312:         Default: uniform random search (poor — replace with a better
   313:         strategy).
   314: 
   315:         Args:
   316:             space: search space definition
   317:             history: list of previously evaluated trials
   318:             budget_left: number of full-fidelity evaluations remaining
   319: 
   320:         Returns:
   321:             config: dict of hyperparameter name -> value
   322:             fidelity: float in (0, 1], fraction of full evaluation
   323:         """
   324:         config = space.sample_uniform(self.rng)
   325:         return config, 1.0
   326: 
   327: 
   328: # ================================================================
   329: # FIXED — HPO loop and evaluation (do not modify below)
   330: # ================================================================
   331: 
   332: 
   333: def run_hpo_loop(benchmark_name: str, seed: int, budget: int,
   334:                  output_dir: str):
   335:     """Run full HPO loop and report metrics."""
   336:     cfg = BENCHMARKS[benchmark_name]
   337:     space, objective = cfg["make_fn"]()
   338: 
   339:     strategy = CustomHPOStrategy(seed=seed)
   340:     history: List[Trial] = []
   341:     best_score = -np.inf
   342:     best_config = None
   343:     total_cost = 0.0
   344:     convergence_curve = []
   345:     convergence_threshold_reached = budget  # default: never reached early
   346: 
   347:     # Determine convergence threshold (90% of budget's potential)
   348:     # We'll compute this after seeing some results
   349: 
   350:     start_time = time.time()
   351: 
   352:     eval_count = 0
   353:     while total_cost < budget:
   354:         budget_left = budget - total_cost
   355:         if budget_left < 0.1:
   356:             break
   357: 
   358:         config, fidelity = strategy.suggest(space, history, int(budget_left))
   359:         fidelity = float(np.clip(fidelity, 0.1, 1.0))
   360:         config = space.clip(config)
   361: 
   362:         score = objective(config, budget=fidelity)
   363:         trial = Trial(config=config, score=score, budget=fidelity)
   364:         history.append(trial)
   365: 
   366:         total_cost += fidelity
   367:         eval_count += 1
   368: 
   369:         if score > best_score:
   370:             best_score = score
   371:             best_config = config.copy()
   372: 
   373:         convergence_curve.append({
   374:             "eval": eval_count,
   375:             "cost": total_cost,
   376:             "best_score": best_score,
   377:         })
   378: 
   379:         if eval_count % 5 == 0 or total_cost >= budget - 0.1:
   380:             elapsed = time.time() - start_time
   381:             print(
   382:                 f"TRAIN_METRICS eval={eval_count} cost={total_cost:.1f}/{budget} "
   383:                 f"best_score={best_score:.6f} elapsed={elapsed:.1f}s",
   384:                 flush=True,
   385:             )
   386: 
   387:     elapsed = time.time() - start_time
   388: 
   389:     # Compute convergence speed: area under the normalized curve (AUC)
   390:     # Higher AUC = faster convergence (found good configs earlier)
   391:     if len(convergence_curve) > 1:
   392:         costs = [c["cost"] / budget for c in convergence_curve]
   393:         scores = [c["best_score"] for c in convergence_curve]
   394:         # Normalize scores to [0, 1] range
   395:         s_min, s_max = min(scores), max(scores)
   396:         if s_max > s_min:
   397:             norm_scores = [(s - s_min) / (s_max - s_min) for s in scores]
   398:         else:
   399:             norm_scores = [1.0] * len(scores)
   400:         # Trapezoidal AUC
   401:         auc = float(np.trapezoid(norm_scores, costs)) if hasattr(np, 'trapezoid') else float(np.trapz(norm_scores, costs))
   402:     else:
   403:         auc = 0.0
   404: 
   405:     # Print final metrics
   406:     print(f"TEST_METRICS best_val_score={best_score:.6f}", flush=True)
   407:     print(f"TEST_METRICS convergence_auc={auc:.6f}", flush=True)
   408:     print(f"TEST_METRICS total_evals={eval_count}", flush=True)
   409: 
   410:     # Save results
   411:     os.makedirs(output_dir, exist_ok=True)
   412:     results = {
   413:         "benchmark": benchmark_name,
   414:         "seed": seed,
   415:         "budget": budget,
   416:         "total_cost": total_cost,
   417:         "total_evals": eval_count,
   418:         "best_score": best_score,
   419:         "best_config": best_config,
   420:         "convergence_auc": auc,
   421:         "elapsed_seconds": elapsed,
   422:         "convergence_curve": convergence_curve,
   423:     }
   424:     with open(os.path.join(output_dir,
   425:                            f"{benchmark_name}_results.json"), "w") as f:
   426:         json.dump(results, f, indent=2)
   427: 
   428:     return best_score, auc
   429: 
   430: 
   431: def main():
   432:     parser = argparse.ArgumentParser(
   433:         description="Hyperparameter Optimization Strategy Benchmark")
   434:     parser.add_argument("--benchmark", type=str, required=True,
   435:                         choices=list(BENCHMARKS.keys()))
   436:     parser.add_argument("--seed", type=int,
   437:                         default=int(os.environ.get("SEED", 42)))
   438:     parser.add_argument("--budget", type=int, default=None,
   439:                         help="Override default budget for this benchmark")
   440:     parser.add_argument("--output-dir", type=str,
   441:                         default=os.environ.get("OUTPUT_DIR", "./output"))
   442:     args = parser.parse_args()
   443: 
   444:     benchmark_budget = args.budget or BENCHMARKS[args.benchmark]["budget"]
   445: 
   446:     print(f"Running HPO benchmark: {args.benchmark} "
   447:           f"(seed={args.seed}, budget={benchmark_budget})", flush=True)
   448:     best_score, auc = run_hpo_loop(
   449:         args.benchmark, args.seed, benchmark_budget, args.output_dir)
   450:     print(f"Final best score on {args.benchmark}: {best_score:.6f} "
   451:           f"(convergence AUC: {auc:.4f})", flush=True)
   452: 
   453: 
   454: if __name__ == "__main__":
   455:     main()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **xgboost** — wall-clock budget `01:00:00`, compute share `0.33`
- **svm** — wall-clock budget `01:00:00`, compute share `0.33`
- **nn** — wall-clock budget `01:00:00`, compute share `0.33`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `random_search` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_hpo.py`:

```python
Lines 255–272:
   252: # ================================================================
   253: 
   254: 
   255: 
   256: class CustomHPOStrategy:
   257:     """Random Search: sample configurations uniformly at random."""
   258: 
   259:     def __init__(self, seed: int = 42):
   260:         self.seed = seed
   261:         self.rng = np.random.RandomState(seed)
   262: 
   263:     def suggest(
   264:         self,
   265:         space: SearchSpace,
   266:         history: List[Trial],
   267:         budget_left: int,
   268:     ) -> Tuple[Dict[str, Any], float]:
   269:         config = space.sample_uniform(self.rng)
   270:         return config, 1.0
   271: 
   272: 
   273: 
   274: # ================================================================
   275: # FIXED — HPO loop and evaluation (do not modify below)
```

### `tpe` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_hpo.py`:

```python
Lines 255–335:
   252: # ================================================================
   253: 
   254: 
   255: 
   256: class CustomHPOStrategy:
   257:     """Tree-structured Parzen Estimator (TPE)."""
   258: 
   259:     def __init__(self, seed: int = 42):
   260:         self.seed = seed
   261:         self.rng = np.random.RandomState(seed)
   262:         self.gamma = 0.25  # fraction of best observations for l(x)
   263:         self.n_startup = 10  # random search before modelling
   264:         self.n_ei_candidates = 24  # candidates to score with EI
   265: 
   266:     def _encode(self, config, space):
   267:         """Encode a config to a numeric vector in [0,1]."""
   268:         vec = []
   269:         for p in space.params:
   270:             val = config[p.name]
   271:             if p.type == "categorical":
   272:                 # One-hot-ish: use index / len
   273:                 idx = p.choices.index(val)
   274:                 vec.append(idx / max(len(p.choices) - 1, 1))
   275:             elif p.type in ("float", "int"):
   276:                 if p.log_scale:
   277:                     v = (np.log(val) - np.log(p.low)) / (np.log(p.high) - np.log(p.low))
   278:                 else:
   279:                     v = (val - p.low) / (p.high - p.low)
   280:                 vec.append(float(np.clip(v, 0, 1)))
   281:         return np.array(vec)
   282: 
   283:     def _kde_logpdf(self, x, samples, bw):
   284:         """Simple Gaussian KDE log-density at x."""
   285:         diffs = x[None, :] - samples  # (n_samples, dim)
   286:         return float(
   287:             np.log(np.mean(np.exp(-0.5 * np.sum(diffs**2 / bw**2, axis=1))) + 1e-30)
   288:         )
   289: 
   290:     def suggest(
   291:         self,
   292:         space: SearchSpace,
   293:         history: List[Trial],
   294:         budget_left: int,
   295:     ) -> Tuple[Dict[str, Any], float]:
   296:         if len(history) < self.n_startup:
   297:             return space.sample_uniform(self.rng), 1.0
   298: 
   299:         # Split observations into good (l) and bad (g)
   300:         scores = np.array([t.score for t in history])
   301:         n_good = max(1, int(self.gamma * len(history)))
   302:         threshold = np.sort(scores)[-n_good]
   303: 
   304:         good_vecs = np.array([
   305:             self._encode(t.config, space)
   306:             for t in history if t.score >= threshold
   307:         ])
   308:         bad_vecs = np.array([
   309:             self._encode(t.config, space)
   310:             for t in history if t.score < threshold
   311:         ])
   312: 
   313:         if len(bad_vecs) == 0:
   314:             bad_vecs = good_vecs.copy()
   315: 
   316:         # Bandwidth: Scott's rule
   317:         bw_good = max(0.05, good_vecs.std() + 1e-6)
   318:         bw_bad = max(0.05, bad_vecs.std() + 1e-6)
   319: 
   320:         # Generate candidates and score them by l(x)/g(x)
   321:         best_score = -np.inf
   322:         best_config = None
   323:         for _ in range(self.n_ei_candidates):
   324:             candidate = space.sample_uniform(self.rng)
   325:             x = self._encode(candidate, space)
   326:             log_l = self._kde_logpdf(x, good_vecs, bw_good)
   327:             log_g = self._kde_logpdf(x, bad_vecs, bw_bad)
   328:             ei = log_l - log_g
   329:             if ei > best_score:
   330:                 best_score = ei
   331:                 best_config = candidate
   332: 
   333:         return best_config, 1.0
   334: 
   335: 
   336: 
   337: # ================================================================
   338: # FIXED — HPO loop and evaluation (do not modify below)
```

### `hyperband` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_hpo.py`:

```python
Lines 255–345:
   252: # ================================================================
   253: 
   254: 
   255: 
   256: class CustomHPOStrategy:
   257:     """Hyperband: multi-fidelity with successive halving."""
   258: 
   259:     def __init__(self, seed: int = 42):
   260:         self.seed = seed
   261:         self.rng = np.random.RandomState(seed)
   262:         self.eta = 3  # halving rate
   263:         self.brackets = []  # list of (configs, fidelity, scores)
   264:         self._initialized = False
   265:         self._queue = []  # queue of (config, fidelity) to suggest
   266: 
   267:     def _init_brackets(self, space, total_budget):
   268:         """Initialize Hyperband brackets (Successive Halving instances)."""
   269:         s_max = max(0, int(np.floor(np.log(total_budget) / np.log(self.eta))))
   270:         s_max = min(s_max, 4)  # cap brackets
   271: 
   272:         for s in range(s_max, -1, -1):
   273:             n = int(np.ceil((s_max + 1) / (s + 1)) * self.eta ** s)
   274:             n = min(n, total_budget)
   275:             r = max(1.0 / self.eta ** s, 0.1)
   276: 
   277:             # Generate random configs for this bracket
   278:             configs = [space.sample_uniform(self.rng) for _ in range(n)]
   279:             # Queue low-fidelity evaluations
   280:             for cfg in configs:
   281:                 self._queue.append((cfg, r))
   282: 
   283:             self.brackets.append({
   284:                 "configs": configs,
   285:                 "fidelity": r,
   286:                 "scores": [None] * len(configs),
   287:                 "round": 0,
   288:                 "s": s,
   289:             })
   290: 
   291:     def _advance_bracket(self, bracket):
   292:         """Advance a bracket: keep top 1/eta, increase fidelity."""
   293:         configs = bracket["configs"]
   294:         scores = bracket["scores"]
   295: 
   296:         # Sort by score, keep top 1/eta
   297:         paired = [(s, c) for s, c in zip(scores, configs) if s is not None]
   298:         if not paired:
   299:             return
   300:         paired.sort(key=lambda x: x[0], reverse=True)
   301:         n_keep = max(1, len(paired) // self.eta)
   302:         survivors = paired[:n_keep]
   303: 
   304:         new_fidelity = min(bracket["fidelity"] * self.eta, 1.0)
   305:         bracket["configs"] = [c for _, c in survivors]
   306:         bracket["scores"] = [None] * len(survivors)
   307:         bracket["fidelity"] = new_fidelity
   308:         bracket["round"] += 1
   309: 
   310:         for cfg in bracket["configs"]:
   311:             self._queue.append((cfg, new_fidelity))
   312: 
   313:     def suggest(
   314:         self,
   315:         space: SearchSpace,
   316:         history: List[Trial],
   317:         budget_left: int,
   318:     ) -> Tuple[Dict[str, Any], float]:
   319:         if not self._initialized:
   320:             self._init_brackets(space, budget_left + len(history))
   321:             self._initialized = True
   322: 
   323:         # Update bracket scores from history
   324:         if history:
   325:             last = history[-1]
   326:             for bracket in self.brackets:
   327:                 for i, cfg in enumerate(bracket["configs"]):
   328:                     if (bracket["scores"][i] is None
   329:                             and cfg == last.config
   330:                             and abs(bracket["fidelity"] - last.budget) < 0.05):
   331:                         bracket["scores"][i] = last.score
   332: 
   333:                 # Check if bracket round complete
   334:                 if all(s is not None for s in bracket["scores"]):
   335:                     if bracket["fidelity"] < 1.0 and len(bracket["configs"]) > 1:
   336:                         self._advance_bracket(bracket)
   337: 
   338:         # Return from queue
   339:         if self._queue:
   340:             return self._queue.pop(0)
   341: 
   342:         # Fallback: random
   343:         return space.sample_uniform(self.rng), 1.0
   344: 
   345: 
   346: 
   347: # ================================================================
   348: # FIXED — HPO loop and evaluation (do not modify below)
```

### `dehb` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_hpo.py`:

```python
Lines 255–466:
   252: # ================================================================
   253: 
   254: 
   255: 
   256: class CustomHPOStrategy:
   257:     """DEHB: Differential Evolution + Hyperband."""
   258: 
   259:     def __init__(self, seed: int = 42):
   260:         self.seed = seed
   261:         self.rng = np.random.RandomState(seed)
   262:         self.eta = 3
   263:         self.mutation_factor = 0.5
   264:         self.crossover_prob = 0.5
   265:         self._initialized = False
   266:         # Queue items: (cfg, fid)
   267:         self._queue = []
   268:         # fid -> list of (vec, score)  -- current evaluated population
   269:         self._populations = {}
   270:         self._fidelities = []
   271:         # fid -> list of (trial_vec, target_idx) for pending DE trials at fid.
   272:         # When score arrives for a trial, do DE selection against target.
   273:         self._pending = {}
   274:         # Ensure we only promote each fid->next once per generation
   275:         self._promoted_rounds = {}
   276: 
   277:     def _encode(self, config, space):
   278:         vec = []
   279:         for p in space.params:
   280:             val = config[p.name]
   281:             if p.type == "categorical":
   282:                 idx = p.choices.index(val)
   283:                 vec.append(idx / max(len(p.choices) - 1, 1))
   284:             elif p.type in ("float", "int"):
   285:                 if p.log_scale:
   286:                     v = (np.log(val) - np.log(p.low)) / (np.log(p.high) - np.log(p.low))
   287:                 else:
   288:                     v = (val - p.low) / (p.high - p.low)
   289:                 vec.append(float(np.clip(v, 0, 1)))
   290:         return np.array(vec)
   291: 
   292:     def _decode(self, vec, space):
   293:         config = {}
   294:         for i, p in enumerate(space.params):
   295:             v = float(np.clip(vec[i], 0, 1))
   296:             if p.type == "categorical":
   297:                 idx = int(round(v * max(len(p.choices) - 1, 1)))
   298:                 idx = min(idx, len(p.choices) - 1)
   299:                 config[p.name] = p.choices[idx]
   300:             elif p.type == "float":
   301:                 if p.log_scale:
   302:                     config[p.name] = float(np.exp(
   303:                         np.log(p.low) + v * (np.log(p.high) - np.log(p.low))))
   304:                 else:
   305:                     config[p.name] = float(p.low + v * (p.high - p.low))
   306:             elif p.type == "int":
   307:                 if p.log_scale:
   308:                     config[p.name] = int(round(np.exp(
   309:                         np.log(p.low) + v * (np.log(p.high) - np.log(p.low)))))
   310:                 else:
   311:                     config[p.name] = int(round(p.low + v * (p.high - p.low)))
   312:         return config
   313: 
   314:     def _de_mutate(self, target_idx, population):
   315:         """DE/rand/1/bin mutation and crossover."""
   316:         pop_vecs = [p[0] for p in population]
   317:         n = len(pop_vecs)
   318:         if n < 4:
   319:             return pop_vecs[target_idx] + self.rng.randn(len(pop_vecs[0])) * 0.1
   320: 
   321:         idxs = list(range(n))
   322:         idxs.remove(target_idx)
   323:         a, b, c = self.rng.choice(idxs, 3, replace=False)
   324:         mutant = pop_vecs[a] + self.mutation_factor * (pop_vecs[b] - pop_vecs[c])
   325:         mutant = np.clip(mutant, 0, 1)
   326: 
   327:         # Crossover
   328:         dim = len(mutant)
   329:         cross_mask = self.rng.rand(dim) < self.crossover_prob
   330:         j_rand = self.rng.randint(dim)
   331:         cross_mask[j_rand] = True
   332:         trial = np.where(cross_mask, mutant, pop_vecs[target_idx])
   333:         return trial
   334: 
   335:     def _init(self, space, total_budget):
   336:         s_max = max(0, int(np.floor(np.log(total_budget) / np.log(self.eta))))
   337:         s_max = min(s_max, 3)
   338:         pop_size = max(4, space.dim + 1)
   339: 
   340:         # Build strictly increasing fidelity ladder with no duplicates.
   341:         # The harness clips fidelity to [0.1, 1.0] in run_hpo_loop, so we
   342:         # must still respect that floor, but we dedupe to avoid wasting
   343:         # SH rounds on near-identical fidelities (e.g. 0.037 -> 0.1 and
   344:         # 0.111 would otherwise both collapse near 0.1).
   345:         seen = set()
   346:         for s in range(s_max, -1, -1):
   347:             raw = 1.0 / self.eta ** s
   348:             fid = max(raw, 0.1)
   349:             key = round(fid, 3)
   350:             if key in seen:
   351:                 continue
   352:             seen.add(key)
   353:             self._fidelities.append(fid)
   354:             self._pending[fid] = []
   355:             pop = []
   356:             for i in range(pop_size):
   357:                 cfg = space.sample_uniform(self.rng)
   358:                 vec = self._encode(cfg, space)
   359:                 pop.append((vec, None))
   360:                 # target_idx = -1 -i means initial eval; we encode as negative
   361:                 # of (i+1) so it's distinguishable from real DE trial indices.
   362:                 self._pending[fid].append((vec, -(i + 1)))
   363:                 self._queue.append((cfg, fid))
   364:             self._populations[fid] = pop
   365:         self._initialized = True
   366: 
   367:     def suggest(
   368:         self,
   369:         space: SearchSpace,
   370:         history: List[Trial],
   371:         budget_left: int,
   372:     ) -> Tuple[Dict[str, Any], float]:
   373:         if not self._initialized:
   374:             self._init(space, budget_left + len(history))
   375: 
   376:         # When a trial comes back, match to pending and do DE selection
   377:         # against the target it was generated from.
   378:         if history:
   379:             last = history[-1]
   380:             last_vec = self._encode(last.config, space)
   381:             # Match by (fidelity, vec) among pending trials
   382:             for fid in self._fidelities:
   383:                 if abs(fid - last.budget) >= 0.05:
   384:                     continue
   385:                 pending = self._pending[fid]
   386:                 pop = self._populations[fid]
   387:                 for j, (trial_vec, tgt_idx) in enumerate(pending):
   388:                     if np.allclose(trial_vec, last_vec, atol=1e-3):
   389:                         if tgt_idx < 0:
   390:                             # Initial-population eval: just fill in score.
   391:                             real_idx = -(tgt_idx + 1)
   392:                             pop[real_idx] = (trial_vec, last.score)
   393:                         else:
   394:                             tgt_vec, tgt_score = pop[tgt_idx]
   395:                             # DE selection: keep better of target/trial.
   396:                             if tgt_score is None or last.score >= tgt_score:
   397:                                 pop[tgt_idx] = (trial_vec, last.score)
   398:                             # else: keep target unchanged.
   399:                         pending.pop(j)
   400:                         break
   401: 
   402:         if self._queue:
   403:             return self._queue.pop(0)
   404: 
   405:         sorted_fids = sorted(self._fidelities)
   406:         # Track generation counter per fidelity to alternate between
   407:         # DE evolution at lowest fidelity and inheritance-based
   408:         # promotion to higher fidelities (DEHB modified SH).
   409:         if not hasattr(self, "_gen_count"):
   410:             self._gen_count = {f: 0 for f in sorted_fids}
   411: 
   412:         lo_fid = sorted_fids[0]
   413:         lo_pop = self._populations[lo_fid]
   414:         lo_ready = (all(s is not None for _, s in lo_pop)
   415:                     and not self._pending[lo_fid])
   416: 
   417:         if lo_ready:
   418:             # Step 1: evolve the lowest-fidelity population.
   419:             for tgt_idx in range(len(lo_pop)):
   420:                 trial_vec = self._de_mutate(tgt_idx, lo_pop)
   421:                 trial_cfg = self._decode(trial_vec, space)
   422:                 trial_cfg = space.clip(trial_cfg)
   423:                 self._pending[lo_fid].append((trial_vec, tgt_idx))
   424:                 self._queue.append((trial_cfg, lo_fid))
   425:             self._gen_count[lo_fid] += 1
   426:             # Step 2: every eta generations, promote top configs to the
   427:             # next fidelity level via successive halving (DEHB design).
   428:             for i in range(len(sorted_fids) - 1):
   429:                 hi_fid = sorted_fids[i + 1]
   430:                 src_fid = sorted_fids[i]
   431:                 if self._pending[hi_fid]:
   432:                     continue
   433:                 src_pop = self._populations[src_fid]
   434:                 if any(s is None for _, s in src_pop):
   435:                     continue
   436:                 # Gate promotion so it runs at most once per fresh src
   437:                 # generation (avoid unbounded queue growth).
   438:                 if self._gen_count[hi_fid] >= self._gen_count[src_fid]:
   439:                     continue
   440:                 scored = sorted(
   441:                     [(s, v) for v, s in src_pop],
   442:                     key=lambda x: x[0],
   443:                     reverse=True,
   444:                 )
   445:                 n_promote = max(1, len(scored) // self.eta)
   446:                 top_vecs = [v for _, v in scored[:n_promote]]
   447:                 hi_pop = self._populations[hi_fid]
   448:                 # Overwrite hi_pop members with promoted vecs, then
   449:                 # re-evaluate each at the higher fidelity.
   450:                 new_hi_pop = []
   451:                 for k in range(len(hi_pop)):
   452:                     vec = top_vecs[k % len(top_vecs)]
   453:                     new_hi_pop.append((vec, None))
   454:                     cfg = self._decode(vec, space)
   455:                     cfg = space.clip(cfg)
   456:                     self._pending[hi_fid].append((vec, -(k + 1)))
   457:                     self._queue.append((cfg, hi_fid))
   458:                 self._populations[hi_fid] = new_hi_pop
   459:                 self._gen_count[hi_fid] = self._gen_count[src_fid]
   460:             if self._queue:
   461:                 return self._queue.pop(0)
   462: 
   463:         # Fallback: random full fidelity
   464:         return space.sample_uniform(self.rng), 1.0
   465: 
   466: 
   467: 
   468: # ================================================================
   469: # FIXED — HPO loop and evaluation (do not modify below)
```

### `bohb` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_hpo.py`:

```python
Lines 255–391:
   252: # ================================================================
   253: 
   254: 
   255: 
   256: class CustomHPOStrategy:
   257:     """BOHB: Bayesian Optimization + Hyperband."""
   258: 
   259:     def __init__(self, seed: int = 42):
   260:         self.seed = seed
   261:         self.rng = np.random.RandomState(seed)
   262:         self.eta = 3
   263:         self.gamma = 0.15  # fraction for good KDE
   264:         self.n_startup = 8  # random configs before model-guided
   265:         self.n_candidates = 24
   266:         self.bw_factor = 1.0
   267:         self._brackets = []
   268:         self._queue = []
   269:         self._initialized = False
   270:         self._all_trials = []  # (vec, score, fidelity)
   271: 
   272:     def _encode(self, config, space):
   273:         vec = []
   274:         for p in space.params:
   275:             val = config[p.name]
   276:             if p.type == "categorical":
   277:                 idx = p.choices.index(val)
   278:                 vec.append(idx / max(len(p.choices) - 1, 1))
   279:             elif p.type in ("float", "int"):
   280:                 if p.log_scale:
   281:                     v = (np.log(val) - np.log(p.low)) / (np.log(p.high) - np.log(p.low))
   282:                 else:
   283:                     v = (val - p.low) / (p.high - p.low)
   284:                 vec.append(float(np.clip(v, 0, 1)))
   285:         return np.array(vec)
   286: 
   287:     def _kde_logpdf(self, x, samples, bw):
   288:         diffs = x[None, :] - samples
   289:         return float(np.log(
   290:             np.mean(np.exp(-0.5 * np.sum(diffs**2 / bw**2, axis=1))) + 1e-30
   291:         ))
   292: 
   293:     def _sample_from_model(self, space):
   294:         """Sample config guided by TPE model or random."""
   295:         if len(self._all_trials) < self.n_startup:
   296:             return space.sample_uniform(self.rng)
   297: 
   298:         vecs = np.array([t[0] for t in self._all_trials])
   299:         scores = np.array([t[1] for t in self._all_trials])
   300:         n_good = max(1, int(self.gamma * len(scores)))
   301:         threshold = np.sort(scores)[-n_good]
   302: 
   303:         good = vecs[scores >= threshold]
   304:         bad = vecs[scores < threshold]
   305:         if len(bad) == 0:
   306:             bad = good.copy()
   307: 
   308:         bw_good = max(0.05, good.std() * self.bw_factor + 1e-6)
   309:         bw_bad = max(0.05, bad.std() * self.bw_factor + 1e-6)
   310: 
   311:         best_ei = -np.inf
   312:         best_cfg = None
   313:         for _ in range(self.n_candidates):
   314:             cfg = space.sample_uniform(self.rng)
   315:             x = self._encode(cfg, space)
   316:             log_l = self._kde_logpdf(x, good, bw_good)
   317:             log_g = self._kde_logpdf(x, bad, bw_bad)
   318:             ei = log_l - log_g
   319:             if ei > best_ei:
   320:                 best_ei = ei
   321:                 best_cfg = cfg
   322:         return best_cfg
   323: 
   324:     def _init_brackets(self, space, total_budget):
   325:         s_max = max(0, int(np.floor(np.log(total_budget) / np.log(self.eta))))
   326:         s_max = min(s_max, 3)
   327: 
   328:         for s in range(s_max, -1, -1):
   329:             n = int(np.ceil((s_max + 1) / (s + 1)) * self.eta ** s)
   330:             n = min(n, total_budget)
   331:             r = max(1.0 / self.eta ** s, 0.1)
   332: 
   333:             configs = [self._sample_from_model(space) for _ in range(n)]
   334:             for cfg in configs:
   335:                 self._queue.append((cfg, r))
   336: 
   337:             self._brackets.append({
   338:                 "configs": configs,
   339:                 "fidelity": r,
   340:                 "scores": [None] * len(configs),
   341:             })
   342: 
   343:     def suggest(
   344:         self,
   345:         space: SearchSpace,
   346:         history: List[Trial],
   347:         budget_left: int,
   348:     ) -> Tuple[Dict[str, Any], float]:
   349:         # Track all completed trials for the TPE model
   350:         if history:
   351:             last = history[-1]
   352:             vec = self._encode(last.config, space)
   353:             self._all_trials.append((vec, last.score, last.budget))
   354: 
   355:         if not self._initialized:
   356:             self._init_brackets(space, budget_left + len(history))
   357:             self._initialized = True
   358: 
   359:         # Update bracket scores
   360:         if history:
   361:             last = history[-1]
   362:             for bracket in self._brackets:
   363:                 for i, cfg in enumerate(bracket["configs"]):
   364:                     if (bracket["scores"][i] is None
   365:                             and cfg == last.config
   366:                             and abs(bracket["fidelity"] - last.budget) < 0.05):
   367:                         bracket["scores"][i] = last.score
   368: 
   369:                 # Advance complete brackets
   370:                 if all(s is not None for s in bracket["scores"]):
   371:                     if bracket["fidelity"] < 1.0 and len(bracket["configs"]) > 1:
   372:                         # Successive halving
   373:                         paired = list(zip(bracket["scores"], bracket["configs"]))
   374:                         paired.sort(key=lambda x: x[0], reverse=True)
   375:                         n_keep = max(1, len(paired) // self.eta)
   376:                         survivors = paired[:n_keep]
   377:                         new_fid = min(bracket["fidelity"] * self.eta, 1.0)
   378:                         bracket["configs"] = [c for _, c in survivors]
   379:                         bracket["scores"] = [None] * len(survivors)
   380:                         bracket["fidelity"] = new_fid
   381:                         for cfg in bracket["configs"]:
   382:                             self._queue.append((cfg, new_fid))
   383: 
   384:         if self._queue:
   385:             return self._queue.pop(0)
   386: 
   387:         # Generate new configs using TPE model at full fidelity
   388:         cfg = self._sample_from_model(space)
   389:         return cfg, 1.0
   390: 
   391: 
   392: 
   393: # ================================================================
   394: # FIXED — HPO loop and evaluation (do not modify below)
```

### `optuna_cma` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_hpo.py`:

```python
Lines 255–436:
   252: # ================================================================
   253: 
   254: 
   255: 
   256: class CustomHPOStrategy:
   257:     """CMA-ES: Covariance Matrix Adaptation Evolution Strategy."""
   258: 
   259:     def __init__(self, seed: int = 42):
   260:         self.seed = seed
   261:         self.rng = np.random.RandomState(seed)
   262:         self._initialized = False
   263:         self._mean = None
   264:         self._sigma = 0.3
   265:         self._C = None  # covariance matrix
   266:         self._p_sigma = None  # evolution path for sigma
   267:         self._p_c = None  # evolution path for C
   268:         self._gen = 0
   269:         self._lam = None  # population size
   270:         self._mu = None
   271:         self._weights = None
   272:         self._mu_eff = None
   273:         self._candidates = []
   274:         self._pending_evals = []
   275: 
   276:     def _encode(self, config, space):
   277:         vec = []
   278:         for p in space.params:
   279:             val = config[p.name]
   280:             if p.type == "categorical":
   281:                 idx = p.choices.index(val)
   282:                 vec.append(idx / max(len(p.choices) - 1, 1))
   283:             elif p.type in ("float", "int"):
   284:                 if p.log_scale:
   285:                     v = (np.log(val) - np.log(p.low)) / (np.log(p.high) - np.log(p.low))
   286:                 else:
   287:                     v = (val - p.low) / (p.high - p.low)
   288:                 vec.append(float(np.clip(v, 0, 1)))
   289:         return np.array(vec)
   290: 
   291:     def _decode(self, vec, space):
   292:         config = {}
   293:         for i, p in enumerate(space.params):
   294:             v = float(np.clip(vec[i], 0, 1))
   295:             if p.type == "categorical":
   296:                 idx = int(round(v * max(len(p.choices) - 1, 1)))
   297:                 idx = min(idx, len(p.choices) - 1)
   298:                 config[p.name] = p.choices[idx]
   299:             elif p.type == "float":
   300:                 if p.log_scale:
   301:                     config[p.name] = float(np.exp(
   302:                         np.log(p.low) + v * (np.log(p.high) - np.log(p.low))))
   303:                 else:
   304:                     config[p.name] = float(p.low + v * (p.high - p.low))
   305:             elif p.type == "int":
   306:                 if p.log_scale:
   307:                     config[p.name] = int(round(np.exp(
   308:                         np.log(p.low) + v * (np.log(p.high) - np.log(p.low)))))
   309:                 else:
   310:                     config[p.name] = int(round(p.low + v * (p.high - p.low)))
   311:         return config
   312: 
   313:     def _init_cma(self, dim):
   314:         self._mean = np.full(dim, 0.5)
   315:         self._C = np.eye(dim)
   316:         self._p_sigma = np.zeros(dim)
   317:         self._p_c = np.zeros(dim)
   318:         self._lam = 4 + int(3 * np.log(dim))
   319:         self._mu = self._lam // 2
   320:         weights = np.log(self._mu + 0.5) - np.log(np.arange(1, self._mu + 1))
   321:         self._weights = weights / weights.sum()
   322:         self._mu_eff = 1.0 / np.sum(self._weights ** 2)
   323:         self._initialized = True
   324: 
   325:     def _sample_population(self, space):
   326:         dim = space.dim
   327:         # Eigendecomposition of C
   328:         eigvals, eigvecs = np.linalg.eigh(self._C)
   329:         eigvals = np.maximum(eigvals, 1e-20)
   330:         sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
   331: 
   332:         self._candidates = []
   333:         self._pending_evals = []
   334:         for _ in range(self._lam):
   335:             z = self.rng.randn(dim)
   336:             x = self._mean + self._sigma * sqrt_C @ z
   337:             x = np.clip(x, 0, 1)
   338:             cfg = self._decode(x, space)
   339:             cfg = space.clip(cfg)
   340:             self._candidates.append((x, cfg, None))
   341:             self._pending_evals.append(cfg)
   342: 
   343:     def _update(self, space):
   344:         """CMA-ES update step after a full generation is evaluated."""
   345:         dim = space.dim
   346: 
   347:         # Sort by score (descending — we maximize)
   348:         scored = [(s, x) for x, _, s in self._candidates if s is not None]
   349:         scored.sort(key=lambda p: p[0], reverse=True)
   350: 
   351:         # Recombination
   352:         old_mean = self._mean.copy()
   353:         self._mean = np.zeros(dim)
   354:         for i in range(self._mu):
   355:             self._mean += self._weights[i] * scored[i][1]
   356: 
   357:         # Evolution paths
   358:         c_sigma = (self._mu_eff + 2) / (dim + self._mu_eff + 5)
   359:         d_sigma = 1 + 2 * max(0, np.sqrt((self._mu_eff - 1) / (dim + 1)) - 1) + c_sigma
   360:         c_c = (4 + self._mu_eff / dim) / (dim + 4 + 2 * self._mu_eff / dim)
   361:         chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
   362: 
   363:         eigvals, eigvecs = np.linalg.eigh(self._C)
   364:         eigvals = np.maximum(eigvals, 1e-20)
   365:         inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
   366: 
   367:         self._p_sigma = (1 - c_sigma) * self._p_sigma + \
   368:             np.sqrt(c_sigma * (2 - c_sigma) * self._mu_eff) * \
   369:             inv_sqrt_C @ (self._mean - old_mean) / self._sigma
   370: 
   371:         h_sigma = 1.0 if (np.linalg.norm(self._p_sigma) /
   372:                           np.sqrt(1 - (1 - c_sigma) ** (2 * (self._gen + 1)))
   373:                           < (1.4 + 2 / (dim + 1)) * chi_n) else 0.0
   374: 
   375:         self._p_c = (1 - c_c) * self._p_c + \
   376:             h_sigma * np.sqrt(c_c * (2 - c_c) * self._mu_eff) * \
   377:             (self._mean - old_mean) / self._sigma
   378: 
   379:         # Covariance matrix update
   380:         c1 = 2.0 / ((dim + 1.3) ** 2 + self._mu_eff)
   381:         c_mu = min(1 - c1, 2 * (self._mu_eff - 2 + 1.0 / self._mu_eff) /
   382:                    ((dim + 2) ** 2 + self._mu_eff))
   383: 
   384:         rank_one = np.outer(self._p_c, self._p_c)
   385:         rank_mu = np.zeros((dim, dim))
   386:         for i in range(self._mu):
   387:             diff = (scored[i][1] - old_mean) / self._sigma
   388:             rank_mu += self._weights[i] * np.outer(diff, diff)
   389: 
   390:         self._C = (1 - c1 - c_mu) * self._C + c1 * rank_one + c_mu * rank_mu
   391:         # Ensure symmetry and positive definiteness
   392:         self._C = (self._C + self._C.T) / 2
   393:         eigvals_check = np.linalg.eigvalsh(self._C)
   394:         if np.min(eigvals_check) < 1e-20:
   395:             self._C += np.eye(dim) * (1e-20 - np.min(eigvals_check))
   396: 
   397:         # Step-size update
   398:         self._sigma *= np.exp(
   399:             (c_sigma / d_sigma) * (np.linalg.norm(self._p_sigma) / chi_n - 1))
   400:         self._sigma = np.clip(self._sigma, 1e-10, 1.0)
   401: 
   402:         self._gen += 1
   403: 
   404:     def suggest(
   405:         self,
   406:         space: SearchSpace,
   407:         history: List[Trial],
   408:         budget_left: int,
   409:     ) -> Tuple[Dict[str, Any], float]:
   410:         if not self._initialized:
   411:             self._init_cma(space.dim)
   412:             self._sample_population(space)
   413: 
   414:         # Update scores for pending candidates
   415:         if history:
   416:             last = history[-1]
   417:             last_vec = self._encode(last.config, space)
   418:             for i, (x, cfg, score) in enumerate(self._candidates):
   419:                 if score is None and np.allclose(x, last_vec, atol=0.01):
   420:                     self._candidates[i] = (x, cfg, last.score)
   421:                     break
   422: 
   423:         # If all candidates evaluated, do CMA update and resample
   424:         if self._candidates and all(s is not None for _, _, s in self._candidates):
   425:             self._update(space)
   426:             self._sample_population(space)
   427: 
   428:         # Return next pending evaluation
   429:         if self._pending_evals:
   430:             cfg = self._pending_evals.pop(0)
   431:             return cfg, 1.0
   432: 
   433:         # Fallback
   434:         return space.sample_uniform(self.rng), 1.0
   435: 
   436: 
   437: 
   438: # ================================================================
   439: # FIXED — HPO loop and evaluation (do not modify below)
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
