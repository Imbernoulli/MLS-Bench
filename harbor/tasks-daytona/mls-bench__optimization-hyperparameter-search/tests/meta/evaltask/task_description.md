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
