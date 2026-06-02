# MLS-Bench: optimization-hyperparameter-search

# Hyperparameter Optimization: Custom Search Strategy Design

## Research Question
Design a novel hyperparameter optimization (HPO) strategy that achieves strong final validation performance and fast convergence within a limited evaluation budget.

## Background
Hyperparameter optimization is a fundamental problem in machine learning: given a model and dataset, find the hyperparameter configuration that maximizes validation performance within a limited evaluation budget. This is a black-box optimization problem where each function evaluation (training + validation) is expensive.

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

The fidelity parameter controls evaluation cost: lower fidelity means cheaper but noisier evaluation.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be rejected.

- `scikit-learn/custom_hpo.py`
- editable lines **255–326**

## Readable Context

### `scikit-learn/custom_hpo.py`  [EDITABLE — lines 255–326 only]

```python
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
```

## Tips

- Keep the function/class signatures of the editable regions identical;
  the editable region is imported by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
