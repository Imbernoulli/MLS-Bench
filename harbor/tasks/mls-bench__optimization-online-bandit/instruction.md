# MLS-Bench: optimization-online-bandit

# Online Bandits: Exploration-Exploitation Strategy Design

## Objective
Design and implement a bandit policy that achieves low cumulative regret across diverse multi-armed bandit settings. Your code goes in `custom_bandit.py`.

## Background
The multi-armed bandit problem is a fundamental model for the exploration-exploitation tradeoff in sequential decision-making. At each round, an agent selects one of K arms and observes a stochastic reward. The goal is to minimize cumulative regret — the gap between the reward of the best arm (in hindsight) and the agent's actual reward.

Key challenges include adapting to different reward distributions, handling contextual information, and detecting non-stationarity.

## Task
Modify the `BanditPolicy` class in `custom_bandit.py` (the EDITABLE section). You must implement:
- `__init__(K, context_dim)`: initialize your policy for K arms with optional context.
- `select_arm(t, context)`: choose which arm to pull at timestep t.
- `update(arm, reward, context)`: update internal state after observing a reward.
- `reset()`: reset state for a new run.

## Interface
```python
class BanditPolicy:
    def __init__(self, K: int, context_dim: int = 0): ...
    def reset(self): ...
    def select_arm(self, t: int, context: np.ndarray | None = None) -> int: ...
    def update(self, arm: int, reward: float, context: np.ndarray | None = None): ...
```

Available utilities (in the FIXED section):
- `kl_bernoulli(p, q)`: KL divergence between Bernoulli distributions.
- `kl_ucb_bound(mu_hat, n, t, c)`: computes a KL-based upper confidence bound.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/SMPyBandits/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be rejected.

- `SMPyBandits/custom_bandit.py`
- editable lines **261–321**

## Readable Context

### `SMPyBandits/custom_bandit.py`  [EDITABLE — lines 261–321 only]

```python
   258: # =====================================================================
   259: # EDITABLE: BanditPolicy
   260: # =====================================================================
   261: class BanditPolicy:
   262:     """Bandit policy: the agent's exploration-exploitation strategy.
   263:
   264:     The evaluation loop calls:
   265:         policy = BanditPolicy(K, context_dim)
   266:         policy.reset()
   267:         for t in range(T):
   268:             context = env.get_context()          # None for context-free settings
   269:             arm = policy.select_arm(t, context)  # choose arm
   270:             reward, _ = env.pull(arm)
   271:             policy.update(arm, reward, context)  # observe reward
   272:
   273:     You MUST implement:
   274:         select_arm(t, context) -> int   : pick an arm in {0, ..., K-1}
   275:         update(arm, reward, context)    : update internal state
   276:         reset()                         : reset state for a new run
   277:
   278:     Available utilities (fixed, importable):
   279:         kl_bernoulli(p, q)              : KL divergence between Bernoulli(p) and Bernoulli(q)
   280:         kl_ucb_bound(mu_hat, n, t, c)   : KL-based upper confidence bound
   281:
   282:     Args:
   283:         K: number of arms
   284:         context_dim: dimension of context vector (0 if no context)
   285:     """
   286:
   287:     def __init__(self, K: int, context_dim: int = 0):
   288:         self.K = K
   289:         self.context_dim = context_dim
   290:         self.counts = np.zeros(K, dtype=np.float64)
   291:         self.rewards = np.zeros(K, dtype=np.float64)
   292:
   293:     def reset(self):
   294:         """Reset internal state for a new run."""
   295:         self.counts[:] = 0
   296:         self.rewards[:] = 0
   297:
   298:     def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
   299:         """Select which arm to pull at timestep t.
   300:
   301:         Args:
   302:             t: current timestep (0-indexed)
   303:             context: context vector of shape (context_dim,), or None
   304:
   305:         Returns:
   306:             arm index in {0, ..., K-1}
   307:         """
   308:         # Placeholder: uniform random — replace with your algorithm
   309:         return int(np.random.randint(self.K))
   310:
   311:     def update(self, arm: int, reward: float, context: np.ndarray | None = None):
   312:         """Update internal state after observing a reward.
   313:
   314:         Args:
   315:             arm: the arm that was pulled
   316:             reward: the observed reward
   317:             context: the context vector that was active, or None
   318:         """
   319:         self.counts[arm] += 1
   320:         self.rewards[arm] += reward
   321:
```

## Tips

- Keep the function/class signatures of the editable regions identical;
  the editable region is imported by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
