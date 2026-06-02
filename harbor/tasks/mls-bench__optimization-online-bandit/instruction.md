# MLS-Bench: optimization-online-bandit

# Online Bandits: Exploration-Exploitation Strategy Design

## Objective
Design and implement a bandit policy that performs well across diverse multi-armed bandit settings. Your code goes in `custom_bandit.py`. Three reference implementations (UCB1, Thompson Sampling, KL-UCB) are available as read-only in the SMPyBandits package.

## Background
The multi-armed bandit problem is a fundamental model for the exploration-exploitation tradeoff in sequential decision-making. At each round, an agent selects one of K arms and observes a stochastic reward. The goal is to approach the performance of the best arm as quickly as possible.

Classic algorithms include:
- **UCB1** (Auer, Cesa-Bianchi, and Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem", *Machine Learning* 47, 2002): plays the arm with the highest upper confidence bound `mu_hat + sqrt(2 log(t) / n_a)`, achieving `O(sqrt(KT log T))` minimax regret.
- **Thompson Sampling** (Thompson, 1933; Agrawal and Goyal, "Analysis of Thompson Sampling for the Multi-armed Bandit Problem", COLT 2012): samples from a Bayesian posterior and plays the arm with the highest sample, achieving optimal Bayesian regret.
- **KL-UCB** (Garivier and Cappé, "The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond", COLT 2011; arXiv:1102.2490; Cappé, Garivier, Maillard, Munos, and Stoltz, *Annals of Statistics* 41, 2013): uses Kullback-Leibler divergence for tighter confidence bounds, provably optimal for Bernoulli bandits.

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
- `kl_ucb_bound(mu_hat, n, t, c)`: computes the KL-UCB upper confidence bound (Garivier and Cappé, 2011).

## Baselines (paper-cited reference implementations from SMPyBandits)
- **ucb1** — Auer, Cesa-Bianchi, and Fischer (*Machine Learning* 2002); paper-default exploration constant `c = 2` in the `sqrt(c log t / n_a)` term.
- **thompson_sampling** — Thompson (1933) / Agrawal and Goyal (COLT 2012); paper-default `Beta(1, 1)` prior per arm for Bernoulli rewards.
- **kl_ucb** — Garivier and Cappé (COLT 2011; arXiv:1102.2490); paper-default exploration function `f(t) = log(t) + 3 log log(t)` and binary KL inversion via bisection.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/SMPyBandits/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `SMPyBandits/custom_bandit.py`
- editable lines **261–321**




## Readable Context


### `SMPyBandits/custom_bandit.py`  [EDITABLE — lines 261–321 only]

```python
     1: # Custom online bandit algorithm for MLS-Bench
     2: #
     3: # EDITABLE section: BanditPolicy class — the exploration-exploitation strategy.
     4: # FIXED sections: everything else (environments, evaluation, main loop).
     5: #
     6: # Three evaluation settings:
     7: #   1. Stochastic MAB (K=10 Bernoulli arms, T=10000)
     8: #   2. Contextual Bandits (d=10 context, K=5 linear arms, T=10000)
     9: #   3. Non-stationary MAB (K=5 Bernoulli arms with 4 abrupt changes, T=10000)
    10: #
    11: # Metric: normalized cumulative regret at horizon T (lower is better).
    12: 
    13: import argparse
    14: import math
    15: import os
    16: import sys
    17: from abc import ABC, abstractmethod
    18: 
    19: import numpy as np
    20: from scipy.optimize import brentq
    21: from scipy.special import rel_entr
    22: 
    23: 
    24: # =====================================================================
    25: # FIXED: Arm Distributions
    26: # =====================================================================
    27: class Arm(ABC):
    28:     """Abstract arm with a mean reward and a draw method."""
    29: 
    30:     @abstractmethod
    31:     def mean(self) -> float:
    32:         ...
    33: 
    34:     @abstractmethod
    35:     def draw(self, rng: np.random.Generator) -> float:
    36:         ...
    37: 
    38: 
    39: class BernoulliArm(Arm):
    40:     """Bernoulli arm with parameter p in [0, 1]."""
    41: 
    42:     def __init__(self, p: float):
    43:         assert 0.0 <= p <= 1.0
    44:         self._p = p
    45: 
    46:     def mean(self) -> float:
    47:         return self._p
    48: 
    49:     def draw(self, rng: np.random.Generator) -> float:
    50:         return float(rng.random() < self._p)
    51: 
    52: 
    53: class GaussianArm(Arm):
    54:     """Gaussian arm with mean mu and std sigma (rewards clipped to [0, 1])."""
    55: 
    56:     def __init__(self, mu: float, sigma: float = 0.25):
    57:         self._mu = mu
    58:         self._sigma = sigma
    59: 
    60:     def mean(self) -> float:
    61:         return self._mu
    62: 
    63:     def draw(self, rng: np.random.Generator) -> float:
    64:         return float(np.clip(rng.normal(self._mu, self._sigma), 0.0, 1.0))
    65: 
    66: 
    67: # =====================================================================
    68: # FIXED: Bandit Environments
    69: # =====================================================================
    70: class StochasticMAB:
    71:     """Standard stochastic multi-armed bandit with K Bernoulli arms.
    72: 
    73:     Arms have fixed, unknown reward probabilities.  The optimal policy
    74:     always plays the arm with the highest mean.
    75:     """
    76: 
    77:     def __init__(self, arm_means: list[float], seed: int = 42):
    78:         self.arms = [BernoulliArm(p) for p in arm_means]
    79:         self.K = len(self.arms)
    80:         self.best_mean = max(a.mean() for a in self.arms)
    81:         self.rng = np.random.default_rng(seed)
    82:         self.context_dim = 0  # no context
    83: 
    84:     def reset(self):
    85:         """Called at the start of each episode (no-op for stationary)."""
    86:         pass
    87: 
    88:     def get_context(self) -> np.ndarray | None:
    89:         """Return context vector, or None for context-free bandits."""
    90:         return None
    91: 
    92:     def pull(self, arm: int) -> tuple[float, float]:
    93:         """Pull an arm.  Returns (reward, instantaneous_regret)."""
    94:         reward = self.arms[arm].draw(self.rng)
    95:         regret = self.best_mean - self.arms[arm].mean()
    96:         return reward, regret
    97: 
    98: 
    99: class ContextualBandit:
   100:     """Linear contextual bandit: reward = context^T theta_arm + noise.
   101: 
   102:     Each arm has a fixed (unknown) parameter vector theta_arm of dimension d.
   103:     At each round, a context x is drawn uniformly from the unit sphere.
   104:     Expected reward for arm a given context x is x^T theta_a.
   105:     Noise is Gaussian with std=0.1, rewards clipped to [0, 1].
   106:     """
   107: 
   108:     def __init__(self, K: int, d: int, seed: int = 42):
   109:         self.K = K
   110:         self.d = d
   111:         self.context_dim = d
   112:         self.rng = np.random.default_rng(seed)
   113:         # Generate arm parameters: theta_a are unit-norm vectors
   114:         raw = self.rng.standard_normal((K, d))
   115:         self.theta = raw / np.linalg.norm(raw, axis=1, keepdims=True) * 0.5
   116:         self._noise_std = 0.1
   117:         self._current_context: np.ndarray | None = None
   118: 
   119:     def reset(self):
   120:         self._current_context = None
   121: 
   122:     def get_context(self) -> np.ndarray:
   123:         """Draw a fresh context vector from the unit sphere."""
   124:         x = self.rng.standard_normal(self.d)
   125:         x = x / np.linalg.norm(x)
   126:         self._current_context = x
   127:         return x.copy()
   128: 
   129:     def pull(self, arm: int) -> tuple[float, float]:
   130:         """Pull an arm given the current context."""
   131:         assert self._current_context is not None, "Call get_context() first"
   132:         x = self._current_context
   133:         expected_rewards = self.theta @ x  # (K,)
   134:         best_reward = expected_rewards.max()
   135:         arm_reward = expected_rewards[arm]
   136:         noise = self.rng.normal(0, self._noise_std)
   137:         reward = float(np.clip(arm_reward + noise, 0.0, 1.0))
   138:         regret = best_reward - arm_reward
   139:         return reward, regret
   140: 
   141: 
   142: class NonStationaryMAB:
   143:     """Piece-wise stationary MAB with abrupt changepoints.
   144: 
   145:     The arm means change at pre-specified timesteps.  A good algorithm
   146:     must detect or adapt to these changes.
   147:     """
   148: 
   149:     def __init__(
   150:         self,
   151:         arm_configs: list[list[float]],
   152:         changepoints: list[int],
   153:         seed: int = 42,
   154:     ):
   155:         """
   156:         Args:
   157:             arm_configs: list of arm-mean vectors, one per segment.
   158:                          arm_configs[i] gives the K arm means for segment i.
   159:             changepoints: sorted list of timesteps where the means change.
   160:                           len(changepoints) == len(arm_configs) - 1.
   161:         """
   162:         assert len(arm_configs) == len(changepoints) + 1
   163:         self.arm_configs = [np.array(c) for c in arm_configs]
   164:         self.changepoints = changepoints
   165:         self.K = len(arm_configs[0])
   166:         self.context_dim = 0
   167:         self.rng = np.random.default_rng(seed)
   168:         self._t = 0
   169:         self._segment = 0
   170: 
   171:     def reset(self):
   172:         self._t = 0
   173:         self._segment = 0
   174: 
   175:     def get_context(self) -> np.ndarray | None:
   176:         return None
   177: 
   178:     def pull(self, arm: int) -> tuple[float, float]:
   179:         # Advance segment if needed
   180:         while (
   181:             self._segment < len(self.changepoints)
   182:             and self._t >= self.changepoints[self._segment]
   183:         ):
   184:             self._segment += 1
   185:         means = self.arm_configs[self._segment]
   186:         best_mean = means.max()
   187:         arm_mean = means[arm]
   188:         reward = float(self.rng.random() < arm_mean)  # Bernoulli
   189:         regret = best_mean - arm_mean
   190:         self._t += 1
   191:         return reward, regret
   192: 
   193: 
   194: # =====================================================================
   195: # FIXED: Environment Factory
   196: # =====================================================================
   197: def make_env(env_name: str, seed: int):
   198:     """Create a bandit environment by name.
   199: 
   200:     Returns (env, horizon) where horizon is the number of rounds.
   201:     """
   202:     if env_name == "stochastic_mab":
   203:         # 10-armed Bernoulli bandit
   204:         arm_means = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]
   205:         return StochasticMAB(arm_means, seed=seed), 10000
   206: 
   207:     elif env_name == "contextual":
   208:         # 5-armed linear contextual bandit, d=10
   209:         return ContextualBandit(K=5, d=10, seed=seed), 10000
   210: 
   211:     elif env_name == "nonstationary":
   212:         # 5-armed piece-wise stationary with 4 changepoints
   213:         configs = [
   214:             [0.9, 0.3, 0.2, 0.1, 0.5],   # segment 0: arm 0 best
   215:             [0.2, 0.8, 0.3, 0.1, 0.4],   # segment 1: arm 1 best
   216:             [0.1, 0.2, 0.7, 0.3, 0.5],   # segment 2: arm 2 best
   217:             [0.3, 0.1, 0.2, 0.85, 0.4],  # segment 3: arm 3 best
   218:             [0.2, 0.4, 0.3, 0.1, 0.9],   # segment 4: arm 4 best
   219:         ]
   220:         changepoints = [2000, 4000, 6000, 8000]
   221:         return NonStationaryMAB(configs, changepoints, seed=seed), 10000
   222: 
   223:     else:
   224:         raise ValueError(f"Unknown environment: {env_name}")
   225: 
   226: 
   227: # =====================================================================
   228: # FIXED: KL-divergence utilities (for reference — usable by the agent)
   229: # =====================================================================
   230: def kl_bernoulli(p: float, q: float) -> float:
   231:     """KL(Bernoulli(p) || Bernoulli(q)), with safe handling of edge cases."""
   232:     p = np.clip(p, 1e-10, 1 - 1e-10)
   233:     q = np.clip(q, 1e-10, 1 - 1e-10)
   234:     return float(rel_entr(p, q) + rel_entr(1 - p, 1 - q))
   235: 
   236: 
   237: def kl_ucb_bound(mu_hat: float, n: int, t: int, c: float = 1.0) -> float:
   238:     """Compute the KL-UCB upper confidence bound for a Bernoulli arm.
   239: 
   240:     Finds max q in [mu_hat, 1] such that n * KL(mu_hat, q) <= c * log(t).
   241:     """
   242:     if n == 0:
   243:         return 1.0
   244:     mu_hat = np.clip(mu_hat, 1e-10, 1 - 1e-10)
   245:     threshold = c * math.log(max(t, 1)) / n
   246: 
   247:     def f(q):
   248:         return kl_bernoulli(mu_hat, q) - threshold
   249: 
   250:     if f(1.0 - 1e-10) <= 0:
   251:         return 1.0
   252:     try:
   253:         return brentq(f, mu_hat, 1.0 - 1e-10, xtol=1e-6)
   254:     except ValueError:
   255:         return 1.0
   256: 
   257: 
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
   268:             context = env.get_context()          # None for MAB
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
   280:         kl_ucb_bound(mu_hat, n, t, c)   : KL-UCB upper confidence bound
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
   322: 
   323: # =====================================================================
   324: # FIXED: Evaluation Protocol
   325: # =====================================================================
   326: def run_bandit(env, policy, horizon: int) -> dict:
   327:     """Run a bandit algorithm for `horizon` steps.
   328: 
   329:     Returns dict with cumulative_regret, normalized_regret, and per-step info.
   330:     """
   331:     env.reset()
   332:     policy.reset()
   333: 
   334:     cumulative_regret = 0.0
   335:     regret_history = []
   336: 
   337:     for t in range(horizon):
   338:         context = env.get_context()
   339:         arm = policy.select_arm(t, context)
   340:         reward, regret = env.pull(arm)
   341:         policy.update(arm, reward, context)
   342:         cumulative_regret += regret
   343:         if (t + 1) % 1000 == 0:
   344:             regret_history.append((t + 1, cumulative_regret))
   345: 
   346:     # Normalized regret: cumulative_regret / horizon
   347:     normalized_regret = cumulative_regret / horizon
   348: 
   349:     return {
   350:         "cumulative_regret": cumulative_regret,
   351:         "normalized_regret": normalized_regret,
   352:         "regret_history": regret_history,
   353:     }
   354: 
   355: 
   356: def evaluate(env_name: str, seed: int, output_dir: str | None = None):
   357:     """Evaluate the BanditPolicy on a given environment."""
   358:     env, horizon = make_env(env_name, seed=seed)
   359:     policy = BanditPolicy(K=env.K, context_dim=env.context_dim)
   360:     result = run_bandit(env, policy, horizon)
   361: 
   362:     # Print training progress
   363:     for step, cum_reg in result["regret_history"]:
   364:         norm_reg = cum_reg / step
   365:         print(
   366:             f"TRAIN_METRICS step={step} cumulative_regret={cum_reg:.4f} "
   367:             f"normalized_regret={norm_reg:.6f}",
   368:             flush=True,
   369:         )
   370: 
   371:     # Print final test metrics
   372:     print(
   373:         f"TEST_METRICS cumulative_regret={result['cumulative_regret']:.4f} "
   374:         f"normalized_regret={result['normalized_regret']:.6f}",
   375:         flush=True,
   376:     )
   377: 
   378:     return result
   379: 
   380: 
   381: # =====================================================================
   382: # FIXED: Main
   383: # =====================================================================
   384: if __name__ == "__main__":
   385:     parser = argparse.ArgumentParser(description="Online Bandit Evaluation")
   386:     parser.add_argument("--env", type=str, required=True,
   387:                         choices=["stochastic_mab", "contextual", "nonstationary"],
   388:                         help="Bandit environment name")
   389:     parser.add_argument("--seed", type=int, default=42, help="Random seed")
   390:     parser.add_argument("--output-dir", type=str, default=None,
   391:                         help="Output directory (optional)")
   392:     args = parser.parse_args()
   393: 
   394:     np.random.seed(args.seed)
   395:     evaluate(args.env, seed=args.seed, output_dir=args.output_dir)
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `ucb1` baseline — editable region  [READ-ONLY — reference implementation]

In `SMPyBandits/custom_bandit.py`:

```python
Lines 261–335:
   258: # =====================================================================
   259: # EDITABLE: BanditPolicy
   260: # =====================================================================
   261: class BanditPolicy:
   262:     """UCB1: Upper Confidence Bound algorithm.
   263: 
   264:     Maintains empirical means and pull counts.  Selects the arm with the
   265:     highest upper confidence bound: mu_hat + sqrt(2 * log(t+1) / N_a).
   266: 
   267:     For non-stationary settings, uses a sliding window of size W with
   268:     an efficient circular buffer (O(1) per step).
   269:     """
   270: 
   271:     def __init__(self, K: int, context_dim: int = 0):
   272:         self.K = K
   273:         self.context_dim = context_dim
   274:         # Cumulative statistics
   275:         self.counts = np.zeros(K, dtype=np.float64)
   276:         self.rewards = np.zeros(K, dtype=np.float64)
   277:         # Sliding window via circular buffer for non-stationary settings
   278:         self._W = 800
   279:         self._buf_arms = np.zeros(self._W, dtype=np.int32)
   280:         self._buf_rewards = np.zeros(self._W, dtype=np.float64)
   281:         self._buf_ptr = 0
   282:         self._buf_full = False
   283:         self._sw_counts = np.zeros(K, dtype=np.float64)
   284:         self._sw_rewards = np.zeros(K, dtype=np.float64)
   285: 
   286:     def reset(self):
   287:         self.counts[:] = 0
   288:         self.rewards[:] = 0
   289:         self._buf_ptr = 0
   290:         self._buf_full = False
   291:         self._sw_counts[:] = 0
   292:         self._sw_rewards[:] = 0
   293: 
   294:     def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
   295:         # Initial round-robin: play each arm once
   296:         if t < self.K:
   297:             return t
   298: 
   299:         # Standard UCB1 index (full history). The SW-UCB fallback here was
   300:         # incorrect — vanilla UCB1 should use the full history regardless of
   301:         # environment; switching to sliding-window inflated regret on
   302:         # stationary MAB from ~960 (theoretical) to ~1450 observed.
   303: 
   304:         mu_hat = self.rewards / np.maximum(self.counts, 1e-10)
   305:         exploration = np.sqrt(2.0 * math.log(t + 1) / np.maximum(self.counts, 1))
   306:         ucb_values = mu_hat + exploration
   307:         return int(np.argmax(ucb_values))
   308: 
   309:     def _sw_select(self, t: int) -> int:
   310:         """Sliding-window UCB using pre-maintained running statistics."""
   311:         unpulled = self._sw_counts == 0
   312:         if unpulled.any():
   313:             return int(np.argmax(unpulled))
   314:         mu_hat = self._sw_rewards / self._sw_counts
   315:         xi = 1.5  # exploration parameter for SW-UCB
   316:         exploration = np.sqrt(xi * math.log(self._W) / self._sw_counts)
   317:         return int(np.argmax(mu_hat + exploration))
   318: 
   319:     def update(self, arm: int, reward: float, context: np.ndarray | None = None):
   320:         self.counts[arm] += 1
   321:         self.rewards[arm] += reward
   322:         # Update circular buffer and running window stats
   323:         if self._buf_full:
   324:             old_arm = int(self._buf_arms[self._buf_ptr])
   325:             old_rew = self._buf_rewards[self._buf_ptr]
   326:             self._sw_counts[old_arm] -= 1
   327:             self._sw_rewards[old_arm] -= old_rew
   328:         self._buf_arms[self._buf_ptr] = arm
   329:         self._buf_rewards[self._buf_ptr] = reward
   330:         self._sw_counts[arm] += 1
   331:         self._sw_rewards[arm] += reward
   332:         self._buf_ptr += 1
   333:         if self._buf_ptr >= self._W:
   334:             self._buf_ptr = 0
   335:             self._buf_full = True
   336: 
   337: # =====================================================================
   338: # FIXED: Evaluation Protocol
```

### `thompson_sampling` baseline — editable region  [READ-ONLY — reference implementation]

In `SMPyBandits/custom_bandit.py`:

```python
Lines 261–356:
   258: # =====================================================================
   259: # EDITABLE: BanditPolicy
   260: # =====================================================================
   261: class BanditPolicy:
   262:     """Thompson Sampling with Beta posterior for Bernoulli arms.
   263: 
   264:     For MAB: samples from Beta(alpha, beta) posterior per arm.
   265:     For contextual bandits: uses Bayesian linear regression (LinTS)
   266:     with Sherman-Morrison incremental inverse updates.
   267:     For non-stationary: uses discounted posterior (gamma < 1).
   268:     """
   269: 
   270:     def __init__(self, K: int, context_dim: int = 0):
   271:         self.K = K
   272:         self.context_dim = context_dim
   273:         self.rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
   274: 
   275:         # Beta posterior params for MAB (alpha=successes+1, beta=failures+1)
   276:         self.alpha = np.ones(K, dtype=np.float64)
   277:         self.beta_param = np.ones(K, dtype=np.float64)
   278: 
   279:         # Discount factor for non-stationary settings
   280:         self._gamma = 0.999
   281: 
   282:         # LinTS parameters for contextual bandits
   283:         if context_dim > 0:
   284:             self._lambda = 1.0  # regularization
   285:             self._v2 = 0.25  # sampling variance scale
   286:             # B_inv_a via Sherman-Morrison updates
   287:             self._B_inv = np.array([np.eye(context_dim) / self._lambda
   288:                                     for _ in range(K)])
   289:             self._f = np.zeros((K, context_dim), dtype=np.float64)
   290:             self._theta_hat = np.zeros((K, context_dim), dtype=np.float64)
   291: 
   292:         # Tracking
   293:         self.counts = np.zeros(K, dtype=np.float64)
   294:         self.rewards = np.zeros(K, dtype=np.float64)
   295: 
   296:     def reset(self):
   297:         self.alpha[:] = 1.0
   298:         self.beta_param[:] = 1.0
   299:         self.counts[:] = 0
   300:         self.rewards[:] = 0
   301:         if self.context_dim > 0:
   302:             d = self.context_dim
   303:             for a in range(self.K):
   304:                 self._B_inv[a] = np.eye(d) / self._lambda
   305:                 self._f[a] = np.zeros(d)
   306:                 self._theta_hat[a] = np.zeros(d)
   307: 
   308:     def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
   309:         if context is not None and self.context_dim > 0:
   310:             return self._lints_select(context)
   311: 
   312:         # Sample from Beta posterior for each arm
   313:         samples = self.rng.beta(self.alpha, self.beta_param)
   314:         return int(np.argmax(samples))
   315: 
   316:     def _lints_select(self, context: np.ndarray) -> int:
   317:         """Linear Thompson Sampling for contextual bandits."""
   318:         best_arm = 0
   319:         best_val = -np.inf
   320:         for a in range(self.K):
   321:             mu_a = self._theta_hat[a]
   322:             # Sample: theta ~ N(mu_a, v2 * B_inv_a)
   323:             # Use Cholesky of B_inv for efficient sampling
   324:             z = self.rng.standard_normal(self.context_dim)
   325:             try:
   326:                 L = np.linalg.cholesky(self._v2 * self._B_inv[a])
   327:                 theta_sample = mu_a + L @ z
   328:             except np.linalg.LinAlgError:
   329:                 theta_sample = mu_a + math.sqrt(self._v2) * z
   330:             val = context @ theta_sample
   331:             if val > best_val:
   332:                 best_val = val
   333:                 best_arm = a
   334:         return best_arm
   335: 
   336:     def update(self, arm: int, reward: float, context: np.ndarray | None = None):
   337:         self.counts[arm] += 1
   338:         self.rewards[arm] += reward
   339: 
   340:         if context is not None and self.context_dim > 0:
   341:             # Sherman-Morrison update: B_inv -= (B_inv x x^T B_inv)/(1 + x^T B_inv x)
   342:             Bx = self._B_inv[arm] @ context
   343:             denom = 1.0 + context @ Bx
   344:             self._B_inv[arm] -= np.outer(Bx, Bx) / denom
   345:             self._f[arm] += reward * context
   346:             self._theta_hat[arm] = self._B_inv[arm] @ self._f[arm]
   347:         else:
   348:             # Discounted Beta posterior update (for non-stationary robustness)
   349:             self.alpha *= self._gamma
   350:             self.beta_param *= self._gamma
   351:             # Clamp to prevent posterior from collapsing
   352:             self.alpha = np.maximum(self.alpha, 1.0)
   353:             self.beta_param = np.maximum(self.beta_param, 1.0)
   354:             # Update the pulled arm
   355:             self.alpha[arm] += reward
   356:             self.beta_param[arm] += (1.0 - reward)
   357: 
   358: # =====================================================================
   359: # FIXED: Evaluation Protocol
```

### `kl_ucb` baseline — editable region  [READ-ONLY — reference implementation]

In `SMPyBandits/custom_bandit.py`:

```python
Lines 261–320:
   258: # =====================================================================
   259: # EDITABLE: BanditPolicy
   260: # =====================================================================
   261: class BanditPolicy:
   262:     """KL-UCB: Kullback-Leibler Upper Confidence Bound.
   263: 
   264:     Vanilla KL-UCB per Garivier & Cappe 2011.  Index for arm a at time t is:
   265:         U_a(t) = sup { q in [0,1] : N_a(t) * kl(mu_hat_a, q) <= c*log(t) }
   266:     with c = 1 (theorem-tight constant) and kl the Bernoulli KL divergence.
   267: 
   268:     Implements the Bernoulli KL-UCB index formula used by SMPyBandits; no
   269:     sliding window (which harms the stationary stochastic regime).
   270:     """
   271: 
   272:     def __init__(self, K: int, context_dim: int = 0):
   273:         self.K = K
   274:         self.context_dim = context_dim
   275:         self.counts = np.zeros(K, dtype=np.float64)
   276:         self.rewards = np.zeros(K, dtype=np.float64)
   277: 
   278:     def reset(self):
   279:         self.counts[:] = 0
   280:         self.rewards[:] = 0
   281: 
   282:     @staticmethod
   283:     def _fast_kl_ucb(p: float, n: int, t: int) -> float:
   284:         """Fast KL-UCB bound via binary search (no scipy dependency)."""
   285:         if n == 0:
   286:             return 1.0
   287:         p = max(min(p, 1 - 1e-10), 1e-10)
   288:         threshold = math.log(max(t, 1)) / n
   289:         lo, hi = p, 1.0 - 1e-10
   290:         for _ in range(32):  # 32 iterations gives ~1e-10 precision
   291:             mid = (lo + hi) * 0.5
   292:             # KL(Bernoulli(p) || Bernoulli(mid))
   293:             kl = p * math.log(p / mid) + (1 - p) * math.log((1 - p) / (1 - mid))
   294:             if kl < threshold:
   295:                 lo = mid
   296:             else:
   297:                 hi = mid
   298:         return (lo + hi) * 0.5
   299: 
   300:     def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
   301:         # Initial round-robin: each arm once.
   302:         if t < self.K:
   303:             return t
   304: 
   305:         # Standard KL-UCB index for each arm (no sliding window).
   306:         best_arm = 0
   307:         best_idx = -1e100
   308:         for a in range(self.K):
   309:             if self.counts[a] == 0:
   310:                 return a
   311:             mu_hat = self.rewards[a] / self.counts[a]
   312:             idx = self._fast_kl_ucb(mu_hat, int(self.counts[a]), t + 1)
   313:             if idx > best_idx:
   314:                 best_idx = idx
   315:                 best_arm = a
   316:         return best_arm
   317: 
   318:     def update(self, arm: int, reward: float, context: np.ndarray | None = None):
   319:         self.counts[arm] += 1
   320:         self.rewards[arm] += reward
   321: 
   322: # =====================================================================
   323: # FIXED: Evaluation Protocol
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
