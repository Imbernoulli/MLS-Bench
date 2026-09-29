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
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `SMPyBandits/custom_bandit.py`
- editable lines **63–123**


Other files you may **read** for context (do not modify):
- `SMPyBandits/mlsb_bandit_harness.py`


## Readable Context


### `SMPyBandits/custom_bandit.py`  [EDITABLE — lines 63–123 only]

```python
     1: # Custom online bandit algorithm for MLS-Bench
     2: #
     3: # EDITABLE section: BanditPolicy class — the exploration-exploitation strategy.
     4: # FIXED sections: everything else (utilities and the worker loop below).
     5: #
     6: # Three evaluation settings (see mlsb_bandit_harness.py, read-only):
     7: #   1. Stochastic MAB (K=10 Bernoulli arms, T=10000)
     8: #   2. Contextual Bandits (d=10 context, K=5 linear arms, T=10000)
     9: #   3. Non-stationary MAB (K=5 Bernoulli arms with 4 abrupt changes, T=10000)
    10: #
    11: # The environments live in the fixed harness, not in this file.  The harness
    12: # runs this file as a separate process (`custom_bandit.py --worker`) and
    13: # drives BanditPolicy over a pipe; every instance (arm means, parameters,
    14: # changepoints, noise, contexts) is drawn from fresh OS entropy in the harness
    15: # and is never visible to this process.  SEED seeds only np.random here.
    16: #
    17: # Metric: normalized cumulative regret at horizon T (lower is better).
    18: 
    19: import argparse
    20: import math
    21: import os
    22: import sys
    23: 
    24: import numpy as np
    25: from scipy.optimize import brentq
    26: from scipy.special import rel_entr
    27: 
    28: 
    29: # =====================================================================
    30: # FIXED: KL-divergence utilities (for reference — usable by the agent)
    31: # =====================================================================
    32: def kl_bernoulli(p: float, q: float) -> float:
    33:     """KL(Bernoulli(p) || Bernoulli(q)), with safe handling of edge cases."""
    34:     p = np.clip(p, 1e-10, 1 - 1e-10)
    35:     q = np.clip(q, 1e-10, 1 - 1e-10)
    36:     return float(rel_entr(p, q) + rel_entr(1 - p, 1 - q))
    37: 
    38: 
    39: def kl_ucb_bound(mu_hat: float, n: int, t: int, c: float = 1.0) -> float:
    40:     """Compute the KL-UCB upper confidence bound for a Bernoulli arm.
    41: 
    42:     Finds max q in [mu_hat, 1] such that n * KL(mu_hat, q) <= c * log(t).
    43:     """
    44:     if n == 0:
    45:         return 1.0
    46:     mu_hat = np.clip(mu_hat, 1e-10, 1 - 1e-10)
    47:     threshold = c * math.log(max(t, 1)) / n
    48: 
    49:     def f(q):
    50:         return kl_bernoulli(mu_hat, q) - threshold
    51: 
    52:     if f(1.0 - 1e-10) <= 0:
    53:         return 1.0
    54:     try:
    55:         return brentq(f, mu_hat, 1.0 - 1e-10, xtol=1e-6)
    56:     except ValueError:
    57:         return 1.0
    58: 
    59: 
    60: # =====================================================================
    61: # EDITABLE: BanditPolicy
    62: # =====================================================================
    63: class BanditPolicy:
    64:     """Bandit policy: the agent's exploration-exploitation strategy.
    65: 
    66:     The evaluation loop calls:
    67:         policy = BanditPolicy(K, context_dim)
    68:         policy.reset()
    69:         for t in range(T):
    70:             context = env.get_context()          # None for MAB
    71:             arm = policy.select_arm(t, context)  # choose arm
    72:             reward, _ = env.pull(arm)
    73:             policy.update(arm, reward, context)  # observe reward
    74: 
    75:     You MUST implement:
    76:         select_arm(t, context) -> int   : pick an arm in {0, ..., K-1}
    77:         update(arm, reward, context)    : update internal state
    78:         reset()                         : reset state for a new run
    79: 
    80:     Available utilities (fixed, importable):
    81:         kl_bernoulli(p, q)              : KL divergence between Bernoulli(p) and Bernoulli(q)
    82:         kl_ucb_bound(mu_hat, n, t, c)   : KL-UCB upper confidence bound
    83: 
    84:     Args:
    85:         K: number of arms
    86:         context_dim: dimension of context vector (0 if no context)
    87:     """
    88: 
    89:     def __init__(self, K: int, context_dim: int = 0):
    90:         self.K = K
    91:         self.context_dim = context_dim
    92:         self.counts = np.zeros(K, dtype=np.float64)
    93:         self.rewards = np.zeros(K, dtype=np.float64)
    94: 
    95:     def reset(self):
    96:         """Reset internal state for a new run."""
    97:         self.counts[:] = 0
    98:         self.rewards[:] = 0
    99: 
   100:     def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
   101:         """Select which arm to pull at timestep t.
   102: 
   103:         Args:
   104:             t: current timestep (0-indexed)
   105:             context: context vector of shape (context_dim,), or None
   106: 
   107:         Returns:
   108:             arm index in {0, ..., K-1}
   109:         """
   110:         # Placeholder: uniform random — replace with your algorithm
   111:         return int(np.random.randint(self.K))
   112: 
   113:     def update(self, arm: int, reward: float, context: np.ndarray | None = None):
   114:         """Update internal state after observing a reward.
   115: 
   116:         Args:
   117:             arm: the arm that was pulled
   118:             reward: the observed reward
   119:             context: the context vector that was active, or None
   120:         """
   121:         self.counts[arm] += 1
   122:         self.rewards[arm] += reward
   123: 
   124: 
   125: 
   126: # =====================================================================
   127: # FIXED: Worker loop (driven by mlsb_bandit_harness.py over a pipe)
   128: # =====================================================================
   129: def _worker_main():
   130:     """Serve BanditPolicy decisions to the harness.
   131: 
   132:     Protocol lines are written to the original stdout only; anything the
   133:     policy prints goes to stderr.
   134:     """
   135:     proto_in = sys.stdin
   136:     proto_out = os.fdopen(os.dup(1), "w", buffering=1)
   137:     os.dup2(2, 1)
   138:     sys.stdout = sys.stderr
   139: 
   140:     policy = None
   141:     prev_context = None
   142:     for line in proto_in:
   143:         parts = line.split()
   144:         if not parts:
   145:             continue
   146:         cmd = parts[0]
   147:         if cmd == "INIT":
   148:             K, context_dim, _horizon, seed = (int(v) for v in parts[1:5])
   149:             np.random.seed(seed)
   150:             policy = BanditPolicy(K=K, context_dim=context_dim)
   151:             policy.reset()
   152:             prev_context = None
   153:             proto_out.write("READY\n")
   154:         elif cmd == "S":
   155:             t, prev_arm = int(parts[1]), int(parts[2])
   156:             prev_reward = float(parts[3])
   157:             if prev_arm >= 0:
   158:                 policy.update(prev_arm, prev_reward, prev_context)
   159:             context = (np.array([float(v) for v in parts[4:]])
   160:                        if len(parts) > 4 else None)
   161:             arm = policy.select_arm(t, None if context is None
   162:                                     else context.copy())
   163:             prev_context = context
   164:             proto_out.write("%d\n" % int(arm))
   165:         elif cmd == "U":
   166:             policy.update(int(parts[1]), float(parts[2]), prev_context)
   167:         elif cmd == "BYE":
   168:             break
   169:     proto_out.close()
   170: 
   171: 
   172: # =====================================================================
   173: # FIXED: Main
   174: # =====================================================================
   175: if __name__ == "__main__":
   176:     parser = argparse.ArgumentParser(description="Online bandit policy worker")
   177:     parser.add_argument("--worker", action="store_true",
   178:                         help="serve the harness protocol on stdin/stdout")
   179:     args = parser.parse_args()
   180:     if not args.worker:
   181:         sys.exit("custom_bandit.py is driven by the harness; run e.g. "
   182:                  "scripts/stochastic_mab.sh (python -I "
   183:                  "SMPyBandits/mlsb_bandit_harness.py ...).")
   184:     _worker_main()
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
Lines 63–137:
    60: # =====================================================================
    61: # EDITABLE: BanditPolicy
    62: # =====================================================================
    63: class BanditPolicy:
    64:     """UCB1: Upper Confidence Bound algorithm.
    65: 
    66:     Maintains empirical means and pull counts.  Selects the arm with the
    67:     highest upper confidence bound: mu_hat + sqrt(2 * log(t+1) / N_a).
    68: 
    69:     For non-stationary settings, uses a sliding window of size W with
    70:     an efficient circular buffer (O(1) per step).
    71:     """
    72: 
    73:     def __init__(self, K: int, context_dim: int = 0):
    74:         self.K = K
    75:         self.context_dim = context_dim
    76:         # Cumulative statistics
    77:         self.counts = np.zeros(K, dtype=np.float64)
    78:         self.rewards = np.zeros(K, dtype=np.float64)
    79:         # Sliding window via circular buffer for non-stationary settings
    80:         self._W = 800
    81:         self._buf_arms = np.zeros(self._W, dtype=np.int32)
    82:         self._buf_rewards = np.zeros(self._W, dtype=np.float64)
    83:         self._buf_ptr = 0
    84:         self._buf_full = False
    85:         self._sw_counts = np.zeros(K, dtype=np.float64)
    86:         self._sw_rewards = np.zeros(K, dtype=np.float64)
    87: 
    88:     def reset(self):
    89:         self.counts[:] = 0
    90:         self.rewards[:] = 0
    91:         self._buf_ptr = 0
    92:         self._buf_full = False
    93:         self._sw_counts[:] = 0
    94:         self._sw_rewards[:] = 0
    95: 
    96:     def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
    97:         # Initial round-robin: play each arm once
    98:         if t < self.K:
    99:             return t
   100: 
   101:         # Standard UCB1 index (full history). The SW-UCB fallback here was
   102:         # incorrect — vanilla UCB1 should use the full history regardless of
   103:         # environment; switching to sliding-window inflated regret on
   104:         # stationary MAB from ~960 (theoretical) to ~1450 observed.
   105: 
   106:         mu_hat = self.rewards / np.maximum(self.counts, 1e-10)
   107:         exploration = np.sqrt(2.0 * math.log(t + 1) / np.maximum(self.counts, 1))
   108:         ucb_values = mu_hat + exploration
   109:         return int(np.argmax(ucb_values))
   110: 
   111:     def _sw_select(self, t: int) -> int:
   112:         """Sliding-window UCB using pre-maintained running statistics."""
   113:         unpulled = self._sw_counts == 0
   114:         if unpulled.any():
   115:             return int(np.argmax(unpulled))
   116:         mu_hat = self._sw_rewards / self._sw_counts
   117:         xi = 1.5  # exploration parameter for SW-UCB
   118:         exploration = np.sqrt(xi * math.log(self._W) / self._sw_counts)
   119:         return int(np.argmax(mu_hat + exploration))
   120: 
   121:     def update(self, arm: int, reward: float, context: np.ndarray | None = None):
   122:         self.counts[arm] += 1
   123:         self.rewards[arm] += reward
   124:         # Update circular buffer and running window stats
   125:         if self._buf_full:
   126:             old_arm = int(self._buf_arms[self._buf_ptr])
   127:             old_rew = self._buf_rewards[self._buf_ptr]
   128:             self._sw_counts[old_arm] -= 1
   129:             self._sw_rewards[old_arm] -= old_rew
   130:         self._buf_arms[self._buf_ptr] = arm
   131:         self._buf_rewards[self._buf_ptr] = reward
   132:         self._sw_counts[arm] += 1
   133:         self._sw_rewards[arm] += reward
   134:         self._buf_ptr += 1
   135:         if self._buf_ptr >= self._W:
   136:             self._buf_ptr = 0
   137:             self._buf_full = True
   138: 
   139: 
   140: # =====================================================================
```

### `thompson_sampling` baseline — editable region  [READ-ONLY — reference implementation]

In `SMPyBandits/custom_bandit.py`:

```python
Lines 63–158:
    60: # =====================================================================
    61: # EDITABLE: BanditPolicy
    62: # =====================================================================
    63: class BanditPolicy:
    64:     """Thompson Sampling with Beta posterior for Bernoulli arms.
    65: 
    66:     For MAB: samples from Beta(alpha, beta) posterior per arm.
    67:     For contextual bandits: uses Bayesian linear regression (LinTS)
    68:     with Sherman-Morrison incremental inverse updates.
    69:     For non-stationary: uses discounted posterior (gamma < 1).
    70:     """
    71: 
    72:     def __init__(self, K: int, context_dim: int = 0):
    73:         self.K = K
    74:         self.context_dim = context_dim
    75:         self.rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
    76: 
    77:         # Beta posterior params for MAB (alpha=successes+1, beta=failures+1)
    78:         self.alpha = np.ones(K, dtype=np.float64)
    79:         self.beta_param = np.ones(K, dtype=np.float64)
    80: 
    81:         # Discount factor for non-stationary settings
    82:         self._gamma = 0.999
    83: 
    84:         # LinTS parameters for contextual bandits
    85:         if context_dim > 0:
    86:             self._lambda = 1.0  # regularization
    87:             self._v2 = 0.25  # sampling variance scale
    88:             # B_inv_a via Sherman-Morrison updates
    89:             self._B_inv = np.array([np.eye(context_dim) / self._lambda
    90:                                     for _ in range(K)])
    91:             self._f = np.zeros((K, context_dim), dtype=np.float64)
    92:             self._theta_hat = np.zeros((K, context_dim), dtype=np.float64)
    93: 
    94:         # Tracking
    95:         self.counts = np.zeros(K, dtype=np.float64)
    96:         self.rewards = np.zeros(K, dtype=np.float64)
    97: 
    98:     def reset(self):
    99:         self.alpha[:] = 1.0
   100:         self.beta_param[:] = 1.0
   101:         self.counts[:] = 0
   102:         self.rewards[:] = 0
   103:         if self.context_dim > 0:
   104:             d = self.context_dim
   105:             for a in range(self.K):
   106:                 self._B_inv[a] = np.eye(d) / self._lambda
   107:                 self._f[a] = np.zeros(d)
   108:                 self._theta_hat[a] = np.zeros(d)
   109: 
   110:     def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
   111:         if context is not None and self.context_dim > 0:
   112:             return self._lints_select(context)
   113: 
   114:         # Sample from Beta posterior for each arm
   115:         samples = self.rng.beta(self.alpha, self.beta_param)
   116:         return int(np.argmax(samples))
   117: 
   118:     def _lints_select(self, context: np.ndarray) -> int:
   119:         """Linear Thompson Sampling for contextual bandits."""
   120:         best_arm = 0
   121:         best_val = -np.inf
   122:         for a in range(self.K):
   123:             mu_a = self._theta_hat[a]
   124:             # Sample: theta ~ N(mu_a, v2 * B_inv_a)
   125:             # Use Cholesky of B_inv for efficient sampling
   126:             z = self.rng.standard_normal(self.context_dim)
   127:             try:
   128:                 L = np.linalg.cholesky(self._v2 * self._B_inv[a])
   129:                 theta_sample = mu_a + L @ z
   130:             except np.linalg.LinAlgError:
   131:                 theta_sample = mu_a + math.sqrt(self._v2) * z
   132:             val = context @ theta_sample
   133:             if val > best_val:
   134:                 best_val = val
   135:                 best_arm = a
   136:         return best_arm
   137: 
   138:     def update(self, arm: int, reward: float, context: np.ndarray | None = None):
   139:         self.counts[arm] += 1
   140:         self.rewards[arm] += reward
   141: 
   142:         if context is not None and self.context_dim > 0:
   143:             # Sherman-Morrison update: B_inv -= (B_inv x x^T B_inv)/(1 + x^T B_inv x)
   144:             Bx = self._B_inv[arm] @ context
   145:             denom = 1.0 + context @ Bx
   146:             self._B_inv[arm] -= np.outer(Bx, Bx) / denom
   147:             self._f[arm] += reward * context
   148:             self._theta_hat[arm] = self._B_inv[arm] @ self._f[arm]
   149:         else:
   150:             # Discounted Beta posterior update (for non-stationary robustness)
   151:             self.alpha *= self._gamma
   152:             self.beta_param *= self._gamma
   153:             # Clamp to prevent posterior from collapsing
   154:             self.alpha = np.maximum(self.alpha, 1.0)
   155:             self.beta_param = np.maximum(self.beta_param, 1.0)
   156:             # Update the pulled arm
   157:             self.alpha[arm] += reward
   158:             self.beta_param[arm] += (1.0 - reward)
   159: 
   160: 
   161: # =====================================================================
```

### `kl_ucb` baseline — editable region  [READ-ONLY — reference implementation]

In `SMPyBandits/custom_bandit.py`:

```python
Lines 63–122:
    60: # =====================================================================
    61: # EDITABLE: BanditPolicy
    62: # =====================================================================
    63: class BanditPolicy:
    64:     """KL-UCB: Kullback-Leibler Upper Confidence Bound.
    65: 
    66:     Vanilla KL-UCB per Garivier & Cappe 2011.  Index for arm a at time t is:
    67:         U_a(t) = sup { q in [0,1] : N_a(t) * kl(mu_hat_a, q) <= c*log(t) }
    68:     with c = 1 (theorem-tight constant) and kl the Bernoulli KL divergence.
    69: 
    70:     Implements the Bernoulli KL-UCB index formula used by SMPyBandits; no
    71:     sliding window (which harms the stationary stochastic regime).
    72:     """
    73: 
    74:     def __init__(self, K: int, context_dim: int = 0):
    75:         self.K = K
    76:         self.context_dim = context_dim
    77:         self.counts = np.zeros(K, dtype=np.float64)
    78:         self.rewards = np.zeros(K, dtype=np.float64)
    79: 
    80:     def reset(self):
    81:         self.counts[:] = 0
    82:         self.rewards[:] = 0
    83: 
    84:     @staticmethod
    85:     def _fast_kl_ucb(p: float, n: int, t: int) -> float:
    86:         """Fast KL-UCB bound via binary search (no scipy dependency)."""
    87:         if n == 0:
    88:             return 1.0
    89:         p = max(min(p, 1 - 1e-10), 1e-10)
    90:         threshold = math.log(max(t, 1)) / n
    91:         lo, hi = p, 1.0 - 1e-10
    92:         for _ in range(32):  # 32 iterations gives ~1e-10 precision
    93:             mid = (lo + hi) * 0.5
    94:             # KL(Bernoulli(p) || Bernoulli(mid))
    95:             kl = p * math.log(p / mid) + (1 - p) * math.log((1 - p) / (1 - mid))
    96:             if kl < threshold:
    97:                 lo = mid
    98:             else:
    99:                 hi = mid
   100:         return (lo + hi) * 0.5
   101: 
   102:     def select_arm(self, t: int, context: np.ndarray | None = None) -> int:
   103:         # Initial round-robin: each arm once.
   104:         if t < self.K:
   105:             return t
   106: 
   107:         # Standard KL-UCB index for each arm (no sliding window).
   108:         best_arm = 0
   109:         best_idx = -1e100
   110:         for a in range(self.K):
   111:             if self.counts[a] == 0:
   112:                 return a
   113:             mu_hat = self.rewards[a] / self.counts[a]
   114:             idx = self._fast_kl_ucb(mu_hat, int(self.counts[a]), t + 1)
   115:             if idx > best_idx:
   116:                 best_idx = idx
   117:                 best_arm = a
   118:         return best_arm
   119: 
   120:     def update(self, arm: int, reward: float, context: np.ndarray | None = None):
   121:         self.counts[arm] += 1
   122:         self.rewards[arm] += reward
   123: 
   124: 
   125: # =====================================================================
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
