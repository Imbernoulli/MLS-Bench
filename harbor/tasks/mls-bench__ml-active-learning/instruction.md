# MLS-Bench: ml-active-learning

# Active Learning: Query Strategy Design

## Research Question
Design a novel pool-based active learning query strategy for tabular classification. Strong strategies trade off uncertainty, diversity, representativeness, and information gain. The fixed harness handles model retraining and data management — the contribution is the *batch acquisition rule itself*, not preprocessing or training-loop changes.

## Background
In pool-based active learning, a query strategy repeatedly selects a batch of `n` examples from an unlabeled pool to be labeled by an oracle, then the model is retrained on the expanded labeled set. The goal is to reach the highest accuracy with the fewest labels.

Reference baselines:
- **Random Sampling** — uniform random batch from the pool.
- **Least-Confidence / Uncertainty Sampling** — select examples with the lowest top-class predicted probability ([Lewis & Gale 1994](https://arxiv.org/abs/cmp-lg/9407020) is the classic single-instance reference).
- **BALD (Bayesian Active Learning by Disagreement)** — Houlsby, Huszár, Ghahramani, Lengyel, 2011 ([arXiv:1112.5745](https://arxiv.org/abs/1112.5745)). Mutual information between predictions and parameters; estimated here via MC Dropout.
- **BADGE (Batch Active Learning by Diverse Gradient Embeddings)** — Ash, Zhang, Krishnamurthy, Langford, Agarwal, ICLR 2020 ([arXiv:1906.03671](https://arxiv.org/abs/1906.03671)). k-means++ seeding in the gradient-embedding space (gradient of loss w.r.t. last layer, using model's predicted label as pseudo-label) — picks batches that are simultaneously high-uncertainty (large gradient norm) and diverse.
- **BAIT (Gone Fishing: Neural Active Learning with Fisher Embeddings)** — Ash, Goel, Krishnamurthy, Kakade, NeurIPS 2021 ([arXiv:2106.09675](https://arxiv.org/abs/2106.09675)). Greedy/swap optimization of a Fisher-information bound on MLE error; selects the batch whose expected Fisher matrix best dominates the pool Fisher.

## Implementation Contract
Modify `CustomSampling` in `badge/query_strategies/custom_sampling.py`:

```python
class CustomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super().__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n) -> np.ndarray:
        # Return n indices into self.X of currently-unlabeled samples to label.
        ...
```

Available from the `Strategy` base class:
- `self.X`, `self.Y`, `self.idxs_lb` — pool features (numpy `[n_pool, n_features]`), labels (LongTensor `[n_pool]`), boolean labeled mask.
- `self.n_pool` — total pool size.
- `self.predict_prob(X, Y)` — softmax probabilities `[len(X), n_classes]`.
- `self.predict_prob_dropout_split(X, Y, n_drop)` — MC dropout probabilities `[n_drop, len(X), n_classes]`.
- `self.get_embedding(X, Y)` — penultimate-layer embeddings `[len(X), emb_dim]`.
- `self.get_grad_embedding(X, Y)` — last-layer gradient embeddings `[len(X), emb_dim * n_classes]`.
- `self.get_exp_grad_embedding(X, Y)` — expected (per-class) Fisher embeddings `[len(X), n_classes, emb_dim]`.

## Fixed Pipeline & Evaluation
- Datasets: 3 OpenML tabular classification datasets — **letter** recognition, **spambase**, **splice**.
- Protocol: 20 rounds of batch active learning; model retrained after each round.
- Metrics (higher is better):
  - `accuracy` — test accuracy after the final round (fixed total label budget).
  - `auc` — area under the learning curve (accuracy vs. # labeled samples) over all 20 rounds, capturing sample efficiency.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/badge/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `badge/query_strategies/custom_sampling.py`
- editable lines **28–54**


Other files you may **read** for context (do not modify):
- `badge/query_strategies/strategy.py`


## Readable Context


### `badge/query_strategies/custom_sampling.py`  [EDITABLE — lines 28–54 only]

```python
     1: """Custom active learning query strategy.
     2: 
     3: This module defines a CustomSampling strategy that inherits from the badge
     4: framework's Strategy base class. The agent must implement the query() method
     5: to select the most informative samples from the unlabeled pool.
     6: 
     7: Interface contract:
     8:   - self.X: numpy array of all pool features, shape (n_pool, n_features)
     9:   - self.Y: torch LongTensor of all pool labels, shape (n_pool,)
    10:   - self.idxs_lb: boolean array, True for labeled samples
    11:   - self.n_pool: total number of pool samples
    12:   - self.clf: the trained neural network model
    13:   - self.predict_prob(X, Y): returns softmax probabilities, shape (len(X), n_classes)
    14:   - self.predict_prob_dropout_split(X, Y, n_drop): returns MC dropout probs, shape (n_drop, len(X), n_classes)
    15:   - self.get_embedding(X, Y): returns penultimate-layer embeddings, shape (len(X), emb_dim)
    16:   - self.get_grad_embedding(X, Y): returns gradient embeddings (for BADGE), shape (len(X), emb_dim * n_classes)
    17:   - self.get_exp_grad_embedding(X, Y): returns expected Fisher embeddings (for BAIT), shape (len(X), n_classes, emb_dim)
    18:   - query(n) must return an array of n indices into self.X (indices of the UNLABELED pool)
    19: """
    20: 
    21: import numpy as np
    22: from query_strategies.strategy import Strategy
    23: 
    24: 
    25: # ================================================================
    26: # EDITABLE REGION — Implement your query strategy below (lines 28-55)
    27: # ================================================================
    28: class CustomSampling(Strategy):
    29:     """Custom active learning query strategy.
    30: 
    31:     Must implement query(n) -> np.ndarray of n indices from the unlabeled pool.
    32:     You may add helper methods, but query(n) is the entry point called by the
    33:     active learning loop.
    34:     """
    35: 
    36:     def __init__(self, X, Y, idxs_lb, net, handler, args):
    37:         super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
    38: 
    39:     def query(self, n):
    40:         """Select n samples from the unlabeled pool to label next.
    41: 
    42:         Args:
    43:             n: number of samples to select
    44: 
    45:         Returns:
    46:             np.ndarray of n indices (into self.X) of selected unlabeled samples
    47:         """
    48:         # Default: random sampling (replace with a better strategy)
    49:         idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    50:         return idxs_unlabeled[np.random.permutation(len(idxs_unlabeled))][:n]
    51: 
    52: # ================================================================
    53: # END EDITABLE REGION
    54: # ================================================================
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **letter** — wall-clock budget `10:00:00`, compute share `0.33`
- **spambase** — wall-clock budget `04:00:00`, compute share `0.33`
- **splice** — wall-clock budget `04:00:00`, compute share `0.33`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `badge` baseline — editable region  [READ-ONLY — reference implementation]

In `badge/query_strategies/custom_sampling.py`:

```python
Lines 28–112:
    25: # ================================================================
    26: # EDITABLE REGION — Implement your query strategy below (lines 28-55)
    27: # ================================================================
    28: class CustomSampling(Strategy):
    29:     """BADGE — Batch Active learning by Diverse Gradient Embeddings.
    30:     Selects batches that are diverse and uncertain in gradient embedding space
    31:     via k-means++ seeding."""
    32: 
    33:     def __init__(self, X, Y, idxs_lb, net, handler, args):
    34:         super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
    35: 
    36:     def query(self, n):
    37:         from scipy import stats
    38:         from sklearn.metrics import pairwise_distances
    39: 
    40:         idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    41:         embs, probs = self.get_embedding(
    42:             self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], return_probs=True
    43:         )
    44:         embs = embs.numpy()
    45:         probs = probs.numpy()
    46: 
    47:         # BADGE: k-means++ in (gradient-embedding x probability-residual) space
    48:         m = len(idxs_unlabeled)
    49:         emb_norms_square = np.sum(embs ** 2, axis=-1)
    50:         max_inds = np.argmax(probs, axis=-1)
    51: 
    52:         prob_residuals = -1.0 * probs
    53:         prob_residuals[np.arange(m), max_inds] += 1.0
    54:         prob_norms_square = np.sum(prob_residuals ** 2, axis=-1)
    55: 
    56:         # k-means++ initialization
    57:         chosen = set()
    58:         chosen_list = []
    59:         mu = None
    60:         D2 = None
    61: 
    62:         def _distance(X1, X2, center):
    63:             Y1, Y2 = center
    64:             X1_vec, X1_norm_sq = X1
    65:             X2_vec, X2_norm_sq = X2
    66:             Y1_vec, Y1_norm_sq = Y1
    67:             Y2_vec, Y2_norm_sq = Y2
    68:             dist = (X1_norm_sq * X2_norm_sq + Y1_norm_sq * Y2_norm_sq
    69:                     - 2.0 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec))
    70:             return np.sqrt(np.clip(dist, a_min=0, a_max=None))
    71: 
    72:         for _ in range(n):
    73:             if len(chosen) == 0:
    74:                 ind = np.argmax(emb_norms_square * prob_norms_square)
    75:                 mu = [((prob_residuals[ind], prob_norms_square[ind]),
    76:                         (embs[ind], emb_norms_square[ind]))]
    77:                 D2 = _distance(
    78:                     (prob_residuals, prob_norms_square),
    79:                     (embs, emb_norms_square),
    80:                     mu[0],
    81:                 ).ravel().astype(float)
    82:                 D2[ind] = 0
    83:                 chosen.add(ind)
    84:                 chosen_list.append(ind)
    85:             else:
    86:                 newD = _distance(
    87:                     (prob_residuals, prob_norms_square),
    88:                     (embs, emb_norms_square),
    89:                     mu[-1],
    90:                 ).ravel().astype(float)
    91:                 D2 = np.minimum(D2, newD)
    92:                 D2[list(chosen)] = 0
    93:                 D2_sq = D2 ** 2
    94:                 total = D2_sq.sum()
    95:                 if total == 0:
    96:                     # Fallback: random from remaining unlabeled
    97:                     remaining = list(set(range(m)) - chosen)
    98:                     ind = np.random.choice(remaining)
    99:                 else:
   100:                     Ddist = D2_sq / total
   101:                     customDist = stats.rv_discrete(
   102:                         name="custm", values=(np.arange(len(Ddist)), Ddist)
   103:                     )
   104:                     ind = customDist.rvs(size=1)[0]
   105:                     while ind in chosen:
   106:                         ind = customDist.rvs(size=1)[0]
   107:                 mu.append(((prob_residuals[ind], prob_norms_square[ind]),
   108:                            (embs[ind], emb_norms_square[ind])))
   109:                 chosen.add(ind)
   110:                 chosen_list.append(ind)
   111: 
   112:         return idxs_unlabeled[chosen_list]
```

### `bait` baseline — editable region  [READ-ONLY — reference implementation]

In `badge/query_strategies/custom_sampling.py`:

```python
Lines 28–283:
    25: # ================================================================
    26: # EDITABLE REGION — Implement your query strategy below (lines 28-55)
    27: # ================================================================
    28: class CustomSampling(Strategy):
    29:     """BAIT — Batch Active Learning via Information Matrices (Fisher embeddings).
    30:     CPU-adapted version of the original BAIT algorithm.
    31: 
    32:     This implementation keeps the Fisher-matrix objective, but makes the
    33:     selection pass tractable on MLS-Bench's CPU setup by:
    34:     1. building Fisher statistics in streaming batches,
    35:     2. projecting very high-dimensional Fisher embeddings before selection,
    36:     3. running BAIT on an entropy-filtered candidate pool instead of the full
    37:        unlabeled set.
    38:     """
    39: 
    40:     def __init__(self, X, Y, idxs_lb, net, handler, args):
    41:         super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
    42:         self.lamb = args.get('lamb', 1)
    43:         self.max_proj_dim = int(args.get('bait_proj_dim', 128))
    44:         self.candidate_pool = int(args.get('bait_candidate_pool', 0))
    45:         self.selection_batch_size = int(args.get('bait_selection_batch_size', 256))
    46:         self.seed = int(args.get('seed', 42))
    47: 
    48:     def _make_projection(self, full_dim):
    49:         import torch
    50: 
    51:         if full_dim <= self.max_proj_dim:
    52:             return None
    53: 
    54:         generator = torch.Generator(device='cpu')
    55:         generator.manual_seed(self.seed)
    56:         projection = torch.randn(
    57:             full_dim,
    58:             self.max_proj_dim,
    59:             generator=generator,
    60:             dtype=torch.float32,
    61:         )
    62:         projection /= np.sqrt(float(self.max_proj_dim))
    63:         return projection
    64: 
    65:     def _build_batch_embeddings(self, embedding, probs, projection):
    66:         import torch
    67: 
    68:         n_lab = probs.shape[1]
    69:         coeffs = -probs.unsqueeze(1).expand(-1, n_lab, -1).clone()
    70:         diag = torch.arange(n_lab)
    71:         coeffs[:, diag, diag] += 1.0
    72:         coeffs *= torch.sqrt(probs.clamp_min(1e-12)).unsqueeze(-1)
    73: 
    74:         fisher = coeffs.unsqueeze(-1) * embedding.unsqueeze(1).unsqueeze(2)
    75:         fisher = fisher.reshape(embedding.shape[0], n_lab, -1)
    76:         if projection is not None:
    77:             fisher = torch.matmul(fisher, projection)
    78:         return fisher.contiguous()
    79: 
    80:     def _candidate_pool_size(self, n, total):
    81:         default_size = max(4 * n, 512)
    82:         if self.candidate_pool > 0:
    83:             default_size = self.candidate_pool
    84:         return min(total, default_size)
    85: 
    86:     def _collect_statistics(self, idxs_unlabeled, n):
    87:         import torch
    88:         import torch.nn.functional as F
    89:         from torch.utils.data import DataLoader
    90: 
    91:         model = self.clf.eval()
    92:         device = next(model.parameters()).device
    93:         n_lab = int(torch.max(self.Y).item() + 1)
    94:         emb_dim = model.get_embedding_dim()
    95:         full_dim = emb_dim * n_lab
    96:         projection = self._make_projection(full_dim)
    97:         target_dim = full_dim if projection is None else projection.shape[1]
    98: 
    99:         fisher = torch.zeros(target_dim, target_dim, dtype=torch.float32)
   100:         init = torch.zeros(target_dim, target_dim, dtype=torch.float32)
   101:         n_labeled = max(int(np.sum(self.idxs_lb)), 1)
   102:         unlabeled_scores = np.empty(len(idxs_unlabeled), dtype=np.float32)
   103:         pool_to_unlabeled = np.full(self.n_pool, -1, dtype=np.int64)
   104:         pool_to_unlabeled[idxs_unlabeled] = np.arange(len(idxs_unlabeled))
   105: 
   106:         loader = DataLoader(
   107:             self.handler(self.X, self.Y, transform=self.args['transformTest']),
   108:             shuffle=False,
   109:             **self.args['loader_te_args']
   110:         )
   111: 
   112:         with torch.no_grad():
   113:             for x, _, idxs in loader:
   114:                 x = x.to(device)
   115:                 logits, embedding = model(x)
   116:                 probs = F.softmax(logits, dim=1).cpu()
   117:                 batch_xt = self._build_batch_embeddings(embedding.cpu(), probs, projection)
   118:                 fisher += torch.sum(
   119:                     torch.matmul(batch_xt.transpose(1, 2), batch_xt),
   120:                     dim=0,
   121:                 ) / float(self.n_pool)
   122: 
   123:                 idxs_np = idxs.numpy()
   124:                 labeled_mask = torch.from_numpy(self.idxs_lb[idxs_np])
   125:                 if labeled_mask.any():
   126:                     init += torch.sum(
   127:                         torch.matmul(
   128:                             batch_xt[labeled_mask].transpose(1, 2),
   129:                             batch_xt[labeled_mask],
   130:                         ),
   131:                         dim=0,
   132:                     ) / float(n_labeled)
   133: 
   134:                 unlabeled_mask = ~labeled_mask
   135:                 if unlabeled_mask.any():
   136:                     unlabeled_rows = pool_to_unlabeled[idxs_np[unlabeled_mask.numpy()]]
   137:                     batch_probs = probs[unlabeled_mask]
   138:                     entropy = -torch.sum(
   139:                         batch_probs * torch.log(batch_probs.clamp_min(1e-12)),
   140:                         dim=1,
   141:                     )
   142:                     unlabeled_scores[unlabeled_rows] = entropy.numpy()
   143: 
   144:         candidate_count = self._candidate_pool_size(n, len(idxs_unlabeled))
   145:         if candidate_count == len(idxs_unlabeled):
   146:             candidate_local = np.arange(len(idxs_unlabeled))
   147:         else:
   148:             candidate_local = np.argpartition(unlabeled_scores, -candidate_count)[-candidate_count:]
   149:         candidate_local = candidate_local[np.argsort(unlabeled_scores[candidate_local])[::-1]]
   150:         candidate_global = idxs_unlabeled[candidate_local]
   151:         pool_to_candidate = np.full(self.n_pool, -1, dtype=np.int64)
   152:         pool_to_candidate[candidate_global] = np.arange(len(candidate_global))
   153:         candidate_xt = torch.empty(
   154:             len(candidate_global),
   155:             n_lab,
   156:             target_dim,
   157:             dtype=torch.float32,
   158:         )
   159: 
   160:         with torch.no_grad():
   161:             for x, _, idxs in loader:
   162:                 idxs_np = idxs.numpy()
   163:                 candidate_rows = pool_to_candidate[idxs_np]
   164:                 keep_mask_np = candidate_rows >= 0
   165:                 if not keep_mask_np.any():
   166:                     continue
   167: 
   168:                 x = x.to(device)
   169:                 logits, embedding = model(x)
   170:                 probs = F.softmax(logits, dim=1).cpu()
   171:                 batch_xt = self._build_batch_embeddings(embedding.cpu(), probs, projection)
   172:                 keep_mask = torch.from_numpy(keep_mask_np)
   173:                 candidate_xt[torch.from_numpy(candidate_rows[keep_mask_np])] = batch_xt[keep_mask]
   174: 
   175:         return fisher, init, candidate_global, candidate_xt
   176: 
   177:     def _trace_scores(self, xt_batch, current_inv, fisher, add_identity):
   178:         import torch
   179: 
   180:         rank = xt_batch.shape[-2]
   181:         eye = torch.eye(rank, dtype=xt_batch.dtype).unsqueeze(0)
   182:         sign = 1.0 if add_identity else -1.0
   183:         info = current_inv @ fisher @ current_inv
   184:         gram = torch.matmul(torch.matmul(xt_batch, current_inv), xt_batch.transpose(1, 2))
   185:         inner = gram + sign * eye
   186:         inner_inv = torch.linalg.pinv(inner)
   187:         middle = torch.matmul(torch.matmul(xt_batch, info), xt_batch.transpose(1, 2))
   188:         scores = torch.diagonal(
   189:             torch.matmul(middle, inner_inv),
   190:             dim1=-2,
   191:             dim2=-1,
   192:         ).sum(-1)
   193:         finfo = torch.finfo(scores.dtype)
   194:         return torch.nan_to_num(scores, nan=-finfo.max, posinf=finfo.max, neginf=-finfo.max)
   195: 
   196:     def _woodbury_update(self, current_inv, xt_sample, add_identity):
   197:         import torch
   198: 
   199:         xt_sample = xt_sample.unsqueeze(0)
   200:         rank = xt_sample.shape[-2]
   201:         eye = torch.eye(rank, dtype=xt_sample.dtype).unsqueeze(0)
   202:         sign = 1.0 if add_identity else -1.0
   203: 
   204:         current = current_inv.unsqueeze(0)
   205:         inner = torch.matmul(torch.matmul(xt_sample, current), xt_sample.transpose(1, 2))
   206:         inner_inv = torch.linalg.pinv(inner + sign * eye)
   207:         updated = current - torch.matmul(
   208:             torch.matmul(torch.matmul(current, xt_sample.transpose(1, 2)), inner_inv),
   209:             torch.matmul(xt_sample, current),
   210:         )
   211:         return updated[0].contiguous()
   212: 
   213:     def _best_forward_index(self, xt_scaled, current_inv, fisher, selected_mask):
   214:         import torch
   215: 
   216:         best_idx = None
   217:         best_score = -float('inf')
   218:         for start in range(0, len(xt_scaled), self.selection_batch_size):
   219:             end = min(start + self.selection_batch_size, len(xt_scaled))
   220:             batch = xt_scaled[start:end]
   221:             scores = self._trace_scores(batch, current_inv, fisher, add_identity=True)
   222:             batch_mask = selected_mask[start:end]
   223:             if np.any(batch_mask):
   224:                 scores[torch.from_numpy(batch_mask)] = -torch.finfo(scores.dtype).max
   225:             score, local_idx = torch.max(scores, dim=0)
   226:             score = score.item()
   227:             if score > best_score:
   228:                 best_score = score
   229:                 best_idx = start + local_idx.item()
   230:         return best_idx
   231: 
   232:     def query(self, n):
   233:         import gc
   234:         import torch
   235: 
   236:         idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
   237:         fisher, init, candidate_global, xt_unlabeled = self._collect_statistics(
   238:             idxs_unlabeled,
   239:             n,
   240:         )
   241:         if len(candidate_global) <= n:
   242:             return candidate_global
   243: 
   244:         n_labeled = int(np.sum(self.idxs_lb))
   245:         K = n
   246:         denom = float(max(n_labeled + K, 1))
   247:         dim = xt_unlabeled.shape[-1]
   248:         currentInv = torch.linalg.pinv(
   249:             self.lamb * torch.eye(dim, dtype=torch.float32)
   250:             + init * n_labeled / denom
   251:         )
   252:         xt_scaled = xt_unlabeled * np.sqrt(K / denom)
   253: 
   254:         indsAll = []
   255:         selected_mask = np.zeros(len(candidate_global), dtype=bool)
   256:         over_sample = 2
   257: 
   258:         for _ in range(min(int(over_sample * K), len(candidate_global))):
   259:             ind = self._best_forward_index(xt_scaled, currentInv, fisher, selected_mask)
   260:             if ind is None:
   261:                 break
   262: 
   263:             indsAll.append(ind)
   264:             selected_mask[ind] = True
   265:             currentInv = self._woodbury_update(
   266:                 currentInv,
   267:                 xt_scaled[ind],
   268:                 add_identity=True,
   269:             )
   270: 
   271:         for _ in range(len(indsAll) - K):
   272:             xt_selected = xt_scaled[indsAll]
   273:             traceEst = self._trace_scores(xt_selected, currentInv, fisher, add_identity=False)
   274:             delInd = torch.argmax(traceEst).item()
   275:             currentInv = self._woodbury_update(
   276:                 currentInv,
   277:                 xt_scaled[indsAll[delInd]],
   278:                 add_identity=False,
   279:             )
   280:             del indsAll[delInd]
   281: 
   282:         gc.collect()
   283:         return candidate_global[np.asarray(indsAll, dtype=int)]
```

### `bald` baseline — editable region  [READ-ONLY — reference implementation]

In `badge/query_strategies/custom_sampling.py`:

```python
Lines 28–51:
    25: # ================================================================
    26: # EDITABLE REGION — Implement your query strategy below (lines 28-55)
    27: # ================================================================
    28: class CustomSampling(Strategy):
    29:     """BALD — Bayesian Active Learning by Disagreement (MC Dropout).
    30:     Selects samples where there is maximal disagreement across stochastic
    31:     forward passes, approximating mutual information."""
    32: 
    33:     def __init__(self, X, Y, idxs_lb, net, handler, args, n_drop=10):
    34:         super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
    35:         self.n_drop = n_drop
    36: 
    37:     def query(self, n):
    38:         import torch
    39:         idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    40:         probs = self.predict_prob_dropout_split(
    41:             self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop
    42:         )
    43:         # Mean prediction across MC samples
    44:         pb = probs.mean(0)
    45:         # Total entropy: H[y | x, D]
    46:         entropy1 = (-pb * torch.log(pb + 1e-10)).sum(1)
    47:         # Expected entropy: E_theta[H[y | x, theta]]
    48:         entropy2 = (-probs * torch.log(probs + 1e-10)).sum(2).mean(0)
    49:         # BALD score = total entropy - expected entropy = mutual information
    50:         U = entropy2 - entropy1
    51:         return idxs_unlabeled[U.sort()[1][:n]]
```

### `least_confidence` baseline — editable region  [READ-ONLY — reference implementation]

In `badge/query_strategies/custom_sampling.py`:

```python
Lines 28–39:
    25: # ================================================================
    26: # EDITABLE REGION — Implement your query strategy below (lines 28-55)
    27: # ================================================================
    28: class CustomSampling(Strategy):
    29:     """Least Confidence (Uncertainty Sampling) — selects samples with lowest
    30:     maximum predicted probability."""
    31: 
    32:     def __init__(self, X, Y, idxs_lb, net, handler, args):
    33:         super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
    34: 
    35:     def query(self, n):
    36:         idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    37:         probs = self.predict_prob(self.X[idxs_unlabeled], np.asarray(self.Y)[idxs_unlabeled])
    38:         U = probs.max(1)[0]
    39:         return idxs_unlabeled[U.sort()[1][:n]]
```

### `random` baseline — editable region  [READ-ONLY — reference implementation]

In `badge/query_strategies/custom_sampling.py`:

```python
Lines 28–36:
    25: # ================================================================
    26: # EDITABLE REGION — Implement your query strategy below (lines 28-55)
    27: # ================================================================
    28: class CustomSampling(Strategy):
    29:     """Random sampling baseline — selects samples uniformly at random."""
    30: 
    31:     def __init__(self, X, Y, idxs_lb, net, handler, args):
    32:         super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
    33: 
    34:     def query(self, n):
    35:         idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    36:         return idxs_unlabeled[np.random.permutation(len(idxs_unlabeled))][:n]
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
