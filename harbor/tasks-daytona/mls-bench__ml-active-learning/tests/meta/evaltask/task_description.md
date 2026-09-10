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
