"""BADGE baseline for ml-active-learning.

Reference: vendor/external_packages/badge/query_strategies/badge_sampling.py
Paper: Ash et al. (2020), "Deep Batch Active Learning by Diverse, Uncertain
       Gradient Lower Bounds" (ICLR 2020)
Uses gradient embeddings with k-means++ initialization for diverse, uncertain
batch selection.
"""

_FILE = "badge/query_strategies/custom_sampling.py"

_CONTENT = """\
class CustomSampling(Strategy):
    \"\"\"BADGE — Batch Active learning by Diverse Gradient Embeddings.
    Selects batches that are diverse and uncertain in gradient embedding space
    via k-means++ seeding.\"\"\"

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        from scipy import stats
        from sklearn.metrics import pairwise_distances

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embs, probs = self.get_embedding(
            self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], return_probs=True
        )
        embs = embs.numpy()
        probs = probs.numpy()

        # BADGE: k-means++ in (gradient-embedding x probability-residual) space
        m = len(idxs_unlabeled)
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(probs, axis=-1)

        prob_residuals = -1.0 * probs
        prob_residuals[np.arange(m), max_inds] += 1.0
        prob_norms_square = np.sum(prob_residuals ** 2, axis=-1)

        # k-means++ initialization
        chosen = set()
        chosen_list = []
        mu = None
        D2 = None

        def _distance(X1, X2, center):
            Y1, Y2 = center
            X1_vec, X1_norm_sq = X1
            X2_vec, X2_norm_sq = X2
            Y1_vec, Y1_norm_sq = Y1
            Y2_vec, Y2_norm_sq = Y2
            dist = (X1_norm_sq * X2_norm_sq + Y1_norm_sq * Y2_norm_sq
                    - 2.0 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec))
            return np.sqrt(np.clip(dist, a_min=0, a_max=None))

        for _ in range(n):
            if len(chosen) == 0:
                ind = np.argmax(emb_norms_square * prob_norms_square)
                mu = [((prob_residuals[ind], prob_norms_square[ind]),
                        (embs[ind], emb_norms_square[ind]))]
                D2 = _distance(
                    (prob_residuals, prob_norms_square),
                    (embs, emb_norms_square),
                    mu[0],
                ).ravel().astype(float)
                D2[ind] = 0
                chosen.add(ind)
                chosen_list.append(ind)
            else:
                newD = _distance(
                    (prob_residuals, prob_norms_square),
                    (embs, emb_norms_square),
                    mu[-1],
                ).ravel().astype(float)
                D2 = np.minimum(D2, newD)
                D2[list(chosen)] = 0
                D2_sq = D2 ** 2
                total = D2_sq.sum()
                if total == 0:
                    # Fallback: random from remaining unlabeled
                    remaining = list(set(range(m)) - chosen)
                    ind = np.random.choice(remaining)
                else:
                    Ddist = D2_sq / total
                    customDist = stats.rv_discrete(
                        name="custm", values=(np.arange(len(Ddist)), Ddist)
                    )
                    ind = customDist.rvs(size=1)[0]
                    while ind in chosen:
                        ind = customDist.rvs(size=1)[0]
                mu.append(((prob_residuals[ind], prob_norms_square[ind]),
                           (embs[ind], emb_norms_square[ind])))
                chosen.add(ind)
                chosen_list.append(ind)

        return idxs_unlabeled[chosen_list]
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 28,
        "end_line": 54,
        "content": _CONTENT,
    },
]
