"""BANANAS-style predictor-guided NAS baseline — rigorous codebase edit ops.

Fits a small MLP ensemble on (path-encoded architecture, val-accuracy) pairs
seen so far, scores a large random candidate pool with the ensemble mean,
and queries the top-scoring unseen candidate. Since we do not ship torch in
the naslib image necessarily, we implement a tiny numpy MLP with Adam.

Reference: White, Neiswanger, Savani, 2021: "BANANAS: Bayesian Optimization
with Neural Architectures for Neural Architecture Search", AAAI.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "naslib/custom_nas_search.py"

_BANANAS = """\
class _TinyMLP:
    \"\"\"2-layer numpy MLP regressor trained with Adam + MSE.\"\"\"

    def __init__(self, in_dim, hidden=64, seed=0):
        rs = np.random.RandomState(seed)
        self.W1 = rs.randn(in_dim, hidden).astype(np.float32) * (1.0 / np.sqrt(in_dim))
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rs.randn(hidden, 1).astype(np.float32) * (1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    def forward(self, X):
        self._X = X
        self._z1 = X @ self.W1 + self.b1
        self._a1 = self._relu(self._z1)
        return (self._a1 @ self.W2 + self.b2).squeeze(-1)

    def fit(self, X, y, epochs=200, lr=1e-2):
        y = y.astype(np.float32).reshape(-1)
        m = {k: np.zeros_like(v) for k, v in self._params().items()}
        v = {k: np.zeros_like(p) for k, p in self._params().items()}
        b1_, b2_, eps, t = 0.9, 0.999, 1e-8, 0
        for _ in range(epochs):
            t += 1
            pred = self.forward(X)
            err = (pred - y) / max(1, len(X))
            dW2 = self._a1.T @ err.reshape(-1, 1)
            db2 = err.sum(keepdims=True)
            dA1 = err.reshape(-1, 1) @ self.W2.T
            dZ1 = dA1 * (self._z1 > 0)
            dW1 = X.T @ dZ1
            db1 = dZ1.sum(axis=0)
            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
            for k, g in grads.items():
                m[k] = b1_ * m[k] + (1 - b1_) * g
                v[k] = b2_ * v[k] + (1 - b2_) * (g * g)
                mhat = m[k] / (1 - b1_ ** t)
                vhat = v[k] / (1 - b2_ ** t)
                setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}


class NASOptimizer:
    \"\"\"BANANAS — predictor-guided sample-efficient NAS.

    Strategy:
    1. Warm start with N0=10 random architectures.
    2. Fit an ensemble of M=5 small MLPs on path-encoded (arch, val_acc) pairs.
    3. Each remaining step: draw a large random pool of candidates, score
       them with ensemble-mean predictions, pick the top unseen candidate,
       query its val accuracy, refit the ensemble.
    \"\"\"

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed

        self.warm_start = min(10, num_epochs)
        self.ensemble_size = 5
        self.candidate_pool = 500

        self.seen = {}           # arch_tuple -> val_acc
        self.best_arch = None
        self.best_val_acc = -1.0

    def _record(self, arch, val_acc):
        self.seen[tuple(arch)] = val_acc
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = list(arch)

    def _fit_ensemble(self):
        X = np.stack([path_encoding(list(a)) for a in self.seen])
        y = np.array([self.seen[a] for a in self.seen], dtype=np.float32)
        ensemble = []
        for i in range(self.ensemble_size):
            mlp = _TinyMLP(X.shape[1], hidden=64, seed=self.seed + i + 1)
            mlp.fit(X, y, epochs=200, lr=1e-2)
            ensemble.append(mlp)
        return ensemble

    def _propose_next(self):
        ensemble = self._fit_ensemble()
        # Large random candidate pool
        cands = []
        while len(cands) < self.candidate_pool:
            a = random_architecture()
            t = tuple(a)
            if t not in self.seen:
                cands.append(a)
        Xc = np.stack([path_encoding(a) for a in cands])
        preds = np.mean([m.forward(Xc) for m in ensemble], axis=0)
        idx = int(np.argmax(preds))
        return cands[idx]

    def search_step(self, epoch):
        if epoch < self.warm_start or len(self.seen) < 2:
            arch = random_architecture()
            while tuple(arch) in self.seen:
                arch = random_architecture()
        else:
            arch = self._propose_next()

        val_acc = self.api.query_val_accuracy(arch)
        self._record(arch, val_acc)

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        return self.best_arch
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 163,
        "end_line": 234,
        "content": _BANANAS,
    },
]
