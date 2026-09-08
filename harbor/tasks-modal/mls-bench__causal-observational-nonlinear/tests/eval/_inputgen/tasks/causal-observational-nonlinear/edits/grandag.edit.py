"""GraN-DAG-inspired baseline using gcastle-style defaults.

Reference: Lachapelle et al., "Gradient-Based Neural DAG Learning", ICLR 2020.

Self-contained implementation based on gcastle's GraNDAG class defaults:
NonLinGaussANM, 2 hidden layers of dim 10, leaky-relu,
RMSprop lr=1e-3, path-normalised weight-product adjacency, convergence-based
augmented Lagrangian, Jacobian-based DAG enforcement.
"""

_FILE = "causal-learn/bench/custom_algorithm.py"

_GRANDAG_FN = """\
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    \"\"\"GraN-DAG (Lachapelle et al., ICLR 2020).

    B[i,j] != 0 means j -> i (causal-learn convention).
    \"\"\"
    import os
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch import distributions

    seed = int(os.environ.get("SEED", "42"))
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n, d = X.shape
    DT = torch.float64

    # ================================================================== #
    # Per-variable MLP model (NonlinearGaussANM, 2x10, leaky-relu)       #
    # ================================================================== #
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            # Explicit adjacency mask (not a Parameter -- updated by clamping only)
            self.adjacency = torch.ones(d, d, dtype=DT) - torch.eye(d, dtype=DT)
            layers = [d, 10, 10, 1]  # [input, hidden1, hidden2, output_dim]
            self.wt = nn.ParameterList()
            self.bi = nn.ParameterList()
            for k in range(len(layers) - 1):
                self.wt.append(nn.Parameter(
                    torch.zeros(d, layers[k + 1], layers[k], dtype=DT)))
                self.bi.append(nn.Parameter(
                    torch.zeros(d, layers[k + 1], dtype=DT)))
            # Per-variable learnable noise log-std (ANM model)
            self.log_std = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, dtype=DT)) for _ in range(d)])
            # Xavier init matching gcastle's reset_params() order
            g = nn.init.calculate_gain('leaky_relu')
            with torch.no_grad():
                for nd in range(d):
                    for w in self.wt:
                        nn.init.xavier_uniform_(w[nd], gain=g)
                    for b in self.bi:
                        b[nd].zero_()

        def _fwd(self, x):
            \"\"\"Per-variable forward pass with adjacency masking on first layer.\"\"\"
            for k in range(3):
                if k == 0:
                    x = torch.einsum("tij,ljt,bj->bti", self.wt[k],
                                     self.adjacency.unsqueeze(0), x) \\
                        + self.bi[k]
                else:
                    x = torch.einsum("tij,btj->bti", self.wt[k], x) \\
                        + self.bi[k]
                if k < 2:
                    x = F.leaky_relu(x)
            return torch.unbind(x, 1)  # d tensors of (batch, 1)

        def log_lik(self, x, detach_target=False):
            \"\"\"(batch, d) per-variable Gaussian log-likelihoods.\"\"\"
            preds = self._fwd(x)
            parts = []
            for i in range(d):
                mu = preds[i].squeeze(1)
                sig = torch.exp(self.log_std[i])
                xi = x[:, i].detach() if detach_target else x[:, i]
                parts.append(
                    distributions.Normal(mu, sig).log_prob(xi).unsqueeze(1))
            return torch.cat(parts, 1)

        def w_adj(self):
            \"\"\"Weighted adjacency via product of |weights|, path-normalised.\"\"\"
            prod = torch.eye(d, dtype=DT)
            pn = torch.eye(d, dtype=DT)
            off = (1.0 - torch.eye(d, dtype=DT)).unsqueeze(0)
            for i, w in enumerate(self.wt):
                wa = torch.abs(w)
                if i == 0:
                    prod = torch.einsum("tij,ljt,jk->tik",
                                        wa, self.adjacency.unsqueeze(0), prod)
                    pn = torch.einsum("tij,ljt,jk->tik",
                                      torch.ones_like(wa), off, pn)
                else:
                    prod = torch.einsum("tij,tjk->tik", wa, prod)
                    pn = torch.einsum("tij,tjk->tik",
                                      torch.ones_like(wa), pn)
            prod = prod.sum(1)
            pn = pn.sum(1)
            return (prod / (pn + torch.eye(d, dtype=DT))).t()

    mdl = _M()

    # ================================================================== #
    # Data split (80/20, no shuffle, no normalise -- gcastle defaults)    #
    # ================================================================== #
    tn = int(n * 0.8)
    Xtr = torch.as_tensor(X[:tn], dtype=DT)
    Xte = torch.as_tensor(X[tn:], dtype=DT)
    rng_tr = np.random.RandomState(seed)
    rng_te = np.random.RandomState(seed + 1)

    def _samp(data, rng, bs):
        idx = rng.choice(data.shape[0], size=int(bs), replace=False)
        return data[torch.as_tensor(idx).long()]

    # ================================================================== #
    # Augmented-Lagrangian training (convergence-based mu/lambda update)  #
    # ================================================================== #
    mu, lamb = 0.001, 0.0        # penalty & dual variable
    opt = torch.optim.RMSprop(mdl.parameters(), lr=0.001)
    a_val, nns, hh = [], [], []  # validation AL, not-nll, constraint history
    BS, ITER, WIN = min(64, tn), 30000, 100

    for it in range(ITER):
        mdl.train()
        xb = _samp(Xtr, rng_tr, BS)
        loss = -mdl.log_lik(xb).mean()
        mdl.eval()

        wa = mdl.w_adj()
        h = torch.trace(torch.matrix_exp(wa)) - d
        al = loss + 0.5 * mu * h ** 2 + lamb * h

        opt.zero_grad()
        al.backward()
        opt.step()

        # Edge clamping -- only apply periodically to avoid premature
        # irreversible edge removal that causes instability across runs.
        # gcastle default threshold is 1e-4, but applying every step is
        # too aggressive; applying every 500 iterations with a stricter
        # weight threshold (1e-3) is more stable.
        if it > 0 and it % 500 == 0:
            with torch.no_grad():
                mdl.adjacency *= (wa > 1e-3).to(DT)

        nns.append(0.5 * mu * h.item() ** 2 + lamb * h.item())

        # Validation every WIN iterations
        if it % WIN == 0:
            with torch.no_grad():
                vl = -mdl.log_lik(
                    _samp(Xte, rng_te, Xte.shape[0])).mean()
                a_val.append([it, vl.item() + nns[-1]])

        # Convergence delta (checked every 2*WIN)
        dl = -np.inf
        if it >= 2 * WIN and it % (2 * WIN) == 0:
            t0, th, t1 = a_val[-3][1], a_val[-2][1], a_val[-1][1]
            if not (min(t0, t1) < th < max(t0, t1)):
                dl = -np.inf
            else:
                dl = (t1 - t0) / WIN

        # Lambda / mu update
        if h.item() > 1e-8:
            if abs(dl) < 1e-3 or dl > 0:
                lamb += mu * h.item()
                hh.append(h.item())
                if len(hh) >= 2 and hh[-1] > hh[-2] * 0.9:
                    mu *= 10
                # Adjust moving-average validation to account for new mu/lambda
                gap = (0.5 * mu * h.item() ** 2
                       + lamb * h.item() - nns[-1])
                a_val[-1][1] += gap
                opt = torch.optim.RMSprop(mdl.parameters(), lr=0.001)
        else:
            # Converged -- final clamping of zero-weight edges
            with torch.no_grad():
                mdl.adjacency *= (wa > 0).to(DT)
            break

    # ================================================================== #
    # DAG enforcement: Jacobian threshold + weakest-edge removal          #
    # ================================================================== #
    mdl.eval()
    xj = Xtr.clone().requires_grad_(True)
    ll = mdl.log_lik(xj, detach_target=True)       # (tn, d)
    lps = torch.unbind(ll, 1)                       # d tensors of (tn,)
    jac = torch.zeros(d, d, dtype=DT)
    for i in range(d):
        g = torch.autograd.grad(
            lps[i], xj, retain_graph=True,
            grad_outputs=torch.ones(Xtr.shape[0], dtype=DT))[0]
        jac[i] = g.abs().mean(0)
    A = jac.t().detach().numpy()

    # Find smallest threshold that produces an acyclic graph
    with torch.no_grad():
        for thr in np.unique(A):
            keep = torch.tensor(A > thr + 1e-8, dtype=DT)
            na = mdl.adjacency * keep
            # Acyclicity check via matrix-power trace
            prod = torch.eye(d, dtype=DT)
            ok = True
            for _ in range(d):
                prod = na @ prod
                if prod.trace() != 0:
                    ok = False
                    break
            if ok:
                mdl.adjacency = na
                break

    # adj[j,t]=1 means j->t; B[i,j]=1 means j->i  =>  B = adj.T
    return mdl.adjacency.t().detach().numpy()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 6,
        "end_line": 13,
        "content": _GRANDAG_FN,
    },
]
