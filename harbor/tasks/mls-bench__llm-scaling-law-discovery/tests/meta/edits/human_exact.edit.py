"""Literature-derived human baseline for llm-scaling-law-discovery.

Implements human-designed scaling laws from referenced literature for the three
harder SLDBench subsets:

- sld-vocab: Tao et al. "Scaling Laws with Vocabulary" — L(N, V, D) =
  E + A * N^-alpha + B * V^-beta + C * D^-gamma applied to the
  unigram-normalised loss (target can be negative, additive constants absorb
  the sign).
- sld-lrbsz: SLDBench Expert-B law (arXiv:2507.21184, App. A.4) — hierarchical
  additive form A/D^a + B/N^b + C + K_l*(l - l0)^2 + E*(log b + b0/b), with
  scale-dependent optima l0 = F*N^g*D^z (Step Law-style, Li et al. 2025) and
  b0 = G*D^h. Reference R^2 = -0.0756 on the held-out split.
- sld-dataconstrained: Muennighoff-style data-constrained language
  Models" — L(N, D, U) = E + A*N^-alpha + B * D_eff^-beta with
  D_eff = U * (1 - exp(-D/U)) as an effective token count.

Fitting uses nonlinear least squares with multi-start initialisations. For
benchmarks whose targets can be negative (sld-vocab), residuals are computed
in the linear (not log) domain.

Reference: SLDBench paper (arXiv:2507.21184); Tao et al. 2024; Muennighoff
et al. 2023. Formulas below are paraphrased human-law forms; specific
coefficient values are fit per group.
"""

_FILE = "scaling-law-lab/custom_scaling_law.py"

_CONTENT = """\
def _safe_log_residuals(pred, y):
    pred = np.clip(np.asarray(pred, dtype=float), EPS, None)
    y = np.clip(np.asarray(y, dtype=float), EPS, None)
    return np.log(pred) - np.log(y)


def _linear_residuals(pred, y):
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    return pred - y


def _fit_generic(X, y, init_u, unpack_fn, predict_fn, n_restarts=6,
                 use_log=True):
    init_u = np.asarray(init_u, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
    candidates = [init_u]
    for scale in np.linspace(0.05, 0.45, max(n_restarts - 1, 0)):
        candidates.append(init_u + rng.normal(scale=scale, size=init_u.shape))
    best_u, best_score = init_u, float("inf")
    for u0 in candidates:
        def residuals(u):
            try:
                pred = predict_fn(X, unpack_fn(u))
                resid = (_safe_log_residuals(pred, y) if use_log
                         else _linear_residuals(pred, y))
                return np.nan_to_num(resid, nan=1e3, posinf=1e3, neginf=-1e3)
            except Exception:
                return np.full_like(y, 1e3, dtype=float)
        try:
            result = least_squares(residuals, u0, method="trf",
                                   loss="soft_l1", f_scale=0.05, max_nfev=5000)
            u_opt = result.x
        except Exception:
            u_opt = np.asarray(u0, dtype=float)
        pred = predict_fn(X, unpack_fn(u_opt))
        if use_log:
            score = float(np.mean(_safe_log_residuals(pred, y) ** 2))
        else:
            score = float(np.mean((np.asarray(pred, dtype=float) - y) ** 2))
        if np.isfinite(score) and score < best_score:
            best_score, best_u = score, u_opt
    return unpack_fn(best_u)


# -------- sld-vocab: L = E + A*N^-alpha + B*V^-beta + C*D^-gamma --------

def _vocab_human_predict(X, params):
    n = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)
    v = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
    d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
    E, A, alpha, B, beta, C, gamma = params
    return (E
            + A * np.power(n, -alpha)
            + B * np.power(v, -beta)
            + C * np.power(d, -gamma))


def _fit_vocab_human(X, y):
    y = np.asarray(y, dtype=float)
    def unpack(u):
        # E unconstrained; scale / exponent parameters exponentiated.
        E = u[0]
        A = np.exp(u[1])
        alpha = np.exp(u[2])
        B = np.exp(u[3])
        beta = np.exp(u[4])
        C = np.exp(u[5])
        gamma = np.exp(u[6])
        return np.array([E, A, alpha, B, beta, C, gamma], dtype=float)
    init = np.array([
        float(np.median(y)),
        np.log(max(abs(np.std(y)), 0.1)), np.log(0.1),
        np.log(max(abs(np.std(y)), 0.1)), np.log(0.3),
        np.log(max(abs(np.std(y)), 0.1)), np.log(0.3),
    ])
    return _fit_generic(X, y, init, unpack, _vocab_human_predict,
                        n_restarts=8, use_log=False)


# -------- sld-lrbsz: Expert-B human law from SLDBench paper --------
# L(D, N, l, b) = A/D^alpha + B/N^beta + C + K_l*(l - l0)^2 + E*(log b + b0/b)
# with l0 = F * N^gamma * D^zeta, b0 = G * D^eta.
# Reference: arXiv:2507.21184v5 Appendix A.4 (Expert B law; R^2 = -0.0756).
# Code parameter K_l is named D_lr in the paper's reference implementation.

def _lrbsz_human_predict(X, params):
    lr = np.clip(np.asarray(X[:, 0], dtype=float), 1e-12, None)
    bsz = np.clip(np.asarray(X[:, 1], dtype=float), 1e-12, None)
    d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
    n = np.clip(np.asarray(X[:, 3], dtype=float), 1.0, None)
    A, alpha, B, beta, C, D_lr, E, F, gamma, zeta, G, eta = params
    l0 = F * np.power(n, gamma) * np.power(d, zeta)
    b0 = G * np.power(d, eta)
    term_data = A * np.power(d, -alpha)
    term_param = B * np.power(n, -beta)
    term_lr = D_lr * (lr - l0) ** 2
    term_bsz = E * (np.log(bsz) + b0 / bsz)
    return term_data + term_param + C + term_lr + term_bsz


def _fit_lrbsz_human(X, y):
    y = np.asarray(y, dtype=float)

    # Reference coefficients from the SLDBench paper (Expert B, "all_data"):
    #   [A, alpha, B, beta, C, D_lr, E, F, gamma, zeta, G, eta]
    paper_params = np.array([
        262.1391, 0.2675, 7.0285, 0.0746, 0.0000136, 1278.595,
        0.0493, 0.3242, -1.0580, 0.6498, 0.0302, 0.3503,
    ], dtype=float)

    # Parameterise so the fitter explores physically meaningful regions while
    # remaining well-conditioned. Positive quantities are exponentiated;
    # signed exponents (gamma, zeta, eta) are unconstrained.
    def unpack(u):
        A = np.exp(u[0]); alpha = np.exp(u[1])
        B = np.exp(u[2]); beta = np.exp(u[3])
        C = u[4]
        D_lr = np.exp(u[5])
        E = np.exp(u[6])
        F = np.exp(u[7]); gamma = u[8]; zeta = u[9]
        G = np.exp(u[10]); eta = u[11]
        return np.array([A, alpha, B, beta, C, D_lr, E, F, gamma, zeta, G, eta],
                        dtype=float)

    def pack(p):
        A, alpha, B, beta, C, D_lr, E, F, gamma, zeta, G, eta = p
        return np.array([
            np.log(max(A, 1e-12)), np.log(max(alpha, 1e-12)),
            np.log(max(B, 1e-12)), np.log(max(beta, 1e-12)),
            C,
            np.log(max(D_lr, 1e-12)),
            np.log(max(E, 1e-12)),
            np.log(max(F, 1e-12)), gamma, zeta,
            np.log(max(G, 1e-12)), eta,
        ], dtype=float)

    init_paper = pack(paper_params)

    # Also include a data-driven init so we degrade gracefully if the training
    # split shifts the optimum.
    y_span = max(float(y.max() - y.min()), 0.1)
    init_data = np.array([
        np.log(max(y_span, 0.1)), np.log(0.25),
        np.log(max(y_span, 0.1)), np.log(0.1),
        float(max(y.min(), 0.01)),
        np.log(1e3), np.log(0.05),
        np.log(0.3), -1.0, 0.65,
        np.log(0.03), 0.35,
    ], dtype=float)

    # Evaluate the reference coefficients directly (no fit) as an absolute
    # fallback — they already achieve the reported R^2 = -0.0756.
    best_params = paper_params
    best_score = float(np.mean((_lrbsz_human_predict(X, paper_params) - y) ** 2))
    if not np.isfinite(best_score):
        best_score = float("inf")

    for u0 in (init_paper, init_data):
        params = _fit_generic(X, y, u0, unpack, _lrbsz_human_predict,
                              n_restarts=3, use_log=False)
        pred = _lrbsz_human_predict(X, params)
        score = float(np.mean((pred - y) ** 2))
        if np.isfinite(score) and score < best_score:
            best_score, best_params = score, params
    return best_params


# -------- sld-dataconstrained: Muennighoff et al. with effective tokens --

def _dconstrained_human_predict(X, params):
    u = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)   # unique_tokens
    n = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)   # params
    d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)   # tokens
    E, A, alpha, B, beta = params
    # Effective tokens: U * (1 - exp(-D/U)) saturates when D >> U (repeated data).
    d_eff = u * (1.0 - np.exp(-np.clip(d / u, 0.0, 50.0)))
    d_eff = np.maximum(d_eff, 1.0)
    return E + A * np.power(n, -alpha) + B * np.power(d_eff, -beta)


def _fit_dconstrained_human(X, y):
    y = np.asarray(y, dtype=float)
    def unpack(u):
        E = np.exp(u[0])
        A = np.exp(u[1]); alpha = np.exp(u[2])
        B = np.exp(u[3]); beta = np.exp(u[4])
        return np.array([E, A, alpha, B, beta], dtype=float)
    init = np.array([
        np.log(max(y.min() * 0.9, 0.1)),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
    ])
    return _fit_generic(X, y, init, unpack, _dconstrained_human_predict,
                        n_restarts=8, use_log=True)


def _human_fit_params(benchmark_name, X, y):
    if benchmark_name == "sld-vocab":
        return _fit_vocab_human(X, y)
    if benchmark_name == "sld-lrbsz":
        return _fit_lrbsz_human(X, y)
    if benchmark_name == "sld-dataconstrained":
        return _fit_dconstrained_human(X, y)
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def _human_predict_params(benchmark_name, X, params):
    if benchmark_name == "sld-vocab":
        return _vocab_human_predict(X, params)
    if benchmark_name == "sld-lrbsz":
        return _lrbsz_human_predict(X, params)
    if benchmark_name == "sld-dataconstrained":
        return _dconstrained_human_predict(X, params)
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


class ScalingLawModel:
    \"\"\"Human law family from the literature for the harder SLDBench subsets.

    Benchmark-specific symbolic forms, fit per group via nonlinear least
    squares:
    - vocab: additive Chinchilla-style with per-axis power terms
    - lrbsz: SLDBench Expert-B hierarchical additive law (arXiv:2507.21184)
    - dataconstrained: Muennighoff-style effective-token saturation
    \"\"\"

    def __init__(self, benchmark_name, numeric_names=None, categorical_names=None):
        self.benchmark_name = benchmark_name
        self.numeric_names = list(numeric_names or [])
        self.categorical_names = list(categorical_names or [])
        self.group_params_ = {}
        self.default_params_ = None

    def fit(self, X_num, X_cat, y):
        X_num = np.asarray(X_num, dtype=float)
        y = np.asarray(y, dtype=float)
        labels = group_labels(X_cat)
        fitted = []
        for group in sorted(set(labels.tolist())):
            mask = labels == group
            params = _human_fit_params(self.benchmark_name, X_num[mask], y[mask])
            self.group_params_[group] = params
            fitted.append(params)
        self.default_params_ = np.median(np.stack(fitted, axis=0), axis=0)
        return self

    def predict(self, X_num, X_cat):
        X_num = np.asarray(X_num, dtype=float)
        labels = group_labels(X_cat)
        preds = np.zeros(len(labels), dtype=float)
        for group in sorted(set(labels.tolist())):
            mask = labels == group
            params = self.group_params_.get(group, self.default_params_)
            preds[mask] = _human_predict_params(self.benchmark_name,
                                                X_num[mask], params)
        # Do not clip to positive: vocab target (unigram_normalized_loss) can
        # be negative.
        return preds
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 183,
        "end_line": 211,
        "content": _CONTENT,
    },
]
