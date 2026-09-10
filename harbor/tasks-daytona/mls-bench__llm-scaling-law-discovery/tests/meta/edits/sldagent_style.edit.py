"""SLDAgent-style symbolic baseline for llm-scaling-law-discovery.

Implements SLDAgent-style discovered scaling-law forms (different from the
established human laws) for the three harder SLDBench subsets. These use
multiplicative cross-axis interactions / mixed log-linear forms to capture
residual variance that pure additive Chinchilla-style laws miss.

Reference: SLDAgent paper. Exact discovered formulas are not published; these
are reasonable stand-in symbolic forms in the same spirit as the paper's
human-vs-discovered comparison.
"""

_FILE = "scaling-law-lab/custom_scaling_law.py"

_CONTENT = """\
def _safe_log_residuals(pred, y):
    pred = np.clip(np.asarray(pred, dtype=float), EPS, None)
    y = np.clip(np.asarray(y, dtype=float), EPS, None)
    return np.log(pred) - np.log(y)


def _linear_residuals(pred, y):
    return np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)


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


# -------- sld-vocab: multiplicative interaction on log scales --------

def _vocab_sldagent_predict(X, params):
    n = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)
    v = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
    d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
    E, A, a1, a2, a3, A_vd, g1, g2 = params
    # Cross term links vocab and data.
    cross = A_vd * np.power(v, -g1) * np.power(d, -g2)
    return E + A * np.power(n, -a1) * np.power(v, -a2) * np.power(d, -a3) + cross


def _fit_vocab_sldagent(X, y):
    y = np.asarray(y, dtype=float)
    def unpack(u):
        E = u[0]
        A = np.exp(u[1])
        a1, a2, a3 = np.exp(u[2]), np.exp(u[3]), np.exp(u[4])
        A_vd = np.exp(u[5])
        g1, g2 = np.exp(u[6]), np.exp(u[7])
        return np.array([E, A, a1, a2, a3, A_vd, g1, g2], dtype=float)
    init = np.array([
        float(np.median(y)),
        np.log(max(abs(np.std(y)), 0.1)),
        np.log(0.1), np.log(0.2), np.log(0.2),
        np.log(max(abs(np.std(y)), 0.05)),
        np.log(0.3), np.log(0.3),
    ])
    return _fit_generic(X, y, init, unpack, _vocab_sldagent_predict,
                        n_restarts=8, use_log=False)


# -------- sld-lrbsz: Chinchilla base + joint (lr, bsz) coupling --------

def _lrbsz_sldagent_predict(X, params):
    lr = np.clip(np.asarray(X[:, 0], dtype=float), 1e-8, None)
    bsz = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
    d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
    n = np.clip(np.asarray(X[:, 3], dtype=float), 1.0, None)
    E, A, alpha, B, beta, k, log_lr_star, log_bsz_star, rho = params
    base = E + A * np.power(n, -alpha) + B * np.power(d, -beta)
    dx = np.log(lr) - log_lr_star
    dy = np.log(bsz) - log_bsz_star
    # Correlated quadratic bowl around (lr*, bsz*) with coupling rho.
    penalty = k * (dx * dx + dy * dy + 2.0 * rho * dx * dy)
    return base + penalty


def _fit_lrbsz_sldagent(X, y):
    y = np.asarray(y, dtype=float)
    lr = np.clip(np.asarray(X[:, 0], dtype=float), 1e-8, None)
    bsz = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
    def unpack(u):
        E = u[0]
        A = np.exp(u[1]); alpha = np.exp(u[2])
        B = np.exp(u[3]); beta = np.exp(u[4])
        k = np.exp(u[5])
        log_lr_star = u[6]; log_bsz_star = u[7]
        rho = np.tanh(u[8])  # keep in (-1, 1)
        return np.array([E, A, alpha, B, beta, k,
                         log_lr_star, log_bsz_star, rho], dtype=float)
    init = np.array([
        float(max(y.min() * 0.9, 0.1)),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.3),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.3),
        np.log(0.05),
        float(np.log(np.median(lr))), float(np.log(np.median(bsz))),
        0.0,
    ])
    return _fit_generic(X, y, init, unpack, _lrbsz_sldagent_predict,
                        n_restarts=10, use_log=True)


# -------- sld-dataconstrained: multiplicative repeat-efficiency term ---

def _dconstrained_sldagent_predict(X, params):
    u = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)
    n = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
    d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
    E, A, alpha, B, beta, R = params
    ratio = np.clip(d / u, 0.0, 200.0)
    # Repeat-efficiency: multiplier decays smoothly with repetition.
    efficiency = 1.0 / (1.0 + ratio / np.maximum(R, 1e-3))
    d_eff = np.maximum(d * efficiency, 1.0)
    return E + A * np.power(n, -alpha) + B * np.power(d_eff, -beta)


def _fit_dconstrained_sldagent(X, y):
    y = np.asarray(y, dtype=float)
    def unpack(u):
        E = np.exp(u[0])
        A = np.exp(u[1]); alpha = np.exp(u[2])
        B = np.exp(u[3]); beta = np.exp(u[4])
        R = np.exp(u[5])
        return np.array([E, A, alpha, B, beta, R], dtype=float)
    init = np.array([
        np.log(max(y.min() * 0.9, 0.1)),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
        np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
        np.log(5.0),
    ])
    return _fit_generic(X, y, init, unpack, _dconstrained_sldagent_predict,
                        n_restarts=8, use_log=True)


def _sldagent_fit_params(benchmark_name, X, y):
    if benchmark_name == "sld-vocab":
        return _fit_vocab_sldagent(X, y)
    if benchmark_name == "sld-lrbsz":
        return _fit_lrbsz_sldagent(X, y)
    if benchmark_name == "sld-dataconstrained":
        return _fit_dconstrained_sldagent(X, y)
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def _sldagent_predict_params(benchmark_name, X, params):
    if benchmark_name == "sld-vocab":
        return _vocab_sldagent_predict(X, params)
    if benchmark_name == "sld-lrbsz":
        return _lrbsz_sldagent_predict(X, params)
    if benchmark_name == "sld-dataconstrained":
        return _dconstrained_sldagent_predict(X, params)
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


class ScalingLawModel:
    \"\"\"SLDAgent-style symbolic baseline for the harder SLDBench subsets.

    Uses discovered-style symbolic forms with cross-axis interactions:
    - vocab: additive power law with extra V*D cross term
    - lrbsz: Chinchilla base + correlated (lr, bsz) quadratic bowl
    - dataconstrained: multiplicative repeat-efficiency factor on D_eff
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
            params = _sldagent_fit_params(self.benchmark_name,
                                          X_num[mask], y[mask])
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
            preds[mask] = _sldagent_predict_params(self.benchmark_name,
                                                   X_num[mask], params)
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
