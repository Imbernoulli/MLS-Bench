"""Held-out scoring module for causal-treatment-effect (NOT agent-visible).

This module lives OUTSIDE every path that is bind-mounted into the agent's
container (workspace package dir, task dir, data binds). It is imported only by
the host-side ``parser.py`` (native) and by the Harbor verifier (tests/) — never
by the agent-editable ``custom_cate.py``. It holds the data-generating
processes (which also produce the true treatment effect ``tau``) and the
metrics, so the agent's process can never reach the answer.

The DGP bodies are byte-identical to the originals that used to live in
``edits/custom_template.py`` — so the data, and therefore every honest result,
is reproduced exactly.
"""

import numpy as np


# =====================================================================
# Data Generating Processes (synthetic benchmark families)
# =====================================================================

def generate_ihdp(n=747, p=25, seed=42):
    """Task-local synthetic IHDP-inspired DGP.

    Returns:
        X: (n, p) covariate matrix
        T: (n,) binary treatment indicator
        Y: (n,) observed outcomes
        tau: (n,) true individual treatment effects (CATE)
        ate: scalar true average treatment effect
    """
    rng = np.random.RandomState(seed)

    # Covariates: mix of continuous and binary
    X_cont = rng.randn(n, 6)
    X_bin = rng.binomial(1, 0.5, size=(n, p - 6)).astype(float)
    X = np.hstack([X_cont, X_bin])

    # Nonlinear propensity score with interactions (strong confounding)
    logit_e = (
        0.5 * X[:, 0]
        - 0.3 * X[:, 1]
        + 0.2 * X[:, 2]
        + 0.15 * X[:, 0] * X[:, 1]       # interaction
        - 0.2 * X[:, 2] ** 2              # quadratic
        + 0.25 * X[:, 3] * X[:, 5]        # confounders shared with tau
        + 0.1 * np.sin(X[:, 4] * np.pi)   # nonlinear
    )
    e = 1.0 / (1.0 + np.exp(-logit_e))
    e = np.clip(e, 0.05, 0.95)
    T = rng.binomial(1, e)

    # Response surfaces (complex nonlinear)
    mu0 = (
        np.exp(0.8 * X[:, 0] + 0.5 * X[:, 1])
        + X[:, 2] * X[:, 3]
        + 0.5 * X[:, 4]
        + 0.3 * X[:, 0] * X[:, 2]
        + 0.2 * np.cos(X[:, 5] * np.pi)
        + rng.randn(n) * 0.5
    )
    # Heterogeneous treatment effect with confounder-dependent terms
    tau = (
        1.0
        + 0.5 * X[:, 0]
        + 0.3 * X[:, 1] ** 2
        - 0.4 * X[:, 2] * X[:, 5]         # interaction with confounder
        + 0.5 * np.sin(X[:, 3] * np.pi)
        + 0.3 * np.maximum(X[:, 4], 0)     # ReLU-like
        - 0.2 * X[:, 0] * X[:, 3]          # cross-interaction
        + 0.15 * X[:, 6]                   # binary covariate effect
    )
    mu1 = mu0 + tau + rng.randn(n) * 0.5

    Y = T * mu1 + (1 - T) * mu0
    ate = tau.mean()

    return X, T, Y, tau, ate


def generate_jobs(n=2000, p=10, seed=42):
    """Task-local synthetic Jobs/LaLonde-inspired DGP."""
    rng = np.random.RandomState(seed)

    # Covariates simulating demographic features
    age = rng.uniform(18, 55, n)
    education = rng.uniform(8, 18, n)
    prior_earnings = np.maximum(0, rng.normal(10000, 5000, n))
    married = rng.binomial(1, 0.4, n).astype(float)
    black = rng.binomial(1, 0.3, n).astype(float)
    hispanic = rng.binomial(1, 0.15, n).astype(float)

    # Additional covariates
    extra = rng.randn(n, p - 6)
    X = np.column_stack([age, education, prior_earnings, married, black, hispanic, extra])

    # Normalize for stability
    X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # Nonlinear propensity with interactions (strong confounding)
    logit_e = (
        -0.3 * X_scaled[:, 0]
        - 0.2 * X_scaled[:, 1]
        - 0.4 * X_scaled[:, 2]
        + 0.1 * X_scaled[:, 3]
        + 0.25 * X_scaled[:, 0] * X_scaled[:, 1]   # age-education interaction
        - 0.15 * X_scaled[:, 2] ** 2                # quadratic earnings effect
        + 0.2 * X_scaled[:, 1] * X_scaled[:, 3]     # education-married interaction
    )
    e = 1.0 / (1.0 + np.exp(-logit_e))
    e = np.clip(e, 0.05, 0.95)
    T = rng.binomial(1, e)

    # Base outcome (earnings) with nonlinearities
    mu0 = (
        5000
        + 200 * X_scaled[:, 0]
        + 500 * X_scaled[:, 1]
        + 0.3 * prior_earnings
        + 1000 * married
        + 300 * X_scaled[:, 0] * X_scaled[:, 1]     # age-education interaction
        + 200 * np.maximum(X_scaled[:, 2], 0)        # ReLU on earnings
        + 150 * X_scaled[:, 4] * X_scaled[:, 5]      # race interaction
        + rng.randn(n) * 800
    )

    # Complex heterogeneous treatment effect
    tau = (
        1500
        + 300 * X_scaled[:, 1]                        # more education -> bigger effect
        - 200 * X_scaled[:, 0]                        # younger -> bigger effect
        + 250 * np.abs(X_scaled[:, 2])                # nonlinear prior earnings
        + 100 * X_scaled[:, 3]
        + 400 * np.sin(X_scaled[:, 0] * np.pi / 2)   # periodic age effect
        - 200 * X_scaled[:, 1] * X_scaled[:, 2]       # education-earnings interaction
        + 300 * np.maximum(X_scaled[:, 6], 0)          # ReLU on extra covariate
        + 150 * X_scaled[:, 0] * X_scaled[:, 3]       # age-married interaction
    )

    mu1 = mu0 + tau + rng.randn(n) * 500
    Y = T * mu1 + (1 - T) * mu0
    ate = tau.mean()

    return X, T, Y, tau, ate


def generate_acic(n=4000, p=50, seed=42):
    """Task-local synthetic ACIC-inspired DGP."""
    rng = np.random.RandomState(seed)

    # High-dimensional covariates with correlations
    mean = np.zeros(p)
    # Block-diagonal correlation structure
    cov = np.eye(p)
    for i in range(0, p - 1, 2):
        cov[i, i + 1] = 0.3
        cov[i + 1, i] = 0.3
    X = rng.multivariate_normal(mean, cov, n)

    # Complex propensity model (strong confounding)
    logit_e = (
        0.4 * X[:, 0]
        + 0.3 * X[:, 1]
        - 0.2 * X[:, 2]
        + 0.15 * X[:, 0] * X[:, 1]
        - 0.1 * X[:, 3] ** 2
        + 0.05 * np.sum(X[:, 4:10], axis=1)
    )
    e = 1.0 / (1.0 + np.exp(-logit_e))
    e = np.clip(e, 0.05, 0.95)  # Overlap enforcement
    T = rng.binomial(1, e)

    # Complex response surface (nonlinear, interactions)
    mu0 = (
        2.0 * np.sin(X[:, 0] * np.pi)
        + X[:, 1] ** 2
        + 0.5 * X[:, 2] * X[:, 3]
        - 1.5 * np.abs(X[:, 4])
        + 0.3 * np.sum(X[:, 5:15], axis=1)
        + rng.randn(n) * 0.5
    )

    # Complex heterogeneous treatment effect
    tau = (
        0.8
        + 0.6 * X[:, 0]
        - 0.3 * X[:, 1] ** 2
        + 0.4 * np.maximum(X[:, 2], 0)
        + 0.2 * X[:, 3] * X[:, 4]
        - 0.15 * np.abs(X[:, 5])
        + 0.1 * np.cos(X[:, 6] * np.pi)
    )

    mu1 = mu0 + tau + rng.randn(n) * 0.3
    Y = T * mu1 + (1 - T) * mu0
    ate = tau.mean()

    return X, T, Y, tau, ate


DATASETS = {
    "ihdp_synth": generate_ihdp,
    "jobs_synth": generate_jobs,
    "acic_synth": generate_acic,
}


# =====================================================================
# Metrics (same formulas as before)
# =====================================================================

def compute_pehe(tau_true, tau_hat):
    """Precision in Estimation of Heterogeneous Effects (lower is better)."""
    return np.sqrt(np.mean((tau_hat - tau_true) ** 2))


def compute_ate_error(ate_true, tau_hat):
    """Absolute error in ATE estimation (lower is better)."""
    return np.abs(np.mean(tau_hat) - ate_true)


# =====================================================================
# Helpers used by the input pre-generator and the host-side scorer
# =====================================================================

def gen_inputs(dataset, data_seed):
    """Return ONLY the agent-visible inputs (X, T, Y) for a given data_seed.

    ``tau``/``ate`` are computed internally by the DGP but deliberately NOT
    returned here, so the pre-generator that writes the agent's input files
    never persists the answer.
    """
    X, T, Y, _tau, _ate = DATASETS[dataset](seed=data_seed)
    return X, T, Y


def truth(dataset, data_seed):
    """Return the held-out ground truth (tau, ate) for the host-side scorer."""
    _X, _T, _Y, tau, ate = DATASETS[dataset](seed=data_seed)
    return tau, ate
