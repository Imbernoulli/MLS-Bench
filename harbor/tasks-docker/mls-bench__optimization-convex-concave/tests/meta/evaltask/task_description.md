# RAIN Convex-Concave

## Research Question
Can you improve gradient-norm convergence on the exact convex-concave benchmark instances used by the official RAIN repository for `src/bilinear_func/exp_gnorm.m` and `src/delta_func/exp_gnorm.m`?

## Background
Convex-concave saddle-point problems `min_x max_y F(x, y)` are a canonical model for minimax optimization (game-theoretic equilibria, robust learning, GANs). Even simple bilinear instances `f(x, y) = xy` make naive simultaneous gradient descent-ascent diverge; extragradient (Korpelevich, 1976), optimistic methods, and noise-robust analogues are needed. The RAIN reference codebase exercises two regimes — a scalar bilinear problem and a structured `(delta, nu)`-strongly-monotone problem — both subject to additive Gaussian update noise.

## What You Can Modify
Edit only the scaffold file `RAIN/optimization_convex_concave/custom_strategy.py` inside the editable block containing:

1. `init_state(problem, initial_z, seed, hyperparameters)`
2. `step(state, oracle, problem, hyperparameters, max_sfo_calls)`
3. `get_hyperparameters(problem_name, sigma)`

The benchmark harness, problem definitions, update-noise model, official iteration counts, initializations, and metric computation are fixed.

## Fixed Setup
- Problems:
  - `bilinear`: official scalar bilinear `f(x, y) = x y` with `n = 900`, `tau = 0.1`, `z0 = [10, 10]^T`, `sigma = 0.001`.
  - `delta_nu`: official `(delta, nu)` problem with `d = 100`, `delta = 1e-2`, `nu = 5e-5`, `n = 6000`, `tau = 1`, `sigma = 0.02`, `z0 ~ N(0, I)` under the script's fixed RNG seed.
- The harness mirrors the official scripts' additive Gaussian update noise, not the earlier generalized SFO sweep variant.
- Evaluation uses the official per-problem iteration counts and the same gradient-norm quantities plotted by the scripts.
- Main metric: `final_gradient_norm` (mean of the two official final gradient norms; lower is better).

## Interface Notes
- `init_state(...)` must preserve the provided starting point in `state["z"]`.
- `step(...)` should implement one official-style iteration of the chosen method.
- The oracle exposes deterministic gradients and fixed-scale Gaussian update noise so the update equations can match the MATLAB scripts directly.
- `get_hyperparameters(...)` should return the per-problem constants used by the method.

## Metrics
The harness prints (lower is better):
- `STEP_METRICS problem=... iteration=... gradient_norm=...`
- `RUN_METRICS problem=... final_gradient_norm=... auc_log_iteration_log_grad=...`
- `FINAL_METRICS final_gradient_norm=...`

## Baselines (reference implementations from the RAIN repo)
- **SEG** — Stochastic Extragradient (Korpelevich, 1976; modern stochastic analyses include Mishchenko et al., AISTATS 2020).
- **R-SEG** — restarted SEG variant from the RAIN repo.
- **SEAG** — Stochastic Extra-Anchored Gradient (RAIN-companion method).
- **RAIN** — the repo's main proposed update (anchor-iteration noise-robust method).

## Read-Only References
- `RAIN/README.md`
- `RAIN/src/bilinear_func/exp_gnorm.m`
- `RAIN/src/delta_func/exp_gnorm.m`

These are the primary references; the task follows them directly rather than the earlier MLS-Bench-specific generalized variant.
