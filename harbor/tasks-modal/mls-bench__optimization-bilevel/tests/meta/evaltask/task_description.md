# Optimization Bilevel

## Research Question
Can you design a single first-order update rule that makes a fixed bilevel-optimization benchmark — Shen and Chen's penalty-based bilevel gradient descent setting — converge faster on a numerical toy and recover more of the clean MNIST data in a hyper-cleaning task?

## Background
A bilevel problem couples an outer objective `f(x, y)` to an inner problem `min_y g(x, y)` whose solution depends on `x`. Penalty-based bilevel gradient descent (PBGD) replaces the inner argmin with a penalty term that constrains the lower-level value gap and then performs first-order updates jointly on `x` and `y`. Two PBGD variants are studied in the reference work:

- **V-PBGD** uses a value-function penalty `g(x, y) - g*(x)` and is the main method of Shen and Chen, "On Penalty-based Bilevel Gradient Descent Method" (ICML 2023; arXiv:2302.05185).
- **G-PBGD** penalizes the squared lower-level gradient norm; an iterative-differentiation baseline (RHG / T-RHG) competes via inner unrolling.

The reference repository `hanshen95/penalized-bilevel-gradient-descent` provides the Section 5/6 toy convergence experiment and the data hyper-cleaning experiment, where a fraction of MNIST training labels are corrupted and the outer problem learns per-example weights so that an inner classifier trained on the weighted set generalizes to clean validation data.

## What You Can Modify
Edit only `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py` inside the editable block. Define:

1. `algorithm(state, hparams, grad_fns)` — one shared update used by both toy and data hyper-cleaning runs. It receives the current state dict, a task hparam dict, and gradient-function callables, and returns the updated state after one outer (or method-equivalent) update.
2. `TOY_HPARAMS` — scalar knobs for toy convergence.
3. `HYPERCLEAN_HPARAMS` — scalar knobs for hyper-cleaning; may contain separate `linear` and `mlp` sub-dicts.

For toy mode, `grad_fns` provides `f`, `df`, `g`, `dg_dy`, `dg_dl`, `proj`, and `init_state`. `df` is the outer gradient, `dg_dy` and `dg_dl` are inner gradients with respect to the lower variable `y` and upper variable, and `proj` projects the upper variable onto the feasible set.

For hyper-cleaning mode, `grad_fns` provides `outer_grad`, `inner_grad`, `inner_val`, and `init_state`, exposing first-order information for the validation loss, weighted training loss, and initial state.

The fixed scaffold also exposes reference helpers `run_v_pbgd(...)`, `run_g_pbgd(...)`, and `run_rhg_family(...)`. You may call them, wrap them, or implement your own update logic on top of the provided state and gradients.

The driver, dataset split, pollution protocol, metrics, and model architectures are fixed.

## Fixed Setup
### Toy / Numerical Verification
- Problem definition follows Section 5.1 / 6.1 of Shen and Chen (2023).
- Upper variable `x` is projected to `[0, 3]`.
- 1000 random initial points are sampled (per the official toy script).
- Primary metric: `convergence_steps`. Secondary: `success_rate`, `final_residual`, `runtime_sec` (lower step counts and residuals are better).

### Data Hyper-Cleaning
- MNIST split: 5000 train / 5000 validation / 10000 test.
- Label-pollution rate: 50% (per the official code).
- Models: linear classifier and 2-layer MLP (`784 -> 300 -> 10`, sigmoid hidden layer).
- Primary metric: `test_accuracy` (higher is better). Secondary: `f1_score`, cleaner precision/recall, runtime to best accuracy.

## Reference Files (read-only)
- `penalized-bilevel-gradient-descent/V-PBGD/toy/toy.py`
- `penalized-bilevel-gradient-descent/V-PBGD/data-hyper-cleaning/data_hyper_clean.py`
- `penalized-bilevel-gradient-descent/G-PBGD/data_hyper_clean_gpbgd.py`
- `penalized-bilevel-gradient-descent/RHG/data_hyper_clean_rhg.py`
- `penalized-bilevel-gradient-descent/RHG/hypergrad/hypergradients.py`

## Baselines (cited reference implementations, all from `hanshen95/penalized-bilevel-gradient-descent`)
- **V-PBGD** — value-function PBGD (Shen and Chen, ICML 2023; arXiv:2302.05185).
- **G-PBGD** — gradient-norm PBGD variant from the same repo.
- **RHG** — Reverse-mode Hyper-Gradient via inner unrolling (Franceschi et al.; used as a baseline by Shen and Chen).
- **T-RHG** — Truncated RHG, the truncated-unroll baseline implemented in the same repo.

## Evaluation
Each command prints structured `TRAIN_METRICS` and `FINAL_METRICS` lines; the parser records final metrics per command label.
