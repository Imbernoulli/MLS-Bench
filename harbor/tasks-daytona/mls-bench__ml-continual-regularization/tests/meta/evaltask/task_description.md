# Continual Learning: Regularization Strategy Optimization

## Research Question
Design a regularization strategy that mitigates catastrophic forgetting in continual learning. The contribution is the *importance estimator* (which parameters matter for each past context) and the *penalty form* (how their changes are penalized while training on later contexts), implemented within the fixed training loop.

## Background
A continual learner trains on a sequence of contexts and must retain performance on earlier ones. Regularization-based methods add `reg_strength * R(theta)` to the per-step loss, penalizing changes to parameters important for previous tasks.

Reference baselines:
- **EWC (Elastic Weight Consolidation)** — Kirkpatrick et al., PNAS 2017 ([arXiv:1612.00796](https://arxiv.org/abs/1612.00796)). Importance = diagonal Fisher Information `F` at the post-training parameter `theta*`. Penalty: `0.5 * sum_i F_i * (theta_i - theta_i*)^2`. Fishers are stored separately per past context (memory grows with context count).
- **SI (Synaptic Intelligence)** — Zenke, Poole, Ganguli, ICML 2017 ([arXiv:1703.04200](https://arxiv.org/abs/1703.04200)). Online importance: at each step accumulate `omega_i ≈ sum (-grad_i * delta_theta_i)`, normalized by total drift `(theta_i - theta_i*)^2 + epsilon` after the context. Penalty: `sum_i omega_i * (theta_i - theta_i*)^2`.
- **Online EWC** — Schwarz et al., ICML 2018 ([arXiv:1805.06370](https://arxiv.org/abs/1805.06370)). Replace the per-task list of Fishers with a single running estimate: `F <- gamma * F_old + F_new`, with `gamma < 1` (often 0.9). Constant memory in number of contexts.

## Implementation Contract
Implement two functions in `continual-learning/custom_regularization.py`:

```python
def estimate_importance(model, dataset, prev_params, device) -> dict:
    """
    Called once after training on each context finishes.
    Returns a dict {param_name: importance_tensor} (same shapes as the params).
    May do forward/backward passes over `dataset`.
    """

def compute_regularization_loss(model, importance_dict, prev_params_dict) -> Tensor:
    """
    Called at every training step. Must be efficient.
    Returns a scalar tensor — the regularization penalty added to the task loss.
    """
```

Available hooks on the framework:
- `model.param_list` — list of generators yielding `(name, param)` over regularized parameters.
- `model._custom_W` — dict tracking per-step gradient-weighted parameter changes (accumulated by the training loop). Useful for SI-style importance.
- `model._custom_p_old` — dict of parameter snapshots from the previous training step.
- `model.gamma` — decay factor for Fisher accumulation (framework default 1.0; Online EWC typically ≈ 0.9).
- `model.epsilon` — damping constant (default 0.1, used by SI).

Constraints: only modify the editable region of `custom_regularization.py`; do not create new files.

## Fixed Pipeline & Evaluation
Three benchmarks:

| Benchmark | Scenario | Contexts | Description |
|-----------|----------|----------|-------------|
| **Split-MNIST** | Task-incremental | 5 (2 classes each) | MNIST digits split into 5 binary tasks. |
| **Permuted-MNIST** | Domain-incremental | 10 | Same digit classes; each context applies a different fixed pixel permutation. |
| **Split-CIFAR100** | Task-incremental | 10 (10 classes each) | CIFAR-100 split into 10 ten-way tasks. |

Primary metric: **average accuracy across all contexts after training completes** (higher is better).
