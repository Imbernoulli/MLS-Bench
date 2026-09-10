# Implicit Q-Learning: Value Function Loss Design for Offline Robot Learning

## Research Question
Design an improved value function loss for Implicit Q-Learning (IQL) in offline robot manipulation. IQL avoids querying out-of-distribution actions by learning V(s) via asymmetric regression against Q(s, a) estimates. The loss function determines how V(s) approximates the upper quantile of Q-values, directly affecting policy quality.

## Background
**Implicit Q-Learning** (Kostrikov, Nair, Levine, ICLR 2022, arXiv:2110.06169) avoids the policy-evaluation step over out-of-distribution actions by fitting V(s) to an upper expectile of Q(s, a) drawn from the dataset. The default objective is the **expectile regression** loss:

```
diff = q_target - vf_pred
loss = mean( |quantile - 1{diff < 0}| * diff**2 )
```

When `quantile = 0.9`, overestimation (V > Q) is penalized 9× more than underestimation, so V(s) tracks the upper-tail of Q. The value function feeds the actor via advantage-weighted regression:

```
w(s, a) = exp((Q(s, a) - V(s)) / beta)
```

so V quality directly impacts policy learning. Alternative asymmetric losses (quantile regression, asymmetric Huber, log-cosh variants, etc.) may yield smoother gradients or better extrapolation.

The training pipeline uses **robomimic** (Mandlekar et al., CoRL 2021, arXiv:2108.03298).

## What You Can Modify
The `custom_vf_loss` function in `custom_iql_vf.py`. This function computes the loss for training the value network V(s).

Interface:
- **Input**:
  - `vf_pred: [B, 1]` — predicted state values V(s)
  - `q_target: [B, 1]` — target Q-values Q(s, a) from the critic ensemble (detached)
  - `quantile: float` — asymmetry parameter τ (default 0.9)
- **Output**: scalar loss tensor

You may restructure the function body, add helper computations, and use any PyTorch operations.

## Evaluation
- **Metric**: `success_rate` — rollout success rate on the task (higher is better)
- **Tasks**: Lift, Can, Square (robot manipulation with proficient human demonstrations)
- **Dataset**: ~200 demonstrations with (s, a, r, s', done) transitions
- **Training**: IQL with Q-ensemble (2 critics), GMM actor (5 modes), 2000 epochs × 100 steps
- **Hyperparameters**: discount = 0.99, target_tau = 0.01, adv_beta = 1.0, vf_quantile = 0.9
- **Rollout evaluation**: 50 episodes per task, horizon 400 steps, every 50 epochs
