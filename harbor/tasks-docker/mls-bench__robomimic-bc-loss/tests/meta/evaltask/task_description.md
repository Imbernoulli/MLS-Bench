# Behavioral Cloning: Loss Function Design for Robot Imitation Learning

## Research Question
Design an improved loss function for GMM-based behavioral cloning (BC) in robot manipulation. The policy outputs a Gaussian Mixture Model (GMM) distribution over actions, and the loss function receives this distribution along with expert demonstration actions. Your goal is to design a loss that maximizes imitation learning quality as measured by rollout success rate.

## Background
The training and evaluation pipeline follows **robomimic** (Mandlekar et al., CoRL 2021, arXiv:2108.03298), the standard imitation-learning study and codebase for robot manipulation from offline human demonstrations. The default GMM-BC objective is the negative log-likelihood (NLL) of the expert action under the predicted mixture:

```
loss = -dist.log_prob(target_actions).mean()
```

NLL is convenient but ignores structure such as which mixture component is responsible for the target action, the shape of action errors (e.g. SE(3) end-effector vs. gripper bit), and the relative weight of low- vs. high-probability components. Alternative losses (e.g. cross-entropy on assignments, robust regression on the mean component, mixture-aware terms) may better exploit demonstration data.

## What You Can Modify
The `CustomBCLoss` class in `custom_bc_loss.py`. This class receives a GMM distribution and target action tensors and must return a scalar loss.

Interface:
- **Input**: `dist` (a `torch.distributions.MixtureSameFamily` GMM distribution with 5 modes) and `target_actions: [B, 7]` — 7-dim robot actions (6D end-effector delta + 1D gripper)
- **Output**: scalar loss tensor
- The default implementation is negative log-likelihood: `-dist.log_prob(target_actions).mean()`

You may add parameters to `__init__`, define helper methods, and use any PyTorch operations. The `dist` object supports `.log_prob()`, `.sample()`, `.component_distribution`, and `.mixture_distribution`.

## Evaluation
- **Metric**: `success_rate` — rollout success rate on the task (higher is better)
- **Tasks**: Lift (pick up cube), Can (pick-and-place can), Square (nut assembly)
- **Dataset**: 200 proficient human demonstrations per task, low-dimensional observations
- **Policy**: GMM with 5 modes, 2-layer MLP backbone (1024, 1024) with ReLU, tanh-squashed means
- **Training**: 2000 epochs, Adam optimizer (lr = 1e-4), batch size 100
- **Rollout evaluation**: 50 episodes per task, horizon 400 steps, every 50 epochs
