"""Custom regularization module for continual learning.

This module provides two core functions that control how a continual learning
model prevents catastrophic forgetting via parameter regularization:

  1. estimate_importance() — called once after each context finishes training
  2. compute_regularization_loss() — called at every training step

The model object may have the following attributes set by the training loop:
  - model._custom_importance: dict mapping param_name -> accumulated importance tensor
  - model._custom_prev_params: dict mapping param_name -> param snapshot tensor
  - model._custom_W: dict for per-step accumulation (available during training)
  - model._custom_p_old: dict for per-step old params (available during training)

You may also attach new attributes to the model object as needed.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ======================================================================
# EDITABLE REGION START — estimate_importance
# ======================================================================
def estimate_importance(model, dataset, prev_params, device):
    """Estimate per-parameter importance after training on a context.

    Called once after each context's training completes. The returned
    importance dict is accumulated (summed) across contexts by the
    training loop and stored in model._custom_importance.

    Args:
        model: The neural network (nn.Module with a ``param_list`` attribute).
               Use model.param_list to get generators for named_parameters.
        dataset: Training dataset of the just-completed context.
        prev_params: Dict mapping param_name -> param tensor from before
                     training on this context started.
        device: torch.device to use.

    Returns:
        importance: Dict mapping param_name -> importance tensor (same shape
                    as the parameter). Higher values mean the parameter is
                    more important for the completed context.
    """
    # Default: diagonal Fisher Information (EWC-style)
    est_fisher = {}
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher[n] = p.detach().clone().zero_()

    mode = model.training
    model.eval()

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_samples = min(len(data_loader), 200)

    for idx, (x, y) in enumerate(data_loader):
        if idx >= n_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1)
        for label_index in range(output.shape[1]):
            label = torch.LongTensor([label_index]).to(device)
            negloglikelihood = F.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(
                retain_graph=True if (label_index + 1) < output.shape[1] else False
            )
            for gen_params in model.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

    est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}

    model.train(mode=mode)
    return est_fisher


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """Compute regularization loss to prevent catastrophic forgetting.

    Called at every training step during forward pass.

    Args:
        model: Current model (nn.Module with ``param_list``).
        importance_dict: Dict from estimate_importance, accumulated across
                         contexts (summed). Maps param_name -> importance tensor.
        prev_params_dict: Dict of parameter snapshots taken after the last
                          context finished. Maps param_name -> tensor.

    Returns:
        loss: Scalar regularization loss (torch scalar tensor).
    """
    # Default: EWC quadratic penalty
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    fisher = importance_dict[n]
                    prev = prev_params_dict[n]
                    losses.append((fisher * (p - prev) ** 2).sum())
    if losses:
        return 0.5 * sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).
CONFIG_OVERRIDES = {}
# ======================================================================
# EDITABLE REGION END
# ======================================================================
