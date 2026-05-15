"""Online EWC baseline.

Same as EWC but with exponential decay when accumulating Fisher across contexts:
  fisher_new = gamma * fisher_old + fisher_current

Reference: Schwarz et al., "Progress & Compress: A scalable framework for
continual learning" (ICML 2018).

Replaces the editable region (lines 25-115) of custom_regularization.py.
"""

_FILE = "continual-learning/custom_regularization.py"

_ONLINE_EWC_IMPL = """\
def estimate_importance(model, dataset, prev_params, device):
    \"\"\"Online EWC: Diagonal Fisher with exponential decay accumulation.

    When accumulating across contexts: fisher = gamma * fisher_old + fisher_new.
    Uses gamma=0.9 as the online Fisher decay for this benchmark.
    \"\"\"
    # Explicitly set gamma on the model to override framework default (1.0).
    # With gamma=1.0, Online EWC reduces to standard EWC.
    model.gamma = 0.9
    gamma = model.gamma
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

    # Apply decay to existing importance before adding new
    existing = getattr(model, '_custom_importance', {})
    for n in est_fisher:
        if n in existing:
            est_fisher[n] = gamma * existing[n] + est_fisher[n]

    # We return the full (decayed + new) Fisher, so the training loop
    # should replace (not add to) _custom_importance. To achieve this
    # with the accumulation logic in mid_edit, we subtract the existing
    # importance so that accumulation yields the correct result.
    result = {}
    for n in est_fisher:
        if n in existing:
            result[n] = est_fisher[n] - existing[n]
        else:
            result[n] = est_fisher[n]

    model.train(mode=mode)
    return result


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    \"\"\"Online EWC: 0.5 * gamma * sum(fisher * (param - prev_param)^2).\"\"\"
    gamma = getattr(model, 'gamma', 0.9)  # Already set to 0.9 in estimate_importance
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
        return 0.5 * gamma * sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 25,
        "end_line": 115,
        "content": _ONLINE_EWC_IMPL,
    },
]
