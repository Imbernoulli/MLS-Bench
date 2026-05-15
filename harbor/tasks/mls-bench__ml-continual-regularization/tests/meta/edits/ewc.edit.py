"""EWC (Elastic Weight Consolidation) baseline.

Diagonal Fisher Information for importance, quadratic penalty for loss.
Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks" (PNAS 2017).

Replaces the editable region (lines 25-115) of custom_regularization.py.
"""

_FILE = "continual-learning/custom_regularization.py"

_EWC_IMPL = """\
def estimate_importance(model, dataset, prev_params, device):
    \"\"\"EWC: Diagonal Fisher Information matrix via squared gradients.\"\"\"
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
    \"\"\"EWC: 0.5 * sum(fisher * (param - prev_param)^2).\"\"\"
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
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 25,
        "end_line": 115,
        "content": _EWC_IMPL,
    },
]
