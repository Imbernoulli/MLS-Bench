"""MAS (Memory Aware Synapses) baseline.

Estimates parameter importance from the sensitivity of the learned function's
output to parameter changes — uses gradient magnitude of the L2 norm of the
network output w.r.t. each parameter.  Unlike EWC, this is *unsupervised*:
it does not require labels, only forward passes on the data.

Reference: Aljundi et al., "Memory Aware Synapses: Learning what (not) to
forget" (ECCV 2018).

Replaces the editable region (lines 25-115) of custom_regularization.py.
"""

_FILE = "continual-learning/custom_regularization.py"

_MAS_IMPL = """\
def estimate_importance(model, dataset, prev_params, device):
    \"\"\"MAS: Estimate importance as the gradient magnitude of output L2 norm.

    For each parameter theta_k, importance is:
        Omega_k = (1/N) * sum_n || d(||F(x_n)||^2) / d(theta_k) ||

    where F(x) is the network output (pre-softmax logits).
    This is unsupervised — no labels needed.
    \"\"\"
    importance = {}
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                importance[n] = p.detach().clone().zero_()

    mode = model.training
    model.eval()

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_samples = min(len(data_loader), 200)

    for idx, (x, _) in enumerate(data_loader):
        if idx >= n_samples:
            break
        x = x.to(device)
        model.zero_grad()
        output = model(x)
        # L2 norm of output (squared) as the scalar objective
        loss = (output ** 2).sum()
        loss.backward()

        for gen_params in model.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        importance[n] += p.grad.detach().abs()

    # Normalize by number of samples
    importance = {n: v / max(n_samples, 1) for n, v in importance.items()}

    model.train(mode=mode)
    return importance


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    \"\"\"MAS: Quadratic penalty weighted by output sensitivity.

    L_reg = sum_k( Omega_k * (theta_k - theta_k^*)^2 )
    \"\"\"
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    omega = importance_dict[n]
                    prev = prev_params_dict[n]
                    losses.append((omega * (p - prev) ** 2).sum())
    if losses:
        return sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 25,
        "end_line": 115,
        "content": _MAS_IMPL,
    },
]
