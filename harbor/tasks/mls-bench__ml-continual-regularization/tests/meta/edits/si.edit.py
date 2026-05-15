"""SI (Synaptic Intelligence) baseline.

Path integral-based importance estimation; quadratic penalty weighted by omega.
Reference: Zenke et al., "Continual Learning Through Synaptic Intelligence"
(ICML 2017).

The per-step W accumulation is tracked via model._custom_W (updated by the
training loop's per-step hook in mid_edit). This function computes the final
omega from the accumulated W and parameter changes.

Replaces the editable region (lines 25-115) of custom_regularization.py.
"""

_FILE = "continual-learning/custom_regularization.py"

_SI_IMPL = """\
def estimate_importance(model, dataset, prev_params, device):
    \"\"\"SI: Compute omega from accumulated path integral W and parameter changes.

    omega_k = W_k / (delta_k^2 + epsilon)

    where W_k is the accumulated per-step gradient-weighted parameter change
    (tracked in model._custom_W by the training loop) and delta_k is the
    total parameter change over the context.
    \"\"\"
    epsilon = getattr(model, 'epsilon', 0.1)
    omega = {}

    # Get the accumulated W from the per-step tracking
    W = getattr(model, '_custom_W', {})

    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                p_current = p.detach().clone()
                p_prev = prev_params.get(n, p_current)
                p_change = p_current - p_prev
                w_val = W.get(n, torch.zeros_like(p_current))
                omega[n] = w_val / (p_change ** 2 + epsilon)

    return omega


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    \"\"\"SI: sum(omega * (param - prev_param)^2).\"\"\"
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

_CONFIG_OVERRIDES = """\
# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).
CONFIG_OVERRIDES = {}
"""

OPS = [
    # CONFIG_OVERRIDES (bottom-to-top: higher line number first)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 117,
        "end_line": 119,
        "content": _CONFIG_OVERRIDES,
    },
    # Main edit (functions)
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 25,
        "end_line": 115,
        "content": _SI_IMPL,
    },
]
