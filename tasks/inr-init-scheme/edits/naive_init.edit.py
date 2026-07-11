"""inr-init-scheme WEAK baseline: sine MLP with NAIVE init + w0=1 (no SIREN init).

Standard small-uniform weights (not the principled SIREN init) plus a first-layer
frequency w0=1 give the sine net no high-frequency support and mis-scaled activations
=> it UNDER-fits => LOWEST PSNR (the documented bad-init / spectral-bottleneck failure).
Reference: vendor/inr-signal-fitting/solution/init_scheme.py (default surface).
"""

_FILE = "inr-signal-fitting/solution/init_scheme.py"

_CONTENT = '''def fit_inr(coords, target, dev):
    # NAIVE init + w0=1.0: no SIREN init, no frequency support (weak reference).
    model = common.SirenMLP(in_dim=2, w0=1.0, w0_hidden=1.0)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.uniform_(-0.05, 0.05)
                if m.bias is not None:
                    m.bias.zero_()
    model = common.train_inr(model, coords, target, dev, label="naive_init")

    @torch.no_grad()
    def predict(coords):
        model.eval()
        return model(coords.to(dev)).detach()

    return predict'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 44, "end_line": 61, "content": _CONTENT},
]
