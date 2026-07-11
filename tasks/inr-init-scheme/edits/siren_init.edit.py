"""inr-init-scheme SOTA baseline: SIREN principled init + tuned w0=30 (Sitzmann 2020).

The paper's initialization (first layer U(-1/in, 1/in); hidden U(-sqrt(6/in)/w0, +...))
plus a first-layer frequency w0=30 keep activations ~unit-variance through depth and
give the right frequency support => HIGHEST PSNR (the SIREN SOTA reference).
Reference: vendor/inr-signal-fitting/baselines/siren.py (w0=30)
"""

_FILE = "inr-signal-fitting/solution/init_scheme.py"

_CONTENT = '''def fit_inr(coords, target, dev):
    # SIREN principled init + tuned frequency w0=30 (Sitzmann et al., NeurIPS 2020) — SOTA.
    model = common.SirenMLP(in_dim=2, w0=30.0, w0_hidden=30.0)
    model = common.train_inr(model, coords, target, dev, label="siren_w0_30")

    @torch.no_grad()
    def predict(coords):
        model.eval()
        return model(coords.to(dev)).detach()

    return predict'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 44, "end_line": 61, "content": _CONTENT},
]
