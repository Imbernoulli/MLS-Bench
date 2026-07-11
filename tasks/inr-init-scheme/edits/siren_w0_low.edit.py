"""inr-init-scheme MIDDLE baseline: SIREN init but a LOW first-layer frequency (w0=5).

The principled SIREN init is applied, but a small w0 gives limited high-frequency
support => the sine net fits low/mid frequencies but under-represents fine detail =>
PSNR clearly above naive-init, clearly below the well-tuned w0=30 SIREN.
Reference: vendor/inr-signal-fitting/baselines/siren.py (w0=5)
"""

_FILE = "inr-signal-fitting/solution/init_scheme.py"

_CONTENT = '''def fit_inr(coords, target, dev):
    # SIREN init with a LOW frequency w0=5 — limited high-frequency support (middle ref).
    model = common.SirenMLP(in_dim=2, w0=5.0, w0_hidden=5.0)
    model = common.train_inr(model, coords, target, dev, label="siren_w0_5")

    @torch.no_grad()
    def predict(coords):
        model.eval()
        return model(coords.to(dev)).detach()

    return predict'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 44, "end_line": 61, "content": _CONTENT},
]
