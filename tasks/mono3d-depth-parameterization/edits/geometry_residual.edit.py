"""mono3d-depth-parameterization STRONG (SOTA-style) baseline: geometry-from-height depth
with a small learned MULTIPLICATIVE residual (Deep3DBox / GUPNet-style height-guided depth).

Start from the projective depth Z0 = f * H / h2d (H = predicted metric height, h2d = 2D box
pixel height, f = focal) and multiply by a small learned correction exp(0.1 * r), r bounded,
predicted from the shared embedding. The residual absorbs the systematic amodal-box / height
estimation bias that pure geometry cannot, so it matches the geometry method's distance
robustness while lowering absolute depth error further -> highest AP3D across ALL distance
regimes. This is the strong reference (the trend Deep3DBox / GUPNet report: learned depth
refinement on top of the height-prior beats both naive regression and raw geometry).
Reference: vendor/mono3d-detection/baselines/geometry_residual.py
"""

_FILE = "mono3d-detection/solution/depth_param.py"

_CONTENT = '''def build_depth_head(emb_dim):
    # STRONG: geometry depth Z = f * H / h2d * exp(0.1 * residual) (Deep3DBox + learned refine).
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        H = ctx["pred_H"].reshape(-1)
        h2d = ctx["h2d"].reshape(-1).clamp(min=1.0)
        geom = ctx["focal"] * H / h2d
        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))

    return head, decode'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 48, "content": _CONTENT},
]
