"""mono3d-depth-parameterization MEDIUM baseline: geometry-from-height (analytic Deep3DBox
depth), NO learned residual.

Recover depth purely from the projective relation Z = f * H / h2d, where f is the focal
length, H the predicted metric object height (from the fixed dims head) and h2d the pixel
height of the amodal 2D box. This is the core monocular-3D cue (Deep3DBox / GS3D /
height-guided depth). Far stronger than direct regression and its advantage grows with
distance, but with no learned correction for the amodal-height / truncation bias it leaves
accuracy on the table vs the residual variant. Reference:
vendor/mono3d-detection/baselines/geometry_height.py
"""

_FILE = "mono3d-detection/solution/depth_param.py"

_CONTENT = '''def build_depth_head(emb_dim):
    # MEDIUM: analytic geometry depth Z = f * H / h2d (no learned residual).
    head = nn.Sequential(nn.Linear(emb_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def decode(raw, ctx):
        H = ctx["pred_H"].reshape(-1)
        h2d = ctx["h2d"].reshape(-1).clamp(min=1.0)
        return ctx["focal"] * H / h2d

    return head, decode'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 48, "content": _CONTENT},
]
