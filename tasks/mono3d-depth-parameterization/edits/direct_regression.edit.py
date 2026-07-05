"""mono3d-depth-parameterization WEAK baseline: DIRECT depth regression.

Regress metric depth Z DIRECTLY as an unbounded scalar (softplus for positivity), IGNORING
the projective geometry (focal / 2D box height / predicted metric height). Direct depth
regression is dominated by far objects and generalizes poorly across distance -> low AP3D,
worst in the far regime. This is the deliberately weak reference the sigmoid floor is anchored
around. Reference: vendor/mono3d-detection/baselines/direct_regression.py
"""

_FILE = "mono3d-detection/solution/depth_param.py"

_CONTENT = '''def build_depth_head(emb_dim):
    # WEAK: regress metric depth DIRECTLY (softplus), ignoring the projective geometry.
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        return F.softplus(raw[:, 0]) + 1.0

    return head, decode'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 48, "content": _CONTENT},
]
