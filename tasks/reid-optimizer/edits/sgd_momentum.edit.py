"""SGD-with-momentum baseline under the harness-owned fixed epoch schedule.
Reference: vendor/torchreid-reid/baselines/optim_sgd_mom.py
"""
_FILE = "torchreid-reid/solution/optimizer.py"
_CONTENT = '''def build_optimizer(params):
    import torch

    return torch.optim.SGD(
        params, lr=3.5e-4, momentum=0.9, weight_decay=5e-4
    )'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 6, "end_line": 9, "content": _CONTENT},
]
