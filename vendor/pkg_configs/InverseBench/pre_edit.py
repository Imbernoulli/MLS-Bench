"""Pre-edit operations for InverseBench package.
Injects TRAIN_METRICS and TEST_METRICS printing into main.py.
Patches dataset.py to handle optional sigpy import.
Fixes eval.Image metric_state accumulation bug (list += float TypeError).

Line references (original file):
- training/dataset.py line 5: import sigpy as sp
- eval.py lines 175, 179: self.metric_state[metric_name] += ...
- main.py line 115: logger.info(f"Metric results: {metric_dict}...")
- main.py line 120: logger.info(f"Final metric results: {metric_state}...")

Operations ordered: different files first, then bottom-to-top within same file.
"""

OPS = [
    # 0. inverse_scatter.py lines 425-430: make the inv-scatter SVD cache load
    #    device-robust. The cache (U/S/Vt/matrix/matrix_inv) is precomputed once
    #    at build time (vendor/data_scripts/InverseBench/precompute_inv_scatter_svd.py)
    #    and shared via the mounted cache dir, so the operator never pays the
    #    expensive float64 torch.svd/pinv at runtime. torch.load without a
    #    map_location restores tensors onto whatever device they were saved on,
    #    which can mismatch self.device (e.g. when the artifact was precomputed
    #    on CPU but the run uses cuda). Pin every loaded tensor to self.device.
    #    Different file from the other ops below, so no line-shift interaction.
    {
        "op": "replace",
        "file": "InverseBench/inverse_problems/inverse_scatter.py",
        "start_line": 425,
        "end_line": 430,
        "content": (
            "            self.U = torch.load(os.path.join(path, 'U.pt'), map_location=self.device)\n"
            "            self.Sigma = torch.load(os.path.join(path, 'S.pt'), map_location=self.device)\n"
            "            self.V_t = torch.load(os.path.join(path, 'Vt.pt'), map_location=self.device)\n"
            "            self.A = torch.load(os.path.join(path, 'matrix.pt'), map_location=self.device)\n"
            "            if os.path.exists(path + '/matrix_inv.pt'):\n"
            "                self.A_inv = torch.load(os.path.join(path, 'matrix_inv.pt'), map_location=self.device)\n"
        ),
    },
    # 1. Patch dataset.py line 5: make sigpy import optional (replace 1 line with 4)
    {
        "op": "replace",
        "file": "InverseBench/training/dataset.py",
        "start_line": 5,
        "end_line": 5,
        "content": "try:\n    import sigpy as sp\nexcept ImportError:\n    sp = None\n",
    },
    # 2. Fix eval.py line 179: metric_state is a list (init in base Evaluator),
    #    so `list += float` raises TypeError. Use .append() instead.
    #    Patch bottom-to-top within eval.py to avoid line shifting.
    {
        "op": "replace",
        "file": "InverseBench/eval.py",
        "start_line": 179,
        "end_line": 179,
        "content": "                self.metric_state[metric_name].append(val)\n",
    },
    # 3. Fix eval.py line 175: metric_dict[metric_name] is a tensor scalar here
    #    (accumulated from metric_func(...).sum()), so cast via .item() before append.
    {
        "op": "replace",
        "file": "InverseBench/eval.py",
        "start_line": 175,
        "end_line": 175,
        "content": "                _v = metric_dict[metric_name]\n                self.metric_state[metric_name].append(_v.item() if hasattr(_v, 'item') else float(_v))\n",
    },
    # 4. Inject TEST_METRICS after main.py line 120 (before TRAIN_METRICS to avoid shifting)
    {
        "op": "insert",
        "file": "InverseBench/main.py",
        "after_line": 120,
        "content": '    for _k, _v in metric_state.items():\n        print(f"TEST_METRICS {_k}={_v}", flush=True)\n',
    },
    # 5. Inject TRAIN_METRICS after main.py line 115
    {
        "op": "insert",
        "file": "InverseBench/main.py",
        "after_line": 115,
        "content": '        _metric_parts = " ".join(f"{k}={v}" for k, v in metric_dict.items())\n        print(f"TRAIN_METRICS sample={data_id} {_metric_parts}", flush=True)\n',
    },
]
