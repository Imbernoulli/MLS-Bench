"""Mid-edit operations for causal-discovery-discrete.

Creates the bench/ evaluation scaffold inside the causal-bnlearn package workspace:
  bench/data_gen.py          — bnlearn data loader and sampler
  bench/metrics.py           — SHD / adjacency / arrow precision-recall
  bench/run_eval.py          — CLI evaluation harness
  bench/custom_algorithm.py  — agent-editable algorithm entry point
"""

from pathlib import Path

_HERE = Path(__file__).parent

OPS = [
    {
        "op": "create",
        "file": "causal-bnlearn/bench/data_gen.py",
        "content": (_HERE / "data_gen_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "causal-bnlearn/bench/metrics.py",
        "content": (_HERE / "metrics_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "causal-bnlearn/bench/run_eval.py",
        "content": (_HERE / "run_eval_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "causal-bnlearn/bench/custom_algorithm.py",
        "content": (_HERE / "custom_template.py").read_text(),
    },
]
