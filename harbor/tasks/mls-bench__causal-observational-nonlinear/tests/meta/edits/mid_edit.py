"""Mid-edit operations for causal-observational-nonlinear.

Creates the bench/ evaluation scaffold inside the causal-learn package workspace:
  bench/data_gen.py          -- synthetic nonlinear ANM data generator
  bench/metrics.py           -- SHD / F1 / precision / recall computation
  bench/run_eval.py          -- CLI evaluation harness
  bench/custom_algorithm.py  -- agent-editable algorithm entry point
"""

from pathlib import Path

_HERE = Path(__file__).parent

OPS = [
    {
        "op": "create",
        "file": "causal-learn/bench/data_gen.py",
        "content": (_HERE / "data_gen_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "causal-learn/bench/metrics.py",
        "content": (_HERE / "metrics_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "causal-learn/bench/run_eval.py",
        "content": (_HERE / "run_eval_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "causal-learn/bench/custom_algorithm.py",
        "content": (_HERE / "custom_template.py").read_text(),
    },
]
