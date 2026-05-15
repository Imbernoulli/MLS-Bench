"""Mid-edit operations for causal-observational-linear-gaussian."""
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
