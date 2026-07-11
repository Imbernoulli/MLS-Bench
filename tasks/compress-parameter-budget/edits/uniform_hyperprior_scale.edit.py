"""Uniform hyperprior_scale policy anchor for compress-parameter-budget."""

OPS = [
    {
        "op": "replace",
        "file": 'compressai/solution/parameter_budget_policy.py',
        "start_line": 6,
        "end_line": 6,
        "content": "    return 'hyperprior_scale'",
    },
]
