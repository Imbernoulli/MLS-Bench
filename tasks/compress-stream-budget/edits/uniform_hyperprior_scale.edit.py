"""Uniform hyperprior_scale policy anchor for compress-stream-budget."""

OPS = [
    {
        "op": "replace",
        "file": 'compressai/solution/stream_budget_policy.py',
        "start_line": 6,
        "end_line": 6,
        "content": "    return 'hyperprior_scale'",
    },
]
