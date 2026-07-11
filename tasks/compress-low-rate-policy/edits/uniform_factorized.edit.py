"""Uniform factorized policy anchor for compress-low-rate-policy."""

OPS = [
    {
        "op": "replace",
        "file": 'compressai/solution/low_rate_policy.py',
        "start_line": 6,
        "end_line": 6,
        "content": "    return ('factorized', 'factorized', 'factorized')",
    },
]
