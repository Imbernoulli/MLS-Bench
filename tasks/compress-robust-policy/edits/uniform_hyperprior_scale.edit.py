"""Uniform hyperprior_scale policy anchor for compress-robust-policy."""

OPS = [
    {
        "op": "replace",
        "file": 'compressai/solution/robust_policy.py',
        "start_line": 6,
        "end_line": 6,
        "content": "    return ('hyperprior_scale', 'hyperprior_scale', 'hyperprior_scale')",
    },
]
