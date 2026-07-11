"""Uniform factorized policy anchor for compress-objective-policy."""

OPS = [
    {
        "op": "replace",
        "file": 'compressai/solution/objective_policy.py',
        "start_line": 6,
        "end_line": 6,
        "content": "    return ('factorized', 'factorized', 'factorized')",
    },
]
