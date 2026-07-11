"""Uniform factorized policy anchor for compress-quality-dispatch."""

OPS = [
    {
        "op": "replace",
        "file": 'compressai/solution/quality_policy.py',
        "start_line": 6,
        "end_line": 6,
        "content": "    return ('factorized', 'factorized', 'factorized')",
    },
]
