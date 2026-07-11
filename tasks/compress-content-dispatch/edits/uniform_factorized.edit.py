"""Uniform factorized policy anchor for compress-content-dispatch."""

OPS = [
    {
        "op": "replace",
        "file": 'compressai/solution/content_policy.py',
        "start_line": 6,
        "end_line": 6,
        "content": "    return ('factorized', 'factorized', 'factorized')",
    },
]
