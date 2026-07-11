"""Uniform meanscale policy anchor for compress-high-rate-policy."""

OPS = [
    {
        "op": "replace",
        "file": 'compressai/solution/high_rate_policy.py',
        "start_line": 6,
        "end_line": 6,
        "content": "    return ('meanscale', 'meanscale', 'meanscale')",
    },
]
