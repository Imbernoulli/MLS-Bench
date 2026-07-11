"""Static reference configuration for offline calibration."""

OPS = [
    {
        "op": "replace",
        "file": 'image-captioning/solution/weighting.py',
        "start_line": 8,
        "end_line": 8,
        "content": "CONFIG = {'scheme': 'uniform'}",
    }
]
