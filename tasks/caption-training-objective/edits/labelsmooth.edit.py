"""Static reference configuration for offline calibration."""

OPS = [
    {
        "op": "replace",
        "file": 'image-captioning/solution/objective.py',
        "start_line": 8,
        "end_line": 8,
        "content": "CONFIG = {'label_smoothing': 0.1}",
    }
]
