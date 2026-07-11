"""Static reference configuration for offline calibration."""

OPS = [
    {
        "op": "replace",
        "file": 'image-captioning/solution/augment.py',
        "start_line": 8,
        "end_line": 8,
        "content": "CONFIG = {'gaussian_std': 0.0, 'dropout_probability': 0.0}",
    }
]
