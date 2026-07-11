"""Static reference configuration for offline calibration."""

OPS = [
    {
        "op": "replace",
        "file": 'image-captioning/solution/optimizer.py',
        "start_line": 8,
        "end_line": 8,
        "content": "CONFIG = {'name': 'adamw', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'schedule': 'warmup_cosine', 'warmup_steps': 500}",
    }
]
