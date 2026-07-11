"""Static reference configuration for offline calibration."""

OPS = [
    {
        "op": "replace",
        "file": 'image-captioning/solution/optimizer.py',
        "start_line": 8,
        "end_line": 8,
        "content": "CONFIG = {'name': 'sgd', 'learning_rate': 0.001, 'weight_decay': 0.0, 'momentum': 0.9, 'schedule': 'constant', 'warmup_steps': 0}",
    }
]
