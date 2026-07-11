"""Static reference configuration for offline calibration."""

OPS = [
    {
        "op": "replace",
        "file": 'image-captioning/solution/decoding.py',
        "start_line": 8,
        "end_line": 8,
        "content": "CONFIG = {'strategy': 'greedy', 'max_length': 24, 'min_length': 3, 'no_repeat_ngram': 2}",
    }
]
