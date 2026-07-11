"""Static reference configuration for offline calibration."""

OPS = [
    {
        "op": "replace",
        "file": 'image-captioning/solution/decoding.py',
        "start_line": 8,
        "end_line": 8,
        "content": "CONFIG = {'strategy': 'sample', 'max_length': 24, 'min_length': 3, 'no_repeat_ngram': 0, 'temperature': 1.0, 'top_p': 1.0}",
    }
]
