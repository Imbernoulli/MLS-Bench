"""Static reference configuration for offline calibration."""

OPS = [
    {
        "op": "replace",
        "file": 'image-captioning/solution/prompt.py',
        "start_line": 8,
        "end_line": 8,
        "content": "CONFIG = {'prefix': 'a photo of ', 'lowercase': False, 'strip_terminal_period': False}",
    }
]
