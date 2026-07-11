"""Weak sampler baseline: plain random shuffle (starves batch-hard triplet mining).
Reference: vendor/torchreid-reid/baselines/sampler_random.py
"""
_FILE = "torchreid-reid/solution/sampler.py"
_CONTENT = '''def build_sampler(items, batch_size):
    from torch.utils.data.sampler import RandomSampler

    return RandomSampler(items)'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 15, "content": _CONTENT},
]
