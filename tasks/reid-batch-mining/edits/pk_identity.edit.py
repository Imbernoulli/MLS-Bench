"""Strong sampler baseline: P x K identity sampler (K=4) enabling batch-hard mining.
Reference: vendor/torchreid-reid/baselines/sampler_pk.py
"""
_FILE = "torchreid-reid/solution/sampler.py"
_CONTENT = '''def build_sampler(items, batch_size):
    from torchreid.data.sampler import RandomIdentitySampler

    return RandomIdentitySampler(items, batch_size, num_instances=4)'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 15, "content": _CONTENT},
]
