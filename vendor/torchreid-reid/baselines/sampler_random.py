"""Weak sampler baseline: plain random shuffle (starves batch-hard triplet mining)."""
def build_sampler(items, batch_size):
    from torch.utils.data.sampler import RandomSampler
    return RandomSampler(items)
