"""Strong sampler baseline: P x K identity sampler (K=4) enabling batch-hard mining."""
def build_sampler(items, batch_size):
    from torchreid.data.sampler import RandomIdentitySampler
    return RandomIdentitySampler(items, batch_size, num_instances=4)
