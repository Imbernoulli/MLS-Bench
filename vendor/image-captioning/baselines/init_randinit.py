"""Design the Mapping-Network Initialization — weak baseline (randinit).

Reference implementation for the caption-mapping-init surface (init_mapping). See tasks/caption-mapping-init/edits/randinit.edit.py.
"""
import torch


def init_mapping(mapping, ctx):
    # Keep PyTorch's default random initialization of the mapping.
    return None
