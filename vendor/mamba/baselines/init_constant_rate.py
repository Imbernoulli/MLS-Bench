"""Initialize the fixed A transform to a constant decay rate A=-1."""

import torch


def init_state(block):
    with torch.no_grad():
        # The fixed transform is A=-exp(A_log), so A_log=0 gives A=-1.
        block.A_log.zero_()
