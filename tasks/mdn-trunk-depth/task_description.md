# MDN Trunk Depth

Choose the number of hidden layers in a fixed-width mixture-density trunk.

Edit `mdn-density/solution/trunk_depth.py` and make `surface_config()` return a
JSON literal with exactly one key, `trunk_depth`. The value must be an integer
from 1 through 4. Component count, width, activation, optimizer, data, and
training budget remain fixed, so this task isolates representational depth.

Evaluation reports held-out mixture NLL. Lower NLL is preferred. Every candidate
uses the complete 20,000-example train and test inventories and 4,000 updates.
