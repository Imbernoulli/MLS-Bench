"""LSUV (Layer-sequential Unit-Variance) initialization baseline.

First initializes with orthogonal matrices, then iteratively rescales
each layer's weights so that the output variance is 1.0, using a small
calibration batch. This data-driven approach ensures proper signal
propagation regardless of network depth.

Reference: Mishkin & Matas, "All you need is a good init" (ICLR 2016)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_init.py"

_CONTENT = """\
def initialize_weights(model, config):
    \"\"\"LSUV: Layer-sequential Unit-Variance (Mishkin & Matas, ICLR 2016).

    Step 1: Orthogonal initialization (gain=sqrt(2) for ReLU) for all
    Conv2d and Linear layers, standard BN init.
    Step 2: Iteratively rescale each layer's weights so that its output
    variance equals 1.0, using a synthetic calibration batch in train
    mode (so BatchNorm uses batch statistics).
    \"\"\"
    import math

    # Step 1: Orthogonal init with ReLU gain
    gain = math.sqrt(2.0)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=gain)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Step 2: Iterative variance normalization with synthetic data
    # Use train mode so BatchNorm computes batch stats (not running stats)
    model.train()
    tgt_var = 1.0
    tol = 0.1
    max_iter = 15

    x_cal = torch.randn(32, 3, 32, 32)

    # Collect target layers in forward order
    target_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            target_layers.append((name, m))

    # Process each layer sequentially
    for layer_name, layer_mod in target_layers:
        hook_out = {}

        def make_hook(storage):
            def hook_fn(mod, inp, out):
                storage['out'] = out.detach().clone()
            return hook_fn

        handle = layer_mod.register_forward_hook(make_hook(hook_out))
        for _ in range(max_iter):
            hook_out.clear()
            with torch.no_grad():
                model(x_cal)
            if 'out' not in hook_out:
                break
            var = hook_out['out'].var().item()
            if abs(var - tgt_var) < tol or var < 1e-8:
                break
            scale = (var / tgt_var) ** 0.5
            layer_mod.weight.data.div_(scale)
        handle.remove()

    model.train()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 228,
        "end_line": 261,
        "content": _CONTENT,
    },
]
