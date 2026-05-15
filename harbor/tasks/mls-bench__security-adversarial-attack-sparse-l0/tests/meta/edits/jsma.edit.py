"""JSMA baseline for security-adversarial-attack-sparse-l0.

torchattacks.JSMA is targeted-only by design
(https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/jsma.py).
The previous baseline had two bugs:

  1. ``gamma`` was set to a constant 0.0098 intended as "10 / 1024" (spatial
     pixels on CIFAR), but torchattacks' JSMA computes
     ``num_features = np.prod(img.shape[1:])`` i.e. ``C*H*W = 3072`` on CIFAR
     and then ``max_iters = ceil(num_features * gamma / 2)``. Each iteration
     modifies 2 features, so the attack could perturb up to
     ``num_features * gamma = 3072 * 0.0098 ≈ 30`` features, which may cover
     up to 30 distinct spatial pixels -- far above the budget of 10. The
     evaluator then rejects every such sample and ASR collapses to 0.

     Fix: set ``gamma = pixels / (C*H*W)`` so ``max_iters = ceil(pixels/2)``
     and the total number of perturbed features is ``<= pixels``, which
     upper-bounds the number of distinct spatial pixels by ``pixels``.

  2. The targeted-map used ``(labels + 1) % n_classes``, a weak fixed target,
     and the previous version even hard-coded ``% 10`` which produced
     out-of-range targets on CIFAR-100 (100 classes).

     Fix: use torchattacks' built-in ``set_mode_targeted_least_likely`` which
     picks the model's least-likely class per sample as the target -- a
     strong untargeted proxy that works for any ``n_classes``.
"""

_FILE = "torchattacks/bench/custom_attack.py"

_JSMA_FN = """\
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torchattacks

    _ = (device, n_classes)
    model.eval()

    # gamma bounds total perturbed features (C*H*W space) to `pixels`, which
    # is a sufficient upper bound on the number of distinct spatial pixels.
    num_features = int(images.shape[1] * images.shape[2] * images.shape[3])
    gamma = float(pixels) / float(num_features)

    attack = torchattacks.JSMA(model, theta=1.0, gamma=gamma)
    # Least-likely class as target -> strong untargeted proxy, works for any
    # n_classes (fixes CIFAR-100 out-of-range target bug).
    attack.set_mode_targeted_least_likely(quiet=True)
    return attack(images, labels)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 23,
        "content": _JSMA_FN,
    }
]
