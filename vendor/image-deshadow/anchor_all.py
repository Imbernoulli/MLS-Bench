#!/usr/bin/env python3
"""Batch anchor runner for ALL image-deshadow editable surfaces.

Runs, for every (surface, setting in {light,medium,heavy}, config in {weak,strong}), the
harness end-to-end and prints one ANCHOR line:

    ANCHOR surface=<S> setting=<L> config=<weak|strong> psnr=<..> shadow_psnr=<..> gain=<..>

so all new RQ anchors are pinned from ONE GPU job. The `network` surface (already validated
in tasks/deshadow-network-design / -mask-guidance) is re-runnable here for a sanity check.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import harness as H  # noqa: E402


def _mod(fn_name: str, cfg: dict):
    m = types.ModuleType("anchor_cfg")
    setattr(m, fn_name, (lambda c=cfg: dict(c)))
    return m


# (surface, hook_name, weak_cfg, strong_cfg)
SPECS = [
    # named-arch surface: 3-way (copy / blind / mask-guided). We anchor the two informative
    # pairs (copy->mask for -network-design, nomask->mask for -mask-guidance) via `network`.
    ("network_copy", "get_network_config", {"arch": "copy"}, {"arch": "unet_mask"}),
    ("network_nomask", "get_network_config", {"arch": "unet_nomask"}, {"arch": "unet_mask"}),
    # new configurable surfaces on the mask-guided residual deshadower:
    ("mask", "get_mask_config", {"use_mask": False}, {"use_mask": True}),
    ("architecture", "get_arch_config", {"depth": 1}, {"depth": 2}),
    ("loss", "get_loss_config", {"ssim": False, "color": False, "comp": False},
     {"ssim": True, "color": True, "comp": True}),
    ("attention", "get_attention_config", {"attention": False}, {"attention": True}),
    ("dilation", "get_dilation_config", {"dilations": [1, 1]}, {"dilations": [2, 4]}),
    ("normalization", "get_norm_config", {"norm": "none"}, {"norm": "in"}),
    ("multiscale", "get_multiscale_config", {"multiscale": False}, {"multiscale": True}),
    ("fusion", "get_fusion_config", {"fusion": False}, {"fusion": True}),
    ("physics", "get_physics_config", {"mode": "residual"}, {"mode": "physics"}),
    ("upsampling", "get_upsampling_config", {"up": "transpose"}, {"up": "bilinear"}),
    ("residual", "get_residual_config", {"mode": "direct"}, {"mode": "residual"}),
]

# map each anchor-spec name to the actual harness --surface it drives.
_SURFACE_OF = {
    "network_copy": "network", "network_nomask": "network",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="root holding light/ medium/ heavy/ subdirs")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default="", help="comma-separated surface subset")
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE {device} torch {torch.__version__}", flush=True)

    settings = ("light", "medium", "heavy")
    for name, hook, weak, strong in SPECS:
        if only and name not in only:
            continue
        surface = _SURFACE_OF.get(name, name)
        for label, cfg in (("weak", weak), ("strong", strong)):
            mod = _mod(hook, cfg)
            for s in settings:
                droot = str(Path(args.data_root) / s)
                m = H.run(surface, mod, droot, device, args.iters, args.seed)
                print(f"ANCHOR surface={name} setting={s} config={label} "
                      f"psnr={m['psnr']:.4f} shadow_psnr={m['shadow_psnr']:.4f} "
                      f"gain={m['psnr_gain']:.4f} ssim={m['ssim']:.4f} "
                      f"full_psnr={m['full_psnr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
