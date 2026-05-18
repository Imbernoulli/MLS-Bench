"""Prepare torchvision datasets and CIFAR model cache for torchattacks tasks."""
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve() / "torchattacks"
    root.mkdir(parents=True, exist_ok=True)

    import torch
    import torchvision

    torchvision.datasets.CIFAR10(root / "cifar10", train=True, download=True)
    torchvision.datasets.CIFAR10(root / "cifar10", train=False, download=True)
    torchvision.datasets.CIFAR100(root / "cifar100", train=True, download=True)
    torchvision.datasets.CIFAR100(root / "cifar100", train=False, download=True)
    torchvision.datasets.MNIST(root / "mnist", train=True, download=True)
    torchvision.datasets.MNIST(root / "mnist", train=False, download=True)

    cache = root / "torch_cache"
    cache.mkdir(parents=True, exist_ok=True)
    names = [
        "cifar10_resnet20",
        "cifar10_vgg11_bn",
        "cifar10_mobilenetv2_x1_0",
        "cifar100_resnet20",
        "cifar100_vgg11_bn",
        "cifar100_mobilenetv2_x1_0",
    ]
    old_hub = torch.hub.get_dir()
    torch.hub.set_dir(str(cache / "hub"))
    try:
        for name in names:
            torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True, trust_repo=True)
    finally:
        torch.hub.set_dir(old_hub)

    # RobustBench L2-robust CIFAR-10 checkpoints for the
    # security-adversarial-attack-sparse-l0 task (Sparse-RS canonical L0
    # setting: k=24, untargeted, robust targets). The vendored RobustBench
    # download_gdrive cannot pass Google Drive's large-file confirm token,
    # so fetch with gdown here (host has network; compute nodes do not).
    rb_dir = root / "robustbench_models" / "cifar10" / "L2"
    rb_dir.mkdir(parents=True, exist_ok=True)
    rb_models = [
        "Rebuffi2021Fixing_R18_cutmix_ddpm",
        "Augustin2020Adversarial",
        "Engstrom2019Robustness",
    ]
    ta_src = Path(__file__).resolve().parents[2] / "external_packages" / "torchattacks"
    sys.path.insert(0, str(ta_src))
    from robustbench.model_zoo.cifar10 import cifar_10_models
    from robustbench.model_zoo.enums import ThreatModel

    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown

    l2_zoo = cifar_10_models[ThreatModel.L2]
    for name in rb_models:
        out = rb_dir / f"{name}.pt"
        # Re-download if missing or a stale HTML interstitial (< 1 MB).
        if out.exists() and out.stat().st_size > 1_000_000:
            continue
        gid = l2_zoo[name]["gdrive_id"]
        gdown.download(id=gid, output=str(out), quiet=True)
        if out.stat().st_size < 1_000_000:
            print(f"RobustBench checkpoint {name} download failed "
                  f"(size={out.stat().st_size}B, likely an HTML page)",
                  file=sys.stderr)
            sys.exit(1)

    checks = [
        root / "cifar10" / "cifar-10-batches-py",
        root / "cifar100" / "cifar-100-python",
        root / "mnist" / "MNIST",
        cache / "hub" / "checkpoints",
        *[rb_dir / f"{n}.pt" for n in rb_models],
    ]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"Missing torchattacks data: {missing}", file=sys.stderr)
        sys.exit(1)
    print("torchattacks data ready")


if __name__ == "__main__":
    main()
