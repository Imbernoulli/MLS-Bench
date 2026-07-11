#!/usr/bin/env python3
"""Prepare the pinned CompressAI-1.2.8 Kodak/zoo verifier bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image


PROTOCOL = "compressai_zoo_kodak24_q1q8_bitstream_v1"
QUALITIES = tuple(range(1, 9))
FAMILIES = {
    "factorized": ("bmshj2018-factorized", "bmshj2018_factorized"),
    "hyperprior_scale": ("bmshj2018-hyperprior", "bmshj2018_hyperprior"),
    "meanscale": ("mbt2018-mean", "mbt2018_mean"),
}
GROUPS = {
    "low": [
        "kodim02.png", "kodim03.png", "kodim09.png", "kodim10.png",
        "kodim12.png", "kodim15.png", "kodim20.png", "kodim23.png",
    ],
    "mid": [
        "kodim04.png", "kodim07.png", "kodim11.png", "kodim16.png",
        "kodim17.png", "kodim19.png", "kodim21.png", "kodim22.png",
    ],
    "high": [
        "kodim01.png", "kodim05.png", "kodim06.png", "kodim08.png",
        "kodim13.png", "kodim14.png", "kodim18.png", "kodim24.png",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kodak-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    import compressai
    import compressai.zoo.image as zoo_image

    if str(compressai.__version__) != "1.2.8":
        raise SystemExit(f"expected CompressAI 1.2.8, got {compressai.__version__}")
    if args.output_root.exists():
        raise SystemExit(f"refusing to reuse output root: {args.output_root}")

    output = args.output_root.resolve()
    kodak_output = output / "kodak"
    checkpoint_output = output / "checkpoints"
    kodak_output.mkdir(parents=True)
    checkpoint_output.mkdir()

    image_records = []
    expected_names = [f"kodim{index:02d}.png" for index in range(1, 25)]
    group_by_image = {
        image: group for group, images in GROUPS.items() for image in images
    }
    if set(group_by_image) != set(expected_names) or len(group_by_image) != 24:
        raise SystemExit("hard-coded Kodak strata do not form an exact partition")
    for name in expected_names:
        source = args.kodak_root / name
        if not source.is_file():
            raise SystemExit(f"missing Kodak image: {source}")
        with Image.open(source) as image:
            image.verify()
        with Image.open(source) as image:
            if image.mode != "RGB" or sorted(image.size) != [512, 768]:
                raise SystemExit(f"unexpected Kodak geometry/mode: {name}")
            width, height = image.size
        target = kodak_output / name
        shutil.copyfile(source, target)
        image_records.append(
            {
                "name": name,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "width": width,
                "height": height,
                "group": group_by_image[name],
            }
        )

    # Record every shipped Python source and native extension that implements
    # the zoo model or entropy coder. The verifier re-hashes this inventory.
    package_root = Path(compressai.__file__).resolve().parent
    package_files = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".so"}:
            continue
        package_files[path.relative_to(package_root).as_posix()] = sha256(path)
    if not package_files or not any(name.endswith(".so") for name in package_files):
        raise SystemExit("incomplete CompressAI source/native-extension inventory")

    family_records = {}
    original_loader = zoo_image.load_state_dict_from_url
    for family, (architecture, factory_name) in FAMILIES.items():
        factory = getattr(zoo_image, factory_name)
        checkpoint_records = []
        for quality in QUALITIES:
            captured_urls: list[str] = []

            def capture_loader(url, *loader_args, **loader_kwargs):
                captured_urls.append(str(url))
                return original_loader(url, *loader_args, **loader_kwargs)

            zoo_image.load_state_dict_from_url = capture_loader
            try:
                model = factory(
                    quality=quality, metric="mse", pretrained=True, progress=True
                )
            finally:
                zoo_image.load_state_dict_from_url = original_loader
            if len(captured_urls) != 1:
                raise SystemExit(
                    f"expected one official URL for {family} q{quality}, got {captured_urls}"
                )
            url = captured_urls[0]
            cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(
                urlparse(url).path
            ).name
            if not cached.is_file():
                raise SystemExit(f"official downloaded checkpoint not found: {cached}")
            target_name = f"{family}-mse-q{quality}.pth.tar"
            target = checkpoint_output / target_name
            shutil.copyfile(cached, target)

            # Prove that the exact pinned file can be reconstructed by the same
            # official from_state_dict path used by the final verifier.
            try:
                state_dict = torch.load(target, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(target, map_location="cpu")
            rebuilt = zoo_image.model_architectures[architecture].from_state_dict(
                __import__("compressai.zoo.pretrained", fromlist=["load_pretrained"])
                .load_pretrained(state_dict)
            )
            if not rebuilt.update(force=True):
                raise SystemExit(f"entropy CDF update was not observed for {family} q{quality}")
            parameter_count = sum(parameter.numel() for parameter in rebuilt.parameters())
            del rebuilt, model, state_dict

            checkpoint_records.append(
                {
                    "quality": quality,
                    "metric": "mse",
                    "file": target_name,
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                    "official_url": url,
                    "parameters": parameter_count,
                }
            )
            print(
                f"COMPRESS_PREP_CHECKPOINT family={family} quality={quality} "
                f"bytes={target.stat().st_size} sha256={checkpoint_records[-1]['sha256']}",
                flush=True,
            )
        family_records[family] = {
            "architecture": architecture,
            "factory": factory_name,
            "checkpoints": checkpoint_records,
        }

    protocol = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "qualities": list(QUALITIES),
        "rd_beta": 12.0,
        "expected_images": 24,
        "expected_cases_per_family": 192,
        "dataset": {
            "name": "Kodak PhotoCD image set",
            "source": "http://r0k.us/graphics/kodak/",
            "files": image_records,
            "groups": GROUPS,
        },
        "families": family_records,
        "runtime": {
            "compressai_version": str(compressai.__version__),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "entropy_coder": "ans",
            "package_files": package_files,
        },
        "final_verifier": {
            "network_access": False,
            "runtime_install": False,
            "runtime_download": False,
            "runtime_compile": False,
            "operation": "compress/decompress each family-quality-image case",
        },
    }
    write_json(output / "protocol.json", protocol)
    protocol_sha = sha256(output / "protocol.json")
    (output / "protocol.sha256").write_text(
        f"{protocol_sha}  protocol.json\n"
    )
    print(
        f"COMPRESS_PREP_COMPLETE protocol={PROTOCOL} images=24 families=3 "
        f"qualities=8 cases_per_family=192 protocol_sha256={protocol_sha}",
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
