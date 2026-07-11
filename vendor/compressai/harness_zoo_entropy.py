#!/usr/bin/env python3
"""Full Kodak-24 bitstream evaluation for official CompressAI zoo codecs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROTOCOL_ID = "compressai_zoo_kodak24_q1q8_bitstream_v1"
QUALITIES = tuple(range(1, 9))
SETTINGS = ("full", "low", "mid", "high")
RD_BETA = 12.0
EXPECTED_IMAGES = 24
EXPECTED_CASES = EXPECTED_IMAGES * len(QUALITIES)
FAMILIES = {
    "factorized": "bmshj2018-factorized",
    "hyperprior_scale": "bmshj2018-hyperprior",
    "meanscale": "mbt2018-mean",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _block_url_loading() -> None:
    def blocked(*_args, **_kwargs):
        raise RuntimeError("network loading is disabled by the fixed verifier")

    torch.hub.load_state_dict_from_url = blocked
    import compressai.zoo.image as zoo_image

    zoo_image.load_state_dict_from_url = blocked
    socket.create_connection = blocked
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_HOME": "/nonexistent-network-cache",
        }
    )


def _load_surface(path: Path) -> str:
    tree = ast.parse(path.read_text(), filename=str(path))
    allowed_module_nodes = (ast.Expr, ast.ImportFrom, ast.FunctionDef)
    if any(not isinstance(node, allowed_module_nodes) for node in tree.body):
        raise TypeError("solution may contain only its docstring, future import, and entropy_model")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise TypeError("solution must define exactly one function")
    fn = functions[0]
    if (
        fn.name != "entropy_model"
        or fn.args.args
        or fn.args.posonlyargs
        or fn.args.kwonlyargs
        or fn.args.vararg is not None
        or fn.args.kwarg is not None
        or fn.decorator_list
        or len(fn.body) != 1
        or not isinstance(fn.body[0], ast.Return)
    ):
        raise TypeError("entropy_model must be a zero-argument literal-return function")
    value = ast.literal_eval(fn.body[0].value)
    if not isinstance(value, str) or value not in FAMILIES:
        raise ValueError(f"entropy_model() must return one of {sorted(FAMILIES)}")
    return value


def _validate_protocol(path: Path, expected_sha256: str) -> dict:
    if _sha256(path) != expected_sha256:
        raise RuntimeError("protocol manifest digest mismatch")
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != 1 or protocol.get("protocol") != PROTOCOL_ID:
        raise RuntimeError("unexpected protocol manifest")
    if protocol.get("qualities") != list(QUALITIES):
        raise RuntimeError("quality inventory mismatch")
    if protocol.get("rd_beta") != RD_BETA:
        raise RuntimeError("R-D objective mismatch")
    if protocol.get("expected_images") != EXPECTED_IMAGES:
        raise RuntimeError("image count mismatch")
    if protocol.get("expected_cases_per_family") != EXPECTED_CASES:
        raise RuntimeError("case count mismatch")
    if set(protocol.get("families", {})) != set(FAMILIES):
        raise RuntimeError("family inventory mismatch")
    for name, architecture in FAMILIES.items():
        record = protocol["families"][name]
        if record.get("architecture") != architecture:
            raise RuntimeError(f"architecture mismatch for {name}")
        checkpoints = record.get("checkpoints", [])
        if [entry.get("quality") for entry in checkpoints] != list(QUALITIES):
            raise RuntimeError(f"checkpoint quality inventory mismatch for {name}")
        for entry in checkpoints:
            sha = str(entry.get("sha256", ""))
            if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
                raise RuntimeError(f"invalid checkpoint digest for {name}")
    dataset = protocol.get("dataset", {})
    files = dataset.get("files", [])
    if len(files) != EXPECTED_IMAGES:
        raise RuntimeError("Kodak inventory must contain exactly 24 images")
    names = [entry.get("name") for entry in files]
    expected_names = [f"kodim{i:02d}.png" for i in range(1, 25)]
    if names != expected_names or len(set(names)) != EXPECTED_IMAGES:
        raise RuntimeError("Kodak filename inventory mismatch")
    groups = dataset.get("groups", {})
    if set(groups) != {"low", "mid", "high"}:
        raise RuntimeError("Kodak group inventory mismatch")
    grouped = [name for group in ("low", "mid", "high") for name in groups[group]]
    if len(grouped) != EXPECTED_IMAGES or set(grouped) != set(expected_names):
        raise RuntimeError("Kodak content groups are not a partition")
    if any(len(groups[group]) != 8 for group in ("low", "mid", "high")):
        raise RuntimeError("each Kodak content group must contain eight images")
    return protocol


def _validate_runtime(protocol: dict) -> None:
    import compressai

    runtime = protocol.get("runtime", {})
    if str(compressai.__version__) != runtime.get("compressai_version"):
        raise RuntimeError("CompressAI runtime version mismatch")
    if str(torch.__version__) != runtime.get("torch_version"):
        raise RuntimeError("PyTorch runtime version mismatch")
    if str(torch.version.cuda) != runtime.get("cuda_version"):
        raise RuntimeError("CUDA runtime version mismatch")
    if runtime.get("entropy_coder") != "ans" or "ans" not in compressai.available_entropy_coders():
        raise RuntimeError("required ANS entropy coder is unavailable")
    package_root = Path(compressai.__file__).resolve().parent
    package_files = runtime.get("package_files", {})
    if not package_files:
        raise RuntimeError("missing pinned CompressAI source inventory")
    for rel, expected in sorted(package_files.items()):
        file_path = package_root / rel
        if not file_path.is_file() or _sha256(file_path) != expected:
            raise RuntimeError(f"CompressAI source digest mismatch: {rel}")


def _validate_inputs(protocol: dict, data_root: Path, checkpoint_root: Path) -> None:
    for entry in protocol["dataset"]["files"]:
        path = data_root / entry["name"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"missing or wrong-sized Kodak image: {entry['name']}")
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Kodak image digest mismatch: {entry['name']}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.mode != "RGB" or sorted(image.size) != [512, 768]:
                raise RuntimeError(f"invalid Kodak image geometry: {entry['name']}")
    for family in FAMILIES:
        for entry in protocol["families"][family]["checkpoints"]:
            path = checkpoint_root / entry["file"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"missing or wrong-sized checkpoint: {entry['file']}")
            if _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"checkpoint digest mismatch: {entry['file']}")


def _load_model(architecture: str, checkpoint: Path, device: torch.device):
    from compressai.zoo.image import model_architectures
    from compressai.zoo.pretrained import load_pretrained

    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint is not a state dictionary")
    model = model_architectures[architecture].from_state_dict(load_pretrained(state_dict))
    model.eval().to(device)
    updated = model.update(force=True)
    cdf_records = []
    for name, module in model.named_modules():
        cdf = getattr(module, "_quantized_cdf", None)
        lengths = getattr(module, "_cdf_length", None)
        offsets = getattr(module, "_offset", None)
        if cdf is None:
            continue
        if cdf.numel() == 0 or lengths is None or lengths.numel() == 0:
            raise RuntimeError(f"entropy CDF was not initialized: {name}")
        for label, tensor in (("cdf", cdf), ("length", lengths), ("offset", offsets)):
            if tensor is None:
                continue
            array = tensor.detach().cpu().contiguous().numpy()
            cdf_records.append(
                {
                    "module": name,
                    "kind": label,
                    "shape": list(array.shape),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
            )
    if not cdf_records:
        raise RuntimeError("model has no initialized entropy CDF")
    return model, bool(updated), _canonical_sha(cdf_records)


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)


def _stream_bytes(strings) -> tuple[int, int]:
    if not isinstance(strings, (list, tuple)) or not strings:
        raise RuntimeError("compress() returned no entropy streams")
    total = 0
    count = 0
    for stream_group in strings:
        if not isinstance(stream_group, (list, tuple)) or len(stream_group) != 1:
            raise RuntimeError("unexpected entropy stream batch structure")
        stream = stream_group[0]
        if not isinstance(stream, (bytes, bytearray)) or not stream:
            raise RuntimeError("empty or invalid entropy stream")
        total += len(stream)
        count += 1
    return total, count


def _encode_decode(model, source: torch.Tensor, device: torch.device):
    _, _, height, width = source.shape
    pad_h = (-height) % 64
    pad_w = (-width) % 64
    padded = F.pad(source.to(device), (0, pad_w, 0, pad_h), mode="constant", value=0)
    with torch.inference_mode():
        encoded = model.compress(padded)
        if not isinstance(encoded, dict) or set(encoded) != {"strings", "shape"}:
            raise RuntimeError("compress() returned an invalid artifact")
        nbytes, nstreams = _stream_bytes(encoded["strings"])
        decoded = model.decompress(encoded["strings"], encoded["shape"])
        if not isinstance(decoded, dict) or "x_hat" not in decoded:
            raise RuntimeError("decompress() returned an invalid artifact")
        reconstruction = decoded["x_hat"][:, :, :height, :width].clamp(0, 1)
    if reconstruction.shape != source.shape or not torch.isfinite(reconstruction).all().item():
        raise RuntimeError("decoded RGB tensor is invalid")
    mse = F.mse_loss(reconstruction.cpu(), source).item()
    if not math.isfinite(mse) or mse <= 0:
        raise RuntimeError("decoded RGB MSE is invalid")
    psnr = -10.0 * math.log10(mse)
    pixels = height * width
    bpp = 8.0 * nbytes / pixels
    recon_u8 = reconstruction.mul(255).round().to(torch.uint8).cpu().numpy()
    recon_sha = hashlib.sha256(recon_u8.tobytes()).hexdigest()
    return nbytes, nstreams, pixels, bpp, psnr, recon_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()

    if len(args.protocol_sha256) != 64:
        raise ValueError("invalid expected protocol digest")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the protocol requires exactly one visible CUDA GPU")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    t0 = time.monotonic()

    protocol_path = Path(args.protocol).resolve()
    protocol = _validate_protocol(protocol_path, args.protocol_sha256)
    _validate_runtime(protocol)
    import compressai

    compressai.set_entropy_coder("ans")
    data_root = Path(args.data_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    _validate_inputs(protocol, data_root, checkpoint_root)
    family = _load_surface(Path(args.solution).resolve())
    architecture = FAMILIES[family]
    _block_url_loading()
    print(
        f"COMPRESS_PROTOCOL protocol={PROTOCOL_ID} family={family} "
        f"images={EXPECTED_IMAGES} qualities={len(QUALITIES)} cases={EXPECTED_CASES} "
        f"protocol_sha={args.protocol_sha256} network=blocked",
        flush=True,
    )

    image_records = protocol["dataset"]["files"]
    group_by_image = {
        name: group
        for group, names in protocol["dataset"]["groups"].items()
        for name in names
    }
    cases = []
    model_records = []
    total_streams = 0

    checkpoints = protocol["families"][family]["checkpoints"]
    for checkpoint_record in checkpoints:
        quality = int(checkpoint_record["quality"])
        checkpoint = checkpoint_root / checkpoint_record["file"]
        model, updated, cdf_sha = _load_model(architecture, checkpoint, device)
        params = sum(parameter.numel() for parameter in model.parameters())
        model_record = {
            "quality": quality,
            "checkpoint_sha": checkpoint_record["sha256"],
            "cdf_sha": cdf_sha,
            "params": params,
            "updated": updated,
        }
        model_records.append(model_record)
        print(
            f"COMPRESS_MODEL quality={quality} checkpoint_sha={checkpoint_record['sha256']} "
            f"cdf_sha={cdf_sha} params={params} updated={int(updated)}",
            flush=True,
        )
        for image_record in image_records:
            name = image_record["name"]
            source = _load_image(data_root / name)
            nbytes, nstreams, pixels, bpp, psnr, recon_sha = _encode_decode(
                model, source, device
            )
            bpp = float(f"{bpp:.12f}")
            psnr = float(f"{psnr:.9f}")
            total_streams += nstreams
            case = {
                "quality": quality,
                "image": name,
                "group": group_by_image[name],
                "pixels": pixels,
                "bytes": nbytes,
                "bpp": bpp,
                "psnr": psnr,
                "recon_sha": recon_sha,
            }
            cases.append(case)
            print(
                f"COMPRESS_CASE quality={quality} image={name} group={case['group']} "
                f"pixels={pixels} bytes={nbytes} bpp={bpp:.12f} "
                f"psnr={psnr:.9f} recon_sha={recon_sha}",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    if len(cases) != EXPECTED_CASES:
        raise RuntimeError("incomplete encode/decode case matrix")
    setting_records = []
    for setting in SETTINGS:
        names = (
            {entry["name"] for entry in image_records}
            if setting == "full"
            else set(protocol["dataset"]["groups"][setting])
        )
        per_quality = []
        for quality in QUALITIES:
            selected = [
                case for case in cases if case["quality"] == quality and case["image"] in names
            ]
            if len(selected) != len(names):
                raise RuntimeError(f"incomplete {setting} quality-{quality} aggregate")
            total_bytes = sum(case["bytes"] for case in selected)
            total_pixels = sum(case["pixels"] for case in selected)
            mean_psnr = sum(case["psnr"] for case in selected) / len(selected)
            bpp = 8.0 * total_bytes / total_pixels
            utility = mean_psnr - RD_BETA * bpp
            bpp = float(f"{bpp:.12f}")
            mean_psnr = float(f"{mean_psnr:.9f}")
            utility = float(f"{utility:.9f}")
            record = {
                "setting": setting,
                "quality": quality,
                "count": len(selected),
                "pixels": total_pixels,
                "bytes": total_bytes,
                "bpp": bpp,
                "psnr": mean_psnr,
                "utility": utility,
            }
            per_quality.append(record)
            print(
                f"COMPRESS_RATE setting={setting} quality={quality} count={len(selected)} "
                f"pixels={total_pixels} bytes={total_bytes} bpp={bpp:.12f} "
                f"psnr={mean_psnr:.9f} utility={utility:.9f}",
                flush=True,
            )
        mean_rd_utility = sum(record["utility"] for record in per_quality) / len(per_quality)
        mean_psnr = sum(record["psnr"] for record in per_quality) / len(per_quality)
        mean_bpp = sum(record["bpp"] for record in per_quality) / len(per_quality)
        mean_rd_utility = float(f"{mean_rd_utility:.9f}")
        mean_psnr = float(f"{mean_psnr:.9f}")
        mean_bpp = float(f"{mean_bpp:.12f}")
        setting_record = {
            "setting": setting,
            "qualities": len(per_quality),
            "images": len(names),
            "cases": len(names) * len(per_quality),
            "mean_rd_utility": mean_rd_utility,
            "mean_psnr": mean_psnr,
            "mean_bpp": mean_bpp,
        }
        setting_records.append(setting_record)
        print(
            f"COMPRESS_SETTING setting={setting} qualities={len(per_quality)} "
            f"images={len(names)} cases={len(names) * len(per_quality)} "
            f"mean_rd_utility={mean_rd_utility:.9f} mean_psnr={mean_psnr:.9f} mean_bpp={mean_bpp:.12f}",
            flush=True,
        )

    cases_sha = _canonical_sha(cases)
    checkpoints_sha = _canonical_sha(model_records)
    settings_sha = _canonical_sha(setting_records)
    elapsed = time.monotonic() - t0
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("invalid elapsed time")
    print(
        f"COMPRESS_FINAL protocol={PROTOCOL_ID} family={family} images={EXPECTED_IMAGES} "
        f"qualities={len(QUALITIES)} cases={len(cases)} streams={total_streams} "
        f"cases_sha={cases_sha} checkpoints_sha={checkpoints_sha} "
        f"settings_sha={settings_sha} elapsed={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
