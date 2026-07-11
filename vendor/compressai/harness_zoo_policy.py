#!/usr/bin/env python3
"""Strict Kodak-24 evaluation for static policies over pinned zoo codecs."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import torch

_HELPER_PATH = Path(__file__).resolve().with_name("harness_zoo_entropy.py")
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "compressai_zoo_entropy_helper", _HELPER_PATH
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError("pinned CompressAI zoo helper is unavailable")
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER)

EXPECTED_CASES = _HELPER.EXPECTED_CASES
EXPECTED_IMAGES = _HELPER.EXPECTED_IMAGES
FAMILIES = _HELPER.FAMILIES
ASSET_PROTOCOL_ID = _HELPER.PROTOCOL_ID
QUALITIES = _HELPER.QUALITIES
SETTINGS = _HELPER.SETTINGS
_block_url_loading = _HELPER._block_url_loading
_canonical_sha = _HELPER._canonical_sha
_encode_decode = _HELPER._encode_decode
_load_image = _HELPER._load_image
_load_model = _HELPER._load_model
_validate_inputs = _HELPER._validate_inputs
_validate_protocol = _HELPER._validate_protocol
_validate_runtime = _HELPER._validate_runtime


PROTOCOL_ID = "compressai_zoo_kodak24_q1q8_policy_v1"
POLICY_MODES = ("global", "content", "quality")
GROUP_ORDER = ("low", "mid", "high")
FAMILY_ORDER = tuple(FAMILIES)
QUALITY_BANDS = {
    "low": (1, 2, 3),
    "mid": (4, 5, 6),
    "high": (7, 8),
}
TARGET_BPP = {
    1: 0.15,
    2: 0.25,
    3: 0.35,
    4: 0.50,
    5: 0.70,
    6: 0.90,
    7: 1.20,
    8: 1.50,
}
TARGET_PENALTY = 12.0


def _validate_identifier(value: str, label: str) -> str:
    if not value or not value.replace("_", "a").isalnum() or value[0].isdigit():
        raise ValueError(f"invalid {label}")
    return value


def _load_surface(path: Path, surface_name: str, mode: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    allowed_module_nodes = (ast.Expr, ast.ImportFrom, ast.FunctionDef)
    if any(not isinstance(node, allowed_module_nodes) for node in tree.body):
        raise TypeError("solution may contain only its docstring, future import, and policy function")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise TypeError("solution must define exactly one policy function")
    fn = functions[0]
    if (
        fn.name != surface_name
        or fn.args.args
        or fn.args.posonlyargs
        or fn.args.kwonlyargs
        or fn.args.vararg is not None
        or fn.args.kwarg is not None
        or fn.decorator_list
        or len(fn.body) != 1
        or not isinstance(fn.body[0], ast.Return)
    ):
        raise TypeError(f"{surface_name} must be a zero-argument literal-return function")
    value = ast.literal_eval(fn.body[0].value)
    if mode == "global":
        if not isinstance(value, str) or value not in FAMILIES:
            raise ValueError(f"{surface_name} must return one pinned family")
        return value
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{surface_name} must return a three-family tuple")
    if any(not isinstance(item, str) or item not in FAMILIES for item in value):
        raise ValueError(f"{surface_name} contains an unknown family")
    return value


def _quality_band(quality: int) -> str:
    for band, qualities in QUALITY_BANDS.items():
        if quality in qualities:
            return band
    raise ValueError(f"quality outside the fixed inventory: {quality}")


def _select_family(choice, mode: str, quality: int, group: str) -> str:
    if mode == "global":
        return choice
    if mode == "content":
        return choice[GROUP_ORDER.index(group)]
    if mode == "quality":
        return choice[GROUP_ORDER.index(_quality_band(quality))]
    raise ValueError(f"unsupported policy mode: {mode}")


def _mean(records: list[dict], key: str) -> float:
    if not records:
        raise RuntimeError(f"cannot aggregate empty {key} records")
    return sum(float(record[key]) for record in records) / len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--surface-name", required=True)
    parser.add_argument("--mode", choices=POLICY_MODES, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()

    surface_name = _validate_identifier(args.surface_name, "surface name")
    if len(args.protocol_sha256) != 64:
        raise ValueError("invalid expected protocol digest")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the protocol requires exactly one visible CUDA GPU")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    started = time.monotonic()

    protocol = _validate_protocol(Path(args.protocol).resolve(), args.protocol_sha256)
    _validate_runtime(protocol)
    import compressai

    compressai.set_entropy_coder("ans")
    data_root = Path(args.data_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    _validate_inputs(protocol, data_root, checkpoint_root)
    choice = _load_surface(Path(args.solution).resolve(), surface_name, args.mode)
    _block_url_loading()

    image_records = protocol["dataset"]["files"]
    group_by_image = {
        name: group
        for group, names in protocol["dataset"]["groups"].items()
        for name in names
    }
    dispatch = [
        {
            "quality": quality,
            "image": image_record["name"],
            "group": group_by_image[image_record["name"]],
            "family": _select_family(
                choice,
                args.mode,
                quality,
                group_by_image[image_record["name"]],
            ),
        }
        for quality in QUALITIES
        for image_record in image_records
    ]
    if len(dispatch) != EXPECTED_CASES:
        raise RuntimeError("policy dispatch did not cover the complete matrix")
    choice_record = {"mode": args.mode, "surface": surface_name, "choice": choice}
    choice_sha = _canonical_sha(choice_record)
    dispatch_sha = _canonical_sha(dispatch)
    target_sha = _canonical_sha(TARGET_BPP)
    print(
        f"COMPRESS_PROTOCOL protocol={PROTOCOL_ID} asset_protocol={ASSET_PROTOCOL_ID} "
        f"mode={args.mode} surface={surface_name} images={EXPECTED_IMAGES} "
        f"qualities={len(QUALITIES)} cases={EXPECTED_CASES} "
        f"protocol_sha={args.protocol_sha256} choice_sha={choice_sha} "
        f"dispatch_sha={dispatch_sha} target_sha={target_sha} network=blocked",
        flush=True,
    )

    dispatch_by_case = {
        (record["quality"], record["image"]): record["family"]
        for record in dispatch
    }
    models = []
    model_by_key = {}
    cases = []
    total_streams = 0
    for quality in QUALITIES:
        case_by_image = {}
        selected_families = {
            dispatch_by_case[(quality, image_record["name"])]
            for image_record in image_records
        }
        for family in FAMILY_ORDER:
            if family not in selected_families:
                continue
            checkpoint_record = protocol["families"][family]["checkpoints"][quality - 1]
            if int(checkpoint_record["quality"]) != quality:
                raise RuntimeError("checkpoint inventory is out of order")
            model, updated, cdf_sha = _load_model(
                FAMILIES[family],
                checkpoint_root / checkpoint_record["file"],
                device,
            )
            params = sum(parameter.numel() for parameter in model.parameters())
            model_record = {
                "family": family,
                "quality": quality,
                "checkpoint_sha": checkpoint_record["sha256"],
                "cdf_sha": cdf_sha,
                "params": params,
                "updated": updated,
            }
            models.append(model_record)
            model_by_key[(family, quality)] = model_record
            print(
                f"COMPRESS_MODEL family={family} quality={quality} "
                f"checkpoint_sha={checkpoint_record['sha256']} cdf_sha={cdf_sha} "
                f"params={params} updated={int(updated)}",
                flush=True,
            )
            for image_record in image_records:
                name = image_record["name"]
                if dispatch_by_case[(quality, name)] != family:
                    continue
                source = _load_image(data_root / name)
                nbytes, nstreams, pixels, bpp, psnr, recon_sha = _encode_decode(
                    model, source, device
                )
                bpp = float(f"{bpp:.12f}")
                psnr = float(f"{psnr:.9f}")
                case_by_image[name] = {
                    "family": family,
                    "quality": quality,
                    "image": name,
                    "group": group_by_image[name],
                    "checkpoint_sha": checkpoint_record["sha256"],
                    "params": params,
                    "streams": nstreams,
                    "pixels": pixels,
                    "bytes": nbytes,
                    "bpp": bpp,
                    "psnr": psnr,
                    "recon_sha": recon_sha,
                }
            del model
            torch.cuda.empty_cache()
        if set(case_by_image) != {record["name"] for record in image_records}:
            raise RuntimeError(f"incomplete encode/decode inventory for quality {quality}")
        for image_record in image_records:
            case = case_by_image[image_record["name"]]
            cases.append(case)
            total_streams += int(case["streams"])
            print(
                f"COMPRESS_CASE family={case['family']} quality={quality} "
                f"image={case['image']} group={case['group']} "
                f"checkpoint_sha={case['checkpoint_sha']} params={case['params']} "
                f"streams={case['streams']} pixels={case['pixels']} bytes={case['bytes']} "
                f"bpp={case['bpp']:.12f} psnr={case['psnr']:.9f} "
                f"recon_sha={case['recon_sha']}",
                flush=True,
            )

    if len(cases) != EXPECTED_CASES:
        raise RuntimeError("incomplete encode/decode case matrix")
    rate_records = []
    setting_records = []
    for setting in SETTINGS:
        names = (
            {record["name"] for record in image_records}
            if setting == "full"
            else set(protocol["dataset"]["groups"][setting])
        )
        per_quality = []
        for quality in QUALITIES:
            selected = [
                case
                for case in cases
                if case["quality"] == quality and case["image"] in names
            ]
            if len(selected) != len(names):
                raise RuntimeError(f"incomplete {setting} quality-{quality} aggregate")
            total_bytes = sum(int(case["bytes"]) for case in selected)
            total_pixels = sum(int(case["pixels"]) for case in selected)
            bpp = 8.0 * total_bytes / total_pixels
            psnr = _mean(selected, "psnr")
            record = {
                "setting": setting,
                "quality": quality,
                "count": len(selected),
                "pixels": total_pixels,
                "bytes": total_bytes,
                "bpp": float(f"{bpp:.12f}"),
                "psnr": float(f"{psnr:.9f}"),
                "rd6": float(f"{psnr - 6.0 * bpp:.9f}"),
                "rd12": float(f"{psnr - 12.0 * bpp:.9f}"),
                "rd18": float(f"{psnr - 18.0 * bpp:.9f}"),
                "target_utility": float(
                    f"{psnr - TARGET_PENALTY * abs(bpp - TARGET_BPP[quality]):.9f}"
                ),
                "mean_params": float(f"{_mean(selected, 'params'):.3f}"),
                "mean_streams": float(f"{_mean(selected, 'streams'):.6f}"),
            }
            per_quality.append(record)
            rate_records.append(record)
            print(
                f"COMPRESS_RATE setting={setting} quality={quality} count={record['count']} "
                f"pixels={record['pixels']} bytes={record['bytes']} bpp={record['bpp']:.12f} "
                f"psnr={record['psnr']:.9f} rd6={record['rd6']:.9f} "
                f"rd12={record['rd12']:.9f} rd18={record['rd18']:.9f} "
                f"target_utility={record['target_utility']:.9f} "
                f"mean_params={record['mean_params']:.3f} "
                f"mean_streams={record['mean_streams']:.6f}",
                flush=True,
            )

        def band_mean(band: str) -> float:
            records = [record for record in per_quality if record["quality"] in QUALITY_BANDS[band]]
            return _mean(records, "rd12")

        setting_record = {
            "setting": setting,
            "qualities": len(per_quality),
            "images": len(names),
            "cases": len(names) * len(per_quality),
            "rd6": float(f"{_mean(per_quality, 'rd6'):.9f}"),
            "rd12": float(f"{_mean(per_quality, 'rd12'):.9f}"),
            "rd18": float(f"{_mean(per_quality, 'rd18'):.9f}"),
            "lowq_rd12": float(f"{band_mean('low'):.9f}"),
            "midq_rd12": float(f"{band_mean('mid'):.9f}"),
            "highq_rd12": float(f"{band_mean('high'):.9f}"),
            "target_utility": float(f"{_mean(per_quality, 'target_utility'):.9f}"),
            "mean_psnr": float(f"{_mean(per_quality, 'psnr'):.9f}"),
            "mean_bpp": float(f"{_mean(per_quality, 'bpp'):.12f}"),
            "mean_params": float(f"{_mean(per_quality, 'mean_params'):.3f}"),
            "mean_streams": float(f"{_mean(per_quality, 'mean_streams'):.6f}"),
        }
        setting_records.append(setting_record)
        print(
            f"COMPRESS_SETTING setting={setting} qualities={setting_record['qualities']} "
            f"images={setting_record['images']} cases={setting_record['cases']} "
            f"rd6={setting_record['rd6']:.9f} rd12={setting_record['rd12']:.9f} "
            f"rd18={setting_record['rd18']:.9f} "
            f"lowq_rd12={setting_record['lowq_rd12']:.9f} "
            f"midq_rd12={setting_record['midq_rd12']:.9f} "
            f"highq_rd12={setting_record['highq_rd12']:.9f} "
            f"target_utility={setting_record['target_utility']:.9f} "
            f"mean_psnr={setting_record['mean_psnr']:.9f} "
            f"mean_bpp={setting_record['mean_bpp']:.12f} "
            f"mean_params={setting_record['mean_params']:.3f} "
            f"mean_streams={setting_record['mean_streams']:.6f}",
            flush=True,
        )

    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("invalid elapsed time")
    print(
        f"COMPRESS_FINAL protocol={PROTOCOL_ID} mode={args.mode} surface={surface_name} "
        f"images={EXPECTED_IMAGES} qualities={len(QUALITIES)} cases={len(cases)} "
        f"models={len(models)} streams={total_streams} choice_sha={choice_sha} "
        f"dispatch_sha={dispatch_sha} cases_sha={_canonical_sha(cases)} "
        f"models_sha={_canonical_sha(models)} rates_sha={_canonical_sha(rate_records)} "
        f"settings_sha={_canonical_sha(setting_records)} elapsed={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
