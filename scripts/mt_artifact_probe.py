#!/usr/bin/env python3
"""Probe pinned machine-translation artifacts without running translation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import socket
from unittest import mock


DIRECTIONS = {
    "de_en": "opus-mt-de-en",
    "fr_en": "opus-mt-fr-en",
    "ru_en": "opus-mt-ru-en",
}
CHECKPOINT_NAMES = {"model.safetensors", "pytorch_model.bin"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_common(path: Path):
    spec = importlib.util.spec_from_file_location("mt_probe_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MT common module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inventory(model_dir: Path) -> list[dict]:
    records = []
    for path in sorted(model_dir.iterdir(), key=lambda item: item.name):
        record = {"path": path.name, "symlink": path.is_symlink()}
        if path.is_file() and not path.is_symlink():
            record.update({
                "kind": "file",
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            })
        elif path.is_dir() and not path.is_symlink():
            record["kind"] = "directory"
        else:
            record["kind"] = "other"
        records.append(record)
    return records


def _tied_aliases(model) -> dict[str, bool]:
    shared = model.model.shared.weight
    return {
        "encoder_embedding_is_shared": model.model.encoder.embed_tokens.weight is shared,
        "decoder_embedding_is_shared": model.model.decoder.embed_tokens.weight is shared,
        "lm_head_is_shared": model.lm_head.weight is shared,
    }


def _probe_data(data_root: Path, common) -> dict:
    os.environ["MT_DATA"] = str(data_root)
    expected_names = {
        split_name for _model_dir, split_name in common.DIRECTIONS.values()
    } | {"source_manifest.json"}
    actual_names = {path.name for path in data_root.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError(
            f"data inventory mismatch: expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}"
        )
    if any(not path.is_file() or path.is_symlink() for path in data_root.iterdir()):
        raise RuntimeError("data inventory contains a non-regular file")
    records = {}
    for direction in common.DIRECTIONS:
        sources, references, proof = common.load_dataset(direction)
        if (
            len(sources) != common.OFFICIAL_TEST_PAIRS
            or len(references) != common.OFFICIAL_TEST_PAIRS
        ):
            raise RuntimeError(f"data row mismatch for {direction}")
        records[direction] = proof
    manifest_path = data_root / "source_manifest.json"
    return {
        "top_level_inventory": [
            {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
                "symlink": path.is_symlink(),
            }
            for path in sorted(data_root.iterdir(), key=lambda item: item.name)
        ],
        "source_manifest_sha256": _sha256(manifest_path),
        "directions": records,
    }


def _probe_direction(
    direction: str, model_dir: Path, common, allow_inventory_mismatch: bool
) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import transformers.modeling_utils as modeling_utils

    if not model_dir.is_dir() or model_dir.is_symlink():
        raise FileNotFoundError(f"missing regular model directory: {model_dir}")
    inventory = _inventory(model_dir)
    checkpoint_candidates = sorted(
        record["path"]
        for record in inventory
        if record["kind"] == "file" and record["path"] in CHECKPOINT_NAMES
    )
    if not checkpoint_candidates:
        raise RuntimeError(
            f"expected at least one checkpoint for {direction}, found none"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    loaded_checkpoint_paths: list[str] = []
    original_load_state_dict = modeling_utils.load_state_dict

    def recording_load_state_dict(checkpoint_file, *args, **kwargs):
        loaded_checkpoint_paths.append(str(checkpoint_file))
        return original_load_state_dict(checkpoint_file, *args, **kwargs)

    with mock.patch.object(
        modeling_utils, "load_state_dict", recording_load_state_dict
    ):
        model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
            output_loading_info=True,
        )
    selected_checkpoints = sorted({
        Path(path).name
        for path in loaded_checkpoint_paths
        if Path(path).name in CHECKPOINT_NAMES
    })
    if len(selected_checkpoints) != 1:
        raise RuntimeError(
            f"could not bind one loaded checkpoint for {direction}: "
            f"candidates={checkpoint_candidates}, loader_paths={loaded_checkpoint_paths}"
        )
    checkpoint = selected_checkpoints[0]
    loading = {
        key: list(loading_info.get(key) or ())
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(loading.values()):
        raise RuntimeError(f"checkpoint loading errors for {direction}: {loading}")

    parameters = list(model.parameters())
    parameter_numel = sum(parameter.numel() for parameter in parameters)
    final_bias = getattr(model, "final_logits_bias", None)
    final_bias_numel = final_bias.numel() if final_bias is not None else 0
    state_dict = model.state_dict()
    spec = common.MODEL_SPECS[direction]
    expected_files = {
        name: {"size": size, "sha256": digest}
        for name, (size, digest) in spec["files"].items()
    }
    expected_manifest = common._canonical_json_bytes(
        common.expected_model_manifest(direction)
    )
    expected_files["model_manifest.json"] = {
        "size": len(expected_manifest),
        "sha256": hashlib.sha256(expected_manifest).hexdigest(),
    }
    actual_files = {
        record["path"]: {
            "size": record["size"],
            "sha256": record["sha256"],
        }
        for record in inventory
        if record["kind"] == "file"
    }
    inventory_matches = actual_files == expected_files
    if not allow_inventory_mismatch and not inventory_matches:
        raise RuntimeError(
            f"runtime inventory mismatch for {direction}: "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    if any(record["symlink"] for record in inventory):
        raise RuntimeError(f"non-regular top-level artifact for {direction}: {inventory}")
    if (
        not allow_inventory_mismatch
        and any(record["kind"] != "file" for record in inventory)
    ):
        raise RuntimeError(f"non-file top-level artifact for {direction}: {inventory}")
    if checkpoint != spec["checkpoint_file"]:
        raise RuntimeError(
            f"Transformers selected {checkpoint} for {direction}, expected "
            f"{spec['checkpoint_file']}"
        )
    if type(model).__name__ != "MarianMTModel":
        raise TypeError(f"unexpected model class for {direction}: {type(model)}")
    if type(tokenizer).__name__ != "MarianTokenizer":
        raise TypeError(f"unexpected tokenizer class for {direction}: {type(tokenizer)}")
    if int(model.config.vocab_size) != spec["vocab_size"]:
        raise RuntimeError(f"model vocab mismatch for {direction}")
    if len(tokenizer) != spec["vocab_size"]:
        raise RuntimeError(f"tokenizer vocab mismatch for {direction}")
    if parameter_numel != spec["parameter_count"]:
        raise RuntimeError(
            f"parameter count mismatch for {direction}: "
            f"expected={spec['parameter_count']}, actual={parameter_numel}"
        )
    if final_bias is None:
        raise RuntimeError(f"missing final_logits_bias for {direction}")
    if parameter_numel + final_bias_numel != spec["checkpoint_tensor_elements"]:
        raise RuntimeError(
            f"parameter+bias count mismatch for {direction}: expected="
            f"{spec['checkpoint_tensor_elements']}, "
            f"actual={parameter_numel + final_bias_numel}"
        )
    tied_aliases = _tied_aliases(model)
    if not all(tied_aliases.values()):
        raise RuntimeError(f"unexpected embedding aliases for {direction}: {tied_aliases}")
    result = {
        "direction": direction,
        "model_dir": str(model_dir),
        "repository": spec["repository"],
        "revision": spec["revision"],
        "top_level_inventory": inventory,
        "checkpoint_candidates": checkpoint_candidates,
        "loader_checkpoint_paths": loaded_checkpoint_paths,
        "expected_runtime_files_match": inventory_matches,
        "selected_checkpoint": checkpoint,
        "selected_checkpoint_sha256": actual_files[checkpoint]["sha256"],
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "config_class": f"{type(model.config).__module__}.{type(model.config).__name__}",
        "vocab_size": int(model.config.vocab_size),
        "tokenizer_length": len(tokenizer),
        "parameter_tensor_count": len(parameters),
        "parameter_numel": parameter_numel,
        "final_logits_bias": None if final_bias is None else {
            "shape": list(final_bias.shape),
            "dtype": str(final_bias.dtype),
            "numel": final_bias_numel,
        },
        "parameter_plus_final_logits_bias_numel": parameter_numel + final_bias_numel,
        "state_dict_tensor_count": len(state_dict),
        "state_dict_numel_with_aliases": sum(tensor.numel() for tensor in state_dict.values()),
        "tied_aliases": tied_aliases,
        "loading_info": loading,
        "expected_parameter_numel": spec["parameter_count"],
        "expected_parameter_plus_bias_numel": spec["checkpoint_tensor_elements"],
    }
    del state_dict, parameters, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--common", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--allow-inventory-mismatch", action="store_true")
    args = parser.parse_args()

    image_match = re.fullmatch(r".+@sha256:([0-9a-f]{64})", args.image_ref)
    if image_match is None:
        raise ValueError("image-ref must be pinned by an exact sha256 digest")

    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    import torch
    import transformers

    common = _load_common(args.common)
    result = {
        "schema_version": 1,
        "image_ref": args.image_ref,
        "image_digest": f"sha256:{image_match.group(1)}",
        "worker": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "models": {},
        "data": _probe_data(args.data_root, common),
    }
    for direction, directory in DIRECTIONS.items():
        result["models"][direction] = _probe_direction(
            direction,
            args.models_root / directory,
            common,
            args.allow_inventory_mismatch,
        )
        probe = result["models"][direction]
        print(
            "MT_ARTIFACT_PROBE "
            f"direction={direction} checkpoint={probe['selected_checkpoint']} "
            f"parameters={probe['parameter_numel']} "
            f"final_logits_bias={probe['final_logits_bias']['numel']} "
            f"tensor_elements={probe['parameter_plus_final_logits_bias_numel']}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"MT_ARTIFACT_PROBE_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
