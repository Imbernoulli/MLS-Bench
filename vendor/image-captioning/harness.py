#!/usr/bin/env python3
"""Fail-closed official Flickr8k ClipCap-style evaluation harness.

The editable surface is a literal ``CONFIG = {...}`` assignment.  This verifier
parses it with :mod:`ast` and never imports or executes agent-authored Python.
Training, decoding, data validation, and metric computation must all complete
before the single versioned result record is emitted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROTOCOL = "flickr8k_official_v1"
TRAIN_IMAGES = 6000
EVAL_IMAGES = 1000
REFS_PER_IMAGE = 5
TRAIN_PAIRS = TRAIN_IMAGES * REFS_PER_IMAGE
EPOCHS = 10
BATCH_SIZE = 40
PREFIX_LENGTH = 10
MAX_CAPTION_TOKENS = 40
EXPECTED_STEPS = EPOCHS * (TRAIN_PAIRS // BATCH_SIZE)
CLIP_DIM = 512
GPT_DIM = 768
GPT_LAYERS = 12
GPT_HEADS = 12
GPT_VOCAB = 50257
MAX_CONFIG_BYTES = 32 * 1024
MAX_CONFIG_AST_NODES = 256
RESULT_PREFIX = "CAPTION_RESULT"
MANIFEST_SCHEMA_VERSION = 3
CANONICAL_ARCHIVE_URL = (
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
)
CANONICAL_ARCHIVE_SHA256 = (
    "4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817"
)
CANONICAL_JSON_MEMBER = "dataset_flickr8k.json"
CANONICAL_JSON_SHA256 = (
    "ce467057af54e8a8b7078fa6000c15cb3605dbfb36c3cc6a202cca90e8a9741e"
)
CANONICAL_FILENAME_SET_SHA256 = {
    "train": "fbb334d8b4d4bab05a65950cb0b8123079c40ba8d1c38d8aa360fa27459e8cf4",
    "test": "25d2fec0836bb4728d4672c46a5694dfbdb953a2ff5ba146f5ffaa7062512489",
}
GPT2_FILE_SHA256 = {
    "model.safetensors": "c7d00560d8910fbed77ffad4065dee5011c41ba401b1064e749c498ba9e20373",
    "config.json": "50fda00afcbf90d2a7655c764fd8879f6ce8bed5624ff8231cae8889a7983cd4",
    "tokenizer.json": "1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e",
}
CLIP_CHECKPOINT_SHA256 = (
    "1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad"
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _normalise_caption(caption: str) -> str:
    return " ".join(caption.strip().split())


def _caption_signature_sha256(captions: list[str]) -> str:
    return _json_sha256(sorted(_normalise_caption(caption) for caption in captions))


def _filename_set_sha256(filenames) -> str:
    return _json_sha256(sorted(filenames))


def _source_split_sha256(rows: list[dict]) -> str:
    return _json_sha256(
        [
            {
                "source_filename": row["source_filename"],
                "decoded_rgb_sha256": row["decoded_rgb_sha256"],
                "captions_sha256": row["captions_sha256"],
                "canonical_captions_sha256": row[
                    "canonical_captions_sha256"
                ],
            }
            for row in rows
        ]
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_keys(config: dict, expected: set[str], label: str) -> None:
    observed = set(config)
    if observed != expected:
        raise ValueError(
            f"{label} CONFIG keys must be exactly {sorted(expected)}, got {sorted(observed)}"
        )


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return result


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _choice(value: Any, name: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return value


def load_literal_config(path: Path, mode: str) -> dict:
    """Parse a single literal CONFIG assignment without executing the module."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError(f"CONFIG file exceeds {MAX_CONFIG_BYTES} bytes")
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("CONFIG file is not valid static Python syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_CONFIG_AST_NODES:
        raise ValueError("CONFIG syntax tree is too large")

    assignment = None
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and all(alias.name == "annotations" for alias in node.names)
        ):
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "CONFIG"
            and assignment is None
        ):
            assignment = node.value
            continue
        raise ValueError(
            "editable file may contain only a docstring, future annotations, and "
            "one literal CONFIG assignment"
        )
    if assignment is None:
        raise ValueError("missing literal CONFIG assignment")
    try:
        config = ast.literal_eval(assignment)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("CONFIG must contain literals only") from exc
    if not isinstance(config, dict) or not config:
        raise TypeError("CONFIG must be a non-empty dictionary")
    if not all(isinstance(key, str) for key in config):
        raise TypeError("every CONFIG key must be a string")
    return validate_config(mode, config)


def validate_config(mode: str, config: dict) -> dict:
    """Validate and normalize the finite design space for one research surface."""
    result = dict(config)
    if mode == "mapping":
        kind = _choice(result.get("type"), "type", {"linear", "mlp", "transformer"})
        if kind == "linear":
            _require_keys(result, {"type"}, mode)
        elif kind == "mlp":
            _require_keys(result, {"type", "hidden_ratio", "activation", "dropout"}, mode)
            result["hidden_ratio"] = _number(result["hidden_ratio"], "hidden_ratio", 0.25, 2.0)
            result["activation"] = _choice(
                result["activation"], "activation", {"tanh", "gelu", "relu"}
            )
            result["dropout"] = _number(result["dropout"], "dropout", 0.0, 0.5)
        else:
            _require_keys(
                result, {"type", "layers", "heads", "clip_tokens", "dropout"}, mode
            )
            result["layers"] = _integer(result["layers"], "layers", 1, 8)
            result["heads"] = _integer(result["heads"], "heads", 1, GPT_HEADS)
            if GPT_DIM % result["heads"]:
                raise ValueError("heads must divide the GPT embedding dimension")
            result["clip_tokens"] = _integer(result["clip_tokens"], "clip_tokens", 1, 16)
            result["dropout"] = _number(result["dropout"], "dropout", 0.0, 0.5)
    elif mode == "decoding":
        strategy = _choice(result.get("strategy"), "strategy", {"greedy", "beam", "sample"})
        common = {"strategy", "max_length", "min_length", "no_repeat_ngram"}
        if strategy == "beam":
            _require_keys(result, common | {"beam_size", "length_penalty"}, mode)
            result["beam_size"] = _integer(result["beam_size"], "beam_size", 2, 8)
            result["length_penalty"] = _number(
                result["length_penalty"], "length_penalty", 0.0, 2.0
            )
        elif strategy == "sample":
            _require_keys(result, common | {"temperature", "top_p"}, mode)
            result["temperature"] = _number(
                result["temperature"], "temperature", 0.2, 2.0
            )
            result["top_p"] = _number(result["top_p"], "top_p", 0.5, 1.0)
        else:
            _require_keys(result, common, mode)
        result["max_length"] = _integer(result["max_length"], "max_length", 10, 40)
        result["min_length"] = _integer(result["min_length"], "min_length", 0, 10)
        if result["min_length"] >= result["max_length"]:
            raise ValueError("min_length must be smaller than max_length")
        result["no_repeat_ngram"] = _integer(
            result["no_repeat_ngram"], "no_repeat_ngram", 0, 4
        )
    elif mode == "objective":
        _require_keys(result, {"label_smoothing"}, mode)
        result["label_smoothing"] = _number(
            result["label_smoothing"], "label_smoothing", 0.0, 0.3
        )
    elif mode == "featureprep":
        _require_keys(result, {"normalization"}, mode)
        result["normalization"] = _choice(
            result["normalization"], "normalization", {"none", "l2", "standardize"}
        )
    elif mode == "init":
        _require_keys(result, {"scheme"}, mode)
        result["scheme"] = _choice(
            result["scheme"],
            "scheme",
            {"pytorch_default", "xavier_uniform", "kaiming_uniform", "caption_mean"},
        )
    elif mode == "sampling":
        _require_keys(result, {"strategy"}, mode)
        result["strategy"] = _choice(
            result["strategy"], "strategy", {"uniform", "length_bucketed"}
        )
    elif mode == "optimizer":
        name = _choice(result.get("name"), "name", {"adamw", "sgd"})
        common = {"name", "learning_rate", "weight_decay", "schedule", "warmup_steps"}
        if name == "sgd":
            _require_keys(result, common | {"momentum"}, mode)
            result["momentum"] = _number(result["momentum"], "momentum", 0.0, 0.99)
        else:
            _require_keys(result, common, mode)
        result["learning_rate"] = _number(
            result["learning_rate"], "learning_rate", 1e-6, 5e-2
        )
        result["weight_decay"] = _number(
            result["weight_decay"], "weight_decay", 0.0, 0.2
        )
        result["schedule"] = _choice(
            result["schedule"], "schedule", {"constant", "cosine", "warmup_cosine"}
        )
        result["warmup_steps"] = _integer(
            result["warmup_steps"], "warmup_steps", 0, EXPECTED_STEPS - 1
        )
        if result["schedule"] != "warmup_cosine" and result["warmup_steps"] != 0:
            raise ValueError("warmup_steps must be zero unless schedule is warmup_cosine")
    elif mode == "prompt":
        _require_keys(result, {"prefix", "lowercase", "strip_terminal_period"}, mode)
        result["prefix"] = _choice(
            result["prefix"], "prefix", {"", "a photo of ", "an image of "}
        )
        for key in ("lowercase", "strip_terminal_period"):
            if not isinstance(result[key], bool):
                raise TypeError(f"{key} must be boolean")
    elif mode == "augment":
        _require_keys(result, {"gaussian_std", "dropout_probability"}, mode)
        result["gaussian_std"] = _number(
            result["gaussian_std"], "gaussian_std", 0.0, 0.1
        )
        result["dropout_probability"] = _number(
            result["dropout_probability"], "dropout_probability", 0.0, 0.5
        )
    elif mode == "weighting":
        scheme = _choice(result.get("scheme"), "scheme", {"uniform", "idf"})
        if scheme == "uniform":
            _require_keys(result, {"scheme"}, mode)
        else:
            _require_keys(result, {"scheme", "idf_power", "idf_cap"}, mode)
            result["idf_power"] = _number(result["idf_power"], "idf_power", 0.1, 2.0)
            result["idf_cap"] = _number(result["idf_cap"], "idf_cap", 1.0, 10.0)
    else:
        raise ValueError(f"unknown caption mode {mode!r}")
    return result


def _validate_manifest(data_root: Path, gpt_dir: Path) -> tuple[dict, str]:
    manifest_path = data_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "dataset_repo": "jxie/flickr8k",
        "dataset_revision": "56f58c967835f7c508d684f36bd7897cca9d7634",
        "train_split": "train",
        "eval_split": "test",
        "train_count": TRAIN_IMAGES,
        "eval_count": EVAL_IMAGES,
        "references_per_image": REFS_PER_IMAGE,
        "gpt2_repo": "openai-community/gpt2",
        "gpt2_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "gpt2_file_sha256": GPT2_FILE_SHA256,
        "clip_model": "ViT-B-32",
        "clip_pretrained_tag": "laion2b_s34b_b79k",
        "clip_repo": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "clip_revision": "1a25a446712ba5ee05982a381eed697ef9b435cf",
        "clip_checkpoint": "open_clip_pytorch_model.bin",
        "clip_checkpoint_sha256": CLIP_CHECKPOINT_SHA256,
        "canonical_archive_url": CANONICAL_ARCHIVE_URL,
        "canonical_archive_sha256": CANONICAL_ARCHIVE_SHA256,
        "canonical_json_member": CANONICAL_JSON_MEMBER,
        "canonical_json_sha256": CANONICAL_JSON_SHA256,
        "canonical_filename_set_sha256": CANONICAL_FILENAME_SET_SHA256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("caption source manifest does not match the official protocol")
    selected = manifest.get("selected_sources")
    if not isinstance(selected, dict):
        raise RuntimeError("caption source manifest lacks selected_sources")
    for split, count in (("train", TRAIN_IMAGES), ("test", EVAL_IMAGES)):
        rows = selected.get(split)
        if not isinstance(rows, list) or len(rows) != count:
            raise RuntimeError(f"caption manifest {split} rows are incomplete")
        expected_keys = {
            "source_filename",
            "decoded_rgb_sha256",
            "captions_sha256",
            "canonical_captions_sha256",
        }
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected_keys:
                raise RuntimeError(f"caption manifest {split} row schema is invalid")
            filename = row.get("source_filename")
            if not isinstance(filename, str) or not filename.endswith(".jpg"):
                raise RuntimeError("caption manifest has an invalid canonical filename")
            for key in (
                "decoded_rgb_sha256",
                "captions_sha256",
                "canonical_captions_sha256",
            ):
                if not _is_sha256(row.get(key)):
                    raise RuntimeError(f"caption manifest has invalid {key}")
        filenames = [row["source_filename"] for row in rows]
        if len(set(filenames)) != count:
            raise RuntimeError(f"caption manifest {split} filenames are not unique")
        if _filename_set_sha256(filenames) != CANONICAL_FILENAME_SET_SHA256[split]:
            raise RuntimeError(
                f"caption manifest {split} filename set is not canonical Flickr8k"
            )
    if set(selected) != {"train", "test"}:
        raise RuntimeError("caption manifest contains unexpected source partitions")
    train_filenames = {row["source_filename"] for row in selected["train"]}
    eval_filenames = {row["source_filename"] for row in selected["test"]}
    if train_filenames & eval_filenames:
        raise RuntimeError("caption train and evaluation filenames overlap")
    train_hashes = {row["decoded_rgb_sha256"] for row in selected["train"]}
    eval_hashes = {row["decoded_rgb_sha256"] for row in selected["test"]}
    if train_hashes & eval_hashes:
        raise RuntimeError("caption train and evaluation image hashes overlap")
    split_hashes = manifest.get("split_sha256")
    recomputed_split_hashes = {
        split: _source_split_sha256(selected[split]) for split in ("train", "test")
    }
    if split_hashes != recomputed_split_hashes:
        raise RuntimeError("caption manifest split proof does not match its source rows")

    prepared = manifest.get("prepared_sha256")
    if not isinstance(prepared, dict):
        raise RuntimeError("caption manifest lacks prepared file hashes")
    data_files = ("train_clip.pt", "train_refs.json", "eval_clip.pt", "eval_refs.json")
    for relative in data_files:
        expected_hash = prepared.get(relative)
        path = data_root / relative
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise RuntimeError(f"caption data hash mismatch: {relative}")
    for filename, expected_hash in GPT2_FILE_SHA256.items():
        relative = f"gpt2/{filename}"
        path = gpt_dir / filename
        if prepared.get(relative) != expected_hash or _sha256(path) != expected_hash:
            raise RuntimeError(f"caption pinned GPT-2 hash mismatch: {relative}")
    for relative, expected_hash in prepared.items():
        if not relative.startswith("gpt2/"):
            continue
        path = gpt_dir / Path(relative).relative_to("gpt2")
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise RuntimeError(f"caption GPT-2 hash mismatch: {relative}")
    return manifest, _sha256(manifest_path)


def load_data(data_root: Path, gpt_dir: Path):
    manifest, manifest_hash = _validate_manifest(data_root, gpt_dir)
    train_emb = torch.load(
        data_root / "train_clip.pt", map_location="cpu", weights_only=True
    )
    eval_emb = torch.load(
        data_root / "eval_clip.pt", map_location="cpu", weights_only=True
    )
    train_refs = json.loads((data_root / "train_refs.json").read_text())
    eval_refs = json.loads((data_root / "eval_refs.json").read_text())
    if tuple(train_emb.shape) != (TRAIN_IMAGES, CLIP_DIM):
        raise RuntimeError(f"train embeddings have shape {tuple(train_emb.shape)}")
    if tuple(eval_emb.shape) != (EVAL_IMAGES, CLIP_DIM):
        raise RuntimeError(f"eval embeddings have shape {tuple(eval_emb.shape)}")
    for label, tensor in (("train", train_emb), ("eval", eval_emb)):
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise RuntimeError(f"{label} embeddings must be finite float32")
    for label, rows, count in (
        ("train", train_refs, TRAIN_IMAGES),
        ("eval", eval_refs, EVAL_IMAGES),
    ):
        if not isinstance(rows, list) or len(rows) != count:
            raise RuntimeError(f"{label} references are incomplete")
        if not all(
            isinstance(refs, list)
            and len(refs) == REFS_PER_IMAGE
            and all(isinstance(caption, str) and caption.strip() for caption in refs)
            for refs in rows
        ):
            raise RuntimeError(f"{label} references violate the five-caption protocol")
    for split, references in (("train", train_refs), ("test", eval_refs)):
        sources = manifest["selected_sources"][split]
        for index, (source, captions) in enumerate(zip(sources, references)):
            if source["captions_sha256"] != _json_sha256(captions):
                raise RuntimeError(
                    f"caption {split} references do not match source row {index}"
                )
            if source["canonical_captions_sha256"] != _caption_signature_sha256(
                captions
            ):
                raise RuntimeError(
                    f"caption {split} references lack canonical proof at row {index}"
                )
    return train_emb, train_refs, eval_emb, eval_refs, manifest, manifest_hash


class LinearMapping(nn.Module):
    def __init__(self, clip_dim: int, gpt_dim: int, prefix_len: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.gpt_dim = gpt_dim
        self.proj = nn.Linear(clip_dim, gpt_dim * prefix_len)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value).view(value.shape[0], self.prefix_len, self.gpt_dim)


class MLPMapping(nn.Module):
    def __init__(
        self,
        clip_dim: int,
        gpt_dim: int,
        prefix_len: int,
        *,
        hidden_ratio: float = 0.5,
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = max(128, int(gpt_dim * prefix_len * hidden_ratio))
        activations = {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}
        self.prefix_len = prefix_len
        self.gpt_dim = gpt_dim
        self.fc1 = nn.Linear(clip_dim, hidden)
        self.activation = activations[activation]()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, gpt_dim * prefix_len)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.fc1(value)))
        return self.fc2(hidden).view(value.shape[0], self.prefix_len, self.gpt_dim)


class TransformerMapping(nn.Module):
    def __init__(
        self,
        clip_dim: int,
        gpt_dim: int,
        prefix_len: int,
        *,
        layers: int,
        heads: int,
        clip_tokens: int,
        dropout: float,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.gpt_dim = gpt_dim
        self.clip_tokens = clip_tokens
        self.clip_projection = nn.Linear(clip_dim, clip_tokens * gpt_dim)
        self.prefix_tokens = nn.Parameter(torch.empty(prefix_len, gpt_dim))
        nn.init.normal_(self.prefix_tokens, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gpt_dim,
            nhead=heads,
            dim_feedforward=4 * gpt_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        clip_tokens = self.clip_projection(value).view(
            value.shape[0], self.clip_tokens, self.gpt_dim
        )
        prefix = self.prefix_tokens.unsqueeze(0).expand(value.shape[0], -1, -1)
        output = self.transformer(torch.cat([clip_tokens, prefix], dim=1))
        return output[:, self.clip_tokens :, :]


def build_mapping(config: dict | None = None) -> nn.Module:
    config = config or {
        "type": "mlp",
        "hidden_ratio": 0.5,
        "activation": "tanh",
        "dropout": 0.0,
    }
    kind = config["type"]
    if kind == "linear":
        return LinearMapping(CLIP_DIM, GPT_DIM, PREFIX_LENGTH)
    if kind == "mlp":
        return MLPMapping(
            CLIP_DIM,
            GPT_DIM,
            PREFIX_LENGTH,
            hidden_ratio=config["hidden_ratio"],
            activation=config["activation"],
            dropout=config["dropout"],
        )
    return TransformerMapping(
        CLIP_DIM,
        GPT_DIM,
        PREFIX_LENGTH,
        layers=config["layers"],
        heads=config["heads"],
        clip_tokens=config["clip_tokens"],
        dropout=config["dropout"],
    )


class CaptionModel(nn.Module):
    def __init__(self, gpt: nn.Module, mapping: nn.Module):
        super().__init__()
        self.gpt = gpt
        self.mapping = mapping
        self.embed = gpt.get_input_embeddings()

    def visual_prefix(self, clip_emb: torch.Tensor) -> torch.Tensor:
        prefix = self.mapping(clip_emb)
        expected = (clip_emb.shape[0], PREFIX_LENGTH, GPT_DIM)
        if not torch.is_tensor(prefix) or tuple(prefix.shape) != expected:
            raise RuntimeError(f"mapping returned {getattr(prefix, 'shape', None)}, expected {expected}")
        if not torch.isfinite(prefix).all():
            raise RuntimeError("mapping returned non-finite prefix embeddings")
        return prefix

    def forward(
        self, clip_emb: torch.Tensor, caption_ids: torch.Tensor, attention: torch.Tensor
    ) -> torch.Tensor:
        prefix = self.visual_prefix(clip_emb)
        caption_emb = self.embed(caption_ids)
        inputs = torch.cat([prefix, caption_emb], dim=1)
        prefix_mask = torch.ones(
            caption_ids.shape[0],
            PREFIX_LENGTH,
            dtype=attention.dtype,
            device=attention.device,
        )
        output = self.gpt(
            inputs_embeds=inputs,
            attention_mask=torch.cat([prefix_mask, attention], dim=1),
            use_cache=False,
        ).logits
        if not torch.isfinite(output).all():
            raise RuntimeError("GPT-2 produced non-finite training logits")
        return output[:, PREFIX_LENGTH - 1 : PREFIX_LENGTH - 1 + caption_ids.shape[1]]


def _apply_feature_prep(
    config: dict, train_emb: torch.Tensor, eval_emb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    kind = config["normalization"]
    if kind == "none":
        train_out, eval_out = train_emb, eval_emb
    elif kind == "l2":
        train_out = F.normalize(train_emb, p=2, dim=1)
        eval_out = F.normalize(eval_emb, p=2, dim=1)
    else:
        mean = train_emb.mean(dim=0, keepdim=True)
        std = train_emb.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        train_out = (train_emb - mean) / std
        eval_out = (eval_emb - mean) / std
    if not torch.isfinite(train_out).all() or not torch.isfinite(eval_out).all():
        raise RuntimeError("feature preprocessing produced non-finite embeddings")
    return train_out.contiguous(), eval_out.contiguous()


def _format_caption(caption: str, config: dict | None) -> str:
    body = caption.strip()
    if config is not None:
        if config["lowercase"]:
            body = body.lower()
        if config["strip_terminal_period"]:
            body = body.rstrip().removesuffix(".").rstrip()
        body = config["prefix"] + body
    if not body:
        raise RuntimeError("caption formatting produced an empty target")
    return " " + body


def _tokenize_pairs(tokenizer, train_refs: list[list[str]], prompt_config: dict | None):
    image_indices: list[int] = []
    sequences: list[list[int]] = []
    for image_index, refs in enumerate(train_refs):
        for caption in refs:
            ids = tokenizer.encode(
                _format_caption(caption, prompt_config), add_special_tokens=False
            )[: MAX_CAPTION_TOKENS - 1]
            if not ids:
                raise RuntimeError("tokenizer produced an empty caption")
            sequences.append(ids + [tokenizer.eos_token_id])
            image_indices.append(image_index)
    if len(sequences) != TRAIN_PAIRS:
        raise RuntimeError(f"tokenized {len(sequences)} pairs, expected {TRAIN_PAIRS}")
    return torch.tensor(image_indices, dtype=torch.long), sequences


def _make_batch(tokenizer, sequences: list[list[int]], pair_indices: torch.Tensor, device):
    selected = [sequences[int(index)] for index in pair_indices.tolist()]
    width = max(len(sequence) for sequence in selected)
    ids = torch.full(
        (len(selected), width), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention = torch.zeros_like(ids)
    for row, sequence in enumerate(selected):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        ids[row, : len(sequence)] = values
        attention[row, : len(sequence)] = 1
    return ids, attention


def _caption_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention: torch.Tensor,
    objective: dict | None,
    token_weights: torch.Tensor | None,
) -> torch.Tensor:
    label_smoothing = objective["label_smoothing"] if objective is not None else 0.0
    flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(targets)
    if attention.shape != targets.shape:
        raise RuntimeError("caption attention mask does not match its targets")
    mask = attention.float()
    if not torch.all((mask == 0) | (mask == 1)):
        raise RuntimeError("caption attention mask must be binary")
    weights = mask if token_weights is None else token_weights * mask
    denominator = weights.sum()
    if not torch.isfinite(denominator) or denominator.item() <= 0:
        raise RuntimeError("training loss has no finite positive token mass")
    loss = (flat * weights).sum() / denominator
    if loss.ndim != 0 or not torch.isfinite(loss) or not loss.requires_grad:
        raise RuntimeError("caption training loss is invalid")
    return loss


def _idf_lut(sequences: list[list[int]], vocab_size: int, device) -> torch.Tensor:
    document_frequency = Counter()
    for sequence in sequences:
        document_frequency.update(set(sequence))
    lut = torch.ones(vocab_size, dtype=torch.float32, device=device)
    for token, count in document_frequency.items():
        lut[token] = math.log((TRAIN_PAIRS + 1.0) / (count + 1.0)) + 1.0
    return lut


def _token_weight_batch(
    config: dict | None,
    targets: torch.Tensor,
    attention: torch.Tensor,
    idf: torch.Tensor | None,
) -> torch.Tensor | None:
    if config is None or config["scheme"] == "uniform":
        return None
    if idf is None:
        raise RuntimeError("IDF configuration is missing its fixed training statistics")
    weights = idf[targets].pow(config["idf_power"]).clamp(max=config["idf_cap"])
    weights = weights * attention
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise RuntimeError("token weighting produced invalid weights")
    return weights


def _apply_initialization(mapping: nn.Module, config: dict, mean_caption_embed: torch.Tensor):
    scheme = config["scheme"]
    if scheme == "pytorch_default":
        return
    linears = [module for module in mapping.modules() if isinstance(module, nn.Linear)]
    if not linears:
        raise RuntimeError("fixed mapping has no linear layers to initialize")
    with torch.no_grad():
        if scheme == "caption_mean":
            if not isinstance(mapping, MLPMapping):
                raise RuntimeError("caption_mean initialization requires the fixed MLP mapping")
            mapping.fc2.weight.zero_()
            mapping.fc2.bias.copy_(mean_caption_embed.repeat(PREFIX_LENGTH))
        else:
            for layer in linears:
                if scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    layer.bias.zero_()
    if any(not torch.isfinite(parameter).all() for parameter in mapping.parameters()):
        raise RuntimeError("mapping initialization produced non-finite parameters")


def _build_optimizer(mapping: nn.Module, config: dict | None):
    config = config or {
        "name": "adamw",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "schedule": "warmup_cosine",
        "warmup_steps": 500,
    }
    parameters = list(mapping.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("mapping parameter set is invalid")
    if config["name"] == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            parameters,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
    return optimizer, config


def _lr_multiplier(config: dict, step: int) -> float:
    schedule = config["schedule"]
    if schedule == "constant":
        return 1.0
    if schedule == "warmup_cosine" and step < config["warmup_steps"]:
        return (step + 1) / max(config["warmup_steps"], 1)
    offset = config["warmup_steps"] if schedule == "warmup_cosine" else 0
    progress = (step - offset + 1) / max(EXPECTED_STEPS - offset, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _epoch_order(
    sampling_config: dict | None,
    sequences: list[list[int]],
    seed: int,
    epoch: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * epoch)
    if sampling_config is None or sampling_config["strategy"] == "uniform":
        order = torch.randperm(TRAIN_PAIRS, generator=generator)
    else:
        # Randomize ties, group similarly sized targets into complete batches,
        # then shuffle the batch order. This is the usual length-bucketed
        # batching tradeoff while preserving exact once-per-epoch coverage.
        shuffled = torch.randperm(TRAIN_PAIRS, generator=generator).tolist()
        shuffled.sort(key=lambda index: len(sequences[index]))
        batches = [
            torch.tensor(shuffled[start : start + BATCH_SIZE], dtype=torch.long)
            for start in range(0, TRAIN_PAIRS, BATCH_SIZE)
        ]
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        order = torch.cat([batches[index] for index in batch_order])
    if not torch.equal(
        torch.sort(order).values, torch.arange(TRAIN_PAIRS, dtype=torch.long)
    ):
        raise RuntimeError("caption epoch order must cover every pair exactly once")
    return order


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_mapping(
    model: CaptionModel,
    tokenizer,
    train_emb: torch.Tensor,
    pair_image_indices: torch.Tensor,
    sequences: list[list[int]],
    *,
    mode: str,
    config: dict,
    device: torch.device,
    seed: int,
) -> int:
    objective = config if mode == "objective" else None
    sampling = config if mode == "sampling" else None
    optimizer_config = config if mode == "optimizer" else None
    augment = config if mode == "augment" else None
    weighting = config if mode == "weighting" else None
    idf = (
        _idf_lut(sequences, GPT_VOCAB, device)
        if weighting is not None and weighting["scheme"] == "idf"
        else None
    )
    optimizer, schedule = _build_optimizer(model.mapping, optimizer_config)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    step = 0
    model.gpt.eval()
    model.mapping.train()
    for epoch in range(EPOCHS):
        order = _epoch_order(sampling, sequences, seed, epoch)
        epoch_losses: list[float] = []
        for start in range(0, TRAIN_PAIRS, BATCH_SIZE):
            pair_batch = order[start : start + BATCH_SIZE]
            if len(pair_batch) != BATCH_SIZE:
                raise RuntimeError("official training pair count is not batch aligned")
            image_batch = pair_image_indices[pair_batch]
            clip_batch = train_emb[image_batch].to(device, non_blocking=True)
            if augment is not None:
                if augment["gaussian_std"]:
                    clip_batch = clip_batch + augment["gaussian_std"] * torch.randn_like(
                        clip_batch
                    )
                if augment["dropout_probability"]:
                    clip_batch = F.dropout(
                        clip_batch,
                        p=augment["dropout_probability"],
                        training=True,
                    )
            if not torch.isfinite(clip_batch).all():
                raise RuntimeError("training feature augmentation produced non-finite values")
            caption_ids, attention = _make_batch(
                tokenizer, sequences, pair_batch, device
            )
            multiplier = _lr_multiplier(schedule, step)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * multiplier
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                logits = model(clip_batch, caption_ids, attention)
                weights = _token_weight_batch(
                    weighting, caption_ids, attention, idf
                )
                loss = _caption_loss(
                    logits, caption_ids, attention, objective, weights
                )
            loss.backward()
            parameters = [p for p in model.mapping.parameters() if p.requires_grad]
            if not parameters or any(
                p.grad is None or not torch.isfinite(p.grad).all() for p in parameters
            ):
                raise RuntimeError("caption training produced missing or non-finite gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("caption gradient norm is non-finite")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise RuntimeError("caption optimizer produced non-finite parameters")
            epoch_losses.append(float(loss.detach().cpu()))
            step += 1
        print(
            f"CAPTION_TRAIN epoch={epoch + 1}/{EPOCHS} steps={step} "
            f"mean_loss={np.mean(epoch_losses):.6f}",
            flush=True,
        )
    if step != EXPECTED_STEPS:
        raise RuntimeError(f"incomplete caption training: {step}/{EXPECTED_STEPS}")
    return step


def _banned_tokens(tokens: list[int], ngram: int) -> list[int]:
    if ngram <= 0 or len(tokens) < ngram - 1:
        return []
    prefix = tuple(tokens[-(ngram - 1) :]) if ngram > 1 else ()
    seen: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for index in range(len(tokens) - ngram + 1):
        gram = tuple(tokens[index : index + ngram])
        seen[gram[:-1]].add(gram[-1])
    return list(seen.get(prefix, set()))


def _reorder_past(past, indices: torch.Tensor):
    return tuple(
        tuple(value.index_select(0, indices) for value in layer) for layer in past
    )


@torch.inference_mode()
def _beam_decode_batch(
    model: CaptionModel,
    tokenizer,
    clip_emb: torch.Tensor,
    config: dict,
    device: torch.device,
) -> list[str]:
    batch = clip_emb.shape[0]
    beam = config["beam_size"] if config["strategy"] == "beam" else 1
    maximum = config["max_length"]
    minimum = config["min_length"]
    no_repeat = config["no_repeat_ngram"]
    eos = tokenizer.eos_token_id
    with _autocast(device):
        prefix = model.visual_prefix(clip_emb.to(device))
        first = model.gpt(inputs_embeds=prefix, use_cache=True)
    logits = first.logits[:, -1, :].float()
    if minimum > 1:
        logits[:, eos] = -torch.inf
    log_probs = F.log_softmax(logits, dim=-1)
    scores, first_tokens = log_probs.topk(beam, dim=-1)
    sequences = first_tokens.unsqueeze(-1)
    finished = first_tokens.eq(eos)
    lengths = torch.ones_like(first_tokens)
    past = tuple(
        tuple(value.repeat_interleave(beam, dim=0) for value in layer)
        for layer in first.past_key_values
    )
    vocabulary = logits.shape[-1]
    for generated in range(1, maximum):
        last_tokens = sequences[:, :, -1].reshape(batch * beam, 1)
        with _autocast(device):
            output = model.gpt(input_ids=last_tokens, past_key_values=past, use_cache=True)
        next_log_probs = F.log_softmax(output.logits[:, -1, :].float(), dim=-1)
        flat_finished = finished.reshape(-1)
        if flat_finished.any():
            next_log_probs[flat_finished] = -torch.inf
            next_log_probs[flat_finished, eos] = 0.0
        if generated + 1 < minimum:
            next_log_probs[~flat_finished, eos] = -torch.inf
        if no_repeat:
            flat_sequences = sequences.reshape(batch * beam, -1)
            for row in range(batch * beam):
                if flat_finished[row]:
                    continue
                banned = _banned_tokens(flat_sequences[row].tolist(), no_repeat)
                if banned:
                    next_log_probs[row, banned] = -torch.inf
        candidates = scores.unsqueeze(-1) + next_log_probs.view(batch, beam, vocabulary)
        scores, indices = candidates.view(batch, beam * vocabulary).topk(beam, dim=-1)
        parents = torch.div(indices, vocabulary, rounding_mode="floor")
        tokens = indices.remainder(vocabulary)
        gather = parents.unsqueeze(-1).expand(-1, -1, sequences.shape[-1])
        sequences = torch.cat([sequences.gather(1, gather), tokens.unsqueeze(-1)], dim=-1)
        previous_finished = finished.gather(1, parents)
        previous_lengths = lengths.gather(1, parents)
        newly_finished = ~previous_finished & tokens.eq(eos)
        lengths = torch.where(newly_finished, generated + 1, previous_lengths)
        finished = previous_finished | newly_finished
        global_parents = (
            torch.arange(batch, device=device).unsqueeze(1) * beam + parents
        ).reshape(-1)
        past = _reorder_past(output.past_key_values, global_parents)
        if finished.all():
            break
    final_lengths = torch.where(
        finished, lengths, torch.full_like(lengths, sequences.shape[-1])
    )
    normalizer = ((5.0 + final_lengths.float()) / 6.0).pow(
        config.get("length_penalty", 0.0)
    )
    best = (scores / normalizer).argmax(dim=1)
    captions: list[str] = []
    for row in range(batch):
        tokens = sequences[row, best[row]].tolist()
        if eos in tokens:
            tokens = tokens[: tokens.index(eos)]
        captions.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
    return captions


@torch.inference_mode()
def _sample_decode_batch(
    model: CaptionModel,
    tokenizer,
    clip_emb: torch.Tensor,
    config: dict,
    device: torch.device,
    generator: torch.Generator,
) -> list[str]:
    batch = clip_emb.shape[0]
    eos = tokenizer.eos_token_id
    sequences = torch.empty(batch, 0, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    with _autocast(device):
        prefix = model.visual_prefix(clip_emb.to(device))
        output = model.gpt(inputs_embeds=prefix, use_cache=True)
    past = output.past_key_values
    logits = output.logits[:, -1, :].float()
    for generated in range(config["max_length"]):
        scaled = logits / config["temperature"]
        if generated + 1 < config["min_length"]:
            scaled[~finished, eos] = -torch.inf
        if config["no_repeat_ngram"]:
            for row in range(batch):
                if finished[row]:
                    continue
                banned = _banned_tokens(
                    sequences[row].tolist(), config["no_repeat_ngram"]
                )
                if banned:
                    scaled[row, banned] = -torch.inf
        sorted_logits, sorted_indices = scaled.sort(descending=True, dim=-1)
        probabilities = F.softmax(sorted_logits, dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative - probabilities >= config["top_p"]
        sorted_logits[remove] = -torch.inf
        filtered = torch.full_like(scaled, -torch.inf).scatter(
            1, sorted_indices, sorted_logits
        )
        probabilities = F.softmax(filtered, dim=-1)
        tokens = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        tokens = torch.where(finished, torch.full_like(tokens, eos), tokens)
        sequences = torch.cat([sequences, tokens.unsqueeze(1)], dim=1)
        finished |= tokens.eq(eos)
        if finished.all() or generated + 1 == config["max_length"]:
            break
        with _autocast(device):
            output = model.gpt(
                input_ids=tokens.unsqueeze(1), past_key_values=past, use_cache=True
            )
        past = output.past_key_values
        logits = output.logits[:, -1, :].float()
    captions: list[str] = []
    for row in range(batch):
        tokens = sequences[row].tolist()
        if eos in tokens:
            tokens = tokens[: tokens.index(eos)]
        captions.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
    return captions


def decode_captions(
    model: CaptionModel,
    tokenizer,
    eval_emb: torch.Tensor,
    config: dict,
    device: torch.device,
    seed: int,
) -> list[str]:
    model.eval()
    # Keep beam batches uniform across the fixed 1,000-image split.  The
    # ragged 16-image tail from a batch size of 24 triggers a reproducible
    # low-level failure in the pinned H20 GPT-2 runtime after 984 captions.
    # Batching is inference-only, so using 20 preserves the exact model,
    # search, predictions, and metrics while covering all images.
    batch_size = 20 if config["strategy"] == "beam" else 64
    generator = torch.Generator(device=device).manual_seed(seed + 7919)
    captions: list[str] = []
    for start in range(0, EVAL_IMAGES, batch_size):
        batch = eval_emb[start : start + batch_size]
        if config["strategy"] == "sample":
            decoded = _sample_decode_batch(
                model, tokenizer, batch, config, device, generator
            )
        else:
            decoded = _beam_decode_batch(model, tokenizer, batch, config, device)
        if len(decoded) != len(batch) or not all(isinstance(item, str) for item in decoded):
            raise RuntimeError("caption decoding returned an incomplete batch")
        captions.extend(decoded)
        print(
            f"CAPTION_DECODE completed={len(captions)}/{EVAL_IMAGES}", flush=True
        )
    if len(captions) != EVAL_IMAGES:
        raise RuntimeError(f"incomplete caption evaluation: {len(captions)}/{EVAL_IMAGES}")
    return captions


def _ptb_tokenize(captions: list[str]) -> list[str]:
    """Run COCO-caption's Stanford PTBTokenizer from a writable temp file."""
    if importlib.metadata.version("pycocoevalcap") != "1.2":
        raise RuntimeError("caption metrics require pinned pycocoevalcap 1.2")
    if importlib.metadata.version("jdk4py") != "17.0.9.2":
        raise RuntimeError("caption metrics require pinned jdk4py 17.0.9.2")

    import jdk4py
    from pycocoevalcap.tokenizer import ptbtokenizer

    if not captions or not all(isinstance(caption, str) for caption in captions):
        raise RuntimeError("PTBTokenizer requires a non-empty string list")
    jar = Path(ptbtokenizer.__file__).parent / ptbtokenizer.STANFORD_CORENLP_3_4_1_JAR
    if not jar.is_file() or not Path(jdk4py.JAVA).is_file():
        raise RuntimeError("pinned PTBTokenizer runtime is incomplete")

    source = "\n".join(caption.replace("\r", " ").replace("\n", " ") for caption in captions)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(source)
            handle.write("\n")
            temp_path = Path(handle.name)
        completed = subprocess.run(
            [
                str(jdk4py.JAVA),
                "-cp",
                str(jar),
                "edu.stanford.nlp.process.PTBTokenizer",
                "-preserveLines",
                "-lowerCase",
                str(temp_path),
            ],
            cwd=jar.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    lines = completed.stdout.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != len(captions):
        raise RuntimeError(
            f"PTBTokenizer returned {len(lines)} lines for {len(captions)} captions"
        )
    punctuation = set(ptbtokenizer.PUNCTUATIONS)
    return [
        " ".join(token for token in line.rstrip().split(" ") if token not in punctuation)
        for line in lines
    ]


def caption_metrics(
    hypotheses: list[str], references: list[list[str]]
) -> tuple[float, float]:
    """Official COCO-caption PTBTokenizer, CIDEr-D, and corpus BLEU-4."""
    if len(hypotheses) != len(references) or not hypotheses:
        raise RuntimeError("caption metrics require aligned non-empty inputs")
    if not all(
        isinstance(refs, list)
        and len(refs) == REFS_PER_IMAGE
        and all(isinstance(ref, str) for ref in refs)
        for refs in references
    ):
        raise RuntimeError("caption metrics require five references per image")

    from pycocoevalcap.bleu.bleu_scorer import BleuScorer
    from pycocoevalcap.cider.cider_scorer import CiderScorer

    tokenized_hypotheses = _ptb_tokenize(hypotheses)
    flat_references = [ref for refs in references for ref in refs]
    tokenized_flat_references = _ptb_tokenize(flat_references)
    tokenized_references = [
        tokenized_flat_references[index : index + REFS_PER_IMAGE]
        for index in range(0, len(tokenized_flat_references), REFS_PER_IMAGE)
    ]

    cider_scorer = CiderScorer(n=4, sigma=6.0)
    bleu_scorer = BleuScorer(n=4)
    for hypothesis, refs in zip(tokenized_hypotheses, tokenized_references):
        cider_scorer += (hypothesis, refs)
        bleu_scorer += (hypothesis, refs)
    cider = float(cider_scorer.compute_score()[0])
    bleu_scores, _ = bleu_scorer.compute_score(option="closest", verbose=0)
    bleu = float(bleu_scores[3])
    if not math.isfinite(cider) or not math.isfinite(bleu):
        raise RuntimeError("COCO-caption metric computation returned non-finite values")
    return cider, bleu


def _fixed_decode_config() -> dict:
    return {
        "strategy": "beam",
        "beam_size": 5,
        "max_length": 24,
        "min_length": 3,
        "length_penalty": 0.8,
        "no_repeat_ngram": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "mapping",
            "decoding",
            "objective",
            "featureprep",
            "init",
            "sampling",
            "optimizer",
            "prompt",
            "augment",
            "weighting",
        ],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--data-root", default=os.environ.get("CAPTION_DATA", "/data/image-captioning")
    )
    parser.add_argument(
        "--gpt-dir",
        default=os.environ.get("CAPTION_GPT2", "/data/image-captioning/gpt2"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("this task protocol requires configured seed 42")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the full caption verifier requires exactly one visible GPU")

    # Fail before the expensive training loop if the official metric runtime is
    # absent or cannot execute on this node.
    _ptb_tokenize(["caption metric runtime check"])
    set_all_seeds(args.seed)
    device = torch.device("cuda")
    config = load_literal_config(Path(args.config), args.mode)
    data_root = Path(args.data_root)
    gpt_dir = Path(args.gpt_dir)
    train_emb, train_refs, eval_emb, eval_refs, manifest, manifest_hash = load_data(
        data_root, gpt_dir
    )
    print(
        f"CAPTION_PROTOCOL protocol={PROTOCOL} train_images={TRAIN_IMAGES} "
        f"train_pairs={TRAIN_PAIRS} eval_images={EVAL_IMAGES} epochs={EPOCHS} "
        f"batch_size={BATCH_SIZE} seed={args.seed} mode={args.mode}",
        flush=True,
    )

    if args.mode == "featureprep":
        train_emb, eval_emb = _apply_feature_prep(config, train_emb, eval_emb)
    mapping_config = config if args.mode == "mapping" else None
    mapping = build_mapping(mapping_config).to(device)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(gpt_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = GPT2LMHeadModel.from_pretrained(gpt_dir, local_files_only=True).to(device)
    if (
        gpt.config.n_embd != GPT_DIM
        or gpt.config.n_layer != GPT_LAYERS
        or gpt.config.n_head != GPT_HEADS
        or gpt.config.vocab_size != GPT_VOCAB
    ):
        raise RuntimeError("pinned GPT-2 architecture does not match the protocol")
    for parameter in gpt.parameters():
        parameter.requires_grad_(False)
    model = CaptionModel(gpt, mapping).to(device)

    prompt_config = config if args.mode == "prompt" else None
    pair_image_indices, sequences = _tokenize_pairs(
        tokenizer, train_refs, prompt_config
    )
    if args.mode == "init":
        flat_tokens = torch.tensor(
            [token for sequence in sequences for token in sequence], dtype=torch.long
        )
        mean_caption_embed = model.embed.weight.detach().cpu()[flat_tokens].mean(dim=0)
        _apply_initialization(model.mapping, config, mean_caption_embed.to(device))

    steps = train_mapping(
        model,
        tokenizer,
        train_emb,
        pair_image_indices,
        sequences,
        mode=args.mode,
        config=config,
        device=device,
        seed=args.seed,
    )
    decode_config = config if args.mode == "decoding" else _fixed_decode_config()
    captions = decode_captions(
        model, tokenizer, eval_emb, decode_config, device, args.seed
    )
    cider, bleu = caption_metrics(captions, eval_refs)
    if not math.isfinite(cider) or not math.isfinite(bleu):
        raise RuntimeError("caption metric computation returned non-finite values")
    if not 0.0 <= cider <= 10.0 or not 0.0 <= bleu <= 1.0:
        raise RuntimeError(f"caption metrics outside valid bounds: {cider}, {bleu}")
    for index in range(3):
        print(
            f"CAPTION_SAMPLE index={index} generated={captions[index]!r} "
            f"reference={eval_refs[index][0]!r}",
            flush=True,
        )
    print(
        f"{RESULT_PREFIX} protocol={PROTOCOL} mode={args.mode} "
        f"train_images={TRAIN_IMAGES} train_pairs={TRAIN_PAIRS} "
        f"eval_images={EVAL_IMAGES} epochs={EPOCHS} batch_size={BATCH_SIZE} "
        f"steps={steps} seed={args.seed} split_sha256={manifest['split_sha256']['test']} "
        f"manifest_sha256={manifest_hash} predictions_sha256={_json_sha256(captions)} "
        f"cider={cider:.6f} bleu4={bleu:.6f} status=ok",
        flush=True,
    )


if __name__ == "__main__":
    main()
