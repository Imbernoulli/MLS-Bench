"""Shared runtime for the text-simplification MLS-Bench tasks.

The tasks use a frozen offline T5 simplification model, source-only public
sentence slices, evaluation-only human targets, and a vendored SARI
implementation. Agent-editable files choose one inference-time behavior; this
module supplies fixed data loading, decoding, and metric utilities.
"""
from __future__ import annotations

import ast
import copy
import functools
import hashlib
import json
import math
import os
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Fixed evaluation constants (shared by ALL simp-* tasks)
# ---------------------------------------------------------------------------
SETTING_COUNTS = {"asset": 359, "turk": 359, "wiki": 720}
SOURCE_SHA256 = {
    "asset": "882a782a2d5ef45382028d8d27cf9de43b1a8d86bbe896d12039453bb7264cf2",
    "turk": "e50505299519f7977117d542a1055a948bab5764410ed748f6b3fd6f07a43fd9",
    "wiki": "478118f163286aafd5275cdcc1420fe34d04be44f13fede2030c950d7bfd4270",
}
REFERENCE_SHA256 = {
    "asset": "65814d0fbb5c7360b0622d5f10a2807108f8d7815dc1b2fee357af0118af1b2e",
    "turk": "091254eac0b60eb51e9619085fcdf59406dc71fd32fe64fdae4b374d8b75b25f",
    "wiki": "dab7fa2e6b36b1e978b8862b6010902e8ab35eafc7981db94bbcc0f4a7ee08b1",
}
MAX_INPUT_TOKENS = 160     # source truncation (sentences are ~20-30 words)
MAX_NEW_TOKENS_CAP = 200   # hard cap on generated length (keeps it minute-scale)
GEN_BATCH_SIZE = 16        # fixed generation batch size
SEED = 42

# T5 simplification models are trained with a task prefix; FIXED here.
SRC_PREFIX = "simplify: "

# The three fixed evaluation settings, staged offline as JSONL.
SETTINGS = ("asset", "turk", "wiki")
PROTOCOL = "gem-full-test-v2"
SETTINGS_BINDING = ",".join(
    f"{setting}:{SETTING_COUNTS[setting]}" for setting in SETTINGS
)
TASK_SURFACES = {
    "simp-beam-width": "beamwidth",
    "simp-decoding-beam": "beam",
    "simp-decoding-strategy": "strategy",
    "simp-decoding-temperature": "temperature",
    "simp-input-truncation": "truncation",
    "simp-length-control": "length",
    "simp-minlen-floor": "minlen",
    "simp-model-capacity": "capacity",
    "simp-nucleus-sampling": "nucleus",
    "simp-source-policy": "policy",
}

DATA_INVENTORY = {
    setting: {
        "count": SETTING_COUNTS[setting],
        "source_sha256": SOURCE_SHA256[setting],
        "reference_sha256": REFERENCE_SHA256[setting],
    }
    for setting in SETTINGS
}

MODEL_SPECS = {
    "small_turk": {
        "directory": "t5-small-finetuned-turk-text-simplification",
        "repo_id": "mrm8488/t5-small-finetuned-turk-text-simplification",
        "revision": "f1c6a63751592c9b51d27acce8ab77e02563c983",
        "architecture": {
            "model_type": "t5", "d_model": 512, "d_ff": 2048,
            "num_layers": 6, "num_decoder_layers": 6,
            "num_heads": 8, "vocab_size": 32128,
        },
        "files": {
            "config.json": {
                "bytes": 1473,
                "sha256": "0e29a6b11425fd91dc3f3e80f55aa6e38f499e08eedb5812d002666f418fe10e",
            },
            "tokenizer.json": {
                "bytes": 2422360,
                "sha256": "f9eadbc572afadad1a76d19519db98e9c03261adb016ada4568c28ef7dd3a854",
            },
            "model.safetensors": {
                "bytes": 242041894,
                "sha256": "4d4e5fcb2ce1bb58134fb86179b01407dc6bc49370960f2f40fa006effba2d24",
            },
        },
    },
    "small_wikiauto": {
        "directory": "t5-small-finetuned-text-simplification",
        "repo_id": "mrm8488/t5-small-finetuned-text-simplification",
        "revision": "6b7f868dad51927dbf8fffd05bc8d71abe379c87",
        "architecture": {
            "model_type": "t5", "d_model": 512, "d_ff": 2048,
            "num_layers": 6, "num_decoder_layers": 6,
            "num_heads": 8, "vocab_size": 32128,
        },
        "files": {
            "config.json": {
                "bytes": 1473,
                "sha256": "b08329c26cb26547cf83a44b03ad1d4407f8bf35d326952450de9d01427bcd90",
            },
            "tokenizer.json": {
                "bytes": 2422360,
                "sha256": "f9eadbc572afadad1a76d19519db98e9c03261adb016ada4568c28ef7dd3a854",
            },
            "pytorch_model.bin": {
                "bytes": 242070267,
                "sha256": "da7890699bcb91de81d46cabe175bbacff9d31d2e9badc83dc2ada3e2345ca88",
            },
        },
    },
    "base_turk": {
        "directory": "t5-base-finetuned-turk-text-simplification",
        "repo_id": "mrm8488/t5-base-finetuned-turk-text-simplification",
        "revision": "3049a645d59a3bb39abfb808b2ac89896876980f",
        "architecture": {
            "model_type": "t5", "d_model": 768, "d_ff": 3072,
            "num_layers": 12, "num_decoder_layers": 12,
            "num_heads": 12, "vocab_size": 32128,
        },
        "files": {
            "config.json": {
                "bytes": 1475,
                "sha256": "6603801d50185a7f2b8955fd3794b1e94813952b21fb1d25dfb6f6456231f20f",
            },
            "tokenizer.json": {
                "bytes": 2422360,
                "sha256": "f9eadbc572afadad1a76d19519db98e9c03261adb016ada4568c28ef7dd3a854",
            },
            "model.safetensors": {
                "bytes": 891644710,
                "sha256": "063b490a880f6164b5aa7e8bc470911825c981ebf7396a31a4c9bf9f642c7fb4",
            },
        },
    },
}


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


DATA_INVENTORY_SHA256 = _canonical_sha256(DATA_INVENTORY)
MODEL_SPEC_SHA256 = {
    choice: _canonical_sha256(spec) for choice, spec in MODEL_SPECS.items()
}

# Source-only jsonl shipped directly alongside this file. Produced host-side by
# holdout/text-simplification/generate_data.py before packaging.
_SIMP_DATA_DIR = Path(__file__).resolve().parent / "_simp_data"


def model_path() -> str:
    """Return the configured path of the frozen base-Turk checkpoint."""
    return os.environ.get(
        "SIMP_MODEL",
        "/data/text-simplification/models/t5-base-finetuned-turk-text-simplification",
    )


def setup(seed: int = SEED):
    """Require the fixed one-GPU runtime, pin the seed, and force offline mode."""
    import random

    import numpy as np
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int) or seed != SEED:
        print(f"SIMP_FAILURE required seed={SEED}, got {seed!r}", flush=True)
        raise ValueError(f"text-simplification verification requires seed {SEED}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        print(
            "SIMP_FAILURE verification requires exactly one visible CUDA device",
            flush=True,
        )
        raise RuntimeError("text-simplification verification requires one GPU")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda:0")


def _refs_data_path(setting: str) -> Path:
    """Resolve the evaluation-only target jsonl for `setting`."""
    task_dir = Path(os.environ.get("TASK_DIR", "/workspace/_task"))
    return task_dir / "data" / f"simp_{setting}_refs.jsonl"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def model_identity(choice: str = "base_turk") -> tuple[str, str]:
    """Verify one staged checkpoint and return its path and manifest digest."""
    if choice not in MODEL_SPECS:
        raise ValueError(f"unknown simplification model choice: {choice!r}")
    spec = MODEL_SPECS[choice]
    path = Path(model_path()).parent / spec["directory"]
    if not path.is_dir():
        raise FileNotFoundError(f"required simplification model is missing: {path}")
    for filename, expected in spec["files"].items():
        file_path = path / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"required model file is missing: {file_path}")
        if file_path.stat().st_size != expected["bytes"]:
            raise ValueError(f"model file size mismatch: {file_path}")
        if _file_sha256(file_path) != expected["sha256"]:
            raise ValueError(f"model file digest mismatch: {file_path}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    observed_architecture = {
        name: config.get(name) for name in spec["architecture"]
    }
    if observed_architecture != spec["architecture"]:
        raise ValueError(
            f"model architecture mismatch for {choice}: {observed_architecture}"
        )
    return str(path), MODEL_SPEC_SHA256[choice]


def load_dataset(setting: str, n_sents: int | None = None):
    """Load one complete official simplification test setting.

    Returns (sources, references): sources is a list[str] and references is a
    list[list[str]]. Source files are available to the action workspace. Target
    files are available only when the verifier runs.
    """
    if setting not in SETTINGS:
        raise SystemExit(f"unknown setting {setting!r}; expected one of {SETTINGS}")
    expected_count = SETTING_COUNTS[setting]
    if n_sents not in (None, expected_count):
        raise ValueError(
            f"setting={setting!r} requires all {expected_count} official test rows"
        )

    def _read_jsonl(fp: Path):
        with fp.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    raise ValueError(f"blank row {line_number} in {fp}")
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON row {line_number} in {fp}") from exc

    src_fp = _SIMP_DATA_DIR / f"simp_{setting}_src.jsonl"
    if not src_fp.exists():
        raise SystemExit(
            f"no frozen source data for setting={setting!r}; expected "
            f"{src_fp}. Regenerate with "
            f"holdout/text-simplification/generate_data.py (host-side only)."
        )
    if _file_sha256(src_fp) != SOURCE_SHA256[setting]:
        raise ValueError(f"source digest mismatch for setting={setting!r}")
    source_rows = list(_read_jsonl(src_fp))
    if any(not isinstance(rec, dict) or set(rec) != {"source"}
           or not isinstance(rec["source"], str) or not rec["source"].strip()
           for rec in source_rows):
        raise ValueError(f"malformed source rows for setting={setting!r}")
    srcs: List[str] = [rec["source"] for rec in source_rows]

    refs_fp = _refs_data_path(setting)
    if not refs_fp.exists():
        raise SystemExit(
            f"no held-out reference data for setting={setting!r}; expected "
            f"{refs_fp}. This file is staged for verifier execution and is "
            f"not available during an agent action session."
        )
    if _file_sha256(refs_fp) != REFERENCE_SHA256[setting]:
        raise ValueError(f"reference digest mismatch for setting={setting!r}")
    reference_rows = list(_read_jsonl(refs_fp))
    if any(not isinstance(rec, dict) or set(rec) != {"references"}
           or not isinstance(rec["references"], list) or not rec["references"]
           or any(not isinstance(ref, str) or not ref.strip()
                  for ref in rec["references"])
           for rec in reference_rows):
        raise ValueError(f"malformed reference rows for setting={setting!r}")
    refs: List[List[str]] = [rec["references"] for rec in reference_rows]

    if len(srcs) != expected_count or len(refs) != expected_count:
        raise SystemExit(
            f"incomplete official setting={setting!r}: expected {expected_count}, "
            f"got {len(srcs)} sources and {len(refs)} reference rows"
        )
    return srcs, refs


def load_model_and_tokenizer(device):
    """FROZEN t5-base simplifier, eval mode, staged offline."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    verified_path, _ = model_identity("base_turk")
    tok = AutoTokenizer.from_pretrained(verified_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        verified_path, local_files_only=True, torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()
    return model, tok


def emit_metrics(
    *, task: str, surface: str, setting: str, sari: float, bleu: float,
    n_sents: int, plen: float, lenratio: float,
) -> str:
    """Emit one canonical, task-bound metric record and return its bytestring."""
    if TASK_SURFACES.get(task) != surface:
        raise ValueError(f"task/surface mismatch: {task!r}/{surface!r}")
    if setting not in SETTINGS or n_sents != SETTING_COUNTS[setting]:
        raise ValueError(f"invalid simplification setting inventory: {setting}/{n_sents}")
    values = (float(sari), float(bleu), float(plen), float(lenratio))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite simplification metric")
    if not 0.0 <= values[0] <= 100.0 or not 0.0 <= values[1] <= 100.0:
        raise ValueError("simplification score outside [0, 100]")
    if not 0.0 <= values[2] <= MAX_NEW_TOKENS_CAP or not 0.0 <= values[3] <= 10.0:
        raise ValueError("simplification length diagnostic outside bounds")
    line = (
        f"SIMP_METRICS protocol={PROTOCOL} task={task} surface={surface} "
        f"setting={setting} sari={values[0]:.6f} bleu={values[1]:.4f} "
        f"n_sents={n_sents} plen={values[2]:.1f} lenratio={values[3]:.3f}"
    )
    print(line, flush=True)
    return line


def emit_done(
    *, task: str, surface: str, seed: int, model_choice: str,
    metric_lines: list[str], elapsed: float,
) -> str:
    """Emit the unique canonical completion record after every metric record."""
    if TASK_SURFACES.get(task) != surface:
        raise ValueError(f"task/surface mismatch: {task!r}/{surface!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    elapsed = float(elapsed)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("completion elapsed must be finite and positive")
    expected_settings = [f"setting={setting}" for setting in SETTINGS]
    if len(metric_lines) != len(SETTINGS) or any(
        expected not in line
        for expected, line in zip(expected_settings, metric_lines, strict=True)
    ):
        raise ValueError("completion requires the ordered full metric inventory")
    _, model_sha256 = model_identity(model_choice)
    metric_payload = ("\n".join(metric_lines) + "\n").encode("utf-8")
    metrics_sha256 = hashlib.sha256(metric_payload).hexdigest()
    line = (
        f"SIMP_DONE protocol={PROTOCOL} task={task} surface={surface} "
        f"settings={SETTINGS_BINDING} seed={seed} "
        f"inventory_sha256={DATA_INVENTORY_SHA256} model={model_choice} "
        f"model_sha256={model_sha256} metrics_sha256={metrics_sha256} "
        f"elapsed={elapsed:.6f} status=ok"
    )
    print(line, flush=True)
    return line


def load_surface(sol_path: str, attr: str):
    """Parse a no-argument finite JSON-literal surface without executing it."""
    path = Path(sol_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {path}")
    source = path.read_text()
    if len(source.encode()) > 64 * 1024:
        raise ValueError("simplification solution exceeds 64 KiB")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"simplification solution does not parse: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise ValueError("simplification solution exceeds 512 AST nodes")

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    matches = [
        node for node in functions
        if isinstance(node, ast.FunctionDef) and node.name == attr
    ]
    if len(matches) != 1 or len(functions) != 1:
        raise ValueError(f"solution must define exactly one `{attr}()` function")
    function = matches[0]
    if (function.decorator_list or function.args.posonlyargs or function.args.args
            or function.args.vararg is not None or function.args.kwarg is not None
            or function.args.kwonlyargs or function.args.defaults
            or function.args.kw_defaults):
        raise ValueError(f"`{attr}` must be undecorated and take no arguments")
    for node in tree.body:
        if node is function:
            continue
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        raise ValueError("simplification solution may not import or execute top-level code")

    body = list(function.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        raise ValueError(f"`{attr}` must contain exactly one literal return")
    try:
        value = ast.literal_eval(body[0].value)
        payload = json.dumps(value, allow_nan=False, separators=(",", ":"))
        value = json.loads(payload)
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError(f"`{attr}` must return a finite JSON literal") from exc

    def surface():
        return copy.deepcopy(value)

    return surface


def require_surface_config(value, required, *, allowed=None, surface="config"):
    """Require a complete editable mapping; never manufacture missing keys."""
    if not isinstance(value, dict):
        print(f"SURFACE_ERROR {surface}: expected dict, got {type(value).__name__}",
              flush=True)
        raise TypeError(f"{surface} must return a dict")
    required = set(required)
    allowed = set(allowed if allowed is not None else required)
    missing = sorted(required - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        print(f"SURFACE_ERROR {surface}: missing={missing} extra={extra}", flush=True)
        raise ValueError(f"invalid {surface} schema")
    return value


def require_surface_choice(value, name, allowed, *, surface):
    """Require a string selector without coercing another type to text."""
    if not isinstance(value, str):
        print(f"SURFACE_ERROR {surface}: {name} must be a string", flush=True)
        raise TypeError(f"{name} must be a string")
    value = value.strip().lower()
    if value not in allowed:
        print(f"SURFACE_ERROR {surface}: unsupported {name}={value!r}", flush=True)
        raise ValueError(f"unsupported {name}")
    return value


def require_surface_int(value, name, low, high, *, surface):
    """Require a built-in integer in range without truncation or clamping."""
    if isinstance(value, bool) or not isinstance(value, int):
        print(f"SURFACE_ERROR {surface}: {name} must be an integer", flush=True)
        raise TypeError(f"{name} must be an integer")
    if not low <= value <= high:
        print(f"SURFACE_ERROR {surface}: {name}={value} outside [{low}, {high}]",
              flush=True)
        raise ValueError(f"{name} outside allowed range")
    return value


def require_surface_number(value, name, low, high, *, surface, low_open=False):
    """Require a finite built-in number in range without string conversion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        print(f"SURFACE_ERROR {surface}: {name} must be numeric", flush=True)
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        print(f"SIMP_NONFINITE {name}={value}", flush=True)
        raise ValueError(f"non-finite {name}")
    valid_low = value > low if low_open else value >= low
    if not valid_low or value > high:
        print(f"SURFACE_ERROR {surface}: {name}={value} outside allowed range",
              flush=True)
        raise ValueError(f"{name} outside allowed range")
    return value


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _sanitize_gen_kwargs(gen_kwargs: dict) -> dict:
    """Validate generation kwargs and reject invalid values without repair."""
    allowed = {
        "num_beams", "min_length", "max_length", "max_new_tokens",
        "length_penalty", "no_repeat_ngram_size", "repetition_penalty",
        "early_stopping",
        # NEW knobs (all default to the ORIGINAL deterministic behaviour when
        # absent, so old callers that never set them are unaffected):
        "do_sample", "temperature", "top_p", "top_k",
        "num_beam_groups", "diversity_penalty",
    }
    if not isinstance(gen_kwargs, dict):
        print("SURFACE_ERROR simplification generation config must be a dict", flush=True)
        raise TypeError("generation config must be a dict")
    out = {}
    for k, v in gen_kwargs.items():
        if k not in allowed:
            raise SystemExit(
                f"decode kwarg {k!r} not allowed; permitted: {sorted(allowed)}"
            )
        out[k] = v
    def _int(name, low, high):
        value = out[name]
        if isinstance(value, bool) or not isinstance(value, int):
            print(f"SURFACE_ERROR simplification {name} must be an integer", flush=True)
            raise TypeError(f"{name} must be an integer")
        if not low <= value <= high:
            print(f"SURFACE_ERROR simplification {name}={value} outside [{low}, {high}]",
                  flush=True)
            raise ValueError(f"{name} outside allowed range")

    def _float(name, low, high, *, low_open=False):
        value = out[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            print(f"SURFACE_ERROR simplification {name} must be numeric", flush=True)
            raise TypeError(f"{name} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            print(f"SIMP_NONFINITE {name}={value}", flush=True)
            raise ValueError(f"non-finite {name}")
        valid_low = value > low if low_open else value >= low
        if not valid_low or value > high:
            print(f"SURFACE_ERROR simplification {name}={value} outside allowed range",
                  flush=True)
            raise ValueError(f"{name} outside allowed range")
        out[name] = value

    for name, low, high in (
        ("num_beams", 1, 12), ("min_length", 0, MAX_NEW_TOKENS_CAP),
        ("max_length", 1, MAX_NEW_TOKENS_CAP),
        ("max_new_tokens", 1, MAX_NEW_TOKENS_CAP),
        ("no_repeat_ngram_size", 0, 20), ("top_k", 0, 200),
        ("num_beam_groups", 1, 12),
    ):
        if name in out:
            _int(name, low, high)
    for name, low, high, low_open in (
        ("length_penalty", 0.0, 10.0, True),
        ("repetition_penalty", 0.0, 10.0, True),
        ("temperature", 0.0, 2.5, True),
        ("top_p", 0.0, 1.0, True),
        ("diversity_penalty", 0.0, 5.0, False),
    ):
        if name in out:
            _float(name, low, high, low_open=low_open)
    for name in ("do_sample", "early_stopping"):
        if name in out and not isinstance(out[name], bool):
            print(f"SURFACE_ERROR simplification {name} must be bool", flush=True)
            raise TypeError(f"{name} must be bool")
    if "min_length" in out and "max_length" in out:
        if out["min_length"] > out["max_length"]:
            print("SURFACE_ERROR simplification min_length exceeds max_length", flush=True)
            raise ValueError("min_length exceeds max_length")
    groups = out.get("num_beam_groups", 1)
    beams = out.get("num_beams", 1)
    if groups > beams or beams % groups:
        print("SURFACE_ERROR simplification beam groups must divide num_beams", flush=True)
        raise ValueError("invalid beam grouping")
    if "diversity_penalty" in out and groups <= 1:
        print("SURFACE_ERROR simplification diversity_penalty requires grouped beams",
              flush=True)
        raise ValueError("diversity_penalty requires grouped beams")
    return out


def simplify(model, tok, sources: List[str], gen_kwargs: dict, device) -> List[str]:
    """Decode a simplification for every source with the agent's config.

    The FIXED task prefix, source truncation (MAX_INPUT_TOKENS), batching, and the
    frozen model are all fixed; only `gen_kwargs` (the agent's decode config) varies.
    """
    import torch

    gk = _sanitize_gen_kwargs(gen_kwargs)
    preds: List[str] = []
    for i in range(0, len(sources), GEN_BATCH_SIZE):
        batch = [SRC_PREFIX + s for s in sources[i:i + GEN_BATCH_SIZE]]
        enc = tok(
            batch,
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                **gk,
            )
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))
    if len(preds) != len(sources):
        print(f"SURFACE_ERROR simplification prediction count {len(preds)} "
              f"for {len(sources)} sources", flush=True)
        raise ValueError("incomplete simplification predictions")
    return [p.strip() for p in preds]


# ---------------------------------------------------------------------------
# Simple text utilities used by policy-style harnesses.
# ---------------------------------------------------------------------------
def copy_input(sources: List[str]) -> List[str]:
    """Return each source sentence unchanged."""
    return [s.strip() for s in sources]


def truncate_tail(sources: List[str], keep_ratio: float = 0.75) -> List[str]:
    """Keep only the first `keep_ratio` of the words in each source."""
    out = []
    for s in sources:
        w = s.split()
        k = max(1, int(round(len(w) * keep_ratio)))
        out.append(" ".join(w[:k]))
    return out


# ---------------------------------------------------------------------------
# Scoring (SARI is the primary metric; BLEU + length ratio are diagnostics)
# ---------------------------------------------------------------------------
def score_sari(sources: List[str], preds: List[str],
               references: List[List[str]]) -> float:
    """Corpus SARI (0-100, higher is better) via the vendored SARI implementation
    (Xu et al. 2016; faithful port of the HuggingFace ``evaluate`` metric)."""
    from sari import corpus_sari

    if (not sources or len(sources) != len(preds) or
            len(sources) != len(references) or any(not refs for refs in references)):
        print("SURFACE_ERROR simplification incomplete sources/predictions/references",
              flush=True)
        raise ValueError("incomplete simplification inputs")
    value = float(corpus_sari(sources, preds, references))
    if not math.isfinite(value):
        print(f"SIMP_NONFINITE sari={value}", flush=True)
        raise ValueError("non-finite SARI")
    return value


def bleu_corpus(preds: List[str], references: List[List[str]]) -> float:
    """Self-contained corpus BLEU-4 (0-100) — an ADEQUACY / meaning-preservation
    diagnostic (a good simplification stays close to a reference). NOT the primary
    score (SARI is), reported only as a sanity trace so over-deletion is visible."""
    import math
    import re
    from collections import Counter

    def tok(s: str):
        return re.findall(r"\w+|[^\w\s]", s.lower())

    def ng(t, n):
        return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))

    if not preds:
        return 0.0
    total = 0.0
    for h, rs in zip(preds, references):
        ht = tok(h)
        rr = [tok(r) for r in rs] or [ht]
        if not ht:
            continue
        log_sum = 0.0
        ok = True
        for n in range(1, 5):
            hn = ng(ht, n)
            hd = sum(hn.values())
            if hd == 0:
                ok = False
                break
            mx = Counter()
            for r in rr:
                for g, c in ng(r, n).items():
                    if c > mx.get(g, 0):
                        mx[g] = c
            clip = sum(min(c, mx.get(g, 0)) for g, c in hn.items())
            if clip <= 0:
                ok = False
                break
            log_sum += math.log(clip / hd)
        if not ok:
            continue
        c = len(ht)
        r = min((len(x) for x in rr), key=lambda rl: (abs(rl - c), rl))
        bp = 1.0 if c > r else math.exp(1 - r / max(1, c))
        total += bp * math.exp(log_sum / 4)
    return 100.0 * total / max(1, len(preds))


def mean_pred_len_words(preds: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(len(p.split()) for p in preds) / len(preds)


def length_ratio(sources: List[str], preds: List[str]) -> float:
    """Mean output/input word-count ratio (diagnostic trace, NOT scored)."""
    if not sources:
        return 0.0
    tot = 0.0
    for s, p in zip(sources, preds):
        ns = max(1, len(s.split()))
        tot += len(p.split()) / ns
    return tot / len(sources)
