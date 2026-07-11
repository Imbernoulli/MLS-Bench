"""Verifier-only runtime shared by the natural-language-inference tasks.

The representative protocol trains one DistilBERT cross-encoder for three
epochs over every labeled SNLI training example, then evaluates that same model
on the complete labeled SNLI test, MultiNLI matched dev, and MultiNLI mismatched
dev splits. Data order, model revision, optimizer, sequence length, batch sizes,
seed, and metric are fixed and authenticated. No run-time installation,
download, score substitution, or incomplete-result fallback is supported.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed evaluation constants (shared by ALL nli-* tasks)
# ---------------------------------------------------------------------------
PROTOCOL = "nli-full-snli-distilbert-v1"
MAX_SEQUENCE_LENGTH = 128
MAX_LEN = 64  # Per-sentence cap for the siamese sibling surfaces.
TRAIN_BATCH = 32
EVAL_BATCH = 128
MAX_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1
SEED_DEFAULT = 42
ENCODER_LR = 2e-5
HEAD_LR = 1e-3

# SNLI/MNLI 3-way labels in a FIXED canonical order (index == class id).
LABEL_NAMES = ["entailment", "neutral", "contradiction"]
NUM_LABELS = len(LABEL_NAMES)

# The ≥3 fixed evaluation DOMAINS (validation settings). Training is ALWAYS on the
# The complete labeled SNLI train split is fixed; only the eval domain changes.
DATASET_INVENTORY = {
    "snli_train": {
        "relative_path": "snli/train.jsonl",
        "rows": 549367,
        "sha256": "3cdde4e94e0c5ca8e7e3d95b0c7c7b9fc03b101d3b9e79c422150bf5c17f1f73",
    },
    "snli": {
        "relative_path": "snli/test.jsonl",
        "rows": 9824,
        "sha256": "e30ea21eb677dab4806e1cc4c646dffc23985ffd982fd6bd15ab3617cd601dd8",
    },
    "mnli_m": {
        "relative_path": "mnli/dev_matched.jsonl",
        "rows": 9815,
        "sha256": "a612ccdf07b2fbe73e2904b061b9e278f552a39b553999bc626de6df6ec4b66d",
    },
    "mnli_mm": {
        "relative_path": "mnli/dev_mismatched.jsonl",
        "rows": 9832,
        "sha256": "a08757b4ddc34421f8f6eac69eb5dd97b2125693078c541cad2d54689013f68d",
    },
}
DOMAINS = {name: DATASET_INVENTORY[name]["relative_path"]
           for name in ("snli", "mnli_m", "mnli_mm")}
TOTAL_EVAL_ROWS = sum(DATASET_INVENTORY[name]["rows"] for name in DOMAINS)
STEPS_PER_EPOCH = (
    DATASET_INVENTORY["snli_train"]["rows"] + TRAIN_BATCH - 1
) // TRAIN_BATCH
TOTAL_OPTIMIZER_STEPS = STEPS_PER_EPOCH * MAX_EPOCHS

MODEL_REPO = "distilbert/distilbert-base-uncased"
MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MODEL_WEIGHTS_FILE = "model.safetensors"
MODEL_WEIGHTS_BYTES = 267954768
MODEL_ASSET_SHA256 = {
    "config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
    "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
MODEL_WEIGHTS_SHA256 = MODEL_ASSET_SHA256[MODEL_WEIGHTS_FILE]
MODEL_PARAMETER_COUNT = 66362880
_AUTHENTICATED_MODEL_ASSETS: dict[str, str] | None = None


def model_path() -> str:
    """FROZEN-vocabulary small transformer encoder staged offline."""
    return os.environ.get(
        "NLI_MODEL",
        "/data/natural-language-inference/models/distilbert-base-uncased")


def data_root() -> Path:
    return Path(os.environ.get(
        "NLI_DATA", "/data/natural-language-inference/data"))


def eval_data_root() -> Path:
    """Evaluation labels must come from the verifier-only staged data root."""
    raw = os.environ.get("NLI_EVAL_DATA")
    if not raw:
        raise RuntimeError("NLI_EVAL_DATA is required for verifier evaluation")
    root = Path(raw)
    if not root.is_absolute():
        raise RuntimeError("NLI_EVAL_DATA must be an absolute verifier path")
    return root


def setup(seed: int = SEED_DEFAULT):
    """Require one assigned GPU, pin seeds, and route caches offline."""
    import random

    import numpy as np
    import torch

    os.environ.setdefault("HF_HOME", str(data_root() / "hf_home"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("full NLI verification requires exactly one visible CUDA GPU")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("NLI_DEVICE type=cuda visible=1", flush=True)
    return torch.device("cuda:0")


def load_surface(sol_path: str, attr: str):
    """Import the agent-editable callable `attr` from solution/<file>.py."""
    p = Path(sol_path)
    spec = importlib.util.spec_from_file_location("agent_surface", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, attr):
        raise SystemExit(f"solution must define `{attr}(...)`")
    return getattr(mod, attr)


def require_surface_config(value, required, *, allowed=None, surface="config"):
    """Require an explicit mapping instead of substituting a selector default."""
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


# ---------------------------------------------------------------------------
# Data: SNLI (staged offline as JSONL)
# ---------------------------------------------------------------------------
# Each staged JSONL row:
#   {"premise": "<sentence>", "hypothesis": "<sentence>", "label": "<name>",
#    "hard": <bool>}
# where "hard" marks rows on which a hypothesis-only classifier is wrong (used by
# the nli-hypothesis-bias task's HARD test subset). The label is one of
# entailment / neutral / contradiction (rows with SNLI's "-" no-consensus label
# are dropped during preparation).

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_model_assets() -> dict[str, str]:
    """Authenticate every runtime model/tokenizer asset once per verifier run."""
    global _AUTHENTICATED_MODEL_ASSETS
    if _AUTHENTICATED_MODEL_ASSETS is not None:
        return dict(_AUTHENTICATED_MODEL_ASSETS)

    root = Path(model_path())
    actual: dict[str, str] = {}
    for filename, expected_digest in MODEL_ASSET_SHA256.items():
        path = root / filename
        if not path.is_file():
            raise RuntimeError(f"missing pinned NLI model asset: {path}")
        digest = _sha256(path)
        if digest != expected_digest:
            raise RuntimeError(
                f"NLI model asset digest mismatch for {filename}: "
                f"{digest} != {expected_digest}"
            )
        actual[filename] = digest
    weights = root / MODEL_WEIGHTS_FILE
    if weights.stat().st_size != MODEL_WEIGHTS_BYTES:
        raise RuntimeError(f"wrong-sized pinned NLI weights: {weights}")
    _AUTHENTICATED_MODEL_ASSETS = actual
    return dict(actual)


def _read_authenticated_jsonl(split: str, path: Path):
    expected = DATASET_INVENTORY[split]
    if not path.is_file():
        raise FileNotFoundError(f"required NLI data file is missing: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected["sha256"]:
        raise RuntimeError(
            f"NLI data digest mismatch for {split}: {actual_sha256} != "
            f"{expected['sha256']}"
        )
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"malformed NLI JSONL for {split} at line {line_number}"
                ) from exc
            if set(row) != {"premise", "hypothesis", "label", "hard"}:
                raise RuntimeError(
                    f"wrong NLI schema for {split} at line {line_number}"
                )
            if (not isinstance(row["premise"], str) or not row["premise"].strip()
                    or not isinstance(row["hypothesis"], str)
                    or not row["hypothesis"].strip()
                    or row["label"] not in LABEL_NAMES
                    or not isinstance(row["hard"], bool)):
                raise RuntimeError(
                    f"invalid NLI row for {split} at line {line_number}"
                )
            rows.append(row)
    if len(rows) != expected["rows"]:
        raise RuntimeError(
            f"wrong full-split row count for {split}: {len(rows)} != "
            f"{expected['rows']}"
        )
    print(
        f"NLI_DATA split={split} rows={len(rows)} sha256={actual_sha256}",
        flush=True,
    )
    return rows


def load_training_data():
    entry = DATASET_INVENTORY["snli_train"]
    return _read_authenticated_jsonl(
        "snli_train", data_root() / entry["relative_path"]
    )


def load_evaluation_data(domain: str):
    if domain not in DOMAINS:
        raise SystemExit(f"unknown domain {domain!r}; expected one of {list(DOMAINS)}")
    return _read_authenticated_jsonl(
        domain, eval_data_root() / DATASET_INVENTORY[domain]["relative_path"]
    )


def load_dataset(domain: str = "snli"):
    """Compatibility helper for sibling harnesses that run one domain."""
    return load_training_data(), load_evaluation_data(domain)


def emit_protocol(
    seed: int,
    *,
    task_id: str,
    surface: str,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> None:
    print(
        f"NLI_PROTOCOL version={PROTOCOL} task={task_id} surface={surface} "
        f"settings={len(DOMAINS)} "
        f"train_rows={DATASET_INVENTORY['snli_train']['rows']} "
        f"eval_rows={TOTAL_EVAL_ROWS} epochs={MAX_EPOCHS} "
        f"train_batch={TRAIN_BATCH} eval_batch={EVAL_BATCH} "
        f"max_length={int(max_length)} seed={seed}",
        flush=True,
    )


def emit_model_proof(model) -> None:
    assets = authenticate_model_assets()
    encoder_parameters = sum(
        parameter.numel() for parameter in model.encoder.parameters()
    )
    if encoder_parameters != MODEL_PARAMETER_COUNT:
        raise RuntimeError(
            "NLI encoder parameter mismatch: "
            f"{encoder_parameters} != {MODEL_PARAMETER_COUNT}"
        )
    head_parameters = sum(
        parameter.numel() for parameter in model.classifier.parameters()
    )
    total_parameters = encoder_parameters + head_parameters
    dtype = str(next(model.encoder.parameters()).dtype).replace("torch.", "")
    print(
        f"NLI_MODEL model={MODEL_NAME} revision={MODEL_REVISION} "
        f"architecture={model.architecture} encoder_params={encoder_parameters} "
        f"head_params={head_parameters} total_params={total_parameters} "
        f"dtype={dtype} "
        f"config_sha256={assets['config.json']} "
        f"weights_sha256={assets['model.safetensors']} "
        f"tokenizer_sha256={assets['tokenizer.json']} "
        f"tokenizer_config_sha256={assets['tokenizer_config.json']} "
        f"vocab_sha256={assets['vocab.txt']}",
        flush=True,
    )


def label_to_id(name: str) -> int:
    return LABEL_NAMES.index(name)


# ---------------------------------------------------------------------------
# Training-set transforms (deterministic, shared) used by the data-oriented
# surfaces (class-weighting / augmentation). The EVAL split is NEVER touched.
# ---------------------------------------------------------------------------

# FIXED per-class keep fraction that turns the nearly balanced full train split
# into a class-imbalanced one (contradiction is the minority). Used by the class-
# weighting surface so that reweighting the loss has something to correct.
IMBALANCE_KEEP = {"entailment": 1.0, "neutral": 1.0, "contradiction": 0.25}


def make_imbalanced(train_rows):
    """Deterministically drop rows to induce a FIXED class imbalance.

    Keeps IMBALANCE_KEEP[label] of each class (a stable prefix after a fixed
    per-label sort), so contradiction becomes the rare minority class. The
    eval split is unchanged, so a model that ignores the imbalance under-predicts
    contradiction and loses accuracy vs one that reweights the loss.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, r in enumerate(train_rows):
        buckets[r["label"]].append((i, r))
    kept = []
    for label, rows in buckets.items():
        rows = sorted(rows, key=lambda ir: ir[0])  # stable, deterministic
        k = max(1, int(round(len(rows) * IMBALANCE_KEEP.get(label, 1.0))))
        kept.extend(rows[:k])
    kept.sort(key=lambda ir: ir[0])
    return [r for _, r in kept]


def class_counts(rows):
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    return [c.get(name, 0) for name in LABEL_NAMES]


def inverse_freq_weights(rows):
    """Inverse-frequency class weights (normalised to mean 1) for `rows`."""
    counts = class_counts(rows)
    total = sum(counts)
    n_cls = len(counts)
    w = [(total / (n_cls * c)) if c > 0 else 0.0 for c in counts]
    mean_w = sum(w) / n_cls
    return [x / mean_w if mean_w > 0 else 1.0 for x in w]


# Fixed negation cue used by the "negation" augmentation. Deterministic, no net.
_NEGATE_MAP = [(" is ", " is not "), (" are ", " are not "),
               (" was ", " was not "), (" were ", " were not ")]


def augment_train(train_rows, mode="none"):
    """Return a deterministic, fixed-size transformed training corpus.

    mode "none"    -> unchanged.
    mode "swap"    -> swap premise and hypothesis for contradiction rows, whose
                      relation is symmetric; all other rows remain unchanged.
    mode "negation"-> replace entailment rows with a negated-hypothesis,
                      contradiction-labeled example when a conservative auxiliary
                      rewrite fires; all other rows remain unchanged.

    Replacement rather than append keeps the full-scale update count identical
    across arms. Evaluation data is never transformed.
    """
    if mode == "none":
        return list(train_rows)
    out = []
    if mode == "swap":
        for r in train_rows:
            if r["label"] == "contradiction":
                out.append({
                    "premise": r["hypothesis"],
                    "hypothesis": r["premise"],
                    "label": "contradiction",
                    "hard": False,
                })
            else:
                out.append(r)
        return out
    if mode == "negation":
        for r in train_rows:
            if r["label"] != "entailment":
                out.append(r)
                continue
            h = r["hypothesis"]
            new_h = None
            for a, b in _NEGATE_MAP:
                if a in h:
                    new_h = h.replace(a, b, 1)
                    break
            if new_h is not None:
                out.append({"premise": r["premise"], "hypothesis": new_h,
                            "label": "contradiction", "hard": False})
            else:
                out.append(r)
        return out
    raise SystemExit(f"unknown augmentation mode {mode!r}")


# ---------------------------------------------------------------------------
# Tokenizer + encoding
# ---------------------------------------------------------------------------

def build_tokenizer():
    from transformers import AutoTokenizer
    authenticate_model_assets()
    return AutoTokenizer.from_pretrained(
        model_path(), local_files_only=True, use_fast=True
    )


class ArrayFeatures:
    """Compact fixed-width NumPy storage used by the full 549k-row corpus."""

    def __init__(self, **fields):
        lengths = {len(value) for value in fields.values()}
        if len(lengths) != 1:
            raise ValueError("encoded NLI feature arrays have inconsistent lengths")
        self.fields = fields
        self.length = lengths.pop()

    def __len__(self):
        return self.length

    def batch(self, indices):
        return {name: values[indices] for name, values in self.fields.items()}


def encode_pairs_cross(rows, tok, max_length=None, mask_premise: bool = False):
    """CROSS-ENCODER encoding: premise + hypothesis as ONE joint sequence
    ``[CLS] premise [SEP] hypothesis [SEP]`` (single-tower, cross attention).

    ``max_length`` (optional) overrides the joint-sequence truncation cap
    (default ``2 * MAX_LEN``). Used by the truncation/sequence-length surface —
    a very short cap chops premise/hypothesis tokens and loses accuracy.

    Returns compact fixed-width arrays rather than one Python dictionary per row.
    """
    import numpy as np

    cap = int(max_length) if max_length is not None else MAX_SEQUENCE_LENGTH
    if cap <= 0 or cap > 512:
        raise ValueError(f"invalid NLI sequence length {cap}")
    premises = ["" if mask_premise else row["premise"] for row in rows]
    count = len(rows)
    input_ids = np.empty((count, cap), dtype=np.int32)
    attention_mask = np.empty((count, cap), dtype=np.uint8)
    chunk_size = 4096
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        encoded = tok(
            premises[start:stop],
            [row["hypothesis"] for row in rows[start:stop]],
            truncation=True,
            padding="max_length",
            max_length=cap,
            return_tensors="np",
            return_offsets_mapping=False,
        )
        input_ids[start:stop] = encoded["input_ids"]
        attention_mask[start:stop] = encoded["attention_mask"]
    labels = np.asarray([label_to_id(row["label"]) for row in rows], dtype=np.int64)
    return ArrayFeatures(input_ids=input_ids, attention_mask=attention_mask), labels


def encode_pairs_siamese(rows, tok, mask_premise: bool = False):
    """SIAMESE encoding: premise and hypothesis tokenised SEPARATELY (two towers
    share the encoder). Returns list of dict with premise_* and hypothesis_*
    input ids/masks plus the label id list.

    ``mask_premise`` (used only by the hypothesis-bias task) replaces the premise
    with an empty/pad-only sequence so the tower sees only the hypothesis — this
    is the HYPOTHESIS-ONLY shortcut model.
    """
    import numpy as np

    count = len(rows)
    p_ids = np.empty((count, MAX_LEN), dtype=np.int32)
    p_mask = np.empty((count, MAX_LEN), dtype=np.uint8)
    h_ids = np.empty((count, MAX_LEN), dtype=np.int32)
    h_mask = np.empty((count, MAX_LEN), dtype=np.uint8)
    chunk_size = 4096
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        premises = ["" if mask_premise else row["premise"]
                    for row in rows[start:stop]]
        hypotheses = [row["hypothesis"] for row in rows[start:stop]]
        prem = tok(
            premises, truncation=True, padding="max_length", max_length=MAX_LEN,
            return_tensors="np", return_offsets_mapping=False,
        )
        hyp = tok(
            hypotheses, truncation=True, padding="max_length", max_length=MAX_LEN,
            return_tensors="np", return_offsets_mapping=False,
        )
        p_ids[start:stop] = prem["input_ids"]
        p_mask[start:stop] = prem["attention_mask"]
        h_ids[start:stop] = hyp["input_ids"]
        h_mask[start:stop] = hyp["attention_mask"]
    labels = np.asarray([label_to_id(row["label"]) for row in rows], dtype=np.int64)
    return ArrayFeatures(
        p_input_ids=p_ids,
        p_attention_mask=p_mask,
        h_input_ids=h_ids,
        h_attention_mask=h_mask,
    ), labels


def _pad_batch(seqs, pad_id):
    import torch
    maxlen = max(len(s) for s in seqs)
    n = len(seqs)
    ids = torch.full((n, maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return ids


def collate_cross(batch_feats, batch_labels, pad_id, device):
    import torch
    if isinstance(batch_feats, dict):
        input_ids = torch.as_tensor(batch_feats["input_ids"], dtype=torch.long)
        attn = torch.as_tensor(batch_feats["attention_mask"], dtype=torch.long)
    else:
        input_ids = _pad_batch([f["input_ids"] for f in batch_feats], pad_id)
        attn = _pad_batch([f["attention_mask"] for f in batch_feats], 0)
    y = torch.as_tensor(batch_labels, dtype=torch.long)
    return ({"input_ids": input_ids.to(device),
             "attention_mask": attn.to(device)}, y.to(device))


def collate_siamese(batch_feats, batch_labels, pad_id, device):
    import torch
    if isinstance(batch_feats, dict):
        p_ids = torch.as_tensor(batch_feats["p_input_ids"], dtype=torch.long)
        p_att = torch.as_tensor(batch_feats["p_attention_mask"], dtype=torch.long)
        h_ids = torch.as_tensor(batch_feats["h_input_ids"], dtype=torch.long)
        h_att = torch.as_tensor(batch_feats["h_attention_mask"], dtype=torch.long)
    else:
        p_ids = _pad_batch([f["p_input_ids"] for f in batch_feats], pad_id)
        p_att = _pad_batch([f["p_attention_mask"] for f in batch_feats], 0)
        h_ids = _pad_batch([f["h_input_ids"] for f in batch_feats], pad_id)
        h_att = _pad_batch([f["h_attention_mask"] for f in batch_feats], 0)
    y = torch.as_tensor(batch_labels, dtype=torch.long)
    return ({"p_input_ids": p_ids.to(device),
             "p_attention_mask": p_att.to(device),
             "h_input_ids": h_ids.to(device),
             "h_attention_mask": h_att.to(device)}, y.to(device))


# ---------------------------------------------------------------------------
# Pooling: mean-pool a sentence vector from encoder hidden states (siamese).
# ---------------------------------------------------------------------------

def mean_pool(hidden, attention_mask):
    """Attention-masked mean pooling over token hidden states -> (B, H)."""
    import torch
    mask = attention_mask.unsqueeze(-1).type_as(hidden)  # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(hidden, attention_mask):
    """Take the first-token ([CLS]) hidden state as the sentence vector -> (B, H).

    This operation selects the first sequence position. It is evaluated under
    the same encoder, interaction, and training budget as the other supported
    pooling functions.
    """
    return hidden[:, 0, :]


def max_pool(hidden, attention_mask):
    """Attention-masked max pooling over token hidden states -> (B, H).

    Masked positions are set to -inf before the max so padding never wins.
    Max pooling keeps the largest activation per dimension. Its measured
    ordering relative to other pooling functions is verifier-only.
    """
    import torch
    mask = attention_mask.unsqueeze(-1).type_as(hidden)  # (B, T, 1)
    very_neg = torch.finfo(hidden.dtype).min
    masked = hidden.masked_fill(mask == 0, very_neg)
    return masked.max(dim=1).values


_POOLERS = {"cls": cls_pool, "mean": mean_pool, "max": max_pool}


def sum_pool(hidden, attention_mask):
    """Attention-masked SUM pooling over token hidden states -> (B, H).

    Unlike mean pooling, sum pooling does not length-normalize, so the sentence
    vector magnitude grows with sentence length. The verifier measures the
    downstream effect without publishing a preferred pooling operation here.
    All other model components remain fixed.
    """
    import torch
    mask = attention_mask.unsqueeze(-1).type_as(hidden)  # (B, T, 1)
    return (hidden * mask).sum(dim=1)


_POOLERS["sum"] = sum_pool


def get_pooler(mode: str):
    """Return the token->sentence pooling fn for `mode` (cls/mean/max/sum)."""
    if mode not in _POOLERS:
        raise SystemExit(f"unknown pooling mode {mode!r}; expected {list(_POOLERS)}")
    return _POOLERS[mode]


def interaction_vector(u, v, mode: str):
    """Combine siamese sentence vectors u (premise) and v (hypothesis).

    mode "concat"    -> [u; v]                       (2H)
    mode "infersent" -> [u; v; |u - v|; u * v]       (4H)   (Conneau et al. 2017)
    """
    import torch
    if mode == "concat":
        return torch.cat([u, v], dim=-1)
    if mode == "infersent":
        return torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
    raise SystemExit(f"unknown interaction mode {mode!r}")


def interaction_out_dim(H, mode: str) -> int:
    return {"concat": 2 * H, "infersent": 4 * H}[mode]


# ---------------------------------------------------------------------------
# Classifier head builder (shared): a linear head, or an MLP head with one
# hidden layer. Used by the classifier-depth / regularization surfaces.
# ---------------------------------------------------------------------------

def build_classifier_head(in_dim, num_labels, device, *, hidden=0,
                          dropout=0.1, activation="gelu"):
    """Return an nn.Module mapping (B, in_dim) -> (B, num_labels).

    hidden == 0  -> a single Linear (the fixed default head).
    hidden  > 0  -> Linear(in,hidden) + activation + Dropout + Linear(hidden,out)
                    (a 2-layer MLP classification head).
    ``dropout`` is applied before the final projection in both cases.
    """
    import torch.nn as nn
    acts = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    if hidden and int(hidden) > 0:
        return nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            acts.get(activation, nn.GELU)(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden), num_labels),
        ).to(device)
    return nn.Sequential(nn.Dropout(dropout),
                         nn.Linear(in_dim, num_labels)).to(device)


# ---------------------------------------------------------------------------
# Loss builder (shared): plain cross-entropy, label-smoothed CE, focal loss,
# and optional per-class weighting. Used by the loss / class-weighting surfaces.
# ---------------------------------------------------------------------------

def build_loss(kind="ce", *, label_smoothing=0.0, focal_gamma=2.0,
               class_weights=None, device=None):
    """Return a callable loss_fn(logits, target) -> scalar tensor.

    kind "ce"    -> nn.CrossEntropyLoss (optionally label-smoothed / weighted).
    kind "focal" -> focal loss (Lin et al., 2017): down-weights easy examples by
                    (1 - p_t)**gamma; optionally class-weighted.
    ``class_weights`` (list[float] over the 3 labels) reweights the per-class
    contribution (useful under class imbalance).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    w = None
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32)
        if device is not None:
            w = w.to(device)

    if kind == "ce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=float(label_smoothing))

    if kind == "focal":
        gamma = float(focal_gamma)

        def focal(logits, target):
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
            p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
            loss = -((1.0 - p_t) ** gamma) * logp_t
            if w is not None:
                loss = loss * w[target]
            return loss.mean()

        return focal

    raise SystemExit(f"unknown loss kind {kind!r}; expected 'ce'/'focal'")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CrossEncoderNLI:
    """Single-tower cross-encoder: joint [CLS] p [SEP] h -> [CLS] -> linear."""

    def __init__(self, num_labels, device, *, dropout=0.1, head_hidden=0):
        import torch.nn as nn
        from transformers import AutoModel
        self.device = device
        self.architecture = "cross"
        self.encoder = AutoModel.from_pretrained(
            model_path(), local_files_only=True, use_safetensors=True
        ).to(device)
        H = self.encoder.config.hidden_size
        self.classifier = build_classifier_head(
            H, num_labels, device, hidden=head_hidden, dropout=dropout)

    def trainable_params(self):
        return (list(self.encoder.parameters())
                + list(self.classifier.parameters()))

    def train(self):
        self.encoder.train(); self.classifier.train()

    def eval(self):
        self.encoder.eval(); self.classifier.eval()

    def forward(self, batch):
        out = self.encoder(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"])
        cls = out.last_hidden_state[:, 0, :]     # [CLS]
        return self.classifier(cls)


class SiameseNLI:
    """Shared-encoder bi-encoder: encode premise & hypothesis separately, pool
    to u, v, form an interaction vector, then a classifier head."""

    def __init__(self, num_labels, interaction_mode, device, *,
                 pooling="mean", dropout=0.1, head_hidden=0):
        from transformers import AutoModel
        self.device = device
        self.architecture = "siamese"
        self.interaction_mode = interaction_mode
        self.pooler = get_pooler(pooling)
        self.encoder = AutoModel.from_pretrained(
            model_path(), local_files_only=True, use_safetensors=True
        ).to(device)
        H = self.encoder.config.hidden_size
        self.classifier = build_classifier_head(
            interaction_out_dim(H, interaction_mode), num_labels, device,
            hidden=head_hidden, dropout=dropout)

    def trainable_params(self):
        return (list(self.encoder.parameters())
                + list(self.classifier.parameters()))

    def train(self):
        self.encoder.train(); self.classifier.train()

    def eval(self):
        self.encoder.eval(); self.classifier.eval()

    def _encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return self.pooler(out.last_hidden_state, mask)

    def forward(self, batch):
        u = self._encode(batch["p_input_ids"], batch["p_attention_mask"])
        v = self._encode(batch["h_input_ids"], batch["h_attention_mask"])
        feat = interaction_vector(u, v, self.interaction_mode)
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# Train + evaluate (FIXED budget + FIXED 3-way accuracy evaluator).
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    if not y_true or len(y_true) != len(y_pred):
        print(f"SURFACE_ERROR NLI prediction/label count {len(y_pred)}/{len(y_true)}",
              flush=True)
        raise ValueError("incomplete NLI predictions")
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    value = correct / len(y_true)
    if not math.isfinite(value):
        print(f"NLI_NONFINITE accuracy={value}", flush=True)
        raise ValueError("non-finite NLI accuracy")
    return value


def majority_class_accuracy(train_rows, test_rows, eval_mask=None):
    """Label-blind majority-class diagnostic.

    Predicts the single most frequent training label for every evaluation row.
    It follows the same completeness and finite-result checks as trained models.
    No score mapping or baseline ordering is exposed here.
    """
    from collections import Counter
    if not train_rows or not test_rows:
        print("SURFACE_ERROR NLI majority path received an empty split", flush=True)
        raise ValueError("empty NLI split")
    train_ids = [label_to_id(r["label"]) for r in train_rows]
    maj = Counter(train_ids).most_common(1)[0][0]
    y_true = [label_to_id(r["label"]) for r in test_rows]
    y_pred = [maj] * len(y_true)
    if eval_mask is not None:
        if len(eval_mask) != len(y_true):
            print("SURFACE_ERROR NLI eval mask length mismatch", flush=True)
            raise ValueError("NLI eval mask length mismatch")
        y_true = [t for t, m in zip(y_true, eval_mask) if m]
        y_pred = [p for p, m in zip(y_pred, eval_mask) if m]
    return accuracy(y_true, y_pred)


def _iter_batches(feats, labels, bs, shuffle, seed_epoch):
    import numpy as np
    idx = np.arange(len(feats))
    if shuffle:
        rng = np.random.RandomState(seed_epoch)
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        if hasattr(feats, "batch"):
            yield feats.batch(j), labels[j]
        else:
            yield [feats[k] for k in j], [labels[k] for k in j]


def train_model(model, tok, train_feats, train_labels, *, collate_fn,
                encoder_lr, head_lr, seed, verbose_tag="model", max_epochs=None,
                loss_fn=None, weight_decay=None,
                sequence_length=MAX_SEQUENCE_LENGTH):
    """Train exactly once and emit authenticated optimizer-completion proofs."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    device = model.device
    if not train_feats or len(train_feats) != len(train_labels):
        print("SURFACE_ERROR NLI incomplete training features or labels", flush=True)
        raise ValueError("incomplete NLI training data")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    n_epochs = int(max_epochs) if max_epochs is not None else MAX_EPOCHS
    if n_epochs <= 0:
        raise ValueError("NLI epoch count must be positive")
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    wd = WEIGHT_DECAY if weight_decay is None else float(weight_decay)

    enc_params = list(model.encoder.parameters())
    if float(encoder_lr) == 0.0:
        for parameter in enc_params:
            parameter.requires_grad_(False)
    enc_ids = {id(parameter) for parameter in enc_params}
    head_params = [p for p in model.trainable_params()
                   if id(p) not in enc_ids and p.requires_grad]
    parameter_groups = []
    trainable_encoder = [p for p in enc_params if p.requires_grad]
    if trainable_encoder:
        parameter_groups.append({"params": trainable_encoder, "lr": float(encoder_lr)})
    if head_params:
        parameter_groups.append({"params": head_params, "lr": float(head_lr)})
    if not parameter_groups:
        raise ValueError("NLI model has no trainable parameters")
    trainable = [parameter for group in parameter_groups
                 for parameter in group["params"]]
    opt = torch.optim.AdamW(parameter_groups, weight_decay=wd)
    steps_per_epoch = (len(train_feats) + TRAIN_BATCH - 1) // TRAIN_BATCH
    total_steps = steps_per_epoch * n_epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(WARMUP_FRAC * total_steps), total_steps)

    print(
        f"NLI_TRAIN mode={verbose_tag} optimizer=adamw "
        f"encoder_lr={float(encoder_lr):.8g} head_lr={float(head_lr):.8g} "
        f"weight_decay={wd:.8g} warmup_ratio={WARMUP_FRAC:.8g} "
        f"epochs={n_epochs} batch={TRAIN_BATCH} "
        f"max_length={int(sequence_length)} expected_steps={total_steps}",
        flush=True,
    )

    optimizer_steps = 0
    for epoch in range(n_epochs):
        model.train()
        if not trainable_encoder:
            model.encoder.eval()
        running = 0.0
        for bf, bl in _iter_batches(train_feats, train_labels, TRAIN_BATCH,
                                    True, seed + epoch):
            batch, y = collate_fn(bf, bl, pad_id, device)
            logits = model.forward(batch)
            if not torch.isfinite(logits).all():
                print(f"NLI_NONFINITE train_logits epoch={epoch}", flush=True)
                raise FloatingPointError("non-finite NLI train logits")
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                print(f"NLI_NONFINITE train_loss epoch={epoch}", flush=True)
                raise FloatingPointError("non-finite NLI train loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                print(f"NLI_NONFINITE grad_norm epoch={epoch}", flush=True)
                raise FloatingPointError("non-finite NLI gradients")
            opt.step()
            sched.step()
            running += float(loss.detach())
            optimizer_steps += 1
        if any(not torch.isfinite(parameter).all() for parameter in trainable):
            print(f"NLI_NONFINITE parameters epoch={epoch + 1}", flush=True)
            raise FloatingPointError("non-finite NLI parameters")
        expected_steps = (epoch + 1) * steps_per_epoch
        print(
            f"NLI_EPOCH mode={verbose_tag} epoch={epoch + 1} "
            f"optimizer_steps={optimizer_steps} expected={expected_steps} "
            f"loss={running / max(1, steps_per_epoch):.6f}",
            flush=True,
        )
    print(
        f"NLI_TRAIN_DONE epochs={n_epochs} optimizer_steps={optimizer_steps} "
        f"expected_steps={total_steps}",
        flush=True,
    )
    return model


def score_model(model, tok, test_feats, test_labels, *, collate_fn,
                eval_mask=None, return_logits=False):
    """Evaluate one complete domain without modifying the trained model."""
    import numpy as np
    import torch

    if not test_feats or len(test_feats) != len(test_labels):
        print("SURFACE_ERROR NLI incomplete evaluation features or labels", flush=True)
        raise ValueError("incomplete NLI evaluation data")
    if eval_mask is not None:
        if len(eval_mask) != len(test_labels) or not any(eval_mask):
            print("SURFACE_ERROR NLI invalid or empty eval mask", flush=True)
            raise ValueError("invalid NLI eval mask")
    device = model.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    model.eval()
    y_true, y_pred, logit_rows = [], [], []
    with torch.no_grad():
        for bf, bl in _iter_batches(test_feats, test_labels, EVAL_BATCH,
                                    False, 0):
            batch, y = collate_fn(bf, bl, pad_id, device)
            logits = model.forward(batch)
            if not torch.isfinite(logits).all():
                print("NLI_NONFINITE eval_logits", flush=True)
                raise FloatingPointError("non-finite NLI eval logits")
            pred = logits.argmax(dim=-1)
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
            if return_logits:
                logit_rows.append(logits.detach().float().cpu().numpy())

    if eval_mask is not None:
        yt = [t for t, m in zip(y_true, eval_mask) if m]
        yp = [p for p, m in zip(y_pred, eval_mask) if m]
        acc = accuracy(yt, yp)
    else:
        acc = accuracy(y_true, y_pred)

    if not math.isfinite(acc):
        print(f"NLI_NONFINITE final_accuracy={acc}", flush=True)
        raise FloatingPointError("non-finite NLI final accuracy")

    if return_logits:
        logits_arr = np.concatenate(logit_rows, axis=0) if logit_rows else np.zeros((0, NUM_LABELS))
        return acc, np.asarray(y_true), logits_arr
    return acc


def train_and_score(model, tok, train_feats, train_labels, test_feats,
                    test_labels, *, collate_fn, encoder_lr, head_lr, seed,
                    verbose_tag="model", max_epochs=None, eval_mask=None,
                    loss_fn=None, weight_decay=None, return_logits=False):
    """Compatibility wrapper for sibling harnesses that evaluate one domain."""
    train_model(
        model, tok, train_feats, train_labels, collate_fn=collate_fn,
        encoder_lr=encoder_lr, head_lr=head_lr, seed=seed,
        verbose_tag=verbose_tag, max_epochs=max_epochs, loss_fn=loss_fn,
        weight_decay=weight_decay,
    )
    return score_model(
        model, tok, test_feats, test_labels, collate_fn=collate_fn,
        eval_mask=eval_mask, return_logits=return_logits,
    )


def emit_setting_result(setting: str, value: float, expected_rows: int) -> None:
    print(
        f"NLI_METRICS setting={setting} acc={value:.8f} n_eval={expected_rows}",
        flush=True,
    )
    print(
        f"NLI_SETTING_DONE setting={setting} predicted={expected_rows} "
        f"expected={expected_rows}",
        flush=True,
    )


def evaluate_all_domains(model, tok, *, encode_rows, collate_fn,
                         eval_mask_for_rows=None) -> int:
    """Evaluate one trained model on every authenticated domain in sequence."""
    import gc

    completed_rows = 0
    for setting in DOMAINS:
        rows = load_evaluation_data(setting)
        expected = DATASET_INVENTORY[setting]["rows"]
        features, labels = encode_rows(rows)
        mask = eval_mask_for_rows(rows) if eval_mask_for_rows else None
        value = score_model(
            model, tok, features, labels, collate_fn=collate_fn, eval_mask=mask
        )
        emit_setting_result(setting, value, expected)
        completed_rows += expected
        del rows, features, labels
        gc.collect()
    print(
        f"NLI_EVAL_DONE settings={len(DOMAINS)} rows={completed_rows}",
        flush=True,
    )
    return completed_rows


def evaluate_majority_all_domains(train_rows) -> int:
    completed_rows = 0
    print("NLI_TRAIN_DONE epochs=0 optimizer_steps=0 expected_steps=0", flush=True)
    for setting in DOMAINS:
        rows = load_evaluation_data(setting)
        expected = DATASET_INVENTORY[setting]["rows"]
        value = majority_class_accuracy(train_rows, rows)
        emit_setting_result(setting, value, expected)
        completed_rows += expected
    print(
        f"NLI_EVAL_DONE settings={len(DOMAINS)} rows={completed_rows}",
        flush=True,
    )
    return completed_rows


def emit_final_completion(*, seed: int, elapsed: float,
                          completed_rows: int) -> None:
    print(
        f"NLI_DONE settings={len(DOMAINS)} "
        f"train_rows={DATASET_INVENTORY['snli_train']['rows']} "
        f"eval_rows={completed_rows} seed={seed} elapsed={elapsed:.1f}",
        flush=True,
    )
