"""Quantization-Aware Training (QAT) for Pythia-1.4B -- finetune + evaluate.

This script:
  1. Loads pretrained Pythia-1.4B (HF ``EleutherAI/pythia-1.4b``).
  2. Replaces every nn.Linear with QATWrapper that applies fake-quant in
     forward (so gradients can flow back through the quantization).
  3. Runs a QAT fine-tune on WikiText-2 train (default ~1500 steps).
  4. Applies a REAL quantize-dequantize roundtrip to every linear weight.
  5. Evaluates perplexity on WikiText-2 test.

The QAT algorithm is defined in the EDITABLE REGION below.  Everything
else (data loading, training loop, real-quant roundtrip, perplexity eval)
is fixed and shared by every baseline and the agent.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE REGION START -- QAT Algorithm (lines 33-176)
# ═══════════════════════════════════════════════════════════════════════════════

# Per-method training hyperparameters.  The training loop reads this dict.
# Override any of these in your method to retune.
CONFIG_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_steps": 500,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "weight_decay": 0.0,
}


def _qrange(num_bits):
    """Symmetric integer range for `num_bits`-bit signed quantization."""
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def fake_quantize_weight(weight, num_bits, group_size):
    """Differentiable fake-quant of a 2D weight tensor.

    Forward: simulates `num_bits` symmetric per-group quantization.
    Backward: straight-through estimator (gradient passes through unchanged).

    Args:
        weight: float tensor of shape (out_features, in_features).
        num_bits: bit width.
        group_size: column group size (>0); in_features must be divisible.

    Returns:
        Tensor of same shape and dtype as `weight`, quantize-dequantized.
    """
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, (
        f"in_features {in_features} not divisible by group_size {group_size}"
    )
    w = weight.float().reshape(out_features, -1, group_size)
    w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = w_max / qmax
    w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    # Straight-through estimator: forward = quantized, backward = identity.
    w_dq = w + (w_q - w).detach()
    return w_dq.reshape(out_features, in_features).to(weight.dtype)


def fake_quantize_activation(x, num_bits):
    """Default identity (weight-only QAT).  Override to add activation QAT."""
    return x


def quantize_dequantize_weight(weight, num_bits, group_size):
    """REAL (non-differentiable) symmetric per-group QDQ for post-training.

    Used after QAT finetune to materialize the quantized weights for eval.
    Returns the same shape/dtype as `weight`.
    """
    qmin, qmax = _qrange(num_bits)
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    with torch.no_grad():
        w = weight.float().reshape(out_features, -1, group_size)
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = w_max / qmax
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
        return w_q.reshape(out_features, in_features).to(weight.dtype)


class QATWrapper(nn.Module):
    """Wraps an nn.Linear and applies fake-quant to its weight in forward.

    The wrapped module exposes the original Linear's weight/bias as
    submodule parameters so the QAT optimizer can update them; the bias
    is left in full precision.

    Attributes
    ----------
    linear : nn.Linear
        Underlying linear layer.  `linear.weight` is the trainable param.
    num_bits : int
    group_size : int
    """

    def __init__(self, linear, num_bits, group_size):
        super().__init__()
        self.linear = linear
        self.num_bits = num_bits
        self.group_size = group_size

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        x = fake_quantize_activation(x, self.num_bits)
        w_q = fake_quantize_weight(self.linear.weight, self.num_bits, self.group_size)
        return F.linear(x, w_q, self.linear.bias)


def prepare_qat_model(model, num_bits, group_size):
    """Replace every nn.Linear in `model` with a QATWrapper in-place.

    The LM head (``model.lm_head`` for GPT-style, ``model.embed_out`` for
    Pythia / GPTNeoX) is restored to a plain Linear after the recursive
    replace so the output projection stays in full precision.  HF GPT-2
    Conv1D layers are converted to nn.Linear before wrapping.
    """
    from transformers.pytorch_utils import Conv1D  # type: ignore

    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                wrapper = QATWrapper(child, num_bits=num_bits, group_size=group_size)
                setattr(parent, name, wrapper)
            elif isinstance(child, Conv1D):
                # Convert Conv1D -> Linear (Conv1D weight is (in, out), Linear is (out, in)).
                in_f, out_f = child.weight.shape
                lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
                                device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    lin.weight.copy_(child.weight.t().contiguous())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                wrapper = QATWrapper(lin, num_bits=num_bits, group_size=group_size)
                setattr(parent, name, wrapper)
            else:
                _replace(child)

    _replace(model)
    # Restore the LM head to full precision (covers GPT-2 `lm_head` and
    # Pythia / GPTNeoX `embed_out`).
    for head_attr in ("lm_head", "embed_out"):
        head = getattr(model, head_attr, None)
        if isinstance(head, QATWrapper):
            setattr(model, head_attr, head.linear)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE REGION END
# ═══════════════════════════════════════════════════════════════════════════════


# ── Model loading ─────────────────────────────────────────────────────────────

def get_model(model_path):
    """Load model in float32 for QAT training stability."""
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.config.use_cache = False
    model.seqlen = 2048
    return model


def find_qat_wrappers(module, prefix=""):
    """Return dict {name: QATWrapper} of all QAT-wrapped layers."""
    out = {}
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, QATWrapper):
            out[full] = child
        else:
            out.update(find_qat_wrappers(child, full))
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_wikitext2(tokenizer, seqlen, split):
    from datasets import load_dataset, Dataset
    import glob

    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/wikitext2")
    try:
        data = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir
        )
    except Exception:
        # Fallback: read arrow file directly
        arrow = glob.glob(f"{cache_dir}/**/wikitext-{split}.arrow", recursive=True)
        if not arrow:
            raise FileNotFoundError(f"WikiText-2 {split} not found in {cache_dir}")
        data = Dataset.from_file(arrow[0])

    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    return enc.input_ids  # (1, total_tokens)


def make_train_batches(ids, batch_size, seqlen, num_steps, gradient_accumulation_steps, seed):
    """Generator yielding randomly-sampled (input, target) blocks of length seqlen."""
    rng = np.random.RandomState(seed)
    total = ids.shape[1]
    n_required = num_steps * gradient_accumulation_steps * batch_size
    starts = rng.randint(0, total - seqlen - 1, size=n_required)
    for k in range(num_steps * gradient_accumulation_steps):
        batch = []
        for b in range(batch_size):
            i = int(starts[k * batch_size + b])
            batch.append(ids[0, i:i + seqlen + 1])
        x = torch.stack([t[:-1] for t in batch], dim=0)
        y = torch.stack([t[1:]  for t in batch], dim=0)
        yield x, y


# ── Training loop ─────────────────────────────────────────────────────────────

def train_qat(model, tokenizer, dev, num_bits, group_size, seed):
    cfg = {
        "learning_rate": 2e-5,
        "num_steps": 500,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "warmup_steps": 50,
        "weight_decay": 0.0,
    }
    cfg.update(CONFIG_OVERRIDES)

    ids = load_wikitext2(tokenizer, model.seqlen, split="train").to(dev)

    # Optimizer over all trainable parameters (includes any extras the
    # editable region added, e.g., LSQ scales or AdaRound betas).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable,
        lr=cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=cfg["weight_decay"],
    )

    def lr_at(step):
        if step < cfg["warmup_steps"]:
            return cfg["learning_rate"] * (step + 1) / max(1, cfg["warmup_steps"])
        # Cosine decay to 10% of base lr
        progress = (step - cfg["warmup_steps"]) / max(1, cfg["num_steps"] - cfg["warmup_steps"])
        return cfg["learning_rate"] * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    model.train()
    batches = make_train_batches(
        ids, cfg["batch_size"], model.seqlen,
        cfg["num_steps"], cfg["gradient_accumulation_steps"], seed,
    )
    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    micro = 0
    step = 0
    running_loss = 0.0
    running_aux = 0.0
    for x, y in batches:
        x = x.to(dev); y = y.to(dev)
        logits = model(x).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
        )
        # Sum any auxiliary losses contributed by per-module ``aux_loss``
        # hooks (e.g. PACT alpha L2, AdaRound beta-annealed regularizer).
        # Modules without an ``aux_loss`` callable are unaffected.
        _aux = 0.0
        for _m in model.modules():
            _al = getattr(_m, "aux_loss", None)
            if callable(_al):
                _v = _al(step=step, total_steps=cfg["num_steps"])
                if _v is not None:
                    _aux = _aux + _v
        loss = loss + _aux
        (loss / cfg["gradient_accumulation_steps"]).backward()
        running_loss += loss.item()
        running_aux += float(_aux) if isinstance(_aux, (int, float)) else float(_aux.detach().item())
        micro += 1
        if micro == cfg["gradient_accumulation_steps"]:
            torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            optim.step()
            optim.zero_grad(set_to_none=True)
            if (step + 1) % 25 == 0 or step == 0:
                avg = running_loss / max(1, micro)
                avg_aux = running_aux / max(1, micro)
                print(
                    f"TRAIN_METRICS: step={step+1}/{cfg['num_steps']} "
                    f"loss={avg:.4f} aux={avg_aux:.4f} lr={lr_at(step):.2e} "
                    f"elapsed={time.time()-t0:.1f}",
                    flush=True,
                )
            running_loss = 0.0
            running_aux = 0.0
            micro = 0
            step += 1
            if step >= cfg["num_steps"]:
                break

    return time.time() - t0


# ── Real-quant materialization ────────────────────────────────────────────────

@torch.no_grad()
def apply_real_quantization(model, num_bits, group_size):
    """After QAT, replace each QATWrapper weight with the real QDQ value.

    The wrapper still applies fake-quant in forward, but with the weight
    already materialized to the quantization grid the result is the true
    INT-N model output (no train-time noise / scale drift).
    """
    wrappers = find_qat_wrappers(model)
    for name, w in wrappers.items():
        w_dq = quantize_dequantize_weight(w.linear.weight.data, num_bits, group_size)
        w.linear.weight.data.copy_(w_dq)
    return len(wrappers)


# ── Perplexity evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dev, seqlen):
    model.eval()
    ids = load_wikitext2(tokenizer, seqlen, split="test").to(dev)
    nsamples = ids.shape[1] // seqlen
    if nsamples == 0:
        return float("nan")
    nlls = []
    for i in range(nsamples):
        x = ids[:, i * seqlen:(i + 1) * seqlen]
        logits = model(x).logits
        shift_logits = logits[:, :-1, :].float().contiguous()
        shift_labels = x[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.float() * (seqlen - 1))
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
    return ppl.item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="QAT for Pythia-1.4B")
    p.add_argument("--model-path", type=str, default="/data/pythia-1.4b")
    p.add_argument("--num-bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    overall_t0 = time.time()

    print(f"Loading model from {args.model_path}...", flush=True)
    model = get_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.seqlen = args.seqlen

    # Enable gradient checkpointing to fit Pythia-1.4B + AdamW on 80GB.
    try:
        model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"warn: gradient_checkpointing_enable failed: {e}", flush=True)

    # FP32 baseline ppl
    print("\n=== FP baseline evaluation ===", flush=True)
    model.to(dev)
    fp_ppl = evaluate_perplexity(model, tokenizer, dev, args.seqlen)
    print(f"FP baseline perplexity: {fp_ppl:.4f}", flush=True)
    print(f"TRAIN_METRICS: fp_perplexity={fp_ppl:.4f}", flush=True)

    # Wrap model for QAT
    print(f"\n=== Preparing QAT (INT{args.num_bits}, group_size={args.group_size}) ===", flush=True)
    model = prepare_qat_model(model, num_bits=args.num_bits, group_size=args.group_size)
    model.to(dev)
    n_wrapped = len(find_qat_wrappers(model))
    print(f"Wrapped {n_wrapped} linear layers as QATWrapper", flush=True)

    # QAT finetune
    print("\n=== QAT fine-tuning ===", flush=True)
    qat_time = train_qat(model, tokenizer, dev, args.num_bits, args.group_size, args.seed)
    print(f"QAT finetune done in {qat_time:.1f}s", flush=True)

    # Real-quant roundtrip
    print("\n=== Materializing real INT-N weights ===", flush=True)
    n_q = apply_real_quantization(model, args.num_bits, args.group_size)
    print(f"Quantized {n_q} layers to INT{args.num_bits}", flush=True)

    # Quantized ppl
    print("\n=== Quantized evaluation ===", flush=True)
    q_ppl = evaluate_perplexity(model, tokenizer, dev, args.seqlen)

    elapsed = time.time() - overall_t0
    degradation = q_ppl - fp_ppl
    print(f"\n=== Results ===", flush=True)
    print(f"FP   perplexity: {fp_ppl:.4f}", flush=True)
    print(f"INT{args.num_bits} perplexity: {q_ppl:.4f}", flush=True)
    print(f"Degradation:     {degradation:.4f}", flush=True)
    print(
        f"TEST_METRICS: wikitext2_ppl={q_ppl:.4f} fp16_ppl={fp_ppl:.4f} "
        f"degradation={degradation:.4f} qat_time={qat_time:.1f} elapsed={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
