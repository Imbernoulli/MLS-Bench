"""Post-Training Quantization (PTQ) for LLMs -- quantize + evaluate pipeline.

This script loads a pretrained LLM (Mistral-7B-v0.1), applies INT4 weight
quantization using a custom algorithm, and evaluates perplexity on WikiText-2.

The quantization algorithm is defined in the EDITABLE REGION below.
Everything else (model loading, calibration data, evaluation) is fixed.
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
# EDITABLE REGION START -- Quantization Algorithm (lines 26-157)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Helper: basic quantize/dequantize primitives ──────────────────────────────

def quantize_tensor(x, scale, zero_point, qmin, qmax):
    """Quantize a float tensor to integers given scale and zero point."""
    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return x_int


def dequantize_tensor(x_int, scale, zero_point):
    """Dequantize integer tensor back to float."""
    return (x_int - zero_point) * scale


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    """Compute per-channel (or per-group) quantization parameters.

    Args:
        weight: float tensor of shape (out_features, in_features)
        num_bits: number of quantization bits
        group_size: if > 0, compute params per group of columns; else per-row
        symmetric: if True, use symmetric quantization (zero_point = 0)

    Returns:
        scale: float tensor broadcastable to weight shape
        zero_point: float tensor broadcastable to weight shape
        qmin, qmax: integer quantization range
    """
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1

    if group_size > 0:
        # Reshape weight into groups for per-group quantization
        out_features, in_features = weight.shape
        assert in_features % group_size == 0, \
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        w_groups = weight.reshape(out_features, -1, group_size)

        if symmetric:
            w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = w_groups.amin(dim=-1, keepdim=True)
            w_max = w_groups.amax(dim=-1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)

        scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
        zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    else:
        # Per-channel (per output row)
        if symmetric:
            w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = weight.amin(dim=1, keepdim=True)
            w_max = weight.amax(dim=1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)

    return scale, zero_point, qmin, qmax


class LayerQuantizer:
    """Quantizes a single nn.Linear layer's weights to low-bit integers.

    This class encapsulates the quantization algorithm. Override the
    `quantize` method to implement custom quantization strategies.

    The calibration data (layer inputs) is provided via `add_batch()`.
    The `quantize()` method uses the collected statistics to quantize
    the weight matrix and returns the quantized-dequantized weight.

    Args:
        layer: nn.Linear module to quantize
        num_bits: target bit width (default: 4)
        group_size: quantization group size; -1 for per-channel (default: -1)
    """

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device

        # Accumulate Hessian (X^T X) for calibration
        self.nsamples = 0
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=self.dev, dtype=torch.float32
        )

    def add_batch(self, inp):
        """Accumulate calibration statistics from a batch of layer inputs.

        Args:
            inp: input tensor of shape (batch, seq_len, in_features) or
                 (batch * seq_len, in_features)
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        n = inp.shape[0]
        inp = inp.float()
        self.H += inp.T @ inp
        self.nsamples += n

    def quantize(self):
        """Quantize the layer weights and return the quantized-dequantized weight.

        Default implementation: simple round-to-nearest (RTN) quantization.
        Override this to implement better algorithms (e.g., GPTQ, AWQ).

        Returns:
            Quantized-dequantized weight tensor of same shape as original weight.
        """
        W = self.layer.weight.data.clone().float()
        scale, zero_point, qmin, qmax = find_scale_zero(
            W, num_bits=self.num_bits, group_size=self.group_size, symmetric=True
        )
        W_q = quantize_tensor(W, scale, zero_point, qmin, qmax)
        W_dq = dequantize_tensor(W_q, scale, zero_point)
        return W_dq.to(self.layer.weight.dtype)

    def free(self):
        """Release calibration buffers."""
        del self.H
        self.H = None


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE REGION END
# ═══════════════════════════════════════════════════════════════════════════════


# ── Model loading ─────────────────────────────────────────────────────────────

def get_model(model_path):
    """Load a pretrained causal LM with weight initialization skipped."""
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    model.seqlen = min(getattr(model.config, "max_position_embeddings", 4096), 4096)
    model.eval()
    return model


def find_linear_layers(module, prefix=""):
    """Recursively find all nn.Linear layers in the model."""
    result = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            result[full_name] = child
        else:
            result.update(find_linear_layers(child, full_name))
    return result


# ── Calibration data ──────────────────────────────────────────────────────────

def get_calibration_data(tokenizer, nsamples=128, seqlen=2048, seed=0):
    """Load WikiText-2 calibration data."""
    from datasets import load_dataset

    # Load from pre-downloaded cache (compute nodes have no network)
    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/wikitext2")
    try:
        traindata = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir
        )
    except Exception:
        # Fallback: load directly from arrow files in cache
        from datasets import Dataset
        import glob
        arrow = glob.glob(f"{cache_dir}/**/wikitext-train.arrow", recursive=True)
        if arrow:
            traindata = Dataset.from_file(arrow[0])
        else:
            raise FileNotFoundError(f"WikiText-2 train data not found in {cache_dir}")

    import random
    random.seed(seed)

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    return trainloader


def get_eval_data(tokenizer, seqlen=2048):
    """Load WikiText-2 test data for perplexity evaluation."""
    from datasets import load_dataset

    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/wikitext2")
    try:
        testdata = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir
        )
    except Exception:
        from datasets import Dataset
        import glob
        arrow = glob.glob(f"{cache_dir}/**/wikitext-test.arrow", recursive=True)
        if arrow:
            testdata = Dataset.from_file(arrow[0])
        else:
            raise FileNotFoundError(f"WikiText-2 test data not found in {cache_dir}")

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return testenc


# ── Layer-by-layer quantization ───────────────────────────────────────────────

@torch.no_grad()
def quantize_model(model, calibration_data, dev, num_bits=4, group_size=-1):
    """Quantize all linear layers in the model using LayerQuantizer.

    Processes the model layer-by-layer (transformer block by block) to
    minimize GPU memory usage. For each block:
      1. Move block to GPU
      2. Run calibration data through to collect Hessian statistics
      3. Quantize each linear sublayer using LayerQuantizer
      4. Replace weights with quantized-dequantized values
      5. Move block back to CPU

    Args:
        model: pretrained causal LM
        calibration_data: list of input_ids tensors for calibration
        dev: torch device (GPU)
        num_bits: target bit width
        group_size: quantization group size; -1 for per-channel

    Returns:
        dict mapping layer name -> quantization error (Frobenius norm)
    """
    print("Starting quantization...", flush=True)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    nsamples = len(calibration_data)
    seqlen = calibration_data[0].shape[1]
    hidden_size = model.config.hidden_size

    # Capture inputs to first layer
    inps = torch.zeros(
        (nsamples, seqlen, hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calibration_data:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    quant_errors = {}

    for i in range(len(layers)):
        print(f"Quantizing layer {i}/{len(layers)}...", flush=True)
        layer = layers[i].to(dev)

        # Find all linear sublayers in this transformer block
        subset = find_linear_layers(layer)

        # Create quantizers and register hooks to collect calibration stats
        quantizers = {}
        for name in subset:
            quantizers[name] = LayerQuantizer(subset[name], num_bits=num_bits, group_size=group_size)

        def make_hook(name):
            def hook(_, inp, out):
                quantizers[name].add_batch(inp[0].data)
            return hook

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(make_hook(name)))

        # Run calibration data through this layer
        for j in range(nsamples):
            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            position_embeddings = cache.get("position_embeddings")
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]

        for h in handles:
            h.remove()

        # Quantize each sublayer
        for name in subset:
            W_orig = subset[name].weight.data.clone()
            W_quant = quantizers[name].quantize()
            error = (W_orig.float() - W_quant.float()).norm().item()
            quant_errors[f"layers.{i}.{name}"] = error
            subset[name].weight.data = W_quant
            quantizers[name].free()

        # Re-run calibration through quantized layer to get outputs for next layer
        for j in range(nsamples):
            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            position_embeddings = cache.get("position_embeddings")
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        del quantizers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    print("Quantization complete.", flush=True)
    return quant_errors


# ── Perplexity evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_perplexity(model, testenc, dev):
    """Evaluate perplexity on test data (layer-by-layer to save memory).

    Args:
        model: (possibly quantized) causal LM
        testenc: tokenized test data
        dev: torch device

    Returns:
        float perplexity value
    """
    print("Evaluating perplexity...", flush=True)
    testenc = testenc.input_ids
    seqlen = model.seqlen
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size
    inps = torch.zeros(
        (nsamples, seqlen, hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            position_embeddings = cache.get("position_embeddings")
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print(f"Perplexity: {ppl.item():.4f}", flush=True)
    return ppl.item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PTQ for LLMs")
    parser.add_argument("--model-path", type=str, default="/data/mistral-7b-v01",
                        help="Path to pretrained model weights")
    parser.add_argument("--num-bits", type=int, default=4,
                        help="Quantization bit width")
    parser.add_argument("--group-size", type=int, default=-1,
                        help="Quantization group size (-1 for per-channel)")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("SEED", "0")),
                        help="Random seed")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for calibration/eval")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device("cuda:0")

    # Load model
    print(f"Loading model from {args.model_path}...", flush=True)
    t0 = time.time()
    model = get_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Evaluate FP16 baseline perplexity first
    print("\n=== FP16 Baseline Evaluation ===", flush=True)
    testenc = get_eval_data(tokenizer, seqlen=args.seqlen)
    fp16_ppl = evaluate_perplexity(model, testenc, dev)
    print(f"TRAIN_METRICS: fp16_perplexity={fp16_ppl:.4f}", flush=True)

    # Load calibration data
    print("\n=== Calibration ===", flush=True)
    t0 = time.time()
    calibration_data = get_calibration_data(
        tokenizer, nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed
    )
    print(f"Calibration data loaded in {time.time() - t0:.1f}s", flush=True)

    # Quantize
    print(f"\n=== INT{args.num_bits} Quantization ===", flush=True)
    t0 = time.time()
    quant_errors = quantize_model(model, calibration_data, dev, num_bits=args.num_bits, group_size=args.group_size)
    quant_time = time.time() - t0
    total_error = sum(quant_errors.values())
    print(f"Quantization completed in {quant_time:.1f}s", flush=True)
    print(f"Total quantization error (Frobenius): {total_error:.4f}", flush=True)
    print(f"TRAIN_METRICS: quant_time={quant_time:.1f} total_quant_error={total_error:.4f}", flush=True)

    # Evaluate quantized perplexity
    print(f"\n=== INT{args.num_bits} Evaluation ===", flush=True)
    quant_ppl = evaluate_perplexity(model, testenc, dev)
    degradation = quant_ppl - fp16_ppl

    print(f"\n=== Results ===", flush=True)
    print(f"FP16 perplexity:  {fp16_ppl:.4f}", flush=True)
    print(f"INT{args.num_bits} perplexity:  {quant_ppl:.4f}", flush=True)
    print(f"Degradation:      {degradation:.4f}", flush=True)
    print(f"TEST_METRICS: wikitext2_ppl={quant_ppl:.4f} fp16_ppl={fp16_ppl:.4f} degradation={degradation:.4f} quant_time={quant_time:.1f}", flush=True)


if __name__ == "__main__":
    main()
