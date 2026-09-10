# LLM Post-Training Quantization (PTQ) Algorithm

## Research Question

Design a post-training quantization algorithm that minimizes accuracy
degradation when quantizing a pretrained Mistral-7B-v0.1 model
(7.24B parameters) to low-bit integer precision, without any retraining
or fine-tuning.

## Background

Post-training quantization (PTQ) compresses neural-network weights from
floating-point to low-bit integer representations after training is
complete. Unlike quantization-aware training (QAT), which modifies the
training procedure, PTQ works on already-trained models and requires no
gradient updates to the original weights, which is attractive for LLMs
where retraining is prohibitively expensive.

The challenge is severe at low bit-widths: INT4 has only 16 discrete
levels (vs 256 for INT8), and INT3 has only 8 levels, so naive rounding
causes significant accuracy loss. This is amplified at 7B+ scale where
weight distributions are complex and quantization errors accumulate
across many transformer layers. Reference families:

- RTN (Round-To-Nearest): round each weight to its nearest quantized
  value. Fast but high degradation.
- SmoothQuant (Xiao et al., ICML 2023; arXiv:2211.10438): migrate
  quantization difficulty from activations to weights via a per-channel
  equivalent transformation, making weight distributions easier to
  quantize.
- GPTQ (Frantar et al., ICLR 2023; arXiv:2210.17323): use calibration data
  to compute an approximate Hessian, then quantize weights column-by-column
  while compensating remaining error using second-order information.
- AWQ (Lin et al., MLSys 2024 Best Paper; arXiv:2306.00978): identify
  salient weight channels via activation magnitudes and protect them with
  per-channel scaling, without requiring Hessian computation.

Quantization here uses symmetric group quantization: weights are
partitioned into groups of consecutive columns (group size 64 or 128),
and one scale factor is computed per group per output row.

## What You Can Modify

The `LayerQuantizer` class and helper functions in `custom_ptq.py`:

- `quantize_tensor()` / `dequantize_tensor()`: basic quantization
  primitives
- `find_scale_zero()`: scale/zero-point computation (per-channel or
  per-group)
- `LayerQuantizer.__init__()`: set hyperparameters; receives `num_bits`
  and `group_size` from the evaluation script
- `LayerQuantizer.add_batch(inp)`: collect statistics from calibration
  data (128 sequences)
- `LayerQuantizer.quantize()`: apply quantization to the layer's weight
  matrix

You can implement any approach: error compensation, weight transformation
(scaling, rotation, smoothing), mixed strategies, outlier handling, or
adaptive grouping schemes that vary by group size or bit-width.

## Architecture

The task loads real Mistral-7B-v0.1 weights (HuggingFace) and quantizes
them. No training is done — the task is purely about the quantization
algorithm quality.

Mistral-7B-v0.1 specs: 32 layers, 32 attention heads, 8 KV heads (GQA),
4096 hidden, 14336 intermediate, ~7.24B parameters.

The script (`custom_ptq.py`):

1. Loads Mistral-7B-v0.1 from `/data/mistral-7b-v01` (pre-downloaded
   HuggingFace snapshot)
2. Evaluates the FP16 (unquantized) model as baseline
3. Runs your `LayerQuantizer.add_batch()` on calibration data layer by
   layer
4. Quantizes each linear layer using your `LayerQuantizer.quantize()`
5. Evaluates the quantized model and reports perplexity degradation

## Interface

```python
class LayerQuantizer:
    def __init__(self, layer, num_bits=4, group_size=-1):
        # layer: nn.Linear to quantize
        # num_bits: target bit width (4 or 3, set by evaluation)
        # group_size: columns per group (-1 = per-channel, 128 or 64)
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        # ... initialize calibration buffers

    def add_batch(self, inp):
        # inp: layer input tensor, shape (batch*seq_len, in_features)
        pass

    def quantize(self):
        # Returns: quantized-dequantized weight tensor
        # Must respect self.num_bits and self.group_size
        return W_dq

    def free(self):
        # Release calibration buffers
        pass
```

Constraints:

- You must NOT retrain or fine-tune the model (no gradient updates to
  original weights)
- All linear layers in each transformer block are quantized (`q_proj`,
  `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- Embeddings, LayerNorm, and the LM head are NOT quantized
- The returned weight must have the same shape and dtype as the original
- `copy`, `math`, `torch`, `torch.nn`, `F`, `np`, `os`, `time` are
  available
- Your algorithm must work for both INT4 and INT3, and for different
  group sizes

## Evaluation

The algorithm is evaluated across multiple quantization settings to test
generalizability:

- `ptq-7b-int4`: INT4 (4-bit) quantization with group size 128 — standard
  PTQ setting
- `ptq-7b-int3`: INT3 (3-bit) quantization with group size 128 — harder
  setting with only 8 levels
- `ptq-7b-int4-g64`: INT4 with group size 64 — finer granularity setting

Primary metric: `wikitext2_ppl` — WikiText-2 perplexity after
quantization (lower is better).
Secondary metric: `degradation` — perplexity increase over FP16 baseline
(lower is better).
Calibration: 128 sequences from WikiText-2 training set, 2048 tokens each.
