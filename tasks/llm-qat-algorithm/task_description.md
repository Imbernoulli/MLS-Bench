# LLM Quantization-Aware Training (QAT) Algorithm

## Research Question

Design a quantization-aware training (QAT) algorithm that minimizes the
perplexity gap between a full-precision Pythia-1.4B and the same model
quantized to very low bit-widths (INT4 / INT3 / INT2) at inference time.
The algorithm must be a *training-side* contribution: how the fake-quant
forward, the gradient flow, the learnable parameters, and the optimizer
schedule are designed. It must work uniformly across 4-, 3-, and 2-bit
settings, not just one.

## Background

Post-training quantization (PTQ) collapses at very low bit-widths because
every weight is rounded to one of `2^B` levels with no chance to repair the
resulting error. Quantization-Aware Training (QAT) attacks this by
inserting *fake quantization* into the forward pass during a short
fine-tune. The key knobs are:

- Gradient estimator: round-then-clamp is non-differentiable. The
  Straight-Through Estimator (STE) (Bengio et al., 2013) simply pretends
  the operation is identity in backward. Learning the step size jointly
  with the weights — Learned Step Size Quantization (LSQ; Esser et al.,
  ICLR 2020; arXiv:1902.08153) — gives a measurably tighter quantization
  grid and tends to dominate STE at INT2.
- Stability: low-bit QAT diverges easily; warming up the quantization
  noise and EMA-smoothing the scales (StableQAT-style) buys back several
  PPL points at INT2.

Group quantization (per-row, per-group of `group_size=128` columns,
symmetric, signed) is the standard low-bit format and is fixed for this
task. Linear layers in every transformer block are quantized; embeddings,
LayerNorm, and the LM head stay full precision.

A control baseline `finetune_then_ptq` runs a full-precision finetune on
WikiText-2 train with the same schedule as the QAT methods (`lr=2e-5`,
500 steps, batch 2, grad-accum 4) and then applies the same RTN
quantize-dequantize as `no_qat`. This isolates the finetune signal from
the QAT signal: a useful QAT method must beat `finetune_then_ptq`,
otherwise its apparent gains over `no_qat` are just the in-domain
finetune talking.

## What You Can Modify

The single file `llm-qat-runtime/custom_qat.py` is created at task setup;
you may only edit the `# EDITABLE REGION START / END` block. It contains:

- `CONFIG_OVERRIDES` dict: per-method training hyperparameters
  (`learning_rate`, `num_steps`, `batch_size`,
  `gradient_accumulation_steps`, `max_grad_norm`, `warmup_steps`,
  `weight_decay`).
- `fake_quantize_weight(weight, num_bits, group_size)`: differentiable
  fake-quant for the QAT forward pass. Must allow gradient flow back to
  the original weight.
- `fake_quantize_activation(x, num_bits)`: optional (default identity for
  weight-only QAT).
- `quantize_dequantize_weight(weight, num_bits, group_size)`: REAL
  (no-grad) per-group symmetric QDQ used after training to materialize the
  integer model for evaluation.
- `class QATWrapper(nn.Module)`: wraps an `nn.Linear`; applies fake quant
  in `forward`; may hold extra learnable parameters (per-group scales for
  LSQ, EMA buffers for StableQAT, etc.). May expose an
  `aux_loss(step, total_steps)` method that the training loop adds to the
  cross-entropy loss.
- `prepare_qat_model(model, num_bits, group_size)`: replace every
  `nn.Linear` (and HF GPT-2 `Conv1D`) in the model with `QATWrapper`,
  initializing any extra learnable parameters. The function must restore
  the LM head (`embed_out` for Pythia / GPTNeoX, `lm_head` for GPT-style
  models) to a plain Linear so the output projection stays in full
  precision.

The fixed (non-editable) region implements: model load (Pythia-1.4B in
FP32 with gradient checkpointing), WikiText-2 train data sampling
(block-1024 random crops), the QAT training loop (`AdamW`, cosine LR with
warmup, gradient accumulation, grad-norm clipping), real-quantization
roundtrip after training, and WikiText-2 test perplexity evaluation.

## Architecture

- Backbone: HuggingFace `EleutherAI/pythia-1.4b` (1.4B parameters,
  GPTNeoX architecture, 24 layers x 16 heads x 2048 hidden, native
  context length 2048). Linear layers are wrapped via the recursive
  traversal in `prepare_qat_model`.
- Optimizer: AdamW, cosine schedule with linear warmup. Default 500 steps
  x batch 2 x grad-accum 4 (~4000 sequences seen, seqlen 1024) — the
  agent may shorten/lengthen via `CONFIG_OVERRIDES`.
- Calibration / training data: WikiText-2 raw v1 train split. Random
  1024-token crops.
- Evaluation: WikiText-2 raw v1 test split, sliding non-overlapping
  blocks of 1024 tokens, exponentiated mean cross-entropy loss.

## Interface

```python
CONFIG_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_steps": 500,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "weight_decay": 0.0,
}

def fake_quantize_weight(weight, num_bits, group_size): ...   # differentiable
def fake_quantize_activation(x, num_bits): ...                # optional, default id
def quantize_dequantize_weight(weight, num_bits, group_size): # no-grad QDQ

class QATWrapper(nn.Module):
    def __init__(self, linear, num_bits, group_size): ...
    @property
    def weight(self) -> torch.Tensor: ...
    @property
    def bias(self): ...
    def forward(self, x): ...

def prepare_qat_model(model, num_bits, group_size): ...
```

Constraints:

- The forward path of every wrapped `nn.Linear` must use
  `fake_quantize_weight` (or an equivalent inside `QATWrapper.forward`)
  so the QAT signal actually trains the integer grid.
- After training, `quantize_dequantize_weight` is applied to every
  `linear.weight` of every `QATWrapper`, then perplexity is measured.
  Your method must produce weights that, after this real QDQ roundtrip,
  still give a low perplexity.
- Keep the LM head at full precision (the template already excludes
  `embed_out` / `lm_head`).
- Available imports in the editable region: `torch`, `torch.nn` (as
  `nn`), `torch.nn.functional` (as `F`), `numpy` (as `np`), `math`,
  `os`, `time`, plus `transformers.pytorch_utils.Conv1D`.
- All seeds and training hyperparameters must be deterministic given
  `--seed`.

## Evaluation

The algorithm is evaluated across three bit-widths:

- `qat-1b-int4`: INT4, group size 128 — easy.
- `qat-1b-int3`: INT3, group size 128 — medium (8 levels).
- `qat-1b-int2`: INT2, group size 128 — extreme (4 levels).

Primary metric: `wikitext2_ppl` — WikiText-2 perplexity after the real
QDQ roundtrip, lower is better.
Secondary metric: `degradation` — `wikitext2_ppl - fp16_ppl`, where
`fp16_ppl` is the FP baseline measured before any quantization.

Note on absolute PPL vs. literature (OmniQuant / EfficientQAT tables):
QAT here finetunes on WikiText-2 train and evaluates on WikiText-2 test
(disjoint articles, but same domain). With 500 steps x bsz 2 x ga 4 =
4000 sequences x 1024 tokens, the FP16 finetune alone can drop test PPL
below the FP16 baseline (cf. `finetune_then_ptq` INT4 < `no_qat` FP16),
because the QAT train domain matches the eval domain. Published OmniQuant
/ EfficientQAT tables on LLaMA-{7B,13B} use C4 calibration and a
held-out WikiText eval, so their absolute W2g128 / W3g128 / W4g128
numbers are not directly comparable to ours. The intended internal
comparison is QAT-method vs `finetune_then_ptq`: a method that beats
`finetune_then_ptq` is showing real QAT signal, beyond the in-domain
finetune effect.

## Reference baselines

### no_qat
Round-to-nearest (RTN) post-training quantization with no fine-tuning —
the pure PTQ lower bound.

### ste
Straight-Through Estimator (Bengio et al., 2013): fake-quantize in the
forward pass, pass the gradient through unchanged (identity) in the
backward pass. The canonical minimal QAT gradient estimator.

### lsq
Learned Step-Size Quantization (Esser et al., ICLR 2020, arXiv:1902.08153):
learnable per-group quantization scales trained jointly with the weights,
giving a tighter quantization grid than STE.

### finetune_then_ptq
Full-precision fine-tune on WikiText-2 (same schedule as QAT methods)
followed by RTN quantization. Isolates the in-domain fine-tune signal
from the QAT signal; a valid QAT method must outperform this baseline.
