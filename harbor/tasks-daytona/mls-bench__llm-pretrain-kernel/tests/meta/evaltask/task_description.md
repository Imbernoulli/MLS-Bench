# LLM Pretraining: Custom GPU Kernel Optimization

## Research Question

Write a custom GPU kernel (Triton or CUDA via PyTorch) to implement a
fused MLP operation for GPT-2 pretraining. Your kernel should fuse
multiple operations to reduce memory bandwidth and improve throughput
while maintaining or improving model quality.

## What You Can Modify

The `fused_mlp_forward` function in `custom_pretrain.py`:

- The MLP activation function (default: GELU via separate PyTorch ops)
- Kernel fusion strategy (fuse linear + activation, save intermediate
  values)
- Memory optimization (avoid materializing intermediate tensors)
- Custom autograd Functions for efficient backward pass

The function signature `fused_mlp_forward(x, w_fc, w_proj)` must be
preserved.

- `x`: input tensor `(B*T, n_embd)`
- `w_fc`: first linear weight `(4*n_embd, n_embd)`
- `w_proj`: second linear weight `(n_embd, 4*n_embd)`
- Returns: output tensor `(B*T, n_embd)`

The MLP class calls this function and handles dropout separately.

## Evaluation

- Metrics: validation loss (cross-entropy, lower is better) and MLP
  kernel throughput — kernel optimizations that also change the
  activation function may improve loss
- Throughput is `mlp_speedup`, measured after training by a fixed
  benchmark: the forward+backward time of the unfused PyTorch GELU MLP
  (the unmodified `fused_mlp_forward`) divided by that of your
  `fused_mlp_forward`, both under `torch.compile` and bf16 autocast at the
  training shape on the same GPU (higher is better; > 1 means faster than
  the unfused reference)
- Model: GPT-2 Medium (24L/16H/1024D, ~355M params)
- Dataset: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N
  Chinchilla-optimal)
- Training: 13535 iterations, BSZ=64, GA=8, 2-GPU DDP
