#!/bin/bash
# FlashAttention-style benchmark case: headdim=128, nheads=16, seqlen=8192, batch=2, causal
# Total tokens = 16384, hidden_dim = 2048
# Throughput is hardware/runtime dependent; local metrics are emitted by the benchmark.

cd /workspace

python flash-attention/custom_triton_bench.py \
    --batch 2 \
    --seqlen 8192 \
    --nheads 16 \
    --headdim 128 \
    --causal \
    --dtype float16 \
    --output-dir "${OUTPUT_DIR:-./output}" \
    --seed ${SEED:-42}
