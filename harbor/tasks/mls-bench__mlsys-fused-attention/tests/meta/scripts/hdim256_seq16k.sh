#!/bin/bash
# FlashAttention-style benchmark case: headdim=256, nheads=8, seqlen=16384, batch=1, causal
# Total tokens = 16384, hidden_dim = 2048
# Throughput is hardware/runtime dependent; local metrics are emitted by the benchmark.

cd /workspace

python flash-attention/custom_triton_bench.py \
    --batch 1 \
    --seqlen 16384 \
    --nheads 8 \
    --headdim 256 \
    --causal \
    --dtype float16 \
    --output-dir "${OUTPUT_DIR:-./output}" \
    --seed ${SEED:-42}
