#!/bin/bash
# FlashAttention-style benchmark case: headdim=64, nheads=32, seqlen=4096, batch=4, causal
# Total tokens = 16384, hidden_dim = 2048
# Throughput is hardware/runtime dependent; local metrics are emitted by the benchmark.

cd /workspace

python flash-attention/custom_triton_bench.py \
    --batch 4 \
    --seqlen 4096 \
    --nheads 32 \
    --headdim 64 \
    --causal \
    --dtype float16 \
    --output-dir "${OUTPUT_DIR:-./output}" \
    --seed ${SEED:-42}
