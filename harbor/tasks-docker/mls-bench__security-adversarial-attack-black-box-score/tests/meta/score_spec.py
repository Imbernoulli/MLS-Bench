"""Score spec for security-adversarial-attack-black-box-score."""
from mlsbench.scoring.dsl import *

# ATTACKER task: ASR = higher better (attacker wants high success rate), bounded [0,1]
# avg_queries: lower better (fewer queries = more efficient attack)
# ASR is the primary metric and avg_queries only a tie-break (task_description.md),
# so each setting weights ASR 0.9 and avg_queries 0.1. With equal weights a
# zero-query attack that returns the images unchanged (ASR 0) out-scored SPSA.
# Settings match config labels: ResNet20-C10, VGG11BN-C10, MobileNetV2-C10, ResNet20-C100, MobileNetV2-C100

term("asr_ResNet20_C10",
    col("asr_ResNet20_C10").higher().id()
    .bounded_power(bound=1.0))

term("avg_queries_ResNet20_C10",
    col("avg_queries_ResNet20_C10").lower().id()
    .sigmoid())

term("asr_VGG11BN_C10",
    col("asr_VGG11BN_C10").higher().id()
    .bounded_power(bound=1.0))

term("avg_queries_VGG11BN_C10",
    col("avg_queries_VGG11BN_C10").lower().id()
    .sigmoid())

term("asr_MobileNetV2_C10",
    col("asr_MobileNetV2_C10").higher().id()
    .bounded_power(bound=1.0))

term("avg_queries_MobileNetV2_C10",
    col("avg_queries_MobileNetV2_C10").lower().id()
    .sigmoid())

term("asr_ResNet20_C100",
    col("asr_ResNet20_C100").higher().id()
    .bounded_power(bound=1.0))

term("avg_queries_ResNet20_C100",
    col("avg_queries_ResNet20_C100").lower().id()
    .sigmoid())

term("asr_MobileNetV2_C100",
    col("asr_MobileNetV2_C100").higher().id()
    .bounded_power(bound=1.0))

term("avg_queries_MobileNetV2_C100",
    col("avg_queries_MobileNetV2_C100").lower().id()
    .sigmoid())

setting("ResNet20-C10", weighted_mean(("asr_ResNet20_C10", 0.9), ("avg_queries_ResNet20_C10", 0.1)))
setting("VGG11BN-C10", weighted_mean(("asr_VGG11BN_C10", 0.9), ("avg_queries_VGG11BN_C10", 0.1)))
setting("MobileNetV2-C10", weighted_mean(("asr_MobileNetV2_C10", 0.9), ("avg_queries_MobileNetV2_C10", 0.1)))
setting("ResNet20-C100", weighted_mean(("asr_ResNet20_C100", 0.9), ("avg_queries_ResNet20_C100", 0.1)))
setting("MobileNetV2-C100", weighted_mean(("asr_MobileNetV2_C100", 0.9), ("avg_queries_MobileNetV2_C100", 0.1)))

task(gmean("ResNet20-C10", "VGG11BN-C10", "MobileNetV2-C10", "ResNet20-C100", "MobileNetV2-C100"))
