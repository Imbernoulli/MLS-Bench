"""Score spec for security-adversarial-attack-white-box-linf."""
from mlsbench.scoring.dsl import *

# ATTACKER task: ASR = higher better (attacker wants high success rate), bounded [0,1]
# Config labels: ResNet20-C10, VGG11BN-C10, ResNet20-C100, VGG11BN-C100, MobileNetV2-C100
# asr_MobileNetV2_C10 appears in leaderboard from prior runs but label no longer in config; included for compatibility
#
# NOTE (eps=2/255): ref values below were calibrated at eps=4/255 where AutoAttack
# saturated to ~1.0 on most envs. Eps was reduced to 2/255 on 2026-04-23 to de-saturate
# (FGSM expected ~60-80%, PGD-40 ~95%, AutoAttack ~98%). Ref values should be
# recomputed from re-run baselines once new AutoAttack numbers land in leaderboard.csv.

term("asr_ResNet20_C10",
    col("asr_ResNet20_C10").higher().id()
    .bounded_power(bound=1.0))

term("asr_VGG11BN_C10",
    col("asr_VGG11BN_C10").higher().id()
    .bounded_power(bound=1.0))

term("asr_ResNet20_C100",
    col("asr_ResNet20_C100").higher().id()
    .bounded_power(bound=1.0))

term("asr_VGG11BN_C100",
    col("asr_VGG11BN_C100").higher().id()
    .bounded_power(bound=1.0))

term("asr_MobileNetV2_C100",
    col("asr_MobileNetV2_C100").higher().id()
    .bounded_power(bound=1.0))

setting("ResNet20-C10", weighted_mean(("asr_ResNet20_C10", 1.0)))
setting("VGG11BN-C10", weighted_mean(("asr_VGG11BN_C10", 1.0)))
setting("ResNet20-C100", weighted_mean(("asr_ResNet20_C100", 1.0)))
setting("VGG11BN-C100", weighted_mean(("asr_VGG11BN_C100", 1.0)))
setting("MobileNetV2-C100", weighted_mean(("asr_MobileNetV2_C100", 1.0)))

task(gmean("ResNet20-C10", "VGG11BN-C10", "ResNet20-C100", "VGG11BN-C100", "MobileNetV2-C100"))
