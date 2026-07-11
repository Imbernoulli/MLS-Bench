# Logit-Only OOD Score

Design a post-hoc OOD score using only logits from a frozen CIFAR-10
ResNet-18 classifier.

Edit `ood-detection-lab/solution/logit_score.py`. `Scorer.fit(ctx)` receives only
ID-fit logits. `Scorer.score(logits)` must return one finite scalar per sample,
with higher values indicating in-distribution inputs. Features, raw pixels, the
model, and OOD data are not passed to the surface.

The verifier performs real frozen-model inference over complete, fixed ID-fit,
ID-evaluation, and multi-domain OOD inventories. The protocol evaluates 106,032
images in total; every configured evaluation contributes to the final score.
Evaluation dataset identities, labels, commands, and resource assignments are
not part of the instruction.

The classifier architecture and 100-epoch CIFAR-10 training recipe follow the
OpenOOD ResNet-18 baseline. The frozen checkpoint, exact dataset inventories,
and normalization are fixed. The verifier reports ID-vs-OOD AUROC; higher is
preferred. A non-zero score requires a complete run over all 106,032 images and
finite metrics for every configured evaluation.
