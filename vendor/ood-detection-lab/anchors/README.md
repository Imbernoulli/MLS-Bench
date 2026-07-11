# OOD full-image protocol and anchor provenance

This file supersedes the earlier 5,000-image SmallCNN/precomputed-logit anchor
notes. Those measurements are not used by `ood-logit-score` full-v1.

## Accepted protocol

- Protocol: `openood_cifar10_resnet18_full_v1`
- Frozen classifier: OpenOOD `ResNet18_32x32`
- Classifier training inventory: complete CIFAR-10 train, 50,000 images
- ID evaluation: complete CIFAR-10 test, 10,000 images
- OOD evaluation:
  - complete SVHN test, 26,032 images
  - complete CIFAR-100 test, 10,000 images
  - complete Tiny-ImageNet validation, 10,000 images, bilinear 32x32 resize
- ID-fit surface input: logits from all 50,000 CIFAR-10 train images
- Actual frozen-model forward inventory: 106,032 images in 832 batches
- Batch size: 128
- Device contract: exactly one visible CUDA GPU; final platform is H20
- Aggregation: all three OOD settings contribute through geometric mean

The classifier reproduces OpenOOD's public CIFAR-10 baseline recipe: 100
epochs, SGD with learning rate 0.1, momentum 0.9, weight decay `5e-4`,
Nesterov, per-step cosine annealing to `1e-6`, random horizontal flip, and
random crop with four-pixel padding. Training is an offline asset-generation
step, never part of final verification.

Pinned inputs:

```text
data SHA256:
796799c9a1c073784b02a3f42a00d0fc8b902387b1f3345b5b3d1f2631e0722d

checkpoint SHA256:
8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc
```

The full data builder was run twice and produced the same archive digest.
Source-release digests and construction details are pinned in
`prepare_full_eval.py`.

## Checkpoint provenance

Mangrove staging task `96476`, container `4936717`, trained on one H20 without
runtime download or installation. The artifact contains exactly 100 sequential
epoch records and 39,100 optimizer steps. The final learning rate is `1e-6`.

```text
environment setup: 41.15 s
100-epoch verifier/training stage: 609.23 s
staging 5k ID accuracy: 0.951600
full 10k ID accuracy in final protocol: 0.951400
checkpoint bytes: 44,777,866
checkpoint SHA256: 8859e0ff...b7dc
```

The staging task is provenance only and is not a final benchmark result.

## Immutable runtime image

Final verification uses one prebuilt repo image:

```text
msai-cn-beijing.cr.volces.com/public/bohanlyu2022/
mlsbench-harbor-ood-detection-lab@
sha256:c96492da2073103f2d59b3aad629b7c9560ed07d9ce6b5d315e0d46ec046fe8e
```

The image contains the full data archive and frozen checkpoint. OCI labels bind
the protocol, both digests, the 106,032-image count, and
`training-in-verifier=false`. Package setup performs only an import/version
check. Verification performs no `pip`, `conda`, `apt`, download, extraction,
compilation, or training.

## Full-image anchors

The final accepted anchors use dataset version `18763`, rendered source commit
`ba0f93fec2a6e041bd87902a9a339c5370f9b161`, task checksum
`5113edabf947c6124ac5b278295adbd4eab82ad54c4ea1696db4b7b5d60010a9`,
and the pinned image digest above. Native MSP is Mangrove task `96612`,
container `4950395`, run `5715452`. Pseudo-cosine is task `96611`, container
`4950394`, run `5715451`. Both used one H20, finished without an exception,
uploaded an indexed artifact, returned command `rc=0`, passed the strict
fail-closed result contract, and ended in the authenticated completion record
binding both asset digests and the 106,032-image / 832-batch inventory.

Immutable artifact evidence:

| evidence | native MSP (`96612`) | pseudo-cosine (`96611`) |
|---|---|---|
| artifact id | `4370070` | `4370032` |
| artifact ZIP SHA256 | `be9e85cbd686012b45010909613e78fbfc78368bf0afdc2b3d1a44b7dc07c0d4` | `63e7771ec5d74015efa4f90e441800ca387d941c0e8cdd4624583ef99cb573ff` |
| verifier log SHA256 | `423b6a26841540806f385b6e72fc9872b8b68ebc6504d77215643d6c9ea43779` | `76fdbede23845652bf551faaa948c3f6d8497996d5615ae999a5180f737e7fe2` |
| eval summary SHA256 | `a952277179353b047cce9339c9aeb0c09e8fe0d17df3097d17612898f3b70cd5` | `e09b870f9f8baafe386d1a8d9124185ae95937e3e251d3ba21264a8527ab085f` |
| metrics SHA256 | `d3fa9a0e3c021277ed9e4bde860b5359fd2ce0e4fcde929f7cb751bcbfad8df0` | `040e3ddcd616acba46b2f428181e3958efe05591117e7c7f9daed906a3c05152` |
| verification result SHA256 | `e6aa1cd8239e2728e481ace7121a23f8eaf78e72c9980ab4d62f998efbbc03b4` | `a79d78193d243c794228f5945fd2779e751b81e60503ff563cf58c919ffb03bd` |
| result JSON SHA256 | `0f2948f411f9acaf80ec2118401d0f9c9744a268ce0a0f5adb2789c9cca805db` | `7ab467d25504c4895908509f7d1faf6b09f869240909cd829264dc9e82bb0fc9` |
| measured script wall | `14.007822 s` | `11.006253 s` |
| reward | `0.1` | `0.5` |

The subsequent task-identity field added to protocol/metric/completion records
is a proof-only hardening change. It does not change the frozen model, inputs,
candidate surfaces, inference inventory, metrics, or score calibration, so the
authenticated runtime and anchor values remain reusable without rerunning
task `96612`.

| baseline | SVHN AUROC / FPR95 | CIFAR-100 AUROC / FPR95 | Tiny-ImageNet AUROC / FPR95 |
|---|---:|---:|---:|
| MSP | 0.92846834 / 0.44875538 | 0.87088166 / 0.57320000 | 0.87007299 / 0.55510000 |
| Pseudo-cosine | 0.93220756 / 0.43661647 | 0.89527483 / 0.51750000 | 0.89377779 / 0.49770000 |

Pseudo-cosine improves the scored AUROC over MSP on all three settings.

Pseudo-cosine uses only permitted ID-fit logits. It assigns each training logit
to its predicted class, averages the logits into ten class prototypes,
L2-normalizes them, and scores a test logit by its maximum cosine similarity to
the prototypes. It uses no labels, pixels, model access, or OOD data.

The measured MSP values map to 0.1 and pseudo-cosine values map to 0.5 per
setting. Each sigmoid scale is `(pseudo_cosine - MSP) / ln(9)`.

## Measured runtime

The final native H20 run reported `14.007822` seconds of test-script wall time
and `15.845338` seconds for the enclosing verifier stage. The pseudo-cosine run
reported `11.006253` and `12.164473` seconds respectively. End-to-end platform
wall includes roughly 98-112 seconds of environment setup for the cached image;
that setup time is reported separately from verifier inference.

Representative model-forward timings from the artifacts:

| split | images | batches | H20 forward seconds |
|---|---:|---:|---:|
| CIFAR-10 train | 50,000 | 391 | 2.23-2.36 |
| CIFAR-10 test | 10,000 | 79 | 0.43 |
| SVHN test | 26,032 | 204 | 1.07-1.08 |
| CIFAR-100 test | 10,000 | 79 | 0.41 |
| Tiny-ImageNet val | 10,000 | 79 | 0.41 |

This runtime is plausible because final verification is batched inference
through a small 32x32 ResNet-18, not classifier training or language-model
generation. The expensive 100-epoch training is separately evidenced above at
about 10 minutes. The verifier still hashes and loads full image data and
executes all 832 model-forward batches at runtime; it does not read prepared
logits.

## Candidate audit

Mangrove task `96554`, container `4942521`, independently executed the complete
forward inventory and exported a temporary 4.2MB logit dump for reproducible
anchor selection:

```text
dump SHA256:
b91adf34b075550ba53658c689de6ed11f6cc696f1b5257f324e419ac64ec72a
```

`evaluate_logit_candidates.py` compares MSP/temperature variants, Energy,
max-logit, centered-logit norm, margin, entropy, pseudo-class Mahalanobis/RMD,
cosine prototypes, and ID-only normalized combinations. The dump is an anchor
research artifact only; it is not shipped in the final image or accepted by the
final verifier.

## Failure semantics

The parser accepts exactly three setting records and exactly one final
completion record. It rejects missing/duplicate records, wrong setting names or
counts, wrong digests, inconsistent classifier accuracy, non-finite values,
zero runtime, traceback/failure markers, and trailing output. A nonzero command
return code also discards otherwise valid-looking metrics. Empty or partial
metrics cannot be submitted, so failed or incomplete verification scores
exactly zero. Agent failure may still leave the native MSP solution to evaluate;
verifier failure cannot receive fallback credit.

## Sibling status

All ten non-logit `ood-*` research questions now use the same authenticated
full archive and frozen ResNet-18 as the representative logit task. Each runs
all three OOD domains serially in one verifier command. The distance and
ensemble paths use GPU-chunked full-50k-bank k-NN; the gradient task exposes a
ten-dimensional per-class gradient-representation scorer instead of a sign
toggle; and the ODIN path performs its explicit two forwards over every scored
split. Their obsolete 5k/SmallCNN anchor rows have been removed. Because AUROC
has a task-independent random floor of 0.5 and a theoretical upper bound of
1.0, these ten siblings use a baseline-free direct score over all three domains
instead of inventing replacement anchors. The accepted logit task retains its
authenticated measured calibration so its historical score scale remains
unchanged.
