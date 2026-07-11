# Person Re-Identification: Learning-Rate Schedule

## Objective

Investigate the learning-rate schedule inside the repository's fixed training and evaluation pipeline. Modify only the declared editable file. No candidate ordering or expected implementation is supplied; select a schedule from the public contract and the feedback produced by valid runs.

## Editable Surface

- File: `torchreid-reid/solution/schedule.py`
- Public symbol: `build_lr_schedule`

Preserve the callable signature and return a deterministic callable mapping each integer update index to a finite positive learning rate. The harness owns the Adam optimizer and every other optimization hyperparameter, so optimizer choice is not part of this task.

The selected surface is active. A load error, exception, malformed return, missing output, or NaN/Inf invalidates the run; the harness does not replace a failed active implementation with another implementation.

## Evaluation

The fixed harness fine-tunes an ImageNet-pretrained ResNet-50 on all 12,936
official Market-1501 training images for 60 epochs with batch 64, a P x K
sampler, fixed image geometry, and seed 42. The complete 3,368-query inventory
is partitioned exactly once into three difficulty groups; every group is
evaluated against the complete 19,732-image gallery, and all groups are scored.
The harness reports mAP and CMC retrieval metrics; higher values are better.

Protocol validity requires authenticated data and checkpoint inventories, exactly
60 epochs, 11,003 updates and 704,192 sampled training images, every query and
gallery item, all required finite metrics, and final
training and evaluation completion records. An interrupted or malformed run is
invalid. Historical 40-identity / ResNet-18 / 200-step anchors do not apply to
this protocol.

Every parser-valid run is scored directly from official mAP across all three
groups. No empirical anchor or historical leaderboard value participates in
the score. A failed, incomplete, malformed, or non-finite verification scores
exactly zero.

Do not modify the harness, scorer, data, scripts, or unrelated solution files.
