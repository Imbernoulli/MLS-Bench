# Mamba Paper-E1 Re-audit (2026-07-11)

> **Terminal update, 2026-07-11.** This file is a pre-terminal handoff snapshot.
> Its `not shipped`, `still pending`, and `remained running` statements below
> are superseded by terminal task [96214](https://mangrove.msh.team/tasks/96214),
> container `4909980`: `finished`, `has_exception=false`, artifact uploaded,
> reward `0.393066`. The measured 1xH20 timing is setup `35.888s`, nop agent
> `0.163s`, harness `15,244.250s`, verifier `15,245.449s`, trial
> `15,282.190s`, and platform wall `15,284.081s`. The authoritative current
> status, full workload proof, and runtime table are in
> [`docs/100_REPO_STATUS_RESEARCH_20260708.md`](../100_REPO_STATUS_RESEARCH_20260708.md#accepted-repo-终态-runtime-总表).
> The historical investigation below is retained only to explain why the old
> 8K-step protocol was rejected.

## Verdict

Mamba is **not shipped**. Mangrove tasks 92609/92635/95310 are useful runtime
forensics only. They do not establish paper-scale performance or secure metric
provenance for an agent-authored solution.

The representative `mamba-selective-scan` task has now been converted to the
selective-copying scale stated in Gu and Dao, *Mamba*, Appendix E.1, but the full
400,000-step anchor and terminal Mangrove artifact are still pending.

## What the old runs actually did

| Task | Agent | Setup | Agent | Verifier | Container | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| 92609 | failed claude-code (429) | 93.2s | 288.9s | 173.2s | 558.0s | Untouched native LTI diagnostic |
| 95310 | nop | 97.0s | 0.17s | 287.8s | 387.4s | Untouched native LTI diagnostic |

Task 95310's Mamba container started 2,449.9 seconds after the 20-item batch was
created. The remembered 40-50 minutes was batch dispatch wait, not Mamba compute.
The whole batch achieved only three overlapping container intervals even though
`max_concurrency=20` was requested.

The 95310 reward `0.09391562778758898` was not hardcoded or a fallback. All three
declared tiny settings returned rc=0 and the unchanged native LTI scaffold was
scored. That is allowed under the policy that a failed/nop agent may evaluate the
native solution. It is not agent performance.

The deployed Mamba-specific proof was nevertheless insecure: it imported editable
agent Python with `importlib.exec_module`, and its parser accepted one final metric
line without binding sequence length, step count, seed, CUDA, or training
completion. A submission could print a forged metric during import and exit zero.
The generic matrix wrapper was fail-closed, but `strict_fail_closed=true` only
proved that its weak Mamba parser had accepted the logs.

## Scale correction

Appendix E.1 states:

- total sequence length 4096;
- vocabulary size 16 and 16 memorized tokens;
- two layers, model dimension 64;
- 400,000 steps at constant learning rate 1e-4;
- batch size 64.

The old task used total lengths 288/416/544, only 6,000 or 8,000 steps, learning
rate 5e-3, and AdamW with weight decay 0.01. Calling that protocol "native scale"
was incorrect. Conversely, calling the 68K-parameter D64/two-layer model itself
too small was also overbroad: the paper's synthetic models are intentionally this
size (the induction-head Mamba row is about 74K parameters). The synthetic task is
official; the decisive defects were sequence length, training budget, optimizer
recipe drift, and ablation confounds.

The new representative proves 25,600,000 training examples and 104,857,600,000
sequence tokens. The paper does not specify the optimizer details for selective
copying, so the implementation explicitly binds Adam, zero weight decay, and
gradient clipping at 1.0 rather than pretending those details came from the paper.

## Ten-question design audit

| Task | Decision | Required correction |
|---|---|---|
| selective scan | Keep as representative | Paper E.1 scale; per-channel constant B/C; fresh LTI/B-C/selective anchors |
| A stability | Keep | Run at paper scale; discard old anchors |
| B/C coupling | Keep | Constant B/C must preserve D x N channel capacity; fresh anchors |
| convolution activation | Keep, revalidate | Remove old four-layer deviation; compare identity/ReLU/SiLU at two layers |
| Delta softplus | Keep after code fix | Identity previously omitted Delta bias while ReLU/softplus included it; all choices now receive the same biased input |
| Delta initialization | Keep | Paper-scale S4D log-uniform comparison; fresh anchors |
| output gating | Keep, revalidate | Paper directly distinguishes architecture gating from selection; remove old four-layer deviation |
| normalization | Redesign | Drop causal-leaking BatchNorm; include RMSNorm, LayerNorm, and no norm at two layers |
| residual | Redesign | Drop the arbitrary half-residual candidate; retain standard add/no-residual and justify any replacement |
| A initialization | Rename and re-anchor | `A_log=0` means A=-1, not a zero state matrix; call it constant-rate versus S4D spectrum |

Only the representative is being promoted now. The other nine remain explicitly
blocked until their corrected question surfaces and fresh numerical anchors exist.
One representative execution can validate the shared runtime, but cannot justify
their old numerical leaderboards.

## Current strict implementation

- Agent code is never imported. A small AST is accepted only when it contains one
  literal `surface_config()` return with one enumerated field.
- Proofs bind protocol, task, label, chosen surface, total length, M/A, model
  dimensions, layers, steps, batch, learning rate, optimizer, weight decay,
  gradient clip, eval batches, seed, parameter count, and train/eval cardinalities.
- Exactly one ordered `POOL_LOADED`, `MAMBA_TRAIN_COMPLETE`,
  `MAMBA_EVAL_COMPLETE`, and task metric record is required.
- Missing/duplicate settings or seeds, timeout, nonzero rc, empty log, traceback,
  failure marker, parser error, non-finite value, incomplete cardinality, score
  error, or interrupted proof publication leaves public reward exactly `0`.
- An unchanged native solution remains valid and can receive its measured score.

Focused verification currently passes:

- Mamba and cross-contract tests: `87 passed`;
- Harbor adapter verifier/scheduler/render tests: `183 passed, 5 skipped`.

## Image and render

Immutable self-contained image:

`msai-cn-beijing.cr.volces.com/public/bohanlyu2022/mlsbench-harbor-mamba-paper-e1@sha256:d15909dfa459913838c8d8f2f91f1d86cc657f8b1dd8bda2fc4a79800ac74452`

Source-manifest SHA-256:

`32e0112f9741cc8b5f3533ad52f96928cbf466857babf859a8f359ddc386069a`

The final render is under `/tmp/mamba-paper-e1-render-final`. It has one H20,
8 CPUs, 128 GiB memory, no Internet, one `paper_e1` setting, serial verifier
execution, a 54-hour setting timeout, and the digest-pinned image above. Its
Dockerfile performs no package installation or data download.

Image validation jobs were submitted to `m3h20`, `k1h20`, and `b0h200` under:

`/mnt/moonfs/lvbohan-b0/mamba-paper-e1/image-validations/20260711T0423Z`

These two-step jobs are dependency/GPU validation only and must never be reported
as benchmark results. Full 400,000-step anchors must use separate evidence paths.

## Formal Mangrove run and measured runtime probes

The clean Harbor branch is `mls-mamba-paper-e1-20260711-0430` at commit
`a5158856ef74e17c6b8495fd17c0b0da4e42d4b5`. Only
`mamba-selective-scan/**` differs from `origin/master`; `dataset.toml` was not
modified. Dataset version `18725` contains exactly one item.

The formal native-solution run is Mangrove task `96214`, container row `4909980`,
remote container `1343286`, run `5678205`. It uses `nop`, dataset version `18725`,
`max_concurrency=1`, one H20 on `virt-m3h20`, and container-level
`disable_internet_access=true`. At the last 2026-07-11 poll it remained running
without an exception or artifact; the public score remained the prewritten zero.
It is not shipped until the terminal artifact audit passes.

Two short jobs measured the real paper-shaped step cost without claiming an eval:

| GPU | Worker | Workload | Harness wall | Raw log |
|---|---|---|---:|---|
| H200 143771 MiB | `dev-xrzfm-1895114-worker-0` | L4096, batch64, 100 train steps, 1 eval batch | 3.5s | `/tmp/mamba_h200_l4096_b64_100step_perf_probe_20260711.log` |
| H20 97871 MiB | `dev-mmwkh-1942713-worker-0` | L4096, batch64, 100 train steps, 1 eval batch | 4.1s | `/tmp/mamba_h20_l4096_b64_100step_perf_probe_20260711.log` |

The logs are also persistent under
`/mnt/moonfs/lvbohan-ksyun/mamba-paper-e1/perf-probes/`. Their SHA-256 values are
`8cee24b8d90a3008f22ea35323ecf318a16866123e68f8f626cf76536145f9d9` (H200)
and `e1cdc964a9ac27c34c43eebab7e7e405724ce1dcb8032a66b39f66d199c34793`
(H20). Both verified every file in the pinned image source manifest, ran on CUDA
without OOM, and produced identical loss/accuracy records. Linear projection is
about 3.9 hours on H200 and 4.6 hours on H20, before small fixed overhead. These
are runtime probes explicitly marked `NOT_AN_EVAL`, not numerical anchors.

The three full independent anchor cells are queued under:

`/mnt/moonfs/lvbohan-b0/mamba-paper-e1/anchor-matrix-20260711T0428Z`

Their workers are `dev-2hdvw-1635177-worker-0` (LTI),
`dev-b9pkn-1635231-worker-0` (B/C-only), and
`dev-whpmg-1635265-worker-0` (selective). A `status=queued` plus `rc=125` is
dispatch state, not a failed or completed benchmark.
