# Handoff: `mamba` → Mangrove verification (SCALE BLOCKED — not in first ship set)

> **Terminal supersession, 2026-07-11.** The title and scale-blocked conclusion
> describe only the historical 8K-step task. They are superseded by the new
> Appendix-E.1 representative in terminal task
> [96214](https://mangrove.msh.team/tasks/96214), container `4909980`, using
> L4096 and 400,000 steps on 1xH20. Its verifier measured `15,245.449s` and its
> platform wall measured `15,284.081s`; it is accepted under the one
> representative plus static sibling-audit policy. See the authoritative
> [`docs/100_REPO_STATUS_RESEARCH_20260708.md`](../100_REPO_STATUS_RESEARCH_20260708.md#accepted-repo-终态-runtime-总表).
> Everything below remains historical evidence for the rejected old image/DV,
> not the current ship status.

> **Authoritative correction, 2026-07-10.** The earlier `COMPLETE`,
> `native scale`, and `non-toy` conclusions below are not valid under the
> community-scale acceptance rule. The deployed workload is a synthetic
> selective-copy diagnostic using a 68,032-parameter, two-layer model
> (`d_model=64`, `d_state=16`) for 8,000 steps at each of three sequence
> lengths. Mangrove task 95310 reproduced the same workload: the verifier
> took 287.816 seconds total and all three commands returned successfully, so
> it is useful evidence that the execution and score paths work. It is not
> evidence of a normal repo-scale Mamba research workload. Historical task
> 92609's longer wall time was mostly agent-side HTTP 429 retries; its verifier
> was only 173.244 seconds. Do not ship this repo in the first 20 and do not
> reuse DV 18046 as scale-acceptance evidence. It needs a new task built on a
> community-scale sequence dataset and material Mamba model, followed by fresh
> anchors and Mangrove runtime. Forensics: `/tmp/mamba-runtime-forensics-20260710.json`
> (SHA-256 `d9c43bcf811b6e9f2f50de33c7c9d94ec01dcc18e25b84e4ec492d2592ea32a9`).

- Owner repo: `mamba` (prefix `mamba-`), upstream `https://github.com/state-spaces/mamba`. Finished 2026-07-09.
- **Historical execution classification only:** Task 92609 (and retry 92635) ran end-to-end on AgentGym/H20 and its unchanged LTI scaffold score is reproducible. The agent-side 429 explains the zero edit, but it does not cure the underscaled verifier workload. Current ship classification is **SCALE_BLOCKED**.

## Mangrove deliverables
- Task link: **https://mangrove.msh.team/tasks/92609** (retry https://mangrove.msh.team/tasks/92635 — same outcome)
- Dataset version id: **18046** (dataset 309, commit `8bda352a740c5c66a8988273eeb2895cf19f4003`, branch `mls-mamba-selective-scan-verify-20260709-043829` in `git@dev.msh.team:mangrove/datasets/l2/mls-bench.git`). Item metadata: agent_to 1800, ver_to 11880, cpus 8, mem 131072, gpus 1, gpu_types ['H20'].
- Final score/reward: **0.06896** (92609) / **0.07246** (92635) — both the unchanged LTI default scaffold (agent produced 0 output tokens). `has_exception=true` (agent `NonZeroAgentExitCodeError` from API 429). Verifier ran all 3 settings rc=0.
- Harbor base image: `msai-cn-beijing.cr.volces.com/public/bohanlyu2022/mlsbench-harbor-mamba:latest` @ **sha256:d6e16c6c8d87408686c41c6e99cd970471304e9f4be194b38fec232b555c1ed0** (also tagged `:20260709-real`). Self-contained: REAL `mamba_ssm==2.2.2+cu122torch2.4` + `causal_conv1d==1.4.0+cu122torch2.4` wheels (official releases, inherited from the prebuilt base) + refreshed REAL `vendor/mamba` (crane-appended layer). Validated by detached k1h20 anchor: `IMPORT_OK mamba_ssm 2.2.2 torch 2.4.0 cuda 12.1 gpu NVIDIA H20`.
- Leak check: **PASS.** Agent-visible `instruction.md` contains NO anchor values (0.995/0.999), NO scale constants (0.1734/0.3013), NO `sigmoid`/`ref=`, NO `leaderboard`/`score_spec`, NO hidden metric labels (`copy_acc_L384/L512`). Only benign: the deliberate disclosure that L384/L512 are hidden generalization settings, and the public baseline name `bc_only` + its read-only reference.

## RQ / grounding / scale (historical; superseded by correction above)
- 10 RQs shipped (`mamba-conv-act, mamba-selective-scan, mamba-a-stability, mamba-bc-coupling, mamba-delta-softplus, mamba-dt-init, mamba-gating, mamba-normalization, mamba-residual, mamba-state-init`), all on the SAME proven harness.
- REAL package: `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` (the real CUDA kernel) inside a tiny 2-layer selective-SSM. The selective-copy diagnostic is scientifically legitimate as a narrow paper diagnostic, but this particular 68k-parameter configuration is **not accepted as a normal repo-scale benchmark workload** (2-layer `d_model=64`, `d_state=16`, 8,000 steps, L=256/384/512).
- Render (FIXED adapter): `Generated 1/1 tasks; 0 failed`. Full `vendor/mamba` (45 files) copied into `_scaffold`, `__init__.py` guaranteed; no edit-range violation (edit region [40,48] = the `parameterize` fn); `allow_create=false`, `rigorous_codebase=true`.

## Anchors (real execution, underscaled workload; historical only)
Detached k1h20 mlaunch (`--preemptible=yes`, image = self-contained harbor-mamba) ran 3 baselines × 3 lengths. TSV `/mnt/moonfs/lvbohan-ksyun/mamba/anchor_results.tsv` (ALL_BASELINES_DONE):
```
lti       256/384/512 = 0.260 / 0.548 / 0.406   (weak; noisy)
bc_only   256/384/512 = 0.877 / 0.956 / 0.968   (mid)
selective 256/384/512 = 0.999 / 0.987 / 1.000   (strong/SOTA)
```
Ordering LTI < bc_only < selective holds at all 3 lengths; consistent with the 2026-07-02 leaderboards. score_spec calibration (strong ref = selective ≈0.995-1.0 → ~0.5; weak = LTI → ~0.1) is confirmed correct.

## Infra notes (hard-won this run)
1. **AgentGym memory cap = 235520 MiB (230 GB).** First launch (dv 18035) fail-fast 39s: `custom spec memory_mb 294912 MiB exceeds maximum 235520 MiB`. The h20 adapter emitted gpus=3 → mem 288 GB; I overrode `gpus=1` but the GPUS-only override did NOT lower `memory_mb`. Fix: set `memory_mb=131072`, `cpus=8` for 1×H20 (dv 18046). Build itself succeeded both times (4.3 GB base pulled + scaffold COPY + pushed to the autobuild registry in ~30s).
2. **opus-4-8 (model_entry_id=1359) 429 credit limit** — both runs: 10× `rate_limit` 429 retries over ~5 min, `output_tokens=0`, `num_turns=1`, `exceeded your credit limit: 233008`. Account-level/shared; NOT a task defect. Did not burn further credit.
3. mlaunch per-user 8-GPU cap saturated (siblings) → k1h20 non-preemptible pend-loops; `--preemptible=yes --preemption-policy-never=false` scheduled immediately (anchor).
4. github release-assets CDN stalls from the devmachine; use `curl -x http://proxy.msh.work:3128` for the wheels.

## Optional resume (ONLY if/when the opus-4-8 credit pool frees and a STRONG score is wanted)
The dataset version 18046 is correct and reusable; just relaunch and poll. Do NOT loop on 429.
```bash
BASE=https://mangrove.msh.team; H="X-Moongate-Access-Token: $MOONGATE_ACCESS_TOKEN"; VID=18046
LP=$(mktemp); cat >$LP <<JSON
{"dataset_version_id":$VID,"agent":"claude-code","agent_version":"2.1.146","model_id":"1359","model_entry_id":1359,"eval_type":"harbor","repeat_count":1,"is_think_enabled":true,"disable_internet_access":false,"use_agent_gym":true,"llm_proxy_enabled":false,"max_concurrency":1}
JSON
TID=$(curl -fsS -X POST "$BASE/api/tasks/launch" -H "$H" -H "Content-Type: application/json" --data-binary @$LP | python3 -c "import sys,json;print(json.load(sys.stdin)['task_id'])"); rm -f $LP
echo "https://mangrove.msh.team/tasks/$TID"
# poll to terminal, then audit (expect a SELECTIVE agent: copy_acc ~0.99 -> score ~0.5):
python ~/.codex/skills/mangrove-red-evals/scripts/mangrove_task.py sync $TID --out /tmp/task_${TID}_sync.json
# A run is only meaningful if agent output_tokens>0 (else it's another 429 default-scaffold run -> ignore).
```

## Second audit 2026-07-09 (subagent independent re-verification)
Re-synced task 92609 (container 4700266): finished, score/reward 0.0689615624352023, 3 verifier logs all rc=0 (49.0/54.0/69.0s), metrics.json keys copy_acc_L256/384/512, NO score_error/violation/parse_errors; has_exception = agent opus-4-8 account-level 429 (credit 233008, 12×429 in agent log, num_turns=1, 0 edits) → EXECUTION_VERIFIED_ONLY; NOT a task defect.
- **Kernel evidence (per assignment)**: harness `vendor/mamba/common.py` imports `selective_scan_fn` from `mamba_ssm.ops.selective_scan_interface` and calls it directly — real CUDA kernel, NO torch fallback path; container log shows real mamba_ssm import from /opt/conda site-packages + selective_scan_interface/triton warnings; POOL_LOADED device=cuda; TRAIN 0→7999 loss 2.8965→1.7098 real descent; default LTI scaffold copy_acc 0.423/0.521/0.354 sits inside the measured noisy-LTI band (0.260/0.548/0.406 @0709, 0.614/0.551/0.338 @0702).
- **Calibration bit-exact**: recomputed reward from container metrics.json through the shipped DSL → 0.0689615624352023 (exact match); leaderboard baselines score lti→0.10004 / bc_only→0.284 / selective→0.500; no floor, no saturation.
- **Defect found + fixed (Gate E)**: 8 sibling task_description.md files leaked rounded measured-baseline values incl. SOTA anchors ("Measured baselines 0.999/..."). De-valued all 8 (a-stability, bc-coupling, conv-act, delta-softplus, dt-init, gating, normalization, residual) to qualitative wording aligned with the deployed representative; synced mangrove-harbor/mamba-a-stability renders (instruction.md + tests/meta/task_description.md). Post-fix grep across all 10 descriptions + 2 renders: CLEAN. Live exposure was zero (DV 18046 contains only leak-free mamba-selective-scan). Sibling DV/render refresh needed only if/when siblings are launched (after credit wall lifts); tasks/ sources are already fixed.
- No mlaunch needed (anchors complete: 0702 leaderboard + 0709 k1h20 TSV, ordering lti < bc_only < selective at all lengths); no relaunch (429 credit wall — do NOT loop; resume instructions above stand: DV 18046 is reusable).
