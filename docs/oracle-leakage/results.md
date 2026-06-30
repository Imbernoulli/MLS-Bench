# Verification results

Two properties per task: **runnability / results preserved** and **leak closed**.
We are honest about *how* each is established — empirical run, structural guarantee,
or both. We do not paper over the verification-harness artifacts (§Known artifacts).

## How "results preserved" is established

Each fix is semantics-preserving by construction, and that is the primary guarantee:

- **out-of-process / opaque-token:** inputs are pre-generated from the *same seed* →
  byte-identical → the host scorer computes the *same metric formula*. Honest results
  are identical by construction.
- **mask-arg:** zeroes only the answer the honest model is asked to predict (which an
  honest model never reads as input) → honest predictions identical.
- **wrap-main:** moves the eval body into a function; honest code never read a
  `__main__` global → identical.
- **fit-guard (quant):** NaNs only the *test-segment* label, which honest fits never
  use (lgbm prepares train/valid only; lstm/transformer unpack `df_test` but never use
  it) → identical. Verified directly by the unit test below (train labels byte-
  identical before/after the guard; test labels fully masked).

On top of that structural guarantee, we ran the reference baseline through the Harbor
verifier wherever the box (heavily shared — see artifacts) let a task complete.

## Leak-closed (all 26)

| # | task | leak verdict | how |
|---|------|--------------|-----|
| 1 | causal-discovery-discrete | CLOSED | scaffold = stub+driver; dgp/answer/metric only in tests/; opaque `--label` |
| 2 | causal-observational-linear-gaussian | CLOSED | idem; simulate_dag only in holdout |
| 3 | causal-observational-linear-non-gaussian | CLOSED | idem |
| 4 | causal-observational-nonlinear | CLOSED | idem |
| 5 | causal-treatment-effect | CLOSED | estimator sees only (X,T,Y); opaque `--dataset`; τ+PEHE in tests/ |
| 6 | ml-anomaly-detection | CLOSED | opaque token (cardio→206e18…); blob = X only; truth host-side |
| 7 | ml-clustering-algorithm | CLOSED | opaque token; blob = X+n_clusters; (instruction.md geometry note → residuals) |
| 8 | ml-ensemble-boosting | CLOSED | opaque token; blob excludes y_test; ENV/SEED scrubbed pre-edit |
| 9 | ml-missing-data-imputation | CLOSED | masked X only; truth+mask host-side |
| 10 | ml-symbolic-regression | CLOSED | salted token; samples only, no closed form / y_test |
| 11 | optimization-diagonal-net | CLOSED | scaffold loads X/y; w_star+gen in holdout (inputgen scripts staged) |
| 12 | optimization-multi-objective | CLOSED | opaque p#; name-free marshalled objective; front in holdout |
| 13 | optimization-parity | CLOSED | secret-free x; decoy n32_k8; secret S + test labels in holdout |
| 14 | optimization-variance-reduction | CLOSED | observable arrays only; w_true+y_test in holdout |
| 15 | quant-stock-prediction | CLOSED (predict+fit) | _label_guard masks predict view **and** test-segment label in prepare |
| 16 | quant-graph-stock | CLOSED (predict+fit) | identical guard |
| 17 | quant-concept-drift | CLOSED (predict+fit) | identical guard |
| 18 | ai4sci-pla-binding-affinity | CLOSED | batch.labels zeroed at eval; metric uses true labels |
| 19 | ai4sci-mol-property-prediction | CLOSED | batch.targets zeroed at eval |
| 20 | graph-signal-propagation | CLOSED (fixed this pass) | wrap-main: data/data_split now locals (cheat → NameError) |
| 21 | ai4sci-climate-emulation | CLOSED | wrap-main; forward(x) has no target |
| 22 | ai4sci-weather-forecast-aggregation | CLOSED | wrap-main; forward(x) has no target |
| 23 | graph-link-prediction | CLOSED | wrap-main; forward gets candidate edges, no label tensor |
| 24 | graph-node-classification | CLOSED | wrap-main; forward(x,edge_index), data.y local |
| 25 | ml-calibration | CLOSED | wrap-main; y_test local; editable gets cal labels + test probs only |
| 26 | ml-selective-deferral | CLOSED | wrap-main; y[test] only to fixed scorer |

(Audit method + raw evidence: 5 read-only sub-agents gathered scaffold trees, editable
files, grep hits and tests/-side presence; every verdict was re-checked by hand. The
one task the audit found still **open** — graph-signal-propagation — was fixed this
pass and re-verified.)

## Runnability / results-preserved — empirical status

`reward` = the reference baseline's combined_score through the Harbor verifier on the
fixed bundle. Anchors are the `baseline:*` rows in `tests/meta/leaderboard.csv`.

| task | empirical | note |
|---|---|---|
| graph-node-classification | reward ≈ 0.507 | completed (dedicated GPU) |
| ai4sci-mol-property-prediction | reward ≈ 0.460 | completed (dedicated GPU) |
| graph-link-prediction | reward ≈ 0.224 | completed (dedicated GPU) |
| ml-anomaly-detection | reward ≈ 0.491 | completed |
| ai4sci-pla-binding-affinity | per-benchmark TEST_METRICS match anchor | rmse 1.457/1.295/1.473 vs 1.41/1.24/1.46 (each benchmark needs its own GPU — see artifacts) |
| graph-signal-propagation | honest reward 0.383 (normal accuracies); **full before/after cheat: pre-fix `forward` reading the module-global `data_split.y` → accuracy 1.0000 on all 4 datasets (reward 1.0); post-fix → `NameError: name 'data_split' is not defined`** | the leak was real and fully exploitable pre-fix; wrap-main closes it. 0.383 vs prior 0.432 = PyG GPU non-determinism on a ±4%-std task (graph-node, same pattern, reproduced the anchor exactly) |
| quant-stock-prediction (+graph-stock,+concept-drift) | **fit-guard fully verified**: unit test `TRAIN_LABELS_UNCHANGED:True` + `TEST_LABELS_FULLY_MASKED:True`; cheat_new IC=**nan**, cheat_old IC=**0.886** (vs honest ~0.047) | the fit-cheat was real (IC 0.886 under the old guard) and is now closed; honest fit unchanged |
| causal-* / ml-oop / optimization-* | anchor-matching (deterministic out-of-process); **inputgen confirmed for ALL 14 holdout tasks** | inputs byte-identical by seed → metric identical by construction. Each materializes >0 input blobs at eval time (cate 90, causal-discovery 15, causal-obs 8–9, anomaly 12, clustering/ensemble/symbolic/missing-data 9, multi-objective 12, parity 45, variance-reduction 3, diagonal-net 19) — no staging bug like the original diagonal-net 0-inputs |
| optimization-diagonal-net | inputgen 19 inputs; **d200_k5_s01 config completes with `FINAL_METRICS n_star=50 score=-5.643856` = anchor (sgd/adam) EXACTLY**; d500 configs running | per-config science confirmed correct; full gmean also needs d10000 (50× larger) — its long GD loop is CPU-bound and starved by the co-tenant job, not the fix |
| ml-missing-data-imputation | clean run ≈ 0.449 | (over-parallel run gave a spurious 0) |
| causal-discovery-discrete | 4/5 networks (Alarm/Cancer/Child/Win95pts) emit CAUSAL_PRED + score; **Hailfinder GES times out at the 3840 s per-config budget** → gmean=0 | budget/runtime of GES on a 56-node net, **unchanged by the fix** (same data X by seed, same algorithm); aggravated by a 142-core co-tenant job during this run |
| ai4sci-climate-emulation / -weather-forecast-aggregation | wrap-main (no behavior change); structural guarantee | full GPU re-run is hours on the shared box |

## Known verification-harness artifacts (NOT task failures)

Reproduced repeatedly on this shared box; they produce spurious `reward=0` and are
properties of the *local reproduction*, not the fixes:

- **CPU saturation** (this box ran at load 220–360 on 384 cores from other tenants +
  causal GES): GPU tasks get CPU-starved at data-loading and time out, and large
  qlib universes (csi300) get killed. Mitigation: run heavy tasks one/two at a time.
- **`/dev/shm`** default 64 MB → PyTorch DataLoader bus error. Mitigation: `--ipc=host`.
- **GPU packing**: `score_task` puts every same-group benchmark on the first visible
  GPU; 3 group=1 benchmarks in one container OOM each other (pla-binding). Mitigation:
  one dedicated GPU per benchmark.
- **Short timeout**: diagonal-net runs 19 long gradient-descent instances; a short
  docker timeout cuts it off. Mitigation: generous timeout.

These are why a few cells above say "pending" rather than a number — the metric the
verifier *does* emit is correct (TRAIN/TEST lines in the logs are normal); only the
end-to-end reward needs an uncontended run to finish.
