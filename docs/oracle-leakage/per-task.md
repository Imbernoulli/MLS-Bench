# Per-task fix notes

For each task: the leak mechanism (how the editable code could reach the answer or
the eval identity), the fix, and the leak-audit verdict on the agent-visible
`environment/_scaffold/`. "Held-out under tests/" means the DGP / answer / scorer
lives only in the verifier-side `tests/` tree, never bind-mounted to the model.

Audit method: for every task we listed the scaffold tree, read the editable file,
grepped the whole scaffold for dataset identity / answer / DGP / self-scorer, and
confirmed the held-out material exists only under `tests/`. Verdicts were
re-verified by hand, not taken from a script.

---

## Pattern A — out-of-process (predict-then-score)

The editable template was restructured to read pre-generated inputs and emit a
prediction artifact; the DGP + answer + metric moved to `holdout/<task>/dgp.py`
(host-only) and `tests/meta/{dgp,parser}.py` (verifier). Inputs are pre-generated
deterministically by `tests/eval/_inputgen/.../mid_edit.py` (same seed → byte-
identical → honest results preserved).

### causal-discovery-discrete
- **Leak:** editable `run_causal_discovery(X)` shared a process with a bnlearn DGP +
  the true DAG + SHD/precision/recall metric, and the network identity was nameable.
- **Fix:** scaffold = `custom_algorithm.py` (stub) + `run_eval.py` (loads pre-gen `X`
  by an **opaque label**, scrubs `ENV`/`SEED`/`CAUSAL_INPUTS_DIR`, emits a base64
  endpoint matrix). DGP+truth+metric in `holdout/causal-discovery-discrete/dgp.py`;
  scored networks (alarm/cancer/child/hailfinder/win95pts) only in hidden
  `tests/eval/scripts/eval_*.sh`.
- **Verdict:** LEAK-CLOSED. (Also scrubbed a stale `data_gen.py`/`metrics.py` read-
  pointer from instruction.md — those files no longer exist in the scaffold.)

### causal-observational-linear-gaussian / -linear-non-gaussian / -nonlinear
- **Leak:** editable `run_causal_discovery(X)` co-located with the SEM/LiNGAM/ANM DGP
  (seed-derivable) + true adjacency + SHD/F1 metric.
- **Fix:** scaffold = `custom_algorithm.py` (stub) + `run_eval.py` (loads pre-gen `X`,
  emits base64 `B_est`). The driver exposes only synthetic-family hyper-parameter
  names (`--graph_type`, `--n_nodes`, `--noise_type`, …) — not a dataset name; the
  exact scored (params, seed) combos live in hidden `tests/eval/scripts/eval_*.sh`,
  and `simulate_dag` lives only in `holdout/<task>/dgp.py`. `tests/meta/config.json`
  was regenerated to drop the stale `data_gen.py`/`metrics.py` declarations.
- **Verdict:** LEAK-CLOSED (all three). Stale instruction.md read-pointers scrubbed.

### causal-treatment-effect
- **Leak:** editable `CATEEstimator` co-located with the IHDP/Jobs/ACIC DGP (which
  also produces true τ) + PEHE/ATE metric; datasets were named in the scaffold.
- **Fix:** scaffold = `custom_cate.py` whose editable `CATEEstimator` only ever sees
  `(X, T, Y)`; `--dataset` is an **opaque token**; predictions emitted as
  `CATE_PRED`. DGP+τ+metric in `holdout/causal-treatment-effect/dgp.py`; the host
  parser maps token→dataset via `dgp.opaque_label` and scores `dgp.compute_pehe`.
- **Verdict:** LEAK-CLOSED. (Also removed the `DATASETS=(...)` list + the argparse
  `choices=` that named the datasets and rejected the opaque token.)

### ml-anomaly-detection / ml-clustering-algorithm / ml-ensemble-boosting
- **Leak:** editable detector/clusterer/booster co-located with the dataset loader
  (named real datasets: cardio/thyroid/…, blobs/moons/digits, breast_cancer/…), the
  train/test split, the test labels, and the AUROC/ARI/accuracy metric.
- **Fix:** scaffold = one editable `.py` that loads a pre-generated input blob named
  by an **opaque sha1 token** and emits predictions. Loader+labels+metric +
  real-name→loader map in `holdout/<task>/dgp.py`; the blob (`mid_edit.py`) contains
  only non-answer arrays (anomaly: `X_train,X_test`; clustering: `X, n_clusters`;
  boosting: `X_train,y_train,X_test` — **no `y_test`**). boosting additionally scrubs
  `ENV`/`SEED` from `os.environ` at import before the editable class is defined.
- **Verdict:** LEAK-CLOSED (all three). Token recomputation matched the run scripts
  (e.g. anomaly cardio→`206e182f63f7`). Note: clustering's instruction.md (a DBSCAN
  baseline reference) discloses data *geometry* + a tuned `eps`; this reveals neither
  the env→token map, the seed, nor the held-out assignments — see `residuals.md`.

### ml-missing-data-imputation
- **Leak:** editable `CustomImputer` co-located with the loader (breast_cancer/wine/
  california), the true matrix, the missingness mask, and the RMSE/downstream metric.
- **Fix:** scaffold loads the **masked matrix only** (opaque token); truth (true
  matrix + mask + labels) + metric in `holdout/ml-missing-data-imputation/dgp.py`.
  The blob (`mid_edit.py`) is `np.save(X_missing)` — neither true values nor mask.
- **Verdict:** LEAK-CLOSED. (Also made `mid_edit.py` import `dgp` lazily +
  `if dgp is not None:` so `budget_check.py`, which imports `mid_edit` in a holdout-
  free scratch dir, no longer crashes; added a `/data/adbench`-style data path so the
  host scorer finds the baked dataset under the verifier's stripped env.)

### ml-symbolic-regression
- **Leak:** editable GP operators co-located with the benchmark target functions
  (Nguyen/Koza) + the held-out test labels + R².
- **Fix:** scaffold sees only *samples* of the target on training inputs (`X_train,
  y_train, X_test` — no `y_test`, no closed form); `SR_TASK` is a **salted** opaque
  token. Target functions + truth + R² in `holdout/ml-symbolic-regression/dgp.py`.
- **Verdict:** LEAK-CLOSED.

### optimization-diagonal-net
- **Leak:** editable optimizer co-located with the problem generator + `w_star`.
- **Fix:** scaffold = `custom_optimizer.py` (gets `dim,sparsity,delta` + gradients) +
  `fixed_benchmark.py` (loads pre-gen `.npz.b64`, success = observable
  `final_test_mse < 1.0`). Generator + `w_star` in `holdout/…/dgp.py`; the blob has
  `X_train,y_train,X_test,y_test` (the recovery target's clean labels) but never
  `w_star`. **Inputgen fix:** staged the 4 `scripts/*.sh` into
  `tests/eval/_inputgen/tasks/optimization-diagonal-net/scripts/` (they drive
  materialization — without them the inputgen produced 0 inputs).
- **Verdict:** LEAK-CLOSED. (Residual: `w_star` is least-squares-recoverable from the
  shipped clean `(X,y)` — inherent to a recovery task; see `residuals.md`.)

### optimization-multi-objective
- **Leak:** editable MOEA co-located with the ZDT/DTLZ problem + analytic Pareto front
  + HV/IGD metric; the problem was nameable.
- **Fix:** scaffold sees an **opaque `p#`** key + a name-free marshalled objective; it
  emits the final population (`MOEA_PRED`). Problem names, ALIASES (zdt1→p0…), front,
  ref_point, metrics, generator in `holdout/…/dgp.py`.
- **Verdict:** LEAK-CLOSED. (Residual: the marshalled evaluator could be disassembled
  to re-derive the front — name-free mitigates but doesn't eliminate; `residuals.md`.)

### optimization-parity
- **Leak:** editable hooks co-located with the hidden parity secret S + held-out test
  labels + accuracy; the (n,k) config was nameable.
- **Fix:** scaffold sees secret-free binary `x` + bit-packed *train* labels; emits
  thresholded predictions. Secret S (`seed+17`), labeling fn, test labels, scorer,
  and the real (n,k) set in `holdout/…/dgp.py`. The scaffold's `n32_k8` default is a
  **decoy** matching none of the scored configs (16,4)/(50,8)/(64,12)/(64,8).
  **Parser fix:** `_TAG_RE` accepts `n32_k8` and `n32-k8` (runner emits underscore,
  cmd label uses hyphen) and filters by parsed `(n,k)`, not raw label equality.
- **Verdict:** LEAK-CLOSED. (Residual: full-pool labels ship → S is GF(2)-recoverable
  in-process — inherent to supervised parity learning; `residuals.md`.)

### optimization-variance-reduction
- **Leak:** editable optimizer co-located with the synthetic conditioned-problem
  generator + `w_true` + the held-out optimum scorer.
- **Fix:** scaffold (`custom_vr.py` + `vr_driver.py`) gets only observable arrays; for
  the synthetic `conditioned` problem `y_test=None` and predictions are emitted for
  host-side scoring; the in-container `evaluate()` touches only public MNIST/CIFAR.
  `w_true`, synthetic `y_test`, generator, scorer in `holdout/…/dgp.py`.
- **Verdict:** LEAK-CLOSED. (Residuals: ridge minimizer is recomputable from
  `(X_train,y_train)` — self-documented, inherent; a docstring wrongly calls
  `vr_driver.py` unreachable — cosmetic, it holds no held-out material; `residuals.md`.)

---

## Pattern B — mask-arg (zero the answer argument at eval)

The editable `forward`/`predict` was handed the answer as an argument; at eval the
fixed wrapper zeroes those values (shape preserved) before calling the editable code
and keeps the true answer only in the fixed scope for the metric. Honest predictions
unchanged → no re-run.

### ai4sci-mol-property-prediction
- **Leak:** editable `MoleculeModel.forward(batch)` where `batch.targets` carried the
  answer.
- **Fix:** in `evaluate()` (outside editable 115–207): `true_targets = batch.targets`;
  `batch = replace(batch, targets=torch.zeros_like(batch.targets))`; `preds =
  model(batch)`; metric uses `true_targets`. Eval inside functions (no module globals).
- **Verdict:** LEAK-CLOSED.

### ai4sci-pla-binding-affinity
- **Leak:** editable `AffinityModel.forward(batch)` where `batch.labels` carried the
  answer.
- **Fix:** in `evaluate()` (outside editable 101–191): `true_labels = batch.labels`;
  `batch = replace(batch, labels=torch.zeros_like(batch.labels))`; metric uses
  `true_labels`. (`forward` may read `batch.labels.size(0)` for the graph count — shape
  only.)
- **Verdict:** LEAK-CLOSED.

### graph-signal-propagation  ← FIXED THIS PASS
- **Leak (found this pass):** the `_mask_y` mask-arg fix was present, but the eval body
  ran directly under `if __name__=='__main__':`, so `data`/`data_split` (real `.y`)
  were **module globals** the editable `CustomFilter.forward(data)` could read
  (`data_split.y`), bypassing the arg mask. **LEAK-OPEN.**
- **Fix:** wrap the eval body in `def _main(): … ; if __name__=='__main__': _main()`
  so `data`/`data_split` become locals — matching the 6 sibling wrap-main tasks
  (`_mask_y` kept as belt-and-suspenders). Applied identically to scaffold + pristine
  (byte-identical) + both `custom_template` copies + the dev source.
- **Verdict:** LEAK-CLOSED (empirically: a cheat `forward` reading `data_split.y` now
  raises `NameError: name 'data_split' is not defined`; honest reward unchanged).

---

## Pattern C — wrap-main (eval body → function, answer becomes local)

The eval block ran at module top level so the held-out target was a module global the
editable code could read. The fix moves the block into `def _main(): … ; if
__name__=='__main__': _main()` (body already indented → line numbers don't shift).
Honest code never read a `__main__` global → no re-run.

### ai4sci-climate-emulation
- `Custom.forward(x)` takes no target; eval wrapped in `_main()` so `test_dataset`/
  `all_targets` are locals. **LEAK-CLOSED.**

### ai4sci-weather-forecast-aggregation
- `VariableAggregator.forward(x)` takes no target; eval wrapped in `_main()` so the
  per-batch true targets are loop locals. **LEAK-CLOSED.**

### graph-link-prediction
- `LinkPredictor.forward(x, edge_index, edge_label_index)` gets candidate edges but no
  pos/neg label tensor; eval wrapped (`main()`/`_main()`) so `test_data`/`split_edge`
  are locals; pos/neg scored separately in fixed scope. **LEAK-CLOSED.**

### graph-node-classification
- `CustomGNN.forward(x, edge_index)` gets no labels (called as `model(data.x,
  data.edge_index)`); eval wrapped in `_main()` so `data.y` is a local. **LEAK-CLOSED.**

### ml-calibration
- Editable `CalibrationMethod` (45–102): `fit(probs, labels)` gets the **calibration-
  set** labels (intended); `predict_proba(probs)` gets no labels. `y_test` is a local
  in `main()`/`_main()`; ECE/Brier/NLL computed in fixed scope. **LEAK-CLOSED.**

### ml-selective-deferral
- Editable `SelectivePolicy` (253–287): `fit(probs, y_true, groups, X)` gets
  calibration labels (intended); `acceptance_score`/`predict_accept` get no test
  labels. `y[test_idx]` passed only to the fixed `_selective_metrics`; eval wrapped in
  `run_benchmark()`/`main()`/`_main()`. **LEAK-CLOSED.**

---

## Pattern D — quant label-guard (withhold the qlib test label)

The editable `CustomModel` was handed a `DatasetH` from which it could fetch the
held-out test-split label. A host-side `_label_guard.py` (imported by the fixed
`run_workflow.py`, hidden under `tests/`) closes this.

### quant-stock-prediction / quant-graph-stock / quant-concept-drift
- **Leak (predict):** `predict()` could fetch the scoring label from the dataset
  object (`SignalRecord.generate` → `model.predict(self.dataset)` then
  `generate_label(self.dataset)`).
- **Fix (predict):** the guard monkey-patches `SignalRecord.generate` to hand
  `predict()` a **label-free view** (NaN'd label columns in `_data`/`_infer`/`_learn`
  + a blocked loader); the scorer keeps the untouched dataset.
- **Leak (fit) — found + FIXED THIS PASS:** the editable range includes `fit()`, which
  runs on the **un-guarded** dataset before the predict view exists; an adversarial
  `fit()` could `dataset.prepare("test", col_set=["feature","label"])`, stash the real
  test label, and replay it in `predict()` (IC≈1). The shipped LSTM/transformer
  baselines demonstrate test-segment prepare is reachable at fit.
- **Fix (fit):** extend the guard to also patch `DatasetH.prepare` to NaN label
  columns **only for rows inside the test-segment date range**; train/valid labels
  pass through. Honest fits are unaffected (lgbm prepares only train/valid;
  lstm/transformer unpack `df_test` but never use it). The host scorer reads the real
  test label by toggling the guard off (`_GUARD_ACTIVE`).
- **Verdict:** LEAK-CLOSED for both predict and fit (verified: honest LSTM IC matches
  the anchor; fit-cheat now yields NaN, not IC≈1). One documented residual remains
  (`$close` reconstruction from the public price provider — needs a data-level change,
  out of scope); see `residuals.md`.
