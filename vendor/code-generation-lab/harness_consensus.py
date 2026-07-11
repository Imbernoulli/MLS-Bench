#!/usr/bin/env python3
"""codegen-self-consensus harness (fixed pipeline).

FIXED pool of 8 (T=0.6, top_p 0.95). Keep candidates passing ALL VISIBLE tests
(execute-filter); cluster survivors by the agent's `canonical(program)`; submit a
representative of the LARGEST cluster (ties -> earliest cluster -> earliest
member). If no candidate passes visible, fall back to candidate 0. Score on the
DISJOINT HIDDEN tests. Only the clustering key is editable.
"""
from __future__ import annotations
import argparse, time
from collections import defaultdict
import common

TASK_ID = "codegen-self-consistency"
POOL_N = 8; POOL_T = 0.6; POOL_P = 0.95; POOL_MAX = 512


def _instruction(prob):
    return ("Complete the following Python function. Return ONLY the complete "
            "function inside a single ```python code block.\n\n" + prob["prompt"])


def _canonical_key(policy, program):
    key = common.call_policy_without_generation(policy, program)
    hash(key)
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=common.EXPECTED_PROBLEMS)
    args = ap.parse_args()
    common.set_seeds(args.seed); t0 = time.time()
    common.start_executor()
    _tok, _ = common.load_model()
    problems = common.load_problems(args.n)
    common.seal_private_data()
    canonical = common.load_surface(args.solution, "canonical")
    common.emit_protocol(TASK_ID, args.seed, problems)
    n = len(problems); n_hidden = 0; n_oracle = 0; top_cl = 0
    total_survivors = 0; total_clusters = 0; n_agreement = 0; n_changed = 0
    for i, prob in enumerate(problems):
        setup = prob.get("test_setup", "")
        raw = common.generate(common.GenConfig(
            prompt=_instruction(prob), n_samples=POOL_N, do_sample=True,
            temperature=POOL_T, top_p=POOL_P, max_new_tokens=POOL_MAX), seed=args.seed)
        cands = [common.extract_code(r) for r in raw]
        survivors = [j for j, c in enumerate(cands)
                     if common.passes_all(c, prob["visible_tests"], setup)]
        cluster_size = 0
        cluster_count = 0
        changed = False
        if survivors:
            groups = defaultdict(list)
            for j in survivors:
                key = _canonical_key(canonical, cands[j])
                groups[key].append(j)
            cluster_count = len(groups)
            best_members, best_size, best_first = [], -1, 10**9
            for members in groups.values():
                first = min(members)
                if (len(members), -first) > (best_size, -best_first):
                    best_size, best_first, best_members = len(members), first, members
            chosen_index = min(best_members)
            chosen = cands[chosen_index]
            top_cl += best_size
            cluster_size = best_size
            changed = chosen_index != survivors[0]
        else:
            chosen = cands[0]
        hid_ok = common.passes_all(chosen, prob["hidden_tests"], setup)
        oracle = any(common.passes_all(c, prob["hidden_tests"], setup) for c in cands)
        agreement = cluster_size > 1
        n_hidden += int(hid_ok); n_oracle += int(oracle)
        total_survivors += len(survivors); total_clusters += cluster_count
        n_agreement += int(agreement); n_changed += int(changed)
        common.emit_item(
            i + 1,
            hid_ok,
            oracle=oracle,
            survivors=len(survivors),
            clusters=cluster_count,
            cluster=cluster_size,
            agreement=agreement,
            changed=changed,
        )
        if i < 3:
            print(f"CG_SAMPLE i={i} id={prob['task_id']} surv={len(survivors)} "
                  f"hid_ok={hid_ok} oracle={oracle}", flush=True)
        if (i + 1) % 20 == 0:
            common.emit_progress(i + 1, n_hidden)
    dt = time.time() - t0
    print(f"CG_METRICS task={TASK_ID} pass_at_1={n_hidden/n:.6f} "
          f"oracle_pass_at_1={n_oracle/n:.6f} "
          f"mean_survivors={total_survivors/n:.6f} "
          f"mean_clusters={total_clusters/n:.6f} "
          f"top_cluster={top_cl/n:.6f} "
          f"agreement_rate={n_agreement/n:.6f} "
          f"changed_selection_rate={n_changed/n:.6f} "
          f"n={n} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
