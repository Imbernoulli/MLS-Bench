#!/usr/bin/env python3
"""codegen-sample-budget harness (fixed pipeline).

The editable policy returns one hardness weight for every problem. A
deterministic capped largest-remainder allocator maps those weights to 1..8
candidates per problem and exactly 1,028 candidates overall. Sampling and
selection are fixed; scoring uses the RESERVED assertions.
"""
from __future__ import annotations
import argparse, math, time
import common

TASK_ID = "codegen-sample-budget"
BUDGET = 4 * common.EXPECTED_PROBLEMS
MIN_SAMPLES = 1
MAX_SAMPLES = 8


def _instruction(prob):
    return ("Complete the following Python function. Return ONLY the complete "
            "function inside a single ```python code block.\n\n" + prob["prompt"])


def allocate_candidates(raw_weights, n):
    if type(raw_weights) not in {list, tuple} or len(raw_weights) != n:
        raise TypeError(f"allocation_weights() must return {n} weights")
    weights = [common.require_real(value, "allocation weight") for value in raw_weights]
    if any(value < 0.0 for value in weights) or not any(value > 0.0 for value in weights):
        raise ValueError("allocation weights must be non-negative and not all zero")
    scale = max(weights)
    weights = [value / scale for value in weights]
    allocations = [MIN_SAMPLES] * n
    capacity = [MAX_SAMPLES - MIN_SAMPLES] * n
    remaining = BUDGET - MIN_SAMPLES * n
    while remaining:
        active = [i for i in range(n) if capacity[i]]
        if not active:
            raise RuntimeError("candidate allocator exhausted its capacity")
        active_weights = {i: weights[i] for i in active}
        if not any(value > 0.0 for value in active_weights.values()):
            active_weights = {i: 1.0 for i in active}
        total_weight = math.fsum(active_weights.values())
        quotas = {i: remaining * active_weights[i] / total_weight for i in active}
        floors = {i: min(capacity[i], int(math.floor(quotas[i]))) for i in active}
        floor_total = sum(floors.values())
        if floor_total:
            for i, amount in floors.items():
                allocations[i] += amount
                capacity[i] -= amount
            remaining -= floor_total
            if not remaining:
                break
        candidates = [i for i in active if capacity[i]]
        candidates.sort(key=lambda i: (-(quotas[i] - math.floor(quotas[i])), i))
        take = min(remaining, len(candidates))
        if not take:
            continue
        for i in candidates[:take]:
            allocations[i] += 1
            capacity[i] -= 1
        remaining -= take
    if sum(allocations) != BUDGET or not all(
        MIN_SAMPLES <= value <= MAX_SAMPLES for value in allocations
    ):
        raise RuntimeError("candidate allocator produced an invalid budget")
    return allocations


def _policy_weights(policy, problems):
    return common.call_policy_without_generation(policy, problems)


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
    allocation_weights = common.load_surface(args.solution, "allocation_weights")
    common.emit_protocol(TASK_ID, args.seed, problems)
    n = len(problems)
    policy_problems = [
        {"task_id": p["task_id"], "entry_point": p["entry_point"], "prompt": p["prompt"]}
        for p in problems
    ]
    allocations = allocate_candidates(
        _policy_weights(allocation_weights, policy_problems), n
    )
    n_hidden = 0; spent = 0
    for i, prob in enumerate(problems):
        setup = prob.get("test_setup", "")
        k = allocations[i]
        raw = common.generate(common.GenConfig(
            prompt=_instruction(prob), n_samples=k, do_sample=True,
            temperature=0.6, top_p=0.95, max_new_tokens=512), seed=args.seed)
        cands = [common.extract_code(r) for r in raw]
        spent += len(cands)
        chosen = cands[0]
        for c in cands:
            if common.passes_all(c, prob["visible_tests"], setup):
                chosen = c; break
        hid_ok = common.passes_all(chosen, prob["hidden_tests"], setup)
        n_hidden += int(hid_ok)
        common.emit_item(i + 1, hid_ok, samples=k)
        if i < 3:
            print(f"CG_SAMPLE i={i} id={prob['task_id']} k={len(cands)} "
                  f"spent={spent} hid_ok={hid_ok}", flush=True)
        if (i + 1) % 20 == 0:
            common.emit_progress(i + 1, n_hidden)
    if spent != BUDGET:
        raise RuntimeError(f"fixed candidate budget mismatch: spent={spent} budget={BUDGET}")
    dt = time.time() - t0
    print(f"CG_METRICS task={TASK_ID} pass_at_1={n_hidden/n:.6f} "
          f"avg_samples={spent/n:.6f} n={n} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
