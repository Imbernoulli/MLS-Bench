"""DEHB (Differential Evolution Hyperband) baseline for opt-hyperparameter-search.

Reference: Awad, Mallik & Hutter (2021). "DEHB: Evolutionary Hyperband for
Scalable, Robust and Efficient Hyperparameter Optimization." IJCAI.

DEHB combines Differential Evolution (DE) with Hyperband's multi-fidelity
scheduling. It maintains a population of configurations at each fidelity
level and uses DE mutation/crossover to generate new candidates, while
Successive Halving promotes the best to higher fidelities.
"""

_FILE = "scikit-learn/custom_hpo.py"

_CONTENT = """\

class CustomHPOStrategy:
    \"\"\"DEHB: Differential Evolution + Hyperband.\"\"\"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.eta = 3
        self.mutation_factor = 0.5
        self.crossover_prob = 0.5
        self._initialized = False
        # Queue items: (cfg, fid)
        self._queue = []
        # fid -> list of (vec, score)  -- current evaluated population
        self._populations = {}
        self._fidelities = []
        # fid -> list of (trial_vec, target_idx) for pending DE trials at fid.
        # When score arrives for a trial, do DE selection against target.
        self._pending = {}
        # Ensure we only promote each fid->next once per generation
        self._promoted_rounds = {}

    def _encode(self, config, space):
        vec = []
        for p in space.params:
            val = config[p.name]
            if p.type == \"categorical\":
                idx = p.choices.index(val)
                vec.append(idx / max(len(p.choices) - 1, 1))
            elif p.type in (\"float\", \"int\"):
                if p.log_scale:
                    v = (np.log(val) - np.log(p.low)) / (np.log(p.high) - np.log(p.low))
                else:
                    v = (val - p.low) / (p.high - p.low)
                vec.append(float(np.clip(v, 0, 1)))
        return np.array(vec)

    def _decode(self, vec, space):
        config = {}
        for i, p in enumerate(space.params):
            v = float(np.clip(vec[i], 0, 1))
            if p.type == \"categorical\":
                idx = int(round(v * max(len(p.choices) - 1, 1)))
                idx = min(idx, len(p.choices) - 1)
                config[p.name] = p.choices[idx]
            elif p.type == \"float\":
                if p.log_scale:
                    config[p.name] = float(np.exp(
                        np.log(p.low) + v * (np.log(p.high) - np.log(p.low))))
                else:
                    config[p.name] = float(p.low + v * (p.high - p.low))
            elif p.type == \"int\":
                if p.log_scale:
                    config[p.name] = int(round(np.exp(
                        np.log(p.low) + v * (np.log(p.high) - np.log(p.low)))))
                else:
                    config[p.name] = int(round(p.low + v * (p.high - p.low)))
        return config

    def _de_mutate(self, target_idx, population):
        \"\"\"DE/rand/1/bin mutation and crossover.\"\"\"
        pop_vecs = [p[0] for p in population]
        n = len(pop_vecs)
        if n < 4:
            return pop_vecs[target_idx] + self.rng.randn(len(pop_vecs[0])) * 0.1

        idxs = list(range(n))
        idxs.remove(target_idx)
        a, b, c = self.rng.choice(idxs, 3, replace=False)
        mutant = pop_vecs[a] + self.mutation_factor * (pop_vecs[b] - pop_vecs[c])
        mutant = np.clip(mutant, 0, 1)

        # Crossover
        dim = len(mutant)
        cross_mask = self.rng.rand(dim) < self.crossover_prob
        j_rand = self.rng.randint(dim)
        cross_mask[j_rand] = True
        trial = np.where(cross_mask, mutant, pop_vecs[target_idx])
        return trial

    def _init(self, space, total_budget):
        s_max = max(0, int(np.floor(np.log(total_budget) / np.log(self.eta))))
        s_max = min(s_max, 3)
        pop_size = max(4, space.dim + 1)

        # Build strictly increasing fidelity ladder with no duplicates.
        # The harness clips fidelity to [0.1, 1.0] in run_hpo_loop, so we
        # must still respect that floor, but we dedupe to avoid wasting
        # SH rounds on near-identical fidelities (e.g. 0.037 -> 0.1 and
        # 0.111 would otherwise both collapse near 0.1).
        seen = set()
        for s in range(s_max, -1, -1):
            raw = 1.0 / self.eta ** s
            fid = max(raw, 0.1)
            key = round(fid, 3)
            if key in seen:
                continue
            seen.add(key)
            self._fidelities.append(fid)
            self._pending[fid] = []
            pop = []
            for i in range(pop_size):
                cfg = space.sample_uniform(self.rng)
                vec = self._encode(cfg, space)
                pop.append((vec, None))
                # target_idx = -1 -i means initial eval; we encode as negative
                # of (i+1) so it's distinguishable from real DE trial indices.
                self._pending[fid].append((vec, -(i + 1)))
                self._queue.append((cfg, fid))
            self._populations[fid] = pop
        self._initialized = True

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        if not self._initialized:
            self._init(space, budget_left + len(history))

        # When a trial comes back, match to pending and do DE selection
        # against the target it was generated from.
        if history:
            last = history[-1]
            last_vec = self._encode(last.config, space)
            # Match by (fidelity, vec) among pending trials
            for fid in self._fidelities:
                if abs(fid - last.budget) >= 0.05:
                    continue
                pending = self._pending[fid]
                pop = self._populations[fid]
                for j, (trial_vec, tgt_idx) in enumerate(pending):
                    if np.allclose(trial_vec, last_vec, atol=1e-3):
                        if tgt_idx < 0:
                            # Initial-population eval: just fill in score.
                            real_idx = -(tgt_idx + 1)
                            pop[real_idx] = (trial_vec, last.score)
                        else:
                            tgt_vec, tgt_score = pop[tgt_idx]
                            # DE selection: keep better of target/trial.
                            if tgt_score is None or last.score >= tgt_score:
                                pop[tgt_idx] = (trial_vec, last.score)
                            # else: keep target unchanged.
                        pending.pop(j)
                        break

        if self._queue:
            return self._queue.pop(0)

        sorted_fids = sorted(self._fidelities)
        # Track generation counter per fidelity to alternate between
        # DE evolution at lowest fidelity and inheritance-based
        # promotion to higher fidelities (DEHB modified SH).
        if not hasattr(self, "_gen_count"):
            self._gen_count = {f: 0 for f in sorted_fids}

        lo_fid = sorted_fids[0]
        lo_pop = self._populations[lo_fid]
        lo_ready = (all(s is not None for _, s in lo_pop)
                    and not self._pending[lo_fid])

        if lo_ready:
            # Step 1: evolve the lowest-fidelity population.
            for tgt_idx in range(len(lo_pop)):
                trial_vec = self._de_mutate(tgt_idx, lo_pop)
                trial_cfg = self._decode(trial_vec, space)
                trial_cfg = space.clip(trial_cfg)
                self._pending[lo_fid].append((trial_vec, tgt_idx))
                self._queue.append((trial_cfg, lo_fid))
            self._gen_count[lo_fid] += 1
            # Step 2: every eta generations, promote top configs to the
            # next fidelity level via successive halving (DEHB design).
            for i in range(len(sorted_fids) - 1):
                hi_fid = sorted_fids[i + 1]
                src_fid = sorted_fids[i]
                if self._pending[hi_fid]:
                    continue
                src_pop = self._populations[src_fid]
                if any(s is None for _, s in src_pop):
                    continue
                # Gate promotion so it runs at most once per fresh src
                # generation (avoid unbounded queue growth).
                if self._gen_count[hi_fid] >= self._gen_count[src_fid]:
                    continue
                scored = sorted(
                    [(s, v) for v, s in src_pop],
                    key=lambda x: x[0],
                    reverse=True,
                )
                n_promote = max(1, len(scored) // self.eta)
                top_vecs = [v for _, v in scored[:n_promote]]
                hi_pop = self._populations[hi_fid]
                # Overwrite hi_pop members with promoted vecs, then
                # re-evaluate each at the higher fidelity.
                new_hi_pop = []
                for k in range(len(hi_pop)):
                    vec = top_vecs[k % len(top_vecs)]
                    new_hi_pop.append((vec, None))
                    cfg = self._decode(vec, space)
                    cfg = space.clip(cfg)
                    self._pending[hi_fid].append((vec, -(k + 1)))
                    self._queue.append((cfg, hi_fid))
                self._populations[hi_fid] = new_hi_pop
                self._gen_count[hi_fid] = self._gen_count[src_fid]
            if self._queue:
                return self._queue.pop(0)

        # Fallback: random full fidelity
        return space.sample_uniform(self.rng), 1.0


"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 255,
        "end_line": 326,
        "content": _CONTENT,
    },
]
