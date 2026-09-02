"""Pre-edit operations for the verl package.

Injects parseable TRAIN_METRICS / VAL_METRICS print statements into
verl's RayPPOTrainer so that training and validation metrics appear on
stdout in a format the task parser can extract.

Target file: verl/trainer/ppo/ray_trainer.py
Target commit: 32705dc135c9a4a06f359361b3d394610ad07e0c
"""

# NOTE on output channel: Ray-captured actor stdout is unreliable in some
# configs (messages get dropped mid-run). stderr passes through Ray to the
# driver reliably (that's how tqdm and warnings come back). So we use
# sys.stderr.write(...) here instead of print() to guarantee our metric
# lines land in the SLURM .out file. Everything after the apptainer exec
# already does `2>&1`, so stderr ends up in the same log as stdout.

# ── Training metrics: after logger.log() at line 1584 ───────────────
# Indentation: 16 spaces (inside epoch → batch → step)
_TRAIN_METRICS_SNIPPET = (
    '                import sys as _sys\n'
    '                _algo_prefixes = ("actor/", "critic/", "training/")\n'
    '                _algo_keys = [k for k in metrics if any(k.startswith(p) for p in _algo_prefixes)]\n'
    '                if _algo_keys:\n'
    '                    _parts = " ".join(\n'
    '                        f"{k}={metrics[k]:.4f}" if isinstance(metrics[k], float) else f"{k}={metrics[k]}"\n'
    '                        for k in sorted(_algo_keys)\n'
    '                    )\n'
    '                    _sys.stderr.write(f"TRAIN_METRICS step={self.global_steps} {_parts}\\n")\n'
    '                    _sys.stderr.flush()'
)

# ── Validation metrics: after metrics.update(val_metrics) at line 1542 ──
# Indentation: 20 spaces (inside epoch → batch → step → if test_freq)
_VAL_METRICS_SNIPPET = (
    '                    import sys as _sys\n'
    '                    if val_metrics:\n'
    '                        _vparts = " ".join(\n'
    '                            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"\n'
    '                            for k, v in sorted(val_metrics.items())\n'
    '                        )\n'
    '                        _sys.stderr.write(f"VAL_METRICS step={self.global_steps} {_vparts}\\n")\n'
    '                        _sys.stderr.flush()'
)

# ── val_before_train metrics: after logger.log at line 1253 ────────
# Indentation: 12 spaces (inside if val_before_train block)
_VAL_BEFORE_TRAIN_SNIPPET = (
    '            import sys as _sys\n'
    '            if val_metrics:\n'
    '                _vparts = " ".join(\n'
    '                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"\n'
    '                    for k, v in sorted(val_metrics.items())\n'
    '                )\n'
    '                _sys.stderr.write(f"VAL_METRICS step={self.global_steps} {_vparts}\\n")\n'
    '                _sys.stderr.flush()'
)

# ── Memory cleanup: end of fit() inner for-loop ────────────────────
# Diagnosis: every training step puts 0.8-1.2 GB of DataProto into Ray's
# plasma shared-memory object store (via _dispatch / parallel_put for each
# of generate_sequences / compute_old_log_prob / compute_ref_log_prob /
# update_actor). Without an explicit reference drop + gc sweep, plasma
# accumulates ~1 GB/step and the cgroup kills vLLM at step ~66 on a 200 GB
# --mem SLURM job. Force the sweep at end of each step.
# Indentation: 16 spaces (inside fit → for epoch → for batch_dict).
_STEP_CLEANUP_SNIPPET = (
    '                # --- pre_edit: free Ray plasma refs from this step ---\n'
    '                try: del batch\n'
    '                except NameError: pass\n'
    '                try: del gen_batch\n'
    '                except NameError: pass\n'
    '                try: del gen_batch_output\n'
    '                except NameError: pass\n'
    '                import gc as _gc; _gc.collect()'
)

# ── Reward scorer routing ──────────────────────────────────────────
# Line 59 of verl/utils/reward_score/__init__.py:
#   elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
# We add "deepmath" and "math500" to this list so our datasets use math_dapo scorer.

# ── Operations (bottom-to-top for stable line numbers) ───────────────

OPS = [
    # Free Ray plasma refs at end of fit() inner for-loop (after line 1608,
    # which is the final `on_batch_end(batch=batch)` statement). Must come
    # before any lower-line insertion to keep anchor line numbers stable.
    {
        "op": "insert",
        "file": "verl/verl/trainer/ppo/ray_trainer.py",
        "after_line": 1608,
        "content": _STEP_CLEANUP_SNIPPET,
    },
    # Inject TRAIN_METRICS after logger.log (line 1584)
    {
        "op": "insert",
        "file": "verl/verl/trainer/ppo/ray_trainer.py",
        "after_line": 1584,
        "content": _TRAIN_METRICS_SNIPPET,
    },
    # Inject VAL_METRICS after metrics.update(val_metrics) (line 1542)
    {
        "op": "insert",
        "file": "verl/verl/trainer/ppo/ray_trainer.py",
        "after_line": 1542,
        "content": _VAL_METRICS_SNIPPET,
    },
    # Inject VAL_METRICS for val_before_train (after logger.log at line 1253)
    {
        "op": "insert",
        "file": "verl/verl/trainer/ppo/ray_trainer.py",
        "after_line": 1253,
        "content": _VAL_BEFORE_TRAIN_SNIPPET,
    },
    # Route deepmath/math500 to math_dapo reward scorer (line 59)
    {
        "op": "replace",
        "file": "verl/verl/utils/reward_score/__init__.py",
        "start_line": 59,
        "end_line": 59,
        "content": '    elif data_source in ["math_dapo", "math", "math_dapo_reasoning", "deepmath", "amc23"] or data_source.startswith("aime"):\n',
    },
    # Fix agent_loop.py KeyError: reward_extra_infos keys differ across batch items
    # Lines 769-771: use union of keys + .get(key, None) instead of first item's keys
    {
        "op": "replace",
        "file": "verl/verl/experimental/agent_loop/agent_loop.py",
        "start_line": 769,
        "end_line": 771,
        "content": (
            "        reward_extra_keys = set()\n"
            "        for info in reward_extra_infos:\n"
            "            reward_extra_keys.update(info.keys())\n"
            "        for key in reward_extra_keys:\n"
            "            non_tensor_batch[key] = np.array([info.get(key, None) for info in reward_extra_infos])\n"
        ),
    },
    # Fix gsm8k scorer: add \boxed{} fallback when #### pattern not found
    # Lines 32-33: when no "#### number" match, try \boxed{number}
    {
        "op": "replace",
        "file": "verl/verl/utils/reward_score/gsm8k.py",
        "start_line": 32,
        "end_line": 33,
        "content": (
            '        if len(solutions) == 0:\n'
            '            boxed_match = re.findall("\\\\\\\\boxed\\\\{(-?[0-9.,]+)\\\\}", solution_str)\n'
            '            if boxed_match:\n'
            '                final_answer = boxed_match[-1].replace(",", "").replace("$", "")\n'
            '            else:\n'
            '                final_answer = None\n'
        ),
    },
    # Fix protocol.py list_of_dict_to_dict_of_list: use intersection of keys
    # Lines 204-209: different dicts may have different keys → use intersection
    {
        "op": "replace",
        "file": "verl/verl/protocol.py",
        "start_line": 204,
        "end_line": 209,
        "content": (
            "    keys = set(list_of_dict[0].keys())\n"
            "    for d in list_of_dict[1:]:\n"
            "        keys &= set(d.keys())\n"
            "    output = {key: [] for key in keys}\n"
            "    for data in list_of_dict:\n"
            "        for key in keys:\n"
            "            output[key].append(data[key])\n"
        ),
    },
    # Fix math_dapo scorer: fallback to \boxed{} extraction when Answer: pattern not found
    # Line 181: add boxed fallback after regex match fails
    {
        "op": "replace",
        "file": "verl/verl/utils/reward_score/math_dapo.py",
        "start_line": 181,
        "end_line": 181,
        "content": (
            '    if match:\n'
            '        extracted_answer = match[-1]\n'
            '    else:\n'
            '        _boxed = last_boxed_only_string(solution_str)\n'
            '        extracted_answer = remove_boxed(_boxed) if _boxed else "[INVALID]"\n'
        ),
    },
    # Fix protocol.py concat: relax strict assert on meta_info key 'reward_extra_keys'
    # Lines 961-966 after the list-of-dicts edit above: different batches may
    # have different reward_extra_keys → merge lists and skip conflicting
    # scalar metadata.
    {
        "op": "replace",
        "file": "verl/verl/protocol.py",
        # list_of_dict_to_dict_of_list above expands by one line, so include
        # the enclosing ``else`` at 961 when the operations are applied.
        "start_line": 961,
        "end_line": 966,
        "content": (
            "                    else:\n"
            "                        if k in merged_meta_info:\n"
            "                            if isinstance(v, (list, set)) and isinstance(merged_meta_info[k], (list, set)):\n"
            "                                merged_meta_info[k] = list(set(list(merged_meta_info[k])) | set(list(v)))\n"
            "                            elif merged_meta_info[k] != v:\n"
            "                                pass  # silently skip conflicting non-list meta_info\n"
            "                        else:\n"
            "                            merged_meta_info[k] = v\n"
        ),
    },
    # ── verl #2490: dynamic_bsz NCCL deadlock fix (dp_actor.py) ─────────
    # With use_dynamic_bsz=True, each DP rank packs micro-batches by *token*
    # count, so ranks can pack a different *number* of micro-batches. Under
    # FSDP every micro-batch forward issues a param all-gather across the
    # sharding group (= WORLD, ulysses SP=1), so a rank with one extra
    # micro-batch issues one extra all-gather with no partner → the NCCL
    # collective stream desyncs → 3600 s watchdog → ActorDiedError at ~step 23.
    # verl already ships the cure: rearrange_micro_batches(same_micro_num_in_dp=True)
    # does all_reduce(num_micro_batches, MAX, group=dp_group) to pad every rank
    # to the same count — but only when a dp_group is passed. The pinned commit
    # (32705dc…) leaves both dp_actor call sites passing no dp_group, so the
    # padding path is dormant. Thread WORLD through both call sites. Padding only
    # equalizes the *number* of forward passes (the same sequences are processed),
    # so this changes no training math and is comparability-safe.
    #
    # Ordered bottom-to-top (line 560 before 468) so the first replace's +line
    # shift does not move the second anchor.
    #
    # update_policy (~L558-560): actor update micro-batch packing.
    {
        "op": "replace",
        "file": "verl/verl/workers/actor/dp_actor.py",
        "start_line": 560,
        "end_line": 560,
        "content": (
            "                    # verl #2490: pass DP group (=WORLD; ulysses SP=1) so dynamic_bsz\n"
            "                    # all_reduce(MAX)-pads every rank to the same micro-batch count and\n"
            "                    # FSDP param all-gathers stay in lockstep (no NCCL deadlock).\n"
            "                    _dp_group = torch.distributed.group.WORLD if torch.distributed.is_initialized() else None\n"
            "                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len, dp_group=_dp_group)\n"
        ),
    },
    # compute_log_prob (~L466-468): old/ref/rollout log-prob recompute micro-batch packing.
    {
        "op": "replace",
        "file": "verl/verl/workers/actor/dp_actor.py",
        "start_line": 468,
        "end_line": 468,
        "content": (
            "            # verl #2490: pass DP group (=WORLD; ulysses SP=1) so dynamic_bsz\n"
            "            # all_reduce(MAX)-pads every rank to the same micro-batch count and\n"
            "            # FSDP param all-gathers stay in lockstep (no NCCL deadlock).\n"
            "            _dp_group = torch.distributed.group.WORLD if torch.distributed.is_initialized() else None\n"
            "            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len, dp_group=_dp_group)\n"
        ),
    },
    # ── validation metrics: tolerate None extra-info values ────────────────
    # The agent_loop.py op above aggregates reward_extra_info over the union of
    # keys and fills None where a scorer did not emit a key.  The Lite/RL
    # validation set mixes float scorers (gsm8k, MATH-500 -> only "acc") with a
    # dict scorer (amc23 -> "score"/"acc"/"pred"), so every gsm8k/MATH-500
    # sample carries pred=None and process_validation_metrics crashed with
    # `unsupported operand type(s) for +: 'NoneType' and 'NoneType'` at the
    # first validation.  Drop None entries before reducing (~L596-598).
    {
        "op": "replace",
        "file": "verl/verl/trainer/ppo/metric_utils.py",
        "start_line": 596,
        "end_line": 598,
        "content": (
            "                # skip empty, string, or missing values (the agent-loop reward\n"
            "                # aggregation fills None for keys a scorer did not emit)\n"
            "                var_vals = [v for v in var_vals if v is not None]\n"
            "                if not var_vals or isinstance(var_vals[0], str):\n"
            "                    continue\n"
        ),
    },
]
