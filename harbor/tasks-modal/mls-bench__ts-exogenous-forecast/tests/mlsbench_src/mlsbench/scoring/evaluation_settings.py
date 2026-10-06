"""Explicit evaluation conditions produced by one or more execution batches.

Score DSL groups are metric aggregations, not necessarily distinct conditions.
Likewise, one process can produce several conditions (e.g. checkpoints) without
retraining. This opt-in contract maps real conditions to their producing command
and score groups, and checks completeness BEFORE seed averaging. It does not
prove evaluator correctness or replace a Harbor execution test.

A task that declares no contract still gets the weaker default rule in
``check_command_seed_coverage``: one successful run per command and seed, or the
record is invalid -- see that function for the coverage rule it enforces.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .spec import load_score_spec


CONDITION_AXES = frozenset({
    "dataset", "subset", "corruption", "resolution", "checkpoint", "nfe",
    "context_length", "guidance_scale", "sequence_length", "protocol",
})


def evaluation_contract(task_dir: Path, config: dict | None = None) -> dict | None:
    """Validate an explicit contract; None preserves legacy command semantics.

Every scoring group must be owned exactly once, including auxiliary cost
groups, which are checked but never counted as evaluation conditions.
"""
    config = config if config is not None else json.loads((task_dir / "config.json").read_text())
    if "evaluation_settings" not in config:
        return None
    declarations = config["evaluation_settings"]
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("evaluation_settings must be a nonempty list")
    commands = [tc.get("label", tc.get("cmd")) for tc in config.get("test_cmds", [])]
    if not commands or any(not isinstance(c, str) or not c for c in commands) or len(set(commands)) != len(commands):
        raise ValueError("evaluation_settings requires unique command labels")
    spec = load_score_spec(task_dir)
    if spec is None or not spec.settings:
        raise ValueError("evaluation_settings requires score_spec settings")
    owned = set()

    def columns(groups):
        if not isinstance(groups, list) or not groups or any(not isinstance(g, str) for g in groups):
            raise ValueError("score_settings must be a nonempty list of names")
        cols = set()
        for group in groups:
            if group not in spec.settings or group in owned:
                raise ValueError("unknown or multiply-owned score setting: " + group)
            owned.add(group)
            setting = spec.settings[group]
            terms = [t for t, w in setting.terms if w > 0] + list(setting.constraints)
            cols.update(spec.terms[t].metric for t in terms if spec.terms[t].role != "drop")
        if not cols:
            raise ValueError("evaluation setting has no required scored metrics")
        return sorted(cols)

    names, conditions, metric_sets = set(), set(), set()
    settings = []
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError("each evaluation setting must be an object")
        name, command = declaration.get("name"), declaration.get("command")
        condition = declaration.get("condition")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("evaluation setting names must be nonempty and unique")
        if command not in commands:
            raise ValueError("unknown producing command for " + name)
        if not isinstance(condition, dict) or not condition or not set(condition) <= CONDITION_AXES:
            raise ValueError("condition must name real dataset/protocol axes: " + name)
        if any(isinstance(v, bool) or not isinstance(v, (str, int, float)) or
               (isinstance(v, str) and not v.strip()) or
               (isinstance(v, (int, float)) and not math.isfinite(v)) for v in condition.values()):
            raise ValueError("invalid condition value: " + name)
        signature = json.dumps(condition, sort_keys=True)
        if signature in conditions:
            raise ValueError("duplicate evaluation condition: " + name)
        cols = columns(declaration.get("score_settings"))
        if tuple(cols) in metric_sets:
            raise ValueError("identical metric sets cannot establish distinct conditions: " + name)
        names.add(name)
        conditions.add(signature)
        metric_sets.add(tuple(cols))
        settings.append(dict(declaration, metrics=cols))
    if {s["command"] for s in settings} != set(commands):
        raise ValueError("every command must declare its evaluation conditions")
    auxiliary = config.get("evaluation_auxiliary_scores", {})
    if not isinstance(auxiliary, dict) or not set(auxiliary) <= set(commands):
        raise ValueError("invalid evaluation_auxiliary_scores command mapping")
    auxiliary = {command: columns(groups) for command, groups in auxiliary.items()}
    if owned != set(spec.settings):
        raise ValueError("unmapped score settings: " + ", ".join(sorted(set(spec.settings) - owned)))
    return {"settings": settings, "auxiliary": auxiliary, "commands": commands}


def _rc_ok(record: dict) -> bool:
    """A successful exit, judged the way score_task's ``_completed_eval_record``
    judges it. ``rc == 0`` alone is looser than it looks: ``False == 0`` is True
    in Python, so a record carrying ``rc: false`` -- which is what a JSON writer
    produces for a missing status in some shapes -- reads as success. The type
    check is what makes "the supervisor recorded a clean exit" mean that."""
    return type(record.get("rc")) is int and record["rc"] == 0


def check_evaluation_records(contract: dict, seeds: list[int], records: list[dict]) -> dict:
    """Require exactly one successful, complete result per command and seed.

Input records are produced by the trusted parser, not by an editable candidate.
Missing/duplicate records, nonzero exit codes, and missing/nonfinite metrics are
explicit errors. Neither another seed nor another command can fill their holes.
"""
    errors, evidence = [], []
    if not seeds or len(set(seeds)) != len(seeds):
        errors.append("expected seeds must be nonempty and unique")
    expected = {(command, seed) for command in contract["commands"] for seed in seeds}
    indexed = {}
    for record in records:
        key = (record.get("label"), record.get("seed"))
        if key not in expected:
            errors.append("unexpected command/seed record: %r" % (key,))
        indexed.setdefault(key, []).append(record)

    def finite(value):
        if isinstance(value, bool):
            return False
        try:
            return value is not None and math.isfinite(float(value))
        except (ValueError, TypeError, OverflowError):
            return False

    for command in contract["commands"]:
        for seed in seeds:
            found = indexed.get((command, seed), [])
            if len(found) != 1:
                errors.append("%s seed %s: expected one result, got %s" % (command, seed, len(found)))
            record = found[0] if len(found) == 1 else {}
            rc_ok = len(found) == 1 and _rc_ok(record) and not record.get("error")
            if not rc_ok:
                errors.append("%s seed %s: %s" % (
                    command, seed,
                    record["error"] if (len(found) == 1 and record.get("error"))
                    else "missing or unsuccessful exit status"))
            metrics = record.get("metrics") or {}
            for setting in contract["settings"]:
                if setting["command"] != command:
                    continue
                missing = [col for col in setting["metrics"] if not finite(metrics.get(col))]
                if missing:
                    errors.append("%s seed %s: missing/nonfinite %s" % (setting["name"], seed, missing))
                evidence.append({"name": setting["name"], "command": command, "seed": seed,
                                 "condition": setting["condition"], "valid": rc_ok and not missing,
                                 "missing": missing,
                                 "metrics": {col: float(metrics[col]) for col in setting["metrics"] if col not in missing}})
            auxiliary_missing = [col for col in contract["auxiliary"].get(command, []) if not finite(metrics.get(col))]
            if auxiliary_missing:
                errors.append("%s seed %s: missing/nonfinite auxiliary metrics %s" % (command, seed, auxiliary_missing))
    return {"schema_version": 1, "valid": not errors, "settings": evidence, "errors": errors}


def check_command_seed_coverage(commands: list[str], seeds: list[int], records: list[dict]) -> dict:
    """Default completeness rule: every (command, seed) must have produced one
    successful run.  Applies to tasks that declare no ``evaluation_settings``.

The explicit contract above is opt-in; without it the default rule still
requires one successful run per (command, seed), so a submission whose runs
fail or are killed cannot score as if they had all succeeded by averaging only
the surviving runs.

The decision this encodes: **a failed run makes the record invalid.**  It is
neither retried nor dropped.  This is the weaker half of the explicit
contract -- exit status and duplication only, no per-setting metric
requirement -- because a task with no contract may legitimately have a command
that produces no scored metric of its own, and a metric genuinely absent from
every seed is already caught downstream by ``_validate_setting``'s
missing-objective rule.

**Harness-generated exit codes count as failures too, and that is deliberate.**
score_task.py writes rc 124 for a wave timeout, 125 for a test_cmd whose
``compute`` cannot be allocated on the reserved GPUs, 126/127 for an unsafe or
missing eval script, and the budget check's own rc for a budget violation.  All
of them zero the whole submission rather than averaging the seeds that did
finish.  Dropping them instead would reopen the exploit above from the other
end: a timeout is submission-reachable (a slow enough editable path runs the
wave out of time), so "the runs that survived" would again be a best-of.  A
task that trips 125 or 127 has a bundle defect rather than a bad submission,
but the honest reading of that run is still "not scored" -- which is what
reward 0.0 means -- and the cause is named in the errors below and in
``eval_summary.json``.
"""
    errors = []
    if not commands:
        errors.append("no evaluation commands declared")
    if not seeds or len(set(seeds)) != len(seeds):
        errors.append("expected seeds must be nonempty and unique")
    expected = {(command, seed) for command in commands for seed in seeds}
    indexed: dict = {}
    for record in records:
        key = (record.get("label"), record.get("seed"))
        if key not in expected:
            errors.append("unexpected command/seed record: %r" % (key,))
        indexed.setdefault(key, []).append(record)
    evidence = []
    for command in commands:
        for seed in seeds:
            found = indexed.get((command, seed), [])
            record = found[0] if len(found) == 1 else {}
            ok = len(found) == 1 and _rc_ok(record) and not record.get("error")
            if len(found) != 1:
                errors.append("%s seed %s: expected one result, got %s" % (command, seed, len(found)))
            elif not _rc_ok(record):
                errors.append("%s seed %s: unsuccessful exit status (rc=%r)"
                              % (command, seed, record.get("rc")))
            elif record.get("error"):
                # The run exited 0 but produced no usable record -- a parser
                # exception, a missing log. Saying "expected one result, got 0"
                # here (which is what dropping the record used to produce) sends
                # the reader looking for a scheduling bug instead of at the
                # parser that raised.
                errors.append("%s seed %s: %s" % (command, seed, record["error"]))
            evidence.append({"command": command, "seed": seed, "valid": ok,
                             "rc": record.get("rc") if len(found) == 1 else None,
                             **({"error": record["error"]} if record.get("error") else {})})
    return {"schema_version": 1, "valid": not errors, "runs": evidence, "errors": errors}
