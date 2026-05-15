"""LocalSchedulerExecutor: submit test jobs to the local GPU scheduler.

Mirrors the SlurmExecutor interface so that agent test execution can
transparently use either SLURM or the local scheduler for GPU allocation.

The local scheduler must be running in the background (`mlsbench.scheduler start`).
When the agent calls test(), this executor:
  1. Builds a self-contained bash script from the local exec spec
  2. Submits it to the scheduler as a "script" job
  3. Waits for the scheduler to assign GPUs and execute it
  4. Reads outputs from the script's output directory
"""

from __future__ import annotations

import math
import os
import shlex
import time
from pathlib import Path


class LocalSchedulerExecutor:
    """Submit grouped local commands to the GPU scheduler and wait for results."""

    resubmit_cancelled_jobs = False

    def __init__(self, project_root: Path, global_config: dict | None = None):
        self.project_root = project_root
        self.logs_dir = project_root / "logs"
        self.global_config = dict(global_config or {})

    # ------------------------------------------------------------------
    # GPU assignment (same logic as SlurmExecutor)
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_gpus(entries: list[dict]) -> list[str]:
        """Return list of CUDA_VISIBLE_DEVICES strings, one per entry."""
        total_gpus = max(1, math.ceil(sum(e["compute"] for e in entries)))
        current = 0
        assignments = []
        for entry in entries:
            compute = entry["compute"]
            n = max(1, math.ceil(compute))
            n = min(n, max(1, total_gpus - current))
            start = min(current, total_gpus - 1)
            end = min(start + n, total_gpus)
            gpus = list(range(start, end))
            if not gpus:
                gpus = [total_gpus - 1]
            assignments.append(",".join(str(g) for g in gpus))
            if compute >= 1:
                current = end
        return assignments

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def _build_script(
        self,
        group_cmds: list[dict],
        out_dir: Path,
    ) -> str:
        """Generate a bash script for a group of local commands.

        Each group_cmd has:
          - label: str
          - local_cmd: list[str]   (conda-wrapped command)
          - local_cwd: str         (working directory)
          - local_env: dict        (environment variables, delta from os.environ)
          - compute: float
        """
        gpu_assignments = self._assign_gpus(group_cmds)

        from mlsbench.cli import local_thread_limit_exports

        lines = [
            "#!/bin/bash",
            "set -o pipefail",
            "",
            "# Limit CPU thread parallelism per job to avoid overloading shared machines.",
            "# Each job gets a moderate thread budget; the scheduler runs multiple jobs.",
            *local_thread_limit_exports(self.global_config),
            "",
        ]

        # Compute env delta: only export vars that differ from current env
        base_env = os.environ

        # GPU handling: The scheduler sets CUDA_VISIBLE_DEVICES to the assigned
        # physical GPUs (e.g. "3,5").  For single-command groups we inherit that
        # directly.  For multi-command groups we split the scheduler allocation
        # across sub-commands using the logical _assign_gpus mapping.
        # Never write CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES from local_env
        # into the script — the scheduler is the authoritative source.
        _gpu_env_keys = {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}

        def _local_env_parts(env: dict) -> list[str]:
            parts = []
            for k, v in sorted((env or {}).items()):
                if k in _gpu_env_keys:
                    continue
                if base_env.get(k) != v:
                    parts.append(f"{k}={shlex.quote(str(v))}")
            return parts

        def _physical_gpu_prefix(logical_gpus: str) -> str:
            physical_parts = []
            for g in logical_gpus.split(","):
                if g:
                    physical_parts.append(f"${{_SCHED_GPUS[{g}]}}")
            physical_gpus_expr = ",".join(physical_parts) if physical_parts else ""
            if not physical_gpus_expr:
                return ""
            return (
                f"CUDA_VISIBLE_DEVICES={physical_gpus_expr} "
                f"NVIDIA_VISIBLE_DEVICES={physical_gpus_expr}"
            )

        def _add_docker_rm_for_script(cmd_list: list[str]) -> list[str]:
            if (
                len(cmd_list) >= 2
                and cmd_list[0] == "docker"
                and cmd_list[1] == "run"
                and "--rm" not in cmd_list
            ):
                return [*cmd_list[:2], "--rm", *cmd_list[2:]]
            return cmd_list

        # Sentinel that survives shlex.quote (only safe chars) and is replaced
        # after rendering with shell-expanded GPU spec so the container only sees
        # the GPUs assigned by the scheduler instead of `--gpus all`.
        _GPUS_SENTINEL = "__MLSB_DOCKER_GPUS_SENTINEL__"

        def _patch_docker_gpus(cmd_list: list[str]) -> list[str]:
            if len(cmd_list) < 2 or cmd_list[0] != "docker" or cmd_list[1] != "run":
                return cmd_list
            out: list[str] = []
            i = 0
            while i < len(cmd_list):
                if (
                    cmd_list[i] == "--gpus"
                    and i + 1 < len(cmd_list)
                    and cmd_list[i + 1] == "all"
                ):
                    out.append(_GPUS_SENTINEL)
                    i += 2
                else:
                    out.append(cmd_list[i])
                    i += 1
            return out

        def _expand_gpus_sentinel(rendered: str) -> str:
            # Docker 28.x rejects bare `--gpus device=0,1` (parses as both
            # Count and DeviceIDs); the spec must arrive at the daemon
            # with literal double quotes around the `device=...` payload.
            # Use shell-escaped \" inside a "..." string so bash both
            # expands $CUDA_VISIBLE_DEVICES and preserves the surrounding
            # quotes that docker expects.
            return rendered.replace(
                _GPUS_SENTINEL,
                '--gpus "\\"device=$CUDA_VISIBLE_DEVICES\\""',
            )

        def _docker_container_name(cmd_list: list[str]) -> str | None:
            if len(cmd_list) < 2 or cmd_list[0] != "docker" or cmd_list[1] != "run":
                return None
            for i, token in enumerate(cmd_list):
                if token == "--name" and i + 1 < len(cmd_list):
                    return cmd_list[i + 1]
                if token.startswith("--name="):
                    return token.split("=", 1)[1]
            return None

        docker_container_names = sorted({
            name
            for cmd in group_cmds
            for key in ("budget_apptainer_cmd", "apptainer_cmd")
            if key in cmd
            for name in [_docker_container_name(list(cmd[key]))]
            if name
        })
        if docker_container_names:
            names = " ".join(shlex.quote(name) for name in docker_container_names)
            lines.extend([
                f"_MLSBENCH_DOCKER_CONTAINERS=({names})",
                "_cleanup_mlsbench_docker() {",
                '  for _container in "${_MLSBENCH_DOCKER_CONTAINERS[@]}"; do',
                '    docker rm -f "$_container" >/dev/null 2>&1 || true',
                "  done",
                "}",
                "trap _cleanup_mlsbench_docker EXIT",
                "trap '_cleanup_mlsbench_docker; exit 143' TERM INT",
                "",
            ])

        def _render_budget_command(cmd: dict, gpu_prefix: str = "") -> str | None:
            if "budget_apptainer_cmd" in cmd:
                cmd_list = _patch_docker_gpus(
                    _add_docker_rm_for_script(list(cmd["budget_apptainer_cmd"]))
                )
                cmd_str = _expand_gpus_sentinel(
                    " ".join(shlex.quote(s) for s in cmd_list)
                )
                return f"{gpu_prefix} {cmd_str}".strip()
            if "budget_local_cmd" not in cmd:
                return None
            cmd_str = " ".join(shlex.quote(s) for s in cmd["budget_local_cmd"])
            env_parts = []
            if gpu_prefix:
                env_parts.append(gpu_prefix)
            env_parts.extend(_local_env_parts(cmd.get("budget_local_env", {})))
            env_str = " ".join(env_parts)
            cwd = shlex.quote(cmd["budget_local_cwd"])
            if env_str:
                return f"(cd {cwd} && {env_str} {cmd_str})"
            return f"(cd {cwd} && {cmd_str})"

        if len(group_cmds) > 1:
            lines.append("# Split scheduler GPU allocation across parallel commands")
            lines.append('IFS="," read -ra _SCHED_GPUS <<< "$CUDA_VISIBLE_DEVICES"')
            lines.append("")

        any_budget = any(
            "budget_apptainer_cmd" in c or "budget_local_cmd" in c
            for c in group_cmds
        )
        if any_budget:
            lines.append("# Budget checks (fail-fast before training)")
            for i, cmd in enumerate(group_cmds):
                gpu_prefix = _physical_gpu_prefix(gpu_assignments[i]) if len(group_cmds) > 1 else ""
                budget_str = _render_budget_command(cmd, gpu_prefix=gpu_prefix)
                if not budget_str:
                    continue
                label = cmd["label"]
                budget_log = f"{out_dir}/{label}_budget_check.out"
                lines.append(f"# --- budget check: {label} ---")
                lines.append(f"{budget_str} > {budget_log} 2>&1")
                lines.append("BUDGET_RC=$?")
                lines.append(f"cat {budget_log}")
                lines.append("if [ $BUDGET_RC -ne 0 ]; then")
                lines.append(f'  echo "{label}:$BUDGET_RC" >> {out_dir}/exit_codes.txt')
                lines.append(f'  echo "[BUDGET CHECK FAILED]" > {out_dir}/{label}.out')
                lines.append(f"  cat {budget_log} >> {out_dir}/{label}.out")
                lines.append("  exit 1")
                lines.append("fi")
            lines.append("")

        if len(group_cmds) == 1:
            cmd = group_cmds[0]
            label = cmd["label"]

            if "apptainer_cmd" in cmd:
                # Container command — run as-is, scheduler provides CUDA_VISIBLE_DEVICES
                apptainer_cmd = _patch_docker_gpus(
                    _add_docker_rm_for_script(list(cmd["apptainer_cmd"]))
                )
                cmd_str = _expand_gpus_sentinel(
                    " ".join(shlex.quote(s) for s in apptainer_cmd)
                )
                lines.append(f"START=$(date +%s)")
                lines.append(f"{cmd_str} > {out_dir}/{label}.out 2>&1")
                lines.append(f'echo "{label}:$?" >> {out_dir}/exit_codes.txt')
                lines.append(f'END=$(date +%s)')
                lines.append(f'echo "{label}:$((END - START))" >> {out_dir}/elapsed.txt')
            else:
                cwd = cmd["local_cwd"]
                env = cmd["local_env"]
                cmd_list = cmd["local_cmd"]

                # Set env vars that differ from base (skip GPU vars — scheduler handles them)
                for k, v in sorted(env.items()):
                    if k in _gpu_env_keys:
                        continue
                    if base_env.get(k) != v:
                        lines.append(f"export {k}={shlex.quote(str(v))}")

                # CUDA_VISIBLE_DEVICES is inherited from the scheduler environment
                lines.append(f"cd {shlex.quote(cwd)}")
                lines.append("")
                cmd_str = " ".join(shlex.quote(s) for s in cmd_list)
                lines.append(f"START=$(date +%s)")
                lines.append(f"{cmd_str} > {out_dir}/{label}.out 2>&1")
                lines.append(f'echo "{label}:$?" >> {out_dir}/exit_codes.txt')
                lines.append(f'END=$(date +%s)')
                lines.append(f'echo "{label}:$((END - START))" >> {out_dir}/elapsed.txt')
        else:
            # Multiple parallel commands — remap logical GPU indices to physical
            # ones from the scheduler's CUDA_VISIBLE_DEVICES allocation.
            for i, (cmd, logical_gpus) in enumerate(zip(group_cmds, gpu_assignments)):
                label = cmd["label"]

                # Remap logical indices to physical: "0,1" → "${_SCHED_GPUS[0]},${_SCHED_GPUS[1]}"
                gpu_prefix = _physical_gpu_prefix(logical_gpus)

                if "apptainer_cmd" in cmd:
                    # Container command (apptainer or docker)
                    apptainer_cmd = _patch_docker_gpus(
                        _add_docker_rm_for_script(list(cmd["apptainer_cmd"]))
                    )
                    cmd_str = _expand_gpus_sentinel(
                        " ".join(shlex.quote(s) for s in apptainer_cmd)
                    )

                    # Build GPU env prefix for remapped devices
                    lines.append(f"START_{i}=$(date +%s)")
                    if gpu_prefix:
                        lines.append(
                            f"({gpu_prefix} {cmd_str}) "
                            f"> {out_dir}/{label}.out 2>&1 &"
                        )
                    else:
                        lines.append(
                            f"({cmd_str}) "
                            f"> {out_dir}/{label}.out 2>&1 &"
                        )
                else:
                    cwd = cmd["local_cwd"]
                    env = cmd["local_env"]
                    cmd_list = cmd["local_cmd"]

                    # Build env string for this command (skip GPU vars)
                    env_parts = []
                    for k, v in sorted(env.items()):
                        if k in _gpu_env_keys:
                            continue
                        if base_env.get(k) != v:
                            env_parts.append(f"{k}={shlex.quote(str(v))}")
                    if gpu_prefix:
                        env_parts.append(gpu_prefix)
                    env_str = " ".join(env_parts)

                    cmd_str = " ".join(shlex.quote(s) for s in cmd_list)
                    lines.append(f"START_{i}=$(date +%s)")
                    lines.append(
                        f"(cd {shlex.quote(cwd)} && {env_str} {cmd_str}) "
                        f"> {out_dir}/{label}.out 2>&1 &"
                    )

                lines.append(f"PID_{i}=$!")
                lines.append("")

            for i, cmd in enumerate(group_cmds):
                label = cmd["label"]
                lines.append(f'wait $PID_{i}; echo "{label}:$?" >> {out_dir}/exit_codes.txt')
                lines.append(f'END_{i}=$(date +%s)')
                lines.append(f'echo "{label}:$((END_{i} - START_{i}))" >> {out_dir}/elapsed.txt')

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def submit_group(
        self,
        group_cmds: list[dict],
        job_name: str,
        out_dir: Path,
    ) -> str:
        """Submit a script job to the local scheduler. Returns the job ID (as string)."""
        from mlsbench.scheduler import submit_script_job

        out_dir.mkdir(parents=True, exist_ok=True)
        script = self._build_script(group_cmds, out_dir)

        script_path = out_dir / "run.sh"
        script_path.write_text(script)
        print(f"[local-scheduler] Script written: {script_path}")

        # Compute total GPUs needed — CPU-only groups need 0
        needs_gpu = any(
            float(cmd.get("compute", 1) or 1) > 0 and cmd.get("use_cuda", True)
            for cmd in group_cmds
        )
        if needs_gpu:
            all_gpu_ids = set()
            gpu_assignments = self._assign_gpus(group_cmds)
            for assignment in gpu_assignments:
                for g in assignment.split(","):
                    if g:
                        all_gpu_ids.add(int(g))
            total_gpus = max(1, len(all_gpu_ids))
        else:
            total_gpus = 0

        # Compute timeout from max time across commands (HH:MM:SS -> seconds)
        # Add 20% buffer to avoid killing jobs that are just slow to finish.
        timeout_secs = 0
        for cmd in group_cmds:
            t = cmd.get("time", "")
            if t:
                parts = t.split(":")
                try:
                    secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    timeout_secs = max(timeout_secs, secs)
                except (ValueError, IndexError):
                    pass
        if timeout_secs > 0:
            timeout_secs = int(timeout_secs * 1.2)

        job_id = submit_script_job(
            script_path=str(script_path),
            gpus_needed=total_gpus,
            job_name=job_name,
            timeout_secs=timeout_secs,
        )
        gpu_label = f"{total_gpus} GPU" if total_gpus else "CPU-only"
        timeout_label = f", timeout {timeout_secs}s" if timeout_secs else ""
        print(f"[local-scheduler] Submitted job {job_id}: {job_name} ({gpu_label}{timeout_label})")
        return str(job_id)

    # ------------------------------------------------------------------
    # Job monitoring
    # ------------------------------------------------------------------

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 5,
        timeout: int | None = None,
    ) -> str:
        """Poll the scheduler until the job reaches a terminal state.

        When no explicit *timeout* is given, we derive one from the job's own
        ``timeout_secs`` (the wall-time limit the scheduler enforces) plus a
        generous buffer for queue wait time.  The old hardcoded 2-hour default
        was far too short on busy machines where jobs can sit in the queue for
        hours before a GPU becomes available, causing the agent to give up and
        report "[output file not found]" even though the job would eventually
        succeed.
        """
        from mlsbench.scheduler import wait_for_job as _wait

        if timeout:
            effective_timeout = float(timeout)
        else:
            # Query the job's own timeout from the scheduler queue so we can
            # wait long enough for both queue time and execution time.
            job_timeout = 0
            try:
                from mlsbench.scheduler import _load_queue
                for j in _load_queue():
                    if j.job_id == int(job_id):
                        job_timeout = getattr(j, "timeout_secs", 0) or 0
                        break
            except Exception:
                pass
            # Wait for the job's execution timeout plus up to 24 hours of
            # queue time.  This matches SLURM executor behaviour where
            # wait_for_job has no timeout by default.
            effective_timeout = max(job_timeout, 7200) + 86400

        state = _wait(int(job_id), poll_interval=poll_interval, timeout=effective_timeout)

        # Map scheduler states to SLURM-like states for compatibility
        state_map = {
            "completed": "COMPLETED",
            "failed": "FAILED",
            "cancelled": "CANCELLED",
            "timeout": "TIMEOUT",
        }
        return state_map.get(state, state.upper())

    # ------------------------------------------------------------------
    # Output reading (same as SlurmExecutor)
    # ------------------------------------------------------------------

    def read_outputs(self, out_dir: Path, labels: list[str]) -> dict[str, str]:
        """Read per-label output files from the output directory."""
        outputs = {}
        for label in labels:
            out_file = out_dir / f"{label}.out"
            if out_file.exists():
                outputs[label] = out_file.read_text()
            else:
                outputs[label] = f"[output file not found: {out_file}]"
        return outputs

    def read_exit_codes(self, out_dir: Path) -> dict[str, int]:
        """Read exit codes from the exit_codes.txt file."""
        exit_codes = {}
        exit_file = out_dir / "exit_codes.txt"
        if exit_file.exists():
            for line in exit_file.read_text().strip().split("\n"):
                if ":" in line:
                    label, code = line.rsplit(":", 1)
                    try:
                        exit_codes[label.strip()] = int(code.strip())
                    except ValueError:
                        pass
        return exit_codes

    def read_elapsed(self, out_dir: Path) -> dict[str, int]:
        """Read per-command elapsed seconds from elapsed.txt."""
        elapsed = {}
        elapsed_file = out_dir / "elapsed.txt"
        if elapsed_file.exists():
            for line in elapsed_file.read_text().strip().split("\n"):
                if ":" in line:
                    label, secs = line.rsplit(":", 1)
                    try:
                        elapsed[label.strip()] = int(secs.strip())
                    except ValueError:
                        pass
        return elapsed
