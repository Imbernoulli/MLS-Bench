from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[2] / "harbor" / "scripts" / "daytona_smoke.py"
    spec = importlib.util.spec_from_file_location("daytona_smoke", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _task(root: Path, name: str, package: str, gpus: int = 0):
    task = root / f"mls-bench__{name}"
    task.mkdir()
    (task / "task.toml").write_text(
        "\n".join(
            [
                "[metadata]",
                f'mls_bench_task_id = "{name}"',
                f'mls_bench_package = "{package}"',
                "[environment]",
                f"gpus = {gpus}",
            ]
        )
    )
    return task


def test_selection_excludes_api_and_picks_gpu_representative(tmp_path: Path):
    module = _module()
    _task(tmp_path, "cpu-one", "pkg-a", 0)
    _task(tmp_path, "gpu-one", "pkg-a", 2)
    _task(tmp_path, "agent-tool-reasoning", "stabletoolbench", 0)

    records = module.discover_tasks(tmp_path)
    assert len(records) == 3
    assert [
        r.task_id
        for r in module.select_tasks(records, scope="task", include_api=False)
    ] == ["cpu-one", "gpu-one"]
    representatives = module.select_tasks(
        records, scope="environment", include_api=False
    )
    assert [(r.package, r.task_id, r.gpus) for r in representatives] == [
        ("pkg-a", "gpu-one", 2)
    ]


def test_build_command_isolated_and_disable_verification():
    module = _module()
    record = module.TaskRecord("demo", "pkg", 1, Path("/tmp/demo"), False)
    command = module.build_command(
        record,
        harbor_cmd="harbor",
        import_path="harbor_env:DaytonaEnvironment",
        jobs_dir=Path("/tmp/jobs"),
        agent="nop",
        verify=False,
        force_build=True,
        delete=True,
        overrides={"--override-gpus": 1, "--override-cpus": None},
    )
    assert command[0:2] == ["harbor", "run"]
    assert (
        "--n-concurrent" in command
        and command[command.index("--n-concurrent") + 1] == "1"
    )
    assert "--disable-verification" in command
    assert command[-1] == "1"  # override-gpus value

    cpu = module.TaskRecord("cpu", "pkg", 0, Path("/tmp/cpu"), False)
    cpu_command = module.build_command(
        cpu,
        harbor_cmd="harbor",
        import_path="harbor_env:DaytonaEnvironment",
        jobs_dir=Path("/tmp/jobs"),
        agent="nop",
        verify=False,
        force_build=True,
        delete=True,
        overrides={"--override-gpus": 1},
    )
    assert "--override-gpus" not in cpu_command


def test_build_command_can_request_gpu_type_without_spot():
    module = _module()
    record = module.TaskRecord("demo", "pkg", 1, Path("/tmp/demo"), False)
    command = module.build_command(
        record,
        harbor_cmd="harbor",
        import_path="harbor_env:DaytonaEnvironment",
        jobs_dir=Path("/tmp/jobs"),
        agent="nop",
        verify=False,
        force_build=True,
        delete=True,
        gpu_type="H100",
        overrides={},
    )
    assert command[command.index("--ek") : command.index("--ek") + 2] == [
        "--ek",
        "gpu_type=H100",
    ]
    assert "spot=true" not in " ".join(command)


def test_daytona_compat_leaves_cleandiffuser_image_untouched(tmp_path: Path):
    """H100/H200 run the published CleanDiffuser base as-is; no torch upgrade."""
    from mls_bench.daytona_compat import apply_daytona_compatibility

    dockerfile = tmp_path / "Dockerfile"
    original = (
        "FROM bohanlyu2022/mlsbench-harbor-cleandiffuser:latest\n"
        "COPY _scaffold/ /workspace/\n"
    )
    dockerfile.write_text(original)
    tests = tmp_path / "tests"
    (tests / "eval/scripts").mkdir(parents=True)
    assert not apply_daytona_compatibility(
        "robo-diffusion-guidance", dockerfile=dockerfile, tests_dir=tests
    )
    assert dockerfile.read_text() == original


def test_daytona_compat_leaves_unrelated_native_task_layer_untouched(tmp_path: Path):
    from mls_bench.daytona_compat import apply_daytona_compatibility

    dockerfile = tmp_path / "Dockerfile"
    original = "FROM example:latest\nCOPY _scaffold/ /workspace/\n"
    dockerfile.write_text(original)
    tests = tmp_path / "tests"
    tests.mkdir()
    assert not apply_daytona_compatibility(
        "ts-imputation", dockerfile=dockerfile, tests_dir=tests
    )
    assert dockerfile.read_text() == original


def test_daytona_compat_patches_only_rendered_verl_verifier(tmp_path: Path):
    from mls_bench.daytona_compat import apply_daytona_compatibility

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM bohanlyu2022/mlsbench-harbor-verl:latest\n"
        "COPY _scaffold/ /workspace/\n"
    )
    tests = tmp_path / "tests"
    (tests / "eval/scripts").mkdir(parents=True)
    (tests / "meta/scripts").mkdir(parents=True)
    native = (
        "python3 -m verl.trainer.main_ppo \\\n"
        "    data.shuffle=False \\\n"
        "    actor_rollout_ref.rollout.enforce_eager=True \\\n"
        "    actor_rollout_ref.rollout.free_cache_engine=False \\\n"
        "    actor_rollout_ref.ref.fsdp_config.param_offload=False \\\n"
        "    trainer.total_epochs=1\n"
    )
    (tests / "eval/scripts/train.sh").write_text(native)
    (tests / "meta/scripts/train.sh").write_text(native)
    for _ in range(2):  # idempotent on re-render
        assert apply_daytona_compatibility(
            "llm-rl-importance-sampling", dockerfile=dockerfile, tests_dir=tests
        )
    rendered = dockerfile.read_text()
    assert "mlsbench-harbor-verl:verl-fixes-20260901" in rendered
    assert rendered.count('test "${found}" -eq 1;') == 1
    for p in (tests / "eval/scripts/train.sh", tests / "meta/scripts/train.sh"):
        text = p.read_text()
        assert "free_cache_engine=False" not in text
        assert text.count("data.dataloader_num_workers=0") == 1
        assert text.count("rollout.agent.num_workers=${AGENT_NUM_WORKERS:-1}") == 1
        assert text.count("reward.num_workers=${REWARD_NUM_WORKERS:-1}") == 1
        # Arguments are inserted after their anchors, keeping the command valid.
        assert text.index("data.shuffle=False") < text.index("dataloader_num_workers=0")


def test_summarize_result_reads_harbor_exceptions(tmp_path: Path):
    module = _module()
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "stats": {
                    "evals": {
                        "nop__adhoc": {
                            "exception_stats": {"DaytonaAuthenticationError": ["x"]}
                        }
                    }
                }
            }
        )
    )
    assert module.summarize_result(result, 0) == (
        "error",
        "DaytonaAuthenticationError",
    )


def test_summarize_result_trusts_terminal_success_over_wrapper_rc(tmp_path: Path):
    """Daytona websocket cleanup may make Harbor's wrapper return non-zero."""
    module = _module()
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "stats": {
                    "n_total_trials": 1,
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "evals": {"nop__adhoc": {"exception_stats": {}}},
                }
            }
        )
    )
    assert module.summarize_result(result, 130) == ("passed", "")


def test_summarize_result_accepts_legacy_root_trial_counts(tmp_path: Path):
    module = _module()
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "evals": {"nop__adhoc": {"exception_stats": {}}},
                },
            }
        )
    )
    assert module.summarize_result(result, 1) == ("passed", "")


def test_harbor_subprocess_timeout_returns_124_and_reaps(monkeypatch):
    module = _module()

    class Hanging:
        pid = 4242
        returncode = None

        def wait(self, timeout=None):
            if timeout is not None and self.returncode is None:
                raise module.subprocess.TimeoutExpired("harbor", timeout)
            self.returncode = -15
            return self.returncode

    child = Hanging()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: child)
    killed = []
    monkeypatch.setattr(
        module, "_terminate_process_group", lambda pid, **kw: killed.append(pid)
    )

    assert (
        module._run_harbor_subprocess(
            ["harbor", "run"],
            cwd=Path("."),
            env={},
            timeout_sec=1,
        )
        == 124
    )
    assert killed == [4242]
    assert not module._ACTIVE_CHILDREN


def test_orphan_cleanup_is_label_scoped(monkeypatch):
    module = _module()
    observed = {}

    class Query:
        def __init__(self, **kwargs):
            observed["query"] = kwargs

    class Sandbox:
        def __init__(self, sandbox_id):
            self.id = sandbox_id
            self.deleted = False

        async def delete(self):
            self.deleted = True
            observed.setdefault("deleted", []).append(self.id)

    owned = Sandbox("owned")

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def list(self, *, query=None):
            observed["client_query"] = query

            async def rows():
                yield owned

            return rows()

        async def close(self):
            pass

    fake_daytona = types.SimpleNamespace(
        AsyncDaytona=Client,
        DaytonaConfig=lambda **kwargs: kwargs,
        ListSandboxesQuery=Query,
    )
    monkeypatch.setitem(sys.modules, "daytona", fake_daytona)

    import asyncio

    assert asyncio.run(
        module._daytona_sandboxes(request_delete=True, run_id="runner-a")
    ) == ["owned"]
    assert observed["query"] == {"labels": {"mlsbench-run-id": "runner-a"}}
    assert observed["deleted"] == ["owned"]


def test_wait_for_daytona_quiescence_zero_timeout():
    module = _module()
    assert module.wait_for_daytona_quiescence(0) == []


def test_real_mls_bench_selection_counts():
    module = _module()
    tasks_dir = Path(__file__).resolve().parents[2] / "harbor" / "tasks"
    records = module.discover_tasks(tasks_dir)
    assert len(records) == 140
    assert len(module.select_tasks(records, scope="task", include_api=False)) == 138
    assert (
        len(module.select_tasks(records, scope="environment", include_api=False)) == 63
    )
