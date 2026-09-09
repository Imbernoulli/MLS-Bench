from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def _module():
    # Keep the test independent of whether the caller installed the adapter as
    # a package or is using the documented PYTHONPATH=src workflow.
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import mls_bench.harbor_env as module

    return module


def _compose(count: int = 2) -> str:
    return yaml.safe_dump(
        {
            "services": {
                "main": {
                    "deploy": {
                        "resources": {
                            "reservations": {
                                "devices": [
                                    {
                                        "driver": "nvidia",
                                        "count": count,
                                        "capabilities": ["gpu"],
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        },
        sort_keys=False,
    )


def test_gpu_reservation_only_compose_is_strict(tmp_path: Path):
    module = _module()
    env = tmp_path / "environment"
    env.mkdir()
    (env / "docker-compose.yaml").write_text(_compose(2))

    assert module.is_gpu_reservation_only_compose(env, gpu_count=2)
    assert not module.is_gpu_reservation_only_compose(env, gpu_count=1)

    (env / "docker-compose.yaml").write_text(
        _compose(2).replace("services:\n", "services:\n  sidecar:\n    image: redis:7\n")
    )
    assert not module.is_gpu_reservation_only_compose(env, gpu_count=2)


def test_gpu_compose_uses_direct_strategy_and_restores_resources(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    config = EnvironmentConfig(
        cpus=2,
        memory_mb=4096,
        storage_mb=10240,
        gpus=2,
    )
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="gpu-task",
        session_id="gpu-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
    )

    assert env.task_env_config.gpus == 2
    assert env._compose_mode is False
    assert env._strategy.__class__.__name__ == "_GpuAwareDirect"
    assert env.capabilities.gpus is True

    captured = []

    async def fake_get_instance():
        return SimpleNamespace(get_client=lambda: asyncio.sleep(0, result=object()))

    async def fake_create_sandbox(*, params):
        captured.append(params)
        env._sandbox = object()

    async def fake_exec(*args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(module.DaytonaClientManager, "get_instance", fake_get_instance)
        monkeypatch.setattr(env, "_configure_daytona_client", fake_exec)
        monkeypatch.setattr(env, "_create_sandbox", fake_create_sandbox)
        monkeypatch.setattr(env, "_sandbox_exec", fake_exec)
        asyncio.run(env.start(force_build=True))
    finally:
        monkeypatch.undo()

    assert len(captured) == 1
    assert captured[0].resources.gpu == 2


def test_gpu_override_keeps_reservation_only_compose_on_direct_path(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="gpu-override-task",
        session_id="gpu-override-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        override_gpus=1,
    )

    assert env.task_env_config.gpus == 1
    assert env._compose_mode is False
    assert env._strategy.__class__.__name__ == "_GpuAwareDirect"

    # An explicit zero-GPU override still uses the direct Dockerfile path; it
    # must not fall back to Daytona DinD merely because the overlay's original
    # reservation requested devices.
    config_zero = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env_zero = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="gpu-override-zero-task",
        session_id="gpu-override-zero-task.1",
        trial_paths=trial_paths,
        task_env_config=config_zero,
        override_gpus=0,
    )
    assert env_zero.task_env_config.gpus == 0
    assert env_zero._compose_mode is False
    assert env_zero._strategy.__class__.__name__ == "_GpuAwareDirect"


def test_spot_h200_forwards_type_and_verifier_environment(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(1))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(
        cpus=2, memory_mb=4096, storage_mb=10240, gpus=1
    )
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="spot-h200-task",
        session_id="spot-h200-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
        gpu_type="H200",
    )

    captured = []

    async def fake_get_instance():
        return SimpleNamespace(get_client=lambda: asyncio.sleep(0, result=object()))

    async def fake_parent_create_sandbox(self, params, *args, **kwargs):
        captured.append(params)
        self._sandbox = object()

    async def fake_exec(*args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(module.DaytonaClientManager, "get_instance", fake_get_instance)
        monkeypatch.setattr(env, "_configure_daytona_client", fake_exec)
        monkeypatch.setattr(
            module._HarborDaytonaEnvironment,
            "_create_sandbox",
            fake_parent_create_sandbox,
        )
        monkeypatch.setattr(env, "_sandbox_exec", fake_exec)
        # Call the adapter hook directly so the assertions exercise the
        # spot/GPU-type/env-var mutation rather than replacing the hook.
        from daytona import CreateSandboxFromImageParams, Image, Resources

        params = CreateSandboxFromImageParams(
            image=Image.base("ubuntu:22.04"),
            resources=Resources(cpu=2, memory=4, disk=10, gpu=1),
        )
        asyncio.run(env._create_sandbox(params))
    finally:
        monkeypatch.undo()

    assert len(captured) == 1
    assert captured[0].spot is True
    assert captured[0].resources.gpu == 1
    assert captured[0].resources.gpu_type.name == "H200"
    assert captured[0].env_vars["MLSBENCH_GPU_TYPE"] == "H200"


def test_h200_uses_native_override_compute_for_daytona_reservation(tmp_path: Path):
    """H200 capacity comes from the rendered task's existing config metadata."""
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    meta = tmp_path / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(
        json.dumps(
            {
                "test_cmds": [
                    {
                        "cmd": "scripts/train.sh",
                        "compute": 2,
                        "group": 1,
                        "h200": {
                            "compute": 1,
                            "env": {"TP_SIZE": "1"},
                        },
                    }
                ],
                "seeds": [42],
            }
        )
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="native-h200-resource-task",
        session_id="native-h200-resource-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
        gpu_type="H200",
    )

    # The verifier still reads the same h200 block; only Daytona's resource
    # request is reduced from the H100 baseline of two cards to one.
    assert env.task_env_config.gpus == 1


def test_h100_keeps_native_baseline_reservation_with_h200_metadata(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    meta = tmp_path / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(
        json.dumps(
            {
                "test_cmds": [
                    {
                        "cmd": "scripts/train.sh",
                        "compute": 2,
                        "h200": {"compute": 1},
                    }
                ]
            }
        )
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="native-h100-resource-task",
        session_id="native-h100-resource-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
        gpu_type="H100",
    )
    assert env.task_env_config.gpus == 2


def test_h200_does_not_scale_tasks_without_native_profile(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    meta = tmp_path / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(
        json.dumps({"test_cmds": [{"cmd": "scripts/train.sh", "compute": 2}]})
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="unprofiled-h200-task",
        session_id="unprofiled-h200-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
        gpu_type="H200",
    )
    assert env.task_env_config.gpus == 2


def test_sandbox_environment_and_resource_floor(tmp_path: Path):
    """Thread caps, NCCL fallback, RAM/CPU floors and time scale reach Daytona."""
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(2))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=4, memory_mb=16384, storage_mb=61440, gpus=2)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="floor-task",
        session_id="floor-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        gpu_memory_gb="64",
        gpu_cpus="16",
    )
    # No explicit gpu_type and no spot: still Hopper, never Daytona's default.
    assert env._gpu_type_requested().name == "H100"

    captured = []

    async def fake_parent_create_sandbox(self, params, *args, **kwargs):
        captured.append(params)
        self._sandbox = object()

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            module._HarborDaytonaEnvironment,
            "_create_sandbox",
            fake_parent_create_sandbox,
        )
        from daytona import CreateSandboxFromImageParams, Image, Resources

        params = CreateSandboxFromImageParams(
            image=Image.base("ubuntu:22.04"),
            resources=Resources(cpu=4, memory=16, disk=60, gpu=2),
        )
        asyncio.run(env._create_sandbox(params))
    finally:
        monkeypatch.undo()

    resources = captured[0].resources
    assert (resources.cpu, resources.memory, resources.disk, resources.gpu) == (16, 64, 60, 2)
    assert resources.gpu_type.name == "H100"
    env_vars = captured[0].env_vars
    assert env_vars["MLSBENCH_GPU_TYPE"] == "H100"
    assert env_vars["OMP_NUM_THREADS"] == "16"
    assert env_vars["MKL_NUM_THREADS"] == "16"
    assert env_vars["NCCL_CUMEM_ENABLE"] == "0"
    assert captured[0].spot is not True


def test_snapshot_salt_appends_noop_layer_and_decoys_default_off(tmp_path: Path):
    """``snapshot_salt`` forces a fresh Daytona snapshot; decoys stay opt-in."""
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(1))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=4, memory_mb=16384, storage_mb=61440, gpus=1)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="salt-task",
        session_id="salt-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        snapshot_salt="rebuild-1",
    )
    captured = []

    async def fake_parent_create_sandbox(self, params, *args, **kwargs):
        captured.append(params)
        self._sandbox = object()

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            module._HarborDaytonaEnvironment,
            "_create_sandbox",
            fake_parent_create_sandbox,
        )
        from daytona import CreateSandboxFromImageParams, Image, Resources

        params = CreateSandboxFromImageParams(
            image=Image.base("ubuntu:22.04"),
            resources=Resources(cpu=4, memory=16, disk=60, gpu=1),
        )
        asyncio.run(env._create_sandbox(params))
    finally:
        monkeypatch.undo()

    assert "mlsbench snapshot salt rebuild-1" in captured[0].image.dockerfile()
    assert env._kwargs.get("toolbox_hold_bad_placements", False) is False


def test_spot_defaults_to_h100_without_explicit_gpu_type(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(1))
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=1)
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="spot-default-task",
        session_id="spot-default-task.1",
        trial_paths=trial_paths,
        task_env_config=config,
        spot=True,
    )

    assert env._gpu_type_requested().name == "H100"


def test_spot_rejects_genuine_gpu_compose(tmp_path: Path):
    module = _module()
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "main": {"deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}}}},
                    "sidecar": {"image": "redis:7"},
                }
            }
        )
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=2, memory_mb=4096, storage_mb=10240, gpus=1)
    with pytest.raises(ValueError, match="genuine multi-container Compose"):
        module.DaytonaEnvironment(
            environment_dir=env_dir,
            environment_name="genuine-compose-spot",
            session_id="genuine-compose-spot.1",
            trial_paths=trial_paths,
            task_env_config=config,
            spot=True,
        )


def test_daytona_run_configs_select_custom_environment():
    for path, expected_dataset in (
        (Path("harbor/run-daytona.yaml"), "tasks-daytona"),
        (Path("harbor_adapter/run-daytona.yaml"), "datasets/mls-bench"),
    ):
        data = yaml.safe_load(path.read_text())
        assert data["environment"]["import_path"].endswith(
            "DaytonaEnvironment"
        )
        assert data["datasets"][0]["path"] == expected_dataset
        assert set(data["datasets"][0]["exclude_task_names"]) == {
            "mls-bench__agent-tool-reasoning",
            "mls-bench__mas-topology",
        }


def test_rendered_bundles_carry_their_own_cpu_ram_and_thread_settings():
    """Everything a downstream harness needs is in the bundle, not in this repo.

    Three variants ship. ``harbor/tasks-docker`` carries the native calibration
    and the Compose overlay a local run needs; ``harbor/tasks-daytona`` carries
    Daytona's CPU-sandbox shape for CPU-only tasks and no compose file at all,
    because stock Harbor's Daytona provider refuses GPU tasks that have one;
    ``harbor/tasks-modal`` is the native calibration without a compose file
    and with CPUs capped at Modal's 64 per sandbox.
    A consumer copies exactly one of them and never imports
    ``mls_bench.harbor_env``.
    """
    module = _module()
    thread_keys = (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    for variant, cpu_shape in (("tasks-docker", (8, 32 * 1024)), ("tasks-daytona", (4, 8 * 1024)), ("tasks-modal", (8, 32 * 1024))):
        cpu_cap = 64 if variant == "tasks-modal" else 10**9  # Modal: 0.125-64 cores per sandbox
        tasks = sorted(p for p in (Path("harbor") / variant).iterdir() if (p / "task.toml").is_file())
        assert len(tasks) == 140, variant
        verl = gpu = cpu = 0
        for task_dir in tasks:
            toml = (task_dir / "task.toml").read_text()
            cpus = int(re.search(r"(?m)^cpus = (\d+)$", toml).group(1))
            memory_mb = int(re.search(r"(?m)^memory_mb = (\d+)$", toml).group(1))
            gpus = int(re.search(r"(?m)^gpus = (\d+)$", toml).group(1))
            package = re.search(r'(?m)^mls_bench_package = "([^"]*)"$', toml).group(1)
            gpu_types = re.search(r'(?m)^gpu_types = (.*)$', toml)
            assert re.search(r"(?m)^build_timeout_sec = 7200$", toml), (variant, task_dir.name, "111 GB bases need > 30 min to pull on Modal")

            if gpus == 0:
                cpu += 1
                assert cpus == cpu_shape[0] and memory_mb >= cpu_shape[1], (variant, task_dir.name)
                assert gpu_types is None, (variant, task_dir.name)
            elif package == "verl":
                verl += 1
                assert (cpus, memory_mb) == (min(max(16, 12 * gpus), cpu_cap), 128 * 1024), (variant, task_dir.name)
            else:
                gpu += 1
                assert (cpus, memory_mb) == (min(max(16, 12 * gpus), cpu_cap), 64 * 1024), (variant, task_dir.name)
            if gpus > 0:
                # Newer Harbor's Daytona provider maps this to a placement
                # constraint; the images' CUDA wheels have no Blackwell kernels.
                assert gpu_types is not None and gpu_types.group(1) == '["h100"]', (variant, task_dir.name)

            dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
            for key in thread_keys:
                assert f"{key}={cpus}" in dockerfile, (variant, task_dir.name, key)
            expected_nccl = gpus > 1
            assert ("NCCL_CUMEM_ENABLE=0" in dockerfile) is expected_nccl, (variant, task_dir.name)
            assert ("NCCL_NVLS_ENABLE=0" in dockerfile) is expected_nccl, (variant, task_dir.name)

            compose = task_dir / "environment" / "docker-compose.yaml"
            if variant == "tasks-docker":
                overlay = yaml.safe_load(compose.read_text())
                assert overlay["services"]["main"]["shm_size"] == "16gb", task_dir.name
                if gpus > 0:
                    assert module.is_gpu_reservation_only_compose(task_dir / "environment", gpu_count=gpus), task_dir.name
                else:
                    assert module.is_shm_only_compose(task_dir / "environment"), task_dir.name
            else:
                assert not compose.exists(), (variant, task_dir.name)
        assert (verl, gpu, cpu) == (4, 112, 24), variant
def test_gpu_cpu_and_memory_kwargs_stay_opt_in(tmp_path: Path):
    """The floors are overrides now: unset leaves `task.toml`'s numbers alone."""
    module = _module()
    from daytona import CreateSandboxFromImageParams, Image, Resources

    for kwargs, expected in (({}, (16, 64)), ({"gpu_cpus": "32", "gpu_memory_gb": "96"}, (32, 96))):
        task_dir = tmp_path / f"task-{len(kwargs)}"
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True)
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        (env_dir / "docker-compose.yaml").write_text(_compose(2))
        trial_paths = TrialPaths(task_dir / "trial")
        trial_paths.mkdir()
        env = module.DaytonaEnvironment(
            environment_dir=env_dir,
            environment_name=f"floor-{len(kwargs)}",
            session_id=f"floor-{len(kwargs)}.1",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(
                cpus=16, memory_mb=65536, storage_mb=61440, gpus=2
            ),
            **kwargs,
        )
        captured = []

        async def fake_parent_create_sandbox(self, params, *args, **kwargs):
            captured.append(params)
            self._sandbox = object()

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                module._HarborDaytonaEnvironment,
                "_create_sandbox",
                fake_parent_create_sandbox,
            )
            params = CreateSandboxFromImageParams(
                image=Image.base("ubuntu:22.04"),
                resources=Resources(cpu=16, memory=64, disk=60, gpu=2),
            )
            asyncio.run(env._create_sandbox(params))
        finally:
            monkeypatch.undo()
        assert (captured[0].resources.cpu, captured[0].resources.memory) == expected
        # Thread pools follow whatever CPU count the sandbox actually got.
        assert captured[0].env_vars["OMP_NUM_THREADS"] == str(expected[0])


def test_lite_config_lists_exactly_the_readme_lite_tasks():
    """harbor/run-daytona-lite.yaml is the README's 30-task table, verbatim.

    Users select MLS-Bench-Lite with ``-c run-daytona-lite.yaml``; this keeps
    the file, the README table and the rendered dataset from drifting apart.
    """
    import re

    readme = Path("README.md").read_text()
    section = re.search(r"## MLS-Bench-Lite(.*?)</details>", readme, re.S).group(1)
    table = {f"mls-bench__{tid}" for tid in re.findall(r"\]\(tasks/([a-z0-9\-]+)\)", section)}
    assert len(table) == 30

    lite = yaml.safe_load(Path("harbor/run-daytona-lite.yaml").read_text())
    full = yaml.safe_load(Path("harbor/run-daytona.yaml").read_text())
    (dataset,) = lite["datasets"]
    assert dataset["path"] == "tasks-daytona"
    names = dataset["task_names"]
    assert len(names) == len(set(names)) == 30
    assert set(names) == table
    for name in names:
        assert (Path("harbor/tasks-daytona") / name / "task.toml").is_file(), name
        assert (Path("harbor/tasks-docker") / name / "task.toml").is_file(), name
        assert (Path("harbor/tasks-modal") / name / "task.toml").is_file(), name
    # Same provider settings as the full-dataset config, so the two cannot drift.
    assert lite["environment"] == full["environment"]
    assert lite["n_concurrent_trials"] == full["n_concurrent_trials"]

    # The Modal pair mirrors the Daytona pair, task list included.
    modal_lite = yaml.safe_load(Path("harbor/run-modal-lite.yaml").read_text())
    modal_full = yaml.safe_load(Path("harbor/run-modal.yaml").read_text())
    (modal_dataset,) = modal_lite["datasets"]
    assert modal_dataset["path"] == "tasks-modal"
    assert modal_dataset["task_names"] == names
    assert modal_full["datasets"][0]["path"] == "tasks-modal"
    assert set(modal_full["datasets"][0]["exclude_task_names"]) == set(full["datasets"][0]["exclude_task_names"])
    assert modal_lite["environment"] == modal_full["environment"]
    assert modal_full["environment"]["import_path"] == "harbor_env:ModalEnvironment"


def test_modal_environment_maps_gpu_type_and_exports_the_h200_profile(tmp_path: Path):
    """``harbor/tasks-modal`` needs nothing from this module; the class only
    turns ``--ek gpu_type=H200`` into Modal's ``h200:<n>`` request (with the
    task's native h200 GPU count when it has such a block) and puts
    ``MLSBENCH_GPU_TYPE`` in the persistent env, like the other two classes."""
    pytest.importorskip("harbor.environments.modal")
    module = _module()
    from harbor.models.task.config import EnvironmentConfig

    task_dir = tmp_path / "task"
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    trial_paths = TrialPaths(task_dir / "trial")
    trial_paths.mkdir()

    def make(**kwargs):
        return module.ModalEnvironment(
            environment_dir=env_dir,
            environment_name="modal-test",
            session_id="modal-test",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(cpus=16, memory_mb=65536, gpus=1, gpu_types=["h100"]),
            **kwargs,
        )

    stock = make()
    assert stock._gpu_config() == "h100:1"
    assert "MLSBENCH_GPU_TYPE" not in stock._persistent_env

    h200 = make(gpu_type="H200")
    assert h200._gpu_config() == "h200:1"  # no native h200 block here: count unchanged
    assert h200._persistent_env["MLSBENCH_GPU_TYPE"] == "H200"


def _h200_task(tmp_path: Path, name: str, gpus: int = 2) -> Path:
    """A rendered bundle with a two-entry h200 profile in tests/meta/config.json."""
    task_dir = tmp_path / name
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(gpus))
    meta = task_dir / "tests" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(json.dumps({
        "seeds": [42],
        "test_cmds": [
            {"cmd": "scripts/train.sh", "compute": 2,
             "h200": {"compute": 1, "env": {"TP_SIZE": "1", "GPU_MEM_UTIL": "0.5"}}},
            {"cmd": "scripts/eval.sh", "compute": 1},
        ],
    }))
    return env_dir


def test_h200_exports_the_native_profile_env_to_the_sandbox(tmp_path: Path):
    """On Daytona the agent's shell sees the h200 block's env, not only the verifier.

    The native runner applies the block to the agent's own test runs; without
    the env the agent would explore with H100 settings on an H200 reservation.
    H100 requests export nothing but the type.
    """
    module = _module()
    from daytona import CreateSandboxFromImageParams, Image, Resources

    seen = {}
    for gpu_type in ("H200", "H100"):
        env_dir = _h200_task(tmp_path, f"task-{gpu_type}")
        trial_paths = TrialPaths(env_dir.parent / "trial")
        trial_paths.mkdir()
        env = module.DaytonaEnvironment(
            environment_dir=env_dir,
            environment_name=f"profile-{gpu_type}",
            session_id=f"profile-{gpu_type}.1",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(
                cpus=4, memory_mb=16384, storage_mb=61440, gpus=2
            ),
            gpu_type=gpu_type,
        )
        captured = []

        async def fake_parent_create_sandbox(self, params, *args, **kwargs):
            captured.append(params)
            self._sandbox = object()

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                module._HarborDaytonaEnvironment,
                "_create_sandbox",
                fake_parent_create_sandbox,
            )
            params = CreateSandboxFromImageParams(
                image=Image.base("ubuntu:22.04"),
                resources=Resources(cpu=4, memory=16, disk=60, gpu=2),
            )
            asyncio.run(env._create_sandbox(params))
        finally:
            monkeypatch.undo()
        seen[gpu_type] = captured[0].env_vars

    assert seen["H200"]["MLSBENCH_GPU_TYPE"] == "H200"
    assert seen["H200"]["TP_SIZE"] == "1"
    assert seen["H200"]["GPU_MEM_UTIL"] == "0.5"
    assert seen["H100"]["MLSBENCH_GPU_TYPE"] == "H100"
    assert "TP_SIZE" not in seen["H100"]


def test_docker_gpu_environment_takes_the_same_gpu_type_switch(tmp_path: Path):
    """Local Docker: ``--ek gpu_type=H200`` reaches every exec, agent and verifier."""
    module = _module()
    env_dir = _h200_task(tmp_path, "docker-task")
    trial_paths = TrialPaths(env_dir.parent / "trial")
    trial_paths.mkdir()
    config = EnvironmentConfig(cpus=4, memory_mb=16384, storage_mb=61440, gpus=2)

    plain = module.DockerGPUEnvironment(
        environment_dir=env_dir,
        environment_name="docker-plain",
        session_id="docker-plain.1",
        trial_paths=trial_paths,
        task_env_config=config,
    )
    assert plain.capabilities.gpus is True
    assert plain._merge_env(None) is None

    h200 = module.DockerGPUEnvironment(
        environment_dir=env_dir,
        environment_name="docker-h200",
        session_id="docker-h200.1",
        trial_paths=trial_paths,
        task_env_config=config,
        gpu_type="H200",
    )
    merged = h200._merge_env({"PER_EXEC": "1"})
    assert merged["MLSBENCH_GPU_TYPE"] == "H200"
    assert merged["TP_SIZE"] == "1"
    assert merged["PER_EXEC"] == "1"

    # harbor/harbor_env.py re-exports the same class for run.yaml.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "harbor_env_reexport", Path("harbor/harbor_env.py")
    )
    reexport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reexport)
    assert reexport.DockerGPUEnvironment is module.DockerGPUEnvironment


def test_gpu_sandboxes_repair_host_overlay_loader_paths_after_toolbox_ready(tmp_path: Path):
    """vLLM dlopens the libcudart path from /proc/self/maps verbatim; on runners
    that report host overlay paths the adapter aliases ``<prefix>/merged`` to
    ``/`` once the toolbox answers.  CPU sandboxes never run it."""
    module = _module()
    from daytona import CreateSandboxFromImageParams, Image, Resources

    ran = {}
    for gpus in (2, 0):
        env_dir = tmp_path / f"repair-{gpus}" / "environment"
        env_dir.mkdir(parents=True)
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        if gpus:
            (env_dir / "docker-compose.yaml").write_text(_compose(gpus))
        trial_paths = TrialPaths(env_dir.parent / "trial")
        trial_paths.mkdir()
        env = module.DaytonaEnvironment(
            environment_dir=env_dir,
            environment_name=f"repair-{gpus}",
            session_id=f"repair-{gpus}.1",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(
                cpus=4, memory_mb=16384, storage_mb=61440, gpus=gpus
            ),
        )
        commands = []

        async def fake_process_exec(command, timeout=None, **kwargs):
            commands.append(command)
            return SimpleNamespace(exit_code=0, result="aliased /var/lib/docker/overlay2/x/merged -> /")

        async def fake_parent_create_sandbox(self, params, *args, **kwargs):
            self._sandbox = SimpleNamespace(process=SimpleNamespace(exec=fake_process_exec))

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                module._HarborDaytonaEnvironment,
                "_create_sandbox",
                fake_parent_create_sandbox,
            )
            params = CreateSandboxFromImageParams(
                image=Image.base("ubuntu:22.04"),
                resources=Resources(cpu=4, memory=16, disk=60, gpu=gpus),
            )
            asyncio.run(env._create_sandbox(params))
        finally:
            monkeypatch.undo()
        ran[gpus] = commands

    assert ran[2][0] == "true"  # toolbox probe first
    assert any("/proc/self/maps" in c and "merged" in c for c in ran[2])
    assert ran[0] == ["true"]


def test_local_docker_clamps_a_cpu_request_the_host_cannot_grant(tmp_path: Path, monkeypatch, caplog):
    """A small host should run a GPU task slowly, not refuse to start it.

    docker rejects a container whose ``--cpus`` exceeds the host core count
    ("range of CPUs is from 0.01 to N"), and every GPU task declares 16.
    """
    module = _module()
    env_dir = _h200_task(tmp_path, "clamp-task")
    trial_paths = TrialPaths(env_dir.parent / "trial")
    trial_paths.mkdir()

    monkeypatch.setattr(module.os, "cpu_count", lambda: 8)
    config = EnvironmentConfig(cpus=16, memory_mb=65536, storage_mb=61440, gpus=2)
    with caplog.at_level("WARNING"):
        clamped = module.DockerGPUEnvironment(
            environment_dir=env_dir,
            environment_name="clamp",
            session_id="clamp.1",
            trial_paths=trial_paths,
            task_env_config=config,
        )
    assert config.cpus == 8
    assert "clamping" in caplog.text
    # The agent's shell must see the granted count, not the image's ENV.
    assert clamped._merge_env({})["OMP_NUM_THREADS"] == "8"

    monkeypatch.setattr(module.os, "cpu_count", lambda: 384)
    roomy = EnvironmentConfig(cpus=16, memory_mb=65536, storage_mb=61440, gpus=2)
    module.DockerGPUEnvironment(
        environment_dir=env_dir,
        environment_name="roomy",
        session_id="roomy.1",
        trial_paths=trial_paths,
        task_env_config=roomy,
    )
    assert roomy.cpus == 16


def test_daytona_clamps_to_the_org_ceiling_on_refusal(tmp_path: Path):
    """`task.toml` declares the calibrated budget; Daytona names its ceiling.

    "CPU request 96 exceeds maximum allowed per sandbox (16)" is the whole
    contract: clamp to 16, re-pin the thread budget, retry. No org-specific
    number is hard-coded in the adapter.
    """
    module = _module()
    from daytona import CreateSandboxFromImageParams, Image, Resources

    task_dir = tmp_path / "task"
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text(_compose(8))
    trial_paths = TrialPaths(task_dir / "trial")
    trial_paths.mkdir()
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="clamp",
        session_id="clamp.1",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(cpus=96, memory_mb=65536, storage_mb=61440, gpus=8),
    )
    seen: list[tuple[int, int]] = []

    async def fake_parent_create_sandbox(self, params, *args, **kwargs):
        seen.append((params.resources.cpu, params.resources.memory))
        if params.resources.cpu > 16:
            raise RuntimeError(
                "Failed to create sandbox: CPU request 96 exceeds maximum allowed per sandbox (16).\n"
                "Need higher resource limits per-sandbox? Contact us at support@daytona.io"
            )
        if params.resources.memory > 192:
            raise RuntimeError("Failed to create sandbox: Memory request 256GB exceeds maximum allowed per sandbox (192GB).")
        self._sandbox = object()
        self._final_env = dict(params.env_vars)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(module._HarborDaytonaEnvironment, "_create_sandbox", fake_parent_create_sandbox)
        params = CreateSandboxFromImageParams(
            image=Image.base("ubuntu:22.04"),
            resources=Resources(cpu=96, memory=256, disk=60, gpu=8),
        )
        asyncio.run(env._create_sandbox(params))
    finally:
        monkeypatch.undo()
    assert seen == [(96, 256), (16, 256), (16, 192)]
    assert env._final_env["OMP_NUM_THREADS"] == "16"
    assert env._final_env["BLIS_NUM_THREADS"] == "16"


def test_shm_only_overlay_keeps_a_cpu_task_on_the_direct_path(tmp_path: Path):
    """CPU bundles carry a /dev/shm-only overlay for local Docker.

    Harbor's Daytona provider would route any compose file through
    Docker-in-Docker; the adapter must recognise this one as a sizing knob
    with no service in it and keep the direct sandbox.
    """
    module = _module()
    task_dir = tmp_path / "cpu-task"
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (env_dir / "docker-compose.yaml").write_text("services:\n  main:\n    shm_size: 16gb\n")
    assert module.is_shm_only_compose(env_dir)
    assert not module.is_gpu_reservation_only_compose(env_dir, gpu_count=0)
    trial_paths = TrialPaths(task_dir / "trial")
    trial_paths.mkdir()
    env = module.DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="cpu-direct",
        session_id="cpu-direct.1",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(cpus=8, memory_mb=32768, storage_mb=30720, gpus=0),
    )
    assert env._compose_mode is False
    assert isinstance(env._strategy, module._GpuAwareDirect)

    # Anything with a real service stays a genuine Compose task.
    (env_dir / "docker-compose.yaml").write_text(
        "services:\n  main:\n    shm_size: 16gb\n    image: other:latest\n"
    )
    assert not module.is_shm_only_compose(env_dir)


def test_local_docker_writes_the_gpu_and_shm_overlay_per_trial(tmp_path: Path):
    """The bundle is Dockerfile-only; the local environment supplies the rest.

    Stock Harbor's Daytona provider refuses a GPU task that ships a compose
    file, and stock Harbor's docker provider cannot run GPU tasks at all, so
    the NVIDIA reservation and the 16 GB /dev/shm live in a per-trial overlay
    that only this class appends to Harbor's own compose file list.
    """
    module = _module()
    task_dir = tmp_path / "gpu-task"
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    trial_paths = TrialPaths(task_dir / "trial")
    trial_paths.mkdir()
    env = module.DockerGPUEnvironment(
        environment_dir=env_dir,
        environment_name="overlay",
        session_id="overlay.1",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(cpus=36, memory_mb=65536, storage_mb=61440, gpus=3),
    )
    # A bundle overlay (tasks-docker ships one) must not be merged alongside
    # ours: Compose appends `devices` lists, which would reserve twice.
    (env_dir / "docker-compose.yaml").write_text("services:\n  main:\n    shm_size: 16gb\n")
    paths = env._docker_compose_paths
    assert env_dir / "docker-compose.yaml" not in paths
    overlay = paths[-1]
    assert overlay.parent == Path(trial_paths.trial_dir)
    doc = yaml.safe_load(overlay.read_text())
    main = doc["services"]["main"]
    assert main["shm_size"] == "16gb"
    device = main["deploy"]["resources"]["reservations"]["devices"][0]
    assert (device["driver"], device["count"], device["capabilities"]) == ("nvidia", 3, ["gpu"])
    # Harbor's own files come first so the overlay's scalars win.
    assert any(p.name == "docker-compose-base.yaml" for p in paths[:-1])

    cpu_env = module.DockerGPUEnvironment(
        environment_dir=env_dir,
        environment_name="overlay-cpu",
        session_id="overlay-cpu.1",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(cpus=4, memory_mb=8192, storage_mb=10240, gpus=0),
    )
    cpu_main = yaml.safe_load(cpu_env._docker_compose_paths[-1].read_text())["services"]["main"]
    assert cpu_main == {"shm_size": "16gb"}



def test_harbor_env_imports_without_the_renderer_dependencies():
    """`harbor_env.py` is what `run.yaml` / `run-daytona.yaml` / `run-modal.yaml`
    load inside a plain `uv tool install "harbor[...]"` environment, which
    has PyYAML and the provider SDKs but not the renderer's Jinja2/tomli_w.
    The 2026-09-08 README check on Harbor 0.22.0 failed every local trial with
    `ModuleNotFoundError: No module named 'tomli_w'` because the Docker class
    reached into `mls_bench.adapter` for the compose overlay text."""
    import os, subprocess, sys
    code = (
        "import sys\n"
        "sys.modules['tomli_w'] = None\n"
        "sys.modules['jinja2'] = None\n"
        "import mls_bench.harbor_env as m\n"
        "from mls_bench.compose_overlay import compose_overlay_text\n"
        "assert 'shm_size: 16gb' in compose_overlay_text(0)\n"
        "assert 'count: 2' in compose_overlay_text(2)\n"
        "assert 'device_ids' in compose_overlay_text(1, ['3'])\n"
        "m.DockerGPUEnvironment, m.DaytonaEnvironment, m.ModalEnvironment\n"
        "print('ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path("harbor_adapter/src").resolve()))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]
