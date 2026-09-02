"""Harbor environments used by the MLS-Bench Daytona run.

The rendered MLS-Bench tasks contain a small Docker Compose override for every
GPU task.  The override is useful with local Docker (it asks Docker to expose
the requested NVIDIA devices), but it is *not* a real multi-container task.
Harbor's Daytona provider quite correctly runs real Compose definitions through
Docker-in-Docker and, in recent releases, rejects GPU+Compose because the
inner container cannot see the Daytona GPU.

``DaytonaEnvironment`` below recognises this adapter-specific, GPU-only
override and runs the task as a direct Daytona GPU sandbox instead.  CPU tasks
and genuine Compose tasks continue to use Harbor's normal provider behaviour.
The small ``Resources`` compatibility shim fills the GPU field on Harbor
0.6.x, whose Daytona implementation omitted it; on newer Harbor releases the
provider already builds the resource object correctly and the shim is a no-op.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml

from harbor.environments.capabilities import EnvironmentCapabilities

try:  # Daytona >=0.205 exposes explicit GPU type selection.
    from daytona import GpuType as _DaytonaGpuType
except ImportError:  # pragma: no cover - only exercised with old SDKs
    _DaytonaGpuType = None


# Harbor 0.6.x keeps the provider in a module; newer Harbor releases moved it
# into the ``harbor.environments.daytona`` package.  Import the implementation
# module first and resolve private strategy classes from there.  Some releases
# re-export those classes from the package while others do not, so importing
# them directly from ``harbor.environments.daytona`` is not portable.
try:  # Harbor >= 0.7/current
    _DAYTONA_IMPL = importlib.import_module("harbor.environments.daytona.environment")
except (ImportError, ModuleNotFoundError):  # Harbor 0.6.x
    _DAYTONA_IMPL = importlib.import_module("harbor.environments.daytona")

try:
    _HarborDaytonaEnvironment = _DAYTONA_IMPL.DaytonaEnvironment
    _DaytonaDirect = _DAYTONA_IMPL._DaytonaDirect
    _DaytonaDinD = _DAYTONA_IMPL._DaytonaDinD
except AttributeError as exc:  # pragma: no cover - provider-version diagnostic
    # A few intermediate releases put the classes in the package's
    # ``__init__`` while retaining an ``environment`` helper module.
    _package = importlib.import_module("harbor.environments.daytona")
    try:
        _HarborDaytonaEnvironment = _package.DaytonaEnvironment
        _DaytonaDirect = _package._DaytonaDirect
        _DaytonaDinD = _package._DaytonaDinD
        _DAYTONA_IMPL = _package
    except AttributeError:
        raise ImportError(
            "Installed Harbor Daytona provider does not expose its strategy classes"
        ) from exc

# Expose the manager for diagnostics/tests and for callers that need to reset
# the provider singleton between isolated jobs.  Resolve it from the package
# as a fallback for releases that keep it out of ``environment.py``.
try:
    DaytonaClientManager = _DAYTONA_IMPL.DaytonaClientManager
except AttributeError:  # pragma: no cover - provider-version diagnostic
    DaytonaClientManager = importlib.import_module(
        "harbor.environments.daytona"
    ).DaytonaClientManager


_GPU_CONTEXT: contextvars.ContextVar[int] = contextvars.ContextVar(
    "mlsbench_daytona_gpu_count", default=0
)


def _h200_gpu_count_from_rendered_task(environment_dir: Path, fallback: int) -> int:
    """Return the GPU reservation implied by the task's native ``h200`` blocks.

    MLS-Bench keeps H200 command/resource overrides in each native task's
    ``config.json``.  Harbor normally renders the H100 baseline into
    ``task.toml`` before the provider sees it, so a Daytona H200 request would
    otherwise reserve the baseline number of cards even though the verifier
    correctly selects the smaller H200 command.  Read the rendered copy of
    that *existing* config and derive the peak reservation; never invent batch,
    tensor-parallel, or other training parameters here.

    If the task has no H200 metadata (or the metadata is malformed), preserve
    the declared baseline reservation.  This keeps non-H200 tasks byte-for-byte
    compatible with the normal Harbor path.
    """
    config_path = environment_dir.parent / "tests" / "meta" / "config.json"
    try:
        config = json.loads(config_path.read_text())
        entries = list(config.get("test_cmds", []) or [])
        if not entries:
            return fallback
        # Do not reinterpret a task that has no native H200 profile.  Its
        # declared reservation is already the source of truth, even if a
        # hand-authored task.toml happens to differ from the computed peak.
        if not any(isinstance(entry.get("h200"), dict) for entry in entries):
            return fallback
        seeds = config.get("seeds") or [42]
        n_seeds = 1 if isinstance(seeds, int) else max(1, len(seeds))
        grouped: dict[object, list[dict]] = {}
        auto_group = 10000
        for entry in entries:
            group = entry.get("group")
            if group is None:
                group = auto_group
                auto_group += 1
            grouped.setdefault(group, []).append(entry)

        def compute(entry: dict) -> float:
            value = entry.get("compute", 1)
            override = entry.get("h200")
            if isinstance(override, dict) and "compute" in override:
                value = override["compute"]
            try:
                return float(value or 1)
            except (TypeError, ValueError):
                return 1.0

        def fractional_bins(values: list[float]) -> int:
            bins: list[float] = []
            for value in sorted(values, reverse=True):
                for index, remaining in enumerate(bins):
                    if remaining >= value:
                        bins[index] = remaining - value
                        break
                else:
                    bins.append(1.0 - value)
            return len(bins)

        peak = 0
        for group_entries in grouped.values():
            computes = [compute(entry) for entry in group_entries for _ in range(n_seeds)]
            whole = sum(max(1, math.ceil(value)) for value in computes if value >= 1.0)
            fractional = fractional_bins([value for value in computes if 0.0 < value < 1.0])
            peak = max(peak, whole + fractional)
        return max(1, min(fallback, peak)) if peak else fallback
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return fallback
# Strategies may be re-exported by a package even though their globals (and
# therefore ``Resources``) live in a different implementation module.
_RESOURCES_IMPL = importlib.import_module(_DaytonaDirect.__module__)
_ORIGINAL_RESOURCES = getattr(_RESOURCES_IMPL, "Resources", None)


if _ORIGINAL_RESOURCES is not None:

    class _GPUAwareResources:
        """Add the task GPU count when running Harbor 0.6.x strategies.

        Harbor's old ``_DaytonaDirect`` and ``_DaytonaDinD`` construct
        ``Resources(cpu=..., memory=..., disk=...)`` themselves.  A context
        variable keeps this compatibility layer safe when Harbor starts trials
        concurrently; newer providers either pass ``gpu`` already or do not
        construct this object in the strategy.
        """

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            gpu_count = _GPU_CONTEXT.get()
            if gpu_count > 0 and "gpu" not in kwargs:
                kwargs["gpu"] = gpu_count
            return _ORIGINAL_RESOURCES(*args, **kwargs)

    # The strategy resolves Resources from its implementation module globals.
    # Install the proxy once; calls made outside our context are byte-for-byte
    # equivalent to the original class.
    if not getattr(_RESOURCES_IMPL, "_MLSBENCH_GPU_RESOURCES", False):
        _RESOURCES_IMPL.Resources = _GPUAwareResources
        _RESOURCES_IMPL._MLSBENCH_GPU_RESOURCES = True


class _GpuAwareStrategyMixin:
    async def start(self, force_build: bool):  # type: ignore[no-untyped-def]
        token = _GPU_CONTEXT.set(max(0, int(self._env.task_env_config.gpus or 0)))
        try:
            return await super().start(force_build)  # type: ignore[misc]
        finally:
            _GPU_CONTEXT.reset(token)


class _GpuAwareDirect(_GpuAwareStrategyMixin, _DaytonaDirect):
    pass


class _GpuAwareDinD(_GpuAwareStrategyMixin, _DaytonaDinD):
    pass


def is_gpu_reservation_only_compose(
    environment_dir: Path,
    *,
    gpu_count: int,
) -> bool:
    """Return whether ``docker-compose.yaml`` only reserves NVIDIA devices.

    This deliberately accepts a narrow schema.  A compose file with another
    service, a build/image override, or any unknown key remains a genuine
    Compose task and is delegated to Harbor (which will report the provider's
    normal GPU+Compose limitation rather than silently dropping a service).
    """

    if gpu_count <= 0:
        return False
    return _is_gpu_reservation_only_compose(
        environment_dir,
        expected_gpu_count=gpu_count,
    )


def _is_gpu_reservation_only_compose(
    environment_dir: Path,
    *,
    expected_gpu_count: int | None,
) -> bool:
    """Validate the reservation-only schema, optionally checking its count.

    ``expected_gpu_count=None`` is used only when Harbor's explicit
    ``override_gpus`` changes the effective allocation.  In that case the
    compose overlay's original count is intentionally different from the
    Daytona request, but the overlay is still safe to ignore because it has no
    service semantics beyond a device reservation.
    """

    path = environment_dir / "docker-compose.yaml"
    if not path.is_file():
        return False
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(document, dict):
        return False
    services = document.get("services")
    if not isinstance(services, dict) or set(services) != {"main"}:
        return False
    main = services.get("main")
    if not isinstance(main, dict) or set(main) != {"deploy"}:
        return False
    deploy = main.get("deploy")
    if not isinstance(deploy, dict) or set(deploy) != {"resources"}:
        return False
    resources = deploy.get("resources")
    if not isinstance(resources, dict) or set(resources) != {"reservations"}:
        return False
    reservations = resources.get("reservations")
    if not isinstance(reservations, dict) or set(reservations) != {"devices"}:
        return False
    devices = reservations.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        return False
    device = devices[0]
    if not isinstance(device, dict):
        return False
    if set(device) != {"driver", "count", "capabilities"}:
        return False
    count = device.get("count")
    if (
        device.get("driver") != "nvidia"
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        return False
    if expected_gpu_count is not None and count != expected_gpu_count:
        return False
    capabilities = device.get("capabilities")
    return isinstance(capabilities, list) and "gpu" in capabilities


class DaytonaEnvironment(_HarborDaytonaEnvironment):
    """Daytona provider with MLS-Bench's GPU-only Compose compatibility."""

    def _int_kwarg(self, name: str) -> int:
        """Return an integer environment kwarg (``--ek name=value``) or 0."""
        value = self._kwargs.get(name)
        if value in (None, "", False):
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            raise ValueError(f"Daytona kwarg {name}={value!r} must be an integer")

    def _spot_requested(self) -> bool:
        """Return whether this run requested a Daytona spot sandbox.

        Harbor forwards ``--ek spot=true`` as an environment kwarg.  The
        upstream Harbor Daytona provider currently does not forward that
        kwarg to Daytona's ``CreateSandbox*Params`` objects, so the adapter
        applies it at the final sandbox-creation boundary below.  Daytona
        rejects ``spot`` for CPU sandboxes; callers therefore only enable it
        when the effective task resource request includes one or more GPUs.
        """
        value = self._kwargs.get("spot", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _gpu_type_requested(self):  # type: ignore[no-untyped-def]
        """Resolve the Daytona GPU type kwarg (default ``H100``).

        MLS-Bench images are built and validated for Hopper (H100/H200,
        sm_90).  Without an explicit type Daytona may place a sandbox on an
        RTX PRO 6000 Blackwell (sm_120), where the pinned CUDA wheels have no
        kernels, so every GPU request defaults to H100.  H200 must be selected
        explicitly and is only meaningful for tasks with a native ``h200``
        profile.
        """
        value = self._kwargs.get("gpu_type") or "H100"
        if _DaytonaGpuType is None:
            return None
        name = str(value).strip().upper().replace("-", "_")
        # Accept the SDK enum names and common CLI spellings while rejecting
        # arbitrary values before they reach the provider API.
        aliases = {
            "RTX_PRO_6000_BLACKWELL": "RTX_PRO_6000",
            "RTX_PRO_6000_BLACKWELL_SERVER": "RTX_PRO_6000",
            "RTX_4090": "RTX_4090",
            "4090": "RTX_4090",
            "RTX_5090": "RTX_5090",
            "5090": "RTX_5090",
        }
        name = aliases.get(name, name)
        try:
            return _DaytonaGpuType[name]
        except (KeyError, TypeError):
            raise ValueError(f"Unsupported Daytona gpu_type={value!r}")

    async def _create_sandbox(self, params, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        """Forward optional spot/GPU-type flags to Daytona's SDK request.

        This hook is shared by Harbor's direct and DinD strategies, including
        the adapter's GPU-only Compose compatibility strategy. Keeping the
        change here avoids duplicating the provider's resource construction and
        preserves the task's declared GPU count verbatim. GPU type selection
        also works for on-demand requests; spot remains opt-in.
        """
        gpu_count = int(self.task_env_config.gpus or 0)
        gpu_type = self._gpu_type_requested()
        resources = getattr(params, "resources", None)
        env_vars = dict(getattr(params, "env_vars", None) or {})
        if gpu_count > 0:
            if self._spot_requested() and hasattr(params, "spot"):
                params.spot = True
            if resources is not None and gpu_type is not None:
                resources.gpu_type = gpu_type
            # The verifier runs inside the Daytona sandbox (and, for GPU-only
            # overlays, the direct strategy). Propagate the selected device
            # type explicitly; the verifier selects a task's native ``h200``
            # profile only from this variable.
            if gpu_type is not None:
                env_vars["MLSBENCH_GPU_TYPE"] = getattr(
                    gpu_type, "value", str(gpu_type)
                )
            # task.toml resources are sized for local Docker, where memory is
            # rarely the binding limit.  Daytona enforces them as hard cgroup
            # limits, so GPU sandboxes may request a larger floor (for example
            # loading a 7B checkpoint needs more than 16 GiB of host RAM).
            if resources is not None:
                min_memory = self._int_kwarg("gpu_memory_gb")
                if min_memory and (resources.memory or 0) < min_memory:
                    resources.memory = min_memory
                min_cpus = self._int_kwarg("gpu_cpus")
                if min_cpus and (resources.cpu or 0) < min_cpus:
                    resources.cpu = min_cpus
            if gpu_count > 1:
                # NCCL >= 2.19 allocates communication buffers through the
                # cuMem driver API by default; inside Daytona's GPU sandbox
                # that path fails with ``cudaErrorIllegalState`` during the
                # first collective.  Fall back to the classic allocator.
                env_vars.setdefault("NCCL_CUMEM_ENABLE", "0")
        # The sandbox cgroup grants only ``resources.cpu`` cores, but the
        # container still reports every host core, so OpenMP/MKL/PyTorch
        # would otherwise start hundreds of threads and thrash the quota.
        cpu_quota = getattr(resources, "cpu", None) if resources is not None else None
        if cpu_quota:
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ):
                env_vars.setdefault(name, str(int(cpu_quota)))
        # ``test_cmds[].time`` budgets were calibrated on MLS-Bench's native
        # hosts; a remote sandbox with a small CPU quota may need longer.
        # ``--ek eval_time_scale=2`` scales the verifier's wall-clock budget.
        time_scale = self._kwargs.get("eval_time_scale")
        if time_scale not in (None, ""):
            env_vars["MLSBENCH_EVAL_TIME_SCALE"] = str(time_scale)
        # Free-form sandbox environment for provider-specific tuning (YAML
        # ``kwargs.sandbox_env``; not expressible through ``--ek``).
        extra_env = self._kwargs.get("sandbox_env") or {}
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                env_vars[str(key)] = str(value)
        if hasattr(params, "env_vars"):
            params.env_vars = env_vars
        # The smoke runner sets a per-invocation label so its optional orphan
        # cleanup can never delete another runner's organization sandbox.
        # Keep this best-effort for older Daytona SDKs that lack ``labels``.
        run_id = os.environ.get("MLSBENCH_DAYTONA_RUN_ID")
        if run_id and hasattr(params, "labels"):
            labels = dict(getattr(params, "labels", None) or {})
            labels["mlsbench-run-id"] = run_id
            params.labels = labels
        # A sandbox occasionally lands on a runner whose in-sandbox toolbox
        # never answers ("Failed to create session" / "Failed to execute
        # command" for the same image that works elsewhere).  Recreate it a
        # few times before giving up; the same image on another placement is
        # fine.
        attempts = max(1, self._int_kwarg("toolbox_ready_retries") or 3)
        # Daytona places a request on whichever runner has the most free
        # capacity, so deleting an unusable sandbox and retrying tends to land
        # on the very same broken runner.  Keep the unusable sandboxes alive
        # ("decoys") until a usable placement exists, so the scheduler must
        # pick another runner; they are deleted before returning.  Set
        # ``--ek toolbox_hold_bad_placements=0`` to release them immediately
        # (cheaper on quota, but the retry may be pointless).
        hold_value = self._kwargs.get("toolbox_hold_bad_placements", True)
        if isinstance(hold_value, str):
            hold_value = hold_value.strip().lower() in {"1", "true", "yes", "on"}
        decoys: list[Any] = []
        last_error: Exception | None = None

        async def _delete_quietly(sandbox: Any) -> None:
            try:
                await sandbox.delete()
            except Exception as delete_exc:  # noqa: BLE001
                self.logger.warning(
                    "Could not delete unusable sandbox %s: %s",
                    getattr(sandbox, "id", "?"),
                    str(delete_exc)[:200],
                )

        try:
            for attempt in range(1, attempts + 1):
                result = await super()._create_sandbox(params, *args, **kwargs)
                try:
                    await self._wait_for_sandbox_toolbox()
                    return result
                except RuntimeError as exc:
                    last_error = exc
                    sandbox = getattr(self, "_sandbox", None)
                    self._sandbox = None
                    self.logger.warning(
                        "Daytona sandbox %s unusable (attempt %d/%d): %s",
                        getattr(sandbox, "id", "?"),
                        attempt,
                        attempts,
                        str(exc)[:300],
                    )
                    if sandbox is None:
                        continue
                    if hold_value and attempt < attempts:
                        decoys.append(sandbox)
                    else:
                        await _delete_quietly(sandbox)
            raise RuntimeError(
                f"Daytona sandbox toolbox never became ready after {attempts} "
                f"placement attempts: {last_error}"
            )
        finally:
            for sandbox in decoys:
                await _delete_quietly(sandbox)

    async def _wait_for_sandbox_toolbox(self) -> None:
        """Block until the sandbox accepts exec sessions.

        ``daytona.create()`` returns once the sandbox is STARTED, but the
        in-sandbox toolbox can still be unreachable for a while, and Harbor's
        very first ``create_session`` would then fail with
        ``DaytonaBadGatewayError`` and abort the trial.  Poll a trivial command
        first; ``--ek toolbox_ready_timeout_sec`` bounds the wait per placement.
        """
        sandbox = getattr(self, "_sandbox", None)
        process = getattr(sandbox, "process", None)
        if process is None:
            return
        timeout = float(self._kwargs.get("toolbox_ready_timeout_sec") or 180)
        deadline = asyncio.get_running_loop().time() + timeout
        attempt = 0
        while True:
            attempt += 1
            try:
                await process.exec("true", timeout=30)
                return
            except Exception as exc:  # noqa: BLE001 - provider transport errors
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(
                        f"Daytona sandbox toolbox not ready after {timeout:g}s "
                        f"({attempt} attempts): {exc}"
                    ) from exc
                self.logger.warning(
                    "Daytona sandbox not ready for exec yet (attempt %d): %s",
                    attempt,
                    str(exc)[:200],
                )
                await asyncio.sleep(10)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Keep all capabilities supplied by the installed Harbor version and
        # advertise GPU allocation (the 0.6.x provider forgot this flag).
        base = super().capabilities
        if hasattr(base, "model_copy"):
            return base.model_copy(update={"gpus": True})
        values = base.model_dump() if hasattr(base, "model_dump") else base.dict()
        values["gpus"] = True
        return EnvironmentCapabilities(**values)

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: Any,
        task_env_config: Any,
        *args: Any,
        **kwargs: Any,
    ):
        declared_gpus = int(getattr(task_env_config, "gpus", 0) or 0)
        override_gpus = kwargs.get("override_gpus")
        if override_gpus is not None:
            # An explicit Harbor override always wins, including zero.
            requested_gpus = int(override_gpus)
        else:
            requested_gpus = declared_gpus
            # The native task config is the sole source of H200 training
            # parameters.  Use its existing ``h200.compute`` values to avoid
            # reserving idle baseline cards on Daytona; no task config is
            # modified and H100 requests retain the declared reservation.
            gpu_type = kwargs.get("gpu_type")
            gpu_type_name = str(getattr(gpu_type, "name", gpu_type or "")).upper()
            if "H200" in gpu_type_name and declared_gpus > 0:
                requested_gpus = _h200_gpu_count_from_rendered_task(
                    environment_dir,
                    declared_gpus,
                )
        extra_compose = kwargs.get("extra_docker_compose")
        gpu_only = (
            not extra_compose
            and declared_gpus > 0
            and (
                is_gpu_reservation_only_compose(
                    environment_dir,
                    gpu_count=declared_gpus,
                )
                or (
                    override_gpus is not None
                    and _is_gpu_reservation_only_compose(
                        environment_dir,
                        expected_gpu_count=None,
                    )
                )
            )
        )

        # A Daytona spot GPU is useful only when the task itself runs in the
        # GPU sandbox. A genuine Compose definition is executed through
        # Docker-in-Docker; its inner containers cannot see the outer
        # Daytona GPU, so accepting spot here would spend scarce capacity on
        # a run that can never satisfy the task's GPU contract.
        spot_value = kwargs.get("spot", False)
        if isinstance(spot_value, str):
            spot_value = spot_value.strip().lower() in {"1", "true", "yes", "on"}
        if bool(spot_value) and declared_gpus > 0 and not gpu_only:
            raise ValueError(
                "Daytona spot GPU requires a direct GPU sandbox; "
                "genuine multi-container Compose tasks are unsupported"
            )

        # Let Harbor run its complete normal initialisation with a zero-GPU
        # clone, then switch this adapter-only override to the direct strategy
        # and restore the effective (possibly overridden) GPU count before any
        # lifecycle call.  The original model is not mutated, which is
        # important when Harbor reuses task config data across trials.
        effective_config = (
            copy.deepcopy(task_env_config) if gpu_only else task_env_config
        )
        if gpu_only:
            effective_config.gpus = 0

        # Newer Harbor releases validate ``override_gpus`` while they still
        # see the compose definition and reject GPU+Compose before this
        # adapter can switch to the direct strategy.  The effective config is
        # already zero-GPU during parent initialization, so omit the override
        # kwarg there and restore the requested count below.
        parent_kwargs = dict(kwargs)
        if gpu_only:
            parent_kwargs.pop("override_gpus", None)

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=effective_config,
            *args,
            **parent_kwargs,
        )

        if gpu_only:
            self.task_env_config.gpus = requested_gpus
            # Newer Harbor caches effective GPU values only through the task
            # model property, so restoring the model is sufficient.
            self._compose_mode = False
            self._strategy = _GpuAwareDirect(self)
        elif not getattr(self, "_compose_mode", False) and requested_gpus > 0:
            # Covers direct GPU tasks authored outside this adapter on Harbor
            # 0.6.x, whose strategy omitted ``Resources.gpu``.
            self._strategy = _GpuAwareDirect(self)


__all__ = [
    "DaytonaEnvironment",
    "DaytonaClientManager",
    "is_gpu_reservation_only_compose",
]
