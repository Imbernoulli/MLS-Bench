#!/bin/bash
# GPU environment preflight for the IsaacGym legs of this task.
#
# Why this exists
# ---------------
# IsaacGym Preview 4 ships a `libPhysXGpu_64.so` whose newest native SASS is
# sm_86, plus a compute_86 PTX payload. On a Hopper GPU (sm_90) there is no
# matching cubin, so the CUDA driver has to JIT-compile that PTX when the sim is
# created. That JIT lives in the *user-space* driver (libcuda +
# libnvidia-ptxjitcompiler), and on 5xx user-space branches older than 570 it can
# segfault instead of emitting sm_90 code — which is what MLS-Bench issue #47
# reported, and what a second site independently confirmed. The kernel driver
# does not have to be touched: a >= 570 user-space CUDA forward-compatibility
# stack inside the container is enough, and a host on 535/550 keeps its driver.
#
# So before the run we:
#   1. print one greppable line describing GPU / driver / PhysX device code,
#   2. transparently activate a bundled >= 570 forward-compat stack when the
#      user-space driver is too old for the GPU it is driving, and
#   3. when we cannot fix it, say exactly what to do — instead of letting the run
#      die later in a bare `Segmentation fault` inside create_sim.
#
# This file is *sourced*, not executed, so the LD_LIBRARY_PATH it exports reaches
# the python child. Everything here is best-effort: it must never abort the run
# (callers use `set -e`), and a healthy node pays one short probe.
#
# Knobs:
#   MLSBENCH_SKIP_GPU_PREFLIGHT=1            skip entirely
#   MLSBENCH_CUDA_COMPAT_DIR=<dir>           forward-compat dir to try first
#   MLSBENCH_GPU_PREFLIGHT_MIN_DRIVER=<maj>  override the 570 threshold

_mls_pf_min_driver="${MLSBENCH_GPU_PREFLIGHT_MIN_DRIVER:-570}"
_mls_pf_physx_so="${ISAACGYM_PHYSX_SO:-/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64/libPhysXGpu_64.so}"

_mls_pf_python() {
    command -v python3 2>/dev/null || command -v python 2>/dev/null
}

# Ask the driver itself, via ctypes, two things at once:
#   - the realpath of the libcuda.so.1 this process actually loads
#   - the compute capability of the first visible device
# Prints "<libcuda-realpath>|<compute-cap>"; either field may be empty. A
# non-empty compute cap also proves cuInit() succeeded, so this doubles as the
# "is this driver stack usable" check after we swap LD_LIBRARY_PATH.
_mls_pf_probe() {
    local py="$1"
    [ -n "$py" ] || return 0
    "$py" - <<'PY' 2>/dev/null
import ctypes, os

path = ""
cc = ""
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError:
    lib = None
if lib is not None:
    try:
        for line in open("/proc/self/maps"):
            candidate = line.rstrip("\n").rsplit(" ", 1)[-1]
            if "/libcuda.so." in candidate:
                path = os.path.realpath(candidate)
                break
    except OSError:
        pass
    try:
        if lib.cuInit(0) == 0:
            dev = ctypes.c_int()
            if lib.cuDeviceGet(ctypes.byref(dev), 0) == 0:
                major, minor = ctypes.c_int(), ctypes.c_int()
                # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}
                rc_major = lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)
                rc_minor = lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)
                if rc_major == 0 and rc_minor == 0:
                    cc = "%d.%d" % (major.value, minor.value)
    except Exception:
        pass
print("%s|%s" % (path, cc))
PY
}

# ".../libcuda.so.570.211.01" -> "570.211.01"; anything unversioned -> ""
_mls_pf_driver_version() {
    local base="${1##*/}"
    case "$base" in
        libcuda.so.[0-9]*.[0-9]*) echo "${base#libcuda.so.}" ;;
        *) echo "" ;;
    esac
}

# Which device code does this PhysX binary actually carry? On Hopper the answer
# is the whole story: no sm_90 cubin and no PTX means nothing can rescue the run,
# which is the second failure mode seen in #47 (community IsaacGym images ship a
# stripped libPhysXGpu with its PTX payload pruned out).
_mls_pf_physx_inventory() {
    local cc="$1" so="$_mls_pf_physx_so" sass ptx want
    [ -f "$so" ] || return 0
    command -v cuobjdump >/dev/null 2>&1 || return 0
    sass="$(timeout 60 cuobjdump --list-elf "$so" 2>/dev/null | grep -o 'sm_[0-9]\+' | sort -u | paste -sd, -)"
    ptx="$(timeout 60 cuobjdump --list-ptx "$so" 2>/dev/null | grep -c '\.ptx')"
    echo "GPU_ENV_PREFLIGHT physx sass=${sass:-none} ptx_payloads=${ptx:-0} so=$so"
    want="sm_${cc%%.*}${cc#*.}"
    if [ "${ptx:-0}" -eq 0 ] && ! printf '%s' ",${sass}," | grep -q ",${want},"; then
        cat >&2 <<EOF
GPU_ENV_PREFLIGHT ERROR: $so carries neither a ${want} cubin nor any PTX, so it
cannot run on this GPU by any driver path. This is a pruned/stripped IsaacGym
copy, not a driver problem — restore the stock Preview 4 binary (the one
vendor/data_scripts/humanoid-gym/prepare_isaacgym.py downloads) and re-run.
EOF
    fi
}

# Prepend the first usable >= min_driver forward-compat directory to
# LD_LIBRARY_PATH, verifying by re-probing that the new libcuda is the one that
# actually loads *and* that it initialises (forward compat is a data-center-GPU
# feature; on other hardware it loads but cuInit fails, so we must roll back).
_mls_pf_activate_compat() {
    local py="$1" from="$2" dir saved cand_ver cand_major probe now
    saved="${LD_LIBRARY_PATH:-}"
    for dir in "${MLSBENCH_CUDA_COMPAT_DIR:-}" \
               /opt/mlsbench/cuda-compat \
               /usr/local/cuda-12.9/compat \
               /usr/local/cuda-12.8/compat; do
        [ -n "$dir" ] || continue
        [ -e "$dir/libcuda.so.1" ] || continue
        cand_ver="$(_mls_pf_driver_version "$(readlink -f "$dir/libcuda.so.1")")"
        cand_major="${cand_ver%%.*}"
        case "$cand_major" in ''|*[!0-9]*) continue ;; esac
        [ "$cand_major" -ge "$_mls_pf_min_driver" ] || continue

        export LD_LIBRARY_PATH="$dir${saved:+:$saved}"
        probe="$(_mls_pf_probe "$py")"
        now="$(_mls_pf_driver_version "${probe%%|*}")"
        if [ "$now" = "$cand_ver" ] && [ -n "${probe#*|}" ]; then
            echo "GPU_ENV_PREFLIGHT remediation=cuda-compat dir=$dir user_driver=${from:-unknown}->$now"
            return 0
        fi
        echo "GPU_ENV_PREFLIGHT compat candidate $dir rejected (loaded='${now:-none}', cuInit=${probe#*|})" >&2
        if [ -n "$saved" ]; then export LD_LIBRARY_PATH="$saved"; else unset LD_LIBRARY_PATH; fi
    done
    return 1
}

_mls_pf_report_unfixable() {
    local cc="$1" udrv="$2" sm="sm_${1%%.*}${1#*.}"
    cat >&2 <<EOF
========================================================================
GPU_ENV_PREFLIGHT WARNING: this container drives a ${sm} GPU with a
user-space CUDA driver of ${udrv:-unknown}, older than ${_mls_pf_min_driver}.x.

IsaacGym Preview 4's libPhysXGpu_64.so carries no ${sm} cubin, so PhysX has
to JIT-compile its compute_86 PTX at create_sim. User-space driver branches
older than ${_mls_pf_min_driver} are known to segfault inside that JIT
(MLS-Bench issue #47): it surfaces as a bare "Segmentation fault" with no
python traceback, and the run then scores 0.

Any one of these fixes it:
  * run on a host whose NVIDIA driver is >= ${_mls_pf_min_driver}; or
  * keep the host driver where it is and upgrade only the container's
    user-space driver, by mounting NVIDIA's cuda-compat package (>= 570,
    e.g. cuda-compat-12-8) at /opt/mlsbench/cuda-compat:
        docker run -v /path/to/compat:/opt/mlsbench/cuda-compat ...
    or by pointing MLSBENCH_CUDA_COMPAT_DIR at wherever you unpacked it.
    This preflight then activates it automatically; or
  * re-pull the MLS-Bench humanoid-gym image, which bundles that directory,
    so no mount is needed.

Continuing anyway — the run may still work, or may segfault in create_sim.
========================================================================
EOF
}

_mls_gpu_env_preflight() {
    [ "${MLSBENCH_SKIP_GPU_PREFLIGHT:-0}" = "1" ] && return 0

    local py probe libcuda cc cc_major kdrv udrv udrv_major
    py="$(_mls_pf_python)"
    probe="$(_mls_pf_probe "$py")"
    libcuda="${probe%%|*}"
    cc="${probe#*|}"
    kdrv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
    udrv="$(_mls_pf_driver_version "$libcuda")"
    cc_major="${cc%%.*}"
    udrv_major="${udrv%%.*}"

    echo "GPU_ENV_PREFLIGHT compute_cap=${cc:-unknown} kernel_driver=${kdrv:-unknown} user_driver=${udrv:-unknown} libcuda=${libcuda:-not-loaded}"

    # Everything below only matters on Hopper and newer, where PhysX has no
    # native cubin and the PTX JIT is on the critical path.
    case "$cc_major" in ''|*[!0-9]*) return 0 ;; esac
    [ "$cc_major" -ge 9 ] || return 0

    _mls_pf_physx_inventory "$cc"

    case "$udrv_major" in ''|*[!0-9]*) return 0 ;; esac
    [ "$udrv_major" -ge "$_mls_pf_min_driver" ] && return 0

    _mls_pf_activate_compat "$py" "$udrv" || _mls_pf_report_unfixable "$cc" "$udrv"
}

_mls_gpu_env_preflight || true
