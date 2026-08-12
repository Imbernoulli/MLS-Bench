#!/bin/bash
# Verifier-only runtime patch for rl-reward-learning's custom_irl.py.
#
# All three seeds of every eval label run concurrently and share
# ${SAVE_PATH}/irl_experts. The original scaffold (a) publishes that
# expert-demo cache non-atomically (np.savez / model.save straight onto the
# final path) and races a bare exists() -> generate -> np.load sequence, so
# a sibling seed can np.load a half-written npz (zipfile.BadZipFile /
# EOFError), zero that seed, and leave a corrupt cache behind; and (b) trains
# the shared expert with whichever seed's RNG state happens to win the race,
# so the shared demonstrations vary with scheduler timing.
#
# The task template was fixed at the source
# (tasks/rl-reward-learning/edits/custom_template.py): temp-file + os.replace
# atomic publish, flock-serialized generate-or-load, corrupt-cache self-heal,
# and deterministic expert generation (fixed env-derived seed with the
# caller's RNG states saved/restored). The scaffold, tests/meta/pristine and
# pristine_manifest.json carry the same fixed bytes, so fresh sessions bake
# the fix and this script is a strict no-op. For a workspace built from a
# PRE-FIX image, this script re-applies exactly that fix at eval time; the
# patched bytes equal the new pristine, so subsequent guard passes stay
# clean.
#
# Anchored on the exact old code blocks, applied all-or-none, and written
# back atomically: if the fixed sections do not match a known pre-fix
# scaffold verbatim (the edit guard forbids the agent from touching them, so
# this only means a future scaffold revision), the file is left byte-for-byte
# untouched. Agent edits live in the editable region (RewardNetwork /
# IRLAlgorithm) and are never modified.
#
# Runs under flock because several eval scripts patch the same file
# concurrently (precedent: mls-bench__cv-dbm-sampler _runtime_patch.sh).

_rl_demo_patch_lock="${RL_DEMO_PATCH_LOCK:-.mlsbench_demo_patch.lock}"
{
if command -v flock >/dev/null 2>&1; then
    flock 9
fi

python3 - <<'PY'
import os
from pathlib import Path

path = Path("custom_irl.py")
if not path.exists():
    print("[runtime-patch] custom_irl.py not found in cwd; nothing to do")
    raise SystemExit(0)

text = path.read_text()

V2_MARKER = "_generate_expert_demos_impl"   # current fixed template
V1_MARKER = "_locked_demo_load"             # intermediate fix (no deterministic seeding)

# ---- pre-fix (gen0) blocks ------------------------------------------------

GEN0_GEN = '''def generate_expert_demos(demo_path, env_id, total_timesteps=2_000_000, n_demos=25000):
    """Train PPO expert and collect demonstrations on GPU."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy as sb3_eval

    os.makedirs(demo_path, exist_ok=True)
    print(f"Training expert for {env_id} ({total_timesteps} steps)...", flush=True)

    train_env = SubprocVecEnv([lambda eid=env_id, i=i: gym.make(eid) for i in range(4)])
    sb3_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO("MlpPolicy", train_env, verbose=0,
                n_steps=2048, batch_size=64, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                max_grad_norm=0.5, device=sb3_device)
    model.learn(total_timesteps=total_timesteps)
    train_env.close()

    eval_env = DummyVecEnv([lambda eid=env_id: gym.make(eid)])
    mean_reward, std_reward = sb3_eval(model, eval_env, n_eval_episodes=20)
    print(f"  Expert {env_id}: {mean_reward:.1f} +/- {std_reward:.1f}", flush=True)
    model.save(os.path.join(demo_path, f"{env_id}_expert"))

    all_obs, all_acts, all_next_obs, all_dones = [], [], [], []
    obs = eval_env.reset()
    for _ in range(n_demos):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = eval_env.step(action)
        all_obs.append(obs[0].copy())
        all_acts.append(action[0].copy())
        all_next_obs.append(next_obs[0].copy())
        all_dones.append(float(done[0]))
        obs = next_obs

    demos = {
        "obs": np.array(all_obs, dtype=np.float32),
        "acts": np.array(all_acts, dtype=np.float32),
        "next_obs": np.array(all_next_obs, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
    }
    np.savez(os.path.join(demo_path, f"{env_id}_demos.npz"), **demos)
    print(f"  Saved {n_demos} transitions for {env_id}", flush=True)
    eval_env.close()
'''

# The v1 intermediate differed from gen0 only in the two publish lines.
V1_GEN = GEN0_GEN.replace(
    '    model.save(os.path.join(demo_path, f"{env_id}_expert"))\n',
    '    _atomic_save_model(model, demo_path, env_id)\n',
).replace(
    '    np.savez(os.path.join(demo_path, f"{env_id}_demos.npz"), **demos)\n',
    '    _atomic_save_npz(demos, os.path.join(demo_path, f"{env_id}_demos.npz"))\n',
)

OLD_LOAD = '''def load_expert_demos(demo_path, env_id, device):
    """Load expert demonstrations, generating them if needed."""
    path = os.path.join(demo_path, f"{env_id}_demos.npz")
    if not os.path.exists(path):
        generate_expert_demos(demo_path, env_id)
    data = np.load(path)
    demos = {
        "obs": torch.tensor(data["obs"], dtype=torch.float32, device=device),
        "acts": torch.tensor(data["acts"], dtype=torch.float32, device=device),
        "next_obs": torch.tensor(data["next_obs"], dtype=torch.float32, device=device),
        "dones": torch.tensor(data["dones"], dtype=torch.float32, device=device),
    }
    print(f"Loaded {len(demos['obs'])} expert transitions from {path}")
    return demos
'''

# ---- fixed (v2) blocks ----------------------------------------------------

NEW_IMPL = '''def _generate_expert_demos_impl(demo_path, env_id, gen_seed, total_timesteps=2_000_000, n_demos=25000):
    """Train the PPO expert under ``gen_seed`` and collect demonstrations."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy as sb3_eval

    os.makedirs(demo_path, exist_ok=True)
    print(f"Training expert for {env_id} ({total_timesteps} steps, seed {gen_seed})...", flush=True)

    train_env = SubprocVecEnv([lambda eid=env_id, i=i: gym.make(eid) for i in range(4)])
    sb3_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO("MlpPolicy", train_env, verbose=0,
                n_steps=2048, batch_size=64, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                max_grad_norm=0.5, device=sb3_device, seed=gen_seed)
    model.learn(total_timesteps=total_timesteps)
    train_env.close()
    eval_env = DummyVecEnv([lambda eid=env_id: gym.make(eid)])
    if hasattr(eval_env, "seed"):
        eval_env.seed(gen_seed)
    mean_reward, std_reward = sb3_eval(model, eval_env, n_eval_episodes=20)
    print(f"  Expert {env_id}: {mean_reward:.1f} +/- {std_reward:.1f}", flush=True)
    _atomic_save_model(model, demo_path, env_id)
    all_obs, all_acts, all_next_obs, all_dones = [], [], [], []
    obs = eval_env.reset()
    for _ in range(n_demos):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = eval_env.step(action)
        all_obs.append(obs[0].copy())
        all_acts.append(action[0].copy())
        all_next_obs.append(next_obs[0].copy())
        all_dones.append(float(done[0]))
        obs = next_obs

    demos = {
        "obs": np.array(all_obs, dtype=np.float32),
        "acts": np.array(all_acts, dtype=np.float32),
        "next_obs": np.array(all_next_obs, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
    }
    _atomic_save_npz(demos, os.path.join(demo_path, f"{env_id}_demos.npz"))
    print(f"  Saved {n_demos} transitions for {env_id}", flush=True)
    eval_env.close()
'''

NEW_LOAD = '''def load_expert_demos(demo_path, env_id, device):
    """Load expert demonstrations, generating them if needed.

    Concurrent runs (e.g. several seeds) share ``demo_path``, so the
    generate-or-load step is serialized with an inter-process file lock, the
    cache is only ever published atomically, and a corrupt cache file is
    regenerated under the lock (see the concurrency helpers further down).
    """
    path = os.path.join(demo_path, f"{env_id}_demos.npz")
    data = _locked_demo_load(demo_path, env_id, path)
    demos = {k: torch.tensor(data[k], dtype=torch.float32, device=device)
             for k in ("obs", "acts", "next_obs", "dones")}
    print(f"Loaded {len(demos['obs'])} expert transitions from {path}")
    return demos
'''

BANNER = (
    "# =====================================================================\n"
    "# FIXED: Main training loop\n"
    "# =====================================================================\n"
)

BASE_HELPERS = '''# =====================================================================
# FIXED: Demo-cache concurrency helpers (atomic publish + file lock)
# =====================================================================
def _atomic_save_model(model, demo_path, env_id):
    """Save the SB3 expert atomically (temp file in the same dir + os.replace)."""
    tmp = os.path.join(demo_path, f".{env_id}_expert.tmp-{os.getpid()}.zip")
    model.save(tmp)
    os.replace(tmp, os.path.join(demo_path, f"{env_id}_expert.zip"))


def _atomic_save_npz(arrays, final_path):
    """np.savez to a temp file in the same dir, then os.replace onto the final
    path, so a concurrent reader can never observe a partially written file."""
    tmp = os.path.join(os.path.dirname(final_path),
                       f".{os.path.basename(final_path)}.tmp-{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, final_path)


def _read_demo_arrays(path):
    """Fully materialize the demo arrays (validates the whole file on read)."""
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in ("obs", "acts", "next_obs", "dones")}


def _locked_demo_load(demo_path, env_id, path):
    """Generate-or-load the shared demo cache under an inter-process lock.

    All runs sharing ``demo_path`` serialize here: the first process trains
    the expert and publishes the cache atomically while the others block on
    the lock and then just load it. A cache file that fails to load (e.g. a
    torn write left behind by a crashed/killed earlier run) is deleted and
    regenerated under the same lock.
    """
    import fcntl
    import zipfile

    os.makedirs(demo_path, exist_ok=True)
    with open(path + ".lock", "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not os.path.exists(path):
                generate_expert_demos(demo_path, env_id)
            try:
                return _read_demo_arrays(path)
            except (zipfile.BadZipFile, EOFError, KeyError, ValueError, OSError) as exc:
                print(f"Corrupt demo cache {path} ({exc!r}); regenerating...", flush=True)
                os.remove(path)
                generate_expert_demos(demo_path, env_id)
                return _read_demo_arrays(path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


'''

EXTRA_HELPERS = '''def _save_rng_states():
    """Snapshot the global RNG states (python, numpy, torch, torch.cuda)."""
    states = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_rng_states(states):
    random.setstate(states["random"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "cuda" in states:
        torch.cuda.set_rng_state_all(states["cuda"])


def generate_expert_demos(demo_path, env_id, total_timesteps=2_000_000, n_demos=25000):
    """Deterministic expert generation, independent of the caller's seed.

    The expert (and thus the shared demo cache) must not depend on which
    concurrent run happens to win the generation lock, so training runs
    under a FIXED seed derived from ``env_id`` alone. The caller's RNG
    states are saved first and restored afterwards, so each run's own
    per-seed randomness is unaffected by whether it generated or loaded.
    """
    import zlib

    gen_seed = zlib.crc32(env_id.encode("utf-8")) % (2 ** 31)
    saved = _save_rng_states()
    try:
        random.seed(gen_seed)
        np.random.seed(gen_seed)
        torch.manual_seed(gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_seed)
        _generate_expert_demos_impl(demo_path, env_id, gen_seed, total_timesteps, n_demos)
    finally:
        _restore_rng_states(saved)


'''


def write_atomic(new_text):
    tmp = path.with_name(".custom_irl.py.patch-tmp")
    tmp.write_text(new_text)
    os.replace(tmp, path)


if V2_MARKER in text:
    print("[runtime-patch] demo-cache fix (v2) already present; no-op")
    raise SystemExit(0)

if V1_MARKER in text:
    # Intermediate fix present (atomic publish + lock, but no deterministic
    # expert seeding): upgrade to v2.
    if not (V1_GEN in text and BANNER in text):
        print("[runtime-patch] v1 anchors not all found; "
              "leaving custom_irl.py untouched")
        raise SystemExit(0)
    new_text = text.replace(V1_GEN, NEW_IMPL, 1)
    idx = new_text.rfind(BANNER)
    new_text = new_text[:idx] + EXTRA_HELPERS + new_text[idx:]
    write_atomic(new_text)
    print("[runtime-patch] upgraded v1 demo-cache fix to v2 "
          "(deterministic expert seeding) in custom_irl.py")
    raise SystemExit(0)

# Pre-fix scaffold: apply the full fix, all-or-none.
if not (GEN0_GEN in text and OLD_LOAD in text and BANNER in text):
    print("[runtime-patch] pre-fix anchors not all found; "
          "leaving custom_irl.py untouched")
    raise SystemExit(0)

new_text = text.replace(GEN0_GEN, NEW_IMPL, 1)
new_text = new_text.replace(OLD_LOAD, NEW_LOAD, 1)
# Insert the helpers right before the LAST occurrence of the main-loop
# banner (the agent's editable region sits above it; using the last
# occurrence keeps a pathological editable-region copy from catching the
# splice).
idx = new_text.rfind(BANNER)
new_text = new_text[:idx] + BASE_HELPERS + EXTRA_HELPERS + new_text[idx:]
write_atomic(new_text)
print("[runtime-patch] applied demo-cache concurrency fix (v2) to custom_irl.py")
PY
} 9>"${_rl_demo_patch_lock}"
