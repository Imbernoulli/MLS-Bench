"""FIXED online-bandit evaluation harness.  READ-ONLY -- do not edit.

This file owns the bandit environments and the regret accounting.  Your
policy (``BanditPolicy`` in ``custom_bandit.py``) never runs inside it:

1. **Your policy runs in a separate interpreter.**  The harness starts
   ``custom_bandit.py --worker`` as a child process and talks to it over a
   pipe, one line per round.  The policy never shares an address space or a
   standard output with the code that measures it; the only thing it can
   influence is which arm it names at each round.  When the harness runs as
   root (the evaluation container), the child is also dropped to an
   unprivileged user.

2. **Every instance is drawn from fresh operating-system entropy.**  Only
   the *distribution* of instances is public (below).  The arm means,
   parameters, changepoints, reward noise and contexts of a run are new every
   time, exist only in this process, and are not a function of ``SEED``.
   ``SEED`` seeds only your policy's own random generator (``np.random.seed``
   in the worker), so the parts you control stay reproducible.

Settings (horizon T = 10000 each):
    stochastic_mab  10 Bernoulli arms.  Best mean ~ U[0.7, 0.9]; the other
                    nine are i.i.d. U[0.1, best - 0.1].
    contextual      5 arms, d = 10; expected reward x^T theta_a with
                    theta_a = 0.5 g_a / ||g_a||, g_a ~ N(0, I); x uniform on
                    the unit sphere; Gaussian noise sd 0.1; rewards clipped
                    to [0, 1].
    nonstationary   5 Bernoulli arms, 5 stationary segments.  Changepoints
                    near t = 2000, 4000, 6000, 8000 (uniform jitter of up to
                    +-400 rounds).  Each segment's best arm differs from the
                    previous one's and has mean ~ U[0.7, 0.9]; the previous
                    best collapses to ~ U[0.1, 0.3]; the rest are i.i.d.
                    U[0.1, best - 0.2].

Metric: normalized regret = cumulative expected regret / T, averaged over
``--replicates`` independent runs (each with a fresh instance draw and a fresh
policy process).

Wire protocol (handled for you by the fixed ``_worker_main`` in
``custom_bandit.py``):

    parent -> child   INIT <K> <context_dim> <horizon> <policy_seed>
    child  -> parent  READY
    parent -> child   S <t> <prev_arm> <prev_reward> [<x_1> ... <x_d>]
                      (prev_arm = -1 at t = 0; the context is omitted when d = 0)
    child  -> parent  <arm>
    parent -> child   U <last_arm> <last_reward>   (final update, no reply)
    parent -> child   BYE
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HORIZON = 10000
PRINT_EVERY = 1000
UNPRIVILEGED_UID = 65534  # nobody
WORKER_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "OMP_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "OMP_THREAD_LIMIT")


# =====================================================================
# Environments (instances are private to this process)
# =====================================================================
class StochasticMAB:
    """K = 10 Bernoulli arms.  Best mean ~ U[0.7, 0.9]; the other nine means
    are i.i.d. U[0.1, best - 0.1], so the best arm leads by at least 0.1."""

    K = 10

    def __init__(self, rng: np.random.Generator, seed: int):
        self.rng = rng
        best = rng.uniform(0.7, 0.9)
        means = rng.uniform(0.1, best - 0.1, size=self.K)
        means[rng.integers(self.K)] = best
        self.means = means
        self.context_dim = 0
        self.best_mean = float(best)

    def get_context(self):
        return None

    def pull(self, arm: int) -> tuple[float, float]:
        reward = float(self.rng.random() < self.means[arm])
        return reward, self.best_mean - float(self.means[arm])


class ContextualBandit:
    """K = 5 arms, d = 10.  theta_a = 0.5 * g_a / ||g_a|| with g_a standard
    normal; contexts uniform on the unit sphere; Gaussian noise sd 0.1;
    rewards clipped to [0, 1]."""

    K, D = 5, 10
    NOISE_STD = 0.1

    def __init__(self, rng: np.random.Generator, seed: int):
        self.rng = rng
        self.context_dim = self.D
        raw = rng.standard_normal((self.K, self.D))
        self.theta = raw / np.linalg.norm(raw, axis=1, keepdims=True) * 0.5
        self._x = None

    def get_context(self) -> np.ndarray:
        x = self.rng.standard_normal(self.D)
        self._x = x / np.linalg.norm(x)
        return self._x

    def pull(self, arm: int) -> tuple[float, float]:
        expected = self.theta @ self._x
        noise = self.rng.normal(0.0, self.NOISE_STD)
        reward = float(np.clip(expected[arm] + noise, 0.0, 1.0))
        return reward, float(expected.max() - expected[arm])


class NonStationaryMAB:
    """K = 5 Bernoulli arms, 5 stationary segments (4 changepoints).
    Changepoints sit near t = 2000, 4000, 6000, 8000 with a uniform jitter of
    up to +-400 rounds.  In every segment the best arm differs from the
    previous segment's, its mean is ~ U[0.7, 0.9], the previous best arm
    collapses to ~ U[0.1, 0.3], and the other means are i.i.d.
    U[0.1, best - 0.2]."""

    K = 5
    N_SEGMENTS = 5
    JITTER = 400

    def __init__(self, rng: np.random.Generator, seed: int):
        self.rng = rng
        seg = HORIZON // self.N_SEGMENTS
        self.changepoints = [
            int(k * seg + rng.integers(-self.JITTER, self.JITTER + 1))
            for k in range(1, self.N_SEGMENTS)
        ]
        self.configs = []
        prev_best = -1
        for _ in range(self.N_SEGMENTS):
            best_arm = int(rng.choice([a for a in range(self.K)
                                       if a != prev_best]))
            best = rng.uniform(0.7, 0.9)
            means = rng.uniform(0.1, best - 0.2, size=self.K)
            if prev_best >= 0:
                means[prev_best] = rng.uniform(0.1, 0.3)
            means[best_arm] = best
            self.configs.append(means)
            prev_best = best_arm
        self.context_dim = 0
        self._t = 0
        self._segment = 0

    def get_context(self):
        return None

    def pull(self, arm: int) -> tuple[float, float]:
        while (self._segment < len(self.changepoints)
               and self._t >= self.changepoints[self._segment]):
            self._segment += 1
        means = self.configs[self._segment]
        reward = float(self.rng.random() < means[arm])
        self._t += 1
        return reward, float(means.max() - means[arm])


ENVS = {
    "stochastic_mab": StochasticMAB,
    "contextual": ContextualBandit,
    "nonstationary": NonStationaryMAB,
}


# =====================================================================
# Policy process
# =====================================================================
class ProtocolError(RuntimeError):
    """The policy process stopped speaking the protocol."""


def _worker_preexec(workspace: str):
    def _fn():
        os.setsid()
        if os.geteuid() == 0:
            os.setgroups([])
            os.setgid(UNPRIVILEGED_UID)
            os.setuid(UNPRIVILEGED_UID)
        os.chdir(workspace)
    return _fn


class PolicyWorker:
    def __init__(self, policy_path: str, workspace: str, stderr_fh):
        self.home = tempfile.mkdtemp(prefix="mlsb_policy_", dir="/tmp")
        os.chmod(self.home, 0o700)
        if os.geteuid() == 0:
            os.chown(self.home, UNPRIVILEGED_UID, UNPRIVILEGED_UID)
            # The unprivileged policy process must be able to read its file.
            os.chmod(policy_path, os.stat(policy_path).st_mode | 0o444)
        env = {k: os.environ[k] for k in WORKER_ENV_KEYS if k in os.environ}
        env["HOME"] = env["TMPDIR"] = self.home
        preexec = _worker_preexec(workspace)

        self.proc = subprocess.Popen(
            [sys.executable, "-B", "-s", policy_path, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_fh,
            env=env,
            text=True,
            bufsize=1,
            close_fds=True,
            preexec_fn=preexec,
        )

    def send(self, line: str) -> None:
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise ProtocolError("policy process is gone (%s)" % exc) from None

    def recv(self) -> str:
        line = self.proc.stdout.readline()
        if line == "":
            raise ProtocolError("policy process closed its output stream")
        return line.strip()

    def close(self) -> None:
        try:
            self.send("BYE")
        except ProtocolError:
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        try:  # reap anything the policy left behind in its session
            os.killpg(self.proc.pid, 9)
        except OSError:
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except Exception:
                pass
        subprocess.run(["rm", "-rf", self.home], check=False)


def play(worker: PolicyWorker, env, horizon: int, policy_seed: int,
         tag: str) -> float:
    worker.send("INIT %d %d %d %d" % (env.K, env.context_dim, horizon,
                                      policy_seed))
    ack = worker.recv()
    if ack != "READY":
        raise ProtocolError("expected READY after INIT, got %r" % ack[:80])

    cumulative = 0.0
    prev_arm, prev_reward = -1, 0.0
    for t in range(horizon):
        x = env.get_context()
        msg = "S %d %d %r" % (t, prev_arm, prev_reward)
        if x is not None:
            msg += " " + " ".join(repr(float(v)) for v in x)
        worker.send(msg)
        answer = worker.recv()
        try:
            arm = int(answer)
        except (TypeError, ValueError):
            raise ProtocolError("policy answered %r, which is not an arm index"
                                % answer[:80]) from None
        if not 0 <= arm < env.K:
            raise ProtocolError("policy returned arm %d outside {0..%d}"
                                % (arm, env.K - 1))
        prev_reward, regret = env.pull(arm)
        prev_arm = arm
        cumulative += regret
        if (t + 1) % PRINT_EVERY == 0:
            print("TRAIN_METRICS %sstep=%d cumulative_regret=%.4f "
                  "normalized_regret=%.6f"
                  % (tag, t + 1, cumulative, cumulative / (t + 1)), flush=True)
    # Deliver the final observation, as the in-process loop used to.
    worker.send("U %d %r" % (prev_arm, prev_reward))
    return cumulative


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(ENVS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--nonce-file", required=True)
    ap.add_argument("--policy", default=None)
    args = ap.parse_args()

    # Read the run nonce and destroy it before any untrusted code exists:
    # only the line carrying it is trusted by the wrapper script.
    with open(args.nonce_file) as fh:
        nonce = fh.read().strip()
    os.unlink(args.nonce_file)
    if not nonce:
        print("[harness] empty nonce", flush=True)
        return 2

    here = os.path.dirname(os.path.abspath(__file__))
    policy_path = os.path.abspath(args.policy or
                                  os.path.join(here, "custom_bandit.py"))
    workspace = os.path.dirname(os.path.dirname(policy_path))
    stderr_path = os.path.join(tempfile.gettempdir(), "policy_stderr.log")

    print("[harness] env=%s horizon=%d replicates=%d policy_seed=%d"
          % (args.env, HORIZON, args.replicates, args.seed), flush=True)
    started = time.time()
    regrets = []
    status = "OK"
    with open(stderr_path, "w") as stderr_fh:
        for r in range(args.replicates):
            env = ENVS[args.env](np.random.default_rng(), args.seed)
            worker = PolicyWorker(policy_path, workspace, stderr_fh)
            try:
                regrets.append(play(worker, env, HORIZON, args.seed + r,
                                    "replicate=%d " % r))
            except ProtocolError as exc:
                status = "PROTOCOL_ERROR"
                print("[harness] %s" % exc, flush=True)
            finally:
                worker.close()
            if status != "OK":
                break
            print("[harness] replicate %d/%d normalized_regret=%.6f "
                  "elapsed=%.1fs" % (r + 1, args.replicates,
                                     regrets[-1] / HORIZON,
                                     time.time() - started), flush=True)

    with open(stderr_path) as fh:
        tail = fh.read()[-4000:]
    if tail.strip():
        print("[harness] policy stderr tail:", flush=True)
        for line in tail.splitlines():
            print("[policy] " + line, flush=True)

    values = np.asarray(regrets, dtype=float)
    if (status != "OK" or len(values) != args.replicates
            or not np.all(np.isfinite(values))):
        print("[harness] run did not complete (%s, %d/%d replicates)"
              % (status, len(values), args.replicates), flush=True)
        return 1
    mean = float(values.mean())
    print("HARNESS_RESULT %s cumulative_regret=%.4f normalized_regret=%.6f"
          % (nonce, mean, mean / HORIZON), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
