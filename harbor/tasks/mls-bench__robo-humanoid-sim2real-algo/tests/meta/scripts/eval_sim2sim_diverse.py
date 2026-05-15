#!/usr/bin/env python3
"""Evaluate humanoid locomotion policy on diverse random commands using sim2sim (MuJoCo)."""

import os
import sys

# IMPORTANT: isaacgym MUST be imported before torch (Isaac Gym's gymdeps enforces this).
# So we do the humanoid.envs import (which transitively imports isaacgym) up front,
# before any module that pulls in torch.
sys.path.insert(0, '/workspace/humanoid-gym')
from humanoid import LEGGED_GYM_ROOT_DIR
try:
    from humanoid.envs import XBotLCustomCfg as EvalBaseCfg
except ImportError:
    from humanoid.envs import XBotLCfg as EvalBaseCfg

import math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import torch


def quaternion_to_euler_array(quat):
    """Convert quaternion to Euler angles."""
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data):
    """Extract observation from MuJoCo data."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0


def get_base_body_id(model):
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body_id < 0:
        base_body_id = 1  # body 0 is world; fall back to first non-world body
    return base_body_id


def evaluate_command(policy, model, cfg, command, base_body_id, eval_duration=10.0):
    """Faithful replica of run_mujoco() inner loop from official sim2sim.py,
    parameterized over the velocity command and instrumented for per-command metrics.
    Differences from official: command sampled (not fixed), per-step vel error tracked,
    fall detection on base_link world height + roll/pitch.
    """
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    cmd.vx, cmd.vy, cmd.dyaw = map(float, command)
    cfg.sim_config.sim_duration = eval_duration
    cfg.sim_config.dt = 0.001
    cfg.sim_config.decimation = 10

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    from collections import deque
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    num_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)

    vel_errors = []
    fell = False

    for _ in range(num_steps):
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # Fall check on world-frame base height + roll/pitch
        base_height = float(data.xpos[base_body_id, 2])
        eu_ang_full = quaternion_to_euler_array(quat)
        eu_ang_full[eu_ang_full > math.pi] -= 2 * math.pi
        if base_height < 0.3 or abs(eu_ang_full[0]) > 0.5 or abs(eu_ang_full[1]) > 0.5:
            fell = True

        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64)
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:41] = action
            obs[0, 41:44] = omega
            obs[0, 44:47] = eu_ang

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

            with torch.no_grad():
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale

            if not fell:
                lin_vel_error = float(np.linalg.norm(np.array([cmd.vx, cmd.vy]) - v[:2]))
                ang_vel_error = float(abs(cmd.dyaw - omega[2]))
                vel_errors.append(lin_vel_error + ang_vel_error)

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau

        mujoco.mj_step(model, data)
        count_lowlevel += 1

    avg_vel_error = float(np.mean(vel_errors)) if vel_errors else float('inf')
    success = (avg_vel_error < 0.5) and (not fell)
    return success, avg_vel_error, fell


def load_policy(policy_path):
    try:
        try:
            policy = torch.load(policy_path, map_location='cpu', weights_only=False)
        except TypeError:
            policy = torch.load(policy_path, map_location='cpu')
    except Exception as pickle_error:
        try:
            policy = torch.jit.load(policy_path)
        except Exception as jit_error:
            raise RuntimeError(
                f"Failed to load policy as plain torch module ({pickle_error}) "
                f"or TorchScript ({jit_error})"
            ) from jit_error
    if hasattr(policy, 'eval'):
        policy.eval()
    return policy


def main():
    # Configuration
    num_commands = int(os.getenv('NUM_COMMANDS', '100'))
    eval_duration = float(os.getenv('EVAL_DURATION', '10.0'))

    # Find the trained model:
    #   1. Prefer $OUTPUT_DIR/exported/policies/ — persists across baseline runs so eval-only
    #      invocations (different workspace from train) can still see the trained policy.
    #   2. Fall back to /workspace/humanoid-gym/logs/<latest>/exported/policies/ — used when
    #      train and eval run in the same workspace (combined runs).
    output_dir = os.environ.get('OUTPUT_DIR', '')
    persistent_dir = os.path.join(output_dir, 'exported', 'policies') if output_dir else ''
    workspace_log_dir = '/workspace/humanoid-gym/logs'
    model_dir = None
    if persistent_dir and os.path.exists(persistent_dir):
        # Only use it if it has at least one non-placeholder .pt
        if any(f.endswith('.pt') and f != 'policy_example.pt' for f in os.listdir(persistent_dir)):
            model_dir = persistent_dir
            print(f"Using persistent OUTPUT_DIR policy: {model_dir}")
    if model_dir is None:
        if not os.path.isdir(workspace_log_dir):
            print(f"ERROR: No persistent policy at {persistent_dir} and no workspace logs at {workspace_log_dir}")
            sys.exit(1)
        experiment_dirs = [d for d in os.listdir(workspace_log_dir) if os.path.isdir(os.path.join(workspace_log_dir, d))]
        if not experiment_dirs:
            print(f"ERROR: No experiment directories found in {workspace_log_dir}/")
            sys.exit(1)
        latest_exp = max(experiment_dirs, key=lambda d: os.path.getmtime(os.path.join(workspace_log_dir, d)))
        model_dir = os.path.join(workspace_log_dir, latest_exp, 'exported', 'policies')
        if not os.path.exists(model_dir):
            print(f"ERROR: No exported policy found at {model_dir}")
            print("Make sure the policy was exported during training.")
            sys.exit(1)

    # Load the policy: pick the most-recently-saved .pt that is NOT the placeholder.
    # `policy_example.pt` ships with the package; the actual trained policy is
    # `policy_1.pt` (or similar). Refuse to silently evaluate the placeholder —
    # that would produce fall_rate=1.0 with a clean exit and look like a real
    # baseline failure.
    candidates = [f for f in os.listdir(model_dir) if f.endswith('.pt') and f != 'policy_example.pt']
    if not candidates:
        print(f"ERROR: Only placeholder policy_example.pt found in {model_dir}.")
        print("Training did not export a real policy_1.pt — refusing to evaluate the placeholder.")
        sys.exit(2)
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
    policy_path = os.path.join(model_dir, candidates[0])
    print(f"Loading policy from: {policy_path}")
    policy = load_policy(policy_path)

    # Create configuration
    class Sim2simCfg(EvalBaseCfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L.xml'
            sim_duration = eval_duration
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)

    cfg = Sim2simCfg()

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    base_body_id = get_base_body_id(model)

    print(f"\nEvaluating on {num_commands} random commands (sim2sim in MuJoCo)...")
    print(f"Evaluation duration per command: {eval_duration}s")

    # Read command ranges from environment variables (with defaults for diverse commands)
    vx_min = float(os.getenv('EVAL_VX_MIN', '-0.5'))
    vx_max = float(os.getenv('EVAL_VX_MAX', '1.0'))
    vy_min = float(os.getenv('EVAL_VY_MIN', '-0.4'))
    vy_max = float(os.getenv('EVAL_VY_MAX', '0.4'))
    dyaw_min = float(os.getenv('EVAL_DYAW_MIN', '-0.5'))
    dyaw_max = float(os.getenv('EVAL_DYAW_MAX', '0.5'))

    print(f"Command ranges: vx=[{vx_min}, {vx_max}], vy=[{vy_min}, {vy_max}], dyaw=[{dyaw_min}, {dyaw_max}]")

    # Sample random commands from specified ranges. MLS-Bench injects SEED;
    # MLSB_SEED is kept as a legacy fallback for older manual launches.
    np.random.seed(int(os.getenv('SEED', os.getenv('MLSB_SEED', '42'))))
    commands = []
    for i in range(num_commands):
        vx = np.random.uniform(vx_min, vx_max)
        vy = np.random.uniform(vy_min, vy_max)
        dyaw = np.random.uniform(dyaw_min, dyaw_max)
        commands.append([vx, vy, dyaw])

    # Evaluate on each command
    successes = []
    vel_errors = []
    falls = []

    for i, cmd in enumerate(commands):
        success, avg_vel_error, fall_detected = evaluate_command(
            policy, model, cfg, cmd, base_body_id, eval_duration
        )
        successes.append(success)
        vel_errors.append(avg_vel_error)
        falls.append(fall_detected)

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_commands} commands evaluated")

    # Compute final metrics
    success_rate = np.mean(successes)
    avg_vel_error = np.mean(vel_errors)
    fall_rate = np.mean(falls)

    # Output metrics in parseable format
    print(f"\n{'='*60}")
    print(f"TEST_METRICS success_rate={success_rate:.4f} avg_vel_error={avg_vel_error:.4f} fall_rate={fall_rate:.4f}")
    print(f"{'='*60}")
    print(f"\nSim2Sim Evaluation Results:")
    print(f"  Success Rate: {success_rate*100:.1f}%")
    print(f"  Average Velocity Error: {avg_vel_error:.3f}")
    print(f"  Fall Rate: {fall_rate*100:.1f}%")
    print(f"  Successful Commands: {sum(successes)}/{num_commands}")


if __name__ == '__main__':
    main()
