"""
Self-contained script for JEPA planning evaluation.

Part 1: Load a released AC Video JEPA checkpoint (or retrain on demand).
Part 2: Define CustomPlanner (EDITABLE REGION).
Part 3: Run planning evaluation and report metrics.
"""

import os
import sys
# Prevent eb_jepa/logging.py from shadowing stdlib logging:
# remove any sys.path entry that contains a `logging.py` file. This is
# path-agnostic and safe when the script lives at the repo root (pwd-script
# dir is /workspace/eb_jepa, which itself has no logging.py — only the
# eb_jepa subpackage does).
sys.path = [p for p in sys.path if not os.path.isfile(os.path.join(p, "logging.py"))]
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    RNNPredictor,
)
from eb_jepa.datasets.two_rooms.env import DotWall
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.planning import (
    Planner,
    PlanningResult,
    ReprTargetDistMPCObjective,
)
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.state_decoder import MLPXYHead

# ============================================================================
# PART 1: Model Training / Checkpoint Loading
# ============================================================================

# Training hyperparameters (from train.yaml).
# NOTE: mlsbench sets the ENV env var to the cmd_label (e.g. "horizon-30"),
# which is NOT the dataset name. The only supported dataset for this task
# is two_rooms, so hardcode it.
TASK_ENV = "two-rooms"
ENV_NAME = "two_rooms"
SEED = int(os.environ.get("SEED", "42"))
PLAN_LENGTH = int(os.environ.get("PLAN_LENGTH", "90"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/eb_jepa/outputs"))
CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_PATH = CKPT_DIR / "ac_jepa.pth.tar"
PRETRAINED_CKPT_PATH = Path(
    os.environ.get(
        "JEPA_PRETRAINED_CKPT",
        "/data/eb_jepa/checkpoints/ac_jepa_e11.pth.tar",
    )
)
FORCE_TRAIN = os.environ.get("JEPA_FORCE_TRAIN", "0") == "1"
EPOCHS = 12
BATCH_SIZE = 384
LR = 0.001
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 2.0
DOBS = 2
HENC = 32
DSTC = 32
NSTEPS = 8
COV_COEFF = 8
STD_COEFF = 16
SIM_COEFF_T = 12
IDM_COEFF = 1


def build_data_cfg(batch_size):
    """Create the shared dataset config for the selected environment."""
    return {
        "env_name": ENV_NAME,
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_mem": False,
        "persistent_workers": False,
    }


def resolve_checkpoint_path():
    """Choose the released checkpoint by default, with local fallback."""
    if not FORCE_TRAIN and PRETRAINED_CKPT_PATH.exists():
        return PRETRAINED_CKPT_PATH
    if CKPT_PATH.exists():
        return CKPT_PATH
    return None


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_env(env, seed):
    for obj in (env, getattr(env, "action_space", None), getattr(env, "observation_space", None)):
        if hasattr(obj, "seed"):
            obj.seed(seed)


def reset_env(env, seed):
    seed_env(env, seed)
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def build_model(device, loader):
    """Build the JEPA model with all components."""
    img_size = loader.dataset.config.img_size

    # Encoder
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, HENC, DSTC),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=DOBS,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(DOBS, img_size, img_size),
    )
    test_input = torch.rand((1, DOBS, 1, img_size, img_size))
    test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape

    # Predictor
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln
    )

    # Action encoder (identity)
    aencoder = nn.Identity()

    # IDM
    idm = InverseDynamicsModel(
        state_dim=h * w * f,
        hidden_dim=256,
        action_dim=2,
    ).to(device)

    # Regularizer
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=COV_COEFF,
        std_coeff=STD_COEFF,
        sim_coeff_t=SIM_COEFF_T,
        idm_coeff=IDM_COEFF,
        idm=idm,
        first_t_only=False,
        projector=None,
        spatial_as_samples=False,
        idm_after_proj=False,
        sim_t_after_proj=False,
    )

    # Loss and JEPA
    ploss = SquareLossSeq()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)

    # Position prober
    xy_head = MLPXYHead(
        input_shape=test_output.shape[1],
        normalizer=loader.dataset.normalizer,
    ).to(device)
    xy_prober = JEPAProbe(jepa=jepa, head=xy_head, hcost=nn.MSELoss())

    return jepa, xy_head, xy_prober, test_output.shape


def train_model(device):
    """Train the AC Video JEPA model and save checkpoint."""
    print("Training AC Video JEPA model...")

    loader, val_loader, data_config = init_data(
        env_name=ENV_NAME,
        cfg_data=build_data_cfg(BATCH_SIZE),
    )

    jepa, xy_head, xy_prober, _ = build_model(device, loader)

    steps_per_epoch = data_config.size // data_config.batch_size
    total_steps = EPOCHS * steps_per_epoch

    jepa_optimizer = AdamW(jepa.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)

    probe_optimizer = AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)
    probe_scheduler = CosineWithWarmup(probe_optimizer, total_steps, warmup_ratio=0.1)

    dtype = torch.bfloat16
    use_amp = True
    scaler = GradScaler(device.type, enabled=use_amp)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch}/{EPOCHS - 1}",
        )
        for idx, (x, a, loc, _, _) in pbar:
            x = x.to(device)
            a = a.to(device)
            loc = loc.to(device)
            total_loss = torch.tensor(0.0, device=device)

            # JEPA loss
            jepa_optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                _, (jepa_loss, regl, regl_unweight, regldict, pl) = jepa.unroll(
                    x, a,
                    nsteps=NSTEPS,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=True,
                    return_all_steps=False,
                )
                total_loss += jepa_loss

            scaler.scale(jepa_loss).backward()
            scaler.unscale_(jepa_optimizer)
            torch.nn.utils.clip_grad_norm_(jepa.encoder.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(jepa.predictor.parameters(), GRAD_CLIP)
            scaler.step(jepa_optimizer)
            scaler.update()
            jepa_scheduler.step()

            # Probe loss
            probe_optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                xy_loss = xy_prober(
                    observations=x[:, :, :1],
                    targets=loc[:, :, :1],
                )
                xy_loss = loader.dataset.normalizer.unnormalize_mse(xy_loss)
                total_loss += xy_loss

            scaler.scale(xy_loss).backward()
            scaler.step(probe_optimizer)
            scaler.update()
            probe_scheduler.step()

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "reg": f"{regl.item():.4f}",
                "pred": f"{pl.item():.4f}",
            })

        epoch_time = time.time() - epoch_start
        print(
            f"TRAIN_METRICS: epoch={epoch}, loss={total_loss.item():.4f}, "
            f"reg={regl.item():.4f}, pred={pl.item():.4f}, "
            f"probe={xy_loss.item():.4f}, time={epoch_time:.1f}s",
            flush=True,
        )

    # Save checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = {
        "epoch": EPOCHS - 1,
        "model_state_dict": jepa.state_dict(),
        "xy_head_state_dict": xy_head.state_dict(),
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"Checkpoint saved to {CKPT_PATH}", flush=True)

    return jepa, xy_head, xy_prober, loader


def load_model(device):
    """Load the JEPA model from checkpoint."""
    loader, val_loader, data_config = init_data(
        env_name=ENV_NAME,
        cfg_data=build_data_cfg(64),
    )

    jepa, xy_head, xy_prober, _ = build_model(device, loader)

    checkpoint_path = resolve_checkpoint_path()
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found at {PRETRAINED_CKPT_PATH} or {CKPT_PATH}"
        )
    print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", {})
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    jepa.load_state_dict(state_dict)

    if "xy_head_state_dict" in checkpoint:
        xy_head.load_state_dict(checkpoint["xy_head_state_dict"])

    print("Model loaded successfully.", flush=True)
    return jepa, xy_head, xy_prober, loader


# ============================================================================
# PART 2: Custom Planner (EDITABLE REGION)
# ============================================================================

# EDITABLE REGION START
class CustomPlanner(Planner):
    """Custom planning algorithm for JEPA world models.

    Uses the learned JEPA world model to search for optimal action
    sequences that reach a specified goal state.

    Available methods (inherited from Planner):
        self.unroll(obs_init, actions): Forward simulate actions through world model
            obs_init: [1, C, 1, H, W] initial observation encoding
            actions: [B, A, T] action sequences to evaluate
            Returns: [B, D, T+1, H, W] predicted state encodings

        self.objective(encodings): Compute cost for predicted encodings
            encodings: [B, D, T, H, W]
            Returns: [B] cost per sample (lower is better)

        self.cost_function(actions, obs_init): Convenience method
            Calls unroll then objective
            Returns: [B] cost per sample

    plan() must return PlanningResult(actions=Tensor[T, A], ...)
    """

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=200, n_iters=20, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length
        # TODO: Implement your planning algorithm here.
        # You have access to:
        #   self.unroll(obs_init, actions) - forward simulate through world model
        #   self.objective(encodings)      - compute cost (lower = better)
        #   self.cost_function(actions, obs_init) - convenience: unroll + objective
        #
        # Return PlanningResult with action sequence of shape [plan_length, action_dim]
        actions = torch.zeros(plan_length, self.action_dim, device=self.device)
        return PlanningResult(actions=actions)
# EDITABLE REGION END


# ============================================================================
# PART 3: Planning Evaluation
# ============================================================================

def create_env(data_config):
    """Create the Two Rooms evaluation environment."""
    if ENV_NAME != "two_rooms":
        raise ValueError(f"Unsupported ENV='{TASK_ENV}' for jepa-planning")
    return DotWall(
        config=data_config,
        n_allowed_steps=200,
        level="normal",
    )


def run_planning_eval(jepa, xy_prober, loader, device, num_episodes=20):
    """Run planning evaluation with the CustomPlanner."""
    jepa.eval()

    data_config = loader.dataset.config
    env = create_env(data_config)
    reset_env(env, SEED)

    # Create a lightweight GCAgent-like wrapper that uses CustomPlanner
    normalizer = env.normalizer

    # Create the planner
    planner = CustomPlanner(
        unroll=None,  # Will be set via agent wrapper
        action_dim=2,
        plan_length=PLAN_LENGTH,
        num_samples=200,
        n_iters=20,
    )

    # We need to wire up the unroll function properly
    # The planner needs access to model unroll through the GCAgent's unroll method
    class PlanningAgent:
        def __init__(self, model, planner, normalizer, env, prober):
            self.model = model
            self.planner = planner
            self.normalizer = normalizer
            self.env = env
            self.device = next(model.parameters()).device
            self.loc_prober = prober
            self.goal_state = None
            self.goal_position = None
            self.goal_state_enc = None
            self.objective = None
            self.num_act_stepped = 1

            # Wire planner's unroll to agent's unroll
            self.planner.unroll = self.unroll

        def unroll(self, obs_init, actions, repeat_batch=True):
            batch_size = actions.shape[0]
            nsteps = actions.shape[2]
            if repeat_batch:
                obs_init_rep = obs_init.repeat(batch_size, 1, 1, 1, 1)
            else:
                obs_init_rep = obs_init
            predicted_states, _ = self.model.unroll(
                obs_init_rep, actions,
                nsteps=nsteps,
                unroll_mode="autoregressive",
                ctxt_window_time=1,
                compute_loss=False,
                return_all_steps=False,
            )
            return predicted_states

        def set_goal(self, goal_state, goal_position=None):
            self.goal_position = goal_position
            self.goal_state = goal_state
            self.goal_state_enc = self.model.encode(
                self.normalizer.normalize_state(goal_state.to(self.device))
                .unsqueeze(0)
                .unsqueeze(2)
            )
            self.objective = ReprTargetDistMPCObjective(
                target_enc=self.goal_state_enc,
                sum_all_diffs=True,
            )
            self.planner.set_objective(self.objective)

        def act(self, obs_tensor, steps_left=None, t0=False):
            planning_result = self.planner.plan(
                obs_tensor,
                steps_left=steps_left,
                eval_mode=True,
                t0=t0,
            )
            return planning_result.actions[:self.num_act_stepped]

    agent = PlanningAgent(jepa, planner, normalizer, env, xy_prober)

    successes = []
    distances = []
    steps_to_success = []

    for ep in range(num_episodes):
        obs, info = reset_env(env, SEED + ep)
        obs, reward, done, truncated, info = env.step(
            np.zeros(env.action_space.shape[0])
        )
        goal_img = info["target_obs"]

        agent.set_goal(
            goal_img.detach().clone().to(dtype=torch.float32),
            info["target_position"],
        )

        steps_left = env.n_allowed_steps
        total_steps = env.n_allowed_steps
        t0 = True
        success = False
        state_dist = float("inf")
        first_success_step = None

        while steps_left > 0:
            obs_tensor = (
                normalizer.normalize_state(
                    obs.detach().clone().to(dtype=torch.float32, device=device)
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )
            with torch.no_grad():
                action = agent.act(
                    obs_tensor, steps_left=steps_left, t0=t0
                ).cpu().numpy()

            for a in action:
                obs, reward, done, truncated, info = env.step(a)
                t0 = False
                steps_left -= 1

                eval_results = env.eval_state(
                    info["target_position"], info["dot_position"]
                )
                success = eval_results["success"]
                state_dist = eval_results["state_dist"]

                if success and first_success_step is None:
                    first_success_step = total_steps - steps_left
                if done or truncated or steps_left <= 0:
                    break
            if done or truncated:
                break

        successes.append(success)
        distances.append(state_dist)
        if first_success_step is not None:
            steps_to_success.append(first_success_step)
        print(
            f"PLAN_METRICS: episode={ep}, success={success}, dist={state_dist:.4f}",
            flush=True,
        )

    success_rate = np.mean(successes)
    mean_dist = np.mean(distances)
    mean_steps = np.mean(steps_to_success) if steps_to_success else float("nan")
    print(
        f"TEST_METRICS: success_rate={success_rate:.2f}, "
        f"mean_dist={mean_dist:.4f}, "
        f"mean_steps_to_success={mean_steps:.2f}",
        flush=True,
    )
    return success_rate, mean_dist


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(
        f"Runtime config: seed={SEED}, env={TASK_ENV}, output_dir={OUTPUT_DIR}",
        flush=True,
    )

    # Part 1: Load released checkpoint by default, retrain only as a fallback.
    if resolve_checkpoint_path() is not None and not FORCE_TRAIN:
        jepa, xy_head, xy_prober, loader = load_model(device)
    else:
        if FORCE_TRAIN:
            print("JEPA_FORCE_TRAIN=1 -> retraining world model", flush=True)
        else:
            print(
                f"Released checkpoint missing at {PRETRAINED_CKPT_PATH}; "
                "falling back to local training.",
                flush=True,
            )
        jepa, xy_head, xy_prober, loader = train_model(device)

    # Part 3: Planning evaluation
    success_rate, mean_dist = run_planning_eval(
        jepa, xy_prober, loader, device, num_episodes=20
    )
    print(f"\nFinal Results: success_rate={success_rate:.2f}, mean_dist={mean_dist:.4f}")


if __name__ == "__main__":
    main()
