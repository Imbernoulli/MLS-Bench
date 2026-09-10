"""Self-contained Video JEPA training script with custom prediction loss.

Trains a JEPA model on Moving MNIST and evaluates detection Average Precision.
The CustomPredictionLoss class is the editable component that the agent modifies.
"""
import os
import sys; sys.path = [p for p in sys.path if not os.path.isfile(os.path.join(p, 'logging.py'))]
import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import (
    DetHead,
    Projector,
    ResNet5,
    ResUNet,
    StateOnlyPredictor,
)
from eb_jepa.datasets.moving_mnist import MovingMNISTDet
from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import VCLoss

# ==============================================================================
# EDITABLE REGION START
# ==============================================================================


class CustomPredictionLoss(nn.Module):
    """Prediction cost function for temporal JEPA.

    Measures discrepancy between predicted and target representations
    in the latent space. Used to train the predictor network.

    Args:
        state: [B, C, T, H, W] target encoded representations
        predicted: [B, C, T, H, W] predicted representations

    Returns:
        Scalar loss tensor
    """

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        return torch.tensor(0.0, device=state.device, requires_grad=True)


# ==============================================================================
# EDITABLE REGION END
# ==============================================================================
# ============================================================================
# FIXED REGION (do not modify below this line)
# ============================================================================


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def validation_loop(val_loader, jepa, detection_head, pixel_decoder, steps, device):
    """Run validation and compute detection AP metrics."""
    jepa.eval()
    detection_head.eval()
    pixel_decoder.eval()

    metrics = collections.defaultdict(list)
    for batch in tqdm(val_loader, desc="Validation"):
        batch = {k: v.to(device) for k, v in batch.items()}
        x = batch["video"]
        loc_map = batch["digit_location"]

        recon_loss = pixel_decoder(x, x)
        det_loss = detection_head(x, loc_map)

        logs = {
            "val/recon_loss": float(recon_loss.item()),
            "val/det_loss": float(det_loss.item()),
        }
        for k, v in logs.items():
            metrics[k].append(v)

        T = x.shape[2]
        preds, _ = jepa.unroll(
            x,
            actions=None,
            nsteps=T - 2,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )
        scores = detection_head.head.score(preds, loc_map[:, 2:])
        for s, score in enumerate(scores):
            metrics[f"AP_{s}"].append(float(score))

    # Aggregate results
    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}

    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    return metrics


def main():
    """Train Video JEPA with custom prediction loss on Moving MNIST."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    seed = int(os.environ.get("SEED", "42"))
    seed_everything(seed)

    # Hyperparameters
    epochs = 50
    lr = 1e-3
    steps = 4
    dobs = 1

    # Model size from environment (small / base / large)
    model_size = os.environ.get("MODEL_SIZE", "base")
    _MODEL_CONFIGS = {
        "small": {"henc": 16, "dstc": 8,  "hpre": 16, "batch_size": 64},
        "base":  {"henc": 32, "dstc": 16, "hpre": 32, "batch_size": 32},
        "large": {"henc": 64, "dstc": 32, "hpre": 64, "batch_size": 16},
    }
    cfg = _MODEL_CONFIGS[model_size]
    henc, dstc, hpre = cfg["henc"], cfg["dstc"], cfg["hpre"]
    batch_size = cfg["batch_size"]
    print(f"Model size: {model_size} (henc={henc}, dstc={dstc}, hpre={hpre}, bs={batch_size})", flush=True)

    # Load datasets
    print("Loading Moving MNIST dataset...", flush=True)
    train_set = MovingMNISTDet(split="train")
    val_set = MovingMNISTDet(split="val")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2,
        worker_init_fn=seed_worker, generator=make_generator(seed)
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2,
        worker_init_fn=seed_worker, generator=make_generator(seed + 1)
    )
    print(
        f"Dataset loaded: {len(train_set)} train, {len(val_set)} val samples",
        flush=True,
    )

    # Initialize model components
    print("Initializing model...", flush=True)
    encoder = ResNet5(dobs, henc, dstc)
    predictor_model = ResUNet(2 * dstc, hpre, dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{dstc}-{dstc * 4}-{dstc * 4}")
    regularizer = VCLoss(std_coeff=10, cov_coeff=100, proj=projector)

    # Use CustomPredictionLoss instead of SquareLossSeq
    ploss = CustomPredictionLoss()
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)

    # Initialize decoder and detection head (for evaluation only)
    decoder = ImageDecoder(dstc, dobs, hidden_dim=dstc)
    dethead = DetHead(dstc, hpre, dobs)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)
    detection_head = JEPAProbe(jepa, dethead, nn.BCELoss()).to(device)

    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    optimizer = Adam(
        [
            {"params": jepa.parameters(), "lr": lr},
            {"params": pixel_decoder.head.parameters(), "lr": lr / 10},
            {"params": detection_head.head.parameters(), "lr": lr},
        ]
    )

    # Training loop
    print(f"Starting training for {epochs} epochs...", flush=True)

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch["video"]
            loc_map = batch["digit_location"]

            optimizer.zero_grad()
            _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
                x,
                actions=None,
                nsteps=steps,
                unroll_mode="parallel",
                compute_loss=True,
                return_all_steps=False,
            )
            recon_loss = pixel_decoder(x, x)
            det_loss = detection_head(x, loc_map)
            total_loss = jepa_loss + recon_loss + det_loss

            total_loss.backward()
            optimizer.step()

            pbar.set_postfix(
                {
                    "loss": f"{jepa_loss.item():.4f}",
                    "vc": f"{regl.item():.4f}",
                    "pred": f"{pl.item():.4f}",
                }
            )

        # Print training metrics every epoch
        print(
            f"TRAIN_METRICS epoch={epoch} "
            f"loss={jepa_loss.item():.6f} "
            f"vc_loss={regl.item():.6f} "
            f"pred_loss={pl.item():.6f}",
            flush=True,
        )

        # Validation every 5 epochs and at last epoch
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_metrics = validation_loop(
                val_loader, jepa, detection_head, pixel_decoder, steps, device
            )

            # Compute mean detection AP across timesteps
            ap_keys = [k for k in val_metrics if k.startswith("AP_")]
            if ap_keys:
                mean_ap = np.mean([val_metrics[k] for k in ap_keys])
            else:
                mean_ap = 0.0

            print(
                f"Validation epoch={epoch}: "
                f"recon_loss={val_metrics.get('val/recon_loss', 0):.4f} "
                f"det_loss={val_metrics.get('val/det_loss', 0):.4f} "
                f"mean_detection_ap={mean_ap:.4f}",
                flush=True,
            )

    # Final evaluation
    print("\nRunning final evaluation...", flush=True)
    val_metrics = validation_loop(
        val_loader, jepa, detection_head, pixel_decoder, steps, device
    )

    ap_keys = [k for k in val_metrics if k.startswith("AP_")]
    if ap_keys:
        mean_ap = np.mean([val_metrics[k] for k in ap_keys])
    else:
        mean_ap = 0.0

    print(f"TEST_METRICS: mean_detection_ap={mean_ap:.4f}", flush=True)
    print("Training complete!", flush=True)


if __name__ == "__main__":
    main()
