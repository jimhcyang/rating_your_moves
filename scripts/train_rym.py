#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .ply_features import NUM_PLANES  # type: ignore
from .rym_models import get_model, MODEL_TYPES  # type: ignore


log = logging.getLogger("train_rym")


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


class RYMNpzDataset(Dataset):
    """
    Simple .npz-backed dataset for Rating Your Moves.

    Expects NPZ keys:
        X       : (N, C, 8, 8) uint8
        y_bin   : (N,) int      # rating band index
        y_elo   : (N,) float32  # numeric rating (e.g. avg Elo)

    Optional:
        game_id    : (N,)
        ply_idx    : (N,)
        num_bins   : scalar
        min_rating : scalar
        max_rating : scalar
    """

    def __init__(self, npz_path: Path) -> None:
        super().__init__()
        self.path = Path(npz_path)
        data = np.load(self.path)

        self.X = data["X"]  # (N, C, 8, 8), uint8
        self.y_bin = data["y_bin"]
        self.y_elo = data["y_elo"]

        self.game_id = data.get("game_id", None)
        self.ply_idx = data.get("ply_idx", None)

        self.N = int(self.X.shape[0])

        # Metadata for rating bands
        self.num_bins = int(data.get("num_bins", 16))
        self.min_rating = float(data.get("min_rating", 0.0))
        self.max_rating = float(data.get("max_rating", 0.0))

        log.info(
            "Loaded %d samples from %s (num_bins=%d, min_rating=%.1f, max_rating=%.1f)",
            self.N,
            self.path,
            self.num_bins,
            self.min_rating,
            self.max_rating,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        x = torch.from_numpy(self.X[idx]).to(torch.float32)  # (C, 8, 8)
        yb = int(self.y_bin[idx])
        ye = float(self.y_elo[idx])

        return {
            "X": x,
            "y_bin": torch.tensor(yb, dtype=torch.long),
            "y_elo": torch.tensor(ye, dtype=torch.float32),
        }


# ---------------------------------------------------------------------
# Loss: distance-aware classification + optional regression + entropy
# ---------------------------------------------------------------------


def compute_losses(
    logits: torch.Tensor,
    rating_pred: torch.Tensor,
    y_bin: torch.Tensor,
    y_elo: torch.Tensor,
    *,
    min_rating: float,
    max_rating: float,
    alpha_reg: float,
    lambda_ent: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute a joint loss where:

      * Classification loss is the expected squared distance between
        predicted band and true band (in *band index* units).
      * Regression loss (optional) is squared error in Elo, but normalised
        by the band width so that it is also in "band units".
      * Optionally, an entropy bonus encourages non-degenerate
        probability distributions over bands.

    When alpha_reg == 0, the regression part is skipped entirely so we
    don't waste compute on it. When lambda_ent == 0 we omit the entropy
    term and recover the original distance-aware loss.
    """
    # logits: (B, num_bins)
    B, num_bins = logits.shape
    device = logits.device

    # 1) Distance-aware classification loss
    probs = F.softmax(logits, dim=1)  # (B, num_bins)

    bin_idx = torch.arange(num_bins, device=device, dtype=torch.float32).view(1, -1)  # (1, K)
    y = y_bin.to(device=device, dtype=torch.float32).view(-1, 1)  # (B, 1)

    # Squared distance (in band index units) for every (sample, bin)
    sq_dist = (bin_idx - y) ** 2  # (B, K)
    per_sample_cls = (probs * sq_dist).sum(dim=1)  # E[(k - y)^2]
    loss_cls = per_sample_cls.mean()

    # Optional entropy bonus to discourage over-confident spikes
    if lambda_ent > 0.0:
        # Small clamp to avoid log(0)
        log_probs = (probs + 1e-8).log()
        entropy = -(probs * log_probs).sum(dim=1).mean()
        # Minimise (distance - λ * entropy):
        loss_cls = loss_cls - lambda_ent * entropy

    # 2) Regression loss, measured in "number of bands"
    if alpha_reg > 0.0:
        band_width = (max_rating - min_rating) / float(num_bins)
        if band_width <= 0:
            raise ValueError(
                f"Non-positive band width: min_rating={min_rating}, "
                f"max_rating={max_rating}, num_bins={num_bins}"
            )
        y_elo = y_elo.to(device=device, dtype=torch.float32)
        rating_pred = rating_pred.to(device=device, dtype=torch.float32)
        diff_in_bands = (rating_pred - y_elo) / band_width
        loss_reg = (diff_in_bands ** 2).mean()
        loss = loss_cls + alpha_reg * loss_reg
    else:
        # Completely skip regression if alpha_reg==0
        loss_reg = torch.zeros((), device=device, dtype=loss_cls.dtype)
        loss = loss_cls

    return {
        "loss": loss,
        "loss_cls": loss_cls,
        "loss_reg": loss_reg,
    }


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    min_rating: float,
    max_rating: float,
    alpha_reg: float,
    lambda_ent: float = 0.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    n_samples = 0

    for batch in loader:
        X = batch["X"].to(device=device, dtype=torch.float32)
        y_bin = batch["y_bin"].to(device=device, dtype=torch.long)
        y_elo = batch["y_elo"].to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits, rating_pred = model(X)  # (B, K), (B,)

        losses = compute_losses(
            logits,
            rating_pred,
            y_bin,
            y_elo,
            min_rating=min_rating,
            max_rating=max_rating,
            alpha_reg=alpha_reg,
            lambda_ent=lambda_ent,
        )

        loss = losses["loss"]
        loss.backward()
        optimizer.step()

        B = X.size(0)
        total_loss += float(loss.item()) * B
        total_cls += float(losses["loss_cls"].item()) * B
        total_reg += float(losses["loss_reg"].item()) * B
        n_samples += B

    if n_samples == 0:
        raise RuntimeError("No samples seen in train_one_epoch.")

    return {
        "loss": total_loss / n_samples,
        "loss_cls": total_cls / n_samples,
        "loss_reg": total_reg / n_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    min_rating: float,
    max_rating: float,
    alpha_reg: float,
    lambda_ent: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_correct = 0
    total_samples = 0
    total_abs_err = 0.0

    # Precompute band centers for an Elo-from-bands estimate
    num_bins = None
    centers = None

    for batch in loader:
        X = batch["X"].to(device=device, dtype=torch.float32)
        y_bin = batch["y_bin"].to(device=device, dtype=torch.long)
        y_elo = batch["y_elo"].to(device=device, dtype=torch.float32)

        logits, rating_pred = model(X)

        # Discover num_bins from logits the first time through
        if num_bins is None:
            num_bins = logits.size(1)
            band_width = (max_rating - min_rating) / float(num_bins)
            centers = (
                min_rating
                + (torch.arange(num_bins, device=device, dtype=torch.float32) + 0.5)
                * band_width
            )

        losses = compute_losses(
            logits,
            rating_pred,
            y_bin,
            y_elo,
            min_rating=min_rating,
            max_rating=max_rating,
            alpha_reg=alpha_reg,
            lambda_ent=lambda_ent,
        )

        B = X.size(0)
        total_loss += float(losses["loss"].item()) * B
        total_cls += float(losses["loss_cls"].item()) * B
        total_reg += float(losses["loss_reg"].item()) * B
        total_samples += B

        # Classification accuracy
        preds_bin = logits.argmax(dim=1)
        total_correct += int((preds_bin == y_bin).sum().item())

        # Elo MAE using the distribution over bands, not necessarily the reg head
        probs = F.softmax(logits, dim=1)  # (B, num_bins)
        rating_from_bands = (probs * centers.view(1, -1)).sum(dim=1)
        total_abs_err += float(torch.abs(rating_from_bands - y_elo).sum().item())

    if total_samples == 0:
        raise RuntimeError("No samples seen in evaluate.")

    return {
        "loss": total_loss / total_samples,
        "loss_cls": total_cls / total_samples,
        "loss_reg": total_reg / total_samples,
        "acc": total_correct / total_samples,
        "mae_rating": total_abs_err / total_samples,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single RYM model on NPZ ply features.",
    )
    parser.add_argument("--train-npz", type=str, required=True)
    parser.add_argument("--val-npz", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=MODEL_TYPES,
        default="resnet",
        help="Model family (linear, mlp, cnn, resnet, conv_transformer).",
    )
    parser.add_argument(
        "--config-id",
        type=int,
        default=0,
        help="Small discrete config index (0..3) within that family.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--alpha-reg",
        type=float,
        default=0.0,
        help="Weight for regression loss vs classification loss (0.0 disables regression).",
    )
    parser.add_argument(
        "--lambda-ent",
        type=float,
        default=0.0,
        help="Entropy bonus weight in the classification loss (0.0 disables entropy term).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device string for torch.device, e.g. 'cuda', 'mps', or 'cpu'.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save final model checkpoint (.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[train_rym] %(levelname)s: %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = RYMNpzDataset(Path(args.train_npz))
    val_ds = RYMNpzDataset(Path(args.val_npz))

    num_bins = train_ds.num_bins
    if num_bins != val_ds.num_bins:
        raise ValueError(f"Train/val num_bins mismatch: {num_bins} vs {val_ds.num_bins}")

    if (train_ds.min_rating != val_ds.min_rating) or (train_ds.max_rating != val_ds.max_rating):
        raise ValueError(
            "Train/val rating range mismatch: "
            f"train=({train_ds.min_rating},{train_ds.max_rating}), "
            f"val=({val_ds.min_rating},{val_ds.max_rating})"
        )

    min_rating = train_ds.min_rating
    max_rating = train_ds.max_rating

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    log.info("Using device: %s", device)

    model = get_model(
        model_type=args.model_type,
        num_planes=NUM_PLANES,
        num_bins=num_bins,
        config_id=args.config_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics: Dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        log.info("Epoch %d/%d", epoch, args.epochs)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            min_rating=min_rating,
            max_rating=max_rating,
            alpha_reg=args.alpha_reg,
            lambda_ent=args.lambda_ent,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            min_rating=min_rating,
            max_rating=max_rating,
            alpha_reg=args.alpha_reg,
            lambda_ent=args.lambda_ent,
        )

        log.info(
            "  [train] loss=%.4f cls=%.4f reg=%.4f",
            train_metrics["loss"],
            train_metrics["loss_cls"],
            train_metrics["loss_reg"],
        )
        log.info(
            "  [val]   loss=%.4f cls=%.4f reg=%.4f acc=%.4f mae=%.2f",
            val_metrics["loss"],
            val_metrics["loss_cls"],
            val_metrics["loss_reg"],
            val_metrics["acc"],
            val_metrics["mae_rating"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)

    log.info("Best val loss: %.4f (epoch %d)", best_val_loss, best_epoch)

    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_type": args.model_type,
                "config_id": args.config_id,
                "num_planes": NUM_PLANES,
                "num_bins": num_bins,
                "min_rating": min_rating,
                "max_rating": max_rating,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
            },
            save_path,
        )
        log.info("Saved model checkpoint to %s", save_path)


if __name__ == "__main__":
    main()
