#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .build_rym_npz import pgn_to_npz  # type: ignore
from .train_rym import RYMNpzDataset  # type: ignore
from .rym_models import get_model, MODEL_TYPES  # type: ignore
from .ply_features import NUM_PLANES  # type: ignore


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def auto_device() -> str:
    """
    Pick a default device.
    We prefer CUDA (GPU) if available, then MPS on Apple Silicon, then CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    # Guard against torch.backends.mps not existing on non-Apple builds
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_npz_for_split(
    split: str,
    pgn_prefix: Path,
    npz_prefix: Path,
    min_rating: int,
    max_rating: int,
    num_bins: int,
    force_rebuild: bool = False,
) -> Path:
    """
    For a split in {train,val,test}, ensure we have an NPZ file.

    PGN is assumed at:  pgn_prefix + f"_{split}.pgn"
    NPZ will be at:     npz_prefix + f"_{split}.npz"
    """
    log = logging.getLogger("run_rym_experiments")
    pgn_path = Path(f"{pgn_prefix}_{split}.pgn")
    npz_path = Path(f"{npz_prefix}_{split}.npz")

    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN for split '{split}' not found: {pgn_path}")

    if npz_path.exists() and not force_rebuild:
        log.info("Found existing NPZ for %s: %s (skipping build)", split, npz_path)
        return npz_path

    log.info("Building NPZ for %s from %s → %s", split, pgn_path, npz_path)
    pgn_to_npz(
        pgn_path=pgn_path,
        out_path=npz_path,
        max_games=None,
        min_rating=min_rating,
        max_rating=max_rating,
        num_bins=num_bins,
    )
    return npz_path


def build_loaders(
    train_npz: Path,
    val_npz: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Build DataLoaders for train/val splits and return num_bins.
    """
    train_ds = RYMNpzDataset(train_npz)
    val_ds = RYMNpzDataset(val_npz)

    num_bins = train_ds.num_bins
    if num_bins != val_ds.num_bins:
        raise ValueError(
            f"Train/val num_bins mismatch: {num_bins} vs {val_ds.num_bins}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_bins


def train_one_epoch_tqdm(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha_reg: float,
) -> Dict[str, float]:
    """
    Training loop with tqdm over batches.
    """
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    n_samples = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        X = batch["X"].to(device)
        y_bin = batch["y_bin"].to(device)
        y_elo = batch["y_elo"].to(device)

        logits, rating_pred = model(X)
        loss_cls = F.cross_entropy(logits, y_bin)
        loss_reg = F.mse_loss(rating_pred, y_elo)
        loss = loss_cls + alpha_reg * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        B = X.size(0)
        total_loss += float(loss.item()) * B
        total_cls += float(loss_cls.item()) * B
        total_reg += float(loss_reg.item()) * B
        n_samples += B

    return {
        "loss": total_loss / n_samples,
        "loss_cls": total_cls / n_samples,
        "loss_reg": total_reg / n_samples,
    }


@torch.no_grad()
def evaluate_tqdm(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha_reg: float,
) -> Dict[str, float]:
    """
    Evaluation loop with tqdm over batches.
    """
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_correct = 0
    total_samples = 0
    total_abs_err = 0.0

    for batch in tqdm(loader, desc="  val", leave=False):
        X = batch["X"].to(device)
        y_bin = batch["y_bin"].to(device)
        y_elo = batch["y_elo"].to(device)

        logits, rating_pred = model(X)
        loss_cls = F.cross_entropy(logits, y_bin)
        loss_reg = F.mse_loss(rating_pred, y_elo)
        loss = loss_cls + alpha_reg * loss_reg

        B = X.size(0)
        total_loss += float(loss.item()) * B
        total_cls += float(loss_cls.item()) * B
        total_reg += float(loss_reg.item()) * B

        preds_bin = logits.argmax(dim=1)
        total_correct += int((preds_bin == y_bin).sum().item())
        total_samples += B

        total_abs_err += float(torch.abs(rating_pred - y_elo).sum().item())

    return {
        "loss": total_loss / total_samples,
        "loss_cls": total_cls / total_samples,
        "loss_reg": total_reg / total_samples,
        "acc": total_correct / total_samples,
        "mae_rating": total_abs_err / total_samples,
    }


def parse_model_list(s: str) -> List[str]:
    s = s.strip()
    if s.lower() == "all":
        return list(MODEL_TYPES)
    names = [x.strip() for x in s.split(",") if x.strip()]
    for name in names:
        if name not in MODEL_TYPES:
            raise ValueError(f"Unknown model type '{name}', valid: {MODEL_TYPES}")
    return names


def parse_config_ids(spec: str) -> List[int]:
    """Parse config-id spec into a sorted list of unique ints.

    Examples:
      '0'       -> [0]
      '0,2,3'   -> [0, 2, 3]
      'all'     -> [0, 1, 2, 3]
    """
    spec = spec.strip()
    if not spec:
        return [0]
    if spec.lower() == "all":
        return [0, 1, 2, 3]

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    cfgs: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise ValueError(
                f"Invalid config-id '{p}', expected integers 0..3 or 'all'"
            ) from exc
        if v < 0 or v > 3:
            raise ValueError(f"config-id out of range: {v}, expected 0..3")
        cfgs.append(v)

    return sorted(set(cfgs))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end RYM runner: ensure NPZs exist, then train "
            "config(s) for one or more model families.\n\n"
            "Example:\n"
            "  python -m scripts.run_rym_experiments \\\n"
            "    --pgn-prefix data/rym_2017-04_bin_1000 \\\n"
            "    --npz-prefix data/rym_2017-04 \\\n"
            "    --min-rating 800 --max-rating 2300 --num-bins 15 \\\n"
            "    --models all --config-id all --epochs 5 --batch-size 256\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pgn-prefix",
        type=str,
        required=True,
        help="Prefix for balanced PGNs, e.g. data/rym_2017-04_bin_1000",
    )
    parser.add_argument(
        "--npz-prefix",
        type=str,
        required=True,
        help="Prefix for NPZ files, e.g. data/rym_2017-04",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=800,
        help="Minimum rating edge for bands (inclusive).",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=2300,
        help="Maximum rating edge for bands (exclusive).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of rating bands between min-rating and max-rating.",
    )
    parser.add_argument(
        "--force-rebuild-npz",
        action="store_true",
        help="Force rebuilding NPZ even if it already exists.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of model families, or 'all'. "
             f"Options: {MODEL_TYPES}",
    )
    parser.add_argument(
        "--config-id",
        type=str,
        default="0",
        help="Config index spec: '0', '0,2,3', or 'all' for [0,1,2,3].",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--alpha-reg",
        type=float,
        default=1.0,
        help="Weight for regression loss vs classification loss.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=auto_device(),
        help="Device to use: 'cuda', 'mps', or 'cpu'. Defaults to a sensible choice.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional directory to save model checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[run_rym_experiments] %(levelname)s: %(message)s",
    )
    log = logging.getLogger("run_rym_experiments")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pgn_prefix = Path(args.pgn_prefix)
    npz_prefix = Path(args.npz_prefix)

    # 1) Ensure NPZs exist for train/val (test NPZ optional here)
    train_npz = ensure_npz_for_split(
        "train",
        pgn_prefix=pgn_prefix,
        npz_prefix=npz_prefix,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        num_bins=args.num_bins,
        force_rebuild=args.force_rebuild_npz,
    )
    val_npz = ensure_npz_for_split(
        "val",
        pgn_prefix=pgn_prefix,
        npz_prefix=npz_prefix,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        num_bins=args.num_bins,
        force_rebuild=args.force_rebuild_npz,
    )

    # (Optional) ensure test NPZ too, for later evaluation
    try:
        ensure_npz_for_split(
            "test",
            pgn_prefix=pgn_prefix,
            npz_prefix=npz_prefix,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            num_bins=args.num_bins,
            force_rebuild=args.force_rebuild_npz,
        )
    except FileNotFoundError:
        log.warning("No test PGN found; skipping test NPZ build.")

    # 2) Build loaders
    train_loader, val_loader, num_bins = build_loaders(
        train_npz,
        val_npz,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = torch.device(args.device)
    log.info("Using device: %s", device)

    # 3) Train each requested model family / config pair
    model_types: Sequence[str] = parse_model_list(args.models)
    config_ids: List[int] = parse_config_ids(args.config_id)

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    for model_type in model_types:
        for cfg in config_ids:
            log.info("=" * 80)
            log.info(
                "Training model_type=%s, config_id=%d, num_bins=%d",
                model_type,
                cfg,
                num_bins,
            )
            log.info("=" * 80)

            model = get_model(
                model_type=model_type,
                num_planes=NUM_PLANES,
                num_bins=num_bins,
                config_id=cfg,
            ).to(device)

            log.info("Model:\n%s", model)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                log.info("Epoch %d/%d", epoch, args.epochs)

                train_metrics = train_one_epoch_tqdm(
                    model,
                    train_loader,
                    optimizer,
                    device=device,
                    alpha_reg=args.alpha_reg,
                )
                val_metrics = evaluate_tqdm(
                    model,
                    val_loader,
                    device=device,
                    alpha_reg=args.alpha_reg,
                )

                log.info(
                    "  train: loss=%.4f (cls=%.4f, reg=%.4f)",
                    train_metrics["loss"],
                    train_metrics["loss_cls"],
                    train_metrics["loss_reg"],
                )
                log.info(
                    "  val  : loss=%.4f (cls=%.4f, reg=%.4f), acc=%.3f, MAE=%.1f",
                    val_metrics["loss"],
                    val_metrics["loss_cls"],
                    val_metrics["loss_reg"],
                    val_metrics["acc"],
                    val_metrics["mae_rating"],
                )

            if save_dir is not None:
                save_path = save_dir / f"rym_{model_type}_cfg{cfg}.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_type": model_type,
                        "config_id": cfg,
                        "num_planes": NUM_PLANES,
                        "num_bins": num_bins,
                    },
                    save_path,
                )
                log.info("Saved checkpoint to %s", save_path)


if __name__ == "__main__":
    main()

"""
python -m scripts.run_rym_experiments \
--pgn-prefix data/rym_2017-04_bin_1000 \
--npz-prefix data/rym_2017-04 \
--min-rating 800 --max-rating 2300 --num-bins 15 \
--models all \
--config-id all \
--epochs 5 \
--batch-size 256 \
--device cuda \
--save-dir models/rym_2017-04_baselines
"""