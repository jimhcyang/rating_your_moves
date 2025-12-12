#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import gc
from pathlib import Path
from typing import List, Sequence, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_rym import RYMNpzDataset, train_one_epoch, evaluate  # type: ignore
from .rym_models import get_model, MODEL_TYPES  # type: ignore
from .ply_features import NUM_PLANES  # type: ignore


log = logging.getLogger("run_rym_experiments")


# ---------------------------------------------------------------------
# Shard discovery helpers
# ---------------------------------------------------------------------


def _parse_shard_indices(spec: str) -> List[int]:
    """
    Helper for parsing shard ranges.

    Examples:
      "all"     -> special value handled elsewhere
      "0"       -> [0]
      "0,1,5"   -> [0, 1, 5]
    """
    spec = spec.strip()
    if spec.lower() == "all":
        raise ValueError("Internal: 'all' should be handled by discover_shards.")
    if not spec:
        return []
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid shard index '{part}' in spec '{spec}'.")
    return sorted(set(out))


def discover_shards(prefix: str, shard_spec: str) -> List[Path]:
    """
    Given a prefix like:
        data/rym_2025_jan_apr_tc300+0_bin200_train_shard

    and a shard_spec of either:
        "all"          → discover shard000, shard001, ... until missing
        "0" or "0,1"   → return exactly those

    return a sorted list of NPZ shard paths.
    """
    base = Path(prefix)
    paths: List[Path] = []

    if shard_spec.lower() == "all":
        idx = 0
        while True:
            p = base.with_name(f"{base.name}{idx:03d}.npz")
            if not p.exists():
                break
            paths.append(p)
            idx += 1
    else:
        indices = _parse_shard_indices(shard_spec)
        for idx in indices:
            p = base.with_name(f"{base.name}{idx:03d}.npz")
            if not p.exists():
                raise FileNotFoundError(f"Requested shard does not exist: {p}")
            paths.append(p)

    if not paths:
        log.warning("No shards found for prefix=%s spec=%s", prefix, shard_spec)
    else:
        log.info("Discovered %d shards for prefix=%s spec=%s", len(paths), prefix, shard_spec)
    return sorted(paths)


# ---------------------------------------------------------------------
# Meta-data extraction
# ---------------------------------------------------------------------


def infer_rating_metadata(shard_paths: Sequence[Path]) -> Dict[str, Any]:
    """
    Load the first available shard and read num_bins / rating range.
    Ensures all shards share the same metadata.
    """
    if not shard_paths:
        raise ValueError("infer_rating_metadata called with empty shard_paths.")

    first_ds = RYMNpzDataset(shard_paths[0])
    num_bins = first_ds.num_bins
    min_rating = first_ds.min_rating
    max_rating = first_ds.max_rating

    for p in shard_paths[1:]:
        ds = RYMNpzDataset(p)
        if ds.num_bins != num_bins:
            raise ValueError(f"num_bins mismatch across shards: {num_bins} vs {ds.num_bins} (in {p})")
        if ds.min_rating != min_rating or ds.max_rating != max_rating:
            raise ValueError(
                "Rating range mismatch across shards: "
                f"({min_rating},{max_rating}) vs ({ds.min_rating},{ds.max_rating}) (in {p})"
            )

    return {
        "num_bins": num_bins,
        "min_rating": min_rating,
        "max_rating": max_rating,
    }


# ---------------------------------------------------------------------
# Training / evaluation over shards
# ---------------------------------------------------------------------


def train_over_shards(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    shard_paths: Sequence[Path],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    min_rating: float,
    max_rating: float,
    alpha_reg: float,
    lambda_ent: float,
) -> Dict[str, float]:
    """
    Run one logical "epoch" over all train shards.
    """
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_samples = 0

    for shard_path in shard_paths:
        log.info("  [train] shard %s", shard_path.name)
        ds = RYMNpzDataset(shard_path)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            device=device,
            min_rating=min_rating,
            max_rating=max_rating,
            alpha_reg=alpha_reg,
            lambda_ent=lambda_ent,
        )

        n = len(ds)
        total_loss += metrics["loss"] * n
        total_cls += metrics["loss_cls"] * n
        total_reg += metrics["loss_reg"] * n
        total_samples += n

        del ds, loader
        gc.collect()

    if total_samples == 0:
        raise RuntimeError("No train samples found across shards.")

    return {
        "loss": total_loss / total_samples,
        "loss_cls": total_cls / total_samples,
        "loss_reg": total_reg / total_samples,
    }


def evaluate_over_shards(
    model: torch.nn.Module,
    shard_paths: Sequence[Path],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    min_rating: float,
    max_rating: float,
    alpha_reg: float,
    lambda_ent: float,
) -> Dict[str, float]:
    """
    Evaluate a model over one or more shards, aggregating metrics
    weighted by number of samples.
    """
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_correct = 0
    total_samples = 0
    total_abs_err = 0.0

    for shard_path in shard_paths:
        log.info("  [eval] shard %s", shard_path.name)
        ds = RYMNpzDataset(shard_path)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        metrics = evaluate(
            model,
            loader,
            device=device,
            min_rating=min_rating,
            max_rating=max_rating,
            alpha_reg=alpha_reg,
            lambda_ent=lambda_ent,
        )

        n = len(ds)
        total_loss += metrics["loss"] * n
        total_cls += metrics["loss_cls"] * n
        total_reg += metrics["loss_reg"] * n
        total_correct += metrics["acc"] * n
        total_abs_err += metrics["mae_rating"] * n
        total_samples += n

        del ds, loader
        gc.collect()

    if total_samples == 0:
        raise RuntimeError("No evaluation samples found across shards.")

    return {
        "loss": total_loss / total_samples,
        "loss_cls": total_cls / total_samples,
        "loss_reg": total_reg / total_samples,
        "acc": total_correct / total_samples,
        "mae_rating": total_abs_err / total_samples,
    }


# ---------------------------------------------------------------------
# CLI + main training loop
# ---------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sharded RYM experiment runner.\n\n"
            "Assumes you have already built NPZ shards named like:\n"
            "  rym_2025_jan_apr_tc300+0_bin200_train_shard000.npz\n"
            "  rym_2025_jan_apr_tc300+0_bin200_val_shard000.npz\n"
            "  rym_2025_jan_apr_tc300+0_bin200_test_shard000.npz\n"
            "  rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard000.npz\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model config
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

    # Optimisation
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
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

    # Shard prefixes + specs
    parser.add_argument(
        "--train-prefix",
        type=str,
        default=str(Path("data") / "rym_2025_jan_apr_tc300+0_bin200_train_shard"),
        help=(
            "Prefix for BALANCED train shards. "
            "Shards are expected as '<prefix><NNN>.npz'."
        ),
    )
    parser.add_argument(
        "--val-prefix",
        type=str,
        default=str(Path("data") / "rym_2025_jan_apr_tc300+0_bin200_val_shard"),
        help="Prefix for BALANCED val shards.",
    )
    parser.add_argument(
        "--test-prefix",
        type=str,
        default=str(Path("data") / "rym_2025_jan_apr_tc300+0_bin200_test_shard"),
        help="Prefix for BALANCED held-out test shards.",
    )
    parser.add_argument(
        "--realtest-prefix",
        type=str,
        default=str(Path("data") / "rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard"),
        help="Prefix for UNBALANCED real-world test shards.",
    )

    parser.add_argument(
        "--train-shards",
        type=str,
        default="all",
        help="Train shard indices: 'all' or comma-separated list like '0,1,2'.",
    )
    parser.add_argument(
        "--val-shards",
        type=str,
        default="all",
        help="Val shard indices: 'all' or comma-separated list like '0,1'.",
    )
    parser.add_argument(
        "--test-shards",
        type=str,
        default="all",
        help="Test shard indices: 'all' or comma-separated list like '0,1'.",
    )
    parser.add_argument(
        "--realtest-shards",
        type=str,
        default="all",
        help="Real-world test shard indices: 'all' or comma-separated list like '0,1'.",
    )

    # Checkpointing
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="rym_experiment_ckpt.pt",
        help="Path to save best model checkpoint + metrics.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[run_rym_experiments] %(levelname)s: %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Discover shard paths according to the user's spec.
    train_shards = discover_shards(args.train_prefix, args.train_shards)
    val_shards = discover_shards(args.val_prefix, args.val_shards)
    test_shards = discover_shards(args.test_prefix, args.test_shards)
    realtest_shards = discover_shards(args.realtest_prefix, args.realtest_shards)

    if not train_shards:
        raise RuntimeError("No train shards discovered; cannot run experiment.")
    if not val_shards:
        raise RuntimeError("No val shards discovered; cannot run experiment.")

    # Infer num_bins / rating range and ensure consistency across all shards we will touch.
    meta = infer_rating_metadata(train_shards + val_shards + test_shards + realtest_shards)
    num_bins = int(meta["num_bins"])
    min_rating = float(meta["min_rating"])
    max_rating = float(meta["max_rating"])

    log.info(
        "Rating meta: num_bins=%d, min_rating=%.1f, max_rating=%.1f",
        num_bins,
        min_rating,
        max_rating,
    )

    device = torch.device(args.device)
    log.info("Using device: %s", device)

    # Build model + optimiser
    model = get_model(
        model_type=args.model_type,
        num_planes=NUM_PLANES,
        num_bins=num_bins,
        config_id=args.config_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Main training loop
    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics: Dict[str, Any] = {}
    best_test_metrics: Dict[str, Any] = {}
    best_realtest_metrics: Dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        log.info("Epoch %d/%d", epoch, args.epochs)

        train_metrics = train_over_shards(
            model,
            optimizer,
            train_shards,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            min_rating=min_rating,
            max_rating=max_rating,
            alpha_reg=args.alpha_reg,
            lambda_ent=args.lambda_ent,
        )

        val_metrics = evaluate_over_shards(
            model,
            val_shards,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
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

            # When we get a new best val, also evaluate on test / realtest shards (if any)
            if test_shards:
                log.info("Evaluating on BALANCED held-out test shards...")
                best_test_metrics = evaluate_over_shards(
                    model,
                    test_shards,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    min_rating=min_rating,
                    max_rating=max_rating,
                    alpha_reg=args.alpha_reg,
                    lambda_ent=args.lambda_ent,
                )
                log.info(
                    "  [test]  loss=%.4f cls=%.4f reg=%.4f acc=%.4f mae=%.2f",
                    best_test_metrics["loss"],
                    best_test_metrics["loss_cls"],
                    best_test_metrics["loss_reg"],
                    best_test_metrics["acc"],
                    best_test_metrics["mae_rating"],
                )

            if realtest_shards:
                log.info("Evaluating on UNBALANCED real-world test shards...")
                best_realtest_metrics = evaluate_over_shards(
                    model,
                    realtest_shards,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    min_rating=min_rating,
                    max_rating=max_rating,
                    alpha_reg=args.alpha_reg,
                    lambda_ent=args.lambda_ent,
                )
                log.info(
                    "  [real]  loss=%.4f cls=%.4f reg=%.4f acc=%.4f mae=%.2f",
                    best_realtest_metrics["loss"],
                    best_realtest_metrics["loss_cls"],
                    best_realtest_metrics["loss_reg"],
                    best_realtest_metrics["acc"],
                    best_realtest_metrics["mae_rating"],
                )

    log.info("Best val loss: %.4f (epoch %d)", best_val_loss, best_epoch)

    # Save best checkpoint
    ckpt_path = Path(args.ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
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
            "test_metrics": best_test_metrics,
            "realtest_metrics": best_realtest_metrics,
            "alpha_reg": args.alpha_reg,
            "lambda_ent": args.lambda_ent,
        },
        ckpt_path,
    )
    log.info("Saved checkpoint + metrics to %s", ckpt_path)


if __name__ == "__main__":
    main()
