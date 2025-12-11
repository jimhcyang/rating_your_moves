#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Dict, Tuple

from .build_month_index import (
    download_month_zst,
    ensure_pgn_from_zst,
    build_index_csv,
)
from . import build_balanced_pgn
from .build_rym_npz import pgn_to_npz_shards

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Default: Jan..Apr 2025
DEFAULT_MONTHS_2025 = [f"2025-{m:02d}" for m in range(1, 5)]

# Shard sizes (games per NPZ shard)
CHUNK_GAMES_BALANCED = 10_000
CHUNK_GAMES_UNBALANCED = 10_000


# ---------------------------------------------------------------------
# Step 1: ensure per-month data + indexes
# ---------------------------------------------------------------------
def ensure_month_indexes(months: Sequence[str]) -> Dict[str, Tuple[Path, Path]]:
    """
    For each month:

      1. Download the .pgn.zst from Lichess into data/ (if needed),
      2. Decompress to .pgn (if needed),
      3. Build the CSV + Parquet index (if needed).

    Returns a mapping: month -> (zst_path, pgn_path) for optional cleanup.
    """
    paths: Dict[str, Tuple[Path, Path]] = {}
    for month in months:
        print(f"[INFO] Processing month {month}...")
        zst_path = download_month_zst(month)
        pgn_path = ensure_pgn_from_zst(zst_path)
        build_index_csv(month, pgn_path, write_parquet=True)
        paths[month] = (zst_path, pgn_path)
    return paths


# ---------------------------------------------------------------------
# Step 2: call build_balanced_pgn programmatically (BALANCED dataset)
# ---------------------------------------------------------------------
def _run_build_balanced_pgn(
    months: Sequence[str],
    out_prefix: Path,
    time_control: str,
    per_band: int,
    bin_size: int,
    min_rating: int,
    max_rating: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> None:
    """
    Wrap scripts.build_balanced_pgn.main(), constructing argv by hand.

    This builds a rating-balanced dataset over the given months, filtered
    by a specific time control (e.g. 5+0) and rating range.
    """
    where_expr = f'time_control == "{time_control}"'

    argv = [
        "build_balanced_pgn",
        *months,
        "--where",
        where_expr,
        "--rating-col",
        "avg_elo",
        "--bin-size",
        str(bin_size),
        "--min-rating",
        str(min_rating),
        "--max-rating",
        str(max_rating),
        "--per-band",
        str(per_band),
        "--train-frac",
        str(train_frac),
        "--val-frac",
        str(val_frac),
        "--test-frac",
        str(test_frac),
        "--seed",
        str(seed),
        "--out-prefix",
        str(out_prefix),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        build_balanced_pgn.main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------
# Step 3: Balanced PGNs → sharded NPZs
# ---------------------------------------------------------------------
def build_npz_shards_for_splits(
    pgn_prefix: Path,
    npz_prefix: Path,
    min_rating: int,
    max_rating: int,
    num_bins: int,
    chunk_games: int = CHUNK_GAMES_BALANCED,
) -> None:
    """
    For each split (train/val/test), convert the balanced PGN to a *set*
    of sharded NPZ files using pgn_to_npz_shards.

    For example:
        pgn_prefix = data/rym_2025_jan_apr_tc300+0_bin200

    produces for the train split:
        pgn:  data/rym_2025_jan_apr_tc300+0_bin200_train.pgn
        npz:  data/rym_2025_jan_apr_tc300+0_bin200_train_shard000.npz
              data/rym_2025_jan_apr_tc300+0_bin200_train_shard001.npz
              ...
    """
    print("[STEP 3] Converting BALANCED PGNs to sharded NPZ...")

    for split in ("train", "val", "test"):
        pgn_path = pgn_prefix.with_name(f"{pgn_prefix.name}_{split}.pgn")
        if not pgn_path.exists():
            print(f"[WARN] PGN for split '{split}' missing: {pgn_path} (skipping)")
            continue

        out_prefix = npz_prefix.with_name(f"{npz_prefix.name}_{split}")
        first_shard = out_prefix.with_name(out_prefix.name + "_shard000.npz")

        if first_shard.exists():
            print(
                f"[INFO] NPZ shards for split '{split}' already exist; "
                f"skipping sharded conversion ({first_shard})"
            )
            continue

        print(
            f"[INFO] Building NPZ shards for {split}: "
            f"{pgn_path.name} -> {out_prefix.name}_shardXXX.npz "
            f"(chunk_games={chunk_games})"
        )

        pgn_to_npz_shards(
            pgn_path=pgn_path,
            out_prefix=out_prefix,
            max_games=None,
            min_rating=min_rating,
            max_rating=max_rating,
            num_bins=num_bins,
            chunk_games=chunk_games,
        )


# ---------------------------------------------------------------------
# Step 4: build UNBALANCED “real-world” 5+0 test PGN
# ---------------------------------------------------------------------
def build_unbalanced_realtest_pgn(
    months: Sequence[str],
    time_control: str,
    min_rating: int,
    max_rating: int,
    total_games: int,
    out_pgn: Path,
    seed: int,
) -> None:
    """
    Build an *unbalanced* 5+0 test dataset:

      - Filter: given time_control (default "300+0" = 5+0 blitz)
      - Compute avg_elo from white_elo / black_elo
      - Keep games with avg_elo in [min_rating, max_rating)
      - Randomly sample total_games across *all* rating bands

    Uses build_balanced_pgn.load_indexes + write_pgn_for_index so it
    integrates with the existing index schema.
    """
    print("[unbalanced] Loading indexes for unbalanced sample...")
    df = build_balanced_pgn.load_indexes(list(months))

    where_expr = f'time_control == "{time_control}"'
    print(f"[unbalanced] Applying filter: {where_expr}")
    df = df.query(where_expr, engine="python")

    if "white_elo" not in df.columns or "black_elo" not in df.columns:
        raise ValueError(
            "Index is missing white_elo/black_elo; cannot compute avg_elo "
            "for unbalanced real-world sample."
        )

    df = df[df["white_elo"].notna() & df["black_elo"].notna()].copy()
    df["avg_elo_tmp"] = (
        df["white_elo"].astype(float) + df["black_elo"].astype(float)
    ) / 2.0

    mask_rating = (df["avg_elo_tmp"] >= min_rating) & (df["avg_elo_tmp"] < max_rating)
    df = df[mask_rating].copy()

    n_avail = len(df)
    if n_avail == 0:
        raise RuntimeError(
            "No games available for unbalanced sample after time_control + rating filters."
        )

    n_sample = min(total_games, n_avail)
    print(
        f"[unbalanced] Sampling {n_sample} games "
        f"(requested {total_games}, available {n_avail})"
    )

    df_sample = df.sample(n=n_sample, random_state=seed).copy()
    df_sample = df_sample.drop(columns=["avg_elo_tmp"])

    print(f"[unbalanced] Writing unbalanced real-world PGN to {out_pgn}")
    build_balanced_pgn.write_pgn_for_index(df_sample, out_pgn)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end RYM data pipeline for Jan–Apr 2025 (5+0 blitz):\n"
            "- downloads Lichess monthly dumps if needed\n"
            "- builds indexes if needed\n"
            "- constructs a BALANCED 5+0 PGN (400–2400, 10×200pt bands, 100k per band)\n"
            "- converts balanced splits to *sharded* NPZ for training\n"
            "- constructs an UNBALANCED 100k 'real-world' 5+0 test PGN + sharded NPZ\n"
        )
    )

    parser.add_argument(
        "--months",
        nargs="+",
        default=DEFAULT_MONTHS_2025,
        help=(
            "Months to include, e.g. 2025-01 2025-02 ... "
            "(default: Jan–Apr 2025)."
        ),
    )

    parser.add_argument(
        "--time-control",
        type=str,
        default="300+0",
        help='Time control filter, e.g. "300+0" for 5+0 blitz (default: 300+0).',
    )

    parser.add_argument(
        "--min-rating",
        type=int,
        default=400,
        help="Minimum rating for banding (inclusive). Default: 400.",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=2400,
        help="Maximum rating for banding (exclusive upper edge). Default: 2400.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=200,
        help=(
            "Rating band width in Elo (default: 200 → 400-600, 600-800, ..., "
            "2200-2400 for min=400, max=2400)."
        ),
    )
    parser.add_argument(
        "--per-band",
        type=int,
        default=100_000,
        help=(
            "Target number of games per rating band for the BALANCED dataset. "
            "With 10 bands from 400-600 to 2200-2400, this aims for 1,000,000 games. "
            "If some bands have fewer games, the sampler shrinks per-band to the "
            "minimum available across bands."
        ),
    )

    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Train fraction per band (default: 0.8).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation fraction per band (default: 0.1).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Test fraction per band (default: 0.1).",
    )

    parser.add_argument(
        "--out-prefix",
        type=str,
        default=str(DATA_DIR / "rym_2025_jan_apr_tc300+0_bin200"),
        help=(
            "Prefix for BALANCED PGN outputs. "
            'Default: data/rym_2025_jan_apr_tc300+0_bin200, which yields '
            '"..._train.pgn", "..._val.pgn", "..._test.pgn".'
        ),
    )
    parser.add_argument(
        "--npz-prefix",
        type=str,
        default=str(DATA_DIR / "rym_2025_jan_apr_tc300+0_bin200"),
        help=(
            "Prefix for BALANCED NPZ outputs. "
            "Shards will be named '<prefix>_<split>_shardXXX.npz'."
        ),
    )

    parser.add_argument(
        "--unbalanced-prefix",
        type=str,
        default=str(DATA_DIR / "rym_2025_jan_apr_tc300+0_unbalanced_realtest"),
        help=(
            "Prefix (without extension) for the UNBALANCED real-world test "
            "dataset. The script will create '<prefix>.pgn' and "
            "'<prefix>_shardXXX.npz'."
        ),
    )
    parser.add_argument(
        "--unbalanced-total",
        type=int,
        default=100_000,
        help=(
            "Total number of games in the unbalanced real-world test dataset "
            "(default: 100000)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=64,
        help="Random seed for banded sampling and unbalanced sampling.",
    )

    parser.add_argument(
        "--cleanup-pgn",
        action="store_true",
        help=(
            "After building balanced/unbalanced PGNs and NPZs, delete the "
            "original monthly .pgn files "
            "(lichess_db_standard_rated_YYYY-MM.pgn) to save disk. "
            "Indexes and final balanced/unbalanced PGNs/NPZs are kept."
        ),
    )
    parser.add_argument(
        "--cleanup-zst",
        action="store_true",
        help=(
            "After the pipeline, delete the compressed .pgn.zst archives "
            "(lichess_db_standard_rated_YYYY-MM.pgn.zst). "
            "Use this if you don't mind re-downloading in the future."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    months = args.months
    time_control = args.time_control
    min_rating = args.min_rating
    max_rating = args.max_rating
    bin_size = args.bin_size
    per_band = args.per_band
    train_frac = args.train_frac
    val_frac = args.val_frac
    test_frac = args.test_frac
    seed = args.seed

    out_prefix = Path(args.out_prefix)
    npz_prefix = Path(args.npz_prefix)
    unbalanced_prefix = Path(args.unbalanced_prefix)

    # Example: min=400, max=2400, bin_size=200 → (2400-400)/200 = 10 bands
    diff = max_rating - min_rating
    if diff <= 0:
        raise ValueError("max-rating must be greater than min-rating.")
    if diff % bin_size != 0:
        raise ValueError(
            f"(max-rating - min-rating) = {diff} is not divisible by bin-size={bin_size}; "
            "this would misalign rating bands."
        )
    num_bins = diff // bin_size

    # Balanced output PGN paths
    balanced_pgn_paths = {
        split: out_prefix.with_name(f"{out_prefix.name}_{split}.pgn")
        for split in ("train", "val", "test")
    }

    # Balanced NPZ shard prefixes and "first shard" paths
    balanced_npz_shard_prefixes = {
        split: npz_prefix.with_name(f"{npz_prefix.name}_{split}")
        for split in ("train", "val", "test")
    }
    balanced_npz_first_shards = {
        split: prefix.with_name(prefix.name + "_shard000.npz")
        for split, prefix in balanced_npz_shard_prefixes.items()
    }

    # Unbalanced output PGN and NPZ shard prefix
    unbalanced_pgn_path = unbalanced_prefix.with_suffix(".pgn")
    unbalanced_npz_prefix = unbalanced_prefix
    unbalanced_npz_first_shard = unbalanced_npz_prefix.with_name(
        unbalanced_npz_prefix.name + "_shard000.npz"
    )

    print("[STEP 1] Ensuring monthly indexes exist (download/decompress if needed)...")
    paths_by_month = ensure_month_indexes(months)

    # ------------------------------------------------------------------
    # STEP 2: Balanced PGNs (only if they don't already exist)
    # ------------------------------------------------------------------
    have_balanced_pgns = all(p.exists() for p in balanced_pgn_paths.values())
    if have_balanced_pgns:
        print("[STEP 2] Balanced PGNs already exist; skipping build_balanced_pgn.")
    else:
        print("[STEP 2] Building BALANCED PGNs across months...")
        _run_build_balanced_pgn(
            months=months,
            out_prefix=out_prefix,
            time_control=time_control,
            per_band=per_band,
            bin_size=bin_size,
            min_rating=min_rating,
            max_rating=max_rating,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # STEP 3: Balanced NPZ shards (only if they don't already exist)
    # ------------------------------------------------------------------
    have_balanced_npz = all(p.exists() for p in balanced_npz_first_shards.values())
    if have_balanced_npz:
        print("[STEP 3] Balanced NPZ shards already exist; skipping conversion.")
    else:
        build_npz_shards_for_splits(
            pgn_prefix=out_prefix,
            npz_prefix=npz_prefix,
            min_rating=min_rating,
            max_rating=max_rating,
            num_bins=num_bins,
            chunk_games=CHUNK_GAMES_BALANCED,
        )

    # ------------------------------------------------------------------
    # STEP 4: Unbalanced real-world PGN (only if it doesn't exist)
    # ------------------------------------------------------------------
    if unbalanced_pgn_path.exists():
        print(
            f"[STEP 4] Unbalanced PGN already exists at {unbalanced_pgn_path}; skipping."
        )
    else:
        print("[STEP 4] Building UNBALANCED real-world test PGN...")
        build_unbalanced_realtest_pgn(
            months=months,
            time_control=time_control,
            min_rating=min_rating,
            max_rating=max_rating,
            total_games=args.unbalanced_total,
            out_pgn=unbalanced_pgn_path,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # STEP 5: Unbalanced NPZ shards (only if they don't already exist)
    # ------------------------------------------------------------------
    if unbalanced_npz_first_shard.exists():
        print(
            f"[STEP 5] Unbalanced NPZ shards already exist at "
            f"{unbalanced_npz_first_shard}; skipping."
        )
    else:
        print("[STEP 5] Converting UNBALANCED PGN to sharded NPZ...")
        pgn_to_npz_shards(
            pgn_path=unbalanced_pgn_path,
            out_prefix=unbalanced_npz_prefix,
            max_games=None,
            min_rating=min_rating,
            max_rating=max_rating,
            num_bins=num_bins,
            chunk_games=CHUNK_GAMES_UNBALANCED,
        )

    # ------------------------------------------------------------------
    # STEP 6: Optional cleanup of raw monthly PGNs / ZSTs
    # ------------------------------------------------------------------
    if args.cleanup_pgn or args.cleanup_zst:
        print("[STEP 6] Cleaning up monthly PGNs / ZSTs...")
        for month, (zst_path, pgn_path) in paths_by_month.items():
            if args.cleanup_pgn and pgn_path.exists():
                print(f"  [cleanup] Removing {pgn_path}")
                pgn_path.unlink()
            if args.cleanup_zst and zst_path.exists():
                print(f"  [cleanup] Removing {zst_path}")
                zst_path.unlink()

    print("\n[DONE] Jan–Apr 2025 5+0 datasets ready:")
    print("  BALANCED dataset (400–2400, 10×200pt bands, ~1M games):")
    print(
        f"    PGNs: {balanced_pgn_paths['train']}, "
        f"{balanced_pgn_paths['val']}, {balanced_pgn_paths['test']}"
    )
    print("    NPZ shards (patterns per split):")
    for split in ("train", "val", "test"):
        prefix = balanced_npz_shard_prefixes[split]
        pattern = prefix.with_name(prefix.name + "_shard*.npz")
        print(f"      {split}: {pattern}")
    print("  UNBALANCED real-world test dataset:")
    print(f"    PGN:  {unbalanced_pgn_path}")
    print(
        "    NPZ shards: "
        f"{unbalanced_npz_prefix.with_name(unbalanced_npz_prefix.name + '_shard*.npz')}"
    )


if __name__ == "__main__":
    main()
