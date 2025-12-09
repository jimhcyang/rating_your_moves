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
from . import build_rym_npz

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Default: Jan..Nov 2025
DEFAULT_MONTHS_2025 = [f"2025-{m:02d}" for m in range(1, 12)]


# ---------------------------------------------------------------------
# Step 1: ensure per-month data + indexes
# ---------------------------------------------------------------------
def ensure_month_indexes(months: Sequence[str]) -> Dict[str, Tuple[Path, Path]]:
    """
    For each month, download the .pgn.zst from Lichess (if needed),
    decompress to .pgn (if needed), and build the CSV + Parquet index.

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
# Step 2: call build_balanced_pgn programmatically
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
) -> None:
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
# Step 3: call build_rym_npz for train / val / test
# ---------------------------------------------------------------------
def _run_build_rym_npz(
    pgn_path: Path,
    npz_path: Path,
    min_rating: int,
    max_rating: int,
    num_bins: int,
) -> None:
    argv = [
        "build_rym_npz",
        "--pgn-path",
        str(pgn_path),
        "--out-path",
        str(npz_path),
        "--min-rating",
        str(min_rating),
        "--max-rating",
        str(max_rating),
        "--num-bins",
        str(num_bins),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        build_rym_npz.main()
    finally:
        sys.argv = old_argv


def build_npz_for_splits(
    pgn_prefix: Path,
    npz_prefix: Path,
    min_rating: int,
    max_rating: int,
    num_bins: int,
) -> None:
    for split in ("train", "val", "test"):
        pgn_path = Path(f"{pgn_prefix}_{split}.pgn")
        npz_path = Path(f"{npz_prefix}_{split}.npz")
        print(f"[INFO] Building NPZ for {split}: {pgn_path.name} -> {npz_path.name}")
        _run_build_rym_npz(pgn_path, npz_path, min_rating, max_rating, num_bins)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end RYM data pipeline for Jan–Nov 2025:\n"
            "- downloads Lichess monthly dumps\n"
            "- builds indexes\n"
            "- constructs a balanced 10+0 PGN\n"
            "- converts to NPZ for training"
        )
    )

    parser.add_argument(
        "--months",
        nargs="+",
        default=DEFAULT_MONTHS_2025,
        help="Months to include, e.g. 2025-01 2025-02 ... (default: Jan–Nov 2025)",
    )

    parser.add_argument(
        "--time-control",
        type=str,
        default="600+0",
        help='Time control filter, e.g. "600+0" for 10+0 rapid (default: 600+0).',
    )

    parser.add_argument(
        "--min-rating",
        type=int,
        default=800,
        help="Minimum rating for banding (inclusive). Default: 800.",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=2400,
        help="Maximum rating for banding (inclusive). Default: 2400.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=100,
        help="Rating band width in Elo (default: 100).",
    )
    parser.add_argument(
        "--per-band",
        type=int,
        default=64000,
        help=(
            "Target number of games per rating band. "
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
        default=str(DATA_DIR / "rym_2025_jan_nov_tc600+0_bin100"),
        help=(
            "Prefix for balanced PGN outputs "
            '(default: data/rym_2025_jan_nov_tc600+0_bin100, which yields '
            '"..._train.pgn", "..._val.pgn", "..._test.pgn").'
        ),
    )
    parser.add_argument(
        "--npz-prefix",
        type=str,
        default=str(DATA_DIR / "rym_2025_jan_nov_tc600+0_bin100"),
        help=(
            "Prefix for NPZ outputs "
            '(default matches out-prefix; yields "..._train.npz", etc.).'
        ),
    )

    parser.add_argument(
        "--cleanup-pgn",
        action="store_true",
        help=(
            "After building balanced PGNs and NPZs, delete the original monthly "
            ".pgn files (lichess_db_standard_rated_YYYY-MM.pgn) to save disk. "
            "Indexes and final balanced PGNs/NPZs are kept."
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

    out_prefix = Path(args.out_prefix)
    npz_prefix = Path(args.npz_prefix)

    num_bins = (max_rating - min_rating) // bin_size
    if num_bins <= 0:
        raise ValueError("max_rating must be greater than min_rating by at least one bin_size.")

    print("[STEP 1] Ensuring monthly indexes exist...")
    paths_by_month = ensure_month_indexes(months)

    print("[STEP 2] Building balanced PGNs across months...")
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
    )

    print("[STEP 3] Converting balanced PGNs to NPZ...")
    build_npz_for_splits(
        pgn_prefix=out_prefix,
        npz_prefix=npz_prefix,
        min_rating=min_rating,
        max_rating=max_rating,
        num_bins=num_bins,
    )

    # Optional cleanup
    if args.cleanup_pgn or args.cleanup_zst:
        print("[STEP 4] Cleaning up monthly PGNs / ZSTs...")
        for month, (zst_path, pgn_path) in paths_by_month.items():
            if args.cleanup_pgn and pgn_path.exists():
                print(f"  [cleanup] Removing {pgn_path}")
                pgn_path.unlink()
            if args.cleanup_zst and zst_path.exists():
                print(f"  [cleanup] Removing {zst_path}")
                zst_path.unlink()

    print("[DONE] Balanced Jan–Nov 2025 10+0 dataset ready at:")
    print(f"  PGNs: {out_prefix}_train/val/test.pgn")
    print(f"  NPZ : {npz_prefix}_train/val/test.npz")


if __name__ == "__main__":
    main()