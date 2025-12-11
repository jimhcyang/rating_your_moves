#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Reuse helpers (and K_TOL_DIFF semantics) from the balanced script
from .build_balanced_pgn import (
    load_indexes,
    prepare_rating_column,
    write_pgn_for_index,
)

ROOT_DIR = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT_DIR / "data"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an *unbalanced* PGN sample from one or more Lichess "
            "monthly indexes.\n\n"
            "Compared to build_balanced_pgn, this script:\n"
            "  - uses the same loading + rating preprocessing (including\n"
            "    the K_TOL_DIFF-based rating-gap filter in avg_elo mode),\n"
            "  - but skips rating-band binning entirely,\n"
            "  - and simply draws a uniform random sample of N games.\n\n"
            "Example:\n"
            "  python -m scripts.build_unbalanced_pgn 2025-01 2025-02 2025-03 2025-04 \\\n"
            "    --where \"time_control == '300+0'\" \\\n"
            "    --rating-col avg_elo --min-rating 400 --max-rating 2400 \\\n"
            "    --total-games 100000 \\\n"
            "    --out-prefix data/rym_2025_jan_apr_tc300+0_unbalanced_realtest\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "months",
        nargs="+",
        help="One or more months in 'YYYY-MM' format, e.g. 2025-01 2025-02",
    )

    parser.add_argument(
        "--where",
        "-w",
        help=(
            "Pandas query over index columns, e.g. "
            "'time_control == \"300+0\" and has_eval and has_clock'. "
            "This is applied *before* rating preprocessing."
        ),
    )

    parser.add_argument(
        "--rating-col",
        default="avg_elo",
        help=(
            "Which rating column to use for rating filtering. "
            "Options: 'white_elo', 'black_elo', or 'avg_elo'. "
            "Default: avg_elo (average of white_elo and black_elo). "
            "In avg_elo mode, games with |white_elo - black_elo| > K_TOL_DIFF*bin_size "
            "are dropped by prepare_rating_column (same as build_balanced_pgn)."
        ),
    )

    parser.add_argument(
        "--bin-size",
        type=int,
        default=200,
        help=(
            "Bin size only matters for the K_TOL_DIFF rating-gap filter when "
            "rating-col == 'avg_elo'. With K_TOL_DIFF=1 and bin-size=200, "
            "we keep only games where |white_elo - black_elo| <= 200."
        ),
    )

    parser.add_argument(
        "--min-rating",
        type=int,
        default=400,
        help=(
            "Minimum rating for inclusion (inclusive). "
            "Applied to the chosen rating_col (or computed avg_elo). "
            "Default: 400."
        ),
    )

    parser.add_argument(
        "--max-rating",
        type=int,
        default=2400,
        help=(
            "Maximum rating for inclusion (exclusive upper edge). "
            "Applied to the chosen rating_col (or computed avg_elo). "
            "Default: 2400."
        ),
    )

    parser.add_argument(
        "--total-games",
        type=int,
        default=100_000,
        help=(
            "Total number of games to sample *after* all filters. "
            "If fewer than this are available, the script will sample all "
            "available games."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=64,
        help="Random seed for sampling.",
    )

    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help=(
            "Output prefix for the unbalanced dataset. "
            "Default: data/rym_unbalanced_<month_or_combined>. "
            "The script writes '<prefix>_index.parquet' and '<prefix>.pgn'."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    months: List[str] = args.months

    print("[unbalanced] Loading monthly indexes...")
    # 1) Load indexes (reuses build_balanced_pgn.load_indexes)
    df = load_indexes(months)

    # 2) Apply optional WHERE filter (e.g. time_control == "300+0")
    if args.where:
        print(f"[unbalanced] Applying filter: {args.where}")
        df = df.query(args.where, engine="python")
        print(f"[unbalanced] Remaining games after filter: {len(df)}")

    if df.empty:
        print("[unbalanced] No games match the given filter.")
        return

    # 3) Rating preprocessing (shared with balanced pipeline)
    df_rating, rating_col = prepare_rating_column(df, args.rating_col, args.bin_size)
    if df_rating.empty:
        print("[unbalanced] No games remain after rating preprocessing.")
        return

    # 4) Rating range filter on the chosen rating column
    min_rating = args.min_rating
    max_rating = args.max_rating

    print(
        f"[unbalanced] Applying rating range filter on '{rating_col}': "
        f"[{min_rating}, {max_rating})"
    )
    mask = (df_rating[rating_col] >= min_rating) & (df_rating[rating_col] < max_rating)
    df_filtered = df_rating[mask].copy()
    print(f"[unbalanced] Games after rating range filter: {len(df_filtered)}")

    if df_filtered.empty:
        print(
            "[unbalanced] No games remain after rating range filter. "
            "Try loosening min-rating / max-rating."
        )
        return

    # 5) Uniform random sample of N games across the remaining pool
    total_avail = len(df_filtered)
    n_sample = min(args.total_games, total_avail)
    if n_sample < args.total_games:
        print(
            f"[unbalanced] Requested {args.total_games} games but only "
            f"{total_avail} available; sampling {n_sample}."
        )
    else:
        print(f"[unbalanced] Sampling {n_sample} games from {total_avail} available.")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(df_filtered.index.to_numpy(), size=n_sample, replace=False)
    df_sample = df_filtered.loc[idx].copy()

    # 6) Decide output prefix
    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
    else:
        if len(months) == 1:
            tag = months[0]
        else:
            tag = "combined"
        out_prefix = DATA_DIR / f"rym_unbalanced_{tag}"

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_pgn = out_prefix.with_suffix(".pgn")

    # 7) Save unbalanced index (mirror build_balanced_pgn convention)
    index_path = out_prefix.parent / f"{out_prefix.name}_index.parquet"
    df_sample.to_parquet(index_path, index=False)
    print(f"[unbalanced] Saved unbalanced index to {index_path}")

    # 8) Write PGN using the existing helper (uses tqdm internally)
    write_pgn_for_index(df_sample, out_pgn)
    print(f"[unbalanced] Finished writing unbalanced PGN to {out_pgn}")


if __name__ == "__main__":
    main()
