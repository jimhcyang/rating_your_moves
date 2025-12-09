#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse helpers from index_stats for loading indexes + rating bands
from .index_stats import load_index_for_month, make_rating_bins  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT_DIR / "data"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a rating-balanced PGN dataset from one or more Lichess "
            "monthly indexes.\n\n"
            "Example:\n"
            "  python -m scripts.build_balanced_pgn 2017-04 \\\n"
            "    --where 'has_eval and has_clock' \\\n"
            "    --rating-col avg_elo --bin-size 100 --per-band 5000 \\\n"
            "    --train-frac 0.8 --val-frac 0.1 --test-frac 0.1 \\\n"
            "    --out-prefix data/rym_2017-04_eval_clock\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "months",
        nargs="+",
        help="One or more months in 'YYYY-MM' format, e.g. 2017-04 2017-05",
    )
    parser.add_argument(
        "--where",
        "-w",
        help=(
            "Pandas query over index columns, e.g. "
            "'result == \"1-0\" and has_eval and has_clock'"
        ),
    )
    parser.add_argument(
        "--rating-col",
        default="avg_elo",
        help=(
            "Which rating column to use for rating bands. "
            "Options: 'white_elo', 'black_elo', or 'avg_elo'. "
            "Default: avg_elo (average of white_elo and black_elo)."
        ),
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=100,
        help="Rating band width in Elo (e.g. 100 → 1000-1100, 1100-1200, ...).",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=None,
        help="Minimum rating for bands (default: inferred from data).",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=None,
        help="Maximum rating for bands (default: inferred from data).",
    )
    parser.add_argument(
        "--per-band",
        type=int,
        default=None,
        help=(
            "Number of games per rating band. If omitted, uses the minimum "
            "available across all bands. If provided together with "
            "--drop-underfull-bands, bands with fewer than this many games "
            "are skipped."
        ),
    )
    parser.add_argument(
        "--drop-underfull-bands",
        action="store_true",
        help=(
            "If set and --per-band is specified, drop rating bands that have "
            "fewer than --per-band games instead of shrinking per-band."
        ),
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of games per band to allocate to train split.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of games per band to allocate to validation split.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of games per band to allocate to test split.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Do not split into train/val/test; write a single balanced PGN.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and splitting.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help=(
            "Output prefix for index + PGN files. "
            "Default: data/rym_balanced_<month_or_combined>"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------


def load_indexes(months: List[str]) -> pd.DataFrame:
    """
    Load indexes for all months and concatenate.

    Uses index_stats.load_index_for_month to reuse Parquet/CSV logic.
    Each DF already gets a 'month' column inside load_index_for_month.
    """
    dfs: List[pd.DataFrame] = []
    for m in months:
        df_m, _ = load_index_for_month(m)
        dfs.append(df_m)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[balance] Loaded {len(df)} total games across {len(months)} month(s).")
    return df


def prepare_rating_column(
    df: pd.DataFrame, rating_col: str, bin_size: int
) -> Tuple[pd.DataFrame, str]:
    """
    Prepare the rating column to be used for banding.

    - If rating_col == 'avg_elo', require both white_elo and black_elo, compute
      avg_elo, and drop games with huge rating gaps (>|2*bin_size|).
    - Else, just drop rows where rating_col is NaN.
    """
    df = df.copy()

    if rating_col == "avg_elo":
        if "white_elo" not in df.columns or "black_elo" not in df.columns:
            raise ValueError(
                "Cannot use rating_col='avg_elo' without 'white_elo' and 'black_elo'."
            )
        mask_wb = df["white_elo"].notna() & df["black_elo"].notna()
        dropped = int((~mask_wb).sum())
        if dropped > 0:
            print(f"[balance] Dropping {dropped} games with missing white/black Elo.")
        df = df[mask_wb].copy()

        df["avg_elo"] = (
            df["white_elo"].astype(float) + df["black_elo"].astype(float)
        ) / 2.0

        # Drop extreme rating differences like in index_stats
        rating_diff = (
            df["white_elo"].astype(float) - df["black_elo"].astype(float)
        ).abs()
        max_diff = 2 * bin_size
        mask_diff_ok = rating_diff <= max_diff
        dropped_diff = int((~mask_diff_ok).sum())
        if dropped_diff > 0:
            print(
                f"[balance] Dropping {dropped_diff} games with |white_elo - black_elo| > {max_diff}."
            )
        df = df[mask_diff_ok].copy()

        return df, "avg_elo"

    # Single-side rating
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in index.")
    mask_rating = df[rating_col].notna()
    dropped = int((~mask_rating).sum())
    if dropped > 0:
        print(
            f"[balance] Dropping {dropped} games with missing ratings for '{rating_col}'."
        )
    df = df[mask_rating].copy()
    return df, rating_col


def sample_balanced_by_band(
    df: pd.DataFrame,
    rating_col: str,
    band_labels: List[str],
    per_band: int | None,
    seed: int,
    drop_underfull_bands: bool,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Given a DF with a 'rating_band' column and band_labels in numeric order,
    sample per_band games from each band.

    If per_band is None:
        - per_band = min(counts across all bands with at least 1 game).

    If per_band is not None and drop_underfull_bands=False:
        - If per_band > min_available, shrink per_band to min_available
          (old behaviour).

    If per_band is not None and drop_underfull_bands=True:
        - Keep per_band as requested.
        - Drop/skips bands with fewer than per_band games.

    Returns the balanced DF and a dict of band -> sampled count.
    """
    counts = df["rating_band"].value_counts()
    counts = counts.reindex(band_labels, fill_value=0)

    print("\n[balance] Band counts before sampling:")
    for band, c in counts.items():
        print(f"  {band}: {c}")

    # Decide per_band
    if per_band is None:
        # Use the smallest non-zero band as cap
        non_zero_counts = counts[counts > 0]
        if non_zero_counts.empty:
            raise ValueError("[balance] No non-empty rating bands found.")
        per_band = int(non_zero_counts.min())
        print(f"[balance] Using per-band size = min count across bands = {per_band}")
        effective_drop_underfull = False
    else:
        min_available = int(counts[counts > 0].min()) if (counts > 0).any() else 0
        if drop_underfull_bands:
            print(
                f"[balance] Requested per-band={per_band} with --drop-underfull-bands; "
                "bands with fewer games will be skipped."
            )
            effective_drop_underfull = True
        else:
            if per_band > min_available:
                print(
                    f"[balance] Requested per-band={per_band}, but smallest band has only {min_available}."
                )
                print(f"[balance] Reducing per-band to {min_available}.")
                per_band = min_available
            effective_drop_underfull = False

    rng = np.random.default_rng(seed)
    sampled_frames: List[pd.DataFrame] = []
    sampled_counts: Dict[str, int] = {}

    for band in band_labels:
        band_df = df[df["rating_band"] == band]
        band_count = len(band_df)
        if band_count == 0:
            continue

        if effective_drop_underfull and band_count < per_band:
            print(
                f"[balance] Skipping band {band} (only {band_count} < per-band={per_band})."
            )
            continue

        n = min(per_band, band_count)
        idx = rng.choice(band_df.index.to_numpy(), size=n, replace=False)
        sampled = band_df.loc[idx]
        sampled_frames.append(sampled)
        sampled_counts[band] = n

    if not sampled_frames:
        raise ValueError(
            "[balance] No bands met the per-band requirement; try lowering --per-band "
            "or disabling --drop-underfull-bands."
        )

    balanced_df = pd.concat(sampled_frames, ignore_index=True)

    # Shuffle overall
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"\n[balance] Total balanced games: {len(balanced_df)}")
    print("[balance] Per-band sampled counts:")
    for band in band_labels:
        if band in sampled_counts:
            print(f"  {band}: {sampled_counts[band]}")
    return balanced_df, sampled_counts


def split_by_band(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the balanced DF into train/val/test **per rating_band** so that
    each split stays balanced across bands.
    """
    if not (0.0 <= train_frac <= 1.0 and 0.0 <= val_frac <= 1.0 and 0.0 <= test_frac <= 1.0):
        raise ValueError("train_frac/val_frac/test_frac must each be in [0, 1].")

    total_frac = train_frac + val_frac + test_frac
    if total_frac <= 0:
        raise ValueError("Sum of train_frac, val_frac, test_frac must be > 0.")

    rng = np.random.default_rng(seed)
    train_frames: List[pd.DataFrame] = []
    val_frames: List[pd.DataFrame] = []
    test_frames: List[pd.DataFrame] = []

    for band, group in df.groupby("rating_band"):
        group = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
        n = len(group)
        n_train = int(round(train_frac / total_frac * n))
        n_val = int(round(val_frac / total_frac * n))
        n_test = n - n_train - n_val

        train_frames.append(group.iloc[:n_train])
        val_frames.append(group.iloc[n_train : n_train + n_val])
        test_frames.append(group.iloc[n_train + n_val :])

    train_df = pd.concat(train_frames, ignore_index=True)
    val_df = pd.concat(val_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    print("\n[balance] Split sizes:")
    print(f"  train: {len(train_df)}")
    print(f"  val  : {len(val_df)}")
    print(f"  test : {len(test_df)}")

    return train_df, val_df, test_df


def write_pgn_for_index(df: pd.DataFrame, out_path: Path) -> None:
    """
    Given a DF with 'pgn_path', 'start_offset', 'end_offset', write all games
    to out_path by slicing from the original PGNs. Handles multiple months/files.
    """
    if df.empty:
        print(f"[balance] No games to write for {out_path.name}.")
        return

    required_cols = {"pgn_path", "start_offset", "end_offset"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            "DF must contain 'pgn_path', 'start_offset', 'end_offset' columns."
        )

    # Sort by (pgn_path, start_offset) to minimize seeks
    df_sorted = df.sort_values(["pgn_path", "start_offset"]).reset_index(drop=True)

    print(f"[balance] Writing {len(df_sorted)} games to {out_path}")
    with out_path.open("w", encoding="utf-8") as f_out:
        for pgn_name, group in df_sorted.groupby("pgn_path"):
            pgn_path = DATA_DIR / pgn_name
            if not pgn_path.exists():
                raise FileNotFoundError(
                    f"PGN file not found: {pgn_path}. "
                    "Run scripts.build_month_index first."
                )

            with pgn_path.open("r", encoding="utf-8") as f_in:
                for _, row in tqdm(
                    group.iterrows(),
                    total=len(group),
                    unit="game",
                    desc=f"  {pgn_name}",
                ):
                    start = int(row["start_offset"])
                    end = int(row["end_offset"])
                    f_in.seek(start)
                    chunk = f_in.read(end - start)
                    f_out.write(chunk)
                    # Ensure at least one blank line between games
                    if not chunk.endswith("\n\n"):
                        if not chunk.endswith("\n"):
                            f_out.write("\n")
                        f_out.write("\n")
    print(f"[balance] Done writing {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    months = args.months
    df = load_indexes(months)

    # Apply optional filter
    if args.where:
        print(f"[balance] Applying filter: {args.where}")
        df = df.query(args.where, engine="python")
        print(f"[balance] Remaining games after filter: {len(df)}")

    if df.empty:
        print("[balance] No games match the given filter.")
        return

    # Rating column preparation
    df_rating, rating_col = prepare_rating_column(df, args.rating_col, args.bin_size)

    if df_rating.empty:
        print("[balance] No games remain after rating preprocessing.")
        return

    # Rating banding
    before_bands = len(df_rating)
    df_bands, band_labels = make_rating_bins(
        df_rating,
        rating_col=rating_col,
        bin_size=args.bin_size,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
    )
    after_bands = len(df_bands)
    dropped_binning = before_bands - after_bands
    if dropped_binning > 0:
        print(f"[balance] Dropped {dropped_binning} games during rating band binning.")

    if df_bands.empty:
        print("[balance] No games remain after rating band binning.")
        return

    # Balanced sampling
    balanced_df, sampled_counts = sample_balanced_by_band(
        df_bands,
        rating_col=rating_col,
        band_labels=band_labels,
        per_band=args.per_band,
        seed=args.seed,
        drop_underfull_bands=args.drop_underfull_bands,
    )

    # Decide output prefix
    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
    else:
        if len(months) == 1:
            tag = months[0]
        else:
            tag = "combined"
        out_prefix = DATA_DIR / f"rym_balanced_{tag}"

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Save balanced index (Parquet)
    index_path = out_prefix.parent / f"{out_prefix.name}_index.parquet"
    balanced_df.to_parquet(index_path, index=False)
    print(f"[balance] Saved balanced index to {index_path}")

    # If no-split, just write one PGN
    if args.no_split:
        pgn_path = out_prefix.parent / f"{out_prefix.name}.pgn"
        write_pgn_for_index(balanced_df, pgn_path)
        return

    # Otherwise, split per band and write train/val/test PGNs
    train_df, val_df, test_df = split_by_band(
        balanced_df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    write_pgn_for_index(train_df, out_prefix.parent / f"{out_prefix.name}_train.pgn")
    write_pgn_for_index(val_df, out_prefix.parent / f"{out_prefix.name}_val.pgn")
    write_pgn_for_index(test_df, out_prefix.parent / f"{out_prefix.name}_test.pgn")


if __name__ == "__main__":
    main()

"""
python -m scripts.build_balanced_pgn 2017-04 \
  --where "time_control == '600+0'" \
  --rating-col avg_elo \
  --bin-size 100 \
  --seed 64 \
  --per-band 1000 \
  --drop-underfull-bands \
  --train-frac 0.8 --val-frac 0.1 --test-frac 0.1 \
  --out-prefix data/rym_2017-04_bin_1000
"""