#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
from fractions import Fraction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

ROOT_DIR = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"


# ---------------------------------------------------------------------
# Helpers: load index, parse time controls, rating bins
# ---------------------------------------------------------------------


def load_index_for_month(month: str) -> Tuple[pd.DataFrame, Path]:
    csv_path = DATA_DIR / f"lichess_db_standard_rated_{month}_index.csv"
    parquet_path = csv_path.with_suffix(".parquet")

    if parquet_path.exists():
        print(f"[stats] Loading Parquet index {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"[stats] Loading CSV index {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No index found for {month}. Expected:\n"
            f"  {parquet_path}\n"
            f"or\n"
            f"  {csv_path}\n"
            f"Run scripts.build_month_index first."
        )

    df["month"] = month
    return df, csv_path


def parse_time_control_str(tc: str) -> Tuple[int | None, int | None]:
    """
    Parse Lichess TimeControl string like "300+0" into (base, inc) in seconds.

    Returns (None, None) if parsing fails.
    """
    if not isinstance(tc, str) or not tc:
        return None, None

    parts = tc.split("+")
    try:
        base = int(parts[0])
    except (ValueError, TypeError):
        return None, None

    if len(parts) > 1:
        try:
            inc = int(parts[1])
        except (ValueError, TypeError):
            inc = 0
    else:
        inc = 0

    return base, inc


def format_tc_label(base_sec: int, inc_sec: int) -> str:
    """
    Convert base seconds to minutes (possibly fractional) and keep increment in seconds.

    Examples:
      300+0 -> "5+0"
      180+2 -> "3+2"
      30+0  -> "1/2+0"
      90+0  -> "3/2+0"
    """
    if base_sec <= 0:
        base_str = "0"
    else:
        frac = Fraction(base_sec, 60).limit_denominator()
        if frac.denominator == 1:
            base_str = str(frac.numerator)
        else:
            base_str = f"{frac.numerator}/{frac.denominator}"
    return f"{base_str}+{inc_sec}"


def add_time_control_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Parse 'time_control' into tc_base, tc_inc (seconds), and tc_label, and
    DROP rows where parsing fails.

    Returns:
      df_tc  : filtered DataFrame with tc_* columns
      dropped: number of rows dropped due to bad time control
    """
    if "time_control" not in df.columns:
        raise ValueError("Index is missing 'time_control' column.")

    base_list: List[int | None] = []
    inc_list: List[int | None] = []
    ok_mask: List[bool] = []

    for tc in df["time_control"]:
        base, inc = parse_time_control_str(tc)
        base_list.append(base)
        inc_list.append(inc)
        ok_mask.append(base is not None and inc is not None)

    ok_mask_arr = np.array(ok_mask, dtype=bool)
    dropped = int((~ok_mask_arr).sum())

    # Filter df and corresponding base/inc lists
    df_tc = df[ok_mask_arr].copy()
    tc_base = [b for b, ok in zip(base_list, ok_mask_arr) if ok]
    tc_inc = [i for i, ok in zip(inc_list, ok_mask_arr) if ok]

    df_tc["tc_base"] = tc_base
    df_tc["tc_inc"] = tc_inc
    df_tc["tc_base"] = df_tc["tc_base"].astype(int)
    df_tc["tc_inc"] = df_tc["tc_inc"].astype(int)
    df_tc["tc_label"] = [
        format_tc_label(b, i) for b, i in zip(df_tc["tc_base"], df_tc["tc_inc"])
    ]

    return df_tc, dropped


def make_rating_bins(
    df: pd.DataFrame,
    rating_col: str,
    bin_size: int,
    min_rating: int | None = None,
    max_rating: int | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add a 'rating_band' categorical column based on rating_col, using bins of
    width bin_size. Returns (df_with_band, band_labels in numeric order).
    """
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in index.")

    df = df.copy()
    ratings = df[rating_col].dropna()
    if ratings.empty:
        raise ValueError(f"No non-NaN values in rating column '{rating_col}'.")

    if min_rating is None:
        min_rating = int((ratings.min() // bin_size) * bin_size)
    if max_rating is None:
        max_rating = int(((ratings.max() // bin_size) + 1) * bin_size)

    bins = np.arange(min_rating, max_rating + bin_size, bin_size)
    labels = [f"{lo}-{hi}" for lo, hi in zip(bins[:-1], bins[1:])]

    df["rating_band"] = pd.cut(
        df[rating_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    before = len(df)
    df = df.dropna(subset=["rating_band"])
    after = len(df)
    # We’ll measure drop_binning outside via before/after if needed.

    df["rating_band"] = df["rating_band"].astype(str)
    return df, labels


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_rating_hist(
    counts: pd.Series,
    rating_order: List[str],
    total: int,
    out_dir: Path,
    fname: str = "rating_hist.png",
    title_suffix: str = "",
) -> None:
    """
    Bar 'histogram' of game counts by rating band, with count and pct annotated.
    Rating bands are ordered numerically via rating_order.
    """
    ensure_out_dir(out_dir)

    counts = counts.reindex(rating_order, fill_value=0)

    bands = counts.index.tolist()
    values = counts.values
    pcts = values / total * 100.0

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bands))
    ax.bar(x, values)

    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=45, ha="right")
    ax.set_xlabel("Rating band")
    ax.set_ylabel("Game count")
    title = "Game count by rating band"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)

    for xi, v, pct in zip(x, values, pcts):
        ax.text(
            xi,
            v,
            f"{v}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[stats] Saved rating histogram to {out_path}")


def plot_time_control_bar(
    tc_df: pd.DataFrame,
    total: int,
    out_dir: Path,
    fname: str,
    title: str,
    label_col: str = "tc_label",
) -> None:
    """
    Generic time-control bar chart with count and pct annotations.
    Labels come from label_col (default: 'tc_label').
    Ordering is determined by the order of tc_df.
    """
    ensure_out_dir(out_dir)

    labels = tc_df[label_col].tolist()
    values = tc_df["count"].tolist()
    pcts = [v / total * 100.0 for v in values]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    ax.bar(x, values)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Time control (minutes + increment seconds)")
    ax.set_ylabel("Game count")
    ax.set_title(title)

    for xi, v, pct in zip(x, values, pcts):
        ax.text(
            xi,
            v,
            f"{v}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[stats] Saved time-control bar chart to {out_path}")


def plot_rating_tc_heatmap(
    data_pct: np.ndarray,
    rating_bands: List[str],
    tc_labels: List[str],
    out_dir: Path,
    fname: str,
    title_suffix: str = "",
    log_scale: bool = False,
) -> None:
    """
    Heatmap of rating_band x time_control in PERCENT of games.
    Time controls are ordered as given in tc_labels (typically by descending
    game count). Optionally log-scale the color mapping.
    """
    ensure_out_dir(out_dir)

    fig, ax = plt.subplots(
        figsize=(max(8, len(tc_labels) * 0.4), max(6, len(rating_bands) * 0.3))
    )

    if log_scale:
        positive = data_pct[data_pct > 0]
        if positive.size == 0:
            norm = None
        else:
            vmin = max(1e-3, float(positive.min()))
            vmax = float(positive.max())
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(
            data_pct,
            aspect="auto",
            cmap="YlOrRd",
            origin="lower",
            norm=norm,
        )
    else:
        im = ax.imshow(
            data_pct,
            aspect="auto",
            cmap="YlOrRd",
            origin="lower",
        )

    ax.set_xticks(np.arange(len(tc_labels)))
    ax.set_yticks(np.arange(len(rating_bands)))
    ax.set_xticklabels(tc_labels, rotation=45, ha="right")
    ax.set_yticklabels(rating_bands)

    ax.set_xlabel("Time control (minutes + increment seconds)")
    ax.set_ylabel("Rating band")

    base_title = "Game percent heatmap (rating band × time control)"
    if log_scale:
        base_title += " [log scale]"
    if title_suffix:
        base_title += f" ({title_suffix})"
    ax.set_title(base_title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percent of games")

    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[stats] Saved rating×time-control heatmap to {out_path}")


# ---------------------------------------------------------------------
# Main: CLI + stats
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute SQL-like stats and visualizations over the monthly index.\n\n"
            "Examples:\n"
            "  # Basic stats for one month\n"
            "  python -m scripts.index_stats 2018-01\n\n"
            "  # Filtered stats (only games with eval and clocks, white wins)\n"
            "  python -m scripts.index_stats 2018-01 2018-02 \\\n"
            "    --where 'result == \"1-0\" and has_clock and has_eval' \\\n"
            "    --rating-col avg_elo --bin-size 100\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "months",
        nargs="+",
        help="One or more months in 'YYYY-MM' format, e.g. 2018-01 2018-02",
    )
    parser.add_argument(
        "--where",
        "-w",
        help=(
            "Pandas query over index columns, e.g. "
            "'white_elo >= 1900 and result == \"1-0\" and has_eval'"
        ),
    )
    parser.add_argument(
        "--rating-col",
        default="avg_elo",
        help=(
            "Which rating column to use for rating bands. "
            "Typical options: 'white_elo', 'black_elo', or 'avg_elo'. "
            "Default: avg_elo (average of white_elo and black_elo). "
            "In avg_elo mode, games with |white_elo - black_elo| > 2 * bin_size "
            "are dropped."
        ),
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=100,
        help="Rating band width (e.g. 100 → 0-100, 100-200, ...).",
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
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Directory for plots (default: <repo>/plots/<month_or_combined>). "
            "If multiple months are provided, 'combined' is used."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only print tables/stats; do not generate plots.",
    )
    parser.add_argument(
        "--max-time-controls",
        type=int,
        default=30,
        help=(
            "Maximum number of time controls to include in the heatmap "
            "and bar charts (top-N by frequency). Use a larger value if "
            "you want everything."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load indexes for all months and concatenate
    dfs = []
    for m in args.months:
        df_m, _ = load_index_for_month(m)
        dfs.append(df_m)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[stats] Loaded {len(df)} total games across {len(args.months)} month(s).")

    # Apply filter if provided (SQL-ish WHERE)
    if args.where:
        print(f"[stats] Applying filter: {args.where}")
        df = df.query(args.where, engine="python")
    print(f"[stats] Remaining games after filter: {len(df)}")

    if df.empty:
        print("[stats] No games match the given filter.")
        return

    # -----------------------------------------------------------------
    # 1) Time control parsing first (drop bad time controls)
    # -----------------------------------------------------------------
    df_tc, drop_time_control = add_time_control_columns(df)

    # -----------------------------------------------------------------
    # 2) Rating preparation: avg_elo / white_elo / black_elo
    #    (drop missing ratings, then large rating diff in avg_elo mode)
    # -----------------------------------------------------------------
    drop_missing_rating = 0
    drop_rating_diff = 0

    if args.rating_col == "avg_elo":
        if "white_elo" not in df_tc.columns or "black_elo" not in df_tc.columns:
            raise ValueError(
                "Cannot use rating_col='avg_elo' without 'white_elo' and 'black_elo'."
            )

        mask_wb = df_tc["white_elo"].notna() & df_tc["black_elo"].notna()
        drop_missing_rating = int((~mask_wb).sum())
        df_rating = df_tc[mask_wb].copy()

        df_rating["avg_elo"] = (
            df_rating["white_elo"].astype(float) + df_rating["black_elo"].astype(float)
        ) / 2.0

        rating_diff = (
            df_rating["white_elo"].astype(float) - df_rating["black_elo"].astype(float)
        ).abs()
        max_diff = 2 * args.bin_size
        mask_diff_ok = rating_diff <= max_diff
        drop_rating_diff = int((~mask_diff_ok).sum())
        df_rating = df_rating[mask_diff_ok].copy()

    else:
        if args.rating_col not in df_tc.columns:
            raise ValueError(f"Rating column '{args.rating_col}' not found in index.")

        mask_rating = df_tc[args.rating_col].notna()
        drop_missing_rating = int((~mask_rating).sum())
        df_rating = df_tc[mask_rating].copy()
        drop_rating_diff = 0  # not applied outside avg_elo mode

    # -----------------------------------------------------------------
    # 3) Rating bands (pd.cut on chosen rating column)
    # -----------------------------------------------------------------
    before_bands = len(df_rating)
    df_bands, rating_labels = make_rating_bins(
        df_rating,
        rating_col=args.rating_col,
        bin_size=args.bin_size,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
    )
    after_bands = len(df_bands)
    drop_binning = before_bands - after_bands

    df_final = df_bands
    total_games = len(df_final)
    print(f"[stats] Total games used in stats: {total_games}")

    # -----------------------------------------------------------------
    # Drop summary in the requested logical order
    # -----------------------------------------------------------------
    print("\n[stats] Drop summary:")
    print(f"  1) Dropped due to unparsable time control: {drop_time_control}")
    print(
        f"  2) Missing ratings for rating_col='{args.rating_col}': {drop_missing_rating}"
    )
    if args.rating_col == "avg_elo":
        print(
            f"  3) |white_elo - black_elo| > 2 * bin_size ({2 * args.bin_size}): {drop_rating_diff}"
        )
    else:
        print("  3) Rating-diff filter not applied (not in avg_elo mode).")
    print(f"  4) Dropped during rating band binning: {drop_binning}")

    if total_games == 0:
        print("[stats] No games remain after rating + time-control + band filters.")
        return

    df = df_final  # from here on we use df with tc_* and rating_band

    # ------------------ Counts: rating bands -------------------------
    rating_counts_raw = df["rating_band"].value_counts()
    rating_counts = rating_counts_raw.reindex(rating_labels, fill_value=0)

    print("\n[stats] Game count by rating band:")
    rating_summary = pd.DataFrame(
        {
            "rating_band": rating_counts.index,
            "count": rating_counts.values,
            "pct": rating_counts.values / total_games * 100.0,
        }
    )
    print(rating_summary.to_string(index=False))

    # ------------------ Counts: time controls ------------------------
    tc_counts = (
        df.groupby(["time_control", "tc_base", "tc_inc", "tc_label"])
        .size()
        .reset_index(name="count")
    )

    tc_counts_by_count = tc_counts.sort_values("count", ascending=False).reset_index(
        drop=True
    )
    if args.max_time_controls is not None and args.max_time_controls > 0:
        tc_top = tc_counts_by_count.head(args.max_time_controls).copy()
    else:
        tc_top = tc_counts_by_count.copy()

    tc_top["pct"] = tc_top["count"] / total_games * 100.0

    print("\n[stats] Top time controls by game count:")
    print(
        tc_top[
            ["tc_label", "time_control", "tc_base", "tc_inc", "count", "pct"]
        ].to_string(index=False)
    )

    tc_for_plot_by_ab = tc_top.sort_values(
        ["tc_base", "tc_inc", "time_control"]
    ).reset_index(drop=True)
    tc_for_plot_by_count = tc_top.sort_values("count", ascending=False).reset_index(
        drop=True
    )

    # ------------------ Heatmap data (rating × time control) ---------
    tc_labels_for_heat = tc_for_plot_by_count["time_control"].tolist()
    tc_display_labels_for_heat = tc_for_plot_by_count["tc_label"].tolist()

    df_heat = df[df["time_control"].isin(tc_labels_for_heat)]

    heat_counts = pd.crosstab(
        df_heat["rating_band"], df_heat["time_control"]
    ).reindex(index=rating_labels, columns=tc_labels_for_heat, fill_value=0)

    heat_counts_values = heat_counts.values.astype(float)
    heat_pct = heat_counts_values / total_games * 100.0

    # ------------------ Plots ----------------------------------------
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if len(args.months) == 1:
            out_dir = PLOTS_DIR / args.months[0]
        else:
            out_dir = PLOTS_DIR / "combined"

    title_suffix = "filtered" if args.where else ""

    if not args.no_plots:
        # 1. Rating histogram
        plot_rating_hist(
            counts=rating_counts,
            rating_order=rating_labels,
            total=total_games,
            out_dir=out_dir,
            fname="rating_hist.png",
            title_suffix=title_suffix,
        )

        # 2. Time-control bar: sorted by A then B
        plot_time_control_bar(
            tc_df=tc_for_plot_by_ab,
            total=total_games,
            out_dir=out_dir,
            fname="time_control_bar_by_ab.png",
            title="Game count by time control (sorted by base+increment)",
        )

        # 3. Time-control bar: sorted by number of games
        plot_time_control_bar(
            tc_df=tc_for_plot_by_count,
            total=total_games,
            out_dir=out_dir,
            fname="time_control_bar_by_count.png",
            title="Game count by time control (sorted by count)",
        )

        # 4. Rating × time-control heatmap in percent
        plot_rating_tc_heatmap(
            data_pct=heat_pct,
            rating_bands=rating_labels,
            tc_labels=tc_display_labels_for_heat,
            out_dir=out_dir,
            fname="rating_tc_heatmap_pct.png",
            title_suffix=title_suffix,
            log_scale=False,
        )

        # 5. Rating × time-control heatmap in percent, log-scaled
        plot_rating_tc_heatmap(
            data_pct=heat_pct,
            rating_bands=rating_labels,
            tc_labels=tc_display_labels_for_heat,
            out_dir=out_dir,
            fname="rating_tc_heatmap_log.png",
            title_suffix=title_suffix,
            log_scale=True,
        )

    print("\n[stats] Done.")


if __name__ == "__main__":
    main()


"""

python -m scripts.index_stats 2017-04 2018-02 \
  --where 'result == "1-0" and has_clock and has_eval' \
  --rating-col avg_elo \
  --bin-size 100
  
python -m scripts.index_stats 2017-04 --no-plots

"""