#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter games using the monthly index and optionally write "
            "a new PGN file.\n\n"
            "Examples:\n"
            "  # Count how many games match a condition\n"
            "  python -m scripts.filter_games 2017-01 \\\n"
            "    --where 'white_elo >= 1900 and white_elo < 2000 and '\n"
            "            'result == \"1-0\" and time_control == \"180+2\" '\n"
            "            'and has_clock and has_eval' \\\n"
            "    --count-only\n\n"
            "  # Actually write up to 1000 such games to a PGN file\n"
            "  python -m scripts.filter_games 2017-01 \\\n"
            "    --where 'white_elo >= 1900 and white_elo < 2000 and '\n"
            "            'result == \"1-0\" and time_control == \"180+2\" '\n"
            "            'and has_clock and has_eval' \\\n"
            "    --limit 1000 \\\n"
            "    --out data/2017-01_white1900_tc180_2_annotated.pgn\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "month",
        help="Month in 'YYYY-MM' format, e.g. 2017-01",
    )
    parser.add_argument(
        "--where",
        "-w",
        help=(
            "Pandas query over index columns, e.g. "
            "'white_elo >= 1900 and white_elo < 2000 and result == \"1-0\"'"
        ),
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        help="Maximum number of games to write (after filtering).",
    )
    parser.add_argument(
        "--out",
        "-o",
        help=(
            "Output PGN path. "
            "Defaults to data/lichess_filtered_<month>.pgn"
        ),
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only print counts and (optionally) a preview, do not write PGN.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print a small preview of the first N matching rows.",
    )
    return parser.parse_args()


def load_index(month: str) -> tuple[pd.DataFrame, Path, Path]:
    """
    Load the monthly index (prefer Parquet, fall back to CSV) and
    return (df, csv_path, pgn_path).
    """
    csv_path = DATA_DIR / f"lichess_db_standard_rated_{month}_index.csv"
    parquet_path = csv_path.with_suffix(".parquet")
    pgn_path = DATA_DIR / f"lichess_db_standard_rated_{month}.pgn"

    if parquet_path.exists():
        print(f"[filter] Loading Parquet index {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"[filter] Loading CSV index {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No index found for {month}. Expected:\n"
            f"  {parquet_path}\n"
            f"or\n"
            f"  {csv_path}\n"
            f"Run scripts.build_month_index first."
        )

    if not pgn_path.exists():
        raise FileNotFoundError(
            f"PGN not found: {pgn_path}. "
            "It should be produced by scripts.build_month_index."
        )

    return df, csv_path, pgn_path


def main() -> None:
    args = parse_args()
    month = args.month

    df, index_path, pgn_path = load_index(month)

    print(f"[filter] Index loaded: {len(df)} total games in {index_path.name}")

    if args.where:
        print(f"[filter] Applying filter: {args.where}")
        df = df.query(args.where, engine="python")

    total_matches = len(df)
    print(f"[filter] Total matches before LIMIT: {total_matches}")

    if args.limit is not None:
        df = df.head(args.limit)
        print(f"[filter] Games selected after LIMIT({args.limit}): {len(df)}")
    else:
        print(f"[filter] Games selected (no LIMIT): {len(df)}")

    if df.empty:
        print("[filter] No games matched your query.")
        return

    # Optional preview
    if args.preview > 0:
        cols = [
            "game_id",
            "white",
            "black",
            "white_elo",
            "black_elo",
            "result",
            "time_control",
            "has_clock",
            "has_eval",
            "num_moves",
        ]
        preview_cols = [c for c in cols if c in df.columns]
        print("\n[filter] Preview of first matches:")
        print(df[preview_cols].head(args.preview).to_string(index=False))
        print()

    if args.count_only:
        # Just counting / inspecting – no PGN writing
        print("[filter] Count-only mode: no PGN written.")
        return

    # Write filtered games to PGN
    out_path = (
        Path(args.out)
        if args.out
        else DATA_DIR / f"lichess_filtered_{month}.pgn"
    )

    # For efficient seeks, sort by start_offset
    if "start_offset" not in df.columns or "end_offset" not in df.columns:
        raise ValueError(
            "Index is missing 'start_offset'/'end_offset' columns, "
            "cannot slice PGN."
        )

    df = df.sort_values("start_offset").reset_index(drop=True)

    print(f"[filter] Writing {len(df)} games to {out_path}")
    with pgn_path.open("r", encoding="utf-8") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), unit="game"):
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

    print("[filter] Done.")


if __name__ == "__main__":
    main()
