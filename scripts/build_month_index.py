#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, Tuple, Iterator

import requests
import zstandard as zstd
from tqdm import tqdm

try:
    import pandas as pd  # for optional Parquet conversion
except ImportError:
    pd = None

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT_DIR / "data"

# Lichess standard DB URL pattern:
#   https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst
LICHESS_URL_TMPL = (
    "https://database.lichess.org/standard/"
    "lichess_db_standard_rated_{month}.pgn.zst"
)

HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')
RESULT_RE = re.compile(r"\b(1-0|0-1|1/2-1/2)\b")


# ---------------------------------------------------------------------
# Download & decompress
# ---------------------------------------------------------------------


def download_month_zst(month: str, overwrite: bool = False) -> Path:
    """
    Download lichess_db_standard_rated_YYYY-MM.pgn.zst into data/.

    month: 'YYYY-MM', e.g. '2017-01'
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zst_path = DATA_DIR / f"lichess_db_standard_rated_{month}.pgn.zst"
    if zst_path.exists() and not overwrite:
        print(f"[download] Using existing {zst_path}")
        return zst_path

    url = LICHESS_URL_TMPL.format(month=month)
    print(f"[download] Fetching {url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("Content-Length", 0)) or None
    with zst_path.open("wb") as f_out, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {month}",
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            f_out.write(chunk)
            if total:
                pbar.update(len(chunk))

    print(f"[download] Saved to {zst_path}")
    return zst_path


def ensure_pgn_from_zst(zst_path: Path, overwrite: bool = False) -> Path:
    """
    Decompress .zst -> .pgn next to it.

    Returns path to the .pgn file.
    """
    if zst_path.suffix != ".zst":
        raise ValueError(f"Expected .zst file, got {zst_path}")
    pgn_path = zst_path.with_suffix("")  # drop .zst

    if pgn_path.exists() and not overwrite:
        print(f"[decompress] Using existing {pgn_path}")
        return pgn_path

    print(f"[decompress] Decompressing {zst_path} -> {pgn_path}")
    dctx = zstd.ZstdDecompressor()
    with zst_path.open("rb") as f_in, pgn_path.open("wb") as f_out:
        with dctx.stream_reader(f_in) as reader, tqdm(
            unit="B", unit_scale=True, desc=f"Decompressing {zst_path.name}"
        ) as pbar:
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.update(len(chunk))

    print(f"[decompress] Done.")
    return pgn_path


# ---------------------------------------------------------------------
# Robust PGN game iterator (header + moves as one unit)
# ---------------------------------------------------------------------


def iter_games_with_offsets(f_in) -> Iterator[Tuple[int, int, str]]:
    """
    Yield (start_offset, end_offset, game_text) for each game in a PGN file.

    We DO NOT assume [Event] is the first tag. We handle:

      * Any header tag line starting with '[' as part of the header block.
      * Blank line between header and moves as part of the same game.
      * Game boundaries at:
          - first blank line AFTER we've seen any movetext, OR
          - the next header line '[...]' appearing after movetext (no blank).

    We also track whether we saw an explicit result token ("1-0", etc.)
    in the movetext, mainly as a sanity signal (we don't hard-rely on it,
    because some games have result only in the header).
    """
    while True:
        # -------------------------------------------------------------
        # Find the first non-blank line: start of next game
        # -------------------------------------------------------------
        start_offset = f_in.tell()
        line = f_in.readline()
        if not line:
            return  # EOF
        while line.strip() == "":
            start_offset = f_in.tell()
            line = f_in.readline()
            if not line:
                return

        game_lines = [line]
        is_header = line.lstrip().startswith("[")
        seen_move_text = not is_header
        seen_result_token = False
        last_nonblank = line.strip()

        if seen_move_text and RESULT_RE.search(last_nonblank):
            seen_result_token = True

        # -------------------------------------------------------------
        # Consume until we decide the game ends
        # -------------------------------------------------------------
        while True:
            pos_before = f_in.tell()
            line = f_in.readline()
            if not line:
                # EOF: game ends here
                end_offset = pos_before
                yield start_offset, end_offset, "".join(game_lines)
                return

            stripped = line.strip()

            if stripped == "":
                # Blank line
                game_lines.append(line)

                if seen_move_text:
                    # Blank line AFTER movetext:
                    # This is our primary "sus" boundary: usually the '\n\n'
                    # right after a result like '... 1-0'.
                    # Even if we never saw the result token in movetext
                    # (e.g. only in [Result]), we still treat this as the
                    # end of the game.
                    end_offset = f_in.tell()
                    break
                else:
                    # Blank between header and first movetext line
                    # -> header-moves separator, stay in same game.
                    continue

            # Non-blank line
            is_header = line.lstrip().startswith("[")
            if seen_move_text and is_header:
                # New header after we've already seen moves:
                # this is the next game's header; current game ends
                # right BEFORE this line.
                f_in.seek(pos_before)
                end_offset = pos_before
                break

            # Still same game
            game_lines.append(line)
            last_nonblank = stripped
            if not is_header:
                seen_move_text = True
                if RESULT_RE.search(last_nonblank):
                    seen_result_token = True

        # Optionally, if you ever want to debug "weird" games where we
        # never saw a moves result token, you could print a warning here
        # for the first few such cases. For now, we just yield.
        yield start_offset, end_offset, "".join(game_lines)


# ---------------------------------------------------------------------
# PGN parsing -> headers + movetext
# ---------------------------------------------------------------------


def split_header_moves(game_text: str) -> Tuple[Dict[str, str], str]:
    """
    Given raw PGN for a single game, return:

    headers: dict[tag -> value]
    moves_str: concatenated movetext lines (with internal spacing normalized)
    """
    headers: Dict[str, str] = {}
    lines = game_text.splitlines()
    move_lines = []
    in_header = True

    for line in lines:
        stripped = line.strip()
        if in_header and stripped.startswith("["):
            m = HEADER_RE.match(line)
            if m:
                tag, val = m.groups()
                headers[tag] = val
        else:
            in_header = False
            if stripped:
                move_lines.append(line)

    moves_str = " ".join(move_lines)
    return headers, moves_str


# ---------------------------------------------------------------------
# CSV / Parquet index builder
# ---------------------------------------------------------------------


def build_index_csv(
    month: str,
    pgn_path: Path,
    overwrite: bool = False,
    write_parquet: bool = True,
) -> Path:
    """
    Stream through the .pgn file, writing one row per game:

    Columns include:
      - month
      - game_id
      - pgn_path (filename)
      - start_offset, end_offset (file offsets usable with .seek)
      - site, event, date, round
      - white, black
      - white_elo, black_elo
      - result, time_control, termination
      - utc_date, utc_time
      - eco, opening
      - has_clock (any [%clk ...] comment)
      - has_eval  (any [%eval ...] comment)
      - num_moves (approx., from move numbers in movetext)

    Additionally, if write_parquet=True and pandas+parquet are available,
    we will convert the CSV index to a .parquet file for faster loading.
    """
    index_path = pgn_path.with_name(
        pgn_path.name.replace(".pgn", "_index.csv")
    )
    if index_path.exists() and not overwrite:
        print(f"[index] Using existing {index_path}")
        if write_parquet:
            maybe_write_parquet(index_path)
        return index_path

    print(f"[index] Building index {index_path} from {pgn_path}")

    fieldnames = [
        "month",
        "game_id",
        "pgn_path",
        "start_offset",
        "end_offset",
        "site",
        "event",
        "date",
        "round",
        "white",
        "black",
        "white_elo",
        "black_elo",
        "result",
        "time_control",
        "termination",
        "utc_date",
        "utc_time",
        "eco",
        "opening",
        "has_clock",
        "has_eval",
        "num_moves",
    ]

    def parse_int(tag_val: str | None) -> int | None:
        if tag_val is None:
            return None
        try:
            return int(tag_val)
        except ValueError:
            return None

    game_id = 0

    with pgn_path.open("r", encoding="utf-8", newline="") as f_in, index_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(desc="Indexing games", unit="game")
        sanity_printed = False

        for start_offset, end_offset, game_text in iter_games_with_offsets(f_in):
            headers, moves_str = split_header_moves(game_text)

            has_clock = "[%clk" in game_text
            has_eval = "[%eval" in game_text

            # Rough number of moves: count move numbers like "1.", "2.", ...
            num_moves = len(re.findall(r"\b\d+\.", moves_str))

            row = {
                "month": month,
                "game_id": game_id,
                "pgn_path": pgn_path.name,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "site": headers.get("Site", ""),
                "event": headers.get("Event", ""),
                "date": headers.get("Date", ""),
                "round": headers.get("Round", ""),
                "white": headers.get("White", ""),
                "black": headers.get("Black", ""),
                "white_elo": parse_int(headers.get("WhiteElo")),
                "black_elo": parse_int(headers.get("BlackElo")),
                "result": headers.get("Result", ""),
                "time_control": headers.get("TimeControl", ""),
                "termination": headers.get("Termination", ""),
                "utc_date": headers.get("UTCDate", ""),
                "utc_time": headers.get("UTCTime", ""),
                "eco": headers.get("ECO", ""),
                "opening": headers.get("Opening", ""),
                "has_clock": has_clock,
                "has_eval": has_eval,
                "num_moves": num_moves,
            }
            writer.writerow(row)

            # -------------------- Sanity preview ------------------------
            if game_id < 3:
                if not sanity_printed:
                    print("\n[sanity] Previewing the first few games "
                          "before indexing everything...")
                    sanity_printed = True

                print("=" * 60)
                print(f"[sanity] Game {game_id}")
                print(f"  Event:       {row['event']}")
                print(f"  Site:        {row['site']}")
                print(f"  TimeControl: {row['time_control']}")
                print(f"  Result:      {row['result']}")
                preview_moves = moves_str[:200].replace("\n", " ")
                print(f"  Moves (preview): {preview_moves}...")
                print(f"  Offsets: [{start_offset}, {end_offset})")
                print(f"  num_moves: {num_moves}")
                print("=" * 60)

                if game_id == 0:
                    # print("[sanity] Sleeping 3 seconds before continuing...")
                    time.sleep(3)

            game_id += 1
            pbar.update(1)

        pbar.close()

    print(f"[index] Indexed {game_id} games into {index_path}")

    if write_parquet:
        maybe_write_parquet(index_path)

    return index_path


def maybe_write_parquet(index_path: Path) -> None:
    """
    Optionally convert the CSV index to Parquet for faster loading.

    Note: requires pandas + a parquet engine (e.g. pyarrow).
    """
    if pd is None:
        print("[parquet] pandas not available; skipping Parquet conversion.")
        return

    parquet_path = index_path.with_suffix(".parquet")
    if parquet_path.exists():
        print(f"[parquet] Using existing {parquet_path}")
        return

    print(f"[parquet] Converting {index_path} -> {parquet_path}")
    df = pd.read_csv(index_path)
    df.to_parquet(parquet_path, index=False)
    print("[parquet] Done.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download lichess monthly standard games, decompress, "
            "and build a CSV/Parquet index for fast slicing."
        )
    )
    parser.add_argument(
        "month",
        help="Month in 'YYYY-MM' format, e.g. 2017-01",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload / redecompress / rebuild index even if files exist.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Do not generate a Parquet copy of the index.",
    )
    args = parser.parse_args()

    month = args.month
    overwrite = args.overwrite
    write_parquet = not args.no_parquet

    zst_path = download_month_zst(month, overwrite=overwrite)
    pgn_path = ensure_pgn_from_zst(zst_path, overwrite=overwrite)
    build_index_csv(month, pgn_path, overwrite=overwrite, write_parquet=write_parquet)


if __name__ == "__main__":
    main()
