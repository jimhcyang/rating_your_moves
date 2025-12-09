#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import chess
import chess.pgn

from .ply_features import encode_ply_planes, NUM_PLANES  # type: ignore

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT_DIR / "data"


# ---------------------------------------------------------------------
# Rating helpers
# ---------------------------------------------------------------------


def parse_avg_elo(game: chess.pgn.Game) -> Optional[float]:
    """Extract average Elo from PGN headers (WhiteElo / BlackElo)."""

    def _safe_int(tag: str | None) -> Optional[int]:
        if not tag:
            return None
        try:
            val = int(tag)
        except ValueError:
            return None
        return val if val > 0 else None

    w = _safe_int(game.headers.get("WhiteElo"))
    b = _safe_int(game.headers.get("BlackElo"))

    if w is not None and b is not None:
        return 0.5 * (w + b)
    if w is not None:
        return float(w)
    if b is not None:
        return float(b)
    return None


def rating_to_band(
    rating: float,
    *,
    min_rating: int,
    max_rating: int,
    num_bins: int,
) -> int:
    """
    Map numeric rating to a band index in [0, num_bins - 1].

    Ratings below min_rating are clipped to the lowest band.
    Ratings >= max_rating are clipped to the highest band.
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    if rating < min_rating:
        rating = float(min_rating)
    if rating >= max_rating:
        rating = float(max_rating - 1)

    width = (max_rating - min_rating) / num_bins
    idx = int((rating - min_rating) // width)
    if idx < 0:
        idx = 0
    if idx >= num_bins:
        idx = num_bins - 1
    return idx


# ---------------------------------------------------------------------
# PGN → NPZ conversion
# ---------------------------------------------------------------------


def pgn_to_npz(
    pgn_path: Path,
    out_path: Path,
    *,
    max_games: Optional[int],
    min_rating: int,
    max_rating: int,
    num_bins: int,
) -> None:
    """
    Convert a PGN of balanced games into an NPZ of per-ply tensors.

    Each ply is encoded using encode_ply_planes(board, move) which returns
    a (NUM_PLANES, 8, 8) uint8 tensor.

    The output NPZ contains:

        X        : uint8, shape (N, NUM_PLANES, 8, 8)
        y_bin    : int16, shape (N,)   # band index in [0, num_bins-1]
        y_elo    : float32, shape (N,) # average Elo used for banding
        game_id  : int32, shape (N,)   # 0-based game index within this PGN
        ply_idx  : int16, shape (N,)   # 0-based ply index within that game

        num_bins   : int16
        min_rating : int16
        max_rating : int16
    """
    log = logging.getLogger("build_rym_npz")
    log.info("Reading PGN from %s", pgn_path)
    log.info(
        "Banding config: min_rating=%d, max_rating=%d, num_bins=%d",
        min_rating,
        max_rating,
        num_bins,
    )

    X_list: List[np.ndarray] = []
    y_bin_list: List[int] = []
    y_elo_list: List[float] = []
    game_ids: List[int] = []
    ply_ids: List[int] = []

    games_seen = 0
    games_used = 0
    plies_used = 0

    with pgn_path.open("r", encoding="utf-8") as f:
        while True:
            if max_games is not None and games_seen >= max_games:
                break

            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_seen += 1

            avg_elo = parse_avg_elo(game)
            if avg_elo is None:
                continue  # drop games with unknown rating

            band_idx = rating_to_band(
                avg_elo,
                min_rating=min_rating,
                max_rating=max_rating,
                num_bins=num_bins,
            )

            board = game.board()
            ply_idx = 0

            # Iterate mainline moves only; side-to-move handled by encoder
            for move in game.mainline_moves():
                planes = encode_ply_planes(board, move)  # (NUM_PLANES, 8, 8)
                X_list.append(planes.astype(np.uint8))
                y_bin_list.append(band_idx)
                y_elo_list.append(float(avg_elo))
                game_ids.append(games_seen - 1)
                ply_ids.append(ply_idx)

                plies_used += 1
                ply_idx += 1
                board.push(move)

            games_used += 1

            if games_seen % 1000 == 0:
                log.info(
                    "Scanned %d games (kept %d), collected %d plies...",
                    games_seen,
                    games_used,
                    plies_used,
                )

    if not X_list:
        log.warning("No plies collected; nothing to save.")
        return

    X = np.stack(X_list, axis=0)  # (N, C, 8, 8)
    y_bin = np.asarray(y_bin_list, dtype=np.int16)
    y_elo = np.asarray(y_elo_list, dtype=np.float32)
    game_id = np.asarray(game_ids, dtype=np.int32)
    ply_idx_arr = np.asarray(ply_ids, dtype=np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y_bin=y_bin,
        y_elo=y_elo,
        game_id=game_id,
        ply_idx=ply_idx_arr,
        num_bins=np.int16(num_bins),
        min_rating=np.int16(min_rating),
        max_rating=np.int16(max_rating),
    )

    log.info(
        "Saved %s with %d plies from %d used games (scanned %d).",
        out_path,
        X.shape[0],
        games_used,
        games_seen,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a (balanced) PGN into an NPZ of per-ply RYM tensors.\n\n"
            "Example:\n"
            "  python -m scripts.build_rym_npz data/rym_2017-04_bin_1000_train.pgn \\\n"
            "    --out data/rym_2017-04_train.npz --min-rating 800 --max-rating 2400 \\\n"
            "    --num-bins 15\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pgn",
        type=str,
        help="Input PGN file (e.g. data/rym_2017-04_bin_1000_train.pgn).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output NPZ path (e.g. data/rym_2017-04_train.npz).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on number of games to process (for quick tests).",
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
        default=2400,
        help="Maximum rating edge for bands (exclusive).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of rating bands between min-rating and max-rating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[build_rym_npz] %(levelname)s: %(message)s",
    )

    pgn_path = Path(args.pgn)
    out_path = Path(args.out)

    if not pgn_path.exists():
        raise FileNotFoundError(f"Input PGN not found: {pgn_path}")

    if args.max_rating <= args.min_rating:
        raise ValueError("max-rating must be > min-rating")

    pgn_to_npz(
        pgn_path,
        out_path,
        max_games=args.max_games,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        num_bins=args.num_bins,
    )


if __name__ == "__main__":
    main()

"""
python -m scripts.build_rym_npz data/rym_2017-04_bin_1000_train.pgn \
--out data/rym_2017-04_train.npz

python -m scripts.build_rym_npz data/rym_2017-04_bin_1000_val.pgn \
--out data/rym_2017-04_val.npz

python -m scripts.build_rym_npz data/rym_2017-04_bin_1000_test.pgn \
--out data/rym_2017-04_test.npz
"""