#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import chess
import chess.pgn

from .train_rym import RYMNpzDataset  # type: ignore
from .rym_models import get_model, MODEL_TYPES  # type: ignore
from .ply_features import NUM_PLANES  # type: ignore
from .build_rym_npz import pgn_to_npz  # type: ignore


def auto_device() -> str:
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models(
    models_dir: Path,
    num_planes: int,
    num_bins: int,
    config_id: int,
    device: torch.device,
    model_types: Sequence[str],
) -> Dict[str, torch.nn.Module]:
    models: Dict[str, torch.nn.Module] = {}
    for mtype in model_types:
        ckpt_path = models_dir / f"rym_{mtype}_cfg{config_id}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
        model = get_model(
            model_type=mtype,
            num_planes=num_planes,
            num_bins=num_bins,
            config_id=config_id,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models[mtype] = model
    return models


def pick_random_game(ds: RYMNpzDataset, seed: int | None = None) -> int:
    if ds.game_id is None:
        raise ValueError("NPZ does not contain 'game_id'; cannot inspect by game.")
    rng = np.random.default_rng(seed)
    game_ids = np.unique(ds.game_id)
    return int(rng.choice(game_ids))


def extract_game_indices(ds: RYMNpzDataset, game_id: int) -> np.ndarray:
    if ds.game_id is None or ds.ply_idx is None:
        raise ValueError("NPZ missing 'game_id' or 'ply_idx' arrays.")
    mask = (ds.game_id == game_id)
    idxs = np.nonzero(mask)[0]
    ply_order = ds.ply_idx[mask]
    order = np.argsort(ply_order)
    return idxs[order]


def load_pgn_game(pgn_path: Path, game_index: int) -> chess.pgn.Game:
    """
    Load the game at position game_index (0-based) from the PGN file.
    """
    with pgn_path.open("r", encoding="utf-8") as f:
        g = None
        for _ in range(game_index + 1):
            g = chess.pgn.read_game(f)
            if g is None:
                raise ValueError(
                    f"PGN has fewer than {game_index + 1} games: {pgn_path}"
                )
        assert g is not None
        return g


def compute_elo_centers(min_rating: float, max_rating: float, num_bins: int) -> np.ndarray:
    span = max_rating - min_rating
    width = span / num_bins
    centers = min_rating + width * (0.5 + np.arange(num_bins))
    return centers


def run_models_on_game(
    ds: RYMNpzDataset,
    idxs: np.ndarray,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    elo_centers: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    For a single game (given indices into ds), run all models and compute:

        - logits_seq: (T, num_bins)
        - reg_elo_seq: (T,)
        - posteriors: (T, num_bins)  Bayes-updated over time
        - cls_elo_seq: (T,)          expected Elo under posterior
    """
    X_np = ds.X[idxs]  # (T, C, 8, 8)
    T = X_np.shape[0]
    num_bins = len(elo_centers)

    results: Dict[str, Dict[str, Any]] = {}

    for mtype, model in models.items():
        logits_list: List[torch.Tensor] = []
        reg_list: List[torch.Tensor] = []

        for t in range(T):
            x = torch.from_numpy(X_np[t]).float().unsqueeze(0).to(device)  # (1, C, 8, 8)
            with torch.no_grad():
                logits, rating_pred = model(x)  # (1, num_bins), (1,)
            logits_list.append(logits.squeeze(0))       # (num_bins,)
            reg_list.append(rating_pred.squeeze(0))     # scalar

        logits_seq = torch.stack(logits_list, dim=0)  # (T, num_bins)
        reg_seq = torch.stack(reg_list, dim=0)        # (T,)

        # Bayes update over time to get posterior per move
        posteriors = []
        p = torch.full(
            (num_bins,),
            1.0 / num_bins,
            device=device,
            dtype=torch.float32,
        )
        for t in range(T):
            lik = F.softmax(logits_seq[t], dim=-1)
            p = p * lik
            p = p / p.sum()
            posteriors.append(p.clone())

        posteriors_t = torch.stack(posteriors, dim=0)  # (T, num_bins)

        # Classification Elo: expectation under posterior
        centers_t = torch.from_numpy(elo_centers).to(device=device, dtype=torch.float32)
        cls_elo_seq = (posteriors_t * centers_t.unsqueeze(0)).sum(dim=1)  # (T,)

        results[mtype] = {
            "logits_seq": logits_seq.cpu(),
            "reg_elo_seq": reg_seq.cpu(),
            "posteriors": posteriors_t.cpu(),
            "cls_elo_seq": cls_elo_seq.cpu(),
        }

    return results


def pretty_print_table(
    game: chess.pgn.Game,
    moves_san: List[str],
    y_elo_true: float,
    per_model: Dict[str, Dict[str, Any]],
) -> None:
    """
    Print a table with rows = plies, columns = per-model class/reg Elo.
    """
    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")
    print(f"\nGame: {white} vs {black}, true avg Elo ≈ {y_elo_true:.1f}")
    print("-" * 120)

    model_types = sorted(per_model.keys())

    # Header
    header = ["ply", "move"]
    for m in model_types:
        header.append(f"{m}_cls")
        header.append(f"{m}_reg")
    print("{:<4s} {:<10s} ".format(header[0], header[1]), end="")
    for h in header[2:]:
        print("{:>10s} ".format(h), end="")
    print()

    # Rows
    T = len(moves_san)
    for t in range(T):
        print("{:<4d} {:<10s} ".format(t + 1, moves_san[t]), end="")
        for m in model_types:
            cls_elo = float(per_model[m]["cls_elo_seq"][t].item())
            reg_elo = float(per_model[m]["reg_elo_seq"][t].item())
            print("{:>10.1f} {:>10.1f} ".format(cls_elo, reg_elo), end="")
        print()
    print("-" * 120)


def plot_per_move_distributions(
    out_dir: Path,
    game: chess.pgn.Game,
    moves_san: List[str],
    y_elo_true: float,
    elo_centers: np.ndarray,
    per_model: Dict[str, Dict[str, Any]],
) -> None:
    """
    For each ply, save a PNG showing:
        - Smooth (Gaussian-shaped) curves over Elo for each model, scaled up.
        - Matching-color vertical lines for cls/reg Elo per model.
        - For conv_transformer, also fill under the curve with very transparent alpha.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model_types = sorted(per_model.keys())
    num_bins = len(elo_centers)
    T = len(moves_san)

    # X-axis (rating) and basic limits
    x_min = float(elo_centers[0])
    x_max = float(elo_centers[-1])

    # Dense grid for plotting Gaussians
    xx = np.linspace(x_min, x_max, 512)

    # Bin width & Gaussian width
    if num_bins > 1:
        width = float(elo_centers[1] - elo_centers[0])
    else:
        width = (x_max - x_min) or 100.0
    sigma = width  # controls how "wide" each bump is

    # Build Gaussian basis, but DO NOT normalize – we care about shape, not area
    diff = xx[:, None] - elo_centers[None, :]
    gaussian_basis = np.exp(-0.5 * (diff / sigma) ** 2)  # (len(xx), num_bins)

    # Scale factor to make curves tall (≈ 0–100 range)
    scale_factor = 100.0

    # Color mapping per model, reused for curves and vertical lines
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i) for i, m in enumerate(model_types)}

    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")

    for t in range(T):
        plt.figure(figsize=(8, 4))

        max_y_this_frame = 0.0

        # Gaussian-mixture curves for each model
        for m in model_types:
            probs_t = per_model[m]["posteriors"][t].numpy()  # (num_bins,)

            # Smooth curve: mixture of Gaussians, then scaled up
            y_smooth = (gaussian_basis @ probs_t) * scale_factor  # (len(xx),)

            color = colors[m]
            plt.plot(
                xx,
                y_smooth,
                alpha=0.5,
                color=color,
                label=f"{m} prob",
            )

            # For conv_transformer, fill under the curve with very transparent alpha
            if m == "conv_transformer":
                plt.fill_between(
                    xx,
                    y_smooth,
                    alpha=0.1,
                    color=color,
                )

            max_y_this_frame = max(max_y_this_frame, float(y_smooth.max()))

        # Vertical lines for cls/reg Elo (matching colors and styles)
        for m in model_types:
            color = colors[m]
            cls_elo = float(per_model[m]["cls_elo_seq"][t].item())
            reg_elo = float(per_model[m]["reg_elo_seq"][t].item())

            # Classification expectation: dashed line
            plt.axvline(
                cls_elo,
                linestyle="--",
                alpha=0.8,
                color=color,
            )
            # Regression prediction: solid line
            plt.axvline(
                reg_elo,
                linestyle="-",
                alpha=0.8,
                color=color,
            )

        # Axes limits – let y go up to around 100
        y_max = max(max_y_this_frame * 1.05, scale_factor * 0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(0.0, y_max)

        plt.xlabel("Rating (Elo)")
        plt.ylabel("Scaled density (arb. units)")

        title = (
            f"{white} vs {black} – ply {t + 1}: {moves_san[t]}  "
            f"(true avg ≈ {y_elo_true:.1f})"
        )
        plt.title(title)
        plt.legend(fontsize=8, loc="upper right")
        plt.tight_layout()

        fname = out_dir / f"game_{white}_vs_{black}_ply_{t+1:03d}.png"
        plt.savefig(fname)
        plt.close()
        
# ---------------------------------------------------------------------
# Optional: Jupyter replay helper
# ---------------------------------------------------------------------


def replay_images(image_dir: str | Path, delay: float = 0.05) -> None:
    """
    In a Jupyter notebook, you can do:

        from scripts.inspect_rym_game import replay_images
        replay_images('plots/rym_inspect', delay=0.05)

    to see the frames as an animation.
    """
    from IPython.display import display, clear_output, Image
    import time

    image_dir = Path(image_dir)
    files = sorted(image_dir.glob("*.png"))
    for f in files:
        clear_output(wait=True)
        display(Image(filename=str(f)))
        time.sleep(delay)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a single game with all RYM models.",
    )
    parser.add_argument(
        "--test-npz",
        type=str,
        default=None,
        help=(
            "Test NPZ file (optional). If omitted, an NPZ will be built "
            "from --test-pgn using min/max rating and num-bins."
        ),
    )
    parser.add_argument(
        "--test-pgn",
        type=str,
        required=True,
        help="PGN file, e.g. data/rym_2017-04_bin_1000_test.pgn or your own games.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory with checkpoints rym_{model}_cfg{config}.pt",
    )
    parser.add_argument(
        "--config-id",
        type=int,
        default=0,
        help="Config index used during training (0..3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=auto_device(),
        help="Torch device string, e.g. 'mps', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for game selection (ignored if --game-index is set).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots/rym_inspect",
        help="Directory to save per-move PNGs.",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=None,
        help="Optional max number of plies to visualize (for very long games).",
    )
    parser.add_argument(
        "--game-index",
        type=int,
        default=None,
        help=(
            "Optional 0-based game index in the PGN/NPZ. "
            "If omitted, a random game is chosen."
        ),
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=800,
        help="Min rating used when auto-building NPZ from PGN.",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=2300,
        help="Max rating (exclusive) when auto-building NPZ from PGN.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of bins when auto-building NPZ from PGN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[inspect_rym_game] %(levelname)s: %(message)s",
    )
    log = logging.getLogger("inspect_rym_game")

    device = torch.device(args.device)
    log.info("Using device: %s", device)

    test_pgn = Path(args.test_pgn)

    # Decide NPZ path and build if needed
    if args.test_npz is not None:
        test_npz = Path(args.test_npz)
    else:
        # Derive NPZ from PGN name, e.g. my_games.pgn -> my_games.npz
        test_npz = test_pgn.with_suffix(".npz")

    if not test_npz.exists():
        log.info(
            "NPZ %s not found, building from PGN %s (min=%d, max=%d, bins=%d)",
            test_npz,
            test_pgn,
            args.min_rating,
            args.max_rating,
            args.num_bins,
        )
        pgn_to_npz(
            pgn_path=test_pgn,
            out_path=test_npz,
            max_games=None,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            num_bins=args.num_bins,
        )

    ds = RYMNpzDataset(test_npz)
    num_bins = ds.num_bins
    # Fallback to CLI defaults if dataset doesn't store min/max
    min_rating = float(getattr(ds, "min_rating", args.min_rating))
    max_rating = float(getattr(ds, "max_rating", args.max_rating))
    elo_centers = compute_elo_centers(min_rating, max_rating, num_bins)

    # Pick a game: either user-specified index or random
    if args.game_index is not None:
        game_id = args.game_index
    else:
        game_id = pick_random_game(ds, seed=args.seed)

    idxs = extract_game_indices(ds, game_id)
    if args.max_plies is not None:
        idxs = idxs[: args.max_plies]

    # Load PGN game (same index order as pgn_to_npz used)
    game = load_pgn_game(test_pgn, game_index=game_id)

    # Collect SAN moves
    moves_san: List[str] = []
    board = game.board()
    for move in game.mainline_moves():
        moves_san.append(board.san(move))
        board.push(move)
    if args.max_plies is not None:
        moves_san = moves_san[: len(idxs)]

    # True Elo (same for all plies of this game)
    y_elo_true = float(ds.y_elo[idxs[0]])

    # Load trained models
    model_types: Sequence[str] = MODEL_TYPES
    models_dir = Path(args.models_dir)
    models = load_models(
        models_dir=models_dir,
        num_planes=NUM_PLANES,
        num_bins=num_bins,
        config_id=args.config_id,
        device=device,
        model_types=model_types,
    )

    # Run models over this game
    per_model = run_models_on_game(
        ds=ds,
        idxs=idxs,
        models=models,
        device=device,
        elo_centers=elo_centers,
    )

    # Pretty-print Elo trajectories
    pretty_print_table(
        game=game,
        moves_san=moves_san,
        y_elo_true=y_elo_true,
        per_model=per_model,
    )

    # Generate per-move probability plots
    out_dir = Path(args.out_dir)
    plot_per_move_distributions(
        out_dir=out_dir,
        game=game,
        moves_san=moves_san,
        y_elo_true=y_elo_true,
        elo_centers=elo_centers,
        per_model=per_model,
    )
    log.info("Saved per-move plots to %s", out_dir)


if __name__ == "__main__":
    main()

"""
python -m scripts.inspect_rym_game \
--test-pgn my_games_rapid.pgn \
--models-dir models/rym_2017-04_baselines \
--config-id 0 \
--device mps \
--out-dir plots/rym_my_games
"""