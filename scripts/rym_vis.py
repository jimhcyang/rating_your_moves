from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Sequence

import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from IPython.display import display, HTML, Image as IPyImage, clear_output
from PIL import Image, ImageDraw, ImageFont

import chess
import chess.pgn
import chess.svg  # only for inline SVG in notebooks

from .ply_features import NUM_PLANES, plane_label_for_index, encode_ply_planes  # type: ignore
from .rym_analysis import (
    _band_edges_and_centers,
    _bayes_update_path,
    _is_post_band_col,
)

# ---------------------------------------------------------------------
# PGN → per-ply metadata
# ---------------------------------------------------------------------


def build_ply_metadata_from_pgn(pgn_path: str | Path) -> Dict[tuple[int, int], Dict[str, str]]:
    """
    Parse the PGN and return a mapping:
        (game_id, ply_idx) -> {
            'fen_pre',  # FEN before the move
            'fen_post', # FEN after the move
            'move_uci', # UCI string of the move
        }

    Assumes a simple sequential ordering of games in the PGN, matching the
    ordering used by pgn_to_npz.
    """
    pgn_path = Path(pgn_path)
    meta: Dict[tuple[int, int], Dict[str, str]] = {}

    with pgn_path.open("r", encoding="utf-8") as f:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            ply_idx = 0

            for move in game.mainline_moves():
                fen_pre = board.fen()
                board.push(move)
                fen_post = board.fen()

                meta[(game_idx, ply_idx)] = {
                    "fen_pre": fen_pre,
                    "fen_post": fen_post,
                    "move_uci": move.uci(),
                }

                ply_idx += 1

            game_idx += 1

    return meta


# ---------------------------------------------------------------------
# Rating-band confusion heatmaps (ply + game)
# ---------------------------------------------------------------------


def _compute_soft_confusion(
    true_bins: np.ndarray,
    prob_mat: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """
    Build a 'soft' confusion matrix:

        M[true, pred] = mean_{samples with y=true} P(pred | sample)

    So each row is a *distribution* over predicted bands for a given true band.
    """
    M = np.zeros((num_bins, num_bins), dtype=float)

    for b in range(num_bins):
        mask = (true_bins == b)
        if np.any(mask):
            M[b] = prob_mat[mask].mean(axis=0)
        else:
            M[b] = np.nan  # no samples in this band

    nan_rows = np.isnan(M).all(axis=1)
    if np.any(nan_rows):
        M[nan_rows] = 0.0

    return M


def plot_band_confusion_heatmap(
    df_out: "pd.DataFrame",
    game_summary: "pd.DataFrame",
    band_labels: list[str] | None = None,
    cmap: str | Any | None = None,
    figsize: tuple[int, int] = (12, 5),
):
    """
    LSTM-style soft confusion heatmaps for rating bands.

    Left:  per-ply  P(predicted band | true band)
    Right: per-game P(predicted band | true band), using the *final Bayesian
           posterior* for each game (post_* columns from summarize_game_posteriors).

    Orientation (matching your description):

        x-axis: TRUE band (0..B-1, left -> right)
        y-axis: PREDICTED band (0..B-1, bottom -> top)

    The heatmap entries for band t (true) and p (predicted) are:

        mean_{samples with y_bin = t} P(band = p | sample)
    """
    # -----------------------------
    # 1) Per-ply probability matrix
    # -----------------------------
    prob_cols_ply = [c for c in df_out.columns if c.startswith("prob_")]
    if not prob_cols_ply:
        raise ValueError("df_out must contain 'prob_*' columns.")

    prob_cols_ply = sorted(prob_cols_ply, key=lambda c: int(c.split("_")[1]))
    num_bins = len(prob_cols_ply)

    probs_ply = df_out[prob_cols_ply].to_numpy(dtype=float)   # (N_plies, B)
    y_ply = df_out["y_bin"].to_numpy(dtype=int)               # (N_plies,)
    M_ply_true_pred = _compute_soft_confusion(y_ply, probs_ply, num_bins)  # (B, B)

    # -------------------------------
    # 2) Per-game posterior matrix
    # -------------------------------
    # Use only true band-type columns post_lo_hi, not summary stats
    post_cols_game = [c for c in game_summary.columns if _is_post_band_col(c)]
    if not post_cols_game:
        raise ValueError(
            "game_summary must contain band-type posterior columns like "
            "post_400_600, post_600_800, ..."
        )

    post_cols_game = sorted(post_cols_game, key=lambda c: int(c.split("_")[1]))

    if len(post_cols_game) != num_bins:
        raise ValueError(
            f"Mismatch between per-ply bands ({num_bins}) and per-game posterior bands "
            f"({len(post_cols_game)})."
        )

    # Each row here is the final Bayesian posterior over bands for a game
    probs_game = game_summary[post_cols_game].to_numpy(dtype=float)  # (N_games, B)
    y_game = game_summary["y_bin"].to_numpy(dtype=int)               # (N_games,)
    M_game_true_pred = _compute_soft_confusion(y_game, probs_game, num_bins)  # (B, B)

    # ---------------------------------------------------
    # 3) Re-orient for plotting
    #
    # We want:
    #   - columns = TRUE band  (0..B-1 left->right)
    #   - rows    = PRED band  (0..B-1 bottom->top)
    #
    # M_true_pred[true, pred] -> H[pred, true]
    # Then flip vertically so row 0 is highest band, bottom is 0.
    # ---------------------------------------------------
    H_ply = M_ply_true_pred.T     # (pred, true)
    H_game = M_game_true_pred.T   # (pred, true)

    H_ply_plot = H_ply[::-1, :]
    H_game_plot = H_game[::-1, :]

    # ---------------------------
    # 4) Colormap (LSTM-style)
    # ---------------------------
    if cmap is None:
        darkest_hex = "#90664b"
        lightest_hex = "#e5d3b3"
        cmap = LinearSegmentedColormap.from_list(
            "rym_confusion",
            [darkest_hex, lightest_hex],
        )

    # Tick labels: use band_labels if provided, otherwise 0..B-1
    if band_labels is not None and len(band_labels) == num_bins:
        x_tick_labels = band_labels                     # true bands (left -> right)
        y_tick_labels = band_labels[::-1]               # predicted bands (top -> bottom)
    else:
        x_tick_labels = list(range(num_bins))
        y_tick_labels = list(range(num_bins - 1, -1, -1))

    # ---------------------------
    # 5) Plot side-by-side
    # ---------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        gridspec_kw={"width_ratios": [0.5, 0.5], "wspace": 0.15},
    )

    # Per-ply panel (no cbar)
    ax0 = axes[0]
    sns.heatmap(
        H_ply_plot,
        ax=ax0,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        xticklabels=x_tick_labels,
        yticklabels=y_tick_labels,
    )
    ax0.set_xlabel("True rating band")
    ax0.set_ylabel("Predicted rating band")
    ax0.set_title("Per-ply confusion (P(pred | true))")
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha="right")
    ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0, va="center", fontsize=10)

    # Per-game panel (colorbar appended to the right, but still inside the panel area)
    ax1 = axes[1]
    sns.heatmap(
        H_game_plot,
        ax=ax1,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        xticklabels=x_tick_labels,
        yticklabels=y_tick_labels,  # keep shared ticks consistent
    )

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="4%", pad=0.10)
    fig.colorbar(ax1.collections[0], cax=cax)

    ax1.set_xlabel("True rating band")
    ax1.set_ylabel("")  # shared with left
    ax1.set_title("Per-game confusion (P(pred | true))")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, va="center", fontsize=10)

    return fig, axes


# ---------------------------------------------------------------------
# Plane helpers
# ---------------------------------------------------------------------


def plane_squares_from_npz_plane(plane_2d: np.ndarray) -> List[chess.Square]:
    """
    Convert an (8,8) 0/1 plane from an NPZ into a list of chess.Square
    indices where the plane == 1.
    """
    assert plane_2d.shape == (8, 8)
    flat = plane_2d.reshape(-1)
    idxs = np.nonzero(flat)[0]
    return [chess.SQUARES[i] for i in idxs]


def plane_squares(plane_8x8: np.ndarray) -> List[chess.Square]:
    """
    For the encode_ply_planes-style planes (8x8, 0/1), convert to
    chess.Square list.
    """
    assert plane_8x8.shape == (8, 8)
    flat = plane_8x8.reshape(-1)
    idxs = np.nonzero(flat)[0]
    return [chess.SQUARES[i] for i in idxs]


# ---------------------------------------------------------------------
# 1) Conceptual view: encode from FEN + move → 64-plane grid
# ---------------------------------------------------------------------


def show_planes_grid(
    fen: str,
    move_uci: str,
    size: int = 120,
    show_labels: bool = True,
) -> None:
    """
    Render all NUM_PLANES feature planes in an 8x8 grid for a given
    (fen, move_uci) by re-encoding with encode_ply_planes.

    Each cell shows the board with that plane's squares highlighted.
    """
    board_pre = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    board_post = board_pre.copy()
    board_post.push(move)

    planes = encode_ply_planes(board_pre, move)  # (NUM_PLANES, 8, 8)

    cell_html_pieces: List[str] = []

    for plane_idx in range(NUM_PLANES):
        if plane_idx < 42:
            ctx_board = board_pre
        else:
            ctx_board = board_post

        plane = planes[plane_idx]
        sqs = plane_squares(plane)
        svg = chess.svg.board(board=ctx_board, squares=sqs, size=size)
        label = plane_label_for_index(plane_idx)

        if show_labels:
            cell_html = f"""
            <div style="border: 1px solid #aaa; padding: 2px; box-sizing: border-box;">
              <div style="font-size: 9px; text-align: center; margin-bottom: 2px;
                          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {plane_idx}: {label}
              </div>
              {svg}
            </div>
            """
        else:
            cell_html = f"""
            <div style="border: 1px solid #aaa; padding: 2px; box-sizing: border-box;">
              {svg}
            </div>
            """
        cell_html_pieces.append(cell_html)

    grid_html = f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(8, {size + 12}px);
        grid-auto-rows: auto;
        gap: 4px;
    ">
      {''.join(cell_html_pieces)}
    </div>
    """

    display(HTML(grid_html))


# ---------------------------------------------------------------------
# 2) NPZ-based view: use stored planes for a given row_index
# ---------------------------------------------------------------------


def show_npz_planes_grid_for_row(
    npz_path: str | Path,
    ply_meta: Dict[tuple[int, int], Dict[str, str]],
    row_index: int,
    size: int = 120,
    show_labels: bool = True,
    *,
    regenerate_npz: bool = True,
    pgn_path: str | Path | None = None,
    pgn_to_npz_kwargs: Dict[str, Any] | None = None,
) -> None:

    """
    Visualize all NUM_PLANES feature planes for a single ply taken from
    an NPZ.

    Inputs:
        npz_path : path to the RYM NPZ (with X, game_id, ply_idx, ...)
        ply_meta : mapping (game_id, ply_idx) -> {fen_pre, fen_post, move_uci}
        row_index : index into X (0 .. N-1) for the ply you want to show

    Output:
        Displays an 8x8 grid of boards in Jupyter, each highlighting one plane.
    """
    if regenerate_npz:
        if pgn_path is None:
            raise ValueError("regenerate_npz=True requires pgn_path=... to rebuild the NPZ.")

        # kill old file
        npz_path.unlink(missing_ok=True)

        # import the builder (support both package + repo usage)
        try:
            from .build_rym_npz import pgn_to_npz  # type: ignore
        except Exception:
            from scripts.build_rym_npz import pgn_to_npz  # type: ignore

        kwargs = dict(pgn_to_npz_kwargs or {})
        kwargs.setdefault("pgn_path", Path(pgn_path))
        kwargs.setdefault("out_path", npz_path)

        pgn_to_npz(**kwargs)

    npz_path = Path(npz_path)
    data = np.load(npz_path)

    X = data["X"]          # (N, NUM_PLANES, 8, 8)
    game_id_arr = data["game_id"]
    ply_idx_arr = data["ply_idx"]

    N = X.shape[0]
    if not (0 <= row_index < N):
        raise IndexError(f"row_index {row_index} out of range [0, {N-1}]")

    planes = X[row_index]  # (NUM_PLANES, 8, 8)
    game_id = int(game_id_arr[row_index])
    ply_idx = int(ply_idx_arr[row_index])

    meta_key = (game_id, ply_idx)
    if meta_key not in ply_meta:
        raise KeyError(f"No PGN metadata for (game_id={game_id}, ply_idx={ply_idx})")

    meta = ply_meta[meta_key]
    fen_pre = meta["fen_pre"]
    fen_post = meta["fen_post"]
    move_uci = meta["move_uci"]

    board_pre = chess.Board(fen_pre)
    board_post = chess.Board(fen_post)

    move = chess.Move.from_uci(move_uci)

    cell_html_pieces: List[str] = []

    for plane_idx in range(NUM_PLANES):
        plane = planes[plane_idx]

        # More explicit mapping for new 64-plane layout
        if 42 <= plane_idx <= 47 or 58 <= plane_idx <= 63:
            ctx_board = board_post
        else:
            ctx_board = board_pre

        sqs = plane_squares_from_npz_plane(plane)
        svg = chess.svg.board(board=ctx_board, squares=sqs, lastmove=move, size=size)
        label = plane_label_for_index(plane_idx)

        if show_labels:
            cell_html = f"""
            <div style="border: 1px solid #aaa; padding: 2px; box-sizing: border-box;">
              <div style="font-size: 9px; text-align: center; margin-bottom: 2px;
                          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {plane_idx}: {label}
              </div>
              {svg}
            </div>
            """
        else:
            cell_html = f"""
            <div style="border: 1px solid #aaa; padding: 2px; box-sizing: border-box;">
              {svg}
            </div>
            """

        cell_html_pieces.append(cell_html)

    grid_html = f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(8, {size + 12}px);
        grid-auto-rows: auto;
        gap: 4px;
    ">
      {''.join(cell_html_pieces)}
    </div>
    """

    header = f"Game {game_id}, ply {ply_idx} — move {move_uci}"

    full_html = f"""
    <div style="display: flex; flex-direction: column; gap: 8px; align-items: flex-start;">
      <div style="font-weight: bold; margin-bottom: 4px;">
        {header}
      </div>
      {grid_html}
    </div>
    """

    display(HTML(full_html))


# ---------------------------------------------------------------------
# 3) Per-game probability / Elo plots
# ---------------------------------------------------------------------


def _sorted_prob_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("prob_")]
    cols = sorted(cols, key=lambda c: int(c.split("_")[1]))
    return cols


def plot_ply_likelihoods_with_elo(
    df_game: pd.DataFrame,
    min_rating: float,
    max_rating: float,
    figsize: Tuple[int, int] = (16, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Per-move likelihoods:

      - Stacked bar plot over rating bands at each ply (prob_* columns)
      - Predicted Elo on a twin y-axis
      - Narrow color-legend column on the right

    df_game should be filtered to a single game_id.
    """
    df_plot = df_game.sort_values("ply_idx").copy()
    plies = df_plot["ply_idx"].to_numpy()
    pred_rating = df_plot["pred_rating"].to_numpy()
    pred_bin = df_plot["pred_bin"].to_numpy().astype(int)

    prob_cols = _sorted_prob_cols(df_plot)
    band_labels = [f"{c.split('_')[1]}–{c.split('_')[2]}" for c in prob_cols]

    probs = df_plot[prob_cols].to_numpy().T  # (num_bands, num_plies)
    num_bands, num_plies = probs.shape

    bottoms = np.zeros_like(probs)
    for b in range(num_bands):
        bottoms[b] = 0.0 if b == 0 else probs[:b].sum(axis=0)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 0.8], wspace=0.30)

    ax_prob = fig.add_subplot(gs[0, 0])
    ax_elo = ax_prob.twinx()
    ax_leg = fig.add_subplot(gs[0, 1])

    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.10)

    colors = [
        '#D32F2F', '#1976D2', '#388E3C', '#7B1FA2', '#F57C00',
        '#C2185B', '#00796B', '#5D4037', '#0097A7', '#AFB42B',
        '#E64A19', '#303F9F', '#689F38', '#512DA8', '#FBC02D',
        '#C62828', '#0288D1', '#00695C', '#8E24AA', '#EF6C00',
        '#AD1457', '#2E7D32', '#4527A0', '#F9A825', '#6A1B9A',
        '#00838F', '#558B2F', '#4A148C', '#BF360C', '#1565C0',
    ]
    while len(colors) < num_bands:
        colors.extend(colors)
    colors = colors[:num_bands]

    bar_width = 0.9

    for b in range(num_bands):
        height = probs[b]
        bottom = bottoms[b]

        bars = ax_prob.bar(
            plies,
            height,
            bottom=bottom,
            width=bar_width,
            color=colors[b],
            edgecolor="none",
            align="center",
        )

        for t, patch in enumerate(bars.patches):
            patch.set_alpha(0.95 if pred_bin[t] == b else 0.20)

    ax_prob.set_xlabel("Ply index", fontsize=11)
    ax_prob.set_ylabel("Probability", fontsize=11)
    ax_prob.set_xlim(plies.min() - 0.5, plies.max() + 0.5)
    ax_prob.set_ylim(0.0, 1.0)
    ax_prob.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_prob.set_xticks(np.arange(plies.min(), plies.max() + 1, 2), minor=True)

    elo_line, = ax_elo.plot(
        plies,
        pred_rating,
        color="black",
        linewidth=2,
        label="Predicted Elo",
        zorder=100,
    )
    ax_elo.set_ylabel("Predicted Elo", fontsize=11)
    ax_elo.set_ylim(min_rating - 50, max_rating + 50)
    ax_elo.legend(loc="upper left", fontsize=10)

    y_pos = np.arange(num_bands)
    ax_leg.barh(
        y_pos,
        np.ones(num_bands),
        color=colors,
        edgecolor="none",
    )
    ax_leg.set_yticks(y_pos)
    ax_leg.set_yticklabels(band_labels, fontsize=9)
    ax_leg.set_ylim(-0.5, num_bands - 0.5)   # 0-band at bottom
    ax_leg.set_xticks([])
    ax_leg.set_xlim(0, 1.0)
    ax_leg.set_title("Rating bands (Elo)", fontsize=10)
    for spine in ["top", "right", "bottom", "left"]:
        ax_leg.spines[spine].set_visible(False)
    ax_leg.xaxis.set_visible(False)

    return fig, ax_prob


def plot_bayesian_posterior_with_elo_and_legend(
    df_game: pd.DataFrame,
    alpha: float = 0.7,
    gamma: float = 1.0,
    eps: float = 1e-8,
    min_rating: float = 0.0,
    max_rating: float = 2500.0,
    figsize: Tuple[int, int] = (16, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Bayesian posterior path:

      - Stacked posterior probabilities per ply (after Bayes updates)
      - Posterior Elo mean (red) and regression head Elo (black, optional)
      - Narrow color legend on the right with 400–600 at the bottom
      - Alpha coding:
          0.95 if band == per-ply raw argmax
          0.50 if band == posterior argmax (and not raw argmax)
          0.20 otherwise
    """
    df_plot = df_game.sort_values("ply_idx").copy()
    plies = df_plot["ply_idx"].to_numpy()
    pred_bin = df_plot["pred_bin"].to_numpy().astype(int)

    prob_cols = _sorted_prob_cols(df_plot)
    band_labels = [f"{c.split('_')[1]}–{c.split('_')[2]}" for c in prob_cols]

    lik_seq = df_plot[prob_cols].to_numpy()  # T x B
    T, B = lik_seq.shape

    _, _, band_centers = _band_edges_and_centers(prob_cols)

    post_seq = _bayes_update_path(lik_seq, alpha=alpha, gamma=gamma, eps=eps)  # T x B
    elo_mean = (post_seq * band_centers).sum(axis=1)    # (T,)

    probs = post_seq.T
    num_bands, num_plies = probs.shape

    post_argmax = post_seq.argmax(axis=1)

    bottoms = np.zeros_like(probs)
    for b in range(num_bands):
        bottoms[b] = 0.0 if b == 0 else probs[:b].sum(axis=0)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 0.8], wspace=0.30)

    ax_prob = fig.add_subplot(gs[0, 0])
    ax_elo = ax_prob.twinx()
    ax_leg = fig.add_subplot(gs[0, 1])

    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.10)

    colors = [
        '#D32F2F',  '#1976D2',  '#388E3C',  '#7B1FA2',  '#F57C00',
        '#C2185B',  '#00796B',  '#5D4037',  '#0097A7',  '#AFB42B',
        '#E64A19',  '#303F9F',  '#689F38',  '#512DA8',  '#FBC02D',
        '#C62828',  '#0288D1',  '#00695C',  '#8E24AA',  '#EF6C00',
        '#AD1457',  '#2E7D32',  '#4527A0',  '#F9A825',  '#6A1B9A',
        '#00838F',  '#558B2F',  '#4A148C',  '#BF360C',  '#1565C0',
    ]
    while len(colors) < num_bands:
        colors.extend(colors)
    colors = colors[:num_bands]

    bar_width = 0.9
    band_handles: List[Any] = []

    for b in range(num_bands):
        height = probs[b]
        bottom = bottoms[b]

        bars = ax_prob.bar(
            plies,
            height,
            bottom=bottom,
            width=bar_width,
            color=colors[b],
            edgecolor="none",
            align="center",
        )

        for t, patch in enumerate(bars.patches):
            if pred_bin[t] == b:
                alpha_patch = 0.95
            elif post_argmax[t] == b:
                alpha_patch = 0.50
            else:
                alpha_patch = 0.20
            patch.set_alpha(alpha_patch)

        band_handles.append(bars)

    ax_prob.set_xlabel("Ply index", fontsize=11)
    ax_prob.set_ylabel("Posterior probability", fontsize=11)
    ax_prob.set_xlim(plies.min() - 0.5, plies.max() + 0.5)
    ax_prob.set_ylim(0.0, 1.0)
    ax_prob.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_prob.set_xticks(np.arange(plies.min(), plies.max() + 1, 2), minor=True)

    line_mean, = ax_elo.plot(
        plies,
        elo_mean,
        color="red",
        linewidth=2,
        label="Posterior Elo (mean)",
        zorder=100,
    )

    line_reg = None
    if "pred_rating" in df_plot.columns:
        line_reg, = ax_elo.plot(
            plies,
            df_plot["pred_rating"].to_numpy(),
            color="black",
            linewidth=1.5,
            alpha=0.6,
            label="Regression head Elo",
        )

    ax_elo.set_ylabel("Elo", fontsize=11)
    ax_elo.set_ylim(min_rating, max_rating)
    ax_elo.set_yticks(np.arange(min_rating, max_rating + 1, 500))

    elo_handles = [line_mean]
    elo_labels = ["Posterior Elo (mean)"]
    if line_reg is not None:
        elo_handles.append(line_reg)
        elo_labels.append("Regression head Elo")

    ax_elo.legend(
        handles=elo_handles,
        labels=elo_labels,
        loc="upper left",
        fontsize=10,
    )

    y_pos = np.arange(num_bands)
    ax_leg.barh(
        y_pos,
        np.ones(num_bands),
        color=colors,
        edgecolor="none",
    )
    ax_leg.set_yticks(y_pos)
    ax_leg.set_yticklabels(band_labels, fontsize=9)
    ax_leg.set_ylim(-0.5, num_bands - 0.5)  # band 0 at bottom
    ax_leg.set_xticks([])
    ax_leg.set_xlim(0, 1.0)
    ax_leg.set_title("Rating bands (Elo)", fontsize=10)
    for spine in ["top", "right", "bottom", "left"]:
        ax_leg.spines[spine].set_visible(False)
    ax_leg.xaxis.set_visible(False)

    return fig, ax_prob


# ---------------------------------------------------------------------
# 4) PNG board rendering (pure PIL, no cairo/cairosvg dependency)
# ---------------------------------------------------------------------


def _render_board_to_pil(board: chess.Board, lastmove=None, size: int = 320) -> Image.Image:
    """
    Render a simple 2D board with Unicode chess glyphs (♙♟♘...), no axes.
    Tries FreeSerif for piece glyphs, falls back to DejaVu Sans or the
    default PIL font if needed.
    """
    sq_size = size // 8
    img = Image.new("RGB", (sq_size * 8, sq_size * 8), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw squares
    for rank in range(8):
        for file in range(8):
            x0 = file * sq_size
            y0 = (7 - rank) * sq_size
            x1 = x0 + sq_size
            y1 = y0 + sq_size
            if (rank + file) % 2 == 0:
                color = (240, 217, 181)   # light
            else:
                color = (181, 136, 99)    # dark
            draw.rectangle([x0, y0, x1, y1], fill=color)

    # Highlight last move squares
    if lastmove is not None:
        for sq in (lastmove.from_square, lastmove.to_square):
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            x0 = file * sq_size
            y0 = (7 - rank) * sq_size
            x1 = x0 + sq_size
            y1 = y0 + sq_size
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=(50, 205, 50),
                width=max(2, sq_size // 12),
            )

    # Robust font selection for Unicode chess glyphs
    # Prefer FreeSerif (nice chess glyphs), then fall back to DejaVu Sans.
    font_size = int(sq_size * 0.8)
    font = None

    # Ordered list of candidates: (matplotlib-family-name, direct-ttf-name)
    candidates = [
        ("FreeSerif", "FreeSerif.ttf"),
        ("DejaVu Sans", "DejaVuSans.ttf"),
    ]

    for family_name, ttf_name in candidates:
        # 1) Try to locate via matplotlib's font manager
        try:
            path = font_manager.findfont(family_name, fallback_to_default=False)
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            pass
        # 2) Try a direct TTF filename in the system font path / cwd
        try:
            font = ImageFont.truetype(ttf_name, font_size)
            break
        except Exception:
            pass

    if font is None:
        font = ImageFont.load_default()

    # Draw pieces as Unicode glyphs (♙♟♘♞…)
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue

        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x0 = file * sq_size
        y0 = (7 - rank) * sq_size

        text = piece.unicode_symbol()  # e.g. ♙, ♟, …
        text_color = (0, 0, 0)

        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textsize(text, font=font)

        tx = x0 + (sq_size - w) / 2
        ty = y0 + (sq_size - h) / 2
        draw.text((tx, ty), text, font=font, fill=text_color)

    return img


# ---------------------------------------------------------------------
# Shared animation schedule (boards + distributions)
# ---------------------------------------------------------------------

def _build_posterior_frame_schedule(num_plies: int) -> list[dict]:
    """
    Common schedule used by:
      - save_board_frames_for_pgn
      - save_dist_frames_for_game

    For T plies, returns 1 + 3T frame descriptors in this order:

        0: prior_only         (initial prior, no ply)
        1: prior_lik (ply 0)  (prior before ply 0 + lik curve)
        2: post_lik  (ply 0)  (posterior after ply 0 + lik curve)
        3: post_only (ply 0)  (same posterior, no lik curve)
        4: prior_lik (ply 1)
        5: post_lik  (ply 1)
        6: post_only (ply 1)
        ...

    Each element is a dict with:
        - "type": one of "prior_only", "prior_lik", "post_lik", "post_only"
        - "ply_index": None (for initial prior) or 0..num_plies-1
    """
    frames: list[dict] = []
    frames.append({"type": "prior_only", "ply_index": None})
    for idx in range(num_plies):
        frames.append({"type": "prior_lik", "ply_index": idx})
        frames.append({"type": "post_lik", "ply_index": idx})
        frames.append({"type": "post_only", "ply_index": idx})
    return frames


def save_board_frames_for_pgn(
    pgn_path: str | Path,
    game_id: int = 0,
    out_root: str | Path = "plots/rym_inspect_conv",
    boards_subdir: str = "boards",
    prefix: str = "frame",
    size: int = 512,
) -> list[Path]:
    """
    Save one PNG board frame per *animation frame*, using the same
    schedule as save_dist_frames_for_game.

    For a game with T plies, this writes 1 + 3T PNGs:

        frame_0000.png  -> initial prior (start position)
        frame_0001.png  -> prior before ply 0 (no highlight)
        frame_0002.png  -> position AFTER ply 0 (highlight that move)
        frame_0003.png  -> same position as 0002 (duplicate)
        frame_0004.png  -> prior before ply 1
        frame_0005.png  -> after ply 1
        frame_0006.png  -> after ply 1 (duplicate)
        ...

    Some board images are deliberately duplicated so they line up
    exactly with the distribution frames.
    """
    pgn_path = Path(pgn_path)
    out_root = Path(out_root)
    boards_dir = out_root / boards_subdir
    boards_dir.mkdir(parents=True, exist_ok=True)

    # Load the requested game (0-based index)
    with pgn_path.open("r", encoding="utf-8") as f:
        game: chess.pgn.Game | None = None
        for _ in range(game_id + 1):
            game = chess.pgn.read_game(f)
            if game is None:
                raise ValueError(
                    f"PGN {pgn_path} has fewer than {game_id + 1} games."
                )

    assert game is not None
    moves: list[chess.Move] = list(game.mainline_moves())
    T = len(moves)

    # Precompute board positions after k moves: boards[k] is after k plies
    board = game.board()
    boards: list[chess.Board] = [board.copy()]  # k = 0 (start position)
    for mv in moves:
        board.push(mv)
        boards.append(board.copy())             # k = 1..T

    # Shared schedule with the dist frames
    schedule = _build_posterior_frame_schedule(T)

    written: list[Path] = []

    for frame_idx, fr in enumerate(schedule):
        kind = fr["type"]
        ply_index = fr["ply_index"]

        if ply_index is None:
            # Initial prior frame: start board, no highlight
            board_idx = 0
            lastmove: chess.Move | None = None
        else:
            t = int(ply_index)
            if kind == "prior_lik":
                # Prior before ply t: board BEFORE move t, no highlight
                board_idx = t
                lastmove = None
            else:
                # Posterior after ply t: board AFTER move t, highlight that move
                board_idx = t + 1
                lastmove = moves[t]

        img = _render_board_to_pil(boards[board_idx], lastmove=lastmove, size=size)
        out_path = boards_dir / f"{prefix}_{frame_idx:04d}.png"
        img.save(out_path)
        written.append(out_path)

    print(f"[boards] Wrote {len(written)} board frames to {boards_dir}")
    return None


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Discrete weighted median for posterior median line."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = weights > 0
    if not mask.any():
        return float(values.mean())

    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]

    cum = np.cumsum(w)
    total = cum[-1]
    if total <= 0:
        return float(v.mean())
    cutoff = 0.5 * total
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    idx = max(0, min(idx, len(v) - 1))
    return float(v[idx])


def _make_gaussian_basis(
    band_centers: np.ndarray,
    x_min: float,
    x_max: float,
    num_points: int = 512,
    width_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a Gaussian basis over rating, one column per band center.

    Columns are normalised so each contributes ~unit area; we then scale
    the mixture curves later so that the global max hits target_ymax.
    """
    band_centers = np.asarray(band_centers, dtype=float)
    xs = np.linspace(x_min, x_max, num_points)

    if len(band_centers) > 1:
        spacing = float(np.median(np.diff(band_centers)))
        if spacing <= 0:
            spacing = (x_max - x_min) / max(len(band_centers), 1)
    else:
        spacing = (x_max - x_min) / max(len(band_centers), 1)

    sigma = max(spacing * width_scale, 1e-6)
    basis = np.exp(-0.5 * ((xs[:, None] - band_centers[None, :]) / sigma) ** 2)

    dx = xs[1] - xs[0]
    col_sums = basis.sum(axis=0, keepdims=True) * dx
    basis = basis / np.maximum(col_sums, 1e-12)
    return xs, basis


def _mixture_curve(probs: np.ndarray, basis: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """Turn a band-probability vector into a smoothed curve over rating."""
    p = np.asarray(probs, dtype=float).clip(min=0.0)
    s = p.sum()
    if s <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / s
    return amplitude * (basis @ p)


def _bayes_update_one(
    prior: np.ndarray,
    lik: np.ndarray,
    alpha: float,
    gamma: float,
    eps: float,
) -> np.ndarray:
    """Single tempered Bayes update step (matching rym_analysis)."""
    prior = np.asarray(prior, dtype=float)
    lik = np.asarray(lik, dtype=float)

    lik = np.clip(lik, eps, None) ** alpha
    lik_sum = lik.sum()
    if lik_sum <= 0:
        lik = np.ones_like(lik) / len(lik)
    else:
        lik = lik / lik_sum

    p_eff = np.clip(prior, eps, None) ** gamma
    p_eff_sum = p_eff.sum()
    if p_eff_sum <= 0:
        p_eff = np.ones_like(p_eff) / len(p_eff)
    else:
        p_eff = p_eff / p_eff_sum

    post = p_eff * lik
    post_sum = post.sum()
    if post_sum <= 0:
        post = np.ones_like(post) / len(post)
    else:
        post = post / post_sum
    return post


def save_dist_frames_for_game(
    df_game: "pd.DataFrame",
    out_root: str | Path,
    dist_subdir: str = "dist",
    prefix: str = "frame",
    rating_min: float = 400.0,
    rating_max: float = 2400.0,
    alpha: float = 0.7,
    gamma: float = 1.0,
    eps: float = 1e-8,
    num_points: int = 512,
    target_ymax: float = 0.2,
    move_labels: list[str] | None = None,
    game_result: str | None = None,
) -> list[Path]:
    """
    Save a sequence of PNG frames visualising the rating distribution
    as the game unfolds.

    Frame schedule (T = # plies):

        0: initial prior (no green curve)
        For each ply t = 0..T-1:
            3t+1: prior before ply t   + per-ply likelihood curve (green)
            3t+2: posterior after ply t + same per-ply curve
            3t+3: posterior after ply t (no green curve)

    Titles:
      - Frames *with* the green curve use move notation only:
            "1. e4"  or  "1...c5"
      - Frames *without* the green curve have a non-empty stub title
        (e.g. "Start" or "1.") to keep layout consistent.

    Y-axis is fixed to [0, target_ymax] and shown as a percentage.
    """
    df_game = df_game.copy()

    # ---- Robust, safe prefix for filenames (no "1." etc.) ----
    safe_prefix = prefix.rstrip(" .")
    if not safe_prefix:
        safe_prefix = "frame"

    # Find ply index column
    ply_col: str | None = None
    for cand in ("ply_idx", "ply_index", "ply", "ply_no"):
        if cand in df_game.columns:
            ply_col = cand
            break
    if ply_col is None:
        raise ValueError("df_game must contain a ply index column, e.g. 'ply_idx'.")

    df_game = df_game.sort_values(ply_col).reset_index(drop=True)

    # Directory
    out_root = Path(out_root)
    dist_dir = out_root / dist_subdir
    dist_dir.mkdir(parents=True, exist_ok=True)

    # prob_* columns and band centers
    prob_cols = [c for c in df_game.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("df_game must contain 'prob_*' columns with band probabilities.")

    band_edges: list[tuple[int, int]] = []
    for col in prob_cols:
        parts = col.split("_")
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            band_edges.append((int(parts[1]), int(parts[2])))
    band_edges = sorted(set(band_edges))
    if not band_edges:
        raise ValueError("Could not parse band edges from prob_* column names.")

    band_centers = np.array([0.5 * (lo + hi) for lo, hi in band_edges], dtype=float)

    lik_seq = df_game[prob_cols].to_numpy(dtype=float)  # (T, B)
    T, B = lik_seq.shape

    # Gaussian basis over [rating_min, rating_max]
    xs, basis = _make_gaussian_basis(
        band_centers,
        x_min=rating_min,
        x_max=rating_max,
        num_points=num_points,
        width_scale=0.5,
    )

    # True Elo (if present)
    true_elo: float | None = None
    for col in ("y_elo", "elo", "rating"):
        if col in df_game.columns:
            true_elo = float(df_game[col].iloc[0])
            break

    # Frame schedule (types + ply indices)
    schedule = _build_posterior_frame_schedule(T)

    # Step through schedule and update priors/posteriors as needed.
    frames: list[dict] = []
    prior = np.ones(B, dtype=float) / B
    last_post: np.ndarray | None = None

    for fr in schedule:
        kind = fr["type"]
        ply_index = fr["ply_index"]

        if kind == "prior_only":
            main = prior.copy()
            ply_probs = None

        elif kind == "prior_lik":
            idx = int(ply_index)
            lik = lik_seq[idx]
            main = prior.copy()
            ply_probs = lik.copy()

        elif kind == "post_lik":
            idx = int(ply_index)
            lik = lik_seq[idx]
            post = _bayes_update_one(prior, lik, alpha=alpha, gamma=gamma, eps=eps)
            main = post.copy()
            ply_probs = lik.copy()
            last_post = post

        elif kind == "post_only":
            if last_post is None:
                raise RuntimeError("post_only encountered before any post_lik.")
            main = last_post.copy()
            ply_probs = None
            prior = last_post
        else:
            raise ValueError(f"Unknown frame type: {kind}")

        frames.append(
            {
                "type": kind,
                "ply_index": ply_index,
                "main": main,
                "ply_probs": ply_probs,
            }
        )

    # Precompute global max for consistent scaling
    global_max = 0.0
    for fr in frames:
        curve_main = _mixture_curve(fr["main"], basis, amplitude=1.0)
        global_max = max(global_max, float(curve_main.max()))
        if fr["ply_probs"] is not None:
            curve_ply = _mixture_curve(fr["ply_probs"], basis, amplitude=0.5)
            global_max = max(global_max, float(curve_ply.max()))

    if global_max <= 0.0:
        global_max = 1.0
    scale = target_ymax / global_max

    written: list[Path] = []
    last_move_title = "Start"

    for frame_idx, fr in enumerate(frames):
        main = fr["main"]
        ply_probs = fr["ply_probs"]
        ply_index = fr["ply_index"]
        kind = fr["type"]

        curve_main = _mixture_curve(main, basis, amplitude=scale)
        curve_ply = (
            _mixture_curve(ply_probs, basis=basis, amplitude=0.5 * scale)
            if ply_probs is not None
            else None
        )

        fig, ax = plt.subplots(figsize=(6, 3))

        # Main prior/posterior curve (blue, shaded)
        ax.fill_between(xs, curve_main, color="C0", alpha=0.25)
        ax.plot(xs, curve_main, color="C0", linewidth=2.0, label="posterior/prior")

        # Per-ply likelihood curve (green, half area)
        if curve_ply is not None:
            ax.plot(xs, curve_ply, color="tab:green", linewidth=1.6, label="per-ply likelihood")

        # -------- Weighted MEAN (not median) over band centers --------
        w = np.asarray(main, dtype=float)
        s = w.sum()
        if s <= 0:
            mean_rating = float(band_centers.mean())
        else:
            mean_rating = float((w * band_centers).sum() / s)

        ax.axvline(
            mean_rating,
            color="C0",
            linestyle="--",
            linewidth=1.5,
            label="posterior mean",
        )

        # True Elo (red dotted), if available
        if true_elo is not None:
            ax.axvline(
                true_elo,
                color="red",
                linestyle=":",
                linewidth=1.5,
                label="true Elo",
            )

        # Axes + style
        ax.set_xlim(rating_min, rating_max)
        ax.set_ylim(0.0, target_ymax)
        ax.set_xlabel("Rating", fontsize=10)
        ax.set_ylabel("Density (%)", fontsize=10)

        # Integer-ish xticks in rating space
        xticks = np.arange(int(rating_min), int(rating_max) + 1, 500)
        ax.set_xticks(xticks)

        # Y ticks: a few evenly spaced levels
        n_yticks = 5
        yticks = np.round(np.linspace(0.0, target_ymax * 100, n_yticks)).astype(int) / 100
        ax.set_yticks(yticks)

        # Percent formatting on y-axis (0 .. 1 mapped to 0–100%)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        # Smaller tick labels
        ax.tick_params(axis="both", labelsize=10)

        ax.grid(alpha=0.2)
        ax.set_facecolor("#f0f0f0")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ----- Title logic -----
        if ply_index is None and kind == "prior_only":
            title_str = "Start"
        else:
            idx = int(ply_index) if ply_index is not None else 0
            move_no = idx // 2 + 1
            is_white = (idx % 2 == 0)

            if kind == "prior_lik":
                # Only move number + side, no SAN
                prefix = f"{move_no}." if is_white else f"{move_no}..."
                title_str = prefix
            else:
                # Posterior frames: keep last full move title
                if (
                    move_labels is not None
                    and 0 <= idx < len(move_labels)
                    and kind == "post_lik"
                ):
                    san = move_labels[idx]
                    prefix = f"{move_no}." if is_white else f"{move_no}..."
                    last_move_title = f"{prefix} {san}"

                title_str = last_move_title

        # Override last frame with result
        if game_result is not None and frame_idx == len(frames) - 1:
            title_str = game_result

        if not title_str.strip():
            title_str = " "

        ax.set_title(title_str, fontsize=12)

        # Legend top-left, deduplicated
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc="upper left",
                frameon=False,
                fontsize=8,
            )

        fig.tight_layout()

        # *** Filenames are PURE SERIAL INDEX with safe_prefix ***
        out_path = dist_dir / f"{safe_prefix}_{frame_idx:04d}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        written.append(out_path)

    print(f"[dist] Wrote {len(written)} distribution frames to {dist_dir}")
    return None


def stitch_board_and_dist(
    root_dir: str | Path,
    boards_subdir: str = "boards",
    dist_subdir: str = "dist",
    out_subdir: str = "combined",
    flatten_factor: float = 1.0,
) -> None:
    """
    Given:
        root_dir/boards/*.png   (board after each move)
        root_dir/dist/*.png     (per-move distribution plots)

    Create:
        root_dir/combined/*.png where each frame is a vertical stack:
            [board image on top]
            [flattened (rescaled) dist image on bottom]

    Assumes that the files in boards/ and dist/ are in 1-1 correspondence
    and aligned by sorted filename order.
    """
    root_dir = Path(root_dir)
    boards_dir = root_dir / boards_subdir
    dist_dir = root_dir / dist_subdir
    out_dir = root_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    board_files = sorted(boards_dir.glob("*.png"))
    dist_files = sorted(dist_dir.glob("*.png"))

    if not board_files or not dist_files:
        print(f"[stitch] No PNGs found in {boards_dir} or {dist_dir}")
        return

    n = min(len(board_files), len(dist_files))
    print(f"[stitch] Found {len(board_files)} board frames, {len(dist_files)} dist frames; stitching {n}")

    for bf, df in zip(board_files[:n], dist_files[:n]):
        board_img = Image.open(bf).convert("RGBA")
        dist_img = Image.open(df).convert("RGBA")

        # Make widths match
        w = max(board_img.width, dist_img.width)

        if board_img.width != w:
            new_h = int(board_img.height * (w / board_img.width))
            board_img = board_img.resize((w, new_h), Image.BICUBIC)

        if dist_img.width != w:
            new_h = int(dist_img.height * (w / dist_img.width))
            dist_img = dist_img.resize((w, new_h), Image.BICUBIC)

        # Optionally squash the dist image vertically
        if 0.0 < flatten_factor < 1.0:
            new_h = max(1, int(dist_img.height * flatten_factor))
            dist_img = dist_img.resize((w, new_h), Image.BICUBIC)

        combined_height = board_img.height + dist_img.height
        combined = Image.new("RGBA", (w, combined_height), (255, 255, 255, 255))
        combined.paste(board_img, (0, 0))
        combined.paste(dist_img, (0, board_img.height))

        out_name = out_dir / bf.name
        combined.save(out_name)

    print(f"[stitch] Wrote stitched frames to {out_dir}")


def replay_images(
    image_dir: str | Path,
    delay: float = 0.05,
    display_width: int = 360,
) -> None:
    """
    Simple Jupyter helper to replay PNGs in a folder as an animation.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing PNG frames.
    delay : float
        Delay between frames (seconds).
    display_width : int
        Width in pixels for display inside the notebook.
    """
    image_dir = Path(image_dir)
    files = sorted(image_dir.glob("*.png"))
    if not files:
        print(f"[replay] No PNGs found in {image_dir}")
        return

    for f in files:
        clear_output(wait=True)
        display(IPyImage(filename=str(f), width=display_width))
        time.sleep(delay)