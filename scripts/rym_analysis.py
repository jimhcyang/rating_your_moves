# scripts/rym_analysis.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .ply_features import NUM_PLANES  # type: ignore
from .rym_models import get_model     # type: ignore


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    num_planes: int | None = None,
    num_bins: int | None = None,
    config_id: int | None = None,
    device: str | torch.device = "cuda",
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Generic loader for RYM models trained via run_rym_experiments.

    Supports checkpoints shaped like:
        {
            "model_state": <state_dict>,
            "model_type": "resnet" | "conv_transformer" | ...,
            "num_planes": 64,
            "num_bins": 10,
            "config_id": 0,
            ...
        }

    Also supports "model_state_dict" or bare state_dict, but in that case
    you MUST provide num_planes / num_bins / config_id.
    """
    device = torch.device(device)
    ckpt_path = Path(ckpt_path)

    state = torch.load(ckpt_path, map_location=device)

    meta: Dict[str, Any] = {}

    # Case 1: training-style checkpoint with "model_state"
    if isinstance(state, dict) and "model_state" in state:
        meta["model_type"] = state.get("model_type", "resnet")
        meta["num_planes"] = state.get("num_planes", num_planes)
        meta["num_bins"] = state.get("num_bins", num_bins)
        meta["config_id"] = state.get("config_id", config_id if config_id is not None else 0)

        if meta["num_planes"] is None or meta["num_bins"] is None:
            raise ValueError("Checkpoint is missing num_planes/num_bins and none were provided.")

        model = get_model(
            model_type=meta["model_type"],
            num_planes=meta["num_planes"],
            num_bins=meta["num_bins"],
            config_id=meta["config_id"],
        ).to(device)

        model.load_state_dict(state["model_state"])
        model.eval()
        return model, meta

    # Case 2: "model_state_dict" style
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
        if num_planes is None or num_bins is None:
            raise ValueError("Need num_planes and num_bins to rebuild model for this checkpoint.")

        meta["model_type"] = "resnet"
        meta["num_planes"] = num_planes
        meta["num_bins"] = num_bins
        meta["config_id"] = config_id if config_id is not None else 0

        model = get_model(
            model_type=meta["model_type"],
            num_planes=num_planes,
            num_bins=num_bins,
            config_id=meta["config_id"],
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()
        return model, meta

    # Case 3: bare state_dict
    if num_planes is None or num_bins is None:
        raise ValueError("Need num_planes and num_bins to rebuild model for bare state_dict")

    meta["model_type"] = "resnet"
    meta["num_planes"] = num_planes
    meta["num_bins"] = num_bins
    meta["config_id"] = config_id if config_id is not None else 0

    model = get_model(
        model_type=meta["model_type"],
        num_planes=num_planes,
        num_bins=num_bins,
        config_id=meta["config_id"],
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model, meta


# ---------------------------------------------------------------------
# Per-ply prediction on an NPZ
# ---------------------------------------------------------------------


def predict_probs_for_npz(
    npz_path: str | Path,
    ckpt_path: str | Path,
    batch_size: int = 256,
    device: str | torch.device | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run a trained RYM model on all plies in an NPZ and return a tidy DataFrame.

    Output schema:
        game_id, ply_idx, y_bin, y_elo,
        pred_bin, pred_rating,
        prob_<lo>_<hi>  (one column per Elo band, e.g. prob_400_600)

    Also returns a `meta` dict with min_rating, max_rating, num_bins, band_edges, model_type, ...
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    X = data["X"]            # (N, NUM_PLANES, 8, 8)
    y_bin = data["y_bin"]
    y_elo = data["y_elo"]
    game_id = data["game_id"]
    ply_idx = data["ply_idx"]

    num_bins_npz = int(data["num_bins"])
    min_rating = float(data["min_rating"])
    max_rating = float(data["max_rating"])

    N, C, H, W = X.shape
    assert C == NUM_PLANES, f"Expected {NUM_PLANES} planes, got {C}"

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    # Load model (no config_id needed – we read it from the checkpoint)
    model, ckpt_meta = load_model_from_checkpoint(
        ckpt_path=ckpt_path,
        num_planes=NUM_PLANES,
        num_bins=num_bins_npz,
        config_id=None,
        device=device,
    )

    # Dataset + loader
    X_tensor = torch.from_numpy(X.astype(np.float32))
    ds = TensorDataset(X_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs: List[np.ndarray] = []
    all_rating_preds: List[np.ndarray] = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits, rating_pred = model(xb)  # (B, num_bins), (B, 1)
            probs = torch.softmax(logits, dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_rating_preds.append(rating_pred.squeeze(-1).cpu().numpy())

    probs_full = np.vstack(all_probs)              # (N, num_bins)
    rating_pred_full = np.concatenate(all_rating_preds)  # (N,)

    assert probs_full.shape[0] == N
    num_bins = probs_full.shape[1]

    # Compute band edges from NPZ metadata
    span = max_rating - min_rating
    width = span / num_bins
    band_edges: List[Tuple[int, int]] = []
    for b in range(num_bins):
        lo = int(min_rating + b * width)
        hi = int(min_rating + (b + 1) * width)
        band_edges.append((lo, hi))

    rows: List[Dict[str, Any]] = []
    for i in range(N):
        row: Dict[str, Any] = {
            "game_id": int(game_id[i]),
            "ply_idx": int(ply_idx[i]),
            "y_bin": int(y_bin[i]),
            "y_elo": float(y_elo[i]),
            "pred_bin": int(probs_full[i].argmax()),
            "pred_rating": float(rating_pred_full[i]),
        }
        for b, (lo, hi) in enumerate(band_edges):
            row[f"prob_{lo}_{hi}"] = float(probs_full[i, b])
        rows.append(row)

    df_out = pd.DataFrame(rows)

    meta: Dict[str, Any] = {
        "min_rating": min_rating,
        "max_rating": max_rating,
        "num_bins": num_bins,
        "band_edges": band_edges,
        "model_type": ckpt_meta.get("model_type", "unknown"),
        "config_id": ckpt_meta.get("config_id", 0),
        "ckpt_path": str(ckpt_path),
        "npz_path": str(npz_path),
    }
    return df_out, meta


# ---------------------------------------------------------------------
# Bayesian updates over plies
# ---------------------------------------------------------------------


def _band_edges_and_centers(prob_cols: List[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lows = np.array([int(c.split("_")[1]) for c in prob_cols], dtype=float)
    highs = np.array([int(c.split("_")[2]) for c in prob_cols], dtype=float)
    centers = 0.5 * (lows + highs)
    return lows, highs, centers


def _bayes_update_path(
    lik_seq: np.ndarray,
    prior: np.ndarray | None = None,
    alpha: float = 0.7,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Sequential Bayes update over plies with tempering.

    lik_seq: (T, B) array of per- ply likelihoods (model softmax outputs)
    prior  : optional prior over bands; defaults to uniform

    alpha < 1.0  -> flattens new evidence (more cautious)
    gamma < 1.0  -> flattens old posterior (prevents runaway spikes)
    eps          -> floor to avoid zeros
    """
    T, B = lik_seq.shape
    post_seq = np.zeros_like(lik_seq, dtype=np.float64)

    if prior is None:
        p_prev = np.ones(B, dtype=np.float64) / B
    else:
        p_prev = np.asarray(prior, dtype=np.float64)
        p_prev /= p_prev.sum()

    for t in range(T):
        lik = np.clip(lik_seq[t].astype(np.float64), eps, 1.0)
        lik = lik ** alpha
        lik /= lik.sum()

        p_eff = np.clip(p_prev, eps, 1.0) ** gamma
        p_eff /= p_eff.sum()

        p = p_eff * lik
        s = float(p.sum())
        if s <= 0.0:
            p[:] = 1.0 / B
        else:
            p /= s

        post_seq[t] = p
        p_prev = p

    return post_seq


def _discrete_quantile(p: np.ndarray, support: np.ndarray, q: float) -> float:
    cdf = np.cumsum(p)
    idx = np.searchsorted(cdf, q, side="left")
    idx = min(max(idx, 0), len(support) - 1)
    return float(support[idx])


# ---------------------------------------------------------------------
# Per-game posterior summary
# ---------------------------------------------------------------------


def summarize_game_posteriors(
    df: pd.DataFrame,
    alpha: float = 0.7,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    df must have columns:
        - game_id, ply_idx, y_bin, y_elo
        - pred_rating (optional for regression metrics)
        - prob_* columns (likelihoods per ply over Elo bands)

    Returns one row per game with:
        - posterior mean Elo and 95% CI (from final posterior)
        - regression Elo (last ply & mean)
        - y_bin, y_elo
        - final posterior probability for each band as post_<lo>_<hi>
    """
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    prob_cols = sorted(prob_cols, key=lambda c: int(c.split("_")[1]))

    band_lows, band_highs, band_centers = _band_edges_and_centers(prob_cols)

    records: List[Dict[str, Any]] = []

    for game_id, df_g in df.groupby("game_id"):
        df_g = df_g.sort_values("ply_idx")
        lik_seq = df_g[prob_cols].to_numpy()  # T x B

        post_seq = _bayes_update_path(lik_seq, alpha=alpha, gamma=gamma, eps=eps)
        final_post = post_seq[-1]  # (B,)

        post_mean_elo = float((final_post * band_centers).sum())
        post_lower_elo = _discrete_quantile(final_post, band_centers, 0.025)
        post_upper_elo = _discrete_quantile(final_post, band_centers, 0.975)

        y_bin = int(df_g["y_bin"].iloc[0])
        y_elo = float(df_g["y_elo"].iloc[0])

        if "pred_rating" in df_g.columns:
            reg_last_elo = float(df_g["pred_rating"].iloc[-1])
            reg_mean_elo = float(df_g["pred_rating"].mean())
        else:
            reg_last_elo = np.nan
            reg_mean_elo = np.nan

        rec: Dict[str, Any] = {
            "game_id": game_id,
            "y_bin": y_bin,
            "y_elo": y_elo,
            "post_mean_elo": post_mean_elo,
            "post_lower_elo": post_lower_elo,
            "post_upper_elo": post_upper_elo,
            "reg_last_elo": reg_last_elo,
            "reg_mean_elo": reg_mean_elo,
        }

        for col, p in zip(prob_cols, final_post):
            # prob_400_600 -> post_400_600
            rec[f"post_{col.split('prob_')[1]}"] = float(p)

        records.append(rec)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------


def _is_post_band_col(col: str) -> bool:
    """
    Returns True only for band-type posterior columns:

        post_400_600, post_600_800, ...

    and False for summary columns like post_mean_elo, post_lower_elo, etc.
    """
    if not col.startswith("post_"):
        return False
    parts = col.split("_")
    if len(parts) != 3:
        return False
    return parts[1].isdigit() and parts[2].isdigit()


def compute_metrics(df_out: pd.DataFrame, game_summary: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregate per-ply and per-game metrics."""
    metrics: Dict[str, float] = {}

    # Basic counts
    num_plies = len(df_out)
    num_games = len(game_summary)
    avg_plies_per_game = float(num_plies) / float(num_games) if num_games > 0 else float("nan")

    metrics["num_plies"] = float(num_plies)
    metrics["num_games"] = float(num_games)
    metrics["avg_plies_per_game"] = float(avg_plies_per_game)

    # -------- Per-ply metrics --------
    y_bin = df_out["y_bin"].to_numpy()
    pred_bin = df_out["pred_bin"].to_numpy()
    y_elo = df_out["y_elo"].to_numpy()
    pred_rating = df_out["pred_rating"].to_numpy()

    metrics["ply_band_accuracy"] = float((pred_bin == y_bin).mean())
    metrics["ply_reg_mae"] = float(np.abs(pred_rating - y_elo).mean())

    # -------- Per-game metrics from posterior + regression --------
    y_elo_g = game_summary["y_elo"].to_numpy()
    post_mean_elo = game_summary["post_mean_elo"].to_numpy()
    reg_last_elo = game_summary["reg_last_elo"].to_numpy()
    reg_mean_elo = game_summary["reg_mean_elo"].to_numpy()

    metrics["game_post_mae"] = float(np.abs(post_mean_elo - y_elo_g).mean())
    metrics["game_reg_last_mae"] = float(np.abs(reg_last_elo - y_elo_g).mean())
    metrics["game_reg_mean_mae"] = float(np.abs(reg_mean_elo - y_elo_g).mean())

    # -------- Posterior band accuracy (final posterior vs y_bin) --------
    post_cols = [c for c in game_summary.columns if c.startswith("post_")]

    post_band_cols: list[str] = []
    for c in post_cols:
        parts = c.split("_")
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            post_band_cols.append(c)

    post_band_cols = sorted(post_band_cols, key=lambda c: int(c.split("_")[1]))

    if post_band_cols:
        post_probs = game_summary[post_band_cols].to_numpy()
        post_argmax = post_probs.argmax(axis=1)
        game_y_bin = game_summary["y_bin"].to_numpy()
        metrics["game_post_band_accuracy"] = float((post_argmax == game_y_bin).mean())
    else:
        metrics["game_post_band_accuracy"] = float("nan")

    return metrics


def compute_band_distribution_table(
    df_out: pd.DataFrame,
    game_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per rating band, columns:

      band_index
      band_label        "lo-hi"
      ply_true_pct      fraction of plies with y_bin == band
      ply_pred_pct      fraction of plies with pred_bin == band
      game_true_pct     fraction of games with y_bin == band
      ply_pred_prob_pct average prob mass per ply assigned to that band
    """
    prob_cols = [c for c in df_out.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("df_out has no prob_* columns; cannot build band distribution table.")

    prob_cols = sorted(prob_cols, key=lambda c: int(c.split("_")[1]))
    band_lows, band_highs, _ = _band_edges_and_centers(prob_cols)

    total_plies = len(df_out)
    total_games = len(game_summary)

    rows: List[Dict[str, Any]] = []
    for b, col in enumerate(prob_cols):
        lo = int(band_lows[b])
        hi = int(band_highs[b])
        label = f"{lo}-{hi}"

        if total_plies > 0:
            ply_true_pct = float((df_out["y_bin"] == b).mean())
            ply_pred_pct = float((df_out["pred_bin"] == b).mean())
            ply_pred_prob_pct = float(df_out[col].mean())
        else:
            ply_true_pct = float("nan")
            ply_pred_pct = float("nan")
            ply_pred_prob_pct = float("nan")

        if total_games > 0:
            game_true_pct = float((game_summary["y_bin"] == b).mean())
        else:
            game_true_pct = float("nan")

        rows.append(
            {
                "band_index": b,
                "band_label": label,
                "ply_true_pct": ply_true_pct,
                "ply_pred_pct": ply_pred_pct,
                "game_true_pct": game_true_pct,
                "ply_pred_prob_pct": ply_pred_prob_pct,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# High-level convenience wrapper
# ---------------------------------------------------------------------


def evaluate_model_on_npz(
    npz_path: str | Path,
    ckpt_path: str | Path,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    alpha: float = 0.7,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """
    End-to-end evaluation for a single NPZ:

        1) Run the model to get per-ply probabilities (df_out)
        2) Run Bayesian updates per game (game_summary)
        3) Compute aggregate metrics (metrics)

    Returns:
        df_out, game_summary, metrics, meta
    """
    df_out, meta = predict_probs_for_npz(
        npz_path=npz_path,
        ckpt_path=ckpt_path,
        batch_size=batch_size,
        device=device,
    )
    game_summary = summarize_game_posteriors(
        df_out,
        alpha=alpha,
        gamma=gamma,
        eps=eps,
    )
    metrics = compute_metrics(df_out, game_summary)
    return df_out, game_summary, metrics, meta


from .build_rym_npz import pgn_to_npz  # put this at the top of the file with other imports


def evaluate_model_on_pgn(
    pgn_path: str | Path,
    ckpt_path: str | Path,
    npz_path: str | Path | None = None,
    min_rating: int = 400,
    max_rating: int = 2400,
    num_bins: int = 10,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    alpha: float = 0.7,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """
    Convenience wrapper for the "single custom game" use case:

      - Takes a PGN path (e.g. 'gte.pgn')
      - Builds an NPZ next to it (e.g. 'gte.npz') if needed
      - Runs evaluate_model_on_npz on that NPZ
      - Returns the same outputs as evaluate_model_on_npz, with extra metadata

    This makes the custom PGN the *only* data source – perfect for demos.
    """
    pgn_path = Path(pgn_path)
    if npz_path is None:
        npz_path = pgn_path.with_suffix(".npz")
    else:
        npz_path = Path(npz_path)

    if not npz_path.exists():
        log.info(
            "NPZ %s not found, building from PGN %s (min=%d, max=%d, bins=%d)",
            npz_path,
            pgn_path,
            min_rating,
            max_rating,
            num_bins,
        )
        pgn_to_npz(
            pgn_path=pgn_path,
            out_path=npz_path,
            max_games=None,   # or 1 if you *know* there's only one game
            min_rating=min_rating,
            max_rating=max_rating,
            num_bins=num_bins,
        )

    df_out, game_summary, metrics, meta = evaluate_model_on_npz(
        npz_path=npz_path,
        ckpt_path=ckpt_path,
        batch_size=batch_size,
        device=device,
        alpha=alpha,
        gamma=gamma,
        eps=eps,
    )

    meta = dict(meta)
    meta.update(
        {
            "pgn_path": str(pgn_path),
            "npz_path": str(npz_path),
            "min_rating_used": min_rating,
            "max_rating_used": max_rating,
            "num_bins_used": num_bins,
        }
    )
    return df_out, game_summary, metrics, meta

# ---------------------------------------------------------------------
# Optional CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained RYM model on an NPZ test shard.",
    )
    p.add_argument("--npz", type=str, required=True, help="Path to NPZ shard.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[rym_analysis] %(levelname)s: %(message)s",
    )
    args = parse_args()

    df_out, game_summary, metrics, meta = evaluate_model_on_npz(
        npz_path=args.npz,
        ckpt_path=args.ckpt,
        batch_size=args.batch_size,
        device=args.device,
        alpha=args.alpha,
        gamma=args.gamma,
        eps=args.eps,
    )

    log.info("NPZ: %s", meta["npz_path"])
    log.info("CKPT: %s", meta["ckpt_path"])
    for k in sorted(metrics.keys()):
        log.info("  %s = %.4f", k, metrics[k])


if __name__ == "__main__":
    main()
