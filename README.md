# Rating Your Moves (RYM)

This repository implements a complete data pipeline for studying the relationship between human moves and playing strength in online chess, using large‐scale Lichess game archives.

The workflow currently covers:

1. **Game indexing and basic statistics** for monthly Lichess dumps.
2. **Filtering and balancing** games across rating bands and time controls.
3. **Move-level feature encoding** into a fixed 64 × 8 × 8 binary tensor per ply.

The next stage (outlined here for completeness) is a **transformer-based model** that consumes these per-ply tensors as a sequence and infers a posterior distribution over player rating bands, updated move by move.

---

## 1. Data Source

The pipeline assumes access to the official Lichess monthly database:

* Standard rated games in `.pgn.zst` format (e.g.
  `lichess_db_standard_rated_2017-04.pgn.zst`).

Archives are typically downloaded from:

* `https://database.lichess.org/standard/`

Each monthly file contains tens of millions of games; the current code is designed to:

* Decompress the `.zst` archive once,
* Build an efficient row-wise index with offsets into the `.pgn`, and
* Store game-level metadata in Parquet for fast filtering and aggregation.

---

## 2. Environment & Installation

### 2.1. Dependencies

Key Python dependencies (see `requirements.txt`):

* Core data & utilities

  * `pandas`, `numpy`, `tqdm`, `python-chess`, `zstandard`, `requests`, `pyarrow`, `matplotlib`
* Modeling (for later stages)

  * `torch`, `transformers`

Install into a virtual environment, for example:

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows

pip install -r requirements.txt
```

The repository assumes a directory layout roughly of the form:

```text
src/
  scripts/
    build_month_index.py
    index_stats.py
    filter_games.py
    build_balanced_pgn.py
    ply_features.py
data/
  lichess_db_standard_rated_YYYY-MM.pgn.zst
  lichess_db_standard_rated_YYYY-MM.pgn
  lichess_db_standard_rated_YYYY-MM_index.parquet
plots/
  YYYY-MM/...
```

Paths are configurable via script arguments or constants in the code.

---

## 3. Monthly Indexing

### 3.1. Script: `build_month_index.py`

**Purpose**

* Download (or reuse) the monthly `.pgn.zst` file for a given month.
* Decompress to `.pgn` if needed.
* Sequentially scan the PGN and build a CSV/Parquet index with one row per game, including:

  * File offsets (`start_offset`, `end_offset`),
  * PGN tags like `Event`, `Site`, `Result`, `TimeControl`,
  * Parsed metadata: player ratings, increment/base time, etc.,
  * Boolean flags like `has_clock`, `has_eval` (engine annotations present).

**Usage**

```bash
python -m scripts.build_month_index YYYY-MM
# Example
python -m scripts.build_month_index 2017-04
```

**Outputs**

* `data/lichess_db_standard_rated_YYYY-MM.pgn`
* `data/lichess_db_standard_rated_YYYY-MM_index.csv`
* `data/lichess_db_standard_rated_YYYY-MM_index.parquet`

The Parquet index is the main input for subsequent scripts.

---

## 4. Index Statistics & Sanity Checks

### 4.1. Script: `index_stats.py`

**Purpose**

* Load one or more monthly Parquet index files.
* Optionally apply a filter (SQL-like expression on columns).
* Compute descriptive statistics for:

  * Rating distributions (via rating bands),
  * Time control distributions,
  * Rating × time-control heatmaps.

**Rating bands**

Games are assigned to rating bands based on the **average Elo** of White and Black:

* `avg_elo = (white_elo + black_elo) / 2`
* Bands: `[600–700), [700–800), …, [2900–3000), [3000–3100)`
* Games with large rating disparity (e.g. `|white_elo - black_elo| > 2 * band_width`) are dropped from stats to avoid mixing very uneven pairings.

**Usage Examples**

Full month:

```bash
python -m scripts.index_stats 2017-04
```

Only games with engine evaluations:

```bash
python -m scripts.index_stats 2017-04 --where 'has_eval'
```

**Outputs**

* Rating histogram (`rating_hist.png`).
* Time control bar charts (by base+increment and by frequency).
* Rating × time control heatmaps in **percentage** and **log-scale**:

  * `rating_tc_heatmap_pct.png`
  * `rating_tc_heatmap_log.png`

Drop summaries are printed (e.g. missing ratings, unparsable time controls, large rating differences) to make the data filtering explicit.

---

## 5. Game Filtering

### 5.1. Script: `filter_games.py`

**Purpose**

* Load a Parquet index for a month.
* Apply SQL-like filters on columns to select a subset of games.
* Either:

  * Return counts only (`--count-only`), or
  * Export the matching games as a new PGN file (using file offsets into the original `.pgn`).

**Filters**

Filters are given as expressions over index columns, for example:

* Only 5+0 games:

  ```bash
  --where 'time_control == "300+0"'
  ```

* 1900–2000 White Elo, White wins, 5+0 blitz, with clock and engine eval:

  ```bash
  --where 'white_elo >= 1900 and white_elo < 2000 and result == "1-0" and time_control == "300+0" and has_clock and has_eval'
  ```

**Usage**

Count only:

```bash
python -m scripts.filter_games 2017-04 \
  --where 'time_control == "300+0" and has_eval' \
  --count-only \
  --preview 5
```

Export games to PGN:

```bash
python -m scripts.filter_games 2017-04 \
  --where 'time_control == "300+0" and has_eval' \
  --out-pgn data/filtered_2017-04_5+0_eval.pgn
```

The preview mechanism shows a small number of sample games to visually inspect the filter.

---

## 6. Balanced PGN Construction

### 6.1. Script: `build_balanced_pgn.py`

**Purpose**

* Construct a **balanced training dataset** across rating bands and other constraints.
* Sample an equal (or capped) number of games per band, optionally restricted by:

  * Time control,
  * Engine evaluations,
  * Clock availability, etc.

**High-level behaviour**

* Read monthly Parquet indexes and associated PGNs.
* Compute the rating band (`avg_elo`) for each game using the same logic as `index_stats.py`.
* For each band, sample up to `N` games (user-configurable).
* Write a combined PGN containing games from all bands in balanced proportions.

This balanced PGN is intended as the primary input for move-level encoding and model training.

---

## 7. Ply-Level Feature Encoding

### 7.1. Script: `ply_features.py`

**Purpose**

Encode a single ply (one half-move) into a fixed **64 × 8 × 8** binary feature tensor.

Input:

* `board`: `chess.Board` in the **pre-move** state
* `move`: `chess.Move` to be played from that position
* Optional: `prev_move` (or `(prev_from, prev_to)`), if known

Output:

* `feats`: `np.ndarray` of shape `(64, 8, 8)`, `dtype=uint8`

#### 7.1.1. Conceptual definitions

* **Presence** (geometric, empty board):
  Squares the piece *could* capture on `if the board were empty` and an enemy piece were there.
* **Any control** (attack/defense, current board):
  Squares the piece currently attacks (including own pieces as defended).
* **Legal control** (attack/defense respecting pins):
  Any control, but filtered to remove moves that violate absolute pins:

  * For pinned rooks/bishops/queens/pawns: only squares along the pin line (from `board.pin`).
  * For pinned knights: no legal control.
* **Net control** (pawns only):
  A square is net-controlled by a side if and only if that side’s **pawns** legally control it and the opponent’s pawns do not.
  Squares with pawn control from both sides, or from neither, are neutral.

#### 7.1.2. Plane layout (64 planes)

All planes are 0/1.

**Board, presence, control (pre-move)**

* `0–11`: `board_pre`

  * One plane per (color, piece_type), 6 types × 2 colors:

    * PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING.
* `12–23`: `presence_pre`

  * Empty-board capture reach per (color, piece_type).
* `24–35`: `control_pre`

  * Any control (attacks/defenses) per (color, piece_type).

**Aggregates (pre/post)**

* `36–41`: pre-move aggregates:

  * 36: `white_any_control_pre`
  * 37: `black_any_control_pre`
  * 38: `white_legal_control_pre`
  * 39: `black_legal_control_pre`
  * 40: `white_net_control_pre`  (pawns only)
  * 41: `black_net_control_pre`  (pawns only)
* `42–47`: post-move aggregates (same semantics after applying the move):

  * 42: `white_any_control_post`
  * 43: `black_any_control_post`
  * 44: `white_legal_control_post`
  * 45: `black_legal_control_post`
  * 46: `white_net_control_post`
  * 47: `black_net_control_post`

**Move squares (current + previous)**

* `48`: from-square of the current move (one-hot).
* `49`: to-square of the current move (one-hot).
* `50`: from-square of the previous move (one-hot; all 0s if none).
* `51`: to-square of the previous move (one-hot; all 0s if none).

**Rule state (pre-move)**

* `52`: `pre_white_can_castle_k` (all 1s if allowed, else 0s).
* `53`: `pre_white_can_castle_q`
* `54`: `pre_black_can_castle_k`
* `55`: `pre_black_can_castle_q`
* `56`: `pre_side_to_move` (all 1s if White to move, else 0).
* `57`: `pre_en_passant_target` (one-hot at the en passant target square, if any).

**Rule state (post-move)**

* `58`: `post_white_can_castle_k`
* `59`: `post_white_can_castle_q`
* `60`: `post_black_can_castle_k`
* `61`: `post_black_can_castle_q`
* `62`: `post_side_to_move` (all 1s if White to move after the move).
* `63`: `post_en_passant_target` (post-move en passant target square, if any).

#### 7.1.3. Usage

In Python:

```python
import chess
import numpy as np
from scripts.ply_features import encode_ply_planes, NUM_PLANES

board = chess.Board()                   # initial position
move = chess.Move.from_uci("e2e4")      # example move
# prev_move can be None or a chess.Move; the encoder handles that
planes = encode_ply_planes(board, move, prev_move=None)  # (64, 8, 8) uint8
```

A Jupyter notebook can visualize individual planes by mapping `1`s back to `chess.Square` indices and overlaying them with `chess.svg.board`.

---

## 8. Dataset Construction for Modeling

The end goal of the preprocessing pipeline is a **move-level dataset** suitable for sequence models:

* Each **game** becomes a sequence of plies:
  `[(features_0, meta_0), (features_1, meta_1), ..., (features_{T-1}, meta_{T-1})]`
* Each **ply** has:

  * Features: `X_t ∈ {0,1}^{64×8×8}` (often flattened or projected before input to the model).
  * Metadata: whose move it is, game ID, rating band label, etc.
* Each **player** in a game has a ground-truth rating or rating band.

A typical downstream dataset for PyTorch would contain:

* An array of game sequences, possibly padded/truncated to a maximum length.
* A per-game label (e.g. rating band index) for:

  * White,
  * Black, or
  * Both.

---

## 9. Planned Modeling: Transformer-Based Rating Inference

The modeling component is not yet implemented in this repository, but its intended structure is:

### 9.1. Problem formulation

* For each game (g) and player (White/Black), there is an unknown **rating band** (y \in {0, \dots, K-1}).
* The game is a sequence of plies ((X_0, X_1, \dots, X_{T-1})) encoded as 64×8×8 binary planes.
* The goal is to infer a **posterior over rating bands** (p(y \mid X_{0:T-1})), with a **Bayesian flavour**:

  * Each ply contributes information to the rating posterior.
  * Early moves yield broad, high-entropy predictions; characteristic errors or brilliancies cause sudden updates.

### 9.2. Data representation for the model

* Each 64×8×8 tensor is typically flattened to a vector and projected into a latent embedding, e.g.:

  ```text
  X_t (64×8×8) → x_t (d_in) → linear → e_t (d_model)
  ```

* A **sequence of embeddings** ((e_0, e_1, ..., e_{T-1})) is fed into a transformer encoder.

Additional scalars (e.g. move number, elapsed time, evaluation differences) can be concatenated or added via separate embeddings on the model side.

### 9.3. Model sketch

* Backbone: Transformer encoder over plies:

  * Positional encoding by ply index.
  * Multi-head self-attention over entire game history.
* Heads:

  * **Per-ply head (optional)**: (p(y \mid X_{0:t})) for each ply (t).
  * **Game-level head**: (p(y \mid X_{0:T-1})), possibly using:

    * CLS-style representation,
    * Global pooling (e.g. mean/max over time),
    * Or an aggregation of per-ply posteriors.

The model can be interpreted in a Bayesian way if per-ply outputs are treated as likelihoods and combined across time; the implementation can either make this explicit or simply train the transformer to output the correct band at every step.

### 9.4. Training objective

* **Classification** over rating bands:

  * Cross-entropy loss on band index:

    * Per-game loss (on final head).
    * Optionally, sum or average of per-ply losses (later plies can be weighted higher).
* **Regression proxy** (optional):

  * Use band midpoints as approximations to Elo scores, and add an MSE or MAE term on predicted Elo.

### 9.5. High-level training script structure

A typical training script (not yet included in this repo) would contain:

1. **Config & arguments**

   * Model hyperparameters: `d_model`, `n_layers`, `n_heads`, `dropout`.
   * Data paths: balanced PGNs, serialized feature datasets.
   * Training parameters: batch size, learning rate, max sequence length, number of epochs.

2. **Dataset & DataLoader**

   * A PyTorch `Dataset` that yields:

     * Tensor of shape `(T, 64, 8, 8)` (or `(T, d_in)` after flatten).
     * Rating band label for White/Black.
   * A collate function:

     * Pads/truncates to a fixed `max_T`.
     * Produces masks for valid timesteps.

3. **Model definition**

   * `nn.Module` containing:

     * Input projection from 64×8×8 to `d_model`.
     * Positional encodings.
     * Transformer encoder stack.
     * Classification head(s) over rating bands.

4. **Training loop**

   * Forward pass on mini-batches of games.
   * Compute loss (cross-entropy, optionally with per-ply aggregation).
   * Backpropagation and optimizer step.
   * Periodic evaluation on a validation set:

     * Top-1 accuracy by rating band.
     * Mean absolute error in Elo.
     * Calibration metrics (e.g. expected calibration error).

5. **Checkpointing & logging**

   * Save model checkpoints.
   * Log training curves and validation metrics.

---

## 10. Summary

* The repository provides a **high-throughput pipeline** from raw Lichess archives to:

  * Indexed, filterable game metadata,
  * Balanced PGNs by rating band, and
  * Rich 64-plane per-ply feature tensors capturing:

    * Piece presence,
    * Empty-board geometric reach,
    * Attack/defense and pin-filtered legal control,
    * Pawn-based net control,
    * Move squares, castling rights, side-to-move, and en passant state.
* The intended modeling stage uses these tensors as inputs to **transformer-based sequence models** that track a player’s rating posterior over the course of a game.