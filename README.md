# Rating Your Moves (RYM)

This repository implements a complete data + modeling pipeline for studying the relationship between human moves and playing strength in online chess, using large-scale Lichess game archives.

At a high level, the workflow now covers:

1. **Game indexing and basic statistics** for monthly Lichess dumps.
2. **Filtering and balancing** games across rating bands and time controls.
3. **Move-level feature encoding** into a fixed `64 × 8 × 8` binary tensor per ply.
4. **Move-level dataset construction** in `.npz` format for efficient PyTorch training.
5. **A small model zoo** (linear → MLP → CNN → ResNet → conv-transformer) for predicting rating bands and Elo from a *single ply*.
6. **End-to-end experiment runner** for training all model families across config grids.
7. **Per-game inspection + visualization**, showing how models update their rating beliefs move by move, with Gaussian-smoothed rating distributions and animated plots over plies.
8. **A preconfigured modern dataset pipeline** (`run_2025_pipeline.py`) that builds a Jan–Apr 2025 **5+0 blitz** dataset (balanced 400–2400 training splits + unbalanced “real-world” test set).

We use **January 2025 (`2025-01`)** as the canonical example month throughout, and a combined **Jan–Apr 2025** 5+0 blitz dataset for the main experiments.

---

## 1. Data Source

The pipeline assumes access to the official Lichess monthly database:

* Standard rated games in `.pgn.zst` format (e.g.
  `lichess_db_standard_rated_2025-01.pgn.zst`).

Archives are typically downloaded from:

* `https://database.lichess.org/standard/`

Each monthly file contains tens of millions of games. The code is designed to:

* Decompress the `.pgn.zst` archive once,
* Build an efficient row-wise index with offsets into the `.pgn`, and
* Store game-level metadata in Parquet for fast filtering and aggregation.

For current experiments, the script `run_2025_pipeline.py` targets **Jan–Apr 2025 5+0 blitz** games by default (details in §12.1).

---

## 2. Environment & Installation

### 2.1. Dependencies

Key Python dependencies (see `requirements.txt`):

* **Core data & utilities**

  * `pandas`, `numpy`, `tqdm`, `python-chess`, `zstandard`, `requests`, `pyarrow`, `matplotlib`
* **Modeling**

  * `torch`, `transformers` (for the conv-transformer baseline and future sequence models)

### 2.2. Windows + CUDA setup (default)

The repo assumes a typical **Windows + NVIDIA GPU (CUDA)** workflow for training:

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
````

If you have a CUDA-capable GPU and drivers installed, PyTorch will typically expose it as `"cuda"`:

* All training scripts accept `--device cuda` for GPU, or `--device cpu` if you want to force CPU.

> Note: On macOS/Linux you can still use the code by adapting the virtualenv activation command (e.g., `source .venv/bin/activate`), but this README defaults to Windows/CUDA usage.

The repository assumes a layout roughly like:

```text
src/
  scripts/
    build_month_index.py
    index_stats.py
    filter_games.py
    build_balanced_pgn.py
    build_unbalanced_pgn.py
    ply_features.py
    build_rym_npz.py
    train_rym.py
    rym_models.py
    run_rym_experiments.py
    run_2025_pipeline.py
    inspect_rym_game.py
data/
  lichess_db_standard_rated_YYYY-MM.pgn.zst
  lichess_db_standard_rated_YYYY-MM.pgn
  lichess_db_standard_rated_YYYY-MM_index.parquet
  rym_YYYY-MM_*.pgn
  rym_YYYY-MM_*.npz
  rym_2025_jan_apr_tc300+0_bin200_*.pgn
  rym_2025_jan_apr_tc300+0_bin200_*.npz
  rym_2025_jan_apr_tc300+0_unbalanced_realtest.* 
models/
  rym_2025-01_baselines/*.pt
  rym_2025_jan_apr_baselines/*.pt
plots/
  2025-01/...
  rym_inspect/...
```

Paths are configurable via script arguments.

---

## 3. Monthly Indexing

### 3.1. Script: `build_month_index.py`

**Purpose**

* Download (or reuse) the monthly `.pgn.zst` file for a given month.
* Decompress to `.pgn` if needed.
* Sequentially scan the PGN and build a CSV/Parquet index with one row per game, including:

  * File offsets (`start_offset`, `end_offset`),
  * PGN tags (`Event`, `Site`, `Result`, `TimeControl`, etc.),
  * Parsed metadata: player ratings, increment/base time,
  * Boolean flags like `has_clock`, `has_eval` (engine annotations present).

**Usage (January 2025 example)**

```bash
python -m scripts.build_month_index 2025-01
```

**Outputs**

* `data/lichess_db_standard_rated_2025-01.pgn`
* `data/lichess_db_standard_rated_2025-01_index.csv`
* `data/lichess_db_standard_rated_2025-01_index.parquet`

The Parquet index is the main input for downstream filtering and balancing.

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
* For the 2025 pipeline, the default bands are:

  * `[400–600), [600–800), …, [2200–2400)` (10 bands of width 200).
* Bands and ranges are configurable via CLI arguments.

Games with very lopsided pairings (e.g. `|white_elo - black_elo|` above a threshold) can be excluded so bands are not polluted by mismatched pairings.

**Usage**

Full month statistics for January 2025:

```bash
python -m scripts.index_stats 2025-01
```

Only 5+0 blitz games:

```bash
python -m scripts.index_stats 2025-01 --where 'time_control == "300+0"'
```

**Outputs**

* Rating histogram (`rating_hist.png`).
* Time-control bar charts (by base+increment and by frequency).
* Rating × time-control heatmaps in percentage and log-scale:

  * `rating_tc_heatmap_pct.png`
  * `rating_tc_heatmap_log.png`

Drop summaries (e.g. missing ratings, unparsable time controls) are printed to make filtering explicit.

---

## 5. Game Filtering

### 5.1. Script: `filter_games.py`

**Purpose**

* Load a Parquet index for a month.
* Apply SQL-like filters on columns to select a subset of games.
* Either:

  * Return counts only (`--count-only`), or
  * Export the matching games as a new PGN file, using file offsets into the original `.pgn`.

**Examples**

Count only (January 2025, 5+0 blitz):

```bash
python -m scripts.filter_games 2025-01 \
  --where 'time_control == "300+0"' \
  --count-only \
  --preview 5
```

Export 5+0 games with clock information to a new PGN:

```bash
python -m scripts.filter_games 2025-01 \
  --where 'time_control == "300+0" and has_clock' \
  --out-pgn data/filtered_2025-01_5+0_clock.pgn
```

The `--preview` mechanism shows a few sample games to visually sanity-check the filter.

---

## 6. PGN Construction (Balanced & Unbalanced)

### 6.1. Script: `build_balanced_pgn.py` (balanced dataset)

**Purpose**

Construct a **balanced training dataset** across rating bands and other constraints (time control, engine evals, etc.).

**High-level behaviour**

* Read monthly Parquet indexes and associated PGNs.

* Compute the rating column (`avg_elo`, or one player’s Elo) using consistent logic with `index_stats.py`:

  * In `avg_elo` mode, games with large rating gaps (e.g. `|white_elo - black_elo| > K_TOL_DIFF × bin_size`) are dropped.

* Bucket games into **rating bands** using `bin_size` and `[min_rating, max_rating)`:

  * For the default 2025 setup: `min_rating = 400`, `max_rating = 2400`, `bin_size = 200` → 10 bands.

* For each band, sample up to `per_band` games (user-configurable; default 100,000 in the 2025 workflow).

* Concatenate sampled games across bands, shuffle, and split into **train/val/test**.

* Write PGNs that are **approximately uniform across bands**.

This balanced PGN is the primary input for move-level encoding and model training.

**Usage (single-month example, January 2025 5+0 blitz)**

```bash
python -m scripts.build_balanced_pgn 2025-01 \
  --where 'time_control == "300+0"' \
  --rating-col avg_elo \
  --bin-size 200 \
  --min-rating 400 \
  --max-rating 2400 \
  --per-band 100000 \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --test-frac 0.1 \
  --out-prefix data/rym_2025-01_tc300+0_bin200
```

This creates PGNs like:

* `data/rym_2025-01_tc300+0_bin200_train.pgn`
* `data/rym_2025-01_tc300+0_bin200_val.pgn`
* `data/rym_2025-01_tc300+0_bin200_test.pgn`

each containing a balanced mix of rating bands between 400 and 2400.

---

### 6.2. Script: `build_unbalanced_pgn.py` (real-world test set)

**Purpose**

Build an **unbalanced** PGN sample that reflects the *natural* rating distribution after filtering, to serve as a more realistic “in-the-wild” test set.

**What it does**

* Reuses the same core helpers as `build_balanced_pgn.py`:

  * `load_indexes(months)` to concatenate monthly Parquet indexes.
  * `prepare_rating_column(df, rating_col, bin_size)` to:

    * Compute `avg_elo` where requested, and
    * Apply the `K_TOL_DIFF × bin_size` rating-gap filter when `rating_col="avg_elo"`.

* Applies a rating range filter `[min_rating, max_rating)` to the chosen rating column.

* **Does not** perform rating-band binning or per-band balancing.

* Draws a **uniform random sample** of `total-games` rows from the filtered pool.

* Uses `write_pgn_for_index(df_sample, out_pgn)` to materialize the PGN by slicing from the original monthly `.pgn` files.

This yields a PGN whose rating distribution mirrors the real data, under your filters.

**Key arguments**

* `months`: one or more `YYYY-MM` strings.
* `--where`: pandas query over index columns (e.g. `time_control == "300+0"`).
* `--rating-col`: `white_elo`, `black_elo`, or `avg_elo` (default: `avg_elo`).
* `--bin-size`: used only to set the rating-gap tolerance in `avg_elo` mode.
* `--min-rating`, `--max-rating`: rating range for inclusion.
* `--total-games`: target number of games after all filters (if fewer are available, all are used).
* `--out-prefix`: output prefix; the script writes `<prefix>.pgn`.

**Example (Jan–Apr 2025 5+0 blitz, 400–2400)**

```bash
python -m scripts.build_unbalanced_pgn 2025-01 2025-02 2025-03 2025-04 \
  --where "time_control == '300+0'" \
  --rating-col avg_elo \
  --bin-size 200 \
  --min-rating 400 \
  --max-rating 2400 \
  --total-games 100000 \
  --seed 64 \
  --out-prefix data/rym_2025_jan_apr_tc300+0_unbalanced_realtest
```

---

## 7. Ply-Level Feature Encoding

### 7.1. Script: `ply_features.py`

**Purpose**

Encode a single ply (one half-move) into a fixed **`64 × 8 × 8`** binary feature tensor.

**Input**

* `board`: `chess.Board` in the **pre-move** state,
* `move`: `chess.Move` to be played from that position,
* `prev_move`: optional `chess.Move` for the previous ply.

**Output**

* `feats`: `np.ndarray` with shape `(64, 8, 8)`, `dtype=uint8`.

#### 7.1.1. Conceptual definitions

* **Presence** (geometric, empty board): squares a piece *could* capture on if the board were empty and an enemy piece sat there.
* **Any control** (attack/defense, current board): squares a piece currently attacks (including own pieces as defended).
* **Legal control** (attack/defense respecting pins): filter any control to remove moves that violate absolute pins:

  * Pinned rooks/bishops/queens/pawns: only squares along the pin line.
  * Pinned knights: no legal control.
* **Net control** (pawns only): a square is net-controlled by a side if and only if that side’s pawns legally control it and the opponent’s pawns do not.

#### 7.1.2. Plane layout (64 planes)

All planes are 0/1.

**Board, presence, control (pre-move)**

* `0–11`: `board_pre`
  1 plane per `(color, piece_type)`, 6 types × 2 colors.
* `12–23`: `presence_pre`
  Empty-board capture reach per `(color, piece_type)`.
* `24–35`: `control_pre`
  Any control (attacks/defenses) per `(color, piece_type)`.

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
* `62`: `post_side_to_move`
* `63`: `post_en_passant_target`

#### 7.1.3. Usage

```python
import chess
from scripts.ply_features import encode_ply_planes, NUM_PLANES

board = chess.Board()
move = chess.Move.from_uci("e2e4")
planes = encode_ply_planes(board, move, prev_move=None)  # (64, 8, 8) uint8
```

---

## 8. Move-Level Dataset Construction (`.npz`)

### 8.1. Script: `build_rym_npz.py`

**Purpose**

Convert a PGN (typically from `build_balanced_pgn.py`, but it can also be an unbalanced test PGN) into a compact `.npz` file suitable for PyTorch training.

Each ply becomes a row in the dataset; each game becomes a sequence of rows linked by `game_id` + `ply_idx`.

**What it does**

* Reads a PGN of games.

* For each game:

  * Parses average Elo `avg_elo` from `WhiteElo`/`BlackElo` tags:

    * If both present: `avg_elo = (WhiteElo + BlackElo)/2`.
    * If only one present: uses that rating.
    * If neither: game is dropped.

  * Assigns a **rating band** index `y_bin ∈ {0, …, num_bins−1}` using `(min_rating, max_rating, num_bins)`.

  * Iterates through all plies in the mainline, encoding each ply as `X_t ∈ {0,1}^{64×8×8}` via `encode_ply_planes`.

* Stores the resulting arrays into a single `.npz` file.

**NPZ schema**

The saved `.npz` contains:

* `X`: `(N, NUM_PLANES, 8, 8)` `uint8` – ply feature tensors.
* `y_bin`: `(N,)` `int16` – rating band index for that game.
* `y_elo`: `(N,)` `float32` – numeric rating (average Elo) used for banding.
* `game_id`: `(N,)` `int32` – integer game index (0-based within this PGN).
* `ply_idx`: `(N,)` `int16` – ply index within the game.
* `num_bins`: scalar `int16`.
* `min_rating`: scalar `int16`.
* `max_rating`: scalar `int16`.

Every ply of a game shares the same `y_bin` and `y_elo`.

**Usage (January 2025, balanced PGNs)**

```bash
python -m scripts.build_rym_npz \
  data/rym_2025-01_tc300+0_bin200_train.pgn \
  --out data/rym_2025-01_tc300+0_bin200_train.npz \
  --min-rating 400 \
  --max-rating 2400 \
  --num-bins 10

python -m scripts.build_rym_npz \
  data/rym_2025-01_tc300+0_bin200_val.pgn \
  --out data/rym_2025-01_tc300+0_bin200_val.npz \
  --min-rating 400 \
  --max-rating 2400 \
  --num-bins 10
```

You typically run this for `train`, `val`, and `test` PGNs.

---

## 9. Local Board Encoders & Bayesian Aggregation

The modeling layer currently focuses on **per-ply encoders** that map a single `64×8×8` tensor to:

* A **distribution over rating bands** (`logits ∈ ℝ^{num_bins}`), and
* A **scalar Elo estimate** (`rating_pred ∈ ℝ`),

and then uses a simple **Bayesian update** across plies to get a game-level posterior.

### 9.1. Script: `rym_models.py`

**Purpose**

Define a small model zoo and a `get_model` factory.

**Model families (`MODEL_TYPES`)**

Each model implements:

```python
logits, rating_pred = model(X_batch)
# X_batch: (B, NUM_PLANES, 8, 8)
# logits: (B, num_bins)
# rating_pred: (B,) or (B, 1)
```

Available families:

* `"linear"`

  * Single linear layer on flattened input.
  * Very simple baseline.

* `"mlp"`

  * Flatten + 1–2 hidden layers with nonlinearity.
  * Tunable hidden size / depth via `config_id`.

* `"cnn"`

  * Small convolutional network over the 8×8 board:

    * Conv → Conv → pooling → MLP.

* `"resnet"`

  * Residual CNN with skip connections:

    * Suitable for deeper spatial feature extraction.

* `"conv_transformer"`

  * Hybrid:

    * Convolutions extract local spatial features.
    * Board squares are flattened to a sequence and passed through a transformer encoder over 64 “tokens”.
    * Final pooled representation feeds the rating heads.

Each family has **4 discrete configurations** (`config_id ∈ {0,1,2,3}`), varying things like width, depth, and number of channels.

**Factory**

```python
from scripts.rym_models import get_model, MODEL_TYPES
from scripts.ply_features import NUM_PLANES

model = get_model(
    model_type="resnet",
    num_planes=NUM_PLANES,
    num_bins=num_bins,
    config_id=0,
)
```

---

### 9.2. Script: `train_rym.py`

**Purpose**

Train a **single** model configuration on a pair of NPZ files (`train` + `val`), with a **joint loss**:

* distance-aware classification over rating bands, and
* regression over Elo.

**Dataset loader**

`train_rym.py` defines `RYMNpzDataset`, a thin wrapper around the `.npz` file:

```python
from pathlib import Path
from scripts.train_rym import RYMNpzDataset

ds = RYMNpzDataset(Path("data/rym_2025-01_tc300+0_bin200_train.npz"))
sample = ds[0]
# sample["X"]: (NUM_PLANES, 8, 8) float32
# sample["y_bin"]: long
# sample["y_elo"]: float32
```

The dataset also exposes:

* `ds.num_bins`
* `ds.min_rating`
* `ds.max_rating`
* `ds.game_id` / `ds.ply_idx` (if you need per-game grouping)

**Distance-aware joint loss**

`compute_losses(logits, rating_pred, y_bin, y_elo, num_bins, min_rating, max_rating, alpha_reg)` does:

1. **Classification loss (bands)**

   * Standard cross-entropy over `y_bin`.
   * Augmented with a **distance penalty**:

     * A band-distance matrix `D[i,j] = |i − j|` is constructed.
     * Misclassifications that are far from the true band are penalized more heavily than near-misses.

2. **Regression loss (Elo)**

   * MSE between `rating_pred` and `y_elo` (the actual average Elo).

3. **Combined loss**

   ```text
   loss = loss_cls + alpha_reg * loss_reg
   ```

   `alpha_reg` controls the relative importance of accurate Elo vs band classification.

**Training / evaluation functions**

* `train_one_epoch(...)` loops over a `DataLoader` and returns average losses.
* `evaluate(...)` runs in `no_grad()` mode and reports:

  * `loss`, `loss_cls`, `loss_reg`
  * `acc` (top-1 accuracy on band index)
  * `mae_rating` (mean absolute error in Elo)

**CLI usage (2025-01 example)**

```bash
python -m scripts.train_rym ^
  --train-npz data/rym_2025-01_tc300+0_bin200_train.npz ^
  --val-npz   data/rym_2025-01_tc300+0_bin200_val.npz   ^
  --model-type resnet ^
  --config-id 0 ^
  --batch-size 256 ^
  --epochs 10 ^
  --lr 1e-3 ^
  --alpha-reg 1.0 ^
  --device cuda ^
  --save-path models/rym_2025-01_baselines/rym_resnet_cfg0.pt
```

(This uses Windows `^` line continuations; you can also write the command on one line.)

---

## 10. End-to-End Experiment Runner

### 10.1. Script: `run_rym_experiments.py`

**Purpose**

A higher-level driver that:

1. Ensures `.npz` files exist for `train`, `val`, and (optionally) `test` splits,
2. Builds `DataLoader`s,
3. Trains **multiple model families and config IDs** in one go,
4. Saves checkpoints and logs metrics.

This is what you use when you’re ready to “turn the crank” and generate all baselines.

**Step 1 – ensure NPZs**

The helper `ensure_npz_for_split(...)` will:

* Expect PGNs at:

  * `"{pgn_prefix}_train.pgn"`
  * `"{pgn_prefix}_val.pgn"`
  * `"{pgn_prefix}_test.pgn"` (optional)

* Create NPZs at:

  * `"{npz_prefix}_train.npz"`
  * `"{npz_prefix}_val.npz"`
  * `"{npz_prefix}_test.npz"`

by calling `build_rym_npz.main()` under the hood if needed.

**Step 2 – DataLoaders**

The script builds `train` / `val` loaders via `RYMNpzDataset` in exactly the same format as `train_rym.py`.

**Step 3 – model grid**

Two CLI arguments control the experiment grid:

* `--models` – comma-separated list, or `"all"`:

  ```bash
  --models resnet,conv_transformer
  # or
  --models all
  ```

* `--config-id` – can be:

  * A single integer as a string, e.g. `"0"`
  * A comma-separated list, e.g. `"0,1,3"`
  * The string `"all"` to use all available config IDs for each model family.

Internally, `parse_model_list` and `parse_config_ids` turn these into Python lists.

**Step 4 – training loop**

For each `(model_type, config_id)` pair:

* Calls `get_model(...)`,
* Sets up `torch.optim.Adam`,
* Runs `train_one_epoch` / `evaluate` (from `train_rym.py`) for `--epochs` epochs,
* Logs metrics like:

  ```text
  train: loss=..., cls=..., reg=...
  val  : loss=..., cls=..., reg=..., acc=..., MAE=...
  ```

**Step 5 – checkpoints**

If `--save-dir` is given, each run writes:

```text
{save_dir}/rym_{model_type}_cfg{config_id}.pt
```

Each checkpoint stores:

* `model_state`
* `model_type`, `config_id`
* `num_planes` (for sanity)
* `num_bins`

**Usage example (2025-01, 400–2400, 200pt bands)**

```bash
python -m scripts.run_rym_experiments ^
  --pgn-prefix data/rym_2025-01_tc300+0_bin200 ^
  --npz-prefix data/rym_2025-01_tc300+0_bin200 ^
  --min-rating 400 ^
  --max-rating 2400 ^
  --num-bins 10 ^
  --models all ^
  --config-id all ^
  --batch-size 256 ^
  --epochs 5 ^
  --lr 1e-3 ^
  --alpha-reg 1.0 ^
  --device cuda ^
  --save-dir models/rym_2025-01_baselines
```

This trains **all model families** with **all configs** on the January 2025 balanced 5+0 dataset, saving a grid of checkpoints you can later use for analysis or inspection.

---

## 11. Inspecting Individual Games (Post-hoc Analysis)

### 11.1. Script: `inspect_rym_game.py`

**Purpose**

Take one or more trained models and a PGN/NPZ, and:

1. Pick a single game (either randomly or by index),
2. Run *all* models on every ply of that game,
3. Maintain a Bayesian posterior over rating bands as the game progresses,
4. Produce:

   * A pretty printed table of Elo estimates per ply, and
   * A **per-move visualization** of each model’s rating distribution:

     * Smooth Gaussian-shaped curves over Elo,
     * Vertical lines for classification vs regression Elo,
     * Optional fill-under-curve highlighting for the conv-transformer,
     * A frame per ply that can be replayed as an animation in Jupyter.

This is where you “watch” the models think: how quickly they latch onto a player’s true strength, which moves cause jumps, and how different architectures disagree.

---

### 11.2. Bayesian update across plies

For a given game:

* For each model, you get per-ply logits over bands:

  ```python
  logits_seq: (T, num_bins)
  rating_seq: (T,)
  ```

* At each ply `t`, you compute a likelihood:

  ```python
  lik_t = torch.softmax(logits_seq[t], dim=-1)
  ```

* You maintain a posterior `p_t`:

  ```text
  p_0 = uniform
  p_t ∝ p_{t-1} * lik_t
  ```

This gives a posterior `p_t` over rating bands after seeing plies `0..t`.

* The **classification Elo** is:

  ```text
  cls_elo_t = Σ_k p_t(k) * center(k)
  ```

  where `center(k)` is the Elo midpoint of band `k`.

* The **regression Elo** is `rating_seq[t]`.

Both are printed per ply and also visualized.

---

### 11.3. Pretty-print output

`pretty_print_table(...)` prints a text table like:

```text
Game: Alice vs Bob, true avg Elo ≈ 1850.0
------------------------------------------------------------------------------------------------------------------------
ply  move        linear_cls linear_reg  mlp_cls    mlp_reg    ... convT_cls convT_reg
1    e4             1650.3     1612.5   1720.8     1690.4     ...  1785.2    1760.9
2    ...c5          1675.1     1630.0   1742.1     1702.7     ...  1801.3    1775.0
3    Nf3            ...
...
------------------------------------------------------------------------------------------------------------------------
```

You can quickly eyeball:

* How fast models converge toward the true Elo,
* Which moves cause sharp jumps,
* How different families behave.

---

### 11.4. Per-move visualization

`plot_per_move_distributions(...)` generates one PNG per ply, with:

* **Gaussian-mixture curves**:

  * For each model, the band posterior is convolved with Gaussian bumps around each band center.
  * Curves are deliberately **scaled up** (not normalized) so the shape is visible; think “scaled density” rather than probability.
  * Each model gets its own color; for the conv-transformer, the area under the curve is filled with a very transparent alpha.

* **Vertical lines:**

  * **Dashed** line at `cls_elo_t` (classification expectation under posterior).
  * **Solid** line at `reg_elo_t` (regression prediction).
  * Lines share the same color as that model’s curve.

* **Axes:**

  * X-axis: Elo (from `min_rating` to `max_rating`).
  * Y-axis: scaled curve height (internally adjusted per frame).

Filenames look like:

```text
plots/rym_inspect/game_Alice_vs_Bob_ply_001.png
plots/rym_inspect/game_Alice_vs_Bob_ply_002.png
...
```

### 11.5. Animation helper (Jupyter)

There is a small helper:

```python
from scripts.inspect_rym_game import replay_images

replay_images("plots/rym_inspect", delay=0.05)
```

which will display all PNGs in order as a simple in-notebook animation.

---

### 11.6. Using your own PGNs

One nice feature: `inspect_rym_game.py` can work with **your own PGN** even if you don’t have an NPZ yet.

**CLI (using 2025-01 baselines)**

```bash
python -m scripts.inspect_rym_game ^
  --test-pgn my_games_rapid.pgn ^
  --models-dir models/rym_2025-01_baselines ^
  --config-id 0 ^
  --device cuda ^
  --out-dir plots/rym_my_games
```

Behaviour:

* If `--test-npz` is **not** given, the script derives one:

  ```text
  my_games_rapid.pgn  →  my_games_rapid.npz
  ```

  by calling `pgn_to_npz(...)` internally with `--min-rating`, `--max-rating`, and `--num-bins` (overridable via CLI).

* It then loads `RYMNpzDataset` on that NPZ, chooses a game:

  * Either a random one (controlled by `--seed`), or
  * A specific `--game-index` if provided.

* The PGN game is loaded with the same index, and the models are run as usual.

**Requirements for custom PGNs**

* Games must have at least one of `WhiteElo` / `BlackElo` tags to compute `avg_elo`.
* Games without any usable rating information are silently dropped from the NPZ.

---

## 12. End-to-End Example (Single Month: January 2025)

Putting everything together for **January 2025 5+0 blitz**, a typical manual workflow looks like:

1. **Index January 2025 Lichess games**

   ```bash
   python -m scripts.build_month_index 2025-01
   ```

2. **Explore rating/time-control structure**

   ```bash
   python -m scripts.index_stats 2025-01 --where 'time_control == "300+0"'
   ```

3. **Filter and build a balanced PGN (5+0, 400–2400, 200pt bands)**

   ```bash
   python -m scripts.build_balanced_pgn 2025-01 ^
     --where 'time_control == "300+0"' ^
     --rating-col avg_elo ^
     --bin-size 200 ^
     --min-rating 400 ^
     --max-rating 2400 ^
     --per-band 100000 ^
     --train-frac 0.8 ^
     --val-frac 0.1 ^
     --test-frac 0.1 ^
     --out-prefix data/rym_2025-01_tc300+0_bin200
   ```

4. **Build NPZs for train/val/test**

   ```bash
   python -m scripts.build_rym_npz ^
     data/rym_2025-01_tc300+0_bin200_train.pgn ^
     --out data/rym_2025-01_tc300+0_bin200_train.npz ^
     --min-rating 400 --max-rating 2400 --num-bins 10

   python -m scripts.build_rym_npz ^
     data/rym_2025-01_tc300+0_bin200_val.pgn ^
     --out data/rym_2025-01_tc300+0_bin200_val.npz ^
     --min-rating 400 --max-rating 2400 --num-bins 10

   python -m scripts.build_rym_npz ^
     data/rym_2025-01_tc300+0_bin200_test.pgn ^
     --out data/rym_2025-01_tc300+0_bin200_test.npz ^
     --min-rating 400 --max-rating 2400 --num-bins 10
   ```

   (Or just let `run_rym_experiments.py` do this automatically.)

5. **Train a grid of baselines**

   ```bash
   python -m scripts.run_rym_experiments ^
     --pgn-prefix data/rym_2025-01_tc300+0_bin200 ^
     --npz-prefix data/rym_2025-01_tc300+0_bin200 ^
     --min-rating 400 --max-rating 2400 --num-bins 10 ^
     --models all ^
     --config-id all ^
     --batch-size 256 ^
     --epochs 5 ^
     --lr 1e-3 ^
     --alpha-reg 1.0 ^
     --device cuda ^
     --save-dir models/rym_2025-01_baselines
   ```

6. **Inspect a single game from the test split**

   ```bash
   python -m scripts.inspect_rym_game ^
     --test-npz data/rym_2025-01_tc300+0_bin200_test.npz ^
     --test-pgn data/rym_2025-01_tc300+0_bin200_test.pgn ^
     --models-dir models/rym_2025-01_baselines ^
     --config-id 0 ^
     --device cuda ^
     --out-dir plots/rym_inspect
   ```

7. **Inspect one of your own games**

   ```bash
   python -m scripts.inspect_rym_game ^
     --test-pgn my_own_games.pgn ^
     --models-dir models/rym_2025-01_baselines ^
     --config-id 0 ^
     --device cuda ^
     --out-dir plots/rym_my_games
   ```

   Then in a Jupyter notebook:

   ```python
   from scripts.inspect_rym_game import replay_images
   replay_images("plots/rym_my_games", delay=0.05)
   ```

---

### 12.1. Jan–Apr 2025 5+0 Blitz Pipeline (`run_2025_pipeline.py`)

For the main RYM experiments in this repo, we provide a **single end-to-end script** that builds a modern dataset from the Lichess 2025 dumps.

By default, it:

* Uses months **2025-01, 2025-02, 2025-03, 2025-04** (`--months` overrideable).

* Filters to **5+0 blitz** via `time_control == "300+0"` (`--time-control`).

* Restricts ratings to **[400, 2400)** with **200-point bands**:

  * `min-rating = 400`, `max-rating = 2400`, `bin-size = 200`.
  * This yields `num_bins = 10` bands: 400–600, 600–800, …, 2200–2400.

* Targets **100k games per band** for the **balanced** dataset (`--per-band 100000`).

* Builds a separate **unbalanced “real-world” test set** with up to **100k games**
  (`--unbalanced-total 100000`).

Conceptually:

* The **balanced dataset** is for training / validation:

  * Roughly uniform over rating bands from 400 to 2400.
  * Split into train / val / test via `train-frac`, `val-frac`, `test-frac`.

* The **unbalanced dataset** is for evaluation:

  * Keeps the same filters (time control + rating range),
  * but **preserves the natural rating distribution** instead of enforcing bands.

**Files produced (by default)**

Under `data/`:

* Balanced PGNs:

  * `rym_2025_jan_apr_tc300+0_bin200_train.pgn`
  * `rym_2025_jan_apr_tc300+0_bin200_val.pgn`
  * `rym_2025_jan_apr_tc300+0_bin200_test.pgn`

* Balanced NPZs:

  * `rym_2025_jan_apr_tc300+0_bin200_train.npz`
  * `rym_2025_jan_apr_tc300+0_bin200_val.npz`
  * `rym_2025_jan_apr_tc300+0_bin200_test.npz`

* Unbalanced real-world test:

  * `rym_2025_jan_apr_tc300+0_unbalanced_realtest.pgn`
  * `rym_2025_jan_apr_tc300+0_unbalanced_realtest.npz`

**Running the pipeline (Windows)**

```bash
python -m scripts.run_2025_pipeline
```

This will:

1. Ensure the monthly `.pgn.zst` dumps and indexes exist (download/decompress if needed).
2. Build the **balanced** PGNs across Jan–Apr 2025 (if they do not already exist).
3. Convert them to NPZs via `build_rym_npz` (if NPZs do not already exist).
4. Build the **unbalanced** PGN via `build_unbalanced_pgn` (if not present).
5. Convert the unbalanced PGN to an NPZ.
6. Optionally clean up the raw monthly `.pgn` / `.pgn.zst` files if
   `--cleanup-pgn` / `--cleanup-zst` are set.

This gives you a **modern, large-scale 5+0 blitz dataset** ready for training and evaluation with the rest of the RYM pipeline.

---

## 13. Future Directions

The current codebase gives you:

* A scalable preprocessing pipeline from raw Lichess archives to balanced, move-level datasets.
* A family of local board encoders with a principled Bayesian aggregation over plies.
* Tools to visualize how rating beliefs evolve move by move, across multiple model families.
* A concrete modern experimental setup (Jan–Apr 2025 5+0 blitz, 400–2400, 200pt bands) for reproducible studies on a **Windows + CUDA** stack.

Future extensions (some hinted at already in the code) include:

* **Full game-sequence transformers**:

  * Consume sequences of plies `(X_0, …, X_{T-1})` directly.
  * Maintain an internal belief state over rating that’s learned end-to-end.

* **Richer metadata inputs**:

  * Clock times, engine evals, time usage patterns, opening families.

* **Multi-task setups**:

  * Predict rating, time control, or even “style” clusters concurrently.

* **Cross-month generalization**:

  * Train on several months, evaluate out-of-distribution on others.

For now, the repository already lets you answer a rich set of questions about **how much information each move carries about a player’s strength**, and how different architectures “see” the same game, all within a reproducible 2025-era Lichess blitz setting.