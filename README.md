# Rating Your Moves (RYM)

This repository implements a complete data + modeling pipeline for studying how human chess moves reveal playing strength, using large scale Lichess game archives.

Conceptually:

> Given a sequence of human moves, what rating distribution is most compatible with those decisions?

The core pieces are:

1. Monthly Lichess indexing (PGN → Parquet index).
2. Balanced and unbalanced PGN builders by rating band and time control.
3. Per-ply feature encoding into a `64 × 8 × 8` tensor.
4. Sharded NPZ datasets for train, validation, test, and a real-world test.
5. A model zoo (linear, MLP, CNN, ResNet, conv transformer).
6. Training and experiment runners over NPZ shards.
7. A new evaluation + visualization stack:

   * `scripts/rym_analysis.py` – model evaluation, Bayesian aggregation, band tables. 
   * `scripts/rym_vis.py` – feature-plane grids, posterior plots, board + distribution frame rendering. 
   * `display.ipynb` – a self-contained demo notebook for running a trained model on NPZ shards + a custom PGN, and rendering replay animations.

The default modern dataset is Jan to Apr 2025 5+0 blitz, with rating bands from 400 to 2400 in 200-point steps.

---

## 1. Data source and scope

The pipeline is built around the official Lichess monthly database:

* Standard rated games in `.pgn.zst` format (for example `lichess_db_standard_rated_2025-01.pgn.zst`).
* Archives are typically fetched from `https://database.lichess.org/standard/`.

Each monthly file contains many millions of games. The code is designed to:

* Download `.pgn.zst` if missing.
* Decompress to `.pgn` once per month.
* Build a row-wise index with file offsets and metadata.
* Store the index in Parquet for fast filtering and aggregation.

The main experimental configuration targets:

* Months 2025-01, 2025-02, 2025-03, 2025-04.
* Time control `300+0` (5+0 blitz).
* Rating range `[400, 2400)`, with bands of width 200 (10 bands total).

A dedicated driver `run_2025_pipeline.py` builds a Jan to Apr 2025 dataset with:

* A balanced rating-band dataset split into train, validation, and test.
* An additional unbalanced “real-world” test set for evaluation.

---

## 2. Repository layout

At a high level:

```text
rating_your_moves/
  data/
    # raw lichess monthly dumps + indexes
    lichess_db_standard_rated_2025-01.pgn[.zst]
    lichess_db_standard_rated_2025-01_index.parquet
    ...
    lichess_db_standard_rated_2025-04.pgn[.zst]
    lichess_db_standard_rated_2025-04_index.parquet

    # Jan–Apr 2025, 5+0 blitz, 400–2400, 200pt bands
    rym_2025_jan_apr_tc300+0_bin200_index.parquet

    # BALANCED splits (only sharded NPZ, no monolithic NPZ by default)
    rym_2025_jan_apr_tc300+0_bin200_train.pgn
    rym_2025_jan_apr_tc300+0_bin200_train_shard000.npz
    rym_2025_jan_apr_tc300+0_bin200_train_shard001.npz
    ...
    rym_2025_jan_apr_tc300+0_bin200_train_shard079.npz

    rym_2025_jan_apr_tc300+0_bin200_val.pgn
    rym_2025_jan_apr_tc300+0_bin200_val_shard000.npz
    ...
    rym_2025_jan_apr_tc300+0_bin200_val_shard009.npz

    rym_2025_jan_apr_tc300+0_bin200_test.pgn
    rym_2025_jan_apr_tc300+0_bin200_test_shard000.npz
    ...
    rym_2025_jan_apr_tc300+0_bin200_test_shard009.npz

    # UNBALANCED "real world" test split
    rym_2025_jan_apr_tc300+0_unbalanced_realtest.pgn
    rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard000.npz
    ...
    rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard009.npz

  scripts/
    build_month_index.py
    index_stats.py
    filter_games.py
    build_balanced_pgn.py
    build_unbalanced_pgn.py
    run_2025_pipeline.py

    ply_features.py
    build_rym_npz.py        # monolithic NPZ for small PGNs (optional)
    rym_models.py
    train_rym.py            # single NPZ train/val (debug or small runs)
    run_rym_experiments.py  # main sharded trainer

    rym_analysis.py         # evaluation + Bayesian aggregation helpers
    rym_vis.py              # visualisation + board/posterior frame helpers

  models/
    rym_2017-04_baselines/
      rym_cnn_cfg0.pt
      rym_conv_transformer_cfg0.pt
      rym_linear_cfg0.pt
      rym_mlp_cfg0.pt
      rym_resnet_cfg0.pt

  display.ipynb             # main end-to-end evaluation + visualisation demo
  README.md
  requirements.txt
```

For Jan to Apr 2025, training is intended to happen on shards via `run_rym_experiments.py`. Monolithic NPZs appear only when built manually with `build_rym_npz.py` for smaller experiments.

---

## 3. Environment and installation

### 3.1. Dependencies

The key Python dependencies (see `requirements.txt`) include:

* Core data and utilities: `pandas`, `numpy`, `tqdm`, `python-chess`, `zstandard`, `requests`, `pyarrow`, `matplotlib`.
* Modeling: `torch`, `transformers` (for the conv transformer baseline and planned sequence models).

### 3.2. Typical setup (Windows with CUDA)

A common setup uses Windows with an NVIDIA GPU and CUDA enabled:

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

PyTorch should expose the GPU as `"cuda"` when the CUDA build and drivers are present.

All training scripts accept:

* `--device cuda` for GPU.
* `--device cpu` to force CPU.
* `--device mps` on Apple Silicon if supported.

On macOS or Linux, the virtual environment activation command changes to `source .venv/bin/activate`, but the rest of the instructions remain the same.

Paths for data and models are configurable through script arguments.

---

## 4. Data pipeline: from Lichess dumps to PGNs

### 4.1. Monthly indexing (`build_month_index.py`)

Purpose:

* Download (or reuse) the monthly `.pgn.zst` file for a given month.
* Decompress to `.pgn` if needed.
* Sequentially scan the PGN and build a CSV and Parquet index with one row per game.

The index includes:

* File offsets: `start_offset`, `end_offset`.
* PGN tags: `Event`, `Site`, `Result`, `TimeControl`, and others.
* Parsed metadata: `white_elo`, `black_elo`, `time_control`, base time and increment.
* Boolean flags such as `has_clock` and `has_eval` (engine annotations present or not).

Example (January 2025):

```bash
python -m scripts.build_month_index 2025-01
```

Outputs:

* `data/lichess_db_standard_rated_2025-01.pgn`
* `data/lichess_db_standard_rated_2025-01_index.csv`
* `data/lichess_db_standard_rated_2025-01_index.parquet`

The Parquet index is the main input for downstream filtering and balancing.

### 4.2. Index statistics and sanity checks (`index_stats.py`)

Purpose:

* Load one or more monthly Parquet index files.
* Optionally apply a filter expression.
* Compute descriptive statistics for rating and time-control structure.

Rating bands:

* Games are assigned to bands using an average Elo:

  * `avg_elo = (white_elo + black_elo) / 2`.

* For the 2025 setup, the default bands are:

  * `[400,600), [600,800), ..., [2200,2400)` (10 bands, width 200).

* Games with very lopsided pairings (large rating gaps between players) can be dropped.

Example: full month statistics for January 2025:

```bash
python -m scripts.index_stats 2025-01
```

Example: only 5+0 games:

```bash
python -m scripts.index_stats 2025-01 --where 'time_control == "300+0"'
```

The script can emit:

* Rating histograms.
* Time-control bar charts.
* Rating × time-control heatmaps.

Drop summaries are printed so that excluded games (missing ratings, unparsable tags, etc.) are explicit.

### 4.3. Game filtering (`filter_games.py`)

Purpose:

* Load a Parquet index for a month.
* Apply SQL-like filters on columns.
* Either count matching games or export them as a new PGN.

Examples:

Count only, January 2025 5+0 games:

```bash
python -m scripts.filter_games 2025-01 ^
  --where 'time_control == "300+0"' ^
  --count-only ^
  --preview 5
```

Export 5+0 games with clock information to a PGN:

```bash
python -m scripts.filter_games 2025-01 ^
  --where 'time_control == "300+0" and has_clock' ^
  --out-pgn data/filtered_2025-01_5+0_clock.pgn
```

The `--preview` option prints a few sample rows for a quick sanity check.

---

## 5. Balanced and unbalanced PGNs

### 5.1. Balanced dataset (`build_balanced_pgn.py`)

Purpose:

* Construct a balanced training dataset across rating bands and other constraints (for example time control and engine annotations).

Behavior:

* Read monthly Parquet indexes and associated PGNs.

* Compute the rating column (for example `avg_elo`) consistently with `index_stats.py`.

  * In `avg_elo` mode, games with rating gaps above a tolerance (relative to the band width) can be dropped.

* Bucket games into rating bands using `bin_size` and `[min_rating, max_rating)`:

  * Default 2025 configuration: `min_rating = 400`, `max_rating = 2400`, `bin_size = 200` → 10 bands.

* For each band, sample up to `per_band` games.

* Concatenate sampled games across bands, shuffle, and split into train, validation, and test.

* Write PGNs that are approximately uniform across bands.

Single month example (January 2025, 5+0):

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

Outputs:

* `data/rym_2025-01_tc300+0_bin200_train.pgn`
* `data/rym_2025-01_tc300+0_bin200_val.pgn`
* `data/rym_2025-01_tc300+0_bin200_test.pgn`

Each contains a balanced mix of rating bands between 400 and 2400.

### 5.2. Unbalanced real-world dataset (`build_unbalanced_pgn.py`)

Purpose:

* Build an unbalanced PGN sample that reflects the natural rating distribution after filtering, for use as a more realistic test set.

Behavior:

* Reuse indexing and rating helpers.
* Apply the same rating range `[min_rating, max_rating)` and filters as the balanced set.
* Do **not** perform per-band balancing.
* Draw a uniform random sample of up to `total-games` rows from the filtered pool.
* Write the result as a PGN by slicing from the original monthly `.pgn` files.

Example (Jan to Apr 2025, 5+0, 400 to 2400):

```bash
python -m scripts.build_unbalanced_pgn 2025-01 2025-02 2025-03 2025-04 ^
  --where "time_control == '300+0'" ^
  --rating-col avg_elo ^
  --bin-size 200 ^
  --min-rating 400 ^
  --max-rating 2400 ^
  --total-games 100000 ^
  --seed 64 ^
  --out-prefix data/rym_2025_jan_apr_tc300+0_unbalanced_realtest
```

The output PGN preserves the empirical rating distribution subject to the filters, instead of balancing bands.

### 5.3. One-shot Jan to Apr 2025 pipeline (`run_2025_pipeline.py`)

For Jan to Apr 2025 there is a single driver that runs the full data pipeline:

1. Reads monthly indexes for 2025-01 to 2025-04.
2. Filters to `time_control == "300+0"`.
3. Bands games into 200-Elo buckets from 400 to 2400.
4. Samples a balanced dataset across bands and splits into train, validation, and test by game.
5. Writes balanced PGNs and associated NPZ shards.
6. Builds an additional unbalanced real-world test set and NPZ shards.

Command:

```bash
python -m scripts.run_2025_pipeline
```

Defaults (overridable):

* Months: `2025-01 2025-02 2025-03 2025-04`.
* Time control: `300+0`.
* Rating range: `400 to 2400`.
* Band width: `200` (10 bands).
* Target per band: `--per-band 100000` (reduced if some bands lack data).
* Size of the unbalanced test: `--unbalanced-total 100000`.

Outputs:

* Balanced PGNs:

  * `data/rym_2025_jan_apr_tc300+0_bin200_train.pgn`
  * `data/rym_2025_jan_apr_tc300+0_bin200_val.pgn`
  * `data/rym_2025_jan_apr_tc300+0_bin200_test.pgn`

* Balanced NPZ shards:

  * `data/rym_2025_jan_apr_tc300+0_bin200_train_shard000.npz`
    …
  * `data/rym_2025_jan_apr_tc300+0_bin200_train_shard079.npz`
  * `data/rym_2025_jan_apr_tc300+0_bin200_val_shard000.npz`
    …
  * `data/rym_2025_jan_apr_tc300+0_bin200_val_shard009.npz`
  * `data/rym_2025_jan_apr_tc300+0_bin200_test_shard000.npz`
    …
  * `data/rym_2025_jan_apr_tc300+0_bin200_test_shard009.npz`

* Unbalanced real-world test:

  * `data/rym_2025_jan_apr_tc300+0_unbalanced_realtest.pgn`
  * `data/rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard000.npz`
    …
  * `data/rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard009.npz`

* Index:

  * `data/rym_2025_jan_apr_tc300+0_bin200_index.parquet` with per-game metadata and band labels.

The driver can optionally remove raw monthly `.pgn` or `.pgn.zst` files after processing if cleanup flags are set.

---

## 6. NPZ schema (per shard and optional monolithic files)

For the main Jan to Apr 2025 dataset, the pipeline produces only sharded NPZ files. Every `*_shardXXX.npz` stores ply-level examples.

Example:

```python
import numpy as np

npz = np.load("data/rym_2025_jan_apr_tc300+0_bin200_train_shard000.npz")

X       = npz["X"]        # (N, NUM_PLANES, 8, 8) float32
y_bin   = npz["y_bin"]    # (N,) int64  band index (0..num_bins-1)
y_elo   = npz["y_elo"]    # (N,) float32 average Elo for the game
game_id = npz["game_id"]  # (N,) int64  game index in the PGN
ply_idx = npz["ply_idx"]  # (N,) int32  ply index within the game

num_bins   = int(npz["num_bins"])
min_rating = float(npz["min_rating"])
max_rating = float(npz["max_rating"])
```

All shards for a given split share `num_bins`, `min_rating`, and `max_rating`. Shards can be treated as chunks of a large conceptual dataset and concatenated if desired.

For small PGNs it is also possible to build a single NPZ via `build_rym_npz.py`. The schema is the same as above, but written into a single file instead of shards.

Example:

```bash
python -m scripts.build_rym_npz ^
  data/my_games_rapid.pgn ^
  --out data/my_games_rapid.npz ^
  --min-rating 400 ^
  --max-rating 2400 ^
  --num-bins 10
```

---

## 7. Per-ply feature planes (`ply_features.py`)

The encoder maps each ply to a 64-plane board tensor:

```python
from scripts.ply_features import encode_ply_planes, NUM_PLANES
import chess

board = chess.Board(fen)
move = chess.Move.from_uci("e2e4")
planes = encode_ply_planes(board, move)  # (64, 8, 8)
```

The planes capture:

* Piece presence and geometric reach.
* Attack and defense structure.
* Aggregate control fields.
* Move-specific squares (from and to) for current and previous moves.
* Rule state (castling rights, side to move, en passant).

A conceptual layout:

**Board, presence, control (pre move)**

* Planes `0–11`: `board_pre`
  One plane for each `(color, piece_type)` combination, 6 types by 2 colors.
* Planes `12–23`: `presence_pre`
  Empty-board capture reach per `(color, piece_type)`, assuming an enemy piece at the target square.
* Planes `24–35`: `control_pre`
  Any control (attacks or defenses) per `(color, piece_type)` on the actual board.

**Aggregates (pre and post)**

* Planes `36–41`: pre-move aggregates:

  * 36: `white_any_control_pre`
  * 37: `black_any_control_pre`
  * 38: `white_legal_control_pre`
  * 39: `black_legal_control_pre`
  * 40: `white_net_control_pre` (pawns only)
  * 41: `black_net_control_pre` (pawns only)

* Planes `42–47`: post-move aggregates (same semantics after applying the move):

  * 42: `white_any_control_post`
  * 43: `black_any_control_post`
  * 44: `white_legal_control_post`
  * 45: `black_legal_control_post`
  * 46: `white_net_control_post`
  * 47: `black_net_control_post`

**Move squares (current and previous)**

* 48: from-square of the current move (one-hot).
* 49: to-square of the current move (one-hot).
* 50: from-square of the previous move (one-hot, or all zeros if none).
* 51: to-square of the previous move (one-hot, or all zeros if none).

**Rule state (pre move)**

* 52: `pre_white_can_castle_k`.
* 53: `pre_white_can_castle_q`.
* 54: `pre_black_can_castle_k`.
* 55: `pre_black_can_castle_q`.
* 56: `pre_side_to_move` (all ones if White to move, otherwise zeros).
* 57: `pre_en_passant_target` (one-hot on the en-passant target square).

**Rule state (post move)**

* 58: `post_white_can_castle_k`.
* 59: `post_white_can_castle_q`.
* 60: `post_black_can_castle_k`.
* 61: `post_black_can_castle_q`.
* 62: `post_side_to_move`.
* 63: `post_en_passant_target`.

All planes are binary. The orientation matches the standard board representation used by `python-chess`. The layout has been selected to align with Jupyter visualization tools that show one board per plane in an 8 × 8 grid.

---

## 8. Model zoo (`rym_models.py`)

`rym_models.py` defines a small model zoo and a `get_model` factory.

Available families:

* `"linear"` – single linear layer baseline on flattened board features.
* `"mlp"` – flattened board features followed by 1–2 hidden layers.
* `"cnn"` – shallow convolutional network over the 8 × 8 board.
* `"resnet"` – residual CNN with skip connections, tailored to 8 × 8 input.
* `"conv_transformer"` – convolutional stem followed by a transformer encoder over 64 square tokens.

Each family has `config_id` in `{0,1,2,3}` for variations in depth, channel count, and hidden dimensions.

Typical usage:

```python
from scripts.rym_models import get_model, MODEL_TYPES
from scripts.ply_features import NUM_PLANES

model = get_model(
    model_type="resnet",
    num_planes=NUM_PLANES,
    num_bins=num_bins,
    config_id=0,
)

logits, rating_pred = model(X_batch)  # X_batch: (B, NUM_PLANES, 8, 8)
# logits: (B, num_bins)
# rating_pred: (B,) or (B, 1) predicted Elo
```

The logits define a distribution over rating bands, and the regression head outputs a scalar Elo estimate. In many experiments, only the classification logits are required, with Elo derived from the band distribution.

---

## 9. Losses and single NPZ training (`train_rym.py`)

`train_rym.py` is a single-NPZ trainer, primarily intended for:

* Smoke tests.
* Small datasets (for example a personal PGN converted to `my_games.npz`).

### 9.1. Dataset wrapper

The dataset wrapper `RYMNpzDataset` provides convenient access to all NPZ fields:

```python
from pathlib import Path
from scripts.train_rym import RYMNpzDataset

ds = RYMNpzDataset(Path("data/my_small_dataset_train.npz"))

sample = ds[0]
sample["X"].shape     # (NUM_PLANES, 8, 8)
sample["y_bin"]       # rating band index
sample["y_elo"]       # average Elo
sample["game_id"]
sample["ply_idx"]

ds.num_bins
ds.min_rating
ds.max_rating
```

Game and ply identifiers make it possible to group predictions back into games if needed.

### 9.2. Loss structure

A helper `compute_losses(...)` combines several terms into a single scalar objective:

1. **Classification loss on rating bands (discrete Gaussian soft labels)**

   Rather than using a hard one-hot target at the true band index `y_bin`, the trainer builds a *soft* target distribution over bands by evaluating a discrete Gaussian in band space:

   * Bands are indexed `k = 0, ..., num_bins-1`.

   * For a given sample, the unnormalised target log-probability for band `k` is

     [
     \log t_y(k) \propto -\frac{(k - y_{\text{bin}})^2}{2\sigma^2},
     ]

     where `σ` is specified in **band units**.

   * After normalisation with a softmax, this gives a discrete Gaussian target distribution centred on the true band, with spread controlled by `σ`.

   This has two effects:

   * Near misses (e.g. predicting a neighbouring band) are penalised less than far-off predictions.
   * The amount of label smoothing is explicit and interpretable: smaller `σ` makes the target very sharp; larger `σ` spreads mass across multiple bands.

   The standard deviation is controlled by:

   * Module-level default: `GAUSSIAN_LABEL_SIGMA`.
   * CLI flag: `--gaussian-sigma` in both `train_rym.py` and `run_rym_experiments.py`.

2. **Regression loss on Elo (optional)**

   Elo is defined as the **expectation of Elo under the predicted band distribution**:

   * Let `band_centers[k]` be the midpoint rating of band `k`.

   * Given predicted band probabilities `p(k)`, the implied Elo is

     [
     \hat{r} = \sum_k p(k),\text{band_centers}[k].
     ]

   * The regression loss is mean squared error between `\hat{r}` and the true `y_elo`.

   The relative weight of this term is controlled by `alpha_reg`:

   * `alpha_reg = 0.0` disables the regression term (pure classification).
   * `alpha_reg > 0.0` adds `alpha_reg · loss_reg` to the total loss.

3. **Optional entropy regulariser**

   An optional entropy bonus is applied to the **predicted** distribution over bands:

   * Entropy:

     [
     H(p) = -\sum_k p(k)\log p(k).
     ]

   * The total classification term becomes:

     [
     \text{loss}*\text{cls} - \lambda*\text{ent}, H(p).
     ]

   * Positive `lambda_ent` encourages the model to avoid over-confident, spiky distributions when the position is uninformative.

Putting it together:

```text
loss = loss_cls + alpha_reg * loss_reg  - lambda_ent * H(pred)
```

where:

* `loss_cls` is cross-entropy against **Gaussian soft labels** with width `σ = --gaussian-sigma`,
* `loss_reg` is the optional Elo MSE from the band distribution,
* `H(pred)` is the entropy of the model’s predicted band distribution.

At inference time it is often sufficient to ignore the regression head and treat the band distribution as the primary object, deriving Elo from its expectation as above. The regression head is kept as an optional auxiliary signal.

### 9.3. CLI example (small monolithic NPZ)

For a small dataset processed via `build_rym_npz.py`:

```bash
python -m scripts.train_rym ^
  --train-npz data/my_small_dataset_train.npz ^
  --val-npz   data/my_small_dataset_val.npz   ^
  --model-type resnet ^
  --config-id 0 ^
  --batch-size 256 ^
  --epochs 10 ^
  --lr 1e-3 ^
  --alpha-reg 0.0 ^        # classification-only
  --lambda-ent 0.0 ^       # no extra entropy bonus
  --gaussian-sigma 1.0 ^   # label width in *band* units
  --device cuda ^
  --save-path models/my_small_experiment/rym_resnet_cfg0.pt
```

Guidelines:

* Smaller `--gaussian-sigma` (e.g. `0.7`) → sharper labels, heavier penalty for nearby-band mistakes.
* Larger `--gaussian-sigma` (e.g. `1.5`) → more smeared labels, more tolerance for nearby-band errors.
* `--lambda-ent` can be increased from `0.0` if you want to explicitly discourage over-confident posteriors on uninformative plies.

---

## 10. Main training entry point on shards (`run_rym_experiments.py`)

`run_rym_experiments.py` is the primary entry point for training on the large sharded Jan to Apr 2025 dataset. It:

1. Discovers NPZ shards according to prefixes and index specifications.
2. Infers rating metadata (`num_bins`, `min_rating`, `max_rating`) consistently across shards.
3. Builds data loaders shard by shard.
4. Trains a **single** model family + configuration per invocation.
5. Optionally evaluates on balanced test shards and unbalanced real-world shards whenever a new best validation loss is found.
6. Saves a checkpoint with the model state and summary metrics.

### 10.1. Shard prefixes and defaults

`run_rym_experiments.py` is configured so that defaults match the data layout produced by `run_2025_pipeline.py`:

```text
--train-prefix    data/rym_2025_jan_apr_tc300+0_bin200_train_shard
--val-prefix      data/rym_2025_jan_apr_tc300+0_bin200_val_shard
--test-prefix     data/rym_2025_jan_apr_tc300+0_bin200_test_shard
--realtest-prefix data/rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard
```

Shard files are expected as:

```text
<frozen prefix><NNN>.npz
# for example
data/rym_2025_jan_apr_tc300+0_bin200_train_shard000.npz
data/rym_2025_jan_apr_tc300+0_bin200_train_shard001.npz
...
```

The script reads `num_bins`, `min_rating`, and `max_rating` from the first train shard and checks that all shards touched by the run share the same metadata.

### 10.2. Selecting shards and running experiments

Shard lists are controlled by simple string arguments:

* `"all"` uses all shards that exist for that prefix.
* `"0"` uses shard `000` only.
* `"0,1,2,3"` uses shards 000, 001, 002, 003.

Applies to:

* `--train-shards`
* `--val-shards`
* `--test-shards`
* `--realtest-shards`

#### Example: smoke test on a few shards

```bash
python -m scripts.run_rym_experiments ^
  --model-type resnet ^
  --config-id 0 ^
  --train-shards 0,1,2,3 ^   # four train shards
  --val-shards   0 ^         # one val shard
  --test-shards  0 ^         # one balanced test shard
  --realtest-shards 0 ^      # one unbalanced real-world shard
  --batch-size 256 ^
  --epochs 2 ^
  --lr 1e-3 ^
  --alpha-reg 0.0 ^
  --lambda-ent 0.0 ^
  --gaussian-sigma 1.0 ^
  --device cuda ^
  --num-workers 8 ^
  --seed 64 ^
  --ckpt-path models/rym_2025_jan_apr_resnet_cfg0_smoke.pt
```

#### Example: full Jan–Apr balanced + real-world evaluation

```bash
python -m scripts.run_rym_experiments ^
  --model-type resnet ^
  --config-id 0 ^
  --train-shards all ^
  --val-shards   all ^
  --test-shards  all ^
  --realtest-shards all ^
  --batch-size 256 ^
  --epochs 5 ^
  --lr 1e-3 ^
  --alpha-reg 0.0 ^
  --lambda-ent 0.0 ^
  --gaussian-sigma 1.0 ^
  --device cuda ^
  --num-workers 8 ^
  --seed 64 ^
  --ckpt-path models/rym_2025_jan_apr_resnet_cfg0_full.pt
```

Internally, the script:

* Streams over train shards and accumulates weighted averages of train loss components.
* Evaluates over validation shards at the end of each epoch.
* Whenever validation loss improves, it:

  * Updates the “best so far” validation metrics.
  * Optionally evaluates on the balanced test shards (if any).
  * Optionally evaluates on the unbalanced real-world shards (if any).

Checkpoints are saved to the path given by `--ckpt-path`. Each checkpoint stores at least:

* `model_state`
* `model_type`, `config_id`
* `num_planes`, `num_bins`, `min_rating`, `max_rating`
* `best_epoch`
* `best_val_metrics`
* `test_metrics`
* `realtest_metrics`
* `alpha_reg`, `lambda_ent`, `gaussian_sigma`

---

## 11. Evaluation + visualisation helpers (`rym_analysis.py`, `rym_vis.py`, `display.ipynb`)

With training and data generation in place, the *new* recommended workflow for inspecting a model is:

1. Evaluate a trained checkpoint on a held-out NPZ shard.
2. Summarise posterior rating distributions per game.
3. Visualise confusion matrices and posterior trajectories.
4. For a specific game (e.g. from `gte.pgn`), render board frames + rating-distribution frames and stitch them into an animation.

This is handled by `rym_analysis.py`, `rym_vis.py`, and the notebook `display.ipynb`.

### 11.1. `rym_analysis.py`: model evaluation + Bayesian aggregation

`rym_analysis.py` provides model-agnostic helpers for running a trained checkpoint on an NPZ shard and summarising the results. 

Core pieces:

* **Checkpoint loading**

  ```python
  from scripts.rym_analysis import load_model_from_checkpoint

  model, meta = load_model_from_checkpoint(
      ckpt_path="models/rym_2025_jan_apr_resnet_cfg3_gauss2.0.pt",
      num_planes=NUM_PLANES,
      num_bins=10,
      device="cuda",
  )
  ```

  This reconstructs the correct model architecture from the checkpoint metadata when possible, and falls back to `num_planes`/`num_bins` arguments when needed.

* **Per-ply prediction on NPZ (`predict_probs_for_npz`)**

  Runs the model on all plies in an NPZ and returns a tidy DataFrame:

  ```text
  game_id, ply_idx, y_bin, y_elo,
  pred_bin, pred_rating,
  prob_<lo>_<hi>  (one column per Elo band, e.g. prob_400_600)
  ```

* **Bayesian updates over plies (`_bayes_update_path`)**

  Given per-ply likelihoods over bands, the code maintains a tempered posterior `p_t` over Elo bands as the game progresses:

  ```text
  p_0  = uniform over bands
  lik_t = softmax(logits_t)      # model per-ply likelihood
  p_t  ∝ (p_{t-1}^γ) * (lik_t^α)
  ```

  where:

  * `α < 1` flattens new evidence (more cautious).
  * `γ < 1` flattens the old posterior (prevents runaway spikes).
  * `eps` floors probabilities away from zero.

* **Per-game posterior summary (`summarize_game_posteriors`)**

  For each game, it:

  * Runs the Bayesian updates across its plies.
  * Extracts the **final posterior** over bands.
  * Computes:

    * Posterior mean Elo (`post_mean_elo`).
    * Posterior 95% interval (`post_lower_elo`, `post_upper_elo`).
    * Regression head Elo (`reg_last_elo`, `reg_mean_elo`, if available).
    * Final posterior probabilities for each band as `post_<lo>_<hi>`.

* **Aggregate metrics (`compute_metrics`)**

  Computes both per-ply and per-game metrics, such as:

  * `ply_band_accuracy` – how often the argmax band matches `y_bin`.
  * `ply_reg_mae` – MAE between `pred_rating` and `y_elo`.
  * `game_post_mae` – MAE between posterior mean Elo and `y_elo`.
  * `game_post_band_accuracy` – accuracy of the final posterior argmax vs `y_bin`.

* **Convenience wrappers**

  * `evaluate_model_on_npz(npz_path, ckpt_path, ...)` – the main entry point used in `display.ipynb`. Returns `(df_out, game_summary, metrics, meta)`.
  * `evaluate_model_on_pgn(pgn_path, ckpt_path, npz_path=..., ...)` – builds a tiny NPZ for a specific PGN (e.g. `gte.pgn`), then runs the same evaluation.

  Example (matching the notebook):

  ```python
  from scripts.rym_analysis import evaluate_model_on_npz, compute_band_distribution_table

  df_out, game_summary, metrics, meta = evaluate_model_on_npz(
      npz_path=TEST_NPZ_PATH,
      ckpt_path=CKPT_PATH,
      batch_size=512,
      device="cuda",
      alpha=0.7,
      gamma=1.0,
      eps=1e-8,
  )

  band_table = compute_band_distribution_table(df_out, game_summary)
  ```

  In `display.ipynb`, this is done both for the balanced test shard and for the unbalanced “real test” shard, with band-distribution tables and confusion heatmaps plotted for each.

There is also a small CLI:

```bash
python -m scripts.rym_analysis \
  --npz data/rym_2025_jan_apr_tc300+0_bin200_test_shard001.npz \
  --ckpt models/rym_2025_jan_apr_resnet_cfg3_gauss2.0.pt \
  --batch-size 512 --device cuda
```

which logs high-level metrics to stdout.

---

### 11.2. `rym_vis.py`: feature planes, posteriors, and replay frames

`rym_vis.py` collects all the plotting and frame-generation utilities used by the notebook. 

It can be roughly grouped into three layers:

#### (a) Feature-plane visualisation

These functions let you visually audit the `64 × 8 × 8` feature encoding.

* `build_ply_metadata_from_pgn(pgn_path)`

  Parses a PGN and returns a mapping:

  ```text
  (game_id, ply_idx) → { "fen_pre", "fen_post", "move_uci" }
  ```

  This matches the `(game_id, ply_idx)` pairs stored in the NPZ.

* `show_planes_grid(fen, move_uci, size=120, show_labels=True)`

  Re-encodes a position + move via `encode_ply_planes` and renders an `8 × 8` HTML grid, one board per plane, using inline SVG. Useful for sanity-checking the conceptual layout.

* `show_npz_planes_grid_for_row(npz_path, ply_meta, row_index, ...)`

  Uses **stored** planes from an NPZ row, plus PGN metadata, to show:

  * Board-before vs board-after,
  * All 64 planes in a grid, with the `lastmove` arrow drawn and each plane labeled.

  In `display.ipynb` this is used on a specific ply of `gte.pgn` to show exactly what the model sees for that move.

#### (b) Per-game probability / Elo plots

Once you have `df_out` for a single game (`df_out_gte[df_out_gte["game_id"] == game_id]`), you can make two main views:

* `plot_ply_likelihoods_with_elo(df_game, min_rating, max_rating, ...)`

  * Stacked per-ply **likelihoods** (`prob_*` columns) as a coloured bar chart over rating bands.
  * Overlaid Elo prediction (`pred_rating`) as a black line.
  * A narrow legend panel listing the band labels.

  The highest-probability band at each ply is alpha-highlighted so you can quickly see when the model is confident vs uncertain.

* `plot_bayesian_posterior_with_elo_and_legend(df_game, alpha, gamma, eps, ...)`

  * Applies the same tempered Bayesian updates as `rym_analysis._bayes_update_path`.

  * Plots stacked **posterior** probabilities per ply.

  * Overlays:

    * Posterior mean Elo (red dashed line).
    * Optional regression head Elo (black line) if available.

  * Uses alpha coding to distinguish:

    * The band that is the per-ply likelihood argmax.
    * The band that is the posterior argmax.
    * All other bands.

Both functions use the same `prob_<lo>_<hi>` columns and band labels, so it is easy to compare “raw per-ply likelihoods” vs “Bayesian belief after seeing the whole game”.

Additionally:

* `plot_band_confusion_heatmap(df_out, game_summary, band_labels=...)`

  * Builds “soft” confusion matrices over rating bands:

    * Left: per-ply `P(predicted band | true band)` from `df_out`.
    * Right: per-game `P(predicted band | true band)` using the **final posterior** from `game_summary`.

  * The matrices are oriented with true band on the x-axis and predicted band on the y-axis, with a custom brown-to-cream colormap.

  This is used in `display.ipynb` twice: once for the balanced test shard, once for the unbalanced real-world shard.

#### (c) PNG board rendering + animation frames

These helpers build board diagrams and posterior plots as **aligned PNG sequences** which can be stitched into replay animations.

* `_render_board_to_pil(board, lastmove, size=320)`

  * Renders a simple 2D board using Unicode chess glyphs (♙♟♘…); no Cairo/cairosvg dependencies.
  * Highlights the last move’s from/to squares with a green rectangle.
  * Picks a font robustly (FreeSerif → DejaVu → PIL default).

* `_build_posterior_frame_schedule(num_plies)`

  * Internal utility that defines a **shared frame schedule** for boards and distributions.

  * For `T` plies it returns `1 + 3T` frames in this order:

    ```text
    0:  prior_only        (initial prior, no ply)
    1:  prior_lik  (ply 0)
    2:  post_lik   (ply 0)
    3:  post_only  (ply 0)
    4:  prior_lik  (ply 1)
    5:  post_lik   (ply 1)
    6:  post_only  (ply 1)
    ...
    ```

  * Both `save_board_frames_for_pgn` and `save_dist_frames_for_game` use this schedule so that filenames line up 1-to-1 by sorted order.

* `save_board_frames_for_pgn(pgn_path, game_id=0, out_root="plots/rym_inspect_conv", ...)`

  * Loads the `game_id`-th game from a PGN.

  * Pre-computes the board after each ply.

  * For each frame in the schedule, writes a PNG board image:

    * `prior_only` and `prior_lik` frames use the board **before** the relevant move.
    * `post_lik` / `post_only` frames use the board **after** that move, with the move highlighted.

  * Some board images are deliberately duplicated (e.g. `post_lik` vs `post_only`) so the frame indices match the distribution frames exactly.

* `save_dist_frames_for_game(df_game, out_root, dist_subdir="dist", prefix="frame", ...)`

  * Takes a single-game `df_game` (from `evaluate_model_on_pgn`).

  * Builds a Gaussian basis over rating, then converts band probabilities to smooth curves.

  * For each frame in the shared schedule:

    * Draws the current main prior/posterior as a blue shaded curve,
    * Optionally draws the per-ply likelihood as a green curve,
    * Adds vertical lines for posterior mean Elo and true Elo (if available),
    * Uses titles of the form `"1."`, `"1. e4"`, `"1...c5"`, with the final frame optionally overwritten by `game_result`.

  * Y-axis is fixed to `[0, target_ymax]` and displayed as a percentage for consistency; curves are globally scaled so they fit that range.

* `stitch_board_and_dist(root_dir, boards_subdir="boards", dist_subdir="dist", out_subdir="combined", flatten_factor=1.0)`

  * Takes sorted PNGs from `boards/` and `dist/` and vertically stacks them:

    * Board on top,
    * Distribution panel on the bottom.

  * Ensures the widths match; optionally squashes the distribution panel vertically by `flatten_factor` (e.g. `0.7` for a flatter plot).

  * Writes the result into `combined/` using the board filenames.

* `replay_images(image_dir, delay=0.05, display_width=360)`

  * Simple Jupyter helper that loops over PNGs in a folder, doing `clear_output(wait=True)` between frames for a lightweight animation preview.

In the new workflow, `display.ipynb` calls:

```python
save_board_frames_for_pgn(...)
save_dist_frames_for_game(...)
stitch_board_and_dist(...)
replay_images(...)
```

to produce a clean “board above, rating distribution below” replay for the chosen game.

---

### 11.3. `display.ipynb`: main end-to-end demo

`display.ipynb` is the recommended entry point for **humans**:

> “Given a trained model checkpoint and some test data, show me the metrics, confusion matrices, feature planes, and a full replay animation for a single game.”

The notebook does four main things:

1. **Evaluate on a balanced test shard**

   * Sets:

     ```python
     TEST_NPZ_PATH = Path("data/rym_2025_jan_apr_tc300+0_bin200_test_shard001.npz")
     CKPT_PATH     = Path("models/rym_2025_jan_apr_resnet_cfg3_gauss2.0.pt")
     DEVICE        = "cuda"
     ```

   * Calls `evaluate_model_on_npz` to get `df_out_test`, `game_summary_test`, `metrics_test`, `meta_test`.

   * Prints aggregate test metrics (per-ply and per-game MAEs, accuracies).

   * Builds a rating-band summary table with `compute_band_distribution_table` and shows it as a styled DataFrame.

   * Plots per-ply and per-game confusion heatmaps via `plot_band_confusion_heatmap`.

2. **Evaluate on an unbalanced “real test” shard**

   * Repeats the same pattern on:

     ```python
     R_TEST_NPZ_PATH = Path("data/rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard001.npz")
     ```

   * Prints a separate metrics block and confusion heatmap for the more realistic rating distribution.

3. **Evaluate a specific PGN (`gte.pgn`)**

   * Uses:

     ```python
     from scripts.rym_analysis import evaluate_model_on_pgn
     ```

     to run the model on `gte.pgn`:

     ```python
     df_out_gte, game_summary_gte, metrics_gte, meta_gte = evaluate_model_on_pgn(
         pgn_path=PGN_PATH,
         ckpt_path=CKPT_PATH,
         npz_path=NPZ_GTE,
         min_rating=400,
         max_rating=2400,
         num_bins=10,
         batch_size=64,
         device=DEVICE,
         alpha=0.7,
         gamma=1.0,
         eps=1e-8,
     )
     ```

   * Sorts `df_out_gte` by `ply_idx` to get `df_game`, and extracts `game_id` and `num_plies`.

   * Builds PGN-derived metadata via:

     ```python
     ply_meta = build_ply_metadata_from_pgn(PGN_PATH)
     ```

   * Calls `show_npz_planes_grid_for_row` for a specific `row_index` to visualise all 64 feature planes for a chosen ply, showing:

     * Pre/post boards,
     * The move being played,
     * Which squares are active in each plane.

   * Plots:

     * `plot_ply_likelihoods_with_elo(df_game, min_rating=0, max_rating=2500)`
     * `plot_bayesian_posterior_with_elo_and_legend(df_game, alpha=0.7, gamma=1.0, ...)`

     to show:

     * Per-ply likelihood bands + Elo,
     * Posterior evolution over the game + posterior vs regression Elo.

4. **Render replay frames (board + distribution) and stitch**

   * Extracts SAN move labels and the game result from `chess.pgn.read_game`.

   * Sets:

     ```python
     OUT_ROOT = Path("plots/rym_inspect_conv")
     ```

     and clears it if it already exists.

   * Calls:

     ```python
     save_board_frames_for_pgn(
         pgn_path=PGN_PATH,
         game_id=game_id,
         out_root=OUT_ROOT,
         boards_subdir="boards",
         size=320,
         prefix="frame",
     )

     save_dist_frames_for_game(
         df_game=df_game,
         out_root=OUT_ROOT,
         dist_subdir="dist",
         prefix="frame",
         move_labels=moves_san,
         game_result=game_result,
     )
     ```

     so that:

     * `plots/rym_inspect_conv/boards/frame_0000.png` … contain boards,
     * `plots/rym_inspect_conv/dist/frame_0000.png` … contain distributions,

     with filenames aligned according to the shared frame schedule.

   * Stitches and replays:

     ```python
     stitch_board_and_dist(
         root_dir=OUT_ROOT,
         boards_subdir="boards",
         dist_subdir="dist",
         out_subdir="combined",
         flatten_factor=1.0,
     )

     replay_images(OUT_ROOT / "combined", delay=0.5, display_width=640)
     ```

     which:

     * Produces `plots/rym_inspect_conv/combined/frame_0000.png`, … with the board stacked above the rating distribution.
     * Plays them as a simple GIF-like replay inside the notebook.

In practice, to adapt the notebook to a different run you typically only touch:

* `TEST_NPZ_PATH` / `R_TEST_NPZ_PATH` (which shards to evaluate).
* `CKPT_PATH` (which trained model to load).
* `PGN_PATH` / `NPZ_GTE` (which custom PGN to inspect).

---

## 12. Summary and future directions

With the current codebase, the RYM pipeline provides:

* A scalable preprocessing stack from raw Lichess PGN dumps to balanced and unbalanced move-level datasets, with both monolithic and sharded NPZ representations.

* A family of local board encoders, with principled Bayesian aggregation over plies at inference time.

* A modern evaluation + visualisation stack:

  * `rym_analysis.py` for running checkpoints on NPZs and summarising per-game posteriors.
  * `rym_vis.py` for feature planes, band confusion heatmaps, and replay frame generation.
  * `display.ipynb` as the main interactive demo.

* Tools to visualise how rating beliefs evolve move by move and to compare architectures on the same dataset, within a consistent and documented Lichess blitz setting.

Natural extensions include:

* Game-level training objectives:

  * Use `game_id` and `ply_idx` to compute a game-level posterior and add a loss term based on the final posterior.

* Full-game sequence transformers:

  * Consume sequences of plies `(X_0, ..., X_{T-1})` directly and learn an internal rating belief state.

* Richer metadata:

  * Clock times, engine evaluations, opening families, and time-usage patterns.

* Multi-task setups:

  * Predict rating, time control, or style clusters jointly.

* Cross-month generalisation:

  * Train on some months and evaluate out-of-distribution on others.

The updated workflow should make it easy to go from **raw PGNs → trained model → interactive visual analysis of a specific game** using a small, composable set of scripts and a single notebook.