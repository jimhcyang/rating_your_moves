# Rating Your Moves (RYM)

This repository implements a complete data + modeling pipeline for studying how human chess moves reveal playing strength, using large scale Lichess game archives.

Conceptually:

> Given a sequence of human moves, what rating distribution is most compatible with those decisions?

The core pieces are:

1. Monthly Lichess indexing (PGN -> Parquet index).
2. Balanced and unbalanced PGN builders by rating band and time control.
3. Per ply feature encoding into a `64 x 8 x 8` tensor.
4. Sharded NPZ datasets for train, validation, test, and a real world test.
5. A model zoo (linear, MLP, CNN, ResNet, conv transformer).
6. Training and experiment runners over NPZ shards.
7. Inspection tools for visualizing per move rating distributions and board planes.

The default modern dataset is Jan to Apr 2025 5+0 blitz, with rating bands from 400 to 2400 in 200 point steps.

---

## 1. Data source and scope

The pipeline is built around the official Lichess monthly database:

* Standard rated games in `.pgn.zst` format (for example `lichess_db_standard_rated_2025-01.pgn.zst`).
* Archives are typically fetched from `https://database.lichess.org/standard/`.

Each monthly file contains many millions of games. The code is designed to:

* Download `.pgn.zst` if missing.
* Decompress to `.pgn` once per month.
* Build a row wise index with file offsets and metadata.
* Store the index in Parquet for fast filtering and aggregation.

The main experimental configuration targets:

* Months 2025-01, 2025-02, 2025-03, 2025-04.
* Time control `300+0` (5+0 blitz).
* Rating range `[400, 2400)`, with bands of width 200 (10 bands total).

A dedicated driver `run_2025_pipeline.py` builds a Jan to Apr 2025 dataset with:

* A balanced rating band dataset split into train, validation, and test.
* An additional unbalanced "real world" test set for evaluation.

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

    inspect_rym_game.py     # visual per move inspector

  models/
    rym_2017-04_baselines/
      rym_cnn_cfg0.pt
      rym_conv_transformer_cfg0.pt
      rym_linear_cfg0.pt
      rym_mlp_cfg0.pt
      rym_resnet_cfg0.pt

  README.md
  requirements.txt
````

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
* Compute descriptive statistics for rating and time control structure.

Rating bands:

* Games are assigned to bands using an average Elo:

  * `avg_elo = (white_elo + black_elo) / 2`.
* For the 2025 setup, the default bands are:

  * `[400,600), [600,800), ..., [2200,2400)` (10 bands, width 200).
* Games with very lopsided pairings (large rating gaps between players) can be dropped.

Example: Full month statistics for January 2025:

```bash
python -m scripts.index_stats 2025-01
```

Example: Only 5+0 games:

```bash
python -m scripts.index_stats 2025-01 --where 'time_control == "300+0"'
```

The script can emit:

* Rating histograms.
* Time control bar charts.
* Rating by time control heatmaps.

Drop summaries are printed so that excluded games (missing ratings, unparsable tags, etc.) are explicit.

### 4.3. Game filtering (`filter_games.py`)

Purpose:

* Load a Parquet index for a month.
* Apply SQL like filters on columns.
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

  * Default 2025 configuration: `min_rating = 400`, `max_rating = 2400`, `bin_size = 200` resulting in 10 bands.
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

### 5.2. Unbalanced real world dataset (`build_unbalanced_pgn.py`)

Purpose:

* Build an unbalanced PGN sample that reflects the natural rating distribution after filtering, for use as a more realistic test set.

Behavior:

* Reuse indexing and rating helpers.
* Apply the same rating range `[min_rating, max_rating)` and filters as the balanced set.
* Do not perform per band balancing.
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

### 5.3. One shot Jan to Apr 2025 pipeline (`run_2025_pipeline.py`)

For Jan to Apr 2025 there is a single driver that runs the full data pipeline:

1. Reads monthly indexes for 2025-01 to 2025-04.
2. Filters to `time_control == "300+0"`.
3. Bands games into 200 Elo buckets from 400 to 2400.
4. Samples a balanced dataset across bands and splits into train, validation, and test by game.
5. Writes balanced PGNs and associated NPZ shards.
6. Builds an additional unbalanced real world test set and NPZ shards.

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
  * ...
  * `data/rym_2025_jan_apr_tc300+0_bin200_train_shard079.npz`
  * `data/rym_2025_jan_apr_tc300+0_bin200_val_shard000.npz`
  * ...
  * `data/rym_2025_jan_apr_tc300+0_bin200_val_shard009.npz`
  * `data/rym_2025_jan_apr_tc300+0_bin200_test_shard000.npz`
  * ...
  * `data/rym_2025_jan_apr_tc300+0_bin200_test_shard009.npz`
* Unbalanced real world test:

  * `data/rym_2025_jan_apr_tc300+0_unbalanced_realtest.pgn`
  * `data/rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard000.npz`
  * ...
  * `data/rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard009.npz`
* Index:

  * `data/rym_2025_jan_apr_tc300+0_bin200_index.parquet` with per game metadata and band labels.

The driver can optionally remove raw monthly `.pgn` or `.pgn.zst` files after processing if cleanup flags are set.

---

## 6. NPZ schema (per shard and optional monolithic files)

For the main Jan to Apr 2025 dataset, the pipeline produces only sharded NPZ files. Every `*_shardXXX.npz` stores ply level examples.

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

## 7. Per ply feature planes (`ply_features.py`)

The encoder maps each ply to a 64 plane board tensor:

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
* Move specific squares (from and to) for current and previous moves.
* Rule state (castling rights, side to move, en passant).

A conceptual layout:

**Board, presence, control (pre move)**

* Planes `0-11`: `board_pre`
  One plane for each `(color, piece_type)` combination, 6 types by 2 colors.
* Planes `12-23`: `presence_pre`
  Empty board capture reach per `(color, piece_type)`, assuming an enemy piece at the target square.
* Planes `24-35`: `control_pre`
  Any control (attacks or defenses) per `(color, piece_type)` on the actual board.

**Aggregates (pre and post)**

* Planes `36-41`: pre move aggregates:

  * 36: `white_any_control_pre`
  * 37: `black_any_control_pre`
  * 38: `white_legal_control_pre`
  * 39: `black_legal_control_pre`
  * 40: `white_net_control_pre` (pawns only)
  * 41: `black_net_control_pre` (pawns only)
* Planes `42-47`: post move aggregates (same semantics after applying the move):

  * 42: `white_any_control_post`
  * 43: `black_any_control_post`
  * 44: `white_legal_control_post`
  * 45: `black_legal_control_post`
  * 46: `white_net_control_post`
  * 47: `black_net_control_post`

**Move squares (current and previous)**

* 48: from square of the current move (one hot).
* 49: to square of the current move (one hot).
* 50: from square of the previous move (one hot, or all zeros if none).
* 51: to square of the previous move (one hot, or all zeros if none).

**Rule state (pre move)**

* 52: `pre_white_can_castle_k`.
* 53: `pre_white_can_castle_q`.
* 54: `pre_black_can_castle_k`.
* 55: `pre_black_can_castle_q`.
* 56: `pre_side_to_move` (all ones if White to move, otherwise zeros).
* 57: `pre_en_passant_target` (one hot on the en passant target square).

**Rule state (post move)**

* 58: `post_white_can_castle_k`.
* 59: `post_white_can_castle_q`.
* 60: `post_black_can_castle_k`.
* 61: `post_black_can_castle_q`.
* 62: `post_side_to_move`.
* 63: `post_en_passant_target`.

All planes are binary. The orientation matches the standard board representation used by `python-chess`. The layout has been selected to align with Jupyter visualization tools that show one board per plane in an 8 x 8 grid.

---

## 8. Model zoo (`rym_models.py`)

`rym_models.py` defines a small model zoo and a `get_model` factory.

Available families:

* `"linear"`
  Single linear layer baseline on flattened board features.

* `"mlp"`
  Flattened board features followed by 1 to 2 hidden layers.

* `"cnn"`
  Shallow convolutional network over the 8 x 8 board.

* `"resnet"`
  Residual CNN with skip connections, tailored to 8 x 8 input.

* `"conv_transformer"`
  Convolutional stem followed by a transformer encoder over 64 square tokens.

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

`train_rym.py` is a single NPZ trainer, primarily intended for:

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

A helper `compute_losses(...)` combines several terms:

1. Classification loss on rating bands:

   * Encourages probability mass near the true band.
   * Penalizes far off predictions more heavily than near misses, reflecting that a one band error is less severe than a large band error.

2. Regression loss on Elo (optional):

   * Mean squared error between `rating_pred` and `y_elo`.

3. Optional regularizers:

   * An entropy term can be added to encourage smoother distributions (avoid overconfident, spiky predictions).
   * Other terms can be attached as needed.

Conceptually:

```text
loss = loss_cls + alpha_reg * loss_reg + lambda_ent * loss_entropy
```

* Setting `alpha_reg = 0` disables the regression term.
* Setting `lambda_ent = 0` disables the entropy bonus.

At inference time it is often sufficient to use the classification head and derive Elo from the predicted band distribution. The regression head is kept as an optional secondary view.

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
  --alpha-reg 0.0 ^  # classification focus
  --device cuda ^
  --save-path models/my_small_experiment/rym_resnet_cfg0.pt
```

For the large Jan to Apr 2025 dataset, `run_rym_experiments.py` with shards is the preferred route.

---

## 10. Main training entry point on shards (`run_rym_experiments.py`)

`run_rym_experiments.py` is the primary entry point for training on the large sharded Jan to Apr 2025 dataset. It:

1. Discovers NPZ shards according to prefixes and index specifications.
2. Builds data loaders across shards.
3. Trains one or more model families and configuration IDs.
4. Optionally evaluates on balanced test shards and unbalanced real world shards.
5. Saves checkpoints and summary metrics.

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

The script reads `num_bins`, `min_rating`, and `max_rating` from the first train shard and assumes all shards share the same metadata.

### 10.2. Selecting shards

Shard lists are controlled by simple string arguments:

* `"all"` uses all shards that exist for that prefix.
* `"0"` uses shard `000` only.
* `"0,1,2,3"` uses shards 000 to 003.

Applies to:

* `--train-shards`
* `--val-shards`
* `--test-shards`
* `--realtest-shards`

Example: a small smoke test with four train shards per one validation shard:

```bash
python -m scripts.run_rym_experiments ^
  --train-shards 0,1,2,3 ^   # four train shards
  --val-shards   0 ^         # one val shard
  --test-shards  '' ^        # skip balanced test
  --realtest-shards '' ^     # skip unbalanced real world test
  --models resnet ^
  --config-id 0 ^
  --batch-size 256 ^
  --epochs 2 ^
  --lr 1e-3 ^
  --alpha-reg 0.0 ^
  --device cuda ^
  --save-dir models/rym_2025_jan_apr_smoke
```

For the full Jan to Apr balanced dataset:

```bash
python -m scripts.run_rym_experiments ^
  --train-shards all ^
  --val-shards   all ^
  --test-shards  all ^
  --realtest-shards all ^
  --models resnet,conv_transformer ^
  --config-id all ^
  --batch-size 256 ^
  --epochs 5 ^
  --lr 1e-3 ^
  --alpha-reg 0.0 ^
  --device cuda ^
  --save-dir models/rym_2025_jan_apr_baselines
```

Internally, the script:

* Builds shard aware datasets and loaders.
* Loops over `(model_type, config_id)` combinations.
* Logs per epoch metrics such as:

  * train: total loss, classification loss, regression loss (if used).
  * validation: total loss, classification loss, regression loss, band accuracy, Elo error.
  * test and real world metrics if shards are specified.

Checkpoints are saved as:

```text
{save_dir}/rym_{model_type}_cfg{config_id}.pt
# for example
models/rym_2025_jan_apr_baselines/rym_resnet_cfg0.pt
```

Each checkpoint stores:

* `model_state`
* `model_type`, `config_id`
* `num_planes`, `num_bins`
* possibly additional training metadata such as best epoch and validation metrics.

---

## 11. Inspecting per move predictions (`inspect_rym_game.py`)

`inspect_rym_game.py` provides tooling for model interpretation at the game level. It can:

* Take a PGN and one or more trained models.
* Re encode each ply using `ply_features.py`.
* Run all selected models on every ply.
* Maintain a Bayesian posterior over rating bands as the game progresses.
* Produce:

  * A table of Elo estimates by ply.
  * Per move visualizations (one image per ply) showing board position and rating distribution.

### 11.1. Bayesian update across plies

For a given game and model, the script collects per ply logits:

```python
logits_seq: (T, num_bins)
rating_seq: (T,)
```

At each ply `t`, it forms a likelihood:

```python
lik_t = softmax(logits_seq[t])
```

The game level posterior `p_t` over rating bands is updated multiplicatively:

```text
p_0 = uniform over bands
p_t proportional to p_{t-1} * lik_t
```

After normalization, `p_t` represents the posterior over rating bands after plies `0..t`.

From `p_t`, an expected Elo can be derived:

```text
cls_elo_t = sum_k p_t(k) * center(k)
```

where `center(k)` is the midpoint of band `k`. If a regression head is present, its Elo prediction `rating_seq[t]` can be plotted alongside the classification based expectation.

### 11.2. Pretty printed tables

A helper prints a text table summarizing model predictions over plies. Example structure:

```text
Game: Alice vs Bob, true avg Elo approximately 1850.0
---------------------------------------------------------------------------
ply  move   resnet_cls  resnet_reg  convT_cls  convT_reg
1    e4       1650.3      1612.5     1720.8     1690.4
2    ...c5    1675.1      1630.0     1742.1     1702.7
3    Nf3      ...
...
---------------------------------------------------------------------------
```

This makes it easy to see how quickly each architecture converges toward the true rating and which moves cause large jumps.

### 11.3. Per move visualization

`inspect_rym_game.py` can also generate PNGs, one per ply, containing:

* A board diagram with the current move played.
* Curves representing smoothed band distributions:

  * Each model's band posterior is convolved with Gaussian bumps centered at the rating band midpoints.
  * Curves are scaled to be visually readable; the shape carries the main information.
* Vertical lines indicating:

  * The expected Elo under the posterior.
  * The regression Elo (if present).

The x axis is Elo from `min_rating` to `max_rating`. The y axis is an arbitrary scale used for visual clarity.

Filenames resemble:

```text
plots/rym_inspect/game_000_ply_001.png
plots/rym_inspect/game_000_ply_002.png
...
```

A small utility allows interactive replay in a notebook:

```python
from scripts.inspect_rym_game import replay_images
replay_images("plots/rym_inspect", delay=0.25)
```

### 11.4. Custom PGNs

`inspect_rym_game.py` can operate on arbitrary PGNs. If `--test-npz` is omitted, the script will derive a small monolithic NPZ for that PGN on the fly using the same encoding logic.

Example:

```bash
python -m scripts.inspect_rym_game ^
  --test-pgn data/rym_2025_jan_apr_tc300+0_bin200_test.pgn ^
  --models-dir models/rym_2025_jan_apr_baselines ^
  --config-id 0 ^
  --device cuda ^
  --out-dir plots/rym_inspect
```

For a personal PGN:

```bash
python -m scripts.inspect_rym_game ^
  --test-pgn my_games_rapid.pgn ^
  --models-dir models/rym_2025_jan_apr_baselines ^
  --config-id 0 ^
  --device cuda ^
  --out-dir plots/rym_my_games
```

---

## 12. Summary and future directions

With the current codebase, the RYM pipeline provides:

* A scalable preprocessing stack from raw Lichess PGN dumps to balanced and unbalanced move level datasets, with both monolithic and sharded NPZ representations.
* A family of local board encoders, with principled Bayesian aggregation over plies at inference time.
* Tools to visualize how rating beliefs evolve move by move and to compare architectures on the same game.
* A modern, reproducible experimental setup focused on Jan to Apr 2025 5+0 blitz, ratings 400 to 2400, with 200 point bands, implemented primarily for a Windows and CUDA environment.

Natural extensions include:

* Game level training objectives:

  * Use `game_id` and `ply_idx` to compute a game level posterior and add a loss term based on the final posterior.
* Full game sequence transformers:

  * Consume sequences of plies `(X_0, ..., X_{T-1})` directly and learn an internal rating belief state.
* Richer metadata:

  * Clock times, engine evaluations, opening families, and time usage patterns.
* Multi task setups:

  * Predict rating, time control, or style clusters jointly.
* Cross month generalization:

  * Train on some months and evaluate out of distribution on others.

The existing code already supports detailed studies of how much information each move carries about playing strength and how different architectures "see" the same game, within a consistent and documented Lichess blitz setting.