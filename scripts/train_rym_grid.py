#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

MODEL_TYPES: List[str] = ["linear", "mlp", "cnn", "resnet", "conv_transformer"]

LOSS_CONFIGS: List[Dict[str, Any]] = [
    {"alpha_reg": 0.5,  "lambda_ent": 0.0,  "gaussian_sigma": -1.0, "tag": "ce_a0p5_l0"},
    {"alpha_reg": 0.32, "lambda_ent": 0.16, "gaussian_sigma": -1.0, "tag": "ce_a0p32_l0p16"},
    {"alpha_reg": 0.0,  "lambda_ent": 0.32, "gaussian_sigma": 1.0,  "tag": "g1_a0_l0p32"},
    {"alpha_reg": 0.0,  "lambda_ent": 0.0,  "gaussian_sigma": 2.0,  "tag": "g2_a0_l0"},
]

LRS: List[float] = [1e-5, 1e-4, 1e-3]


def run(cmd: List[str]) -> None:
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def make_outdir(
    base_prefix: Path, *, config_id: int, seed: int, epochs: int, test_shards: str, realtest_shards: str
) -> Path:
    name = f"{base_prefix.name}_c{config_id}_s{seed}_e{epochs}_t{test_shards}_r{realtest_shards}"
    return base_prefix.parent / name


# ✅ OLD filename convention: include _c{config_id}_ in the checkpoint name
def ckpt_path_for(outdir: Path, model_type: str, config_id: int, cfg: Dict[str, Any], lr: float) -> Path:
    lr_tag = f"lr{lr:.0e}".replace("+", "")
    ckpt_name = f"{model_type}_c{config_id}_{cfg['tag']}_{lr_tag}.pt"
    return outdir / ckpt_name


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RYM grid over models × config_id × loss cfg × lr.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=64)

    ap.add_argument(
        "--config-ids",
        type=str,
        default="0,1,2,3",
        help="Comma-separated config ids to sweep (default: '0,1,2,3').",
    )

    ap.add_argument("--train-prefix", type=str, default=str(Path("data") / "rym_2025_jan_apr_tc300+0_bin200_train_shard"))
    ap.add_argument("--val-prefix", type=str, default=str(Path("data") / "rym_2025_jan_apr_tc300+0_bin200_val_shard"))
    ap.add_argument("--test-prefix", type=str, default=str(Path("data") / "rym_2025_jan_apr_tc300+0_bin200_test_shard"))
    ap.add_argument("--realtest-prefix", type=str, default=str(Path("data") / "rym_2025_jan_apr_tc300+0_unbalanced_realtest_shard"))

    ap.add_argument("--train-shards", type=str, default="all")
    ap.add_argument("--val-shards", type=str, default="all")
    ap.add_argument("--test-shards", type=str, default="4")
    ap.add_argument("--realtest-shards", type=str, default="4")

    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print planned runs + skip counts, then exit.")
    ap.add_argument("--outdir-prefix", type=str, default=str(Path("models") / "rym_grid"))

    args = ap.parse_args()
    config_ids = parse_int_list(args.config_ids)

    base_prefix = Path(args.outdir_prefix)
    base_prefix.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pre-flight summary
    # ------------------------------------------------------------------
    per_config: List[Tuple[int, int, int, int]] = []  # (c, total, existing, will_run)
    grand_total = grand_existing = grand_will_run = 0

    jobs_per_config = len(MODEL_TYPES) * len(LOSS_CONFIGS) * len(LRS)

    for c in config_ids:
        outdir = make_outdir(
            base_prefix,
            config_id=c,
            seed=args.seed,
            epochs=args.epochs,
            test_shards=args.test_shards,
            realtest_shards=args.realtest_shards,
        )

        planned_paths: List[Path] = []
        for model_type in MODEL_TYPES:
            for cfg in LOSS_CONFIGS:
                for lr in LRS:
                    planned_paths.append(ckpt_path_for(outdir, model_type, c, cfg, lr))

        total = len(planned_paths)
        existing = sum(1 for p in planned_paths if p.exists())
        will_run = (total - existing) if args.skip_existing else total

        per_config.append((c, total, existing, will_run))
        grand_total += total
        grand_existing += existing
        grand_will_run += will_run

    print("============================================================")
    print("[grid] Summary")
    print(f"  outdir_prefix: {base_prefix}  (per-config dirs will be created under {base_prefix.parent})")
    print(f"  config_ids:    {config_ids}")
    print(f"  models:        {MODEL_TYPES}")
    print(f"  loss_cfgs:     {[c['tag'] for c in LOSS_CONFIGS]}")
    print(f"  lrs:           {LRS}")
    print(f"  epochs:        {args.epochs}")
    print(f"  seed:          {args.seed}")
    print(f"  test_shards:   {args.test_shards}")
    print(f"  realtest:      {args.realtest_shards}")
    print(f"  jobs/config:   {jobs_per_config} (= {len(MODEL_TYPES)} models × {len(LOSS_CONFIGS)} loss × {len(LRS)} lrs)")
    print("------------------------------------------------------------")
    for c, total, existing, will_run in per_config:
        outdir = make_outdir(
            base_prefix,
            config_id=c,
            seed=args.seed,
            epochs=args.epochs,
            test_shards=args.test_shards,
            realtest_shards=args.realtest_shards,
        )
        print(f"  c{c}: total={total:3d} existing={existing:3d} will_run={will_run:3d} -> {outdir}")
    print("------------------------------------------------------------")
    print(
        f"  GRAND: total={grand_total} existing={grand_existing} will_run={grand_will_run} "
        f"(skip_existing={args.skip_existing})"
    )
    print("============================================================")

    if args.dry_run:
        return

    # ------------------------------------------------------------------
    # Main sweep
    # ------------------------------------------------------------------
    for c in config_ids:
        outdir = make_outdir(
            base_prefix,
            config_id=c,
            seed=args.seed,
            epochs=args.epochs,
            test_shards=args.test_shards,
            realtest_shards=args.realtest_shards,
        )
        outdir.mkdir(parents=True, exist_ok=True)

        for model_type in MODEL_TYPES:
            for cfg in LOSS_CONFIGS:
                for lr in LRS:
                    ckpt_path = ckpt_path_for(outdir, model_type, c, cfg, lr)

                    if args.skip_existing and ckpt_path.exists():
                        print(f"[skip] {ckpt_path}")
                        continue

                    cmd = [
                        sys.executable, "-m", "scripts.run_rym_experiments",
                        "--model-type", model_type,
                        "--config-id", str(c),

                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--lr", str(lr),
                        "--device", args.device,
                        "--num-workers", str(args.num_workers),
                        "--seed", str(args.seed),

                        "--alpha-reg", str(cfg["alpha_reg"]),
                        "--lambda-ent", str(cfg["lambda_ent"]),
                        "--gaussian-sigma", str(cfg["gaussian_sigma"]),

                        "--train-prefix", args.train_prefix,
                        "--val-prefix", args.val_prefix,
                        "--test-prefix", args.test_prefix,
                        "--realtest-prefix", args.realtest_prefix,

                        "--train-shards", args.train_shards,
                        "--val-shards", args.val_shards,
                        "--test-shards", args.test_shards,
                        "--realtest-shards", args.realtest_shards,

                        "--ckpt-path", str(ckpt_path),
                    ]
                    run(cmd)


if __name__ == "__main__":
    main()
