#!/usr/bin/env python
"""Driver for the additional-evaluation sweeps required by the assignment.

    python scripts/studies.py --suite main        # full FT vs FG-LoRA vs head-only
    python scripts/studies.py --suite rank        # r = 1,2,4,8,16,32
    python scripts/studies.py --suite placement   # where in the block LoRA goes
    python scripts/studies.py --suite data        # 10 / 25 / 50 / 100 % of TRAIN
    python scripts/studies.py --suite ablation    # Delta-LoRA on/off, A-LoRA on/off, ties
    python scripts/studies.py --suite all

Each entry becomes an independent ``scripts/train.py`` process writing
``results/<tag>/metrics.json``; ``scripts/analyze.py`` then aggregates them.

Budget hint (Kaggle T4, VideoMamba-S, 16 frames @ 224): ~5-7 min/epoch for full
fine-tuning and ~4-5 min/epoch for LoRA, so the sweeps use ``--epochs 8`` while the main
comparison uses ``--epochs 20``.  Pass ``--epochs`` to override.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def base_args(a):
    return [
        "--ckpt", a.ckpt, "--num-frames", str(a.num_frames), "--img-size", str(a.img_size),
        "--batch-size", str(a.batch_size), "--num-workers", str(a.num_workers),
        "--out-dir", a.out_dir, "--seed", str(a.seed),
    ] + (["--cache", a.cache] if a.cache else [])


def plan(a) -> list:
    """Return a list of (tag, argv, epochs) tuples."""
    jobs = []
    ep_main = a.epochs_main
    ep_sweep = a.epochs_sweep

    if a.suite in ("main", "all"):
        jobs += [
            ("full_ft", ["--mode", "full"], ep_main),
            ("head_only", ["--mode", "head"], ep_main),
            ("fglora_r8", ["--mode", "lora", "--lora-r", "8",
                           "--lora-targets", "in_proj,out_proj,dt_proj,a_log"], ep_main),
        ]

    if a.suite in ("rank", "all"):
        for r in (1, 2, 4, 8, 16, 32):
            jobs.append((f"rank_r{r}", ["--mode", "lora", "--lora-r", str(r),
                                        "--lora-targets", "in_proj,out_proj,dt_proj,a_log"],
                         ep_sweep))

    if a.suite in ("placement", "all"):
        placements = [
            ("p_inproj", ["--lora-targets", "in_proj"]),
            ("p_outproj", ["--lora-targets", "out_proj"]),
            ("p_inout", ["--lora-targets", "in_proj,out_proj"]),
            ("p_gate_only", ["--lora-targets", "dt_proj"]),
            ("p_ssm_only", ["--lora-targets", "x_proj,dt_proj"]),
            ("p_a_only", ["--lora-targets", "a_log"]),
            ("p_inout_alog", ["--lora-targets", "in_proj,out_proj,a_log"]),
            ("p_inout_dt", ["--lora-targets", "in_proj,out_proj,dt_proj"]),
            ("p_full", ["--lora-targets", "in_proj,out_proj,x_proj,dt_proj,a_log"]),
            ("p_full_conv", ["--lora-targets",
                             "in_proj,out_proj,x_proj,dt_proj,a_log,conv1d"]),
            ("p_x_slice", ["--lora-targets", "in_proj,out_proj", "--in-proj-slice", "x"]),
            ("p_z_slice", ["--lora-targets", "in_proj,out_proj", "--in-proj-slice", "z"]),
        ]
        for tag, extra in placements:
            jobs.append((tag, ["--mode", "lora", "--lora-r", "8", *extra], ep_sweep))
        for spec in ("first8", "last8", "0-11", "12-23"):
            jobs.append((f"p_layers_{spec.replace('-', '_')}",
                         ["--mode", "lora", "--lora-r", "8",
                          "--lora-targets", "in_proj,out_proj,dt_proj,a_log",
                          "--lora-layers", spec], ep_sweep))

    if a.suite in ("data", "all"):
        for frac in (0.1, 0.25, 0.5):
            pct = int(frac * 100)
            jobs.append((f"full_ft_data{pct}", ["--mode", "full",
                                                "--fraction", str(frac)], ep_sweep))
            jobs.append((f"fglora_data{pct}", ["--mode", "lora", "--lora-r", "8",
                                               "--fraction", str(frac)], ep_sweep))

    if a.suite in ("ablation", "all"):
        jobs += [
            ("abl_no_dt", ["--mode", "lora", "--lora-r", "8",
                           "--lora-targets", "in_proj,out_proj,a_log"], ep_sweep),
            ("abl_no_a", ["--mode", "lora", "--lora-r", "8",
                          "--lora-targets", "in_proj,out_proj,dt_proj"], ep_sweep),
            ("abl_no_tie_dt", ["--mode", "lora", "--lora-r", "8", "--no-dt-tie",
                               "--lora-targets", "in_proj,out_proj,dt_proj,a_log"], ep_sweep),
            ("abl_no_tie_a", ["--mode", "lora", "--lora-r", "8", "--no-a-tie",
                              "--lora-targets", "in_proj,out_proj,dt_proj,a_log"], ep_sweep),
            ("abl_no_gate", ["--mode", "lora", "--lora-r", "8", "--no-gate",
                             "--lora-targets", "in_proj,out_proj,dt_proj,a_log"], ep_sweep),
        ]
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="main",
                   choices=["main", "rank", "placement", "data", "ablation", "all"])
    p.add_argument("--ckpt", default="ckpt/videomamba_s16_k400_f16_res224.pth")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--cache", default=None)
    p.add_argument("--epochs-main", type=int, default=20)
    p.add_argument("--epochs-sweep", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args(argv)

    jobs = plan(args)
    print(f"{len(jobs)} runs planned\n")
    t_all = time.time()
    for i, (tag, extra, epochs) in enumerate(jobs, 1):
        out = os.path.join(args.out_dir, tag)
        if args.skip_existing and os.path.isfile(os.path.join(out, "metrics.json")):
            print(f"[{i}/{len(jobs)}] {tag}: exists, skipped")
            continue
        cmd = [sys.executable, os.path.join(HERE, "train.py"), *base_args(args),
               "--tag", tag, "--epochs", str(epochs), *extra]
        print(f"[{i}/{len(jobs)}] {tag}: {' '.join(cmd[1:])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT)
        print(f"      -> exit {rc} in {time.time() - t0:.0f}s", flush=True)
    print(f"\nall done in {(time.time() - t_all) / 60:.1f} min")
    print(f"now run: python scripts/analyze.py --results {args.out_dir}")


if __name__ == "__main__":
    main()
