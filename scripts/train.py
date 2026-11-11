#!/usr/bin/env python
"""Train / evaluate VideoMamba on EchoNet-Dynamic EF regression.

One script, three conditions (the *only* difference is which parameters are updated):

    --mode full   full fine-tuning of the pretrained backbone + regression head
    --mode lora   frozen backbone + FG-LoRA adapters + regression head   (proposed)
    --mode head   frozen backbone + regression head only                 (control)

Examples
--------
# full fine-tuning
python scripts/train.py --mode full --tag full_s16 --epochs 20 --batch-size 8

# proposed: FG-LoRA
python scripts/train.py --mode lora --tag fglora_r8 --epochs 20 --batch-size 8 \
        --lora-r 8 --lora-targets in_proj,out_proj,dt_proj,a_log

# rank / placement / head-only studies
python scripts/train.py --mode lora --tag fglora_r4 --lora-r 4 ...
python scripts/train.py --mode head --tag head_only ...

# evaluate a finished run on the official test split
python scripts/train.py --mode lora --eval-only --resume results/fglora_r8/best.pt ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecomamba.data import build_loaders, resolve_root, dataset_stats          # noqa: E402
from ecomamba.engine import (                                                  # noqa: E402
    Trainer, build_optimizer, build_scheduler, count_trainable, make_criterion,
    peak_gpu_memory, resolve_precision, set_seed, benchmark_inference, file_size,
    human_bytes,
)
from ecomamba.lora import (                                                    # noqa: E402
    LoRAConfig, apply_lora, count_parameters, gate_report, lora_state_dict,
    merge_and_strip,
)
from ecomamba.metrics import compute_metrics, format_metrics                   # noqa: E402
from ecomamba.videomamba import build_videomamba_ef                           # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser("VideoMamba + EchoNet-Dynamic EF regression")

    # data
    g = p.add_argument_group("data")
    g.add_argument("--data-root", default=None, help="EchoNet-Dynamic directory")
    g.add_argument("--num-frames", type=int, default=16)
    g.add_argument("--img-size", type=int, default=224,
                   help="224 keeps the pretrained 14x14 patch grid")
    g.add_argument("--fraction", type=float, default=1.0,
                   help="fraction of the TRAIN split (reduced-data study)")
    g.add_argument("--cache", default=None, help="path to the .npy clip cache")
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--limit-items", type=int, default=0,
                   help="debug: use at most N clips per split")
    g.add_argument("--limit-batches", type=int, default=0,
                   help="debug: at most N batches per epoch / evaluation")

    # model
    g = p.add_argument_group("model")
    g.add_argument("--ckpt", default="ckpt/videomamba_s16_k400_f16_res224.pth")
    g.add_argument("--drop-path", type=float, default=0.1)
    g.add_argument("--head-drop", type=float, default=0.1)
    g.add_argument("--head-bias", default="mean", choices=["mean", "zero"])
    g.add_argument("--mamba-fast-path", default="off", choices=["on", "off"],
                   help="'on' fuses the SSM kernels but bypasses the projection "
                        "modules, so it is only valid without adapters")

    # condition
    g = p.add_argument_group("condition")
    g.add_argument("--mode", default="lora", choices=["full", "lora", "head"])
    g.add_argument("--lora-r", type=int, default=8)
    g.add_argument("--lora-alpha", type=float, default=None, help="default 2*r")
    g.add_argument("--lora-r-dt", type=int, default=None, help="Delta adapter rank (r//2)")
    g.add_argument("--lora-r-a", type=int, default=None, help="A adapter rank (r)")
    g.add_argument("--lora-targets", default="in_proj,out_proj,dt_proj,a_log")
    g.add_argument("--lora-layers", default="all")
    g.add_argument("--in-proj-slice", default=None, choices=["x", "z"])
    g.add_argument("--dt-tie", action="store_true", default=True)
    g.add_argument("--no-dt-tie", dest="dt_tie", action="store_false")
    g.add_argument("--a-tie", action="store_true", default=True)
    g.add_argument("--no-a-tie", dest="a_tie", action="store_false")
    g.add_argument("--no-gate", dest="gate", action="store_false", default=True)

    # optimisation
    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--eval-batch-size", type=int, default=0)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--lr-head", type=float, default=None)
    g.add_argument("--min-lr", type=float, default=0.0)
    g.add_argument("--weight-decay", type=float, default=None)
    g.add_argument("--layer-decay", type=float, default=0.75)
    g.add_argument("--warmup-epochs", type=float, default=1.0)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--criterion", default="huber",
                   choices=["huber", "smooth_l1", "l1", "mae", "l2", "mse"])
    g.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])

    # bookkeeping
    g = p.add_argument_group("bookkeeping")
    g.add_argument("--tag", default="run")
    g.add_argument("--out-dir", default="results")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--eval-only", action="store_true")
    g.add_argument("--resume", default=None, help="checkpoint to start from / evaluate")
    g.add_argument("--save-every", type=int, default=0, help="also save every N epochs")
    g.add_argument("--patience", type=int, default=0, help="early stop after N bad epochs")
    return p.parse_args(argv)


def default_lrs(args):
    if args.mode == "full":
        lr = args.lr if args.lr is not None else 1e-4
        lr_head = args.lr_head if args.lr_head is not None else 1e-3
        wd = args.weight_decay if args.weight_decay is not None else 0.05
        layer_decay = args.layer_decay
    elif args.mode == "lora":
        lr = args.lr if args.lr is not None else 1e-3
        lr_head = args.lr_head if args.lr_head is not None else 1e-3
        wd = args.weight_decay if args.weight_decay is not None else 0.0
        layer_decay = 1.0
    else:  # head
        lr = args.lr if args.lr is not None else 1e-3
        lr_head = args.lr_head if args.lr_head is not None else 1e-3
        wd = args.weight_decay if args.weight_decay is not None else 0.0
        layer_decay = 1.0
    return lr, lr_head, wd, layer_decay


def build_condition(model, args, lora_cfg=None):
    """Set ``requires_grad`` according to --mode.  Returns a dict describing the setup."""
    if args.mode == "head":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)
        return {"mode": "head", "lora": None}
    if args.mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return {"mode": "full", "lora": None}
    report = apply_lora(model, lora_cfg, verbose=True)
    return {"mode": "lora", "lora": lora_cfg.as_dict(),
            "lora_by_kind": report.by_kind, "lora_modules": len(report.names)}


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = resolve_precision(args.precision)
    print(f"[setup] device={device} precision={precision} "
          f"torch={torch.__version__} tag={args.tag}")

    # ---------------- data ----------------
    root = resolve_root(args.data_root)
    print(f"[data] root={root}")
    if not args.eval_only:
        print(f"[data] split sizes: {dataset_stats(root)}")

    cache_index = args.cache + ".json" if args.cache else None
    train_loader, val_loader, test_loader, stats = build_loaders(
        root=root, num_frames=args.num_frames, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        fraction=args.fraction, cache=args.cache, cache_index=cache_index,
        seed=args.seed, max_items=args.limit_items or None,
        eval_batch_size=args.eval_batch_size or None,
    )
    print(f"[data] train={stats['train']} val={stats['val']} test={stats['test']} "
          f"EF mean={stats['ef_mean']:.2f} sd={stats['ef_std']:.2f}")
    if args.limit_batches:
        from itertools import islice
        train_loader = list(islice(train_loader, args.limit_batches))
        val_loader = list(islice(val_loader, args.limit_batches))
        test_loader = list(islice(test_loader, args.limit_batches))

    # ---------------- model ----------------
    head_bias = stats["ef_mean"] if args.head_bias == "mean" else 0.0
    model = build_videomamba_ef(
        args.ckpt, num_frames=args.num_frames, img_size=args.img_size,
        drop_path_rate=args.drop_path, head_drop_rate=args.head_drop,
        head_bias=head_bias, use_fast_path=(args.mamba_fast_path == "on"),
    ).to(device)

    lora_cfg = None
    if args.mode == "lora":
        lora_cfg = LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha if args.lora_alpha else 2.0 * args.lora_r,
            r_dt=args.lora_r_dt,
            r_a=args.lora_r_a,
            targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
            in_proj_slice=args.in_proj_slice,
            dt_tie=args.dt_tie, a_tie=args.a_tie, gate=args.gate,
            layers=args.lora_layers,
        )
    cond = build_condition(model, args, lora_cfg)
    total_params, trainable_params = count_trainable(model)
    print(f"[model] total {total_params:,} | trainable {trainable_params:,} "
          f"({100.0 * trainable_params / total_params:.3f}%)")

    lr, lr_head, wd, layer_decay = default_lrs(args)
    print(f"[opt] mode={args.mode} lr={lr:g} lr_head={lr_head:g} wd={wd:g} "
          f"layer_decay={layer_decay:g}")

    # ---------------- optional resume ----------------
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[resume] loaded {args.resume} (missing={len(missing)}, "
              f"unexpected={len(unexpected)})")
        if args.eval_only:
            del ck

    # ---------------- training ----------------
    history = []
    train_seconds = []
    if not args.eval_only:
        criterion = make_criterion(args.criterion)
        optimizer = build_optimizer(model, lr=lr, lr_head=lr_head, weight_decay=wd,
                                    layer_decay=layer_decay, depth=model.depth)
        steps = len(train_loader)
        scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, steps)
        trainer = Trainer(model, criterion, optimizer, scheduler, device=device,
                          precision=precision, grad_clip=args.grad_clip,
                          log_every=args.log_every, out_dir=out_dir, depth=model.depth)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        best_val, best_state, bad = float("inf"), None, 0
        t_start = time.time()
        for epoch in range(start_epoch, args.epochs):
            tr = trainer.train_epoch(train_loader, epoch + 1)
            val_m, _, _ = trainer.evaluate(val_loader)
            train_seconds.append(tr["seconds"])
            row = {"epoch": epoch + 1, "train_loss": tr["loss"], "train_seconds": tr["seconds"],
                   "samples_per_s": tr["samples_per_s"], "skipped": tr["skipped"],
                   **{f"val_{k}": v for k, v in val_m.items()}}
            history.append(row)
            print(f"[epoch {epoch + 1}/{args.epochs}] loss {tr['loss']:.4f} "
                  f"({tr['seconds']:.1f}s) | val {format_metrics(val_m)}", flush=True)

            if val_m["mae"] < best_val - 1e-6:
                best_val, bad = val_m["mae"], 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                trainer.save(os.path.join(out_dir, "best.pt"),
                             extra={"epoch": epoch + 1, "val_mae": best_val,
                                    "args": vars(args), "condition": cond})
            else:
                bad += 1
            trainer.save(os.path.join(out_dir, "last.pt"),
                         extra={"epoch": epoch + 1, "val_mae": val_m["mae"],
                                "args": vars(args), "condition": cond})
            if args.save_every and (epoch + 1) % args.save_every == 0:
                trainer.save(os.path.join(out_dir, f"ep{epoch + 1}.pt"))
            if args.patience and bad >= args.patience:
                print(f"[early-stop] no improvement for {bad} epochs")
                break

        total_train_time = time.time() - t_start
        mem = peak_gpu_memory()
        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"[train] total {total_train_time:.1f}s | best val MAE {best_val:.3f} | "
              f"peak GPU {mem.get('peak_allocated_mb', 0):.0f} MB")

    # ---------------- parameter / memory / storage accounting ----------------
    total_params, trainable_params = count_trainable(model)
    peak_train_mem = peak_gpu_memory()

    # ---------------- test evaluation ----------------
    trainer_eval = Trainer(model, make_criterion(args.criterion), None, None,
                           device=device, precision=precision,
                           log_every=0, out_dir=out_dir, depth=model.depth)
    test_m, y_true, y_pred = trainer_eval.evaluate(test_loader)
    val_m, _, _ = trainer_eval.evaluate(val_loader)
    print(f"[test]  {format_metrics(test_m)}")
    print(f"[test]  R2={test_m['r2']:.4f} RMSE={test_m['rmse']:.3f} "
          f"bias={test_m['bias']:+.3f} LoA=[{test_m['loa_lower']:.2f},"
          f"{test_m['loa_upper']:.2f}] within5={100 * test_m['agree_5']:.1f}% "
          f"mae_lowEF={test_m['mae_low_ef']:.3f}")

    # ---------------- storage (adapters must be written before merging) ----------------
    sizes = {}
    if args.mode == "lora":
        adapter_path = os.path.join(out_dir, "adapters.pt")
        torch.save(lora_state_dict(model), adapter_path)
        sizes["adapter_file_bytes"] = file_size(adapter_path)
    full_path = os.path.join(out_dir, "full_model.pt")
    torch.save(model.state_dict(), full_path)
    sizes["full_state_bytes"] = file_size(full_path)
    if os.path.isfile(os.path.join(out_dir, "best.pt")):
        sizes["trainable_checkpoint_bytes"] = file_size(os.path.join(out_dir, "best.pt"))

    # ---------------- inference benchmark (merged vs original) ----------------
    inf_before = benchmark_inference(model, test_loader, device, precision,
                                     batch_size=args.batch_size)
    baseline = build_videomamba_ef(
        args.ckpt, num_frames=args.num_frames, img_size=args.img_size,
        drop_path_rate=0.0, head_drop_rate=0.0, head_bias=head_bias,
        use_fast_path=(args.mamba_fast_path == "on"),
    ).to(device)
    inf_baseline = benchmark_inference(baseline, test_loader, device, precision,
                                       batch_size=args.batch_size)
    del baseline
    if args.mode == "lora":
        merged = merge_and_strip(model)
        inf_merged = benchmark_inference(merged, test_loader, device, precision,
                                         batch_size=args.batch_size)
        # merging must not change the predictions
        merged_m, _, _ = trainer_eval.evaluate(test_loader)
        merge_delta = abs(merged_m["mae"] - test_m["mae"])
        print(f"[merge] MAE before {test_m['mae']:.4f} after {merged_m['mae']:.4f} "
              f"(|d|={merge_delta:.2e})")
        merged_model_path = os.path.join(out_dir, "merged_model.pt")
        torch.save(merged.state_dict(), merged_model_path)
        sizes["merged_state_bytes"] = file_size(merged_model_path)
    else:
        inf_merged = dict(inf_before)
        merged_m, merge_delta = test_m, 0.0

    # ---------------- write results ----------------
    result = {
        "tag": args.tag,
        "mode": args.mode,
        "args": vars(args),
        "data": {"root": root, **{k: v for k, v in stats.items()}},
        "params": {"total": total_params, "trainable": trainable_params,
                   "trainable_ratio_pct": 100.0 * trainable_params / total_params},
        "timing": {
            "total_train_seconds": (sum(train_seconds) if train_seconds else 0.0),
            "mean_epoch_seconds": (float(np.mean(train_seconds)) if train_seconds else 0.0),
            "epochs_run": len(train_seconds),
            "samples_per_s": (float(np.mean([h["samples_per_s"] for h in history]))
                              if history else 0.0),
        },
        "memory": peak_train_mem,
        "test": test_m,
        "val": val_m,
        "test_merged": merged_m,
        "merge_delta_mae": merge_delta,
        "inference": {"unmerged": inf_before, "merged": inf_merged,
                      "pretrained_baseline": inf_baseline},
        "storage": sizes,
        "lora_config": cond.get("lora"),
        "lora_by_kind": cond.get("lora_by_kind"),
        "condition": cond,
        "history": history,
        "predictions": {"y_true": y_true.tolist(), "y_pred": y_pred.tolist()},
    }
    if args.mode == "lora":
        result["gates"] = gate_report(model)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n================ SUMMARY ================")
    print(f"mode            : {args.mode}  ({args.tag})")
    print(f"trainable params: {trainable_params:,} / {total_params:,} "
          f"({100.0 * trainable_params / total_params:.3f}%)")
    if train_seconds:
        print(f"train time      : {np.mean(train_seconds):.1f}s / epoch, "
              f"{sum(train_seconds):.1f}s total ({len(train_seconds)} epochs)")
    print(f"peak GPU        : {peak_train_mem.get('peak_allocated_mb', 0):.0f} MB allocated, "
          f"{peak_train_mem.get('peak_reserved_mb', 0):.0f} MB reserved")
    print(f"inference       : {inf_merged.get('inf_ms_per_clip', 0):.2f} ms/clip merged "
          f"({inf_merged.get('inf_clips_per_s', 0):.1f} clips/s), "
          f"pretrained {inf_baseline.get('inf_ms_per_clip', 0):.2f} ms/clip")
    if sizes:
        print(f"storage         : " + ", ".join(f"{k}={human_bytes(v)}"
                                               for k, v in sizes.items()))
    print(f"test            : {format_metrics(test_m)}")
    print(f"results written to {os.path.join(out_dir, 'metrics.json')}")
    return result


if __name__ == "__main__":
    main()
