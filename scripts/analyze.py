#!/usr/bin/env python
"""Aggregate every ``results/<tag>/metrics.json`` into the tables and figures the
assignment asks for (Section 1.7 + "Additional Evaluation").

Produces, in ``--out``:

    comparison_table.md / .csv   the single main table (both models, every metric)
    efficiency.png               MAE vs number of trainable parameters
    training_curves.png          val MAE per epoch, all runs
    bland_altman.png             Bland-Altman of EF predictions (best full vs best LoRA)
    scatter.png                  predicted vs true EF
    gates.png                    learned FG-LoRA gate values per layer
    rank_study.md                rank sweep table
    placement_study.md           placement sweep table
    data_fraction.md             reduced-data table
    report.json                  everything, machine readable

Usage
-----
python scripts/analyze.py --results results --out analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def load_runs(results_dir: str) -> List[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "metrics.json"))):
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipping {path}: {e}")
            continue
        r["_path"] = path
        runs.append(r)
    return runs


def fmt(v, nd=3, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:.{nd}f}"


def main_table(runs: List[dict]) -> str:
    cols = [
        ("tag", "run", "tag"),
        ("mode", "mode", "tag"),
        ("params.trainable", "trainable", "int"),
        ("params.trainable_ratio_pct", "ratio", "pct1"),
        ("test.mae", "MAE", "3"),
        ("test.rmse", "RMSE", "3"),
        ("test.r2", "R^2", "4"),
        ("test.pearson", "Pearson", "4"),
        ("test.bias", "bias", "3s"),
        ("test.agree_5", "<=5 pt", "pct"),
        ("timing.mean_epoch_seconds", "s/epoch", "1"),
        ("memory.peak_allocated_mb", "peak MB", "0"),
        ("inference.merged.inf_ms_per_clip", "ms/clip", "2"),
    ]

    def get(r, path):
        cur = r
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def render(r, path, spec):
        v = get(r, path)
        if spec == "tag":
            return str(v) if v is not None else "-"
        if v is None:
            return "-"
        if spec == "int":
            return f"{int(v):,}"
        if spec == "pct1":
            return f"{v:.2f}%"
        if spec == "pct":
            return f"{100 * v:.1f}%"
        if spec == "3s":
            return f"{v:+.3f}"
        if spec in ("3", "4", "1", "2", "0"):
            return fmt(v, int(spec))
        return str(v)

    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in sorted(runs, key=lambda x: (x.get("mode", ""), x.get("tag", ""))):
        lines.append("| " + " | ".join(render(r, c[0], c[2]) for c in cols) + " |")
    md = "\n".join(lines)
    return md


def subset_table(runs: List[dict], pred, title: str) -> str:
    sel = [r for r in runs if pred(r)]
    if not sel:
        return f"### {title}\n\n(nothing to report)\n"
    lines = [f"### {title}", "",
             "| run | r | trainable | MAE | RMSE | R2 |", "|---|---|---|---|---|---|"]
    for r in sorted(sel, key=lambda x: x.get("tag", "")):
        lc = r.get("lora_config") or {}
        lines.append("| {t} | {rr} | {p:,} | {m:.3f} | {rm:.3f} | {r2:.4f} |".format(
            t=r.get("tag"), rr=lc.get("r", "-"),
            p=r["params"]["trainable"], m=r["test"]["mae"],
            rm=r["test"]["rmse"], r2=r["test"]["r2"]))
    return "\n".join(lines) + "\n"


def plots(runs: List[dict], out: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    os.makedirs(out, exist_ok=True)

    # ---- efficiency: MAE vs trainable parameters -------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    styles = {"full": ("C3", "o"), "lora": ("C0", "s"), "head": ("C2", "^")}
    for r in runs:
        c, m = styles.get(r.get("mode"), ("gray", "x"))
        ax.scatter(r["params"]["trainable"], r["test"]["mae"], c=c, marker=m, s=46,
                   label=r.get("mode") if r is runs[0] else None, zorder=3)
        ax.annotate(r.get("tag", ""), (r["params"]["trainable"], r["test"]["mae"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    handles = [plt.Line2D([], [], marker=v[1], color=v[0], ls="", label=k)
               for k, v in styles.items()]
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("test MAE (EF points)")
    ax.set_title("Adaptation efficiency")
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    p = os.path.join(out, "efficiency.png")
    fig.savefig(p)
    plt.close(fig)
    made.append(p)

    # ---- training curves -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    for r in runs:
        h = r.get("history") or []
        if not h:
            continue
        ax.plot([e["epoch"] for e in h], [e["val_mae"] for e in h], marker=".",
                lw=1.3, label=r.get("tag"))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MAE")
    ax.set_title("Validation curves")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6)
    fig.tight_layout()
    p = os.path.join(out, "training_curves.png")
    fig.savefig(p)
    plt.close(fig)
    made.append(p)

    # ---- Bland-Altman + scatter (best full vs best LoRA) ------------------------
    def best(mode):
        sel = [r for r in runs if r.get("mode") == mode and r.get("predictions")]
        return min(sel, key=lambda r: r["test"]["mae"]) if sel else None

    pairs = [r for r in (best("full"), best("lora")) if r]
    if pairs:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=160)
        for r in pairs:
            y = np.array(r["predictions"]["y_true"])
            p_ = np.array(r["predictions"]["y_pred"])
            axes[0].scatter(y, p_, s=5, alpha=0.35, label=r["tag"])
        lim = [0, 100]
        axes[0].plot(lim, lim, "k--", lw=1)
        axes[0].set_xlabel("true EF")
        axes[0].set_ylabel("predicted EF")
        axes[0].set_title("Prediction scatter (test split)")
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.25)

        for r in pairs:
            y = np.array(r["predictions"]["y_true"])
            p_ = np.array(r["predictions"]["y_pred"])
            d = p_ - y
            axes[1].scatter((y + p_) / 2, d, s=5, alpha=0.35, label=r["tag"])
            axes[1].axhline(d.mean(), color="k", lw=1)
        axes[1].axhline(0, color="gray", ls=":", lw=0.8)
        axes[1].set_xlabel("mean of true and predicted EF")
        axes[1].set_ylabel("predicted - true")
        axes[1].set_title("Bland-Altman")
        if axes[1].get_legend_handles_labels()[0]:
            axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        p = os.path.join(out, "bland_altman.png")
        fig.savefig(p)
        plt.close(fig)
        made.append(p)

    # ---- learned gate profile --------------------------------------------------
    gated = [r for r in runs if r.get("gates")]
    if gated:
        r = min(gated, key=lambda x: x["test"]["mae"])
        g = r["gates"]
        fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=160)
        layers = [x["layer"] for x in g]
        for key in ("a_gate_f", "a_gate_b", "dt_gate_f", "dt_gate_b"):
            if key in g[0]:
                ax.plot(layers, [x.get(key, np.nan) for x in g], marker=".", label=key)
        ax.set_xlabel("layer")
        ax.set_ylabel("tanh(gate)")
        ax.set_title(f"Learned FG-LoRA gates ({r['tag']})")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(out, "gates.png")
        fig.savefig(p)
        plt.close(fig)
        made.append(p)
    return made


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="analysis")
    args = p.parse_args(argv)

    runs = load_runs(args.results)
    if not runs:
        print(f"no metrics.json under {args.results}")
        return
    os.makedirs(args.out, exist_ok=True)
    print(f"loaded {len(runs)} runs")

    md = ["# VideoMamba on EchoNet-Dynamic - aggregated results", "",
          f"({len(runs)} runs)", "", "## Main comparison", "", main_table(runs), ""]

    md.append(subset_table(runs, lambda r: r.get("args", {}).get("lora_targets") is not None
                           and r.get("mode") == "lora", "LoRA runs"))
    md.append(subset_table(runs, lambda r: r.get("mode") in ("full", "head"),
                           "Baselines"))

    text = "\n".join(md)
    with open(os.path.join(args.out, "comparison_table.md"), "w") as f:
        f.write(text)
    print(text)

    # csv
    import csv
    with open(os.path.join(args.out, "comparison_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "mode", "trainable", "total", "mae", "rmse", "r2", "pearson",
                    "bias", "sd_diff", "agree5", "agree10", "epoch_s", "total_s",
                    "peak_mem_mb", "inf_ms_merged", "adapter_bytes", "full_bytes"])
        for r in runs:
            t = r["timing"]
            w.writerow([
                r.get("tag"), r.get("mode"), r["params"]["trainable"],
                r["params"]["total"], r["test"]["mae"], r["test"]["rmse"],
                r["test"]["r2"], r["test"]["pearson"], r["test"]["bias"],
                r["test"]["sd_diff"], r["test"]["agree_5"], r["test"]["agree_10"],
                t.get("mean_epoch_seconds"), t.get("total_train_seconds"),
                r.get("memory", {}).get("peak_allocated_mb"),
                r.get("inference", {}).get("merged", {}).get("inf_ms_per_clip"),
                r.get("storage", {}).get("adapter_file_bytes"),
                r.get("storage", {}).get("full_state_bytes"),
            ])

    made = plots(runs, args.out)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(runs, f, indent=2)
    print("\nfigures:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
