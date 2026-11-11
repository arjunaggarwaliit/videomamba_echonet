"""Generate the Kaggle notebook (run once; the .ipynb is then a normal repo file)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "kaggle_videomamba_echonet.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in src.strip("\n").split("\n")]}


cells = [
    md("# VideoMamba → EchoNet-Dynamic: full fine-tuning vs. FG-LoRA",
       "",
       "End-to-end Kaggle notebook. **Settings → Accelerator: GPU T4 x2**, "
       "**Internet: On**, attach the EchoNet-Dynamic dataset and the VideoMamba weights.",
       "",
       "1. install the selective-scan kernels (20–40 min compile, once per session)",
       "2. attach the dataset + pretrained checkpoint",
       "3. pre-decode the clips",
       "4. train full fine-tuning, FG-LoRA and head-only",
       "5. aggregate + plot"),

    md("## 1. Environment",
       "",
       "`mamba-ssm` and `causal-conv1d` are **sdist-only** on PyPI, so they compile for "
       "15-40 min, once per session. They are *optional*: the pipeline falls back to a "
       "pure-PyTorch selective scan automatically (~3x slower, identical math), so a failed "
       "build costs speed, not the run.",
       "",
       "**We do not need VideoMamba's vendored `mamba` fork.** This code only uses "
       "`selective_scan_fn` (present in upstream) and implements the bidirectional scan "
       "itself. That sidesteps two known traps: the fork is the only source of "
       "`bimamba`/`mamba_inner_fn_no_out_proj` (upstream is incompatible with it), and the "
       "fork's `setup.py` hard-codes `-gencode` for sm_70/80/90 with no sm_75, so it fails "
       "on a T4 even with `TORCH_CUDA_ARCH_LIST` set."),

    code(r'''
import os, sys, subprocess, torch
print(torch.__version__, "|", torch.cuda.get_device_name(0), "| cuda", torch.version.cuda)

# T4 = 7.5 | P100 = 6.0 | A100 = 8.0 | L4/L40S = 8.9 | A10G = 8.6
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.call(cmd, shell=True)

ok = sh("pip install -q causal-conv1d") == 0
ok = ok and sh("pip install -q mamba-ssm") == 0
if not ok:
    print("\nupstream build failed -> trying the vendored fork with a patched arch list")
    sh("git clone -q --depth 1 https://github.com/OpenGVLab/VideoMamba.git /kaggle/working/VideoMamba")
    sh("cd /kaggle/working/VideoMamba && "
       "sed -i \"s/-gencode=[^,]*,[^']*'//g\" causal-conv1d/setup.py mamba/setup.py")
    sh("pip install -q /kaggle/working/VideoMamba/causal-conv1d")
    sh("pip install -q /kaggle/working/VideoMamba/mamba")

try:
    import causal_conv1d, mamba_ssm  # noqa: F401
    print("kernels installed:", mamba_ssm.__version__)
except Exception as e:
    print("kernels NOT available - the pure-PyTorch scan will be used (about 3x slower):", e)
'''),

    md("## 2. Code + data",
       "",
       "Upload the `videomamba_echonet` folder as a Kaggle dataset (or `git clone` your "
       "repo). Expected layout: `ecomamba/`, `scripts/`, `tests/`, and "
       "`ckpt/videomamba_s16_k400_f16_res224.pth`."),

    code(r'''
PROJECT = "/kaggle/input/videomamba-echonet/videomamba_echonet"   # <- adjust
WORK    = "/kaggle/working"
os.makedirs(f"{WORK}/ckpt", exist_ok=True)

# pretrained weights (HuggingFace: OpenGVLab/VideoMamba)
if not os.path.exists(f"{WORK}/ckpt/videomamba_s16_k400_f16_res224.pth"):
    sh(f"curl -sL -o {WORK}/ckpt/videomamba_s16_k400_f16_res224.pth "
       "https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_s16_k400_f16_res224.pth")

sh(f"cp -r {PROJECT} {WORK}/src")          # writable copy
os.chdir(f"{WORK}/src")
sys.path.insert(0, os.getcwd())

# the code reports which kernels it found - this is the single most useful check
import ecomamba.mamba as mm
print("CUDA selective scan :", mm.HAS_CUDA_SCAN)
print("fused bimamba kernel:", mm.HAS_FUSED_INNER, "(only needed for --mamba-fast-path on)")
print("causal_conv1d       :", mm.HAS_CAUSAL_CONV1D)

# correctness: scan numerics, checkpoint load, 10 LoRA placements, merge exactness (~1 min)
sh("python tests/test_core.py")
'''),

    code(r'''
os.environ["ECHONET_ROOT"] = "/kaggle/input/echonet-dynamic/EchoNet-Dynamic"  # <- adjust
from ecomamba.data import resolve_root, dataset_stats
root = resolve_root(os.environ["ECHONET_ROOT"])
print(root); print(dataset_stats(root))

# pre-decode 10 031 AVIs x 32 frames @112  -> 4.0 GB memmap (once, ~4 min)
sh("python scripts/prepare_data.py --out cache/clips.npy --frames 32")
'''),

    md("## 3. Train",
       "",
       "`--mamba-fast-path off` is the default and is *required* for LoRA (the fused "
       "kernel bypasses the projection modules). Keep every other flag identical between "
       "the two conditions — the scripts already do."),

    code(r'''
COMMON = ("--ckpt /kaggle/working/ckpt/videomamba_s16_k400_f16_res224.pth "
          "--cache cache/clips.npy --num-frames 16 --img-size 224 "
          "--batch-size 8 --num-workers 4 --epochs 20 --precision fp16")

sh(f"python scripts/train.py --mode full --tag full_ft {COMMON}")
sh(f"python scripts/train.py --mode head --tag head_only {COMMON}")
sh(f"python scripts/train.py --mode lora --tag fglora_r8 {COMMON} "
   "--lora-r 8 --lora-targets in_proj,out_proj,dt_proj,a_log")
'''),

    md("## 4. Required sweeps",
       "",
       "Rank study (r = 1…32), placement study, head-only, reduced data (10/25/50 %) and "
       "the FG-LoRA ablations. Budget ~8 epochs each; drop to `--epochs-sweep 4` if the "
       "30 h/week GPU quota is tight."),

    code(r'''
sh("python scripts/studies.py --suite rank      --epochs-sweep 8 "
   "--ckpt /kaggle/working/ckpt/videomamba_s16_k400_f16_res224.pth --cache cache/clips.npy")
sh("python scripts/studies.py --suite placement --epochs-sweep 8 "
   "--ckpt /kaggle/working/ckpt/videomamba_s16_k400_f16_res224.pth --cache cache/clips.npy")
sh("python scripts/studies.py --suite data      --epochs-sweep 8 "
   "--ckpt /kaggle/working/ckpt/videomamba_s16_k400_f16_res224.pth --cache cache/clips.npy")
sh("python scripts/studies.py --suite ablation  --epochs-sweep 8 "
   "--ckpt /kaggle/working/ckpt/videomamba_s16_k400_f16_res224.pth --cache cache/clips.npy")
'''),

    md("## 5. Analyse", "",
       "Writes the single comparison table (all Section 1.7 metrics), the "
       "MAE-vs-trainable-parameters efficiency plot, Bland–Altman, prediction scatter, "
       "training curves and the learned FG-LoRA gate profile."),

    code(r'''
sh("python scripts/analyze.py --results results --out analysis")
from IPython.display import Markdown, Image, display
display(Markdown(open("analysis/comparison_table.md").read()))
for f in ("efficiency.png", "bland_altman.png", "gates.png", "training_curves.png"):
    p = f"analysis/{f}"
    if os.path.exists(p):
        display(Image(p))
'''),

    md("## 6. Persisting results",
       "",
       "`/kaggle/working` is capped at 20 GB and is only kept when you use "
       "*Save Version → Save & Run All*. Checkpoints are written atomically "
       "(`last.pt`, `best.pt`) so a killed session never leaves a corrupt file; to resume, "
       "attach the previous run's output as an input dataset and pass `--resume`."),

    code(r'''
sh("du -sh results cache analysis | sort -h")
sh("zip -qr /kaggle/working/results.zip results analysis")
print("download results.zip from the Output pane")
'''),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
