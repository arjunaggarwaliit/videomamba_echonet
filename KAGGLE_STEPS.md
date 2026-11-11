# What to do on Kaggle — step by step

Everything in this repository runs on CPU (that is how it was validated), but the **actual
experiment needs a GPU**. This file is the complete list of things only you can do, in order,
with the exact clicks and the exact commands.

Total wall-clock: **~30 min setup + ~7 GPU-hours** of training, spread over 2–3 sessions.

---

## 0. What you need to have ready

| item | where | size |
|---|---|---|
| EchoNet-Dynamic | Kaggle dataset (search "echonet-dynamic") or upload `EchoNet-Dynamic.zip` | 7.9 GB |
| this repository, zipped | upload as a private Kaggle **Dataset** | ~1 MB (without `ckpt/`) |
| `videomamba_s16_k400_f16_res224.pth` | `https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_s16_k400_f16_res224.pth` | 102 MB |

Do **not** upload `ckpt/`, `cache/`, `results/`, `analysis/` — the notebook downloads the
checkpoint and rebuilds the cache itself. A 1 MB zip uploads in seconds; a 600 MB one does not.

```bash
# from the folder that contains videomamba_echonet/
cd videomamba_echonet
zip -r ../videomamba_echonet_code.zip . -x "ckpt/*" "cache/*" "results/*" "analysis/*" "*.pyc" "__pycache__/*"
```

---

## 1. Create the two Kaggle inputs

1. **Code dataset** — kaggle.com → *Datasets* → *New Dataset* → upload
   `videomamba_echonet_code.zip` → name it `videomamba-echonet` → visibility *Private*.
   Kaggle unzips it automatically, so the mounted path becomes
   `/kaggle/input/videomamba-echonet/videomamba_echonet/...`.
   **Check the real path** with `!ls /kaggle/input/videomamba-echonet` once the notebook is
   open — if you uploaded the folder rather than the zip, the path gains or loses a level.

2. **Data** — kaggle.com → *Datasets* → search **`echonet-dynamic`** → *Add to notebook*.
   The mount is usually `/kaggle/input/echonet-dynamic/EchoNet-Dynamic`; confirm you can see
   `FileList.csv` and `Videos/` there. If your copy has a different layout, set
   `ECHONET_ROOT` to whatever directory contains `FileList.csv` — `resolve_root()` also
   searches one level down automatically.

---

## 2. Notebook settings (these four matter)

| setting | value | why |
|---|---|---|
| Accelerator | **GPU T4 ×2** | the only general-purpose GPU Kaggle now offers |
| Internet | **On** | needed for `pip`, HuggingFace, and the kernel build. Requires phone verification on your account; the toggle does not appear until you do it |
| Persistence | *Files only* | keeps `/kaggle/working` smaller than persisting variables |
| Language | Python | — |

Then *File → Import notebook* → upload
`notebooks/kaggle_videomamba_echonet.ipynb`.

---

## 3. Run the notebook, in this order

| cell | what it does | expected time |
|---|---|---|
| 1. Environment | compiles `causal-conv1d` + `mamba-ssm` | **15–40 min** |
| 2. Code + data | copies the repo to `/kaggle/working/src`, downloads the checkpoint, runs the test suite | ~2 min |
| 3. Data | pre-decodes 10 031 AVIs into a 4 GB memmap | ~5 min |
| 4. Train | three runs | hours |
| 5. Sweeps | 36 runs | hours |
| 6. Analyse | builds the table and the figures | ~1 min |

**After cell 1, look at these three lines and send them to me if anything is `False`:**

```
CUDA selective scan : True
fused bimamba kernel: False      <- expected, and fine
causal_conv1d       : True
```

`False` on the first line is not fatal: the pipeline falls back to a pure-PyTorch scan that is
about 3× slower but numerically identical. If you see that, either re-run the install cell or
accept the slowdown — the run still produces correct results.

### Before committing to 20 epochs, calibrate

The timing estimates in the report are computed from FLOP counts, not measured. Spend 6
minutes to replace them with facts:

```bash
python scripts/train.py --mode lora --tag calib --epochs 1 --limit-items 256 \
    --ckpt /kaggle/working/ckpt/videomamba_s16_k400_f16_res224.pth \
    --cache cache/clips.npy --batch-size 8 --precision fp16
```

Read the `time` and `peak GPU` printed in the summary. If the epoch time is far from the
4–7 min predicted, adjust `--epochs` so a run fits inside the 12 h session limit; if peak
memory is close to 16 GB, drop `--batch-size` to 4.

### Then run the two headline experiments

```bash
COMMON="--ckpt /kaggle/working/ckpt/videomamba_s16_k400_f16_res224.pth \
        --cache cache/clips.npy --num-frames 16 --img-size 224 \
        --batch-size 8 --num-workers 4 --epochs 20 --precision fp16"

python scripts/train.py --mode full --tag full_ft   $COMMON
python scripts/train.py --mode lora --tag fglora_r8 $COMMON \
       --lora-r 8 --lora-targets in_proj,out_proj,dt_proj,a_log
```

Do these **first**, before the sweeps. If the GPU quota or the session limit bites, these two
are the deliverable; the sweeps are the "additional evaluation".

---

## 4. Save the results

*File → Save Version → Save & Run All (Commit)* keeps `/kaggle/working` (capped at 20 GB) as
the notebook's output. Before that, make the download small:

```bash
zip -qr results.zip results analysis
```

and download `results.zip` from the *Output* pane.

**Then send me `results.zip`** (or just `results/*/metrics.json`). Section 6 of the report is
written against exactly those files: I can drop the real numbers, the comparison table, the
efficiency plot and the Bland–Altman figure straight in without re-deriving anything.

---

## 5. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `no kernel image is available` | compiled without sm_75, or `TORCH_CUDA_ARCH_LIST` ignored | the notebook sets it to `7.5` before `pip`; if you installed by hand, `export TORCH_CUDA_ARCH_LIST=7.5` first. Or just let it fall back to the PyTorch scan |
| `unexpected keyword argument 'bimamba'` | you installed the **vendored** `mamba/` fork of the repo instead of upstream | not needed here: `pip install mamba-ssm` (upstream) is correct for this codebase |
| `ModuleNotFoundError: mamba_ssm` | build failed | harmless — the fallback scan takes over. Training is slower, results identical |
| CUDA OOM | batch 8 at 224²/16 frames is near the T4 limit | `--batch-size 4`, or `--img-size 112` (4× fewer tokens) |
| session dies at 12 h | Kaggle limit | `last.pt` is written atomically after every epoch; attach the previous output as an input dataset and use `--resume /kaggle/input/<prev>/results/<tag>/last.pt` |
| `could not locate EchoNet-Dynamic` | wrong mount path | pass `--data-root` explicitly, or check that `FileList.csv` is inside the directory you named |
| quota exhausted | 30 h/week, reset weekly | the two headline runs alone are ~2 h; they fit comfortably |

---

## 6. Optional: a second seed

The report's "Threats to validity" section says a single-seed difference of 0.1 MAE is not
evidence. If the quota allows, re-run the two headline experiments with `--seed 43`
(tag them `full_ft_s43`, `fglora_r8_s43`); `analyze.py` picks them up automatically and the
table gains a mean ± sd column.

---

## What I need back from you

1. The three `HAS_*` lines from cell 2.
2. The calibration run's epoch time and peak GPU memory.
3. `results.zip` after the two headline runs.

With (2) I can correct the timing table in §6.2 of the report; with (3) I can write §6 with
real EF numbers and regenerate every figure.
