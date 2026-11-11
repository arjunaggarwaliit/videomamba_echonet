# VideoMamba → EchoNet-Dynamic: full fine-tuning vs. FG-LoRA

Parameter-efficient cardiac AI: predict the left-ventricular **ejection fraction (EF)**
from apical four-chamber echocardiography videos of **EchoNet-Dynamic** by adapting a
pretrained **VideoMamba** backbone, in two conditions that differ *only* in which
parameters are updated:

| condition | what is trainable | role |
|---|---|---|
| `--mode full` | the whole pretrained backbone + the regression head | baseline |
| `--mode lora` | frozen backbone + **FG-LoRA** adapters + the regression head | proposed |
| `--mode head` | frozen backbone + the regression head only | control |

**FG-LoRA** (Forget-Gate LoRA) is this project's contribution: instead of adapting only
the Transformer-style projections, it adapts **both factors of the Mamba forget gate**

  Abar_t = exp(Δ_t ⊙ A),  Δ from `x_proj`→`dt_proj`,  A = `A_log` (a parameter matrix)

with low-rank matrices on `dt_proj` (Δ-LoRA) **and directly on `A_log`** (A-LoRA), the
latter being provably unreachable by any projection adapter (SDT, ICML 2025, Lemma 1).
See `docs/DESIGN.md` and the PDF report for the full argument and literature review.

---

## 1. Quick start

```bash
pip install -r requirements.txt          # torch, numpy, opencv-python, matplotlib
export ECHONET_ROOT=/path/to/EchoNet-Dynamic    # dir containing FileList.csv + Videos/

# 0. sanity check of the whole stack (CPU ok, ~1 min)
python tests/test_core.py

# 1. (optional but recommended) pre-decode the 10k AVIs into one memmap
python scripts/prepare_data.py --out cache/echonet_f32_112.npy --frames 32   # 4 GB

# 2. full fine-tuning
python scripts/train.py --mode full --tag full_ft --epochs 20 --batch-size 8 \
       --ckpt ckpt/videomamba_s16_k400_f16_res224.pth --cache cache/echonet_f32_112.npy

# 3. the proposed model
python scripts/train.py --mode lora --tag fglora_r8 --epochs 20 --batch-size 8 \
       --ckpt ckpt/videomamba_s16_k400_f16_res224.pth --cache cache/echonet_f32_112.npy \
       --lora-r 8 --lora-targets in_proj,out_proj,dt_proj,a_log

# 4. head-only control
python scripts/train.py --mode head --tag head_only --epochs 20 --batch-size 8

# 5. all required sweeps, then aggregate
python scripts/studies.py --suite all --epochs-main 20 --epochs-sweep 8
python scripts/analyze.py --results results --out analysis
```

`scripts/train.py` writes `results/<tag>/{metrics.json, best.pt, last.pt, adapters.pt,
full_model.pt, merged_model.pt}`. `metrics.json` contains everything the assignment asks
for (MAE/RMSE/R², trainable params + ratio, s/epoch + total, peak GPU memory, ms/clip
after merging, storage bytes, Bland–Altman quantities, per-epoch history, raw test
predictions).

---

## 2. Directory structure

```
videomamba_echonet/
├── README.md                     this file
├── requirements.txt              minimal deps (torch, numpy, opencv, matplotlib)
├── ecomamba/                     the library
│   ├── __init__.py               public API re-exports
│   ├── mamba.py                  Mamba block (bidirectional), checkpoint-compatible,
│   │                             pure-PyTorch selective-scan fallback + fused CUDA path
│   ├── videomamba.py             VideoMamba encoder + EF regression head,
│   │                             architecture inference and checkpoint loading
│   │                             (pos-embed / temporal-pos / 2D→3D inflation)
│   ├── lora.py                   LoRALinear, LoRAConv1d, DtLoRA, ALogLoRA,
│   │                             LoRAConfig, apply_lora, merge/unmerge, gate_report
│   ├── data.py                   EchoNet splits, clip sampling, augmentation,
│   │                             grayscale→RGB policy, memmap cache reader
│   ├── metrics.py                MAE/RMSE/R²/Pearson/Spearman/Bland–Altman/clinical
│   ├── engine.py                 Trainer, AMP, AdamW + layer-wise decay, schedules,
│   │                             peak-memory and latency benchmarks, atomic ckpts
│   └── ...
├── scripts/
│   ├── prepare_data.py           AVI → uint8 memmap + metadata JSON
│   ├── train.py                  one entry point for all three conditions
│   ├── studies.py                rank / placement / reduced-data / ablation sweeps
│   └── analyze.py                aggregates results/ into tables and figures
├── tests/
│   └── test_core.py              scan numerics, checkpoint load, every LoRA placement,
│                                 merge-exactness (0.0 max |Δ| after merging)
├── notebooks/
│   └── kaggle_videomamba_echonet.ipynb   end-to-end Kaggle notebook (install → train → analyse)
├── ckpt/                         pretrained VideoMamba weights (not in git)
├── results/                      per-run metrics + checkpoints
└── cache/                        optional pre-decoded clips
```

---

## 3. Design decisions

**Input.** EchoNet clips are 112×112, ~50 fps, ~175 frames, one greyscale speckle
channel. We sample `T` frames uniformly over the whole record (so several beats are
covered), replicate the single channel to 3 channels (a grey clip `v` then meets the
pretrained RGB stem as `W·[v,v,v]ᵀ`, i.e. the sum of the pretrained RGB filters — the
standard way to reuse an RGB stem) and normalise with ImageNet statistics. Clips are
resized to 224 so the pretrained 14×14 patch grid and `pos_embed` are untouched.

**Head.** `Dropout(0.1) → Linear(d_model, 1)`, bias initialised to the TRAIN EF mean
(55.78). Trained in *both* conditions and counted in both.

**Loss.** Smooth-L1 (β = 1 EF point): the EF distribution has a long low tail, and an
L1-type loss optimises the reported MAE more directly than MSE.

**Why the unfused Mamba path.** The released VideoMamba fork fuses the whole block into
`mamba_inner_fn_no_out_proj`, which consumes **raw weights** — any LoRA adapter on
`in_proj`/`out_proj`/`x_proj`/`dt_proj` would be silently bypassed. `ecomamba.mamba`
therefore implements the block with the same parameter names and *algebraically
identical* math but calling the projection modules, so adapters take effect. The fused
kernel remains available for the full-fine-tuning baseline (`--mamba-fast-path on`) but
is off by default so both conditions execute the same kernels.

**LoRA learning rate.** LoRA adapters use `lr = 1e-3` with `weight_decay = 0`; full
fine-tuning uses `lr = 1e-4` on the backbone with layer-wise decay 0.75 and
`weight_decay = 0.05`, `lr_head = 1e-3`. Both use AdamW, cosine schedule, 1 epoch warmup,
gradient clipping at 1.0, identical augmentation and seed.

---

## 4. The three conditions and the innovation

`ecomamba/lora.py` implements four adapter types:

| adapter | target | params per layer (VideoMamba-S) |
|---|---|---|
| `LoRALinear` | `in_proj`, `out_proj`, `x_proj` (optionally only the X or Z slice) | `r·(d_in+d_out)` |
| `DtLoRA` | `dt_proj` (both directions, shared basis) | `r_Δ·(dt_rank + 2·d_inner)` |
| `ALogLoRA` | `A_log` (and `A_b_log`), rank-`r_a` on a `d_inner×d_state` matrix | `r_a·(d_state + 2·d_inner)` |
| `LoRAConv1d` | depth-wise `conv1d` as a channel-coupled rank-`r` kernel | `r·(d_inner + k)` |

plus a learnable scalar `gate` per adapter (init 0, through `tanh`) reported by
`gate_report()` as a "how much did this layer re-time itself" diagnostic.

Default proposed configuration: `--lora-r 8 --lora-targets in_proj,out_proj,dt_proj,a_log`
→ **3.6 %** of the parameters of full fine-tuning, with the same head and the same clips.

---

## 5. Environment notes

* **CPU**: everything runs (a pure-PyTorch chunked selective scan replaces the CUDA
  kernel), but a VideoMamba-S forward at 224/16 frames takes seconds, so CPU is for
  correctness tests only. The fallback scan is wrapped in **gradient checkpointing**: it
  otherwise keeps one `(b, d, C, C, n)` decay matrix per chunk alive in the autograd graph,
  which measured 3.7 GB of host RAM for a single forward pass of a 7 M model. Recomputation
  costs ~1/3 extra time on the scan and removes the problem.
* **GPU**: install `mamba-ssm` and `causal-conv1d` for the fast CUDA scan. Both are
  sdist-only and compile for ~20–40 min; the pipeline falls back to the PyTorch scan
  automatically if the import fails. `--precision fp16` (T4) or `bf16` (A100/L4).
* **Kaggle**: see [`KAGGLE_STEPS.md`](KAGGLE_STEPS.md) for the exact click-by-click
  procedure, and `notebooks/kaggle_videomamba_echonet.ipynb` for the runnable version.
