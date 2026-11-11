"""Training / evaluation engine: one code path used by *both* experimental conditions.

The only thing that differs between full fine-tuning and LoRA is the set of parameters
with ``requires_grad=True`` (plus, as a consequence, the recommendation for the learning
rate).  Everything else - data, augmentation, clips, loss, schedule, seed, head - is
shared, which is what the assignment asks for.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD
from .metrics import compute_metrics, format_metrics


# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_precision(name: str) -> str:
    if name != "auto":
        return name
    if not torch.cuda.is_available():
        return "fp32"
    if torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def build_optimizer(
    model: nn.Module,
    lr: float,
    lr_head: Optional[float] = None,
    weight_decay: float = 0.05,
    layer_decay: float = 1.0,
    depth: int = 24,
) -> torch.optim.Optimizer:
    """AdamW with (optional) layer-wise LR decay and no weight decay on norms/biases.

    Layer-wise decay ``gamma`` multiplies the LR of layer ``i`` by ``gamma ** (depth - i)``
    so that the pretrained early layers move least - standard practice when fully
    fine-tuning a large video backbone on a small dataset.
    """
    lr_head = lr_head if lr_head is not None else lr
    buckets: Dict[Tuple[float, float], List[nn.Parameter]] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        wd = 0.0 if no_decay else weight_decay
        if name.startswith("layers."):
            idx = int(name.split(".")[1])
            scale = layer_decay ** (depth - idx) if layer_decay < 1.0 else 1.0
        else:
            scale = 1.0
        base_lr = lr_head if name.startswith("head.") else lr * scale
        buckets.setdefault((round(base_lr, 10), wd), []).append(p)

    param_groups = [
        {"params": ps, "lr": l, "weight_decay": w, "name": f"lr{l:.2e}_wd{w}"}
        for (l, w), ps in sorted(buckets.items())
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(optimizer, epochs: int, warmup_epochs: float, steps_per_epoch: int,
                    kind: str = "cosine"):
    warmup = max(1, int(round(warmup_epochs * steps_per_epoch)))
    total = max(1, epochs * steps_per_epoch)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# --------------------------------------------------------------------------------------
@dataclass
class TrainHistory:
    epochs: List[dict] = field(default_factory=list)

    def add(self, row: dict):
        self.epochs.append(row)

    def best(self, key: str = "val_mae") -> dict:
        if not self.epochs:
            return {}
        return min(self.epochs, key=lambda r: r.get(key, float("inf")))

    def to_json(self) -> str:
        return json.dumps(self.epochs, indent=2)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        precision: str = "fp16",
        grad_clip: float = 1.0,
        log_every: int = 50,
        out_dir: Optional[str] = None,
        depth: int = 24,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.precision = precision
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.out_dir = out_dir
        self.depth = depth

        self.use_amp = precision in ("fp16", "bf16") and device.startswith("cuda")
        self.amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        self.scaler = None
        if self.use_amp and precision == "fp16":
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):  # older torch
                self.scaler = torch.cuda.amp.GradScaler()

    # -- core loops --------------------------------------------------------------------
    def _autocast(self):
        if not self.use_amp:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def train_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        t0 = time.time()
        total, n = 0.0, 0
        skipped = 0
        for step, (x, y, _) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).float()
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pred = self.model(x)
                loss = self.criterion(pred.float(), y)

            if not torch.isfinite(loss):
                skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip,
                    )
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            bs = x.shape[0]
            total += float(loss.detach()) * bs
            n += bs
            if self.log_every and step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"    e{epoch} step {step}/{len(loader)} loss {float(loss):.4f} lr {lr:.2e}",
                      flush=True)
        dt = time.time() - t0
        return {"loss": total / max(1, n), "seconds": dt, "n": n,
                "skipped": skipped, "samples_per_s": n / max(1e-9, dt)}

    @torch.no_grad()
    def predict(self, loader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        self.model.eval()
        preds, trues, names = [], [], []
        for x, y, nm in loader:
            x = x.to(self.device, non_blocking=True)
            with self._autocast():
                p = self.model(x)
            preds.append(p.float().cpu().numpy())
            trues.append(np.asarray(y, dtype=np.float32))
            names.extend(nm)
        return np.concatenate(trues), np.concatenate(preds), names

    def evaluate(self, loader) -> Tuple[dict, np.ndarray, np.ndarray]:
        y, p, _ = self.predict(loader)
        return compute_metrics(y, p), y, p

    # -- checkpointing -----------------------------------------------------------------
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"model": self.model.state_dict(),
                   "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                   "scaler": self.scaler.state_dict() if self.scaler else None,
                   "extra": extra or {}}
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic: a killed session never leaves a corrupt file


def count_trainable(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def peak_gpu_memory() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
    }


@torch.no_grad()
def benchmark_inference(model: nn.Module, loader, device: str, precision: str = "fp16",
                        warmup: int = 5, iters: int = 30, batch_size: int = 1) -> dict:
    """Wall-clock inference latency after merging (single clip) and throughput (batch)."""
    model.eval()
    if not device.startswith("cuda"):
        # CPU is ~100x slower than a T4; a full 30-iteration benchmark is pointless there.
        warmup, iters = 1, 2
    use_amp = precision in ("fp16", "bf16") and device.startswith("cuda")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    # cache a few real batches so that data loading is not measured
    batches = []
    for x, _, _ in loader:
        batches.append(x[:batch_size].to(device))
        if len(batches) >= max(iters, warmup + 1):
            break
    if not batches:
        return {}

    def run(x):
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                return model(x)
        return model(x)

    for i in range(warmup):
        run(batches[i % len(batches)])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for i in range(iters):
        run(batches[i % len(batches)])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    per_iter = dt / iters
    return {
        "inf_batch_size": batch_size,
        "inf_ms_per_clip": 1000.0 * per_iter / batch_size,
        "inf_clips_per_s": batch_size / per_iter,
        "inf_peak_mem_mb": (torch.cuda.max_memory_allocated() / 2**20
                            if device.startswith("cuda") else 0.0),
    }


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.isfile(path) else 0


class HuberLoss(nn.Module):
    """Smooth L1 on EF in percentage points (beta = 1 point).

    Chosen over MSE because the EF distribution is left-skewed with a long low tail
    (heart failure); L1-type losses optimise the reported MAE more directly and are less
    sensitive to the few very low EF outliers.  Identical for both conditions.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, beta=self.beta)


def make_criterion(name: str) -> nn.Module:
    name = name.lower()
    if name in ("l1", "mae"):
        return nn.L1Loss()
    if name in ("l2", "mse"):
        return nn.MSELoss()
    if name in ("huber", "smooth_l1"):
        return HuberLoss(1.0)
    raise ValueError(name)
