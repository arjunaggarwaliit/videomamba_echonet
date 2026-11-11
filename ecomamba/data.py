"""EchoNet-Dynamic loading: clips, splits, augmentations.

Two backends
------------
``mode="video"``  decode each AVI with OpenCV on the fly (a 174-frame 112x112 AVI takes
                  ~20 ms, so ~4 workers feed a T4 comfortably).
``mode="cache"``  read from a single ``.npy`` uint8 memmap built by
                  ``scripts/prepare_data.py`` - much faster per epoch, 4 GB on disk for
                  32 stored frames at 112x112.

Both return the *same* tensor layout ``(3, T, S, S)`` float32, so the two experimental
conditions see identical inputs.

Grayscale -> RGB
----------------
VideoMamba is pretrained on RGB frames with ImageNet statistics.  Echocardiography is a
single speckle-intensity channel.  We replicate the channel three times and apply the
ImageNet mean/std.  Replication (rather than, e.g., a learned 1->3 projection) keeps the
pretrained first-layer filters meaningful: for a grey input ``v`` the patch embedding of
the replicated frame equals ``W_conv @ (v * [1,1,1]^T)``, i.e. the sum of the pretrained
RGB filters, which is the standard way to reuse an RGB stem on greyscale data.  The
alternative (feeding grey into only one channel) wastes two thirds of the stem.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Common locations of the dataset (local machine + the Kaggle mirror).
DEFAULT_ROOTS = (
    r"C:\Users\Arjun Aggarwal\Desktop\courses\AI512\Project\Dataset\EchoNet-Dynamic\EchoNet-Dynamic",
    "/kaggle/input/echonet-dynamic/EchoNet-Dynamic",
    "/kaggle/input/echonet-dynamic",
    "/kaggle/input/echonetdynamic/EchoNet-Dynamic",
)


def resolve_root(root: Optional[str] = None) -> str:
    """Find the EchoNet-Dynamic directory (the one containing ``FileList.csv``)."""
    candidates: List[str] = []
    if root:
        candidates.append(root)
    env = os.environ.get("ECHONET_ROOT")
    if env:
        candidates.append(env)
    candidates.extend(DEFAULT_ROOTS)
    for c in candidates:
        if not c:
            continue
        if os.path.isfile(os.path.join(c, "FileList.csv")):
            return c
        nested = os.path.join(c, "EchoNet-Dynamic")
        if os.path.isfile(os.path.join(nested, "FileList.csv")):
            return nested
    raise FileNotFoundError(
        "Could not locate EchoNet-Dynamic (a directory containing FileList.csv). "
        "Pass --data-root or set ECHONET_ROOT. Tried:\n  " + "\n  ".join(candidates)
    )


def load_filelist(root: str) -> List[Dict]:
    """Parse FileList.csv -> list of {file, ef, esv, edv, frames, fps, split}."""
    import csv

    rows = []
    with open(os.path.join(root, "FileList.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                dict(
                    file=r["FileName"],
                    ef=float(r["EF"]),
                    esv=float(r.get("ESV", "nan")),
                    edv=float(r.get("EDV", "nan")),
                    frames=int(r.get("NumberOfFrames", 0) or 0),
                    fps=float(r.get("FPS", 50) or 50),
                    split=r["Split"].strip().upper(),
                )
            )
    return rows


class EchoNetClips(Dataset):
    """Fixed-length clips + EF targets.

    Parameters
    ----------
    root      : EchoNet-Dynamic directory
    split     : ``TRAIN`` / ``VAL`` / ``TEST`` (the official split from FileList.csv)
    num_frames: clip length fed to VideoMamba
    img_size  : spatial size fed to VideoMamba (the clips are 112x112; 224 keeps the
                pretrained 14x14 patch grid)
    train     : enables augmentation
    fraction  : use only this fraction of the split (reduced-data study); the subset is
                drawn with a fixed seed so every run sees the same videos
    cache     : optional path to the ``.npy`` clip cache
    """

    def __init__(
        self,
        root: str,
        split: str = "TRAIN",
        num_frames: int = 16,
        img_size: int = 224,
        train: bool = False,
        fraction: float = 1.0,
        cache: Optional[str] = None,
        cache_index: Optional[str] = None,
        hflip_p: float = 0.5,
        color_jitter: float = 0.2,
        scale_jitter: float = 0.1,
        temporal_jitter: bool = True,
        seed: int = 0,
        max_items: Optional[int] = None,
    ):
        self.root = root
        self.split = split.upper()
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train
        self.hflip_p = hflip_p
        self.color_jitter = color_jitter
        self.scale_jitter = scale_jitter
        self.temporal_jitter = temporal_jitter
        self.rng = random.Random(seed * 1000 + hash(self.split) % 1000)

        all_rows = load_filelist(root)
        rows = [r for r in all_rows if r["split"] == self.split]
        if not rows:
            raise ValueError(f"no rows for split {self.split}; "
                             f"available: {sorted({r['split'] for r in all_rows})}")
        if fraction < 1.0:
            rng = random.Random(1234)  # fixed: identical subsets across runs
            k = max(1, int(round(len(rows) * fraction)))
            rows = sorted(rng.sample(rows, k), key=lambda r: r["file"])
        if max_items:
            rows = rows[:max_items]
        self.rows = rows
        self.ef = np.array([r["ef"] for r in rows], dtype=np.float32)
        self.video_dir = os.path.join(root, "Videos")

        self.cache = None
        self.cache_index = None
        if cache and os.path.isfile(cache):
            self.cache = np.load(cache, mmap_mode="r")
            with open(cache_index or (cache + ".json")) as f:
                idx = json.load(f)
            # map file name -> row of the memmap
            self._pos = {name: i for i, name in enumerate(idx["files"])}
            missing = [r["file"] for r in rows if r["file"] not in self._pos]
            if missing:
                raise ValueError(f"{len(missing)} clips missing from the cache, "
                                 f"e.g. {missing[:3]}. Re-run prepare_data.py.")
            self._store_frames = int(self.cache.shape[1])
        else:
            if cache:
                print(f"[data] cache {cache} not found - decoding AVI on the fly")
            self._store_frames = 0

    # -- helpers -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def _read_video(self, name: str) -> np.ndarray:
        import cv2

        path = os.path.join(self.video_dir, name + ".avi")
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr[:, :, 0] if fr.ndim == 3 else fr)  # grey channel of the AVI
        cap.release()
        if not frames:
            raise RuntimeError(f"failed to decode {path}")
        return np.stack(frames)  # (T, H, W) uint8

    def _pick_indices(self, n_avail: int) -> np.ndarray:
        T = self.num_frames
        if n_avail <= T:
            return np.linspace(0, n_avail - 1, T).round().astype(np.int64)
        # dense uniform sampling with a random phase during training
        span = (n_avail - 1) / (T - 1)
        if self.train and self.temporal_jitter:
            base = random.uniform(0.0, span * 0.6)
        else:
            base = 0.0
        idx = np.round(base + np.arange(T) * span).astype(np.int64)
        return np.clip(idx, 0, n_avail - 1)

    def _augment(self, clip: np.ndarray) -> np.ndarray:
        """clip: (T, H, W) float32 in [0, 1]."""
        T, H, W = clip.shape
        if self.scale_jitter > 0:
            s = 1.0 - random.uniform(0.0, self.scale_jitter)
            ch = max(8, int(round(H * s)))
            y0 = random.randint(0, H - ch)
            x0 = random.randint(0, W - ch)
            clip = clip[:, y0 : y0 + ch, x0 : x0 + ch]
            clip = torch.from_numpy(clip).unsqueeze(0)
            clip = torch.nn.functional.interpolate(
                clip, size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0).numpy()
        if self.color_jitter > 0:
            gain = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            bias = random.uniform(-self.color_jitter, self.color_jitter) * 0.15
            clip = np.clip(clip * gain + bias, 0.0, 1.0)
        if random.random() < self.hflip_p:
            clip = clip[:, :, ::-1]
        return clip

    def __getitem__(self, i: int):
        row = self.rows[i]
        if self.cache is not None:
            store = self.cache[self._pos[row["file"]]]           # (T_s, H, W) uint8
            idx = self._pick_indices(store.shape[0])
            clip = store[idx].astype(np.float32) / 255.0
        else:
            video = self._read_video(row["file"])                # (T, 112, 112) uint8
            idx = self._pick_indices(video.shape[0])
            clip = video[idx].astype(np.float32) / 255.0

        if self.train:
            clip = self._augment(clip)

        if clip.shape[-1] != self.img_size:
            t = torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(1)   # (T,1,H,W)
            t = torch.nn.functional.interpolate(
                t, size=(self.img_size, self.img_size), mode="bicubic", align_corners=False
            )
            clip = t.squeeze(1).numpy()

        # (T, H, W) -> (3, T, H, W), replicate the single channel
        x = torch.from_numpy(np.ascontiguousarray(clip)).float().unsqueeze(0)
        x = x.repeat(1, 1, 1, 1).expand(3, -1, -1, -1).contiguous()
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1, 1)
        x = (x - mean) / std
        return x, float(row["ef"]), row["file"]


def build_loaders(
    root: str,
    num_frames: int,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
    fraction: float = 1.0,
    cache: Optional[str] = None,
    cache_index: Optional[str] = None,
    seed: int = 0,
    max_items: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
):
    from torch.utils.data import DataLoader

    common = dict(root=root, num_frames=num_frames, img_size=img_size, fraction=fraction,
                  cache=cache, cache_index=cache_index, max_items=max_items)
    train_ds = EchoNetClips(split="TRAIN", train=True, seed=seed, **common)
    val_ds = EchoNetClips(split="VAL", train=False, seed=seed, **common)
    test_ds = EchoNetClips(split="TEST", train=False, seed=seed, **common)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, drop_last=len(train_ds) > batch_size,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
    )
    eval_bs = eval_batch_size or batch_size
    val_loader = DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=num_workers > 0,
    )
    # Statistics of the FULL TRAIN split, regardless of --fraction / --limit-items.
    # The regression-head bias is initialised from ``ef_mean``, so it must not change when
    # only a subset of the training data is used; otherwise the reduced-data study would be
    # confounded by a different head initialisation.
    all_train = [r for r in load_filelist(root) if r["split"] == "TRAIN"]
    train_ef = np.array([r["ef"] for r in all_train], dtype=np.float32)
    stats = dict(
        train=len(train_ds), val=len(val_ds), test=len(test_ds),
        train_used=len(train_ds),
        ef_mean=float(train_ef.mean()), ef_std=float(train_ef.std()),
        ef_min=float(train_ef.min()), ef_max=float(train_ef.max()),
    )
    return train_loader, val_loader, test_loader, stats


def dataset_stats(root: str) -> Dict[str, float]:
    rows = load_filelist(root)
    out = {}
    for sp in ("TRAIN", "VAL", "TEST"):
        ef = np.array([r["ef"] for r in rows if r["split"] == sp], dtype=np.float32)
        out[sp] = dict(n=len(ef), mean=float(ef.mean()), std=float(ef.std()),
                       min=float(ef.min()), max=float(ef.max()))
    return out
