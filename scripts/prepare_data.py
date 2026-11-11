#!/usr/bin/env python
"""Pre-decode EchoNet-Dynamic into a single uint8 memmap of fixed-length clips.

Decoding 10k AVIs on the fly costs ~0.02 s per video (~150 s per epoch); reading from a
memmap costs almost nothing.  The cache is the difference between a data-bound and a
compute-bound training loop, which matters on Kaggle where the GPU is the expensive
resource and ``/kaggle/working`` has a 20 GB budget.

Output (default ``cache/echonet_f32_112.npy`` + ``.json``):

    .npy   uint8  (N, T_store, 112, 112)
    .json  {"files": [...], "frames": T_store, "size": 112, "rows": [...]}

Size for 10 031 videos x 32 frames x 112 x 112 = 4.0 GB.

Usage
-----
python scripts/prepare_data.py --out cache/echonet_f32_112.npy --frames 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecomamba.data import load_filelist, resolve_root  # noqa: E402


def decode_video(path: str) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr[:, :, 0] if fr.ndim == 3 else fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"failed to decode {path}")
    return np.stack(frames)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="cache/echonet_f32_112.npy")
    p.add_argument("--frames", type=int, default=32,
                   help="clips stored per video (uniformly sampled over the cycle)")
    p.add_argument("--size", type=int, default=112, help="spatial size of the stored clips")
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N videos")
    args = p.parse_args(argv)

    root = resolve_root(args.data_root)
    rows = load_filelist(root)
    if args.limit:
        rows = rows[: args.limit]
    video_dir = os.path.join(root, "Videos")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    shape = (len(rows), args.frames, args.size, args.size)
    arr = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.uint8, shape=shape)

    t0 = time.time()
    kept, failed = [], []
    for i, r in enumerate(rows):
        path = os.path.join(video_dir, r["file"] + ".avi")
        try:
            v = decode_video(path)
        except Exception as e:  # noqa: BLE001
            failed.append((r["file"], str(e)))
            arr[i] = 0
            continue
        n = v.shape[0]
        idx = np.linspace(0, n - 1, args.frames).round().astype(np.int64)
        clip = v[idx]
        if clip.shape[-1] != args.size:
            import cv2

            clip = np.stack([cv2.resize(f, (args.size, args.size),
                                        interpolation=cv2.INTER_AREA) for f in clip])
        arr[i] = clip
        kept.append(r["file"])
        if (i + 1) % 200 == 0 or i == len(rows) - 1:
            el = time.time() - t0
            print(f"  {i + 1}/{len(rows)}  {el:.0f}s  "
                  f"eta {el / (i + 1) * (len(rows) - i - 1):.0f}s", flush=True)

    arr.flush()
    del arr

    meta = {
        "files": [r["file"] for r in rows],
        "rows": rows,
        "frames": args.frames,
        "size": args.size,
        "failed": failed,
        "root": root,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f)
    gb = np.prod(shape) / 2**30
    print(f"\ndone: {args.out}  shape={shape}  {gb:.2f} GB  "
          f"({len(failed)} videos failed)  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
