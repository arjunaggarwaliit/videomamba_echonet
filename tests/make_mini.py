"""Build a tiny, self-consistent EchoNet subset (its own FileList.csv + Videos/).

Used to exercise the clip-cache path end-to-end without needing 4 GB of address space.

    python tests/make_mini.py --n-train 24 --n-val 8 --n-test 8
"""
import argparse
import csv
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ecomamba.data import load_filelist, resolve_root  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=os.path.join(HERE, "mini_echonet"))
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--n-val", type=int, default=8)
    p.add_argument("--n-test", type=int, default=8)
    args = p.parse_args(argv)

    root = resolve_root(args.data_root)
    rows = load_filelist(root)
    want = {"TRAIN": args.n_train, "VAL": args.n_val, "TEST": args.n_test}
    picked, seen = [], {"TRAIN": 0, "VAL": 0, "TEST": 0}
    for r in rows:
        sp = r["split"]
        if sp in want and seen[sp] < want[sp]:
            picked.append(r)
            seen[sp] += 1
    assert all(seen.values()), f"not enough rows: {seen}"

    os.makedirs(os.path.join(args.out, "Videos"), exist_ok=True)
    for r in picked:
        src = os.path.join(root, "Videos", r["file"] + ".avi")
        dst = os.path.join(args.out, "Videos", r["file"] + ".avi")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    with open(os.path.join(args.out, "FileList.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FileName", "EF", "ESV", "EDV", "FrameHeight", "FrameWidth",
                    "FPS", "NumberOfFrames", "Split"])
        for r in picked:
            w.writerow([r["file"], r["ef"], r["esv"], r["edv"], 112, 112,
                        r["fps"], r["frames"], r["split"]])
    print(f"wrote {len(picked)} clips to {args.out}  {dict(seen)}")


if __name__ == "__main__":
    main()
