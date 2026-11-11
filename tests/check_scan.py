import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecomamba.mamba import selective_scan_naive, selective_scan_chunked

torch.manual_seed(0)
ok = True
for (b, d, n, L) in [(2, 16, 8, 37), (2, 8, 4, 512), (3, 64, 16, 193)]:
    u = torch.randn(b, d, L)
    dt = torch.rand(b, d, L) * 0.05
    A = -torch.exp(torch.log(torch.arange(1, n + 1).float()).repeat(d, 1))
    B = torch.randn(b, n, L)
    C = torch.randn(b, n, L)
    D = torch.ones(d)
    z = torch.randn(b, d, L)
    ref = selective_scan_naive(u, dt, A, B, C, D, z=z, delta_softplus=True)
    for ch in (1, 16, 64):
        got = selective_scan_chunked(u, dt, A, B, C, D, z=z, delta_softplus=True, chunk_size=ch)
        rel = float((ref - got).abs().max() / ref.abs().max())
        good = rel < 1e-4
        ok = ok and good
        print(f"b={b} d={d} L={L:4d} chunk={ch:2d}  rel={rel:.2e}  {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "FAILURES")
