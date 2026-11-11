"""Unit / integration tests for the ecomamba core (run on CPU, no GPU needed).

    python tests/test_core.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecomamba.mamba import (  # noqa: E402
    selective_scan_naive, selective_scan_chunked, Mamba, HAS_CUDA_SCAN,
)
from ecomamba.videomamba import build_videomamba_ef, infer_arch, VideoMambaEF  # noqa: E402
from ecomamba.lora import (  # noqa: E402
    LoRAConfig, apply_lora, count_parameters, merge_and_strip, lora_state_dict,
)

torch.manual_seed(0)


def _scan_inputs(b=2, d=16, n=8, L=37):
    u = torch.randn(b, d, L)
    dt = torch.rand(b, d, L) * 0.05
    A = -torch.exp(torch.log(torch.arange(1, n + 1).float()).repeat(d, 1))
    B = torch.randn(b, n, L)
    C = torch.randn(b, n, L)
    D = torch.ones(d)
    z = torch.randn(b, d, L)
    return u, dt, A, B, C, D, z


def test_scan_matches_naive():
    u, dt, A, B, C, D, z = _scan_inputs()
    ref = selective_scan_naive(u, dt, A, B, C, D, z=z, delta_softplus=True)
    for chunk in (1, 4, 8, 16, 64):
        got = selective_scan_chunked(u, dt, A, B, C, D, z=z, delta_softplus=True, chunk_size=chunk)
        err = (ref - got).abs().max().item()
        rel = err / ref.abs().max().item()
        assert rel < 1e-5, f"chunk={chunk} rel err {rel:.2e}"
        print(f"  chunk={chunk:3d}  max abs err {err:.3e}  rel {rel:.2e}   OK")
    # long sequence, strong decay (worst case for numerics)
    u, dt, A, B, C, D, z = _scan_inputs(b=2, d=8, n=4, L=512)
    dt = dt * 3.0
    ref = selective_scan_naive(u, dt, A, B, C, D, z=z, delta_softplus=True)
    got = selective_scan_chunked(u, dt, A, B, C, D, z=z, delta_softplus=True, chunk_size=16)
    rel = (ref - got).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-4, rel
    print(f"  L=512 strong-decay rel err {rel:.2e}   OK")


def test_mamba_shapes():
    m = Mamba(64, d_state=8, d_conv=4, expand=2, bimamba=True, use_fast_path=False)
    x = torch.randn(3, 21, 64)
    y = m(x)
    assert y.shape == x.shape, y.shape
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  Mamba(64) params={n_params:,} out={tuple(y.shape)}  OK")


def test_checkpoint_load(ckpt):
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = infer_arch(raw)
    print(f"  inferred arch: {arch}")
    model = build_videomamba_ef(ckpt, num_frames=8, img_size=112, drop_path_rate=0.0)
    tot, tr = count_parameters(model)
    print(f"  total params {tot:,}")
    x = torch.randn(2, 3, 8, 112, 112)
    t0 = time.time()
    with torch.no_grad():
        y = model(x)
    print(f"  forward {tuple(x.shape)} -> {tuple(y.shape)} in {time.time() - t0:.2f}s")
    assert y.shape == (2,)
    return model


def test_lora(model):
    for cfg in [
        LoRAConfig(r=4, alpha=8, targets=("in_proj", "out_proj")),
        LoRAConfig(r=4, alpha=8, targets=("in_proj", "out_proj", "x_proj", "dt_proj")),
        LoRAConfig(r=4, alpha=8, targets=("in_proj", "out_proj", "x_proj", "dt_proj", "conv1d")),
        LoRAConfig(r=4, alpha=8, targets=("dt_proj",), dt_tie=False),
        LoRAConfig(r=2, alpha=4, targets=("in_proj", "out_proj"), layers="last2"),
        LoRAConfig(r=4, alpha=8, targets=("a_log",)),
        LoRAConfig(r=4, alpha=8, targets=("a_log",), a_tie=False),
        LoRAConfig(r=4, alpha=8, targets=("in_proj", "out_proj", "dt_proj", "a_log")),
        LoRAConfig(r=4, alpha=8, targets=("in_proj", "out_proj"), in_proj_slice="x"),
        LoRAConfig(r=4, alpha=8, targets=("in_proj", "out_proj"), in_proj_slice="z"),
    ]:
        import copy
        m = copy.deepcopy(model)
        apply_lora(m, cfg, verbose=False)
        tot, tr = count_parameters(m)
        x = torch.randn(2, 3, 8, 112, 112)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2,)
        print(f"  targets={cfg.targets} r={cfg.r} tie={cfg.dt_tie} layers={cfg.layers}: "
              f"trainable {tr:,}/{tot:,} ({100*tr/tot:.2f}%)  OK")

        # merge must leave the predictions unchanged and restore a clean module
        m.eval()
        with torch.no_grad():
            y_merged_ref = m(x)
        sd = lora_state_dict(m)
        m2 = merge_and_strip(m)
        with torch.no_grad():
            y_merged = m2(x)
        err = (y_merged - y_merged_ref).abs().max().item()
        assert err < 1e-4, f"merge changed the output by {err}"
        assert not any(".lora_" in k for k in m2.state_dict())
        print(f"    merge ok (max |dy| = {err:.2e}), adapter tensors = {len(sd)}")


def main():
    print(f"\n[1] selective scan  (CUDA kernel available: {HAS_CUDA_SCAN})")
    test_scan_matches_naive()
    print("\n[2] Mamba block")
    test_mamba_shapes()

    ckpts = [p for p in ("ckpt/videomamba_t16_k400_f16_res224.pth",
                         "ckpt/videomamba_s16_k400_f16_res224.pth") if os.path.exists(p)]
    if not ckpts:
        print("\n[3] checkpoint tests SKIPPED (no checkpoint in ./ckpt)")
        return
    print(f"\n[3] checkpoint loading: {ckpts[0]}")
    model = test_checkpoint_load(ckpts[0])
    print("\n[4] LoRA placement / merge")
    test_lora(model)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
