"""Checkpoint-compatible Mamba block for VideoMamba (bidirectional / "bimamba").

Why this file exists
--------------------
VideoMamba ships a *fork* of ``mamba_ssm`` (``vendor/VideoMamba/mamba``) that adds a
bidirectional scan: ``conv1d_b``, ``x_proj_b``, ``dt_proj_b``, ``A_b_log``, ``D_b`` and a
fused kernel ``mamba_inner_fn_no_out_proj``.  Two problems make that fork a poor
foundation for this project:

1.  The fused kernel ``mamba_inner_fn_no_out_proj`` reads the *raw weights*
    (``self.x_proj.weight``, ``self.dt_proj.weight``, ...) instead of calling the
    projection modules.  Any LoRA adapter attached to those modules would therefore be
    silently bypassed -> LoRA on ``x_proj`` / ``dt_proj`` / ``in_proj`` / ``out_proj``
    would be a no-op.  LoRA experiments *must* use the unfused path.

2.  The unfused branch of ``mamba_simple.Mamba.forward`` **ignores the backward
    direction entirely** (``bimamba`` is only implemented in the fused kernel).  So
    turning the fused path off silently changes the model.

We therefore re-implement the block here with

* identical parameter *names and shapes* (so the released checkpoints load unchanged),
* a correct **bidirectional** unfused forward that calls the projection modules
  (so LoRA adapters are actually used),
* an optional fused path (used only for the full-fine-tuning baseline when the fork is
  installed), and
* a pure-PyTorch ``selective_scan`` fallback so the whole pipeline runs on CPU / on a
  GPU without ``mamba-ssm`` (smoke tests, CI, debugging).

Numerics are unchanged: the unfused path is algebraically identical to the fused kernel.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Optional CUDA kernels.  Everything degrades gracefully.
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - depends on the environment
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _CUDA_SCAN
except Exception:  # ImportError, or a broken/incompatible build
    _CUDA_SCAN = None

try:  # pragma: no cover
    from mamba_ssm.ops.selective_scan_interface import (
        mamba_inner_fn_no_out_proj as _CUDA_INNER_NO_OUT,
    )
except Exception:
    _CUDA_INNER_NO_OUT = None

try:  # pragma: no cover
    from causal_conv1d import causal_conv1d_fn as _CAUSAL_CONV1D
except Exception:
    _CAUSAL_CONV1D = None

try:  # pragma: no cover
    from mamba_ssm.ops.triton.layernorm import (
        RMSNorm as _TritonRMSNorm,
        layer_norm_fn as _triton_layer_norm_fn,
        rms_norm_fn as _triton_rms_norm_fn,
    )
except Exception:
    _TritonRMSNorm = None
    _triton_layer_norm_fn = None
    _triton_rms_norm_fn = None


HAS_CUDA_SCAN = _CUDA_SCAN is not None
HAS_FUSED_INNER = _CUDA_INNER_NO_OUT is not None
HAS_CAUSAL_CONV1D = _CAUSAL_CONV1D is not None
HAS_TRITON_NORM = _TritonRMSNorm is not None


# --------------------------------------------------------------------------------------
# LayerNorm / RMSNorm (fused add+norm semantics of mamba_ssm's triton kernels)
# --------------------------------------------------------------------------------------
def _add_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-5,
    is_rms_norm: bool = True,
):
    """Pure-PyTorch equivalent of ``mamba_ssm.ops.triton.layernorm.rms_norm_fn``.

    ``residual is None`` -> the input is the first residual in the block.
    ``prenorm=True``  -> return ``(normalised, new_residual)`` for the next sub-layer.
    ``prenorm=False`` -> return only the normalised tensor.
    """
    if residual is None:
        residual = x
    else:
        residual = residual + x
    src = residual
    xf = src.to(torch.float32)
    if is_rms_norm:
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + eps)
    else:
        mu = xf.mean(dim=-1, keepdim=True)
        var = (xf - mu).pow(2).mean(dim=-1, keepdim=True)
        xf = (xf - mu) * torch.rsqrt(var + eps)
    if weight is not None:
        xf = xf * weight.to(torch.float32)
    if bias is not None:
        xf = xf + bias.to(torch.float32)
    out = xf.to(src.dtype)
    if prenorm:
        return out, (residual.to(torch.float32) if residual_in_fp32 else residual)
    return out


def layer_norm_fn(
    x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-5, **kwargs
):
    return _add_norm(x, weight, bias, residual, prenorm, residual_in_fp32, eps, is_rms_norm=False)


def rms_norm_fn(
    x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-5, **kwargs
):
    return _add_norm(x, weight, bias, residual, prenorm, residual_in_fp32, eps, is_rms_norm=True)


class RMSNorm(nn.Module):
    """Drop-in replacement for ``mamba_ssm``'s triton RMSNorm (weight only, no bias)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        # The triton module exposes ``.bias`` (set to None); the fused Block code reads it.
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_fn(x, self.weight, None, eps=self.eps, prenorm=False)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"


# Use the triton kernels when they exist (GPU), otherwise the pure-torch versions.
_FUSED_RMS = _triton_rms_norm_fn if _triton_rms_norm_fn is not None else rms_norm_fn
_FUSED_LN = _triton_layer_norm_fn if _triton_layer_norm_fn is not None else layer_norm_fn
_RMSNormImpl = _TritonRMSNorm if _TritonRMSNorm is not None else RMSNorm


# --------------------------------------------------------------------------------------
# Selective scan (pure PyTorch fallback)
# --------------------------------------------------------------------------------------
def selective_scan_naive(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    """Reference recurrence, O(L) python loop.  Used to validate the chunked version."""
    b, d, L = u.shape
    n = A.shape[1]
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, d, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    dt = delta.float()
    uf, Bf, Cf = u.float(), B.float(), C.float()
    Aneg = A.float()
    h = torch.zeros(b, d, n, device=u.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        dA = torch.exp(dt[:, :, t].unsqueeze(-1) * Aneg.view(1, d, n))
        dB = dt[:, :, t].unsqueeze(-1) * Bf[:, :, t].unsqueeze(1)
        h = h * dA + dB * uf[:, :, t].unsqueeze(-1)
        ys.append((h * Cf[:, :, t].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + uf * D.float().view(1, d, 1)
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(u.dtype)


def selective_scan_chunked(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True,
    chunk_size: int = 16, max_elements: int = 32_000_000,
):
    """Exact, memory-bounded parallel scan.

    Splits the sequence into chunks of ``C`` steps.  Inside a chunk the recurrence is
    solved in closed form with the lower-triangular decay matrix

        G[t, s] = exp(m_t - m_s),  s <= t,      m = cumsum(dt * A) <= 0

    so ``G`` is a product of factors in (0, 1]: no overflow, no ``exp(+big)``.  The
    chunk-to-chunk state is propagated sequentially (``ceil(L / C)`` cheap iterations).
    ``chunk_size`` is reduced automatically so that ``G`` stays within ``max_elements``.
    """
    b, d, L = u.shape
    n = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, d, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    # Adaptive chunk size: G has b*d*C*C*n elements.
    C_ch = max(1, min(chunk_size, int(math.sqrt(max_elements / max(1, b * d * n)))))
    pad = (-L) % C_ch
    if pad:
        u = F.pad(u, (0, pad))
        delta = F.pad(delta, (0, pad))
        B = F.pad(B, (0, pad))
        C = F.pad(C, (0, pad))
    Lp = u.shape[-1]
    nc = Lp // C_ch

    # fp32 throughout: the decay factors span many orders of magnitude.
    uf = u.float()
    dtf = delta.float()
    Bf = B.float()
    Cf = C.float()
    Aneg = A.float().view(1, d, 1, n)

    # log dA and the "dB * u" term, reshaped into chunks  (b, d, nc, C, n)
    log_dA = (dtf.unsqueeze(-1) * Aneg).view(b, d, nc, C_ch, n)
    dBu = (dtf.unsqueeze(-1) * Bf.transpose(1, 2).unsqueeze(1)) * uf.unsqueeze(-1)
    dBu = dBu.view(b, d, nc, C_ch, n)

    mask = torch.tril(torch.ones(C_ch, C_ch, dtype=torch.bool, device=u.device))
    mask_f = mask.view(1, 1, C_ch, C_ch, 1)

    ys = torch.empty(b, d, Lp, device=u.device, dtype=torch.float32)
    h = torch.zeros(b, d, n, device=u.device, dtype=torch.float32)  # carried SSM state

    for c in range(nc):
        ldA = log_dA[:, :, c]                       # (b, d, C, n)
        m = torch.cumsum(ldA, dim=2)                # (b, d, C, n)  non-increasing (A<0, dt>0)
        # G[t, s] = exp(m_t - m_s) for s <= t.  For s > t the exponent would be positive
        # and can overflow to +inf; clamp *before* exp so that ``inf * 0`` cannot occur.
        diff = (m.unsqueeze(3) - m.unsqueeze(2)).clamp_(max=0.0)
        G = torch.exp(diff) * mask_f                # (b, d, C, C, n)
        # Contract one state dimension at a time: the 5-D einsum would materialise a
        # (b, d, C, C) intermediate *per state*, which at d_inner = 768, C = 16 is 12 MB
        # per call - enough to matter on memory-constrained machines.  Looping over
        # n (16) keeps the peak at one (b, d, C, C) slab and is the same arithmetic.
        intrab = torch.empty(b, d, C_ch, n, device=u.device, dtype=torch.float32)
        for k in range(n):
            intrab[..., k] = torch.einsum("bdts,bds->bdt", G[..., k], dBu[:, :, c, :, k])
        prev = torch.exp(m) * h.unsqueeze(2)                          # (b, d, C, n)
        H = intrab + prev
        ys[:, :, c * C_ch : (c + 1) * C_ch] = torch.einsum(
            "bdtn,bnt->bdt", H, Cf[:, :, c * C_ch : (c + 1) * C_ch]
        )
        h = H[:, :, -1, :]

    y = ys[:, :, :L]
    if D is not None:
        y = y + uf[:, :, :L] * D.float().view(1, d, 1)
    if z is not None:
        y = y * F.silu(z.float()[:, :, :L])
    return y.to(u.dtype)


def selective_scan(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True, chunk_size=16
):
    """Dispatch to the CUDA kernel when possible, else the chunked PyTorch version.

    The PyTorch version keeps one ``(b, d, C, C, n)`` decay matrix *per chunk* alive in the
    autograd graph, i.e. ``b*d*L*C*n`` floats for one scan - several hundred MB per block at
    ``d_inner=768, C=16``.  Under ``torch.no_grad`` (inference) that is harmless, but during
    training it is not, so the fallback is wrapped in gradient checkpointing: the chunk loop
    is recomputed in the backward pass instead of being stored.  That trades ~1/3 extra
    compute on the scan (a small part of the block) for an O(1) activation footprint.
    """
    if _CUDA_SCAN is not None and u.is_cuda:
        return _CUDA_SCAN(
            u, delta, A, B, C, D,
            z=z, delta_bias=delta_bias, delta_softplus=delta_softplus,
        )

    kwargs = dict(z=z, delta_bias=delta_bias, delta_softplus=delta_softplus,
                  chunk_size=chunk_size)
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (u, delta, B, C, A, D, z, delta_bias)
    )
    if not needs_grad:
        return selective_scan_chunked(u, delta, A, B, C, D, **kwargs)

    def _fn(u_, d_, A_, B_, C_, D_, z_, db_):
        return selective_scan_chunked(u_, d_, A_, B_, C_, D_, z=z_, delta_bias=db_,
                                      delta_softplus=delta_softplus, chunk_size=chunk_size)

    try:
        return torch.utils.checkpoint.checkpoint(
            _fn, u, delta, A, B, C, D, z, delta_bias, use_reentrant=False
        )
    except Exception:  # pragma: no cover - very old torch, or a non-tensor edge case
        return selective_scan_chunked(u, delta, A, B, C, D, **kwargs)


# --------------------------------------------------------------------------------------
# The Mamba mixer
# --------------------------------------------------------------------------------------
class Mamba(nn.Module):
    """Bidirectional selective state-space mixer, weight-compatible with VideoMamba.

    Parameter names match ``mamba_ssm.modules.mamba_simple.Mamba`` exactly:

        in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj
        conv1d_b, x_proj_b, dt_proj_b, A_b_log, D_b        (bidirectional branch)
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba
        # Set to True by ``ecomamba.lora`` once adapters are attached.
        self.lora_active = False
        # Optional A-LoRA adapter (see ecomamba.lora.ALogLoRA); ``None`` by default so
        # state_dict keys stay identical to the released VideoMamba checkpoints.
        self.a_lora = None

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # --- identical initialisation to mamba_simple (must not diverge: weights are loaded)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
        A = A.repeat(self.d_inner, 1).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        if self.bimamba:
            A_b = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
            A_b = A_b.repeat(self.d_inner, 1).contiguous()
            A_b_log = torch.log(A_b)
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    # -- helpers -----------------------------------------------------------------------
    def fast_path_available(self) -> bool:
        """The fused kernel bypasses the projection modules, so it is only valid for
        full fine-tuning (no adapters) and only for the bidirectional variant."""
        return bool(
            self.use_fast_path and self.bimamba and HAS_FUSED_INNER and not self.lora_active
        )

    def _short_conv(self, x, conv):
        L = x.shape[-1]
        # Use the fused causal-conv kernel only when the module is the *unmodified*
        # depth-wise conv, so that full fine-tuning and LoRA runs execute the same ops.
        # ``activation`` is passed by keyword: the vendored causal-conv1d takes it as the
        # 4th positional argument, while upstream >= 1.5 takes ``seq_idx`` there.
        if isinstance(conv, nn.Conv1d) and _CAUSAL_CONV1D is not None:
            try:
                return _CAUSAL_CONV1D(
                    x, conv.weight.squeeze(1), conv.bias, activation=self.activation
                )
            except TypeError:  # signature mismatch -> fall through to nn.Conv1d
                pass
        return self.act(conv(x)[..., :L])

    def _ssm_branch(self, xz, conv, x_proj, dt_proj, A, D, direction="f"):
        """One scan direction.  ``xz``: (b, d_inner*2, L).  Returns y: (b, d_inner, L)."""
        x, z = xz.chunk(2, dim=1)
        L = x.shape[-1]

        x = self._short_conv(x, conv)

        x_dbl = x_proj(x.transpose(1, 2).reshape(-1, x.shape[1]))       # (b*L, dt_rank+2n)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        if hasattr(dt_proj, "project_f"):
            # DtLoRA adapter: it owns the base weights and the low-rank correction.
            # A tied adapter serves both directions (has_b=True); untied adapters are
            # two separate modules that each only handle their own direction.
            if direction == "b" and getattr(dt_proj, "has_b", False):
                dt = dt_proj.project_b(dt)
            else:
                dt = dt_proj.project_f(dt)
            delta_bias = dt_proj.bias_for(direction)
        else:
            dt = F.linear(dt, dt_proj.weight)                          # (b*L, d_inner)
            delta_bias = dt_proj.bias
        dt = dt.view(x.shape[0], L, -1).transpose(1, 2)                # (b, d_inner, L)
        B = B.view(x.shape[0], L, -1).transpose(1, 2).contiguous()
        C = C.view(x.shape[0], L, -1).transpose(1, 2).contiguous()

        return selective_scan(
            x, dt, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True
        )

    # -- forward -----------------------------------------------------------------------
    def forward(self, hidden_states, inference_params=None):
        b, seqlen, dim = hidden_states.shape
        if inference_params is not None:
            raise NotImplementedError("incremental decoding is not used in this project")

        if self.fast_path_available():
            # Bit-identical to the VideoMamba fork's fused path.
            xz = (
                self.in_proj.weight @ hidden_states.transpose(1, 2).reshape(dim, b * seqlen)
            ).view(-1, b, seqlen).contiguous()
            out = _CUDA_INNER_NO_OUT(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                -torch.exp(self.A_log.float()),
                None,
                None,
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            out_b = _CUDA_INNER_NO_OUT(
                xz.flip([-1]),
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                -torch.exp(self.A_b_log.float()),
                None,
                None,
                self.D_b.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )
            y = out + out_b.flip([-1])
            return self.out_proj(y.transpose(1, 2))

        xz = self.in_proj(hidden_states).transpose(1, 2).contiguous()   # (b, 2*d_inner, L)
        y = self._ssm_branch(
            xz, self.conv1d, self.x_proj, self.dt_proj,
            self._neg_exp_A("f"), self.D, direction="f",
        )
        if self.bimamba:
            y_b = self._ssm_branch(
                xz.flip([-1]), self.conv1d_b, self.x_proj_b, self.dt_proj_b,
                self._neg_exp_A("b"), self.D_b, direction="b",
            )
            y = y + y_b.flip([-1])
        return self.out_proj(y.transpose(1, 2))

    def _neg_exp_A(self, direction: str) -> torch.Tensor:
        """``A = -exp(A_log)``, with the optional rank-``r`` A-LoRA correction applied."""
        a_log = (self.A_log if direction == "f" else self.A_b_log).float()
        if self.a_lora is not None:
            a_log = a_log + self.a_lora.delta(direction)
        return -torch.exp(a_log)


# --------------------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm residual block:  residual + LN -> mixer, with optional fused add+norm."""

    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
                 residual_in_fp32=False, drop_path=0.0):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert _RMSNormImpl is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, _RMSNormImpl)), (
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            )

    def forward(self, hidden_states, residual=None, inference_params=None, use_checkpoint=False):
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = _FUSED_RMS if isinstance(self.norm, _RMSNormImpl) else _FUSED_LN
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                getattr(self.norm, "bias", None),
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            # ``torch.utils.checkpoint`` cannot take the (unused) kwargs object.
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.mixer, hidden_states, None, use_reentrant=False
            )
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


class DropPath(nn.Module):
    """Stochastic depth (per-sample drop of the whole residual branch)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"
