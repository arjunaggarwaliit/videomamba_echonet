"""LoRA adapters for VideoMamba, and the proposed **FG-LoRA** strategy.

1.  Where does LoRA belong inside a Mamba block?
================================================
A bidirectional VideoMamba block computes

    xz    = in_proj(h)              (d -> 2 d_inner)      content + gate
    x, z  = xz.chunk(2)                                   z is the output gate
    x     = silu(dwconv1d(x))                             depth-wise short conv, k = 4
    d,B,C = split(x_proj(x))        (d_inner -> dt_rank + 2 d_state)   selective params
    d     = dt_proj(d)              (dt_rank -> d_inner)  discretisation step  Delta
    A     = -exp(A_log)             (d_inner x d_state)   state transition   (a PARAMETER)
    h_t   = exp(dt_t * A) . h_{t-1} + dt_t B_t x_t        <-- the recurrence
    y_t   = C_t . h_t + D . x_t
    out   = out_proj(y * silu(z))

so the length of the model's memory is set by a single quantity, the **forget gate**

        Abar_t = exp(Delta_t . A)          in (0, 1)^(d_inner x d_state)

``Abar`` near 1 -> the state is kept (long memory); ``Abar`` near 0 -> the state is
overwritten (local, motion-sensitive).  Two things drive it:

* ``Delta`` (input dependent, produced by ``x_proj`` -> ``dt_proj``),
* ``A`` (input *independent*, a learned parameter matrix ``A_log``).

Why default LoRA is not enough
------------------------------
The LoRA paper targets Transformer attention projections.  Applied to Mamba blocks the
usual recipe is "adapt ``in_proj`` and ``out_proj``".  Two published results matter here:

* Deleon & Thampi, *SDT* (ICML 2025, arXiv:2410.09016), Lemma 1: a LoRA update on the
  fused ``in_proj`` can *exactly reproduce* a LoRA update on ``x_proj`` / ``dt_proj``
  (the Delta-path) - but it **cannot modify ``A``**.  ``A_log`` is the one matrix in the
  block that no projection adapter can reach.
* Yoshimura et al., *MambaPEFT* (ICLR 2025, arXiv:2411.03855): ``A``-tuning beats
  ``D``-tuning by a wide margin; the ``conv1d`` adapter is the most
  parameter-efficient single choice.

**FG-LoRA (this project's contribution).**  Instead of adapting only one factor of
``Abar``, adapt *both* factors of the forget gate with low-rank matrices:

    Abar_t = exp( (Delta_t + s_d B_d A_d u_t) . (A + s_a B_a A_a) )

    * **A-LoRA**   : W' = A_log + (alpha_a / r_a) B_a A_a,  B_a in R^{d_inner x r_a},
                     A_a in R^{r_a x d_state}.  ``A_log`` is a (d_inner x d_state)
                     *weight matrix*, so this is textbook LoRA - just applied to the
                     matrix that standard placements never touch.
    * **Delta-LoRA**: W' = W_dt + (alpha_d / r_d) B_d A_d, applied to ``dt_proj``.
    * **content LoRA**: the usual adapters on ``in_proj`` / ``out_proj`` (and the
      X- or Z-slice of ``in_proj``, following MambaPEFT's ``LoRA_p(X)``).

Both legs are rank-``r`` and, in the recommended configuration, the low-rank basis of
the A adapter is **shared between the two scan directions** (the forward and backward
scans read the same channels), so the cost is ``r_a (d_inner + 2 d_state)`` per layer -
about 3 orders of magnitude below full fine-tuning.

Each adapter also carries an optional scalar ``gate`` (init 0, applied through ``tanh``)
whose trained value says *how much* that particular layer needed to re-time itself; we
plot it as a diagnostic.  ``gate`` is applied to the low-rank *delta* only, so the model
still starts exactly at the pretrained function.

Everything else - ``D``, the LayerNorms, ``patch_embed``, the position embeddings - stays
frozen, as the assignment requires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import Mamba


# --------------------------------------------------------------------------------------
# Projection adapters
# --------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """``W0 x + scale * B (A x)`` with ``W0`` frozen and ``B`` zero-initialised.

    ``out_slice=(a, b)`` restricts the update to rows ``[a, b)`` of the output, which is
    how ``LoRA_p(X)`` / ``LoRA_p(Z)`` (MambaPEFT) are implemented: the fused ``in_proj``
    produces ``[x ; z]`` and only one half is adapted.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0,
                 name: str = "", out_slice: Optional[Tuple[int, int]] = None):
        super().__init__()
        assert r > 0, "LoRA rank must be positive"
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.name = name
        self.out_slice = out_slice
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad_(False)

        out_dim = base.out_features if out_slice is None else (out_slice[1] - out_slice[0])
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def raw_delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        d = self.raw_delta()
        if self.out_slice is None:
            return d
        return F.pad(d, (0, 0, self.out_slice[0], self.base.out_features - self.out_slice[1]))

    def forward(self, x):
        out = self.base(x)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        delta = delta * self.scaling
        if self.out_slice is not None:
            # zero-pad the update to the full output width (last dim, any leading dims)
            delta = F.pad(delta, (self.out_slice[0],
                                  self.base.out_features - self.out_slice[1]))
        return out + delta

    @torch.no_grad()
    def merge(self):
        """Fold the adapter into the frozen weight (zero inference overhead)."""
        self.base.weight.data += self.delta_weight().to(self.base.weight.dtype)

    @torch.no_grad()
    def unmerge(self):
        self.base.weight.data -= self.delta_weight().to(self.base.weight.dtype)

    def extra_repr(self) -> str:
        sl = "" if self.out_slice is None else f" slice={self.out_slice}"
        return f"{self.name} r={self.r} alpha={self.alpha}{sl}"


class LoRAConv1d(nn.Module):
    """Depth-wise conv + a rank-``r`` **channel-coupled** temporal filter.

    ``conv1d(x, B_c @ A_c)`` with ``B_c`` (d_inner, r) and ``A_c`` (r, kernel) adds a
    low-rank temporal basis that mixes channels - something the pretrained depth-wise
    conv cannot express at all.  ``B_c @ A_c`` is itself a channel-dependent depth-wise
    kernel, so it merges into the pretrained conv with no residual cost.
    """

    def __init__(self, base: nn.Conv1d, r: int, alpha: float, name: str = "", gate: bool = True):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.name = name
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.kernel_size[0]))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.gate = nn.Parameter(torch.zeros(())) if gate else None
        if self.gate is None:
            self.register_parameter("gate", None)
        self.pad = base.padding[0]

    def delta_weight(self) -> torch.Tensor:
        d = (self.lora_B @ self.lora_A) * self.scaling
        if self.gate is not None:
            d = d * torch.tanh(self.gate)
        return d  # (d_inner, k)

    def forward(self, x):
        out = self.base(x)
        w = self.delta_weight().unsqueeze(1)  # (d_inner, 1, k)
        delta = F.conv1d(x, w, bias=None, stride=1, padding=self.pad, groups=x.shape[1])
        return out + delta[..., : out.shape[-1]]

    @torch.no_grad()
    def merge(self):
        self.base.weight.data += self.delta_weight().unsqueeze(1).to(self.base.weight.dtype)

    @torch.no_grad()
    def unmerge(self):
        self.base.weight.data -= self.delta_weight().unsqueeze(1).to(self.base.weight.dtype)

    def extra_repr(self) -> str:
        return f"{self.name} r={self.r} k={self.base.kernel_size[0]}"


# --------------------------------------------------------------------------------------
# The two forget-gate adapters
# --------------------------------------------------------------------------------------
class DtLoRA(nn.Module):
    """**Delta-LoRA** - low-rank correction of the discretisation step Delta.

    Wraps ``dt_proj`` (``dt_rank -> d_inner``).  With ``tie=True`` one module serves both
    scan directions (shared low-rank basis ``A_d``, direction-specific ``B_d``);
    with ``tie=False`` two independent modules are created.

    Parameters (tied): ``A_d`` (r_d, dt_rank) + ``B_d``, ``B_d'`` (d_inner, r_d) + gate.
    """

    def __init__(self, base_f: nn.Linear, base_b: Optional[nn.Linear], r: int, alpha: float,
                 tie: bool = True, gate: bool = True, name: str = "dt_proj"):
        super().__init__()
        assert isinstance(base_f, nn.Linear)
        self.base_f = base_f
        self.base_b = base_b
        self.has_b = base_b is not None
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.tie = tie
        self.name = name
        for p in self.base_f.parameters():
            p.requires_grad_(False)
        if self.has_b:
            for p in self.base_b.parameters():
                p.requires_grad_(False)

        dt_rank = base_f.in_features
        d_inner = base_f.out_features
        self.lora_A = nn.Parameter(torch.empty(r, dt_rank))
        self.lora_B = nn.Parameter(torch.zeros(d_inner, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.has_b:
            self.lora_B_b = nn.Parameter(torch.zeros(d_inner, r))
        else:
            self.register_parameter("lora_B_b", None)
        self.gate = nn.Parameter(torch.zeros(())) if gate else None
        if self.gate is None:
            self.register_parameter("gate", None)
        if gate and self.has_b:
            self.gate_b = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("gate_b", None)

    def project_f(self, dt_raw: torch.Tensor) -> torch.Tensor:
        """(N, dt_rank) -> (N, d_inner) **without** bias; the bias is added by the
        selective scan as ``delta_bias``, exactly as in the reference implementation."""
        out = F.linear(dt_raw, self.base_f.weight)
        corr = F.linear(F.linear(dt_raw, self.lora_A), self.lora_B) * self.scaling
        if self.gate is not None:
            return out + corr * torch.tanh(self.gate)
        return out + corr

    def project_b(self, dt_raw: torch.Tensor) -> torch.Tensor:
        assert self.has_b, "this adapter only serves the forward direction"
        out = F.linear(dt_raw, self.base_b.weight)
        corr = F.linear(F.linear(dt_raw, self.lora_A), self.lora_B_b) * self.scaling
        if self.gate_b is not None:
            return out + corr * torch.tanh(self.gate_b)
        return out + corr

    def bias_for(self, direction: str) -> torch.Tensor:
        if direction == "b" and self.has_b:
            return self.base_b.bias
        return self.base_f.bias

    @torch.no_grad()
    def merge(self):
        d = (self.lora_B @ self.lora_A) * self.scaling
        if self.gate is not None:
            d = d * torch.tanh(self.gate)
        self.base_f.weight.data += d.to(self.base_f.weight.dtype)
        if self.has_b:
            db = (self.lora_B_b @ self.lora_A) * self.scaling
            if self.gate_b is not None:
                db = db * torch.tanh(self.gate_b)
            self.base_b.weight.data += db.to(self.base_b.weight.dtype)

    @torch.no_grad()
    def unmerge(self):
        d = (self.lora_B @ self.lora_A) * self.scaling
        if self.gate is not None:
            d = d * torch.tanh(self.gate)
        self.base_f.weight.data -= d.to(self.base_f.weight.dtype)
        if self.has_b:
            db = (self.lora_B_b @ self.lora_A) * self.scaling
            if self.gate_b is not None:
                db = db * torch.tanh(self.gate_b)
            self.base_b.weight.data -= db.to(self.base_b.weight.dtype)

    def extra_repr(self) -> str:
        return f"{self.name} r={self.r} tie={self.tie} dt_rank={self.lora_A.shape[1]}"


class ALogLoRA(nn.Module):
    """**A-LoRA** - low-rank correction of the state-transition matrix ``A_log``.

    ``A_log`` is a real ``(d_inner x d_state)`` weight matrix, so this *is* LoRA, just
    applied to the one matrix in the block that no projection adapter can reach
    (SDT, Lemma 1).  It reshapes the forget-gate profile ``exp(Delta . A)``, i.e. the
    memory horizon of every channel.

        A_log_eff = A_log + (alpha_a / r_a) B_a A_a
        A_a in R^{r_a x d_state}, B_a in R^{d_inner x r_a}

    With ``tie=True`` the basis ``A_a`` is shared by both scan directions and only
    ``B_a`` is per-direction; the pretrained ``A_log`` / ``A_b_log`` stay separate.
    """

    def __init__(self, r: int, alpha: float, d_inner: int, d_state: int,
                 tie: bool = True, gate: bool = True, name: str = "a_log"):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.tie = tie
        self.name = name
        self.lora_A = nn.Parameter(torch.empty(r, d_state))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(d_inner, r))
        if tie:
            self.lora_B_b = nn.Parameter(torch.zeros(d_inner, r))
        else:
            self.register_parameter("lora_B_b", None)
        self.gate = nn.Parameter(torch.zeros(())) if gate else None
        if self.gate is None:
            self.register_parameter("gate", None)
        if gate and tie:
            self.gate_b = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("gate_b", None)

    def delta(self, direction: str = "f") -> torch.Tensor:
        """(d_inner, d_state) additive correction of ``A_log``."""
        B = self.lora_B if (direction == "f" or not self.tie) else self.lora_B_b
        g = self.gate if (direction == "f" or not self.tie) else self.gate_b
        d = (B @ self.lora_A) * self.scaling
        if g is not None:
            d = d * torch.tanh(g)
        return d

    @torch.no_grad()
    def merge(self, a_log_f: torch.Tensor, a_log_b: Optional[torch.Tensor]) -> None:
        a_log_f += self.delta("f").to(a_log_f.dtype)
        if a_log_b is not None:
            a_log_b += self.delta("b").to(a_log_b.dtype)

    def extra_repr(self) -> str:
        return f"{self.name} r={self.r} d_state={self.lora_A.shape[1]} tie={self.tie}"


# --------------------------------------------------------------------------------------
# Placement configuration
# --------------------------------------------------------------------------------------
DEFAULT_TARGETS = ("in_proj", "out_proj", "dt_proj", "a_log")
ALL_TARGETS = ("in_proj", "out_proj", "x_proj", "dt_proj", "a_log", "conv1d")


@dataclass
class LoRAConfig:
    """Everything that defines *where* and *how much* capacity LoRA gets."""

    r: int = 8
    alpha: float = 16.0
    r_dt: Optional[int] = None        # Delta adapter rank   (default max(1, r // 2))
    alpha_dt: Optional[float] = None
    r_a: Optional[int] = None         # A adapter rank       (default r)
    alpha_a: Optional[float] = None
    dropout: float = 0.0
    targets: Sequence[str] = DEFAULT_TARGETS
    in_proj_slice: Optional[str] = None   # None | "x" | "z" | "xz"
    dt_tie: bool = True
    a_tie: bool = True
    gate: bool = True
    layers: Optional[str] = None      # "all" | "0-11" | "last6" | "first6"
    name: str = "fglora"

    def resolved(self) -> "LoRAConfig":
        if self.r_dt is None:
            self.r_dt = max(1, self.r // 2)
        if self.alpha_dt is None:
            self.alpha_dt = float(self.r_dt) * 2.0
        if self.r_a is None:
            self.r_a = self.r
        if self.alpha_a is None:
            self.alpha_a = float(self.r_a) * 2.0
        return self

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["targets"] = list(self.targets)
        return d


def parse_layer_spec(spec: Optional[str], depth: int) -> List[int]:
    """``None``/``all`` -> every layer; ``"0-11"`` -> [0..11]; ``"last4"`` -> last four."""
    if spec is None or str(spec).lower() in ("all", "*"):
        return list(range(depth))
    spec = str(spec).replace(" ", "")
    if spec.lower().startswith("last"):
        n = int(spec[4:])
        return list(range(depth - n, depth))
    if spec.lower().startswith("first"):
        n = int(spec[5:])
        return list(range(n))
    out: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(i for i in out if 0 <= i < depth))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class LoRAReport:
    names: List[str] = field(default_factory=list)
    by_kind: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"  {k:34s} {v:>10,d}" for k, v in sorted(self.by_kind.items())]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------------------
def apply_lora(model: nn.Module, cfg: LoRAConfig, verbose: bool = True) -> LoRAReport:
    """Freeze the backbone and insert LoRA adapters according to ``cfg``."""
    cfg = cfg.resolved()
    report = LoRAReport()

    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.layers
    layer_ids = parse_layer_spec(cfg.layers, len(layers))
    targets = set(cfg.targets)
    d_model = model.embed_dim

    for li in layer_ids:
        mixer: Mamba = layers[li].mixer
        prefix = f"layers.{li}.mixer"

        for tgt in ("in_proj", "out_proj", "x_proj"):
            if tgt not in targets:
                continue
            for suffix in ("", "_b"):
                name = f"{tgt}{suffix}"
                mod = getattr(mixer, name, None)
                if not isinstance(mod, nn.Linear):
                    continue
                out_slice, tag = None, ""
                if tgt == "in_proj" and cfg.in_proj_slice in ("x", "z"):
                    d_inner = mixer.d_inner
                    out_slice = (0, d_inner) if cfg.in_proj_slice == "x" else (d_inner, 2 * d_inner)
                    tag = "." + cfg.in_proj_slice
                elif tgt == "in_proj" and cfg.in_proj_slice not in (None, "xz"):
                    raise ValueError(f"in_proj_slice={cfg.in_proj_slice}")
                wrapped = LoRALinear(mod, cfg.r, cfg.alpha, cfg.dropout,
                                     name=f"{prefix}.{name}", out_slice=out_slice)
                setattr(mixer, name, wrapped)
                n = cfg.r * (mod.in_features + wrapped.lora_B.shape[0])
                report.names.append(f"{prefix}.{name}{tag}")
                report.by_kind[f"{tgt}{tag}"] = report.by_kind.get(f"{tgt}{tag}", 0) + n

        if "dt_proj" in targets:
            if cfg.dt_tie:
                wrapped = DtLoRA(mixer.dt_proj, mixer.dt_proj_b, cfg.r_dt, cfg.alpha_dt,
                                 tie=True, gate=cfg.gate, name=f"{prefix}.dt_proj")
                mixer.dt_proj = wrapped
                mixer.dt_proj_b = wrapped
                n = cfg.r_dt * (mixer.dt_rank + 2 * mixer.d_inner) + 2
                report.names.append(f"{prefix}.dt_proj[_b]")
                report.by_kind["dt_proj (tied)"] = report.by_kind.get("dt_proj (tied)", 0) + n
            else:
                for suffix in ("", "_b"):
                    name = f"dt_proj{suffix}"
                    mod = getattr(mixer, name)
                    wrapped = DtLoRA(mod, None, cfg.r_dt, cfg.alpha_dt, tie=False,
                                     gate=cfg.gate, name=f"{prefix}.{name}")
                    setattr(mixer, name, wrapped)
                    n = cfg.r_dt * (mixer.dt_rank + mixer.d_inner) + 1
                    report.names.append(f"{prefix}.{name}")
                    report.by_kind["dt_proj (separate)"] = (
                        report.by_kind.get("dt_proj (separate)", 0) + n
                    )

        if "a_log" in targets:
            adapter = ALogLoRA(
                cfg.r_a, cfg.alpha_a, mixer.d_inner, mixer.d_state,
                tie=cfg.a_tie, gate=cfg.gate, name=f"{prefix}.a_log",
            )
            mixer.a_lora = adapter
            n_b = 2 if cfg.a_tie else 1
            n = cfg.r_a * mixer.d_state + n_b * cfg.r_a * mixer.d_inner + n_b
            report.names.append(f"{prefix}.a_log")
            report.by_kind["a_log"] = report.by_kind.get("a_log", 0) + n

        if "conv1d" in targets:
            for suffix in ("", "_b"):
                name = f"conv1d{suffix}"
                mod = getattr(mixer, name, None)
                if not isinstance(mod, nn.Conv1d):
                    continue
                wrapped = LoRAConv1d(mod, cfg.r, cfg.alpha, name=f"{prefix}.{name}",
                                     gate=cfg.gate)
                setattr(mixer, name, wrapped)
                report.names.append(f"{prefix}.{name}")
                n = cfg.r * (mod.out_channels + mod.kernel_size[0]) + 1
                report.by_kind["conv1d"] = report.by_kind.get("conv1d", 0) + n

        mixer.lora_active = True

    # the regression head is trained in both conditions (assignment 1.3)
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        for p in model.head.parameters():
            p.requires_grad_(True)

    if verbose:
        tot, tr = count_parameters(model)
        print(f"[lora] adapters on {len(report.names)} modules / {len(layer_ids)} layers "
              f"(targets={sorted(targets)}, r={cfg.r}, r_dt={cfg.r_dt}, r_a={cfg.r_a})")
        print(report.summary())
        print(f"[lora] trainable {tr:,} / {tot:,} ({100.0 * tr / tot:.3f}%)")
    return report


class _IdentityLinear(nn.Module):  # pragma: no cover - reserved for future use
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features


# --------------------------------------------------------------------------------------
# Merge / unmerge / checkpoints
# --------------------------------------------------------------------------------------
def iter_lora_modules(model: nn.Module) -> Iterable[nn.Module]:
    seen = set()
    for m in model.modules():
        if isinstance(m, (LoRALinear, LoRAConv1d, DtLoRA, ALogLoRA)) and id(m) not in seen:
            seen.add(id(m))
            yield m


def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into the frozen base weights (in place).  Returns #modules."""
    n = 0
    for layer in model.layers:
        mixer = layer.mixer
        if getattr(mixer, "a_lora", None) is not None:
            mixer.a_lora.merge(mixer.A_log.data, mixer.A_b_log.data)
            n += 1
    for m in iter_lora_modules(model):
        if isinstance(m, ALogLoRA):
            continue  # already handled above (needs access to the parameters)
        m.merge()
        n += 1
    return n


def unmerge_lora(model: nn.Module) -> int:
    n = 0
    for layer in model.layers:
        mixer = layer.mixer
        if getattr(mixer, "a_lora", None) is not None:
            mixer.A_log.data -= mixer.a_lora.delta("f")
            mixer.A_b_log.data -= mixer.a_lora.delta("b")
            n += 1
    for m in iter_lora_modules(model):
        if isinstance(m, ALogLoRA):
            continue
        m.unmerge()
        n += 1
    return n


def lora_state_dict(model: nn.Module) -> dict:
    """Only the adapter tensors + the head: this is what a LoRA checkpoint stores."""
    out = {}
    for k, v in model.state_dict().items():
        if ".lora_" in k or ".gate" in k or k.startswith("head."):
            out[k] = v
    return out


def merge_and_strip(model: nn.Module) -> nn.Module:
    """Merge adapters, then replace the wrappers by their plain base layers.

    Afterwards the module has exactly the structure (and speed) of the original
    VideoMamba - the correct setting for the "inference after merge" measurement.
    """
    merge_lora(model)
    linear_attrs = ("in_proj", "in_proj_b", "out_proj", "x_proj", "x_proj_b")
    conv_attrs = ("conv1d", "conv1d_b")
    for layer in model.layers:
        mixer = layer.mixer
        for a in linear_attrs:
            m = getattr(mixer, a, None)
            if isinstance(m, LoRALinear):
                setattr(mixer, a, m.base)
        if isinstance(getattr(mixer, "dt_proj", None), DtLoRA):
            d = mixer.dt_proj
            mixer.dt_proj = d.base_f
            mixer.dt_proj_b = d.base_b if d.has_b else d.base_f
        elif isinstance(getattr(mixer, "dt_proj_b", None), DtLoRA):
            mixer.dt_proj_b = mixer.dt_proj_b.base_f
        for a in conv_attrs:
            m = getattr(mixer, a, None)
            if isinstance(m, LoRAConv1d):
                setattr(mixer, a, m.base)
        mixer.a_lora = None
        mixer.lora_active = False
    return model


def gate_report(model: nn.Module) -> List[dict]:
    """Per-layer learned gate values (how much each layer re-timed itself)."""
    rows = []
    for i, layer in enumerate(model.layers):
        mixer = layer.mixer
        row = {"layer": i}
        if getattr(mixer, "a_lora", None) is not None and mixer.a_lora.gate is not None:
            row["a_gate_f"] = float(torch.tanh(mixer.a_lora.gate))
            if mixer.a_lora.gate_b is not None:
                row["a_gate_b"] = float(torch.tanh(mixer.a_lora.gate_b))
        if isinstance(mixer.dt_proj, DtLoRA) and mixer.dt_proj.gate is not None:
            row["dt_gate_f"] = float(torch.tanh(mixer.dt_proj.gate))
            if mixer.dt_proj.gate_b is not None:
                row["dt_gate_b"] = float(torch.tanh(mixer.dt_proj.gate_b))
        if getattr(mixer, "a_lora", None) is not None:
            row["a_delta_norm"] = float(mixer.a_lora.delta("f").norm())
        rows.append(row)
    return rows
