"""VideoMamba encoder with an EF regression head, plus a checkpoint loader.

The released VideoMamba weights are ImageNet/Kinetics-400 classification models with a
1000/400-way head, a fixed 224x224x8-frame input grid, and a PatchEmbed whose tubelet
size differs between checkpoints (``kernel_size`` 1, 2 or 4).  This module

* infers the architecture from the checkpoint itself (depth, width, ``d_state``,
  ``dt_rank``, tubelet size),
* resizes ``pos_embed`` and ``temporal_pos_embedding`` to the resolution / clip length
  actually used here,
* drops the classification head and installs a 1-unit regression head.

Nothing else about the backbone is changed, so the two experimental conditions
(full fine-tuning vs. LoRA) start from *bit-identical* weights.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import Block, Mamba, _RMSNormImpl, _FUSED_LN, _FUSED_RMS


# --------------------------------------------------------------------------------------
VARIANT_CFG = {
    "tiny":   dict(embed_dim=192, depth=24),
    "small":  dict(embed_dim=384, depth=24),
    "middle": dict(embed_dim=576, depth=32),
}

# The checkpoints we support out of the box (HuggingFace: OpenGVLab/VideoMamba).
CKPT_ZOO = {
    "videomamba_t16_in1k":           dict(variant="tiny",   tag="in1k"),
    "videomamba_s16_in1k":           dict(variant="small",  tag="in1k"),
    "videomamba_m16_in1k":           dict(variant="middle", tag="in1k"),
    "videomamba_t16_k400_f8":        dict(variant="tiny",   tag="k400"),
    "videomamba_t16_k400_f16":       dict(variant="tiny",   tag="k400"),
    "videomamba_t16_k400_f32":       dict(variant="tiny",   tag="k400"),
    "videomamba_s16_k400_f8":        dict(variant="small",  tag="k400"),
    "videomamba_s16_k400_f16":       dict(variant="small",  tag="k400"),
    "videomamba_s16_k400_f32":       dict(variant="small",  tag="k400"),
    "videomamba_m16_k400_f8":        dict(variant="middle", tag="k400"),
    "videomamba_m16_k400_f16":       dict(variant="middle", tag="k400"),
    "videomamba_m16_k400_f32":       dict(variant="middle", tag="k400"),
    "videomamba_s16_ssv2_f16":       dict(variant="small",  tag="ssv2"),
    "videomamba_m16_ssv2_f16":       dict(variant="middle", tag="ssv2"),
}


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Timm-compatible truncated normal init (avoid a hard ``timm`` dependency)."""
    with torch.no_grad():
        if not torch.isfinite(tensor).all():
            pass
        tensor.normal_(mean, std)
        if a is not None and b is not None:
            while True:
                bad = (tensor < a) | (tensor > b)
                if not bad.any():
                    break
                tensor[bad] = tensor[bad].normal_(mean, std)
    return tensor


class PatchEmbed(nn.Module):
    """Tubelet embedding: (B, 3, T, H, W) -> (B, C, T/k, H/p, W/p)."""

    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = kernel_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x):
        return self.proj(x)


class VideoMambaEF(nn.Module):
    """VideoMamba (bidirectional Mamba) + linear EF regression head."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        depth: int = 24,
        embed_dim: int = 384,
        channels: int = 3,
        kernel_size: int = 1,
        num_frames: int = 16,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        ssm_cfg: Optional[dict] = None,
        norm_epsilon: float = 1e-5,
        fused_add_norm: bool = True,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        bimamba: bool = True,
        head_drop_rate: float = 0.0,
        head_bias: float = 0.0,
        use_fast_path: bool = True,
        num_classes: int = 1,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.temporal_pos_embedding = nn.Parameter(
            torch.zeros(1, num_frames // kernel_size, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr

        self.layers = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    mixer_cls=_make_mixer(ssm_cfg, bimamba, use_fast_path),
                    norm_cls=_make_norm(rms_norm, norm_epsilon),
                    drop_path=inter_dpr[i],
                    fused_add_norm=fused_add_norm,
                    residual_in_fp32=residual_in_fp32,
                )
                for i in range(depth)
            ]
        )
        for i, blk in enumerate(self.layers):
            blk.layer_idx = i

        self.norm_f = (_RMSNormImpl if rms_norm else nn.LayerNorm)(embed_dim, eps=norm_epsilon)
        self.head_drop = nn.Dropout(head_drop_rate) if head_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.temporal_pos_embedding, std=0.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            nn.init.constant_(self.head.bias, head_bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # -- forward -----------------------------------------------------------------------
    def forward_features(self, x):
        """x: (B, 3, T, H, W) -> (B, C) class-token feature."""
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # temporal position embedding: (b t) n m -> (b n) t m -> b (t n) m
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        n_tok = x.shape[1]                                            # H*W
        x = x.reshape(B, T, n_tok, C).permute(0, 2, 1, 3).reshape(B * n_tok, T, C)
        x = x + self.temporal_pos_embedding
        x = x.reshape(B, n_tok, T, C).permute(0, 2, 1, 3).reshape(B, T * n_tok, C)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = _FUSED_RMS if isinstance(self.norm_f, _RMSNormImpl) else _FUSED_LN
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                getattr(self.norm_f, "bias", None),
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states[:, 0, :]

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(self.head_drop(feat)).squeeze(-1)


def _make_mixer(ssm_cfg, bimamba, use_fast_path):
    cfg = dict(ssm_cfg or {})
    cfg.pop("bimamba", None)
    cfg.pop("use_fast_path", None)
    return lambda dim: Mamba(dim, bimamba=bimamba, use_fast_path=use_fast_path, **cfg)


def _make_norm(rms_norm, eps):
    if rms_norm:
        return lambda dim: _RMSNormImpl(dim, eps=eps)
    return lambda dim: nn.LayerNorm(dim, eps=eps)


# --------------------------------------------------------------------------------------
# Architecture inference + checkpoint loading
# --------------------------------------------------------------------------------------
def infer_arch(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Read depth / width / ssm hyper-parameters straight off the tensors."""
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

    n_layers = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("layers."))
    embed_dim = state_dict["pos_embed"].shape[-1]
    pos_tokens = state_dict["pos_embed"].shape[1] - 1
    grid = int(round(math.sqrt(pos_tokens)))
    proj_w = state_dict["patch_embed.proj.weight"]
    if proj_w.ndim == 5:
        kernel, patch = proj_w.shape[2], proj_w.shape[-1]
    else:  # 2D patch embedding from an ImageNet checkpoint
        kernel, patch = 1, proj_w.shape[-1]
    dt_rank = state_dict["layers.0.mixer.dt_proj.weight"].shape[1]
    d_state = state_dict["layers.0.mixer.A_log"].shape[1]
    tpe = state_dict["temporal_pos_embedding"].shape[1]
    return dict(
        depth=n_layers, embed_dim=embed_dim, img_size=grid * patch, patch_size=patch,
        kernel_size=kernel, num_frames=tpe * kernel, dt_rank=dt_rank, d_state=d_state,
    )


def _interp_pos_embed(pos: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Resize the (1, 1+grid^2, C) spatial position embedding to a new grid."""
    if pos.shape[1] - 1 == num_patches:
        return pos
    dim = pos.shape[-1]
    g_src = int(round(math.sqrt(pos.shape[1] - 1)))
    g_dst = int(round(math.sqrt(num_patches)))
    cls, grid = pos[:, :1], pos[:, 1:]
    grid = grid.reshape(1, g_src, g_src, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(g_dst, g_dst), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, g_dst * g_dst, dim)
    return torch.cat([cls, grid], dim=1)


def _interp_temporal(pos: torch.Tensor, target_len: int) -> torch.Tensor:
    if pos.shape[1] == target_len:
        return pos
    dim = pos.shape[-1]
    x = pos.transpose(1, 2)  # (1, C, T)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.transpose(1, 2)


def load_videomamba_weights(
    model: VideoMambaEF,
    ckpt_path: str,
    verbose: bool = True,
) -> Dict[str, list]:
    """Load released VideoMamba weights into ``model`` (head is discarded).

    Returns a report with the lists of loaded / interpolated / missing / unexpected keys.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    tgt = model.state_dict()
    report = {"loaded": [], "interpolated": [], "missing": [], "unexpected": []}

    for k, v in sd.items():
        if k not in tgt:
            report["unexpected"].append(k)
            continue
        if k == "head.weight" or k == "head.bias":
            report["unexpected"].append(k)
            continue
        if v.shape == tgt[k].shape:
            report["loaded"].append(k)
            continue
        if k == "pos_embed":
            tgt[k].copy_(_interp_pos_embed(v.float(), tgt[k].shape[1] - 1))
            report["interpolated"].append(f"{k} {tuple(v.shape)}->{tuple(tgt[k].shape)}")
        elif k == "temporal_pos_embedding":
            tgt[k].copy_(_interp_temporal(v.float(), tgt[k].shape[1]))
            report["interpolated"].append(f"{k} {tuple(v.shape)}->{tuple(tgt[k].shape)}")
        elif k == "patch_embed.proj.weight" and v.ndim == 4:
            # 2D patch embedding (ImageNet checkpoint) -> inflate over the time axis,
            # keeping the central frame ("inflate_weight(center=True)" from the repo).
            t = tgt[k].shape[2]
            w3 = torch.zeros_like(tgt[k])
            w3[:, :, t // 2] = v.float()
            tgt[k].copy_(w3)
            report["interpolated"].append(f"{k} inflated 2D->3D (T={t}, centred)")
        else:
            report["missing"].append(f"{k} (shape {tuple(v.shape)} vs {tuple(tgt[k].shape)})")

    for k in tgt:
        if k not in sd and k not in report["missing"] and not k.startswith("head."):
            report["missing"].append(k)

    model.load_state_dict(tgt, strict=True)

    if verbose:
        n_tot = len(tgt)
        print(f"[videomamba] loaded {len(report['loaded'])}/{n_tot} tensors from "
              f"{os.path.basename(ckpt_path)}")
        if report["interpolated"]:
            print(f"[videomamba] {len(report['interpolated'])} tensors resized/re-initialised:")
            for s in report["interpolated"]:
                print("    ", s)
        if report["missing"]:
            print(f"[videomamba] WARNING: {len(report['missing'])} tensors missing in the "
                  f"checkpoint (left at init): {report['missing'][:6]}")
    return report


def build_videomamba_ef(
    ckpt_path: str,
    num_frames: int = 16,
    img_size: Optional[int] = None,
    drop_path_rate: float = 0.1,
    head_drop_rate: float = 0.0,
    head_bias: float = 0.0,
    use_fast_path: bool = True,
    ssm_cfg: Optional[dict] = None,
) -> VideoMambaEF:
    """Build the model with the architecture implied by ``ckpt_path`` and load it."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    arch = infer_arch(raw)
    del raw

    img_size = img_size or 224
    model = VideoMambaEF(
        img_size=img_size,
        patch_size=arch["patch_size"],
        depth=arch["depth"],
        embed_dim=arch["embed_dim"],
        kernel_size=arch["kernel_size"],
        num_frames=num_frames,
        drop_path_rate=drop_path_rate,
        head_drop_rate=head_drop_rate,
        head_bias=head_bias,
        use_fast_path=use_fast_path,
        ssm_cfg=ssm_cfg,
        bimamba=True,
    )
    load_videomamba_weights(model, ckpt_path)
    return model
