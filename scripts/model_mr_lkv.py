#!/usr/bin/env python3
"""
MR-LKV with cross-stage skip connections (U-Net style) and test-time augmentation (TTA) hooks.
- Encoder: patch_embed + 4 multi-block stages, each followed by downsample except last.
- Skips: outputs of each stage before downsample.
- Decoder: upsample & fuse skips via concatenation and 1*1 convs.
- TTA: flips & transpose transforms in tta_predict.
"""
from __future__ import annotations
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# LayerNorm helper & norm getter
# -----------------------------
class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm (ConvNeXt style)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        sigma = ((x - mu) ** 2).mean(1, keepdim=True)
        x_norm = (x - mu) / torch.sqrt(sigma + self.eps)
        return self.weight[:, None, None] * x_norm + self.bias[:, None, None]


def get_norm(norm: str, C: int) -> nn.Module:
    n = norm.lower()
    if n == 'bn': return nn.BatchNorm2d(C)
    if n == 'ln': return LayerNorm2d(C)
    if n == 'gn': return nn.GroupNorm(1, C)
    return nn.Identity()

# -----------------------------
# Fourier Unit and FEB
# -----------------------------
class FourierUnit(nn.Module):
    """FFT -> channel mixing -> iFFT."""
    def __init__(self, C: int):
        super().__init__()
        self.mix_real = nn.Conv2d(C, C, 1)
        self.mix_imag = nn.Conv2d(C, C, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
        real = self.mix_real(fft.real)
        imag = self.mix_imag(fft.imag)
        return torch.fft.ifftn(torch.complex(real, imag), dim=(-2, -1), norm='ortho').real

class FEB(nn.Module):
    """Frequency-Enhanced Branch."""
    def __init__(self, C: int, norm: str = 'bn'):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 1)
        self.funit = FourierUnit(C)
        self.conv2 = nn.Conv2d(C, C, 1)
        self.norm = get_norm(norm, C)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.funit(y)
        y = self.conv2(y)
        return self.act(self.norm(y) + x)

# -----------------------------
# Residual Large-Kernel Block
# -----------------------------
class RLKB(nn.Module):
    def __init__(self, C: int, K: int, dilation: int = 1, depthwise: bool = True, norm: str = 'bn'):
        super().__init__()
        groups = C if depthwise else 1
        padding = dilation * (K - 1) // 2
        self.conv = nn.Conv2d(C, C, K, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.pw = nn.Conv2d(C, C, 1)
        self.norm = get_norm(norm, C)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.pw(y)
        return self.act(self.norm(y) + x)

# -----------------------------
# Channel Attention
# -----------------------------
class CA(nn.Module):
    def __init__(self, C: int, r: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, max(C//r,1)), nn.GELU(), nn.Linear(max(C//r,1), C), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.avg(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

# -----------------------------
# MR-LKV with U-Net Skip + TTA
# -----------------------------
class MR_LKV(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        C0: int = 32,
        depths: Sequence[int] = (1,1,1,1),
        kernels: Sequence[int] = (31,51,71,91),
        dilations: Sequence[int] = (1,1,1,1),
        depthwise: bool = True,
        norm: str = 'bn',
        **kwargs,
    ):
        super().__init__()
        # legacy kwargs
        if 'in_channels' in kwargs: in_ch = kwargs.pop('in_channels')
        if 'base_channels' in kwargs: C0 = kwargs.pop('base_channels')
        if 'norm_type' in kwargs:
            nt = kwargs.pop('norm_type'); norm = {'batch':'bn','instance':'gn','none':'id'}.get(nt, nt)
        kwargs.pop('use_decoder', None); kwargs.pop('final_activation', None)
        if kwargs: raise TypeError(f"Unexpected kwargs: {kwargs}")
        assert len(depths)==4 and len(kernels)==4 and len(dilations)==4
        # channel dims
        Cs = [C0*(2**i) for i in range(4)]  # [C0,2C0,4C0,8C0]
        # encoder
        self.patch_embed = nn.Conv2d(in_ch, Cs[0], 3, stride=2, padding=1)
        self.stage0 = nn.Sequential(*[RLKB(Cs[0], kernels[0], dilation=dilations[0], depthwise=depthwise, norm=norm) for _ in range(depths[0])])
        self.down0  = nn.Conv2d(Cs[0], Cs[1], 2, stride=2)
        self.stage1 = nn.Sequential(*[RLKB(Cs[1], kernels[1], dilation=dilations[1], depthwise=depthwise, norm=norm) for _ in range(depths[1])])
        self.down1  = nn.Conv2d(Cs[1], Cs[2], 2, stride=2)
        self.stage2 = nn.Sequential(*[RLKB(Cs[2], kernels[2], dilation=dilations[2], depthwise=depthwise, norm=norm) for _ in range(depths[2])])
        self.down2  = nn.Conv2d(Cs[2], Cs[3], 2, stride=2)
        self.stage3 = nn.Sequential(*[RLKB(Cs[3], kernels[3], dilation=dilations[3], depthwise=depthwise, norm=norm) for _ in range(depths[3])])
        # frequency & attention branches
        self.freq0, self.attn0 = FEB(Cs[0], norm), CA(Cs[0])
        self.freq1, self.attn1 = FEB(Cs[1], norm), CA(Cs[1])
        self.freq2, self.attn2 = FEB(Cs[2], norm), CA(Cs[2])
        self.freq3, self.attn3 = FEB(Cs[3], norm), CA(Cs[3])
        # decoder fuses
        self.dec2 = nn.Conv2d(Cs[3]+Cs[2], Cs[2], 1)
        self.dec1 = nn.Conv2d(Cs[2]+Cs[1], Cs[1], 1)
        self.dec0 = nn.Conv2d(Cs[1]+Cs[0], Cs[0], 1)
        self.final_conv = nn.Conv2d(Cs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        # encoder + skips
        x0 = self.patch_embed(x)
        f0 = self.freq0(x0);  a0 = self.attn0(f0);
        s0 = self.stage0(a0); e0 = self.down0(s0)
        f1 = self.freq1(e0);  a1 = self.attn1(f1);
        s1 = self.stage1(a1); e1 = self.down1(s1)
        f2 = self.freq2(e1);  a2 = self.attn2(f2);
        s2 = self.stage2(a2); e2 = self.down2(s2)
        f3 = self.freq3(e2);  a3 = self.attn3(f3);
        e3 = self.stage3(a3)
        # decoder
        d2 = F.interpolate(e3, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = F.interpolate(d2, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        d0 = F.interpolate(d1, size=s0.shape[-2:], mode='bilinear', align_corners=False)
        d0 = self.dec0(torch.cat([d0, s0], dim=1))
        out = F.interpolate(d0, size=(H, W), mode='bilinear', align_corners=False)
        return self.final_conv(out)

    def tta_predict(self, x: torch.Tensor) -> torch.Tensor:
        funcs = [
            (lambda y: y, lambda y: y),
            (lambda y: torch.flip(y, [-1]), lambda y: torch.flip(y, [-1])),  # horiz
            (lambda y: torch.flip(y, [-2]), lambda y: torch.flip(y, [-2])),  # vert
            (lambda y: y.transpose(-2, -1), lambda y: y.transpose(-2, -1)),  # swap H/W
        ]
        preds = []
        for tf, inv in funcs:
            p = self.forward(tf(x))
            preds.append(inv(p))
        return torch.stack(preds).mean(0)

# -----------------------------
# Utility: count parameters
# -----------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
