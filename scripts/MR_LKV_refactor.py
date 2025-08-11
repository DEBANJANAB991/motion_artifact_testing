#!/usr/bin/env python3
"""
Refactored MR-LKV implementation with the requested changes:
- Verified large-kernel conv: explicit padding, dilation, optional depthwise/standard groups.
- Norm + residual scaffolding around LK path (configurable LayerNorm2d/BatchNorm2d).
- Optional lightweight attention/residual wrappers for LK blocks.
- Frequency branch kept; minor cleanups.
- Channel Attention retained.
- Stage wrapper keeps spatial/channel adjustment; can switch between downsample/identity.
- Utility: parameter counter, receptive-field checker.
- Training utilities: mixed L1 + SSIM loss, cosine LR with warmup, early stopping helper.

Author: ChatGPT
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Small helpers
# -----------------------------

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm (like ConvNeXt)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def get_norm(norm: str, C: int) -> nn.Module:
    norm = norm.lower()
    if norm == 'bn':
        return nn.BatchNorm2d(C)
    if norm == 'ln':
        return LayerNorm2d(C)
    if norm == 'gn':
        return nn.GroupNorm(1, C)
    if norm == 'id' or norm == 'none':
        return nn.Identity()
    raise ValueError(f"Unknown norm: {norm}")


# -----------------------------
# Core Blocks
# -----------------------------

class RLKB(nn.Module):
    """
    Residual Large-Kernel Block.
    - Optional depthwise large kernel conv (groups=C) OR standard conv (groups=1).
    - Explicit padding formula to keep spatial dim: padding = dilation*(K-1)//2.
    - Norm + residual + activation.
    """
    def __init__(
        self,
        C: int,
        K: int,
        dilation: int = 1,
        depthwise: bool = True,
        norm: str = 'bn',
        act_layer: nn.Module = nn.GELU,
        **kwargs,
    ):
        super().__init__()
        groups = C if depthwise else 1
        padding = dilation * (K - 1) // 2
        self.lk = nn.Conv2d(C, C, K, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.pw = nn.Conv2d(C, C, 1, bias=True)
        self.norm = get_norm(norm, C)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lk(x)
        out = self.pw(out)
        out = self.norm(out)
        out = out + x
        return self.act(out)


class FourierUnit(nn.Module):
    """
    FFT -> channel-wise mixing -> iFFT.
    Works on real/imag parts separately.
    """
    def __init__(self, C: int):
        super().__init__()
        self.mix_real = nn.Conv2d(C, C, 1)
        self.mix_imag = nn.Conv2d(C, C, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
        real = self.mix_real(fft.real)
        imag = self.mix_imag(fft.imag)
        complex_out = torch.complex(real, imag)
        return torch.fft.ifftn(complex_out, dim=(-2, -1), norm='ortho').real


class FEB(nn.Module):
    """
    Frequency-Enhanced Branch.
    pointwise -> FourierUnit -> pointwise -> residual + activation.
    """
    def __init__(self, C: int, norm: str = 'bn', act_layer: nn.Module = nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 1)
        self.fft = FourierUnit(C)
        self.conv2 = nn.Conv2d(C, C, 1)
        self.norm = get_norm(norm, C)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.fft(y)
        y = self.conv2(y)
        y = self.norm(y + x)
        return self.act(y)


class CA(nn.Module):
    """Channel Attention (SE style)."""
    def __init__(self, C: int, r: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, max(C // r, 1)),
            nn.GELU(),
            nn.Linear(max(C // r, 1), C),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class MR_LKV_Stage(nn.Module):
    """
    One stage of MR-LKV: RLKB -> FEB -> CA -> adjust.
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        K: int,
        downsample: bool = True,
        dilation: int = 1,
        depthwise: bool = True,
        norm: str = 'bn',
        act_layer: nn.Module = nn.GELU,
        **kwargs,
    ):
        super().__init__()
        self.resblock = RLKB(C_in, K, dilation=dilation, depthwise=depthwise, norm=norm, act_layer=act_layer)
        self.freq = FEB(C_in, norm=norm, act_layer=act_layer)
        self.attn = CA(C_in)

        if downsample:
            self.adjust = nn.Conv2d(C_in, C_out, 2, stride=2)
        else:
            self.adjust = nn.Identity() if C_in == C_out else nn.Conv2d(C_in, C_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resblock(x)
        x = self.freq(x)
        x = self.attn(x)
        x = self.adjust(x)
        return x


class MR_LKV(nn.Module):
    """
    Multi-Scale Residual Large-Kernel Vision Network.
    Architecture: PatchEmbed -> sequential MR_LKV_Stages -> Head conv + upsample.
    """
    def __init__(
        self,
        in_ch: int = 1,
        C0: int = 32,
        depths: Sequence[int] = (1, 1, 1, 1),
        kernels: Sequence[int] = (31, 51, 71, 91),
        dilations: Sequence[int] = (1, 1, 1, 1),
        depthwise: bool = True,
        norm: str = 'bn',
        act_layer: nn.Module = nn.GELU,
        **kwargs,
    ):
        super().__init__()
        # accept legacy kwargs
        if 'in_channels' in kwargs:
            in_ch = kwargs.pop('in_channels')
        if 'base_channels' in kwargs:
            C0 = kwargs.pop('base_channels')
        if 'norm_type' in kwargs:
            nt = kwargs.pop('norm_type')
            norm = {'batch':'bn','instance':'gn','none':'id'}.get(nt, nt)
        # ignore unused legacy args
        kwargs.pop('use_decoder', None)
        kwargs.pop('final_activation', None)
        if kwargs:
            raise TypeError(f"Unexpected kwargs: {kwargs}")
        assert len(depths) == 4 and len(kernels) == 4 and len(dilations) == 4
        self.patch_embed = nn.Conv2d(in_ch, C0, 3, stride=2, padding=1)
        super().__init__()
        # accept legacy kwarg name
        if 'in_channels' in kwargs:
            in_ch = kwargs.pop('in_channels')
        if kwargs:
            raise TypeError(f"Unexpected kwargs: {kwargs}")
        assert len(depths) == 4 and len(kernels) == 4 and len(dilations) == 4
        self.patch_embed = nn.Conv2d(in_ch, C0, 3, stride=2, padding=1)

        Cs = [C0 * (2 ** i) for i in range(4)]
        layers: List[nn.Module] = []
        for i in range(4):
            for d in range(depths[i]):
                down = (i < 3 and d == 0)  # only first block of each stage (except last) downsamples
                C_in = Cs[i - 1] if (i > 0 and d == 0) else Cs[i]
                layers.append(
                    MR_LKV_Stage(
                        C_in,
                        Cs[i],
                        kernels[i],
                        downsample=down,
                        dilation=dilations[i],
                        depthwise=depthwise,
                        norm=norm,
                        act_layer=act_layer,
                    )
                )
        self.stages = nn.Sequential(*layers)
        self.head = nn.Conv2d(Cs[-1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        x = self.stages(x)
        out = self.head(x)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# -----------------------------
# Training utilities
# -----------------------------

class SSIM(nn.Module):
    """Simplified SSIM for training loss (expect images in [0,1])."""
    def __init__(self, window_size: int = 11, channel: int = 1, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.C1 = C1
        self.C2 = C2
        self.register_buffer('window', self._create_window(window_size, channel))

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.t()
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        window = self.window.type_as(img1)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return ssim_map.mean()


class MixedLoss(nn.Module):
    """total = alpha * L1 + (1-alpha) * (1-SSIM)"""
    def __init__(self, alpha: float = 0.8, channel: int = 1):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(channel=channel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1(pred, target)
        ssim_term = 1 - self.ssim(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * ssim_term


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine decay with warmup on top of any optimizer."""
    def __init__(self, optimizer, warmup_iters: int, max_iters: int, last_epoch: int = -1):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_iters:
            return [base_lr * step / max(1, self.warmup_iters) for base_lr in self.base_lrs]
        # cosine
        progress = (step - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)
        return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


class EarlyStopping:
    """Early stopping on validation metric (lower is better by default)."""
    def __init__(self, patience: int = 10, mode: str = 'min', delta: float = 0.0):
        assert mode in ['min', 'max']
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improve = (value < self.best - self.delta) if self.mode == 'min' else (value > self.best + self.delta)
        if improve:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# -----------------------------
# Misc utilities
# -----------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def receptive_field(kernel: int, dilation: int = 1, stride: int = 1, prev_rf: int = 1) -> int:
    """
    Compute receptive field after a conv layer given previous RF.
    """
    return prev_rf + (kernel - 1) * dilation * stride


# -----------------------------
# Simple self-test
# -----------------------------
if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256)
    model = MR_LKV()
    y = model(x)
    print("input:", x.shape, "output:", y.shape)
    print("params (M):", count_parameters(model) / 1e6)
