#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class RLKB(nn.Module):
    """
    Residual Large-Kernel Block: depthwise large-kernel conv + pointwise + skip.
    """
    def __init__(self, C, K):
        super().__init__()
        self.dw = nn.Conv2d(C, C, K, padding=K//2, groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, 1)
        self.norm = nn.BatchNorm2d(C)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        out = self.norm(out)
        return self.act(out + x)

class FourierUnit(nn.Module):
    """
    FFT, channel-wise mixing, inverse FFT
    """
    def __init__(self, C):
        super().__init__()
        self.mix_real = nn.Conv2d(C, C, 1)
        self.mix_imag = nn.Conv2d(C, C, 1)

    def forward(self, x):
      
        fft = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')  # complex tensor
        real = fft.real
        imag = fft.imag
        real = self.mix_real(real)
        imag = self.mix_imag(imag)
        complex_out = torch.complex(real, imag)
        return torch.fft.ifftn(complex_out, dim=(-2, -1), norm='ortho').real

class FEB(nn.Module):
    """
    Frequency-Enhanced Branch: pointwise conv, FourierUnit, pointwise conv, residual.
    """
    def __init__(self, C):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 1)
        self.fft = FourierUnit(C)
        self.conv2 = nn.Conv2d(C, C, 1)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.fft(y)
        y = self.conv2(y)
        return self.act(y + x)

class CA(nn.Module):
    """
    Channel Attention: squeeze-and-excitation style.
    """
    def __init__(self, C, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, C//r),
            nn.GELU(),
            nn.Linear(C//r, C),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class MR_LKV_Stage(nn.Module):
    """
    One stage of MR-LKV: RLKB -> FEB -> CA -> channel/possibly spatial adjustment
    """
    def __init__(self, C_in, C_out, K, downsample=True):
        super().__init__()
        self.resblock = RLKB(C_in, K)
        self.freq     = FEB(C_in)
        self.attn     = CA(C_in)
        # spatial/channel adjustment
        if downsample:
            # halve H×W and map C_in→C_out
            self.adjust = nn.Conv2d(C_in, C_out, 2, stride=2)
        else:
            # preserve H×W but adjust channels if needed
            if C_in != C_out:
                self.adjust = nn.Conv2d(C_in, C_out, 1)
            else:
                self.adjust = nn.Identity()

    def forward(self, x):
        x = self.resblock(x)
        x = self.freq(x)
        x = self.attn(x)
        return self.adjust(x)

class MR_LKV(nn.Module):
    """
    Multi-Scale Residual Large-Kernel Vision Network.
    Architecture: PatchEmbed -> sequential MR_LKV_Stages -> Head conv + upsample
    """
    def __init__(self, C0=32, depths=[1,1,1,1], kernels=[31,51,71,91]):
        super().__init__()
        # Patch embedding: increase channels and downsample by 2
        self.patch_embed = nn.Conv2d(1, C0, 3, stride=2, padding=1)
        # channel sizes per stage
        Cs = [C0 * (2**i) for i in range(4)]  # [32, 64, 128, 256]
        layers = []
        for i in range(4):
            for d in range(depths[i]):
                # downsample only on first block of stages 0-2
                down = (i < 3 and d == 0)
                C_in = Cs[i-1] if (i>0 and d==0) else Cs[i]
                layers.append(MR_LKV_Stage(C_in, Cs[i], kernels[i], downsample=down))
        self.stages = nn.Sequential(*layers)
        # Final head to reduce channels to 1
        self.head = nn.Conv2d(Cs[-1], 1, 1)

    def forward(self, x):
        # remember original H×W
        H, W = x.shape[-2], x.shape[-1]
        x = self.patch_embed(x)
        x = self.stages(x)
        out = self.head(x)
        # upsample back to original spatial size
        out = F.interpolate(
            out,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        return out
