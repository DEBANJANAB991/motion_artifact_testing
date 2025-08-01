#!/usr/bin/env python3
"""
UNet.py — Standard 2D U-Net with:
  • Optional Batch/InstanceNorm
  • Optional dropout in bottleneck
  • Xavier init for conv layers
  • Safe skip concatenation via center-crop (handles odd sizes)
  • Optional final activation (e.g. sigmoid)

Author: <your name>
"""
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Helpers
# ----------------------------

def init_weights(m: nn.Module) -> None:
    """Xavier-uniform init for Conv/Deconv weights, zero bias."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def center_crop(t: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Crop tensor `t` (N,C,H,W) at center to match spatial size `target_hw`.
    Assumes target size <= current size.
    """
    _, _, h, w = t.shape
    th, tw = target_hw
    dh = (h - th) // 2
    dw = (w - tw) // 2
    return t[:, :, dh:dh + th, dw:dw + tw]


class ConvBlock(nn.Module):
    """Two Conv(3x3)->Norm(optional)->ReLU blocks with optional dropout."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: Optional[Literal['batch', 'instance']] = 'batch',
        dropout: float = 0.0
    ):
        super().__init__()
        layers = []
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if norm_type == 'batch':
                layers.append(nn.BatchNorm2d(out_channels))
            elif norm_type == 'instance':
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Flexible U-Net architecture with safe skip connections.

    Args:
        in_channels:   Input channels.
        base_channels: #channels of first encoder level.
        levels:        Encoder/decoder depth (>=1).
        norm_type:     'batch', 'instance', or None.
        dropout_bottleneck: Dropout prob in bottleneck block.
        final_activation: 'sigmoid' or None. (Add others if needed.)
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        levels: int = 4,
        norm_type: Optional[Literal['batch', 'instance']] = 'batch',
        dropout_bottleneck: float = 0.0,
        final_activation: Optional[Literal['sigmoid', 'none']] = None,
    ):
        super().__init__()
        assert levels >= 1, "levels must be >= 1"

        self.final_activation = final_activation if final_activation != 'none' else None
        self.pool = nn.MaxPool2d(2)

        # ----- Encoder -----
        encs = []
        prev_ch = in_channels
        for i in range(levels):
            out_ch = base_channels * (2 ** i)
            encs.append(ConvBlock(prev_ch, out_ch, norm_type, dropout=0.0))
            prev_ch = out_ch
        self.encoders = nn.ModuleList(encs)

        # ----- Bottleneck -----
        self.bottleneck = ConvBlock(prev_ch, prev_ch * 2, norm_type, dropout=dropout_bottleneck)
        bottleneck_ch = prev_ch * 2

        # ----- Decoder -----
        ups = []
        decs = []
        for i in reversed(range(levels)):
            out_ch = base_channels * (2 ** i)
            ups.append(nn.ConvTranspose2d(bottleneck_ch, out_ch, kernel_size=2, stride=2))
            decs.append(ConvBlock(bottleneck_ch, out_ch, norm_type, dropout=0.0))
            bottleneck_ch = out_ch
        self.up_convs = nn.ModuleList(ups)
        self.decoders = nn.ModuleList(decs)

        # ----- Final -----
        self.final_conv = nn.Conv2d(bottleneck_ch, 1, kernel_size=1)

        # init
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_feats = []
        out = x
        # Encoder
        for enc in self.encoders:
            out = enc(out)
            enc_feats.append(out)
            out = self.pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        for up, dec, enc_feat in zip(self.up_convs, self.decoders, reversed(enc_feats)):
            out = up(out)
            # Crop encoder feat if mismatch
            if enc_feat.shape[-2:] != out.shape[-2:]:
                enc_feat = center_crop(enc_feat, out.shape[-2:])
            out = torch.cat([out, enc_feat], dim=1)
            out = dec(out)

        out = self.final_conv(out)
        if self.final_activation == 'sigmoid':
            out = torch.sigmoid(out)
        return out


if __name__ == "__main__":
    # quick sanity test
    model = UNet(in_channels=1, base_channels=64, levels=4, norm_type='batch', dropout_bottleneck=0.1)
    x = torch.randn(2, 1, 355, 447)  # odd dims to test crop
    y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
