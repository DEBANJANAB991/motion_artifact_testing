#!/usr/bin/env python3
from pathlib import Path

import torch
from ptflops import get_model_complexity_info

# import your models
from MR_LKV_refactorv2 import MR_LKV
from UNet import UNet
from train_test_split import RepLKNetReg

# configure input size (modify H,W to your sinogram dims)
H, W = 512, 512

def complexity(model, name):
    macs, params = get_model_complexity_info(
        model, (1, H, W),
        as_strings=False,
        print_per_layer_stat=False
    )
    flops = 2 * macs
    print(f"{name}:")
    print(f"  Params: {params/1e6:.2f} M")
    print(f"  FLOPs : {flops/1e9:.2f} G\n")

def main():
    # instantiate each model with the same init-chars you use in training
    mr_lkv = MR_LKV(in_channels=1, base_channels=32,
                    depths=[1,1,1,1],
                    kernels=[31,51,71,91],
                    norm_type="batch",
                    use_decoder=True,
                    final_activation=None).to('cpu')
    unet   = UNet(in_channels=1, base_channels=64,
                  levels=4,
                  norm_type="batch",
                  dropout_bottleneck=0.1,
                  final_activation=None).to('cpu')
    replk  = RepLKNetReg(
        large_kernel_sizes=[31,29,27,13],
        layers=[2,2,18,2],
        channels=[64,128,256,512],
        drop_path_rate=0.0,
        small_kernel=5,
        in_ch=1
    ).to('cpu')

    # compute
    complexity(mr_lkv, "MR_LKV")
    complexity(unet,   "U-Net")
    complexity(replk,  "RepLKNetReg")

if __name__ == "__main__":
    main()
