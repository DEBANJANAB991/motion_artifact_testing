#!/usr/bin/env python3
import sys
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "Restormer"))

sys.path.insert(0, str(repo_root  / "swin2sr"))


from train_test_split import SinogramDataset, SwinIRWrapper, RestormerWrapper, RepLKNetReg, MR_LKV, Swin2SRWrapper

from models.network_swin2sr import Swin2SR

from config import CLEAN_SINOGRAM_ROOT, ARTIFACT_ROOT, CKPT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
NUM_WORKERS = 0

def compute_metrics(clean_np, recon_np):
    dr = clean_np.max() - clean_np.min()
    psnr = peak_signal_noise_ratio(clean_np, recon_np, data_range=dr)
    ssim = structural_similarity(clean_np, recon_np, data_range=dr)
    return psnr, ssim

# registry of model factories and checkpoint subfolders
MODELS = {
    "mr_lkv":    {"factory": lambda: MR_LKV(1, base_channels=32, depths=[1,1,1,1], kernels=[31,51,71,91], norm_type="batch", use_decoder=True, final_activation=None),
                   "folder": "mr_lkv"},
    "unet": {"factory": lambda: __import__("UNet").UNet(in_channels=1, base_channels=64,levels=4, norm_type="batch", dropout_bottleneck=0.1, final_activation=None),"folder": "unet"},
    "replknet":  {"factory": lambda: RepLKNetReg([31,29,27,13],[2,2,18,2],[64,128,256,512],0.0,5,1),
                   "folder": "replknet"},
    "swinir":    {"factory": lambda: SwinIRWrapper(img_size=512, window_size=8, in_chans=1, out_chans=1, depths=[4,4,4,4], embed_dim=64, num_heads=[2,2,2,2], mlp_ratio=2, use_checkpoint=True),
                   "folder": "swinir"},
    "restormer": {"factory": lambda: RestormerWrapper(1,1,48,[2,2,2,4],2,[1,2,2,4],2.0,False,"WithBias"),
                   "folder": "restormer"},
    "swin2sr":   {"factory": lambda: Swin2SRWrapper(
                       in_ch=1, embed_dim=64, depths=(4,4,4,4), num_heads=(4,4,4,4),
                       window_size=8, upscale=1, upsampler='', img_range=1.0, img_size=(64,64)
                   ),
                   "folder": "swin2sr"},
}

def main():
    # build dataset splits
    ds = SinogramDataset(CLEAN_SINOGRAM_ROOT, ARTIFACT_ROOT, patch=64)
    N = len(ds)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    n_test  = N - n_train - n_val
    _, _, test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # get a sample’s spatial dims
    art0, clean0 = next(iter(loader))
    H, W = clean0.shape[-2], clean0.shape[-1]

    results = []
    for name, info in MODELS.items():
        print(f"\n=== Evaluating {name} ===")
        ckpt_dir = Path(CKPT_DIR) / info["folder"]
        ckpt = ckpt_dir / "best_model.pth"
        if not ckpt.exists():
            print(f"  Skip {name}: no checkpoint at {ckpt}")
            continue

        model = info["factory"]().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        # params
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # FLOPs
        flops_str, _ = get_model_complexity_info(model, (1, H, W),
                                                 as_strings=True,
                                                 print_per_layer_stat=False,
                                                 verbose=False)

        # inference timing
        art_w = art0.to(DEVICE)
        with torch.no_grad():
            for _ in range(5): _ = model(art_w)
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(len(loader)):
            art, _ = next(iter(loader))
            _ = model(art.to(DEVICE))
        end.record()
        torch.cuda.synchronize()
        inf_ms = start.elapsed_time(end) / len(loader)

        # PSNR, SSIM
        psnrs, ssims = [], []
        with torch.no_grad():
            for art, clean in loader:
                out = model(art.to(DEVICE))
                o = out.cpu().numpy()[0,0]
                c = clean.numpy()[0,0]
                p, s = compute_metrics(c, o)
                psnrs.append(p); ssims.append(s)
        results.append({
            "model":      name,
            "params (M)": n_params/1e6,
            "FLOPs":      flops_str,
            "Inf (ms)":   inf_ms,
            "PSNR (dB)":  np.mean(psnrs),
            "SSIM":       np.mean(ssims),
        })

    # print summary table
    import pandas as pd
    df = pd.DataFrame(results)
    cols = ["model","params (M)","FLOPs","Inf (ms)","PSNR (dB)","SSIM"]
    df = df[cols]
    print("\nModel Comparison:\n")
    print(df.to_markdown(index=False))

    out_dir = Path("results") / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "model_comparison.csv", index=False)
    df.to_json(out_dir / "model_comparison.json", orient="records", lines=True)

    print("Saved full results to model_comparison.csv")
if __name__ == "__main__":
    main()
