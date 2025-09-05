#!/usr/bin/env python3
import sys
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))


from model_mr_lkv    import MR_LKV
from UNet            import UNet
from replknet        import RepLKNet
from train_test_split import SinogramDataset, SwinIRWrapper, RestormerWrapper

#---------------------------------------------------------------------------
#  Configuration
#---------------------------------------------------------------------------
from config import (
    CLEAN_SINOGRAM_ROOT,
    ARTIFACT_ROOT,
    CKPT_DIR,
   
)

BATCH_SIZE  = 1     
NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------------------------------
#  Model registry: class constructors and checkpoint paths
#---------------------------------------------------------------------------
MODELS = {
    "mr_lkv":    {"cls": MR_LKV,                   "ckpt": CKPT_DIR / "MR_LKV_models"    / "best_model.pth"},
    "unet":      {"cls": lambda: UNet(in_channels=1, base_channels=64, levels=4, norm_type="batch", final_activation=None),
                   "ckpt": CKPT_DIR / "unet"      / "best_model.pth"},
    "replknet":  {"cls": lambda: RepLKNet(
                       large_kernel_sizes=[31,29,27,13],
                       layers=[2,2,18,2],
                       channels=[64,128,256,512],
                       drop_path_rate=0.0,
                       small_kernel=5,
                       in_ch=1),
                   "ckpt": CKPT_DIR / "replknet"  / "best_model.pth"},
    "swinir":    {"cls": lambda: SwinIRWrapper(img_size=512, window_size=8, in_chans=1, out_chans=1),
                   "ckpt": CKPT_DIR / "swinir"    / "best_model.pth"},
    "restormer": {"cls": lambda: RestormerWrapper(
                       inp_channels=1,
                       out_channels=1,
                       dim=48,
                       num_blocks=[2,2,2,4],
                       num_refinement_blocks=2,
                       heads=[1,2,2,4],
                       ffn_expansion_factor=2.0,
                       bias=False,
                       LayerNorm_type='WithBias'
                   ),
                   "ckpt": CKPT_DIR / "restormer" / "best_model.pth"},
}

#---------------------------------------------------------------------------
#  Metric computation
#---------------------------------------------------------------------------
def compute_metrics(clean_np, recon_np):
    dr = clean_np.max() - clean_np.min()
    psnr = peak_signal_noise_ratio(clean_np, recon_np, data_range=dr)
    ssim = structural_similarity(clean_np, recon_np, data_range=dr)
    return psnr, ssim

#---------------------------------------------------------------------------
#  Main comparison script
#---------------------------------------------------------------------------

def main():
    # 1) Build test split
    ds = SinogramDataset(CLEAN_SINOGRAM_ROOT, ARTIFACT_ROOT)
    N = len(ds)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    n_test  = N - n_train - n_val
    _, _, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 2) Loop over models
    results = []
    for name, info in MODELS.items():
        print(f"\n=== Evaluating {name} ===")
        # instantiate & load
        model = info["cls"]().to(DEVICE)
        ckpt = info["ckpt"]
        if not ckpt.exists():
            print(f"  Skip {name}: checkpoint not found at {ckpt}")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        # param & checkpoint size
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ckpt_size = ckpt.stat().st_size / 1e6

        # FLOPs estimation (use sinogram shape from first sample)
        # get a sample to infer H,W
        art0, clean0 = next(iter(loader))
        H, W = clean0.shape[-2], clean0.shape[-1]
        flops_str, _ = get_model_complexity_info(
            model, (1, H, W), as_strings=True,
            print_per_layer_stat=False, verbose=False
        )

        # inference timing (5 warmup + timed)
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

        # full eval: MSE, PSNR, SSIM
        mse_sum = 0.0
        psnrs, ssims = [], []
        with torch.no_grad():
            for art, clean in loader:
                art, clean = art.to(DEVICE), clean.to(DEVICE)
                out = model(art)
                if out.shape[-2:] != clean.shape[-2:]:
                    out = F.interpolate(
                        out, size=clean.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                mse = F.mse_loss(out, clean, reduction='mean').item()
                mse_sum += mse
                c_np = clean.cpu().numpy()[0,0]
                o_np = out.cpu().numpy()[0,0]
                p, s = compute_metrics(c_np, o_np)
                psnrs.append(p); ssims.append(s)
        mse_mean  = mse_sum / len(loader)
        psnr_mean = np.mean(psnrs)
        ssim_mean = np.mean(ssims)

        results.append({
            "model":   name,
            "params":  n_params/1e6,
            "ckpt_MB": ckpt_size,
            "flops":   flops_str,
            "mse":     mse_mean,
            "psnr":    psnr_mean,
            "ssim":    ssim_mean,
            "inf_ms":  inf_ms
        })

    # 3) Print table
    header = f"{'Model':10s} | {'Params(M)':>8s} | {'CKPT(MB)':>8s} | {'FLOPs':>10s} | {'MSE':>10s} | {'PSNR(dB)':>9s} | {'SSIM':>6s} | {'Inf(ms)':>7s}"
    print('\n' + header)
    print('-'*len(header))
    for r in results:
        print(f"{r['model']:10s} | {r['params']:8.1f} | {r['ckpt_MB']:8.1f} | {r['flops']:>10s} | "
              f"{r['mse']:10.3e} | {r['psnr']:9.2f} | {r['ssim']:6.4f} | {r['inf_ms']:7.2f}")

if __name__ == '__main__':
    main()
