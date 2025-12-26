#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from config import (
    ARTIFACT_SINOGRAM_2D_TEST,
    CLEAN_SINOGRAM_2D_TEST,
    PREDICTED_SINOGRAM_2D_TEST,
)

# -----------------------------
# Parameters
# -----------------------------
MAX_SAMPLES = 1000      # use 500–1000 for fast evaluation
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Metrics
# -----------------------------
def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(max_val**2 / (mse + 1e-12))

def compute_ssim(a, b):
    return ssim(a, b, data_range=1.0)

# -----------------------------
# Main
# -----------------------------
def main():

    art_dir  = Path(ARTIFACT_SINOGRAM_2D_TEST)
    clean_dir = Path(CLEAN_SINOGRAM_2D_TEST)
    pred_dir = Path(PREDICTED_SINOGRAM_2D_TEST)

    files = sorted([p.name for p in pred_dir.glob("*.npy")])

    if len(files) == 0:
        raise RuntimeError("No predicted sinograms found")

    # sample subset
    if len(files) > MAX_SAMPLES:
        files = random.sample(files, MAX_SAMPLES)

    psnr_art, psnr_pred = [], []
    ssim_art, ssim_pred = [], []

    print(f"Evaluating {len(files)} sinogram views...\n")

    for fname in tqdm(files):

        art = np.load(art_dir / fname).astype(np.float32)
        clean = np.load(clean_dir / fname).astype(np.float32)
        pred = np.load(pred_dir / fname).astype(np.float32)

        # normalize (same as training)
        # normalize using CLEAN reference only
        min_val = clean.min()
        max_val = clean.max()

        clean = (clean - min_val) / (max_val - min_val + 1e-12)
        art   = (art   - min_val) / (max_val - min_val + 1e-12)
        pred  = (pred  - min_val) / (max_val - min_val + 1e-12)


        # to torch
        art_t = torch.from_numpy(art)
        clean_t = torch.from_numpy(clean)
        pred_t = torch.from_numpy(pred)

        # metrics
        psnr_art.append(psnr(art_t, clean_t).item())
        psnr_pred.append(psnr(pred_t, clean_t).item())

        ssim_art.append(compute_ssim(art, clean))
        ssim_pred.append(compute_ssim(pred, clean))

    # -----------------------------
    # Results
    # -----------------------------
    print("\n================ SINOGRAM METRICS ================")
    print(f"Artifact → Clean : PSNR = {np.mean(psnr_art):.2f} dB | SSIM = {np.mean(ssim_art):.4f}")
    print(f"MR_LKV → Clean   : PSNR = {np.mean(psnr_pred):.2f} dB | SSIM = {np.mean(ssim_pred):.4f}")
    print("-------------------------------------------------")
    print(f"Δ PSNR  = {np.mean(psnr_pred) - np.mean(psnr_art):+.2f} dB")
    print(f"Δ SSIM  = {np.mean(ssim_pred) - np.mean(ssim_art):+.4f}")
    print("=================================================")


if __name__ == "__main__":
    main()
