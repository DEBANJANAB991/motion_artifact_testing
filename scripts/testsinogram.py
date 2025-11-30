#!/usr/bin/env python3
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pydicom
from scipy import ndimage

from diffct.differentiable import FanBackprojectorFunction
from config import CLEAN_SINOGRAM_ROOT, DATASET_PATH

# ---------------------------------------------------------
# FIXED GEOMETRY (from your geometry search)
# ---------------------------------------------------------
SID = 560.0
SDD = 1100.0
DET_SPACING = 1.2
DET_COUNT = 736
VOXEL_SPACING = 1.0
MU_WATER = 0.019

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
NUM_SAMPLES = 40
OUT_DIR = Path("recon_preview")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1) HEAD CENTER DETECTION + RECENTERING
# ---------------------------------------------------------
def detect_head_center(hu, thresh=-300):
    """
    Detects head region by thresholding (> -300 HU),
    chooses largest connected component,
    returns centroid (cy,cx).
    """

    mask = hu > thresh
    labeled, n = ndimage.label(mask)

    if n == 0:
        # fallback to center
        H, W = hu.shape
        return H//2, W//2

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    comp = (labeled == largest)

    ys, xs = np.nonzero(comp)
    if len(ys) == 0:
        H, W = hu.shape
        return H//2, W//2

    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))
    return cy, cx


def recenter_sinogram_from_hu(sino, hu):
    """
    Compute HU centroid shift and apply same shift to sinogram.
    Sino is shape (N_views, det_count).
    """

    H, W = hu.shape
    N_views, N_det = sino.shape

    cy, cx = detect_head_center(hu)
    target_cy, target_cx = H//2, W//2

    dy = target_cy - cy
    dx = target_cx - cx

    # 💡 Only horizontal shift affects sinogram → shift detector axis (dx)
    # vertical shift dy affects projection path → would require view-dependent correction.
    # For head CT it is enough to correct dx (dominant).
    sino_shifted = torch.roll(sino, shifts=dx, dims=1)
    return sino_shifted


# ---------------------------------------------------------
# RAMP + HANN FILTER
# ---------------------------------------------------------
def ramp_hann_filter(sino):
    device = sino.device
    n_det = sino.shape[-1]

    freqs = torch.fft.rfftfreq(n_det, d=1.0, device=device)
    ramp = torch.abs(freqs)

    hann = 0.5 * (1.0 + torch.cos(np.pi * freqs / (freqs.max() + 1e-8)))
    filt = ramp * hann

    S = torch.fft.rfft(sino, dim=-1)
    S = S * filt.unsqueeze(0)
    return torch.fft.irfft(S, n_det, dim=-1)


# ---------------------------------------------------------
# LOAD MATCHING HU SLICE FOR A SINOGRAM
# ---------------------------------------------------------
def load_corresponding_hu(dcm_path):
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + inter
    hu = np.clip(hu, -1024, 3071)
    return hu


def find_matching_dicom(sino_path):
    """
    Reconstruct original DICOM path from sinogram path.
    Assumes CLEAN_SINOGRAM_ROOT mirrors DATASET_PATH structure.
    """
    relative = sino_path.relative_to(CLEAN_SINOGRAM_ROOT)
    dcm_path = Path(DATASET_PATH) / relative.with_suffix(".dcm")
    return dcm_path


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N_VIEWS = 720
    angles = torch.linspace(0, 2*np.pi, N_VIEWS, device=device)[:-1]

    sino_files = list(Path(CLEAN_SINOGRAM_ROOT).rglob("*.npy"))[:NUM_SAMPLES]

    print(f"Reconstructing {len(sino_files)} sinograms...")

    for idx, sino_path in enumerate(tqdm(sino_files)):

        # 1) Load sinogram
        sino_np = np.load(sino_path)
        sino_np = sino_np.astype(np.float32)
        sino = torch.from_numpy(sino_np).to(device)

        # 2) Load matching HU slice for centering
        dcm_path = find_matching_dicom(sino_path)
        hu = load_corresponding_hu(dcm_path)

        # 3) Recenter sinogram using detected head position
        sino = recenter_sinogram_from_hu(sino, hu)

        # 4) Filter sinogram
        sino_filt = ramp_hann_filter(sino)

        # 5) Backproject
        H, W = 512, 512
        with torch.no_grad():
            recon = FanBackprojectorFunction.apply(
                sino_filt,
                angles,
                float(DET_SPACING),
                int(H),
                int(W),
                float(SDD),
                float(SID),
                float(VOXEL_SPACING)
            )

        # 6) Normalize
        recon = recon * (np.pi / (2 * N_VIEWS))

        # 7) Convert to HU
        recon_mu = recon.cpu().numpy()
        recon_hu = 1000.0 * (recon_mu / MU_WATER - 1.0)

        # 8) Save PNG
        save_path = OUT_DIR / f"recon_{idx:03d}.png"
        disp = np.clip((recon_hu + 1000) / 3000, 0, 1)
        plt.imsave(save_path, disp, cmap="gray")

    print(f"\n✅ Saved reconstructions in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
