#!/usr/bin/env python3
import os, math
import torch
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from config import DATASET_PATH
from diffct.differentiable import FanProjectorFunction, FanBackprojectorFunction

# ---------------------------------------------------------
# SCANNER GEOMETRY (FROM FAST SEARCH RESULTS)
# ---------------------------------------------------------
SID = 560.0           # Source → Isocenter (mm)
SDD = 1100.0          # Source → Detector (mm)
DET_SPACING = 1.2     # Detector spacing (mm)
DET_COUNT = 736       # Number of detector elements
VOXEL_SPACING = 1.0   # MUST remain 1.0 for diffct
# ---------------------------------------------------------

MU_WATER = 0.019      # 1/mm (standard water attenuation)

def find_first_dicom(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".dcm"):
                return os.path.join(dirpath, f)
    return None

def load_dicom(path):
    ds = pydicom.dcmread(path, force=True)
    px = ds.PixelSpacing
    arr = ds.pixel_array.astype(np.float32)
    hu = arr * float(getattr(ds,"RescaleSlope",1.0)) + float(getattr(ds,"RescaleIntercept",0.0))
    return hu, [float(px[0]), float(px[1])], ds

def ramp_hann_filter(sino):
    device = sino.device
    n_det = sino.shape[-1]

    freqs = torch.fft.rfftfreq(n_det, d=1.0, device=device)
    ramp = torch.abs(freqs)

    hann = 0.5 * (1.0 + torch.cos(math.pi * freqs / (freqs.max() + 1e-8)))
    filt = ramp * hann

    S = torch.fft.rfft(sino, dim=-1)
    S = S * filt.unsqueeze(0)

    return torch.fft.irfft(S, n_det, dim=-1)

def run_round_trip(dataset_root, n_views=720):

    # ------------------- Load clean CT ---------------------
    dcm_path = find_first_dicom(dataset_root)
    if dcm_path is None:
        raise RuntimeError("No DICOM found!")
    print("Using:", dcm_path)

    hu, px, ds = load_dicom(dcm_path)
    hu = np.clip(hu, -1000, 2000)
    H, W = hu.shape

    print("Loaded CT:", H, W, "pixel spacing:", px)

    # ------------------- CT → mu ----------------------------
    mu = MU_WATER * (1.0 + hu / 1000.0)
    img_t = torch.from_numpy(mu.astype(np.float32)).cuda()

    # ------------------- Projection Angles ------------------
    angles = torch.linspace(0, 2*math.pi, n_views+1, device=img_t.device)[:-1]

    # ------------------- Forward Projection ----------------
    with torch.no_grad():
        sino = FanProjectorFunction.apply(
            img_t, angles,
            int(DET_COUNT),
            float(DET_SPACING),
            float(SDD),
            float(SID),
            float(VOXEL_SPACING)
        )

    # If projector returns (views, detector_v, detector_u)
    if sino.ndim == 3:
        sino = sino[:, sino.shape[1]//2, :].contiguous()

    print("Sinogram shape:", sino.shape)

    # ------------------- Filter Sinogram -------------------
    sino_filt = ramp_hann_filter(sino)

    # ------------------- FBP Reconstruction ----------------
    with torch.no_grad():
        reco = FanBackprojectorFunction.apply(
            sino_filt,
            angles,
            float(DET_SPACING),
            H, W,
            float(SDD),
            float(SID),
            float(VOXEL_SPACING)
        )

    # Correct FBP scaling for fan-beam
    reco = reco * (math.pi / (2.0 * n_views))

    recon_mu = reco.cpu().numpy()
    recon_hu = 1000.0 * (recon_mu/MU_WATER - 1.0)

    # ------------------- Visualization Panels -------------------

    # Original CT
    disp_ct = np.clip((hu + 1000)/3000, 0, 1)

    # Unfiltered sinogram
    sino_raw = sino.cpu().numpy()
    sino_raw_disp = (sino_raw - sino_raw.min()) / (sino_raw.max() - sino_raw.min() + 1e-8)

    # Filtered sinogram
    sino_f = sino_filt.cpu().numpy()
    sino_f_disp = (sino_f - sino_f.min()) / (sino_f.max() - sino_f.min() + 1e-8)

    # Reconstruction
    disp_recon = np.clip((recon_hu + 1000)/3000, 0, 1)

    # ------------------- Plot ------------------------

    plt.figure(figsize=(26,6))

    plt.subplot(1,4,1)
    plt.imshow(disp_ct, cmap='gray')
    plt.title("Original CT (HU)")
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(sino_raw_disp, cmap='gray', aspect='auto')
    plt.title("Unfiltered Sinogram")
    plt.xlabel("Detector")
    plt.ylabel("View")

    plt.subplot(1,4,3)
    plt.imshow(sino_f_disp, cmap='gray', aspect='auto')
    plt.title("Filtered Sinogram")
    plt.xlabel("Detector")
    plt.ylabel("View")

    plt.subplot(1,4,4)
    plt.imshow(disp_recon, cmap='gray')
    plt.title("Reconstruction (FBP)")
    plt.axis('off')

    plt.tight_layout()
    out = "round_trip_final_with_unfiltered.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print("Saved:", out)

    return recon_hu, sino_raw, sino_f


if __name__ == "__main__":
    run_round_trip(DATASET_PATH)
