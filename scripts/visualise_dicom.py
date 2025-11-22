#!/usr/bin/env python3
"""
FBP round-trip with HU->mu conversion, DICOM geometry, ramp+Hann filter,
circular cropping and recon -> HU conversion. Uses diffct low-level functions.
"""
import os
import math
import torch
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from config import DATASET_PATH
from diffct.differentiable import FanProjectorFunction, FanBackprojectorFunction

# Physical constant: approximate linear attenuation of water at 120 kVp (1/mm)
# ~0.19 cm^-1 = 0.019 mm^-1
MU_WATER = 0.019

# -------------------------
# Utilities
# -------------------------
def find_first_dicom(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".dcm"):
                return os.path.join(dirpath, f)
    return None

def load_dicom(path):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept                    # exact HU
    px = ds.get("PixelSpacing", None)
    if px is None:
        raise ValueError("Missing PixelSpacing in DICOM")
    px = [float(px[0]), float(px[1])]
    # get SDD and SAD if present (units mm)
    sdd = float(getattr(ds, "DistanceSourceToDetector", 0.0))
    sad = float(getattr(ds, "DistanceSourceToPatient", 0.0))
    return hu, px, ds, sdd, sad

# -------------------------
# Filter: Ram-Lak with Hann window
# -------------------------
def ramp_hann_filter(sino_t):
    """
    sino_t: torch.Tensor shape (n_views, n_det) on device
    Returns filtered sinogram same device/shape.
    """
    device = sino_t.device
    n_det = sino_t.shape[-1]
    # rfftfreq: frequencies in cycles per sample (0 .. 0.5)
    freqs = torch.fft.rfftfreq(n_det, d=1.0, device=device, dtype=torch.float32)  # shape (n_r)
    # Ram-Lak: magnitude ~ |f|
    ramp = torch.abs(freqs)
    # Hann window (smooth high frequencies): window = 0.5*(1 + cos(pi * f / f_max))
    f_max = freqs.max() if freqs.numel() > 0 else 0.5
    if f_max == 0:
        window = torch.ones_like(ramp)
    else:
        window = 0.5 * (1.0 + torch.cos(math.pi * freqs / f_max))
    filt = ramp * window  # shape (n_r)
    # rFFT of sinogram along detector axis
    sino_fft = torch.fft.rfft(sino_t, dim=-1)
    sino_fft = sino_fft * filt.unsqueeze(0)  # broadcast over views
    sino_filt = torch.fft.irfft(sino_fft, n=n_det, dim=-1)
    return sino_filt

# -------------------------
# Circular mask (in pixels)
# -------------------------
def circular_mask(H, W, margin=0.0):
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    y, x = np.ogrid[:H, :W]
    r = min(H, W) / 2.0 * (1.0 - margin)
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    return mask.astype(np.float32)

# -------------------------
# Main improved round-trip
# -------------------------
def round_trip_improved(dataset_root,
                        n_views=360,
                        det_count_factor=1.5,
                        default_SAD=538.52,
                        default_SDD=946.746,
                        hu_scale_preserve=False):
    dcm_path = find_first_dicom(dataset_root)
    if dcm_path is None:
        raise FileNotFoundError(f"No DICOM found under {dataset_root!r}")
    print("Using DICOM:", dcm_path)

    hu, px_spacing, ds, sdd_hdr, sad_hdr = load_dicom(dcm_path)
    H, W = hu.shape
    print(f"Image size: {H} x {W}, pixel spacing (mm): {px_spacing}")

    # Use header geometry if present (not zero), else defaults
    SDD = sdd_hdr if sdd_hdr and sdd_hdr > 1e-3 else default_SDD
    SAD = sad_hdr if sad_hdr and sad_hdr > 1e-3 else default_SAD
    print(f"Using SDD={SDD:.3f} mm, SAD={SAD:.3f} mm")

    # Convert HU -> linear attenuation mu (1/mm)
    # mu = mu_water * (1 + HU/1000)
    mu_img = MU_WATER * (1.0 + (hu / 1000.0))
    # For projection numerical stability we don't need additional scaling.
    # Optionally scale mu for projector if values are too large/small
    mu_proj = mu_img.astype(np.float32)

    # Choose detector count based on image FOV in mm and spacing
    fov_mm = W * px_spacing[1]
    det_spacing = px_spacing[1]  # mm
    # choose number of detectors as factor * image width (cap to typical scanner sizes)
    n_detectors = max(int(W * det_count_factor), W)
    print(f"Detector bins: {n_detectors}, detector spacing: {det_spacing:.6f} mm, FOV (mm): {fov_mm:.2f}")

    # Build angles (avoid duplicate 0/2pi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    angles = torch.linspace(0.0, 2.0 * math.pi, n_views + 1, dtype=torch.float32, device=device)[:-1]

    # Prepare image tensor for FanProjectorFunction.apply (shape HxW)
    img_t = torch.from_numpy(mu_proj).to(device=device, dtype=torch.float32)

    # Forward projection (fan-beam) -> sinogram tensor (n_views x n_detectors)
    with torch.no_grad():
        sino_t = FanProjectorFunction.apply(
            img_t,
            angles,
            int(n_detectors),
            float(det_spacing),
            float(SDD),
            float(SAD),
            float(px_spacing[0])
        )
    print("Produced sinogram shape:", tuple(sino_t.shape))

    # Filter sinogram with ramp+hann
    sino_filt_t = ramp_hann_filter(sino_t)

    # Backproject filtered sinogram -> returns reconstruction in same units as mu
    with torch.no_grad():
        recon_t = FanBackprojectorFunction.apply(
            sino_filt_t,
            angles,
            float(det_spacing),
            int(H),
            int(W),
            float(SDD),
            float(SAD),
            float(px_spacing[0])
        )

    recon_mu = recon_t.cpu().numpy()
    print("Recon (mu) min/max:", recon_mu.min(), recon_mu.max())

    # Convert reconstruction mu -> HU: HU = 1000*(mu / mu_water - 1)
    recon_hu = 1000.0 * (recon_mu / MU_WATER - 1.0)

    # Crop to circular FOV to remove outer ring artifact
    mask = circular_mask(H, W, margin=0.0)
    recon_hu_masked = recon_hu * mask
    hu_masked = hu * mask

    # Normalize for display: scale original HU to 0..1 for plotting
    disp_orig = np.clip((hu + 1000.0) / 3000.0, 0.0, 1.0)
    # Normalize recon for display (clip to reasonable HU range, then scale)
    # limit HU display range to [-1000, 2000] same mapping
    recon_disp = np.clip((recon_hu_masked + 1000.0) / 3000.0, 0.0, 1.0)

    # MSE on HU within mask (ignore zero outside)
    valid = mask > 0
    if valid.sum() > 0:
        mse_hu = float(np.mean((hu_masked[valid] - recon_hu_masked[valid]) ** 2))
    else:
        mse_hu = float('nan')

    print(f"Round-trip MSE (HU, masked): {mse_hu:.6f}")

    # Save figure
    out = os.path.join(os.getcwd(), "round_trip_physically_corrected.png")
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(disp_orig, cmap="gray")
    plt.title("Original (scaled HU for display)")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(sino_filt_t.cpu().numpy(), cmap="gray", aspect="auto")
    plt.title("Sinogram (filtered display)")
    plt.xlabel("Detector")
    plt.ylabel("View")

    plt.subplot(1,3,3)
    plt.imshow(recon_disp, cmap="gray")
    plt.title("Reconstruction (HU, FBP + crop)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close() : this code generated these images. why does the reconstructed image look wrong although the original image and sinograms are correct ? 
    print("Saved:", out)

    return {
        "dcm": dcm_path,
        "sino": sino_t.cpu().numpy(),
        "sino_filtered": sino_filt_t.cpu().numpy(),
        "recon_mu": recon_mu,
        "recon_hu_masked": recon_hu_masked,
        "mse_hu": mse_hu,
        "out_path": out
    }

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    res = round_trip_improved(DATASET_PATH)
