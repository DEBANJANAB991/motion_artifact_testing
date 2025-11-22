#!/usr/bin/env python3
"""
FBP round-trip, improved geometry and normalization.

Usage:
    python fbp_roundtrip_fixed.py

Output:
    - round_trip_physically_corrected.png (3-panel: original HU, filtered sinogram, reconstruction HU)
    - returns a dict if used as module (not necessary)

Notes:
    - Requires diffct.differentiable FanProjectorFunction and FanBackprojectorFunction available.
    - Adjust MU_WATER if you know a different value for your spectrum (default ~0.019 mm^-1 for 120 kVp).
"""
import os
import math
import torch
import pydicom
import numpy as np
import matplotlib.pyplot as plt

from diffct.differentiable import FanProjectorFunction, FanBackprojectorFunction

# ---------- User-tweakable constants ----------
# Approx linear attenuation of water (~1/mm) near 120 kVp. Change if needed.
MU_WATER = 0.019

# Defaults if header does not contain geometry
DEFAULT_SDD = 946.746   # mm
DEFAULT_SAD = 538.52    # mm

# Number of projection views to synthesize (use typical scanner value)
DEFAULT_N_VIEWS = 360

# ------------------------- Utilities -------------------------
def find_first_dicom(root):
    """Return the path of the first .dcm file in `root` tree."""
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".dcm"):
                return os.path.join(dirpath, f)
    return None


def load_dicom(path):
    """
    Read a DICOM and return (hu_image, pixel_spacing_mm, ds, sdd, sad).
    If header fields missing, sdd/sad may be 0.0 (caller will handle defaults).
    """
    ds = pydicom.dcmread(path, force=True)
    # Access pixels (may raise if pixel data malformed)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept  # exact HU values

    px = ds.get("PixelSpacing", None)
    if px is None:
        raise ValueError("Missing PixelSpacing in DICOM header")
    # PixelSpacing: [rowSpacing, colSpacing] in mm
    px = [float(px[0]), float(px[1])]

    # try to read geometry values; if missing will be 0.0
    sdd = float(getattr(ds, "DistanceSourceToDetector", 0.0))
    sad = float(getattr(ds, "DistanceSourceToPatient", 0.0))

    return hu.astype(np.float32), px, ds, sdd, sad


# ------------------------- Filter: Ram-Lak with Hann window -------------------------
def ramp_hann_filter(sino_t):
    """
    Filter sinogram with Ram-Lak * Hann window.
    Input:
        sino_t: torch.Tensor (n_views, n_det)
    Returns:
        sino_filt (same device/shape)
    """
    device = sino_t.device
    n_det = int(sino_t.shape[-1])

    # frequencies for rfft (cycles per sample)
    freqs = torch.fft.rfftfreq(n_det, d=1.0, device=device, dtype=torch.float32)  # shape (n_r)
    ramp = torch.abs(freqs)  # Ram-Lak magnitude

    f_max = freqs.max() if freqs.numel() > 0 else 0.5
    if f_max == 0.0:
        window = torch.ones_like(ramp)
    else:
        window = 0.5 * (1.0 + torch.cos(math.pi * freqs / f_max))

    filt = ramp * window  # shape (n_r,)

    sino_fft = torch.fft.rfft(sino_t, dim=-1)
    # broadcast filter over views
    sino_fft = sino_fft * filt.unsqueeze(0)
    sino_filt = torch.fft.irfft(sino_fft, n=n_det, dim=-1)
    return sino_filt


# ------------------------- Circular mask -------------------------
def circular_mask(H, W, margin=0.0):
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    y, x = np.ogrid[:H, :W]
    r = min(H, W) / 2.0 * (1.0 - margin)
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return mask.astype(np.float32)


# ------------------------- Main improved round-trip -------------------------
def round_trip_improved(dataset_root,
                        n_views=DEFAULT_N_VIEWS,
                        default_SAD=DEFAULT_SAD,
                        default_SDD=DEFAULT_SDD,
                        mu_water=MU_WATER,
                        use_cuda=True,
                        save_out=True):
    """
    - dataset_root: folder containing DICOM(s). We pick the first DICOM found.
    - n_views: number of projection angles for forward/back projection.
    - Returns a dict containing results and paths saved (if save_out True).
    """

    dcm_path = find_first_dicom(dataset_root)
    if dcm_path is None:
        raise FileNotFoundError(f"No DICOM found under {dataset_root!r}")
    print("Using DICOM:", dcm_path)

    # Load DICOM and geometry (pixel spacing, SDD, SAD may be zero)
    hu_img, px_spacing, ds, sdd_hdr, sad_hdr = load_dicom(dcm_path)
    H, W = hu_img.shape
    print(f"Image size: {H} x {W}, pixel spacing (mm): {px_spacing}")

    # Use header geometry if present and sensible (>1e-3), else fallback to defaults
    SDD = sdd_hdr if (sdd_hdr and sdd_hdr > 1e-3) else default_SDD
    SAD = sad_hdr if (sad_hdr and sad_hdr > 1e-3) else default_SAD
    print(f"Using SDD={SDD:.3f} mm, SAD={SAD:.3f} mm")

    # Convert HU -> linear attenuation mu (1/mm)
    # mu = mu_water * (1 + HU/1000)
    mu_img = mu_water * (1.0 + (hu_img / 1000.0))
    mu_proj = mu_img.astype(np.float32)

    # choose detector spacing and number of detectors based on FOV and pixel spacing
    # use column pixel spacing (px_spacing[1]) as detector bin physical size
    det_spacing = float(px_spacing[1])
    fov_mm = W * det_spacing
    n_detectors = max(int(round(fov_mm / det_spacing)), W)  # at least image width
    # Optionally cap detector count to reasonable upper bound (e.g., 2x width)
    n_detectors = min(n_detectors, max(W * 3, n_detectors))
    print(f"Detector bins: {n_detectors}, detector spacing: {det_spacing:.6f} mm, FOV (mm): {fov_mm:.2f}")

    # Build angles (avoid duplicate 0/2pi)
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    angles = torch.linspace(0.0, 2.0 * math.pi, n_views + 1, dtype=torch.float32, device=device)[:-1]  # length n_views
    delta_theta = float(2.0 * math.pi / float(n_views))

    # Prepare image tensor for FanProjectorFunction.apply (HxW)
    img_t = torch.from_numpy(mu_proj).to(device=device, dtype=torch.float32)

    # Forward projection (fan-beam) -> sinogram tensor shape (n_views, n_detectors)
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

    # Multiply by Δθ to account for discrete angular sampling (angular normalization)
    # This is important for correct amplitude in backprojection
    sino_filt_t = sino_filt_t * float(delta_theta)

    # Backproject filtered sinogram -> returns reconstruction in same units as mu (1/mm)
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
    print("Recon (mu) min/max:", float(recon_mu.min()), float(recon_mu.max()))

    # Convert reconstruction mu -> HU: HU = 1000*(mu / mu_water - 1)
    recon_hu = 1000.0 * (recon_mu / mu_water - 1.0)

    # Crop to circular FOV to remove outer ring artifact
    mask = circular_mask(H, W, margin=0.0)
    recon_hu_masked = recon_hu * mask
    hu_masked = hu_img * mask

    # Normalize for display: scale original HU to [0,1] for plotting (windowing done visually similarly)
    # choose display mapping of HU range [-1000, 2000] -> [0,1]
    disp_lo, disp_hi = -1000.0, 2000.0
    disp_orig = np.clip((hu_img - disp_lo) / (disp_hi - disp_lo), 0.0, 1.0)
    recon_disp = np.clip((recon_hu_masked - disp_lo) / (disp_hi - disp_lo), 0.0, 1.0)

    # MSE on HU within mask (ignore zero outside)
    valid = mask > 0
    if valid.sum() > 0:
        mse_hu = float(np.mean((hu_masked[valid] - recon_hu_masked[valid]) ** 2))
    else:
        mse_hu = float('nan')
    print(f"Round-trip MSE (HU, masked): {mse_hu:.6f}")

    # Save figure(s)
    out = os.path.join(os.getcwd(), "round_trip_physically_corrected.png")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(disp_orig, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("Original (scaled HU for display)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    # show filtered sinogram (move to CPU numpy)
    plt.imshow(sino_filt_t.cpu().numpy(), cmap="gray", aspect="auto")
    plt.title("Sinogram (filtered x Δθ)")
    plt.xlabel("Detector")
    plt.ylabel("View")

    plt.subplot(1, 3, 3)
    plt.imshow(recon_disp, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("Reconstruction (HU, FBP + crop)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

    results = {
        "dcm": dcm_path,
        "angles": angles.cpu().numpy(),
        "sino": sino_t.cpu().numpy(),
        "sino_filtered": sino_filt_t.cpu().numpy(),
        "recon_mu": recon_mu,
        "recon_hu_masked": recon_hu_masked,
        "mse_hu": mse_hu,
        "out_path": out
    }
    return results


# ------------------------- Run as script -------------------------
if __name__ == "__main__":
    # pick dataset root from env var or current working dir's 'data' or adapt this path
    # You had earlier used DATASET_PATH in config; if you want to reuse that, import config and replace below.
    import argparse
    parser = argparse.ArgumentParser(description="Improved FBP round-trip from DICOM")
    parser.add_argument("--dataset", type=str, default=".", help="root folder containing DICOM series (we pick first DICOM found)")
    parser.add_argument("--nviews", type=int, default=DEFAULT_N_VIEWS, help="number of projection views")
    parser.add_argument("--muwater", type=float, default=MU_WATER, help="mu_water (1/mm)")
    parser.add_argument("--use-cuda", action="store_true", help="use CUDA if available")
    args = parser.parse_args()

    res = round_trip_improved(args.dataset, n_views=args.nviews, mu_water=args.muwater, use_cuda=args.use_cuda)
    print("Done. Outputs:", res["out_path"])
