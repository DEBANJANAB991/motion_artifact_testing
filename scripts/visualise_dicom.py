#!/usr/bin/env python3
import os, math
import torch
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from config import DATASET_PATH
# original operators
from diffct.differentiable import FanProjectorFunction, FanBackprojectorFunction

MU_WATER = 0.019  # 1/mm

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
    hu = arr * slope + intercept
    px = ds.get("PixelSpacing", None)
    if px is None:
        raise ValueError("Missing PixelSpacing in DICOM")
    px = [float(px[0]), float(px[1])]
    sdd = float(getattr(ds, "DistanceSourceToDetector", 0.0))
    sad = float(getattr(ds, "DistanceSourceToPatient", 0.0))
    return hu, px, ds, sdd, sad

# ramp+hann filter (expects sino shape (n_views, n_det))
def ramp_hann_filter(sino_t):
    device = sino_t.device
    n_det = sino_t.shape[-1]
    freqs = torch.fft.rfftfreq(n_det, d=1.0, device=device, dtype=torch.float32)
    ramp = torch.abs(freqs)
    f_max = freqs.max() if freqs.numel() > 0 else 0.5
    window = 0.5 * (1.0 + torch.cos(math.pi * freqs / f_max)) if f_max != 0 else torch.ones_like(ramp)
    filt = ramp * window
    sino_fft = torch.fft.rfft(sino_t, dim=-1)
    sino_fft = sino_fft * filt.unsqueeze(0)
    sino_filt = torch.fft.irfft(sino_fft, n=n_det, dim=-1)
    return sino_filt

# Optional: compute fan-beam pre-weight cos(gamma)
def fan_beam_preweight(det_u, du, sid):
    # detector indices centered at 0
    u = (np.arange(det_u) - (det_u - 1) / 2.0) * du  # mm from center
    gamma = np.arctan(u / sid)   # fan angle per detector column
    weight = np.cos(gamma).astype(np.float32)      # cos(gamma)
    return torch.from_numpy(weight).float()  # shape (det_u,)

def round_trip_fixed(dataset_root,
                     n_views=360,
                     # recommended CQ500-ish defaults (change if you know exact scanner)
                     n_detectors=888,
                     det_spacing=1.285,   # mm
                     default_SAD=595.0,   # mm (SAD ~ source->isocenter)
                     default_SDD=1085.6,  # mm (source->detector)
                     apply_fan_weight=True):
    dcm_path = find_first_dicom(dataset_root)
    if dcm_path is None:
        raise FileNotFoundError(f"No DICOM found under {dataset_root!r}")
    print("Using DICOM:", dcm_path)

    hu, px_spacing, ds, sdd_hdr, sad_hdr = load_dicom(dcm_path)
    H, W = hu.shape
    pixel_spacing_H = float(px_spacing[0])
    pixel_spacing_W = float(px_spacing[1])
    print(f"Image size: {H}x{W}   pixel spacing H={pixel_spacing_H} W={pixel_spacing_W}")

    # Use header or defaults
    SDD = sdd_hdr if sdd_hdr and sdd_hdr > 1e-3 else default_SDD
    SAD = sad_hdr if sad_hdr and sad_hdr > 1e-3 else default_SAD
    print(f"Using SDD={SDD:.3f} mm, SAD={SAD:.3f} mm")

    # Clip HU -> avoid outliers causing large mu values
    hu = np.clip(hu, -1000, 2000)

    # HU -> mu (1/mm)
    mu_img = MU_WATER * (1.0 + (hu / 1000.0))
    img_t = torch.from_numpy(mu_img.astype(np.float32)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Build angles
    device = img_t.device
    angles = torch.linspace(0.0, 2.0 * math.pi, n_views+1, dtype=torch.float32, device=device)[:-1]

    # Project (fan-beam)
    with torch.no_grad():
        sino_t = FanProjectorFunction.apply(
            img_t,
            angles,
            int(n_detectors),
            float(det_spacing),
            float(SDD),
            float(SAD),
            float(pixel_spacing_W)   # IMPORTANT: use pixel_spacing_W (image column spacing)
        )
    print("Raw sino shape:", sino_t.shape)
    print("Raw sino stats: min", float(sino_t.min()), "max", float(sino_t.max()), "mean", float(sino_t.mean()))

    # Extract shape: ensure sino shape is (n_views, n_det)
    # Many projectors return (n_views, n_det) or (n_views, n_det, ?) — adapt if needed
    if sino_t.ndim == 3:
        # some implementations return (views, det_v, det_u). If so, pick central det_v row:
        # assume shape (views, det_v, det_u)
        sino2d = sino_t[:, sino_t.shape[1] // 2, :].contiguous()
    else:
        sino2d = sino_t.contiguous()

    print("Using sinogram shape for FBP:", sino2d.shape)

    # Optional fan-beam pre-weight (cos gamma)
    if apply_fan_weight:
        weight = fan_beam_preweight(sino2d.shape[1], det_spacing, SAD)  # shape (det_u,)
        weight = weight.to(device)
        sino2d = sino2d * weight.unsqueeze(0)

    # Scale sinogram by detector spacing (convert per-mm integrals -> discrete samples)
    sino_scaled = sino2d * float(det_spacing)

    print("Sino scaled stats: min", float(sino_scaled.min()), "max", float(sino_scaled.max()), "mean", float(sino_scaled.mean()))

    # Filter
    sino_filt = ramp_hann_filter(sino_scaled)
    print("Filtered sino stats: min", float(sino_filt.min()), "max", float(sino_filt.max()), "mean", float(sino_filt.mean()))

    # Backproject
    # FanBackprojectorFunction signature (sino, angles, det_spacing, H, W, SDD, SAD, pixel_spacing)
    with torch.no_grad():
        recon_t = FanBackprojectorFunction.apply(
            sino_filt,
            angles,
            float(det_spacing),
            int(H),
            int(W),
            float(SDD),
            float(SAD),
            float(pixel_spacing_W)
        )

    print("Raw recon (mu) stats:", float(recon_t.min()), float(recon_t.max()), float(recon_t.mean()))

    # Normalize backprojection by number of views
    recon_t = recon_t / float(n_views)

    recon_mu = recon_t.cpu().numpy()
    recon_hu = 1000.0 * (recon_mu / MU_WATER - 1.0)
    print("Recon HU stats:", recon_hu.min(), recon_hu.max(), recon_hu.mean())

    # Display / save images
    disp_orig = np.clip((hu + 1000.0) / 3000.0, 0.0, 1.0)
    recon_disp = np.clip((recon_hu + 1000.0) / 3000.0, 0.0, 1.0)

    out = os.path.join(os.getcwd(), "round_trip_fixed.png")
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow(disp_orig, cmap='gray'); plt.title("Original (scaled HU)"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(sino_filt.cpu().numpy(), cmap='gray', aspect='auto'); plt.title("Filtered sinogram"); plt.xlabel("Detector"); plt.ylabel("View")
    plt.subplot(1,3,3); plt.imshow(recon_disp, cmap='gray'); plt.title("Reconstruction (HU, FBP)"); plt.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    print("Saved:", out)

    # Also run a synthetic phantom test to validate pipeline
    def phantom_test():
        H_, W_ = H, W
        phantom = np.zeros((H_, W_), dtype=np.float32)
        cy, cx = H_//2, W_//2
        r = int(min(H_, W_) * 0.35)
        y,x = np.ogrid[:H_, :W_]
        phantom[(x-cx)**2 + (y-cy)**2 <= r*r] = MU_WATER  # uniform mu circle
        vol = torch.from_numpy(phantom).to(device).float()
        with torch.no_grad():
            s = FanProjectorFunction.apply(vol, angles, int(n_detectors), float(det_spacing), float(SDD), float(SAD), float(pixel_spacing_W))
        if s.ndim == 3:
            s2 = s[:, s.shape[1]//2, :].contiguous()
        else:
            s2 = s
        s2 = s2 * float(det_spacing)
        s2_f = ramp_hann_filter(s2)
        with torch.no_grad():
            rvol = FanBackprojectorFunction.apply(s2_f, angles, float(det_spacing), H_, W_, float(SDD), float(SAD), float(pixel_spacing_W))
        rvol = rvol / float(n_views)
        return float(s2.min()), float(s2.max()), float(rvol.min()), float(rvol.max())
    ph_stats = phantom_test()
    print("Phantom test: sino min/max, reco min/max:", ph_stats)

    return {
        "sino_raw": sino2d.cpu().numpy(),
        "sino_filtered": sino_filt.cpu().numpy(),
        "recon_hu": recon_hu,
        "out_path": out
    }

if __name__ == "__main__":
    res = round_trip_fixed(DATASET_PATH)
