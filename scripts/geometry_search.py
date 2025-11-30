#!/usr/bin/env python3
"""
geometry_search.py (updated)

Coarse + refine search for SID, SDD, detector spacing, detector count
to maximize reconstruction sharpness (variance of Laplacian).

Saves:
 - best_recon.png
 - geometry_search_log.csv

Usage:
    python geometry_search.py

Note:
 - This script uses your installed diffct FanProjectorFunction & FanBackprojectorFunction.
 - It forces voxel_spacing=1.0 when calling the CUDA kernels (required for your version).
"""
import os
import math
import csv
import time
import numpy as np
import torch
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from config import DATASET_PATH
from diffct.differentiable import FanProjectorFunction, FanBackprojectorFunction

# ----------------- constants (some from header) -----------------
MU_WATER = 0.0192
# NOTE: we will search SDD as well; keep header value as a fallback/starting point
SDD_HEADER = 946.746
PIXEL_SIZE = 0.488281
H = W = 512
# ----------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def find_first_dicom(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".dcm"):
                return os.path.join(dirpath, f)
    return None

def load_first_slice(path):
    dcm = find_first_dicom(path)
    if dcm is None:
        raise FileNotFoundError("No DICOM found at path")
    ds = pydicom.dcmread(dcm)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    hu = np.clip(hu, -1000, 2000)
    px = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
    return hu, px, ds, dcm

def focus_score(image, center=None, radius=None):
    """Variance of absolute Laplacian inside circular mask."""
    if center is None:
        center = (image.shape[0]//2, image.shape[1]//2)
    if radius is None:
        radius = min(image.shape)//2 * 0.9
    lap = laplace(image)
    yy, xx = np.indices(image.shape)
    mask = ((yy - center[0])**2 + (xx - center[1])**2) <= (radius**2)
    vals = lap[mask]
    return float(np.var(np.abs(vals)))

def do_projection_and_backprojection(mu_img, sid, sdd, det_count, det_spacing, n_views=720, preweight=True):
    """
    Project + FBP using the calling convention observed in your diffct build.
    IMPORTANT: voxel_spacing is fixed to 1.0 for this diffct version.
    """
    img_t = torch.from_numpy(mu_img).float().to(device)
    angles = torch.linspace(0.0, 2.0 * math.pi, n_views + 1, device=device)[:-1].float()

    # Forward project (observed signature: image, angles, num_detectors, detector_spacing, sdd, sid, voxel_spacing)
    with torch.no_grad():
        sino = FanProjectorFunction.apply(
            img_t, angles,
            int(det_count), float(det_spacing),
            float(sdd), float(sid), 1.0
        )

    if sino.ndim == 3:
        sino2d = sino[:, sino.shape[1]//2, :].contiguous()
    else:
        sino2d = sino.contiguous()

    if preweight:
        u = (np.arange(sino2d.shape[1]) - (sino2d.shape[1]-1)/2.0) * det_spacing
        gamma = np.arctan(u / sid)
        weight = torch.from_numpy(np.cos(gamma).astype(np.float32)).to(device)
        sino2d = sino2d * weight.unsqueeze(0)

    # Use convention: multiply by det_spacing, filter, backproject, then divide by n_views
    sino_scaled = sino2d * float(det_spacing)

    # Ramp + Hann filter (GPU)
    n_det = sino_scaled.shape[-1]
    freqs = torch.fft.rfftfreq(n_det, d=1.0, device=device)
    ramp = torch.abs(freqs)
    fmax = float(freqs.max().cpu().numpy()) if freqs.numel() > 0 else 1.0
    hann = 0.5 * (1.0 + torch.cos(math.pi * freqs / fmax))
    filt = ramp * hann
    S = torch.fft.rfft(sino_scaled, dim=-1)
    S = S * filt.unsqueeze(0)
    sino_f = torch.fft.irfft(S, n_det, dim=-1)

    # Backproject (note voxel_spacing=1.0)
    with torch.no_grad():
        recon = FanBackprojectorFunction.apply(
            sino_f, angles,
            float(det_spacing),
            int(H), int(W),
            float(sdd), float(sid), 1.0
        )

    recon = recon / float(n_views)
    recon_mu = recon.cpu().numpy()
    recon_hu = 1000.0 * (recon_mu / MU_WATER - 1.0)
    return recon_hu

def coarse_search(mu_img):
    # NEW coarse ranges (suggested)
    sid_candidates = np.arange(550.0, 621.0, 10.0)     # 550..620 step 10
    sdd_candidates = np.arange(1000.0, 1151.0, 20.0)   # 1000..1150 step 20
    det_spacing_candidates = [1.0, 1.1, 1.2, 1.3]
    det_counts = [736, 888, 1024]

    total = len(sid_candidates) * len(sdd_candidates) * len(det_spacing_candidates) * len(det_counts)
    print(f"Coarse search combos: {total}")
    best = {"score": -1.0}
    log = []
    idx = 0
    start = time.time()
    for sid in sid_candidates:
        for sdd in sdd_candidates:
            for det_spacing in det_spacing_candidates:
                for det_count in det_counts:
                    idx += 1
                    try:
                        reco = do_projection_and_backprojection(mu_img, sid=float(sid), sdd=float(sdd),
                                                               det_count=int(det_count), det_spacing=float(det_spacing))
                        score = focus_score(reco)
                    except Exception as e:
                        print("Error at", sid, sdd, det_spacing, det_count, e)
                        score = -1.0
                        reco = None
                    log.append((sid, sdd, det_spacing, det_count, score))
                    if score > best["score"]:
                        best.update({"sid": float(sid), "sdd": float(sdd), "det_spacing": float(det_spacing), "det_count": int(det_count), "score": float(score), "reco": reco})
                    if idx % 50 == 0 or idx == total:
                        elapsed = time.time() - start
                        print(f"{idx}/{total} combos tested — best score {best['score']:.4f} (sid={best['sid']}, sdd={best['sdd']}, det={best['det_count']}, dsp={best['det_spacing']}) — elapsed {elapsed:.1f}s")
    return best, log

def refine_search(mu_img, coarse_best):
    sid0 = coarse_best["sid"]
    sdd0 = coarse_best["sdd"]
    ds0 = coarse_best["det_spacing"]
    det_count = coarse_best["det_count"]

    # refine ranges around coarse best
    sid_candidates = np.linspace(max(540.0, sid0-15.0), sid0+15.0, 31)
    sdd_candidates = np.linspace(max(1020.0, sdd0-30.0), sdd0+30.0, 31)
    det_spacing_candidates = np.linspace(max(0.9, ds0-0.15), ds0+0.15, 21)

    total = len(sid_candidates) * len(sdd_candidates) * len(det_spacing_candidates)
    print(f"Refine search combos: {total} (det_count={det_count})")
    best = {"score": -1.0}
    log = []
    idx = 0
    start = time.time()
    for sid in sid_candidates:
        for sdd in sdd_candidates:
            for det_spacing in det_spacing_candidates:
                idx += 1
                try:
                    reco = do_projection_and_backprojection(mu_img, sid=float(sid), sdd=float(sdd),
                                                           det_count=int(det_count), det_spacing=float(det_spacing))
                    score = focus_score(reco)
                except Exception as e:
                    score = -1.0
                    reco = None
                log.append((sid, sdd, det_spacing, det_count, score))
                if score > best["score"]:
                    best.update({"sid": float(sid), "sdd": float(sdd), "det_spacing": float(det_spacing), "det_count": int(det_count), "score": float(score), "reco": reco})
                if idx % 500 == 0 or idx == total:
                    elapsed = time.time() - start
                    print(f"{idx}/{total} tested — current best {best['score']:.4f} sid={best['sid']} sdd={best['sdd']} dsp={best['det_spacing']} — elapsed {elapsed:.1f}s")
    return best, log

if __name__ == "__main__":
    hu, px, ds, dcmfile = load_first_slice(DATASET_PATH)
    print("Loaded DICOM:", dcmfile)
    mu_img = MU_WATER * (1.0 + hu / 1000.0)

    # coarse
    coarse_best, coarse_log = coarse_search(mu_img)
    print("Coarse best:", coarse_best['sid'], coarse_best['sdd'], coarse_best['det_spacing'], coarse_best['det_count'], "score:", coarse_best['score'])

    # refine
    refine_best, refine_log = refine_search(mu_img, coarse_best)
    print("Refined best:", refine_best['sid'], refine_best['sdd'], refine_best['det_spacing'], refine_best['det_count'], "score:", refine_best['score'])

    # write CSV log
    csvfile = "geometry_search_log.csv"
    with open(csvfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sid","sdd","det_spacing","det_count","score","stage"])
        for sid, sdd, dsp, dc, sc in coarse_log:
            writer.writerow([sid, sdd, dsp, dc, sc, "coarse"])
        for sid, sdd, dsp, dc, sc in refine_log:
            writer.writerow([sid, sdd, dsp, dc, sc, "refine"])
    print("Saved log:", csvfile)

    # pick best of refine vs coarse
    best = refine_best if refine_best['score'] > coarse_best['score'] else coarse_best
    best_reco = best['reco']
    out = "best_recon.png"
    v = np.clip((best_reco + 1000.0) / 3000.0, 0.0, 1.0)
    plt.imsave(out, v, cmap='gray', dpi=200)
    print("Saved best recon:", out)
    print("FINAL BEST GEOMETRY -> SID:", best['sid'], "SDD:", best['sdd'], "det_spacing:", best['det_spacing'], "det_count:", best['det_count'], "score:", best['score'])
