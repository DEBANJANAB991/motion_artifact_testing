#!/usr/bin/env python3
"""
FDK reconstruction (cone-beam) that reads sinogram .npy files and writes PNG images only.
No .npy reconstructions are saved.

Usage:
    python fdk_reconstruct_only_png.py
"""
import os
import math
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from diffct.differentiable import ConeBackprojectorFunction
from config import TEST_ARTIFACTED_SINOGRAM  # folder containing .npy sinograms (and optional .json metadata)
#from config import ARTIFACTED_SINOGRAM_3D
# -------------------------gi
# User parameters
# -------------------------
SINO_DIR = Path(TEST_ARTIFACTED_SINOGRAM)
OUT_ROOT = Path.cwd() / "artifact_reconstruction_png"   # output folder (per-sinogram subfolders)
OUT_ROOT.mkdir(exist_ok=True)

MAX_FILES = 10            # number of sinograms to reconstruct (sorted)
USE_GPU = True            # use GPU if available
NUM_VIEWS_DEFAULT = 540 #360   # fallback if metadata is missing
SID_DEFAULT = 530.0
SDD_DEFAULT = 1095.0 #1086.0
DU_DEFAULT = 1.0
DV_DEFAULT = 1.0
RECO_SHAPE_DEFAULT = 512 #None  # if None -> use DET_V for cubic grid (requires DET_V present)

# -------------------------
# Helpers
# -------------------------
def load_sino_and_meta(path: Path) -> Tuple[np.ndarray, dict]:
    """Load sinogram .npy and optional .json metadata (same stem)."""
    sino = np.load(path)
    meta_path = path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf8"))
        except Exception:
            meta = {}
    return sino, meta

def ramp_filter_windowed(sino: torch.Tensor, cutoff_ratio: float = 0.9) -> torch.Tensor:
    """
    1D ramp filter along detector-u (dim=1) with Hann window to reduce ringing.
    sino: (views, det_u, det_v)
    """
    device = sino.device
    _, det_u, _ = sino.shape

    freqs = torch.fft.fftfreq(det_u, device=device)
    omega = 2.0 * math.pi * freqs
    ramp = torch.abs(omega).reshape(1, det_u, 1)

    # Hann window around cutoff
    fmax = freqs.abs().max().item() if freqs.numel() > 0 else 0.0
    if fmax == 0:
        hann = torch.ones_like(freqs, device=device)
    else:
        cutoff = cutoff_ratio * fmax
        x = freqs / (cutoff + 1e-12)
        x = torch.clamp(x, -1.0, 1.0)
        hann = 0.5 * (1.0 + torch.cos(x * math.pi))
    hann = hann.reshape(1, det_u, 1)

    filt = ramp * hann
    S = torch.fft.fft(sino, dim=1)
    S_f = S * filt
    out = torch.real(torch.fft.ifft(S_f, dim=1))
    return out

def save_slices_per_slice_norm(volume: np.ndarray, out_folder: Path, prefix: str = "slice"):
    """
    Normalize and save each slice independently so PNGs are not gray.
    volume: numpy array with shape (Z, Y, X)
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    Nz = volume.shape[0]
    for i in range(Nz):
        sl = volume[i].astype(np.float32)
        # per-slice normalization
        sl = sl - np.min(sl)
        mx = np.max(sl)
        if mx > 0:
            sl = sl / mx
        else:
            sl = np.zeros_like(sl)
        img = (sl * 255.0).astype(np.uint8)
        Image.fromarray(img).save(out_folder / f"{prefix}_{i:03d}.png")

def save_mid_preview(volume: np.ndarray, out_file: Path, title: str = ""):
    """Save a single mid-slice preview using matplotlib (better display)."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mid = volume.shape[0] // 2
    plt.figure(figsize=(6,6))
    plt.imshow(volume[mid], cmap="gray", interpolation="nearest")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=150)
    plt.close()

# -------------------------
# Main per-sinogram reconstruction
# -------------------------
def reconstruct_one(path: Path, device: torch.device):
    sino_np, meta = load_sino_and_meta(path)

    if sino_np.ndim != 3:
        raise ValueError(f"Sinogram {path.name} shape {sino_np.shape} not supported (expect 3D)")

    views, det_u, det_v = sino_np.shape

    # read geometry from metadata if present, else use defaults
    geom = meta.get("geometry", {})
    NUM_VIEWS = int(geom.get("NUM_VIEWS", views if views else NUM_VIEWS_DEFAULT))
    SID = float(geom.get("SID", SID_DEFAULT))
    SDD = float(geom.get("SDD", SDD_DEFAULT))
    DU = float(geom.get("DU", DU_DEFAULT))
    DV = float(geom.get("DV", DV_DEFAULT))

    # determine reconstruction grid size
    if "reco_shape" in meta:
        RECO_Z, RECO_Y, RECO_X = meta["reco_shape"]
    elif RECO_SHAPE_DEFAULT is not None:
        RECO_Z = RECO_Y = RECO_X = RECO_SHAPE_DEFAULT
    else:
        # default: use det_v (height) as cubic grid
        RECO_Z = RECO_Y = RECO_X = det_v

    # angles (consistent with projection creation)
    angles = torch.linspace(0.0, 2.0 * math.pi, NUM_VIEWS + 1, dtype=torch.float32, device=device)[:-1]

    # convert sino to tensor on device
    sino = torch.tensor(sino_np.astype(np.float32), device=device).contiguous()

    # cone-beam weighting (per projection sample)
    u = (torch.arange(det_u, device=device) - (det_u - 1) / 2.0) * DU
    v = (torch.arange(det_v, device=device) - (det_v - 1) / 2.0) * DV
    uu, vv = torch.meshgrid(u, v, indexing="ij")  # (det_u, det_v)
    D = float(SDD)
    W = D / torch.sqrt(D * D + uu*uu + vv * vv)  # (det_u, det_v)
    W = W.unsqueeze(0)  # (1, det_u, det_v)
    sino_w = sino * W

    # ramp filter with window
    sino_filt = ramp_filter_windowed(sino_w, cutoff_ratio=0.9).contiguous()

    # voxel_size — best read from metadata if present
    #voxel_size = float(meta.get("voxel_size", DU * (SID / SDD)))  # fallback heuristic
    voxel_size = float(meta.get("voxel_size", 0.5))   # or your actual PixelSpacing

    # Backprojection
    reco = ConeBackprojectorFunction.apply(
        sino_filt, angles,
        int(RECO_Z), int(RECO_Y), int(RECO_X),
        DU, DV,
        SDD, SID, voxel_size
    )

    # normalization factor for discrete angular sampling (FDK)
    reco = reco * (math.pi / float(NUM_VIEWS))

    reco_np = reco.detach().cpu().numpy().astype(np.float32)

    # Save outputs (PNG only)
    base = path.stem
    out_folder = OUT_ROOT / base
    # per-slice PNGs
    save_slices_per_slice_norm(reco_np, out_folder / "slices", prefix="slice")
    # mid-slice preview (matplotlib)
    save_mid_preview(reco_np, out_folder / f"{base}_mid.png", title=base)
    # also save sinogram preview image (optional)
    # create central-row/column preview and save
    try:
        # plot same preview as earlier pipeline
        num_views, U, V = sino_np.shape
        mid_u = U // 2
        mid_v = V // 2
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        axs[0].imshow(sino_np[:, :, mid_v].T, cmap="gray", aspect="auto"); axs[0].set_title("Central detector-row")
        axs[1].imshow(sino_np[:, mid_u, :].T, cmap="gray", aspect="auto"); axs[1].set_title("Central detector-column")
        axs[2].imshow(sino_np[num_views//2], cmap="gray"); axs[2].set_title("One projection")
        plt.tight_layout()
        preview_png = out_folder / f"{base}_sino_preview.png"
        out_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(preview_png, dpi=150)
        plt.close()
    except Exception:
        pass

    return out_folder

# -------------------------
# Entrypoint
# -------------------------
def main():
    device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    files = sorted([p for p in SINO_DIR.iterdir() if p.suffix == ".npy"])
    files = files[:MAX_FILES]
    if not files:
        print("No .npy sinograms found in", SINO_DIR)
        return

    print(f"Found {len(files)} sinograms; reconstructing up to {MAX_FILES}...")

    for p in tqdm(files, desc="Reconstructing"):
        try:
            out = reconstruct_one(p, device)
            print("Saved PNGs in:", out)
        except Exception as e:
            print(f"Failed {p.name}: {e}")

    print("Done. All PNG outputs saved to:", OUT_ROOT)

if __name__ == "__main__":
    main()
