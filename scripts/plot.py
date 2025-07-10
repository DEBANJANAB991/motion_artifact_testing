#!/usr/bin/env python3
"""
convert_and_artifact_sinograms.py

Walks a DICOM-derived NumPy sinogram directory, applies per-angle
rigid-motion artifacts (translation + in-plane rotation) via periodic
splines, and writes out the artifacted sinograms.
"""
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from config import (
    CLEAN_SINOGRAM_ROOT,
    ARTIFACT_ROOT,
    N_VIEWS,
    DET_SPACING,
)
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates


def apply_motion(
    sino_np: np.ndarray,
    n_views: int = N_VIEWS,
    det_spacing: float = DET_SPACING,
    n_nodes: int = 10,
    max_trans: float = 5.0,   # pixels
    max_rot: float = 1.0      # degrees
) -> np.ndarray:
    """
    Introduce per-angle rigid motion into a 2D fan-beam sinogram:
      - In-plane translations (±max_trans pixels)
      - In-plane rotations about z-axis (±max_rot degrees)

    Both translation and rotation are parameterized by periodic splines
    over `n_nodes` control points.
    """
    # 1) build control-point grid
    t_nodes = np.linspace(0, n_views, n_nodes, endpoint=False)

    # sample random offsets at control points
    tx_nodes = np.random.uniform(-max_trans, max_trans, n_nodes)
    ty_nodes = np.random.uniform(-max_trans, max_trans, n_nodes)
    rot_nodes = np.random.uniform(-max_rot, max_rot, n_nodes)
    # enforce periodicity
    tx_nodes[-1] = tx_nodes[0]
    ty_nodes[-1] = ty_nodes[0]
    rot_nodes[-1] = rot_nodes[0]

    # create periodic splines
    cs_tx  = CubicSpline(t_nodes, tx_nodes,  bc_type='periodic')
    cs_ty  = CubicSpline(t_nodes, ty_nodes,  bc_type='periodic')
    cs_rot = CubicSpline(t_nodes, rot_nodes, bc_type='periodic')

    # 2) evaluate per-angle offsets
    angles = np.arange(n_views, dtype=np.float32)
    txs  = cs_tx(angles)                     # pixel shifts
    tys  = cs_ty(angles)
    rzs  = np.deg2rad(cs_rot(angles))        # convert to radians

    # prepare output
    n_det = sino_np.shape[1]
    det_idx = np.arange(n_det, dtype=np.float32)
    out = np.zeros_like(sino_np)

    # 3) loop per view: first rotate, then translate
    for i in range(n_views):
        #--- rotation (view-axis warp) ---
        # map_coordinates expects coords in (row, col) order
        # shifting along the view axis by rot_offset
        rot_offset = rzs[i] * n_views / (2 * np.pi)
        coords = np.vstack((
            np.full(n_det, i + rot_offset, dtype=np.float32),
            det_idx
        ))
        row_rot = map_coordinates(
            sino_np,
            coords,
            order=1,
            mode='constant',
            cval=0.0
        )

        #--- translation (detector-axis shift) ---
        shift = txs[i] * np.cos(2*np.pi*i/n_views) + \
                tys[i] * np.sin(2*np.pi*i/n_views)
        shift_px = shift / det_spacing
        row_trans = np.interp(
            det_idx,
            det_idx - shift_px,
            row_rot,
            left=0.0,
            right=0.0
        )

        out[i] = row_trans

    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_root = Path(ARTIFACT_ROOT)
    artifact_root.mkdir(parents=True, exist_ok=True)

    # process each sinogram file
    for sino_path in tqdm(sorted(Path(CLEAN_SINOGRAM_ROOT).rglob("*.npy")),
                           desc="Applying motion artifacts"):
        rel = sino_path.relative_to(CLEAN_SINOGRAM_ROOT)
        dst = artifact_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        sino = np.load(sino_path)
        art_sino = apply_motion(sino)
        np.save(dst, art_sino)

        # optionally remove original
        try:
            sino_path.unlink()
        except Exception:
            pass

    print(f"✅ Generated artifacted sinograms in: {ARTIFACT_ROOT}")


if __name__ == "__main__":
    # Plot and save the first 5 samples; adjust as needed
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 6):
        rel = sorted(Path(CLEAN_SINOGRAM_ROOT).rglob("*.npy"))[i-1].relative_to(CLEAN_SINOGRAM_ROOT)
        clean_path = Path(CLEAN_SINOGRAM_ROOT) / rel
        art_path = Path(ARTIFACT_ROOT) / rel
        if not art_path.exists():
            print(f"⚠️ Missing artifact for {rel}")
            continue
        clean = np.load(clean_path)
        art = np.load(art_path)
        diff = art - clean

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax in axes:
            ax.axis('off')
        axes[0].imshow(clean,   cmap='gray', vmin=clean.min(), vmax=clean.max())
        axes[0].set_title(f"Sample {i}: Clean")
        axes[1].imshow(art,     cmap='gray', vmin=clean.min(), vmax=clean.max())
        axes[1].set_title("Motion Artifact")
        v = np.max(np.abs(diff))
        axes[2].imshow(diff,    cmap='RdBu',  vmin=-v, vmax=v)
        axes[2].set_title("Difference")
        plt.tight_layout()
        # save figure
        fig.savefig(out_dir / f"comparison_sample_{i}.png", bbox_inches='tight')
        plt.close(fig)
    print(f"✅ Saved comparison plots to {out_dir}")
