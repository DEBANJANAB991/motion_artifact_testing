#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import CLEAN_SINOGRAM_ROOT, ARTIFACT_ROOT, DET_SPACING
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates


def apply_motion(sino_np: np.ndarray,
                 det_spacing: float,
                 n_nodes: int = 10,
                 max_trans: float = 10.0) -> np.ndarray:
    """
    Apply random motion artifacts to a sinogram of any size.

    Parameters
    ----------
    sino_np : np.ndarray
        Input sinogram of shape (n_views, n_detectors)
    det_spacing : float
        Detector pixel spacing in mm
    n_nodes : int
        Number of control points for motion splines
    max_trans : float
        Maximum translation in pixels

    Returns
    -------
    out : np.ndarray
        Motion-affected sinogram of same shape as sino_np
    """
    n_views, n_det = sino_np.shape

    # --- create periodic control points for translation and rotation ---
    t_nodes = np.linspace(0, n_views, n_nodes, endpoint=False)
    tx = np.random.uniform(-max_trans, max_trans, n_nodes)
    ty = np.random.uniform(-max_trans, max_trans, n_nodes)
    rot_nodes = np.random.uniform(-1.0, 1.0, n_nodes)

    # periodic endpoints
    tx[-1] = tx[0]
    ty[-1] = ty[0]
    rot_nodes[-1] = rot_nodes[0]

    # create splines
    cs_tx  = CubicSpline(t_nodes, tx, bc_type='periodic')
    cs_ty  = CubicSpline(t_nodes, ty, bc_type='periodic')
    cs_rot = CubicSpline(t_nodes, rot_nodes, bc_type='periodic')

    # evaluate per-view offsets
    view_idx = np.arange(n_views, dtype=np.float32)
    txs  = cs_tx(view_idx)
    tys  = cs_ty(view_idx)
    rzs  = np.deg2rad(cs_rot(view_idx))

    # prepare output sinogram
    out = np.zeros_like(sino_np)
    det_idx = np.arange(n_det, dtype=np.float32)

    # loop per view
    for i in range(n_views):
        # rotation along view axis
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

        # translation along detector axis
        shift = txs[i] * np.cos(2*np.pi*i/n_views) + tys[i] * np.sin(2*np.pi*i/n_views)
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
    artifact_root = Path(ARTIFACT_ROOT)
    artifact_root.mkdir(parents=True, exist_ok=True)

    # process each sinogram file
    sinogram_files = sorted(Path(CLEAN_SINOGRAM_ROOT).rglob("*.npy"))
    for sino_path in tqdm(sinogram_files, desc="Applying motion artifacts"):
        rel = sino_path.relative_to(CLEAN_SINOGRAM_ROOT)
        dst = artifact_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        # load sinogram
        sino = np.load(sino_path)

        # apply motion artifacts
        art_sino = apply_motion(sino, det_spacing=DET_SPACING)

        # save artifacted sinogram
        np.save(dst, art_sino)

        # optionally delete original
        #try:
         #   sino_path.unlink()
        #except Exception:
         #   pass

    print(f"\n✅ Generated artifacted sinograms in: {ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()
