#!/usr/bin/env python3
"""
apply_cubic_spline_motion_to_sinograms.py

Load clean cone-beam sinograms (.npy) from CLEAN_SINOGRAM_ROOT1 (from config.py),
apply cubic-spline rotation + translation motion (Option A), and write the
artifacted sinograms to ARTIFACT_TEST_ROOT (from config.py).

Output:
 - ARTIFACT_TEST_ROOT/<stem>_artifact.npy

No JSON or PNG files are created by this script.
"""

import os
import math
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm
from scipy.ndimage import rotate, map_coordinates

# Import your project config which must define CLEAN_SINOGRAM_ROOT1 and ARTIFACT_TEST_ROOT
from config import TEST_CLEAN_SINOGRAM, TEST_ARTIFACT_ROOT

# ---------------------------------------------------------------------
# User parameters (tweak if you want)
# ---------------------------------------------------------------------
MAX_FILES: Optional[int] = None # process all files if None, else limit
SEED = 12345

# Default cubic-spline motion parameters (these match the example you posted)
# You can override per-run by editing these lists.
DEFAULT_ROT_EVENTS = [250, 320]       # projection indices where rotations are applied
DEFAULT_ROT_DEGS   = [3.0, -4.0]      # degrees for each rotation event
DEFAULT_TRANS_EVENTS = [260, 330]     # projection indices where translations are applied
DEFAULT_TRANS_PX     = [10.0, -14.0]  # pixel shifts (u-axis) for each translation event

# If a listed event index >= num_views it will be ignored for that sinogram.
# ---------------------------------------------------------------------

# -------------------------
# Helpers
# -------------------------
def seed_everything(seed: int):
    np.random.seed(seed)


def load_sino(path: Path) -> np.ndarray:
    """Load a .npy sinogram from disk."""
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D sinogram (views, det_u, det_v), got shape {arr.shape} for {path}")
    return arr.astype(np.float32)


# ---------------------------------------------------------------------
# Core: cubic-spline rotation + translation function (Option A)
# ---------------------------------------------------------------------
def apply_cubic_spline_motion(
    sino: np.ndarray,
    angles: Optional[np.ndarray] = None,
    rot_events: Optional[List[int]] = None,
    rot_degs: Optional[List[float]] = None,
    trans_events: Optional[List[int]] = None,
    trans_px: Optional[List[float]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply cubic-spline interpolation for rotation + translation to a sinogram.

    Parameters
    ----------
    sino : np.ndarray
        Input sinogram shape (num_views, det_u, det_v).
    angles : Optional[np.ndarray]
        Per-view angles (radians). If provided and rotations are applied,
        returned angles will be modified accordingly.
    rot_events : list[int]
        Projection indices where a rotation event occurs.
    rot_degs : list[float]
        Rotation magnitudes (degrees) corresponding to rot_events.
    trans_events : list[int]
        Projection indices where a translation event occurs.
    trans_px : list[float]
        Translation amounts in pixels (u-direction) corresponding to trans_events.

    Returns
    -------
    sino_out, angles_out
        The motion-applied sinogram and (optionally) the modified angles array
        if `angles` was passed in. If `angles` was None, returns (sino_out, None).
    """

    # Use the provided defaults if None
    if rot_events is None:
        rot_events = list(DEFAULT_ROT_EVENTS)
    if rot_degs is None:
        rot_degs = list(DEFAULT_ROT_DEGS)
    if trans_events is None:
        trans_events = list(DEFAULT_TRANS_EVENTS)
    if trans_px is None:
        trans_px = list(DEFAULT_TRANS_PX)

    sino_out = sino.copy()
    if angles is not None:
        angles_out = angles.copy()
    else:
        angles_out = None

    num_views, det_u, det_v = sino.shape

    # --- Rotation events: update per-view angles and rotate projection images ---
    # For each rotation event (ev, deg) we:
    # - add deg (in radians) to all subsequent angles
    # - rotate each projection image from ev .. end by `deg` degrees (cubic)
    for ev, deg in zip(rot_events, rot_degs):
        if ev < 0:
            continue
        if ev >= num_views:
            # skip events beyond available views
            continue

        # modify angles if present
        if angles_out is not None:
            angles_out[ev:] = angles_out[ev:] + np.deg2rad(deg)

        # rotate pixel image for each projection >= ev
        # NOTE: rotate(..., reshape=False) keeps the same array shape
        for v in range(ev, num_views):
            # order=3 -> cubic interpolation
            # mode='nearest' keeps out-of-bounds reasonable
            try:
                sino_out[v] = rotate(sino_out[v], angle=deg, reshape=False, order=3, mode="nearest")
            except Exception:
                # fallback to linear if cubic fails
                sino_out[v] = rotate(sino_out[v], angle=deg, reshape=False, order=1, mode="nearest")

    # --- Translation events: apply per-projection translations (u direction) with cubic interpolation ---
    # For each translation event (ev, shift) shift all subsequent projections by `shift` pixels in u.
    # We'll use map_coordinates with order=3 (cubic).
    # Note: we only translate along the u-axis (first axis of projection image),
    # keeping v coordinate the same (like in your example).
    if len(trans_events) > 0:
        # precompute grid coordinates
        grid_u, grid_v = np.meshgrid(np.arange(det_u), np.arange(det_v), indexing="ij")  # shapes (det_u, det_v)

        for ev, shift in zip(trans_events, trans_px):
            if ev < 0:
                continue
            if ev >= num_views:
                continue

            # translate projections v in range [ev, end)
            for v in range(ev, num_views):
                # new_u = grid_u - shift  (we sample input at shifted coords -> effectively translate)
                coords = np.vstack(( (grid_u - shift).ravel(), grid_v.ravel() ))
                try:
                    warped = map_coordinates(sino_out[v], coords, order=3, mode='nearest')
                except Exception:
                    # fallback to linear interpolation if cubic fails
                    warped = map_coordinates(sino_out[v], coords, order=1, mode='nearest')
                sino_out[v] = warped.reshape(det_u, det_v)

    return sino_out, angles_out


# ---------------------------------------------------------------------
# Main loop: process all .npy sinograms in CLEAN_SINOGRAM_ROOT
# ---------------------------------------------------------------------
def main():
    seed_everything(SEED)

    src = Path(TEST_CLEAN_SINOGRAM)
    dst = Path(TEST_ARTIFACT_ROOT)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir() if p.suffix.lower() == ".npy"])
    if not files:
        print("No .npy sinograms found in", src)
        return

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    print(f"Processing {len(files)} sinograms from {src} -> {dst} (Option A cubic-spline motion)")

    for p in tqdm(files, desc="Applying cubic-spline motion"):
        try:
            sino_clean = load_sino(p)
            num_views = sino_clean.shape[0]

            # If file has an accompanying .json with geometry, attempt to use angles from it.
            # If not present, we will not modify angles (angles_in = None).
            angles_in = None
            meta_path = p.with_suffix(".json")
            if meta_path.exists():
                try:
                    import json
                    meta = json.loads(meta_path.read_text(encoding="utf8"))
                    geom = meta.get("geometry", {})
                    num_views_meta = int(geom.get("NUM_VIEWS", num_views))
                    # angles: create default angles if geometry suggests number of views
                    if num_views_meta == num_views:
                        angles_in = np.linspace(0.0, 2.0 * math.pi, num_views + 1)[:-1].astype(np.float32)
                    else:
                        # fallback: build angles matching actual sinogram length
                        angles_in = np.linspace(0.0, 2.0 * math.pi, num_views + 1)[:-1].astype(np.float32)
                except Exception:
                    angles_in = np.linspace(0.0, 2.0 * math.pi, num_views + 1)[:-1].astype(np.float32)
            else:
                # build default equally spaced angles
                angles_in = np.linspace(0.0, 2.0 * math.pi, num_views + 1)[:-1].astype(np.float32)

            # Apply the cubic-spline motion with the example event lists (Option A)
            sino_art, _ = apply_cubic_spline_motion(
                sino_clean,
                angles=angles_in,
                rot_events=DEFAULT_ROT_EVENTS,
                rot_degs=DEFAULT_ROT_DEGS,
                trans_events=DEFAULT_TRANS_EVENTS,
                trans_px=DEFAULT_TRANS_PX
            )

            # Save artifacted sinogram (float32)
            out_path = dst / f"{p.stem}_artifact.npy"
            np.save(out_path, sino_art.astype(np.float32))

        except Exception as e:
            print(f"Failed {p.name}: {e}")
            continue

    print("Done. Artifacted sinograms saved in:", dst)


if __name__ == "__main__":
    main()
