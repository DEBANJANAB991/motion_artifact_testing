#!/usr/bin/env python3
"""
motion_pipeline.py (clean version)

Applies rigid-motion artifacts to 2D fan-beam sinograms in projection domain.

- Loads clean sinograms (.npy) from CLEAN_SINOGRAM_ROOT
- Applies motion:
    mode = "simple"  → integer roll per angle (fast, robust)
    mode = "smooth"  → subpixel spline shifts (bonus)
- Saves artifacted sinograms in ARTIFACT_ROOT, keeping folder structure
- No reconstruction
- No arguments
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import map_coordinates
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------
# USER SETTINGS (EDIT THESE)
# ---------------------------------------------------------
MOTION_MODE = "simple"       # "simple" or "smooth"
MAX_SHIFT_PX = 8.0           # max detector shift (pixels)
SPLINE_KNOTS = 6             # used only for smooth spline mode
RNG_SEED = None              # set int for reproducible motion

# ---------------------------------------------------------
# PATHS (taken from your config)
# ---------------------------------------------------------
from config import CLEAN_SINOGRAM_ROOT, ARTIFACT_ROOT, DET_SPACING


# ---------------------------------------------------------
# MOTION CURVES
# ---------------------------------------------------------
def simple_random_roll_curve(n_views, max_shift_px=8.0, rng=None):
    """Smooth-ish integer roll curve."""
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, 2*np.pi, n_views, endpoint=False)
    slow = (max_shift_px * 0.6) * np.sin(1.0 * t)
    fast = (max_shift_px * 0.2) * np.sin(6.0 * t)
    jitter = rng.normal(0, max_shift_px * 0.08, n_views)

    return slow + fast + jitter


def smooth_spline_curve(n_views, max_shift_px=6.0, n_knots=6, rng=None):
    """Smooth periodic spline shift curve (subpixel)."""
    if rng is None:
        rng = np.random.default_rng()

    xs = np.linspace(0, n_views, n_knots, endpoint=False)
    ys = rng.uniform(-max_shift_px, max_shift_px, n_knots)
    ys[-1] = ys[0]  # enforce periodicity

    cs = CubicSpline(xs, ys, bc_type="periodic")
    return cs(np.arange(n_views))


# ---------------------------------------------------------
# MOTION APPLICATION
# ---------------------------------------------------------
def apply_motion_integer_roll(sino_np, shift_curve):
    """Fast rigid motion via integer roll per view."""
    n_views, _ = sino_np.shape
    out = np.empty_like(sino_np)

    for i in range(n_views):
        shift_px = int(round(shift_curve[i]))
        out[i] = np.roll(sino_np[i], shift_px)

    return out


def apply_motion_subpixel(sino_np, shift_curve):
    """Subpixel shifting using map_coordinates."""
    n_views, n_det = sino_np.shape
    det_idx = np.arange(n_det, dtype=np.float32)
    out = np.zeros_like(sino_np)

    for i in range(n_views):
        shift = float(shift_curve[i])

        coords = np.vstack((np.full(n_det, i, np.float32), det_idx - shift))
        row = map_coordinates(sino_np, coords, order=1, mode="nearest")
        out[i] = row

    return out


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)

    CLEAN_ROOT = Path(CLEAN_SINOGRAM_ROOT)
    ART_ROOT = Path(ARTIFACT_ROOT)
    ART_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted(CLEAN_ROOT.rglob("*.npy"))
    print(f"Found {len(files)} clean sinograms.")
    print(f"Motion mode: {MOTION_MODE}, max_shift_px={MAX_SHIFT_PX}")

    for src in tqdm(files, desc="Applying motion"):
        sino_np = np.load(src).astype(np.float32)
        n_views, _ = sino_np.shape

        # Generate shift curve
        if MOTION_MODE == "simple":
            curve = simple_random_roll_curve(n_views, MAX_SHIFT_PX, rng)
            art = apply_motion_integer_roll(sino_np, curve)

        elif MOTION_MODE == "smooth":
            curve = smooth_spline_curve(n_views, MAX_SHIFT_PX, SPLINE_KNOTS, rng)
            art = apply_motion_subpixel(sino_np, curve)

        else:
            raise ValueError("MOTION_MODE must be 'simple' or 'smooth'")

        # Save artifacted sinogram preserving structure
        rel = src.relative_to(CLEAN_ROOT)
        dst = ART_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, art)

    print(f"\n✅ Done! Artifacted sinograms saved to:\n   {ART_ROOT.resolve()}")


if __name__ == "__main__":
    main()
