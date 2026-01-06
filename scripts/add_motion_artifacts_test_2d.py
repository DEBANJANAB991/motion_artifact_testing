#!/usr/bin/env python3
"""
Paper-style motion artifact simulation for 2D sinograms.

Applies smooth inter-projection rigid motion using Akima splines.
Designed for training/testing 2D sinogram models (MR_LKV, UNet, etc.)
"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.ndimage import shift
from pathlib import Path
from tqdm import tqdm
from config import CLEAN_SINOGRAM_2D_TEST, ARTIFACT_SINOGRAM_2D_TEST_v2

# ============================================================
# Generate smooth motion curves (Akima splines)
# ============================================================

def generate_motion_curve(
    num_views,
    max_shift_px=8.0,
    num_knots=6,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    knot_x = np.linspace(0, num_views - 1, num_knots)
    knot_y = np.random.uniform(-max_shift_px, max_shift_px, size=num_knots)
    knot_y -= knot_y.mean()  # zero-mean motion

    akima = Akima1DInterpolator(knot_x, knot_y)
    return akima(np.arange(num_views))


# ============================================================
# Apply motion to a 2D sinogram
# ============================================================

def apply_motion_to_2d_sinogram(
    sino_2d,
    max_shift_px=8.0,
    num_knots=6,
    seed=None
):
    """
    sino_2d: (V, U)
    """
    V, U = sino_2d.shape

    shift_curve = generate_motion_curve(
        V,
        max_shift_px=max_shift_px,
        num_knots=num_knots,
        seed=seed
    )

    out = np.zeros_like(sino_2d, dtype=np.float32)

    for v in range(V):
        out[v] = shift(
            sino_2d[v],
            shift=(shift_curve[v],),
            order=1,
            mode="constant",
            cval=0.0
        )

    return out


# ============================================================
# Batch processing
# ============================================================

def process_folder(
    clean_root: Path,
    out_root: Path,
    max_shift_px=8.0,
    num_knots=6
):
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(clean_root.glob("*.npy"))

    for f in tqdm(files, desc="Adding 2D motion"):
        sino = np.load(f)

        if sino.ndim != 2:
            raise ValueError(f"{f.name} is not 2D")

        motion = apply_motion_to_2d_sinogram(
            sino,
            max_shift_px=max_shift_px,
            num_knots=num_knots,
            seed=None
        )

        np.save(out_root / f.name, motion)


# ============================================================
# Example usage
# ============================================================

def main():
    CLEAN_2D = CLEAN_SINOGRAM_2D_TEST
    ART_2D   = ARTIFACT_SINOGRAM_2D_TEST_v2

    process_folder(
        clean_root=CLEAN_2D,
        out_root=ART_2D,
        max_shift_px=4.0,   # motion severity
        num_knots=6
    )


if __name__ == "__main__":
    main()
