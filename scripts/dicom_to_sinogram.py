#!/usr/bin/env python3
"""
Converts files from volume domain to projection domain
"""

import os, sys
from pathlib import Path
import numpy as np
import pydicom
import torch
from tqdm import tqdm

from config import (
    DATASET_PATH,
    SINOGRAM_ROOT,
    N_VIEWS,
    N_DET,
    DET_SPACING,
    SRC_ISO_PIXELS,
    SRC_DET_PIXELS,
    STEP_SIZE,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffct.differentiable import FanProjectorFunction

MAX_SAMPLES = 10000  


def dicom_to_hu(path: Path) -> np.ndarray:
    try:
        ds  = pydicom.dcmread(str(path), force=True)
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        return img * slope + inter
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return None


@torch.no_grad()
def slice_to_sino(img: np.ndarray, angles: torch.Tensor) -> np.ndarray:
    phantom = torch.from_numpy(img).to(angles.device)
    sino = FanProjectorFunction.apply(
        phantom,
        angles,
        N_DET, DET_SPACING,
        STEP_SIZE,
        SRC_DET_PIXELS, SRC_ISO_PIXELS
    )
    return sino.cpu().numpy()


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not available - abort.")

    device = torch.device("cuda")
    angles = torch.linspace(0, 2*np.pi, N_VIEWS,
                             device=device, dtype=torch.float32)

    processed = 0
    # Walk the DICOM tree
    for root, _, files in os.walk(DATASET_PATH):
        if processed >= MAX_SAMPLES:
            break

        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue

        out_dir = SINOGRAM_ROOT / Path(root).relative_to(DATASET_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Process each slice, but stop at MAX_SAMPLES
        for f in tqdm(dcm_files, desc=f"Converting ({processed}/{MAX_SAMPLES})"):
            if processed >= MAX_SAMPLES:
                break

            src = Path(root) / f
            dst = out_dir / (f.replace(".dcm", ".npy"))
            if dst.exists():
                continue

            img = dicom_to_hu(src)
            if img is None:
                continue

            sino = slice_to_sino(img, angles)
            np.save(dst, sino)
            processed += 1

      

    print(f"✅ Finished: wrote {processed} sinograms (limit {MAX_SAMPLES}) to {SINOGRAM_ROOT}")


if __name__ == "__main__":
    main()
