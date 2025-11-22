#!/usr/bin/env python3
import os
import numpy as np
import torch
import pydicom
import matplotlib.pyplot as plt
from config import DATASET_PATH 
# diffct cone-beam operators
from diffct.differentiable import ConeProjectorFunction, ConeBackprojectorFunction


# ============================================================
# 1. EXTRACT GEOMETRY FROM DICOM
# ============================================================
def extract_geometry(ds):
    """Return SID, SDD, detector resolution, detector pixel sizes, voxel spacing."""
    # Voxel spacing
    spacing_x, spacing_y = map(float, ds.PixelSpacing)
    spacing_z = float(ds.SliceThickness)

    # Philips DICOM uses these fields consistently
    sid = float(ds.DistanceSourceToPatient)            # source–isocenter
    sdd = float(ds.DistanceSourceToDetector)           # source–detector

    # Approx detector pixel size based on FOV
    # This is not perfect, but good enough for cone-beam testing
    det_u_size = float(ds.ReconstructionDiameter) / ds.Columns
    det_v_size = det_u_size

    # Detector resolution (choose square detector)
    det_u = 512
    det_v = 512

    return sid, sdd, det_u, det_v, det_u_size, det_v_size, (spacing_x, spacing_y, spacing_z)


# ============================================================
# 2. BUILD 3D VOLUME FROM SINGLE 2D SLICE
# ============================================================
def make_3d_volume_from_slice(img2d, nz=128):
    """
    Create a fake 3-D volume by stacking the same CT slice along z.
    Cone-beam projector requires 3D input.
    """
    vol = np.repeat(img2d[np.newaxis, :, :], nz, axis=0)
    return torch.tensor(vol, dtype=torch.float32).cuda()


# ============================================================
# 3. CONE-BEAM FORWARD PROJECTION
# ============================================================
def cone_forward(volume, sid, sdd, det_u, det_v, du, dv, n_views=360):
    angles = torch.linspace(0, 2*np.pi, n_views, device="cuda")
    sino = ConeProjectorFunction.apply(
        volume,
        angles,
        det_u, det_v,
        du, dv,
        sdd, sid,
        voxel_spacing=1.0
    )
    return sino, angles


# ============================================================
# 4. CONE-BEAM BACKPROJECTION
# ============================================================
def cone_backproject(sino, angles, D, H, W, du, dv, sdd, sid):
    reco = ConeBackprojectorFunction.apply(
        sino,
        angles,
        D, H, W,
        du, dv,
        sdd, sid,
        voxel_spacing=1.0
    )
    return reco


# ============================================================
# 5. FULL PIPELINE
# ============================================================
def main():

    # ============================================================
    # CHANGE THIS TO YOUR REAL CQ500 DICOM
    # ============================================================
    DICOM_PATH = DATASET_PATH

    # 1. Load DICOM
    ds = pydicom.dcmread(DICOM_PATH)
    img2d = ds.pixel_array.astype(np.float32)

    # 2. Extract geometry
    sid, sdd, det_u, det_v, du, dv, (sx, sy, sz) = extract_geometry(ds)

    # 3. Create fake 3D volume
    D = 128
    H, W = img2d.shape
    volume = make_3d_volume_from_slice(img2d, nz=D)

    # 4. Forward projection
    sino, angles = cone_forward(volume, sid, sdd, det_u, det_v, du, dv, n_views=360)

    # 5. Backprojection
    reco = cone_backproject(sino, angles, D, H, W, du, dv, sdd, sid)

    # 6. Display result
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.title("Original Slice"); plt.imshow(img2d, cmap='gray')
    plt.subplot(1, 2, 2); plt.title("Reconstruction (Cone-beam)"); plt.imshow(reco[D//2].cpu(), cmap='gray')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
