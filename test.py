#!/usr/bin/env python3
import h5py
from pathlib import Path

# adjust these to your layout:
ROOT_DCM  = Path("/home/hpc/iwi5/iwi5293h/CT-Motion-Artifact-Reduction_bkp/data/dicom_raw")
ROOT_H5   = Path("/home/vault/iwi5/iwi5293h/CT-Motion-Artifact-Reduction_bkp/data")

errors = []

for dcm_folder in sorted(ROOT_DCM.rglob("*")):
    if not dcm_folder.is_dir():
        continue

    # find .dcm files here
    dcm_files = list(dcm_folder.glob("*.dcm"))
    if not dcm_files:
        continue

    rel = dcm_folder.relative_to(ROOT_DCM)
    h5_path = ROOT_H5 / rel / "sinograms.h5"
    if not h5_path.exists():
        errors.append(f"❌ Missing HDF5 for folder: {rel}")
        continue

    # count datasets in the HDF5
    with h5py.File(h5_path, "r") as f:
        n_h5 = len(f.keys())

    n_dcm = len(dcm_files)
    if n_h5 != n_dcm:
        errors.append(f"❌ Mismatch in {rel}: DCM={n_dcm} vs H5={n_h5}")
    else:
        print(f"✅ {rel}: {n_dcm} ⇢ {n_h5}")

if errors:
    print("\nSummary of issues:")
    for e in errors:
        print("  ", e)
else:
    print("\n🎉 All folders match 1:1 DCM→sinogram datasets.")
