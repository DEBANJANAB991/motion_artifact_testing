#!/usr/bin/env python3
"""
Hard-coded DICOM path version with automatic JSON-safe conversion.
"""
import os
import json
import pydicom
from pprint import pprint
from pydicom.datadict import tag_for_keyword
from pydicom.multival import MultiValue
from config import DATASET_PATH
# ---------------------------------------------------------
# EDIT THIS ONLY:
DICOM_PATH = DATASET_PATH
# ---------------------------------------------------------


def to_json_safe(value):
    """Convert DICOM values (including MultiValue) into JSON-safe types."""
    if value is None:
        return None

    # MultiValue -> list
    if isinstance(value, MultiValue) or isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]

    # Numeric values
    if isinstance(value, (int, float, str)):
        return value

    # Pydicom value wrappers -> convert to str
    try:
        return str(value)
    except:
        return None


def find_dicom(path):
    """Find first readable DICOM file in folder."""
    if os.path.isfile(path):
        return path

    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                _ = pydicom.dcmread(fp, stop_before_pixels=True)
                return fp
            except:
                continue

    raise FileNotFoundError("No DICOM file found in the given path.")


def safe_get(ds, keyword):
    """Extract DICOM field safely."""
    try:
        if hasattr(ds, keyword):
            return to_json_safe(getattr(ds, keyword))

        tag = tag_for_keyword(keyword)
        elem = ds.get(tag)
        if elem is not None:
            return to_json_safe(elem.value)
    except:
        return None
    return None


def extract_fields(ds):
    """Important geometry & scaling fields."""
    keys = [
        "Manufacturer", "ManufacturerModelName", "StationName",
        "DistanceSourceToDetector", "DistanceSourceToPatient",
        "GantryDetectorTilt", "ReconstructionDiameter",
        "Rows", "Columns", "PixelSpacing",
        "SliceThickness", "RescaleIntercept", "RescaleSlope",
        "DetectorElementSpacing", "DetectorElementSize",
        "KVP", "ImagePositionPatient", "ImageOrientationPatient",
    ]
    return {k: safe_get(ds, k) for k in keys}


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
dicom_file = find_dicom(DICOM_PATH)
print("Found DICOM file:", dicom_file)

ds = pydicom.dcmread(dicom_file, stop_before_pixels=False)

# Save full header
with open("header.txt", "w", encoding="utf-8") as f:
    f.write(str(ds))
print("Saved: header.txt")

# Extracted fields
extracted = extract_fields(ds)

print("\n=== Extracted geometry fields ===")
pprint(extracted)

# Save JSON safely
with open("header_extracted.json", "w", encoding="utf-8") as f:
    json.dump(extracted, f, indent=2)

print("Saved: header_extracted.json")
print("\nDone! Upload header_extracted.json here.")
