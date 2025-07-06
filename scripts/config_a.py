# config.py  (root of your repo)

from pathlib import Path

# ------------------------------------------------------------------
# Mandatory paths  – adjust if your data live somewhere else
# ------------------------------------------------------------------
DATASET_PATH   = Path("/home/vault/iwi5/iwi5293h/CT-Motion-Artifact-Reduction_bkp/data/dicom_raw")
/home/vault/iwi5/iwi5293h/CT-Motion-Artifact-Reduction_bkp/data/sinograms_artifact/dicom_raw
CLEAN_SINOGRAM_ROOT = Path("/scratch/iwi5/iwi5293h/sinograms/dicom_raw")

ARTIFACT_ROOT  = DATASET_PATH.parent / "sinograms_artifact"

SINOGRAM_ROOT  = DATASET_PATH.parent / "sinograms"    

# Where to save model checkpoints (scratch)
CKPT_DIR = Path("/scratch/iwi5/iwi5293h/models")


RECON_ROOT     = DATASET_PATH.parent / "recon"    

# ------------------------------------------------------------------
# Fan-beam geometry (pixel units, square detector)
# ------------------------------------------------------------------
N_VIEWS         = 360
N_DET           = 512
DET_SPACING     = 1.0
SRC_ISO_PIXELS  = 500.0   # source-to-iso
SRC_DET_PIXELS  = 1000.0  # source-to-detector
STEP_SIZE       = 0.5
# Training hyperparameters
# ------------------------------------------------------------------
# Total number of epochs
EPOCHS = 100

# Batch size per GPU
BATCH_SIZE = 16

# Learning rate for optimizer
LR = 1e-3

# How often (in epochs) to save a checkpoint
SAVE_INTERVAL = 10