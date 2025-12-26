# config.py  (root of your repo)

from pathlib import Path

# ------------------------------------------------------------------
# Mandatory paths  – adjust if your data live somewhere else
# ------------------------------------------------------------------
DATASET_PATH   = Path("/home/vault/iwi5/iwi5293h/CT-Motion-Artifact-Reduction_bkp/data/dicom_raw")


#CLEAN_SINOGRAM_ROOT = DATASET_PATH.parent / "sinograms"
#ARTIFACT_ROOT = Path("/home/vault/iwi5/iwi5293h/CT-Motion-Artifact-Reduction_bkp/data/art_sinograms") #name to be changed back to ARTIFACT_ROOT 
CLEAN_SINOGRAM_ROOT=Path("/home/woody/iwi5/iwi5293h/clean_sinograms")
ARTIFACT_ROOT=Path("/home/woody/iwi5/iwi5293h/art_sinograms")
TEST_CLEAN_SINOGRAM = Path("/home/woody/iwi5/iwi5293h/clean_sinograms_test")
TEST_ARTIFACT_ROOT = Path("/home/woody/iwi5/iwi5293h/art_sinograms_test")
CLEAN_SINOGRAM_2D=Path("/home/woody/iwi5/iwi5293h/clean_sinograms_2d")
ARTIFACT_SINOGRAM_2D=Path("/home/woody/iwi5/iwi5293h/art_sinograms_2d")
CLEAN_SINOGRAM_2D_TEST=Path("/home/woody/iwi5/iwi5293h/clean_sinograms_2d_test")
ARTIFACT_SINOGRAM_2D_TEST=Path("/home/woody/iwi5/iwi5293h/art_sinograms_2d_test")
PREDICTED_SINOGRAM_2D_TEST=Path("/home/woody/iwi5/iwi5293h/predicted_sinograms_2d_test")
MERGED_SINOGRAM_3D_TEST=Path("/home/woody/iwi5/iwi5293h/merged_sinograms_3d_test")
# Where to save model checkpoints (scratch)

CKPT_DIR=Path("/home/vault/iwi5/iwi5293h/models")


RECON_ROOT     = DATASET_PATH.parent / "recon"    

# ------------------------------------------------------------------
# Fan-beam geometry (pixel units, square detector)
# ------------------------------------------------------------------
N_VIEWS         = 360
N_DET           = 512
STEP_SIZE       = 0.5
MAX_SAMPLES=20000
MAX_SAMPLES_TEST=100
SID = 560.0           # Source → Isocenter (mm)
SDD = 1100.0          # Source → Detector (mm)
DET_SPACING = 1.2     # Detector spacing (mm)
DET_COUNT = 736       # Number of detector elements
VOXEL_SPACING = 1.0   # MUST remain 1.0 for diffct
# Training hyperparameters
# ------------------------------------------------------------------
# Total number of epochs
EPOCHS = 100
MU_WATER = 0.019 

# Batch size per GPU
BATCH_SIZE = 16
BATCH_SIZE=8


# Learning rate for optimizer
LR = 1e-3

# How often (in epochs) to save a checkpoint
SAVE_INTERVAL = 10