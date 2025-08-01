#!/bin/bash -l
#
# SLURM Job Script for Optuna Hyperparameter Optimization of FixMatch Model
#
# ===============================
# SLURM Directives
# ===============================
#SBATCH --gres=gpu:a100:1               # Request 1 NVIDIA rtx3080 GPU
#SBATCH --partition=a100               # Specify the GPU partition rtx3080
#SBATCH --time=24:00:00                 # Maximum runtime of 24 hours
#SBATCH --export=NONE                   # Do not export current environment variables
#SBATCH --job-name=REPLKNET        # Job name
#SBATCH --output=REPLKNET.out      # Standard output log file (%j expands to job ID)
#SBATCH --error=REPLKNET.err       # Standard error log file (%j expands to job ID)
 
# ===============================
# Environment Configuration
# ===============================

 
# Set HTTP and HTTPS proxies if required (uncomment and modify if needed)
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
 
# Unset SLURM_EXPORT_ENV to prevent SLURM from exporting environment variables
unset SLURM_EXPORT_ENV
 
# Load the necessary modules
module load python/3.12-conda        # Load Python Anaconda module
 
# Activate the Conda environment
conda activate /home/woody/iwi5/iwi5293h/software/private/conda/envs/thesis-gpu  # Replace with your Conda environment path
 
# ===============================
# Navigate to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5293h/Debanjana_Master_Thesis/scripts  # Replace with your script directory path
 




 
# ===============================
# Execute the Python Training Script
# ===============================
 
# Run the Optuna-based FixMatch HPO Python script with necessary arguments
#python3 -u train_test_split.py --model unet
#python3 train_test_split.py --model mr_lkv --base-ch 32 --norm batch
python train_test_split.py --model replk




