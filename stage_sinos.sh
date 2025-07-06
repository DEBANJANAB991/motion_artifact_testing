#!/bin/bash
#SBATCH --job-name=stage_sinos
#SBATCH --output=stage_sinos.%j.out
#SBATCH --error=stage_sinos.%j.err
#SBATCH --time=24:00:00

#SBATCH --partition=rtx3080   # Use rtx3080 or whatever GPU partition is valid
#SBATCH --gres=gpu:1
set -euxo pipefail

mkdir -p /scratch/iwi5/iwi5293h/sinograms
rsync -avh --progress \
  /home/vault/iwi5/iwi5293h/CT-Motion-Artifact-Reduction_bkp/data/sinograms/ \
  /scratch/iwi5/iwi5293h/sinograms/   
