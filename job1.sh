#!/bin/bash -l
#SBATCH --job-name=test-cuda
#SBATCH --output=test-cuda.out
#SBATCH --error=test-cuda.err
#SBATCH --partition=rtx3080   # Use rtx3080 or whatever GPU partition is valid
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# Load modules (adjust to your HPC system)
module load python/3.12-conda
module load cuda/11.8

# Activate your GPU-enabled conda env
conda activate /home/woody/iwi5/iwi5293h/software/private/conda/envs/thesis-gpu

# Print info
which python
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
