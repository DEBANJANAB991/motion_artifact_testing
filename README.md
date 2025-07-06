# Motion Artifact Reduction in CT with Super-Large-Kernel CNNs

This repository contains PyTorch code for training, evaluating, and visualizing a novel super-large-kernel convolutional neural network (CNN) to mitigate rigid-motion artifacts in fan-beam CT sinograms. It implements a U-shaped backbone with depthwise 31×31→41×41→51×51 convolutions, channel/spatial attention, and optional FFT processing, and reports both pixel-wise (MSE) and perceptual (PSNR, SSIM) metrics.


# Prerequisites

- Python 3.8+  
- PyTorch 1.12+ with CUDA support  
- NumPy, Matplotlib  
- A CUDA-enabled GPU for training  (Optional, but recommended)

# Repository structure:
All the codes are present inside the scripts folder:
-> dicom_to_sinogram.py (To convert from volume domain to projection domain).
-> add_motion_artifacts.py (Adding cubic spline artifacts).
-> model_mr_lkv.py (model for training).
-> train_test_split.py (For training script).
-> evaluate.py (For testing data and computing metrics).
-> config.py



