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

# Takeaway from the Observations and Comparison:
# 1. Parameter‐efficiency

    MR_LKV: 3.65 M params
    U-Net: 31.04 M params
    RepLKNet: 21.79 M
    SwinIR: 0.73 M
    Restormer: 12.16 M
    Swin2SR: 1.06 M


# 2. Inference time

    MR_LKV: 6.17 ms
    U-Net: 4.84 ms
    RepLKNet: 20.09 ms
    SwinIR: 47.43 ms
    Restormer: 22.61 ms
    Swin2SR: 25.66 ms

# MR_LKV is significantly more parameter‐efficient than most baselines and offers one of the fastest inference times (only U-Net is slightly faster), all while maintaining comparable reconstruction quality
