# Large-Kernel CNN for Motion Artifact Reduction

This repository contains the code and results for a Master's thesis on motion artifact reduction in CT imaging using super large-kernel convolutional neural networks.

## Overview

The pipeline includes:

1. Motion artifact simulation in sinogram space using ConeProjection.
2. Training models for artifact removal from 2D sinograms.
3. CT reconstruction of artifact-cleaned sinograms.
4. Quantitative comparison (PSNR, SSIM, Params, FLOPs, Inference time).
5. Qualitative comparison on real CT data.

## Models

- Proposed MR-LKV (nobel model)
- U-Net baseline
- RepLKNet baseline
- SwinIR baseline
- Restormer baseline

## Results

Quantitative results are provided under `results/quantitative/`.
Qualitative visual comparisons are under `results/qualitative/`.

## Reproducing Results

### Requirements

```bash
pip install -r requirements.txt
