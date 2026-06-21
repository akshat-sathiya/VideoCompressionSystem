# Video Compression for Resource-Constrained Devices

Developed an efficient video compression system using an encoder-decoder architecture with ConvLSTM-based temporal prediction and bitmask block-skipping.

**Course Project** — EE604, Odd Semester 2025

**Team:**
- Akshat Sathiya Narayanan — 240087
- Arya Achal Mehta — 240202
- Pradyumn Vikram — 240759

The project report can be accessed in `/report.pdf`

---

## Overview

An end-to-end learned video compression pipeline built in PyTorch, designed to fit within an 8GB GPU budget.

### Architecture
- **Encoder-Decoder** with residual blocks and batch normalization
- **ConvLSTM** for temporal prediction across frames
- Frames converted to **YUV color space** and resized to 128×128

### Key Techniques
- **Bitmask Block-Skipping:** Computes per-block L2 residuals between consecutive frames and skips blocks below a threshold, reducing compute with only 0.04 dB PSNR loss
- **Parametric Gaussian Rate Model:** Models latent distributions for differentiable rate estimation, replacing discrete entropy computation

### Results
- **28.07 dB PSNR** (MSE: 0.0016)
- **-5.49 bpp** rate
- Trainable within **8GB GPU** memory

## Tech Stack
Python · PyTorch · OpenCV · NumPy · Matplotlib
