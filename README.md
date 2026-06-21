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

An end-to-end learned video compression pipeline built in PyTorch, designed to fit within an 8GB GPU budget. The system encodes video frames into compact latent representations, exploits temporal redundancy across frames, and reconstructs output with minimal quality loss.

## Architecture

### Encoder
- Two-stage downsampling (128→64→32) using strided convolutions (5×5 and 3×3 kernels)
- Residual blocks with batch normalization after each downsampling stage
- Outputs a 24-channel latent representation at 32×32 spatial resolution

### Decoder
- Mirrors the encoder with transposed convolutions for upsampling (32→64→128)
- Residual blocks with batch normalization for stable reconstruction
- Final sigmoid activation to constrain output to [0, 1] range

### Temporal Modeling
- **ConvLSTM Cell** captures temporal dependencies across frames in latent space
- Maintains hidden and cell states across the sequence to predict future latent representations
- Refinement network (two 3×3 conv layers) post-processes ConvLSTM output for sharper predictions

## Key Techniques

### Bitmask Block-Skipping
- Divides each frame into 8×8 blocks and computes L2 norm of the residual between consecutive frames per block
- Blocks below a threshold (0.01) are zeroed out and skipped during encoding
- Generates a binary skip mask that propagates through the encoder, reusing the previous frame's latent for skipped regions
- Reduces compute with only 0.04 dB PSNR loss

### Parametric Gaussian Rate Model
- Models the distribution of quantized latent vectors as a learned Gaussian (per-channel mean and variance)
- Estimates bitrate differentiably using the negative log-likelihood under the Gaussian, avoiding discrete Shannon entropy
- Enables joint rate-distortion optimization via a tunable α parameter that balances reconstruction quality vs compression rate

### Preprocessing
- Frames converted from BGR to **YUV color space** to separate luminance from chrominance, improving compression efficiency
- Resized to 128×128 and normalized to [0, 1]

## Results
- **28.07 dB PSNR** (MSE: 0.0016)
- **-5.49 bpp** rate
- Trainable within **8GB GPU** memory

## Tech Stack
Python · PyTorch · OpenCV · NumPy · Matplotlib
