# AudioPG: Physics-Guided Audio Masked Autoencoders

Official implementation of the AudioPG system presented in the paper "Physics-Guided Audio Masked Autoencoders for Robust Sound Representation Learning".



## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Pre-training](#pre-training)
- [Fine-tuning](#fine-tuning)
- [Results](#results)
- [Citation](#citation)

## Overview

AudioPG introduces a novel physics-guided approach to audio representation learning using Masked Autoencoders (MAE). Our key contributions include:

### 🌍 World Model (Core Innovation)
- **Physics-guided synthesizer** that generates diverse audio samples using physical principles
- **Harmonic Additive Synthesis**: Implements formula (3) with k^(-γ) decay
- **Frequency Modulation (FM)**: Implements formula (4) with modulation index and carrier-modulator ratios
- **Broadband Pulses**: Sawtooth, square, triangle waveforms for rich timbre generation
- **Temporal Dynamics**: ADSR envelopes with randomized parameters and multi-event placement
- **Transient Injection**: Randomized short noise bursts for realistic transients
- **Post-processing Chain**: Spectral damping, background noise addition, and peak normalization

### 🏗️ AudioMAE Architecture
- Modified Vision Transformer with audio-specific patch embedding
- 75% masking ratio for effective self-supervised learning
- 16×16 patches of Log-Mel spectrograms (10.24s × 128 mel-bins)

## Architecture

```
Physics-Guided Synthesizer → Log-Mel Spectrograms → AudioMAE → Downstream Tasks
       ↓                          ↓                    ↓              ↓
Harmonic/FM/Pulse        10.24s × 128 mel-bins   ViT-Base    ESC-50/US8K/SCv2
+ Temporal Dynamics      16×16 patches           75% mask    FSD50K
+ Transients
+ Post-processing
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/AudioPG.git
cd AudioPG

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Pre-training with On-the-fly Generation

```bash
python pretrain.py
```

This trains the AudioPG model using the physics-guided synthesizer to generate audio samples on-the-fly.

### Fine-tuning on Downstream Tasks

```bash
# Fine-tune on ESC-50
python finetune.py --task esc50 --data_path /path/to/esc-50

# Fine-tune on UrbanSound8K
python finetune.py --task us8k --data_path /path/to/urbansound8k

# Fine-tune on Speech Commands V2
python finetune.py --task scv2 --data_path /path/to/speech_commands_v2
```

## Pre-training

Our pre-training methodology:

1. **On-the-fly Generation**: Audio samples are synthesized in real-time during training
2. **Physics-based Diversity**: Multiple synthesis methods ensure diverse audio content
3. **Temporal Complexity**: Multi-event placement and ADSR envelopes create realistic temporal structures
4. **Robust Representations**: Masked autoencoding learns robust audio features

### Key Parameters:
- **Duration**: 10.24 seconds per sample
- **Sample Rate**: 16 kHz
- **Mel-bins**: 128
- **Window**: 25ms (400 samples)
- **Hop**: 10ms (160 samples)
- **Masking Ratio**: 75%
- **Patch Size**: 16×16
- **Model Size**: ViT-Base (86M parameters)

## Fine-tuning

The pre-trained model can be fine-tuned for various downstream tasks:

### Supported Tasks:
- **ESC-50**: Environmental sound classification (50 classes)
- **UrbanSound8K**: Urban sound classification (10 classes) 
- **Speech Commands V2**: Wake word detection (35 classes)
- **FSD50K**: Large-scale sound event detection (mAP evaluation)

## Visualization & Analysis Tools

We provide tools to reproduce key analyses from the paper:

```bash
# Visualize learned filters
python visualize_filters.py

# Analyze disentanglement
python analyze_disentangle.py
```

## Citation

The citation will be updated upon acceptance.

## License

This project is licensed under the MIT License.

## Contact

For questions about the code or paper, please contact anonymous@university.edu.
