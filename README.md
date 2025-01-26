# Speech Command Recognition

This repository contains the code and resources for the **Speech Command Recognition** project, developed by Roger Baiges Trilla and Pau Hidalgo Pujol at TVD - GIA @ UPC. The work focuses on designing, implementing, and evaluating deep learning models for classifying audio recordings into predefined command categories with high accuracy.

---

## Overview

This project explores advanced techniques in speech command recognition using:

- **Audio Preprocessing**: Spectrograms, MEL-Spectrograms, MFCCs
- **Model Architectures**: CNNs, RNNs, Attention mechanisms, Transformers
- **Optimization**: Data augmentation (SpecAugment), normalization, and regularization
- **Transfer Learning**: Fine-tuning pre-trained models like DistilHuBERT and HuBERT

The final model achieves a test accuracy of **98.6%**, outperforming many state-of-the-art methods.

---

## Performance Metrics

| Model                | Validation Accuracy | Test Accuracy |
|----------------------|---------------------|---------------|
| Our ResNet (1.6M)      | 97.8%              | 98.0%         |
| HuBERT-Large (fine-tune) | 98.5%          | 98.6%         |
| AttMH-RNN + SpecAug           | 96.36%              |  - |
| AttRNN           | 95.77%              |  - |
| ConvLSTM2            | 95.4%              |  - |
---

For more results, look at the report.

## Key Features

- **Audio Preprocessing**
  - Linear Spectrogram, MEL-Spectrogram (32, 64, 128 bins)
  - MFCC (13, 40, 64 features)
  - SpecAugment for data augmentation

- **Model Architectures**
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks (RNNs, GRUs, LSTMs, and bidirectional variants)
  - Transformers with CNN integration
  - Residual and Attention-based networks

- **Training Enhancements**
  - Normalization (LayerNorm, BatchNorm)
  - Regularization (Dropout, L1/L2 penalties)
  - Learning rate schedules and early stopping

- **Transfer Learning**
  - Pre-trained models (DistilHuBERT, HuBERT-Large, HuBERT-XLarge)
  - Fine-tuning for domain-specific tasks

---

## Repository Structure

```
├── Convolutionals.ipynb    # Experiments with convolutional models
├── Normalization.ipynb     # Impact of normalization on models
├── notebook.ipynb          # Main analysis notebook
├── plotting.ipynb          # Plotting and visualization scripts
├── README.md               # Project documentation (this file)
├── Recurrents.ipynb        # Experiments with recurrent models
├── Regularization.ipynb    # Regularization techniques and experiments
├── ResNet9800.ipynb        # Best-performing ResNet model
├── SpecAugment.py          # Implementation of SpecAugment
├── SpectrogramsBig.ipynb   # Spectrogram preprocessing (big model)
├── SpectrogramsSmall.ipynb # Spectrogram preprocessing (small model)
├── training_plots.py       # Scripts for training visualization
```

---

## References

This project used the **TensorFlow Speech Commands Dataset** as the primary dataset for training and evaluation. The models were trained using Kaggle GPUs and achieved **first place** in the [TVD-2024 Kaggle competition](https://www.kaggle.com/competitions/tvd-2024-reconocimiento-de-comandos-de-voz/leaderboard). For further technical details, refer to the project report included in [this repository](https://github.com/pauhidalgoo/speech-recognition/blob/main/Report_TVD_SpeechRecognition.pdf).

### Related Works

1. **SpecAugment**: Park, D. S., et al. *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*.
2. **HuBERT**: Hsu, W. N., et al. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*.
3. **Temporal Convolutional Networks**: Lea, C., et al. *Temporal Convolutional Networks for Action Segmentation and Detection*.
4. **Rare Sound Events Detection**: Mesaros, A., et al. *An Attention-Based Neural Network for Detecting Rare Sound Events*.
5. **Streaming Architectures for Keyword Spotting**: Lugosch, L., et al. *Streaming Keyword Spotting on Mobile Devices*.
6. **Google Speech Commands Dataset Benchmarks**: Papers with Code (2024).
7. **Kaggle Competition Leaderboard**: Kaggle, *TVD 2024 - Reconocimiento de Comandos de Voz*.


---
Feel free to contribute or raise issues for discussion :)
