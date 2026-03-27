# Harmonized Feature Conditioning and Frequency-Prompt Personalization for Multi-Rater Medical Segmentation

Official PyTorch implementation of the CVPR 2026 paper:

> **Harmonized Feature Conditioning and Frequency-Prompt Personalization for Multi-Rater Medical Segmentation**
> Sanaz Karimijafarbigloo, Armin Khosravi, Alireza Kheyrkhah, Reza Azad, Mauricio Reyes, Dorit Merhof
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026*

---

## Introduction

Medical image segmentation often involves multiple expert annotations, reflecting both intrinsic clinical ambiguity and subjective differences among annotators. Traditional approaches collapse these annotations into a single consensus mask, discarding valuable uncertainty and personalization information.

We propose a **probabilistic multi-rater segmentation framework** that explicitly models two complementary sources of variability:

- **Data-level variability** — scanner noise, acquisition artifacts, and domain shifts
- **Rater-level variability** — annotator-specific boundary and texture preferences

The framework introduces three core components:

| Component | Role |
|---|---|
| **Noise Harmonizer** | Reduces scanner-induced variability in the latent space |
| **Frequency-based Personalization Module** | Models annotator-specific styles via high-frequency prompts |
| **GED-based Objective** | Aligns predicted distributions with true annotation diversity |

---

## Highlights

- Probabilistic modeling of multi-rater segmentation uncertainty
- Explicit disentanglement of data-level vs. rater-level variability
- Frequency-domain personalization via wavelet-based prompts
- Improved distribution learning using **Generalized Energy Distance (GED)**
- State-of-the-art performance on **LIDC-IDRI** and **NPC-170** benchmarks

---

## Method Overview

The framework follows a **two-stage training paradigm**:

### Stage I — Harmonized Probabilistic Learning

- Learn a shared latent representation across raters
- Apply the **Noise Harmonizer** to suppress acquisition artifacts
- Train the full backbone (Encoder + Decoder + Harmonizer)

### Stage II — Rater Personalization

- Freeze the pre-trained backbone
- Train a lightweight **Personalization Module**
- Apply frequency decomposition to adapt predictions per annotator

This design yields clean, artifact-robust features in Stage I, and efficient, interpretable rater-specific adaptation in Stage II.

---

## Repository Structure

```
.
├── code/
│   ├── train.py
│   ├── evaluate.py
│   ├── networks/
│   ├── utils/
│   └── ...
├── dataset/
│   ├── LIDC/
│   └── NPC/
├── models/
├── assets/
└── README.md
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

**2. Create a conda environment**

```bash
conda create -n multirater python=3.10 -y
conda activate multirater
```

**3. Install PyTorch**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

---

## Training

### Stage I — Backbone and Noise Harmonizer

```bash
cd code
python train.py \
    --stage 1 \
    --dataset lidc \
    --gpu 0
```

### Stage II — Rater Personalization

Copy the Stage I checkpoint, then launch Stage II training:

```bash
cp ../models/stage1_model.pth ./code/

python train.py \
    --stage 2 \
    --dataset lidc \
    --gpu 0 \
    --pretrained stage1_model.pth
```

---

## Evaluation

### Stage I — Distribution Quality

```bash
python evaluate.py \
    --stage 1 \
    --dataset lidc \
    --save_path ../models/model.pth \
    --test_num 50
```

### Stage II — Personalized Predictions

```bash
python evaluate.py \
    --stage 2 \
    --dataset lidc \
    --save_path ../models/model.pth \
    --test_num 500
```

---

## Acknowledgements

This codebase builds upon the following open-source project:

- [D-Persona](https://github.com/ycwu1997/D-Persona) — Dong et al., CVPR 2023

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@InProceedings{Karimijafarbigloo2026CVPR,
  author    = {Karimijafarbigloo, Sanaz and Khosravi, Armin and Kheyrkhah, Alireza and Azad, Reza and Reyes, Mauricio and Merhof, Dorit},
  title     = {Harmonized Feature Conditioning and Frequency-Prompt Personalization for Multi-Rater Medical Segmentation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
