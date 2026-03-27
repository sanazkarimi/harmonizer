# Harmonized Feature Conditioning and Frequency-Prompt Personalization for Multi-Rater Medical Segmentation

Official PyTorch implementation of the CVPR paper:

**Harmonized Feature Conditioning and Frequency-Prompt Personalization for Multi-Rater Medical Segmentation**

---

## 🧠 Introduction

Medical image segmentation often involves multiple expert annotations, reflecting both intrinsic clinical ambiguity and subjective differences among annotators. Traditional approaches typically collapse these annotations into a single consensus mask, losing valuable uncertainty and personalization information.

In this work, we propose a **probabilistic multi-rater segmentation framework** that explicitly models:

- **Data-level variability** (scanner noise, acquisition artifacts, domain shifts)
- **Rater-level variability** (annotator-specific boundary and texture preferences)

Our framework introduces:

- **Noise Harmonizer** → reduces scanner-induced variability in latent space  
- **Frequency-based Personalization Module** → models annotator-specific styles using high-frequency prompts  
- **GED-based objective** → aligns predicted distributions with true annotation diversity  

---

## 🚀 Highlights

- Probabilistic modeling of multi-rater segmentation
- Explicit disentanglement of data vs. rater uncertainty
- Frequency-domain personalization via wavelet-based prompts
- Improved distribution learning using **Generalized Energy Distance (GED)**
- Strong performance on **LIDC-IDRI** and **NPC-170**

---

## 🏗️ Method Overview

The framework follows a **two-stage training paradigm**:

### 🔹 Stage I: Harmonized Probabilistic Learning
- Learn shared latent representation
- Apply **Noise Harmonizer** to suppress artifacts
- Train backbone (Encoder + Decoder + Harmonizer)

### 🔹 Stage II: Rater Personalization
- Freeze backbone
- Train lightweight **Personalization Module**
- Use frequency decomposition to adapt predictions per annotator

This design ensures:
- clean, artifact-robust features  
- efficient and interpretable rater-specific adaptation  

---

## 📁 Repository Structure

```bash
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
