# 🎧 Multiclass Sound Classification using Deep Learning

This repository contains a mini-project on multiclass sound classification using a deep learning approach. The project focuses on urban sound scene recognition using audio preprocessing techniques, data augmentation strategies, and a modified ResNet18 architecture.

---

## Project Overview

- Task: Classify audio recordings into 10 urban environment categories (e.g., airport, metro, park).
- Techniques Used:
  - Mel Spectrogram Extraction
  - Data Augmentation: SpecAugment, Mixup
  - Modified ResNet18 Architecture
  - Label Smoothing and Gradient Clipping
- Tools: Python, PyTorch, Librosa, Streamlit

---

## Repository Structure

- `notebook.ipynb` – Main notebook with full pipeline and results
- `report.pdf` – Final project report
- `requirements.txt` – Required packages to run the project
- `app/` – Streamlit app to test predictions
- `results/` – Sample outputs, model weights, and spectrograms

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `notebook.ipynb` in Jupyter or Colab.

### 3. Launch Streamlit app

```bash
cd app
streamlit run app.py
```

## Dataset

The dataset used in this project is the **TUT Urban Acoustic Scenes 2018 Development Dataset**, sourced from Hugging Face. It contains `.wav` files categorized into 10 urban sound scenes, with each class stored in its own folder under `train/`.

**Dataset Source:** [TUT-urban-acoustic-scenes-2018-development-16bit](https://huggingface.co/datasets/wetdog/TUT-urban-acoustic-scenes-2018-development-16bit)

The dataset includes recordings from various urban environments such as:

- Airport
- Metro/Subway
- Park
- Shopping mall
- Street (pedestrian)
- Street (traffic)
- Tram
- Bus
- Public square
- Residential area

## Model Architecture

We use a modified ResNet18 adapted for single-channel spectrogram input. The model is trained with techniques like SpecAugment, mixup, and label smoothing for better generalization.

## Collaborators

- Senhadji M Said
- Ali Abbou Oussama
- DJEZIRI Oussama
- Baidar Samir
