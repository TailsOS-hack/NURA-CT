<div align="center">

# 🧠 NURA-CT

**Neural Understanding & Radiological Analysis for CT**

An ML-powered toolkit for detecting strokes and classifying brain tumors from medical imaging — with a fully interactive Streamlit GUI.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-GUI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22d3ee.svg)](LICENSE)

<br>

<img src="https://github.com/user-attachments/assets/placeholder-screenshot" alt="NURA-CT GUI" width="90%">

*Upload a brain scan → choose your analysis mode → get instant results*

</div>

---

## ✨ Features

| Mode | Description | Input | Model |
|------|-------------|-------|-------|
| 🔍 **Bright-Region Detection** | Fast threshold-based anomaly spotter (mean + σ) | Any 2D brain image (JPG/PNG) | None — works instantly |
| 🧬 **Tumor Classification** | 4-layer CNN classifies Glioma, Meningioma, or Pituitary tumors | MRI image | `brain_tumor_model.h5` |
| 🩻 **Stroke Segmentation** | 3D U-Net segments stroke regions in volumetric CT scans | NIfTI `.nii` file | `stroke_segmentation_model_3d.h5` |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/TailsOS-hack/NURA-CT.git
cd NURA-CT
pip install -r requirements.txt
```

### 2. Launch the GUI

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**. Use the sidebar to switch between the three analysis modes.

> **Bright-Region Detection** works immediately — no model training needed.  
> For the other two modes, train the models first (see below).

---

## 🏋️ Training the Models

### Brain Tumor CNN

```bash
python "Tumor Train.py"
```

Trains a 4-layer convolutional neural network on the MRI dataset in `brain-tumor-data-mri/` for 30 epochs. Saves the best model checkpoint as `brain_tumor_model.h5`.

**Architecture:** Conv2D(32) → Conv2D(64) → Conv2D(128) → Conv2D(128) → Dense(512) → Softmax(3)

### Stroke 3D U-Net

```bash
cd "Stroke Code and Data"
python "Main Code.py"
```

Trains a 3D U-Net encoder–decoder on NIfTI CT volumes with skip connections. Saves as `stroke_segmentation_model_3d.h5`.

**Architecture:** Encoder (64 → 128 → 256) → Decoder with skip connections → Sigmoid output

---

## 📂 Project Structure

```
NURA-CT/
├── app.py                          # 🖥️  Streamlit GUI (start here!)
├── requirements.txt                # Python dependencies
├── check_specs.py                  # GPU / system info checker
├── LLM.py                         # Bright-region detector (standalone)
├── LLM Tumor.py                   # (placeholder)
├── Tumor Train.py                  # Tumor CNN training script
├── tumor_ex.jpeg                   # Example brain MRI image
├── brain-tumor-data-mri/           # MRI training dataset
│   └── New folder (6)/
│       ├── train/                  # Training images (3 classes)
│       └── val/                    # Validation images
└── Stroke Code and Data/
    ├── Main Code.py                # 3D U-Net training script
    ├── Main Stroke Segmentation.py # Stroke inference script
    ├── stroke_segmentation_model_3d.h5
    ├── Segmented Data/             # NIfTI volumes & masks
    │   ├── ct_scans/
    │   └── masks/
    ├── Normal/                     # Normal CT samples
    └── Stroke/                     # Stroke CT samples
```

---

## 🔧 System Requirements

Run `python check_specs.py` to verify your setup.

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| GPU | — | CUDA-compatible (6 GB+ VRAM) |
| RAM | 8 GB | 16 GB+ |
| Disk | 10 GB (with datasets) | 15 GB |

---

## 📖 How It Works

### Bright-Region Detection

1. Convert image to grayscale
2. Compute adaptive threshold: **T = μ + σ**
3. Extract bounding box of all pixels above T
4. Draw detection overlay + generate report

### Tumor Classification (CNN)

1. Resize MRI to 150×150, normalize to [0, 1]
2. Data augmentation: rotation, shift, shear, zoom, flip
3. Forward pass through 4 conv blocks + dense layers
4. Softmax output → Glioma / Meningioma / Pituitary

### Stroke Segmentation (3D U-Net)

1. Load NIfTI volume, resize to 128×128×64
2. Normalize per-volume to [0, 1]
3. Encoder–decoder with skip connections produces voxel-wise probability map
4. Threshold at 0.5 → binary mask overlay on CT slice

---

## ⚠️ Disclaimer

> **This software is for educational and research purposes only.**  
> NURA-CT is not a certified medical device and must not be used for clinical diagnosis, treatment planning, or any medical decision-making. Always consult a qualified radiologist or medical professional.

---

## 🤝 Contributors

<a href="https://github.com/TailsOS-hack"><img src="https://github.com/TailsOS-hack.png" width="60" style="border-radius:50%"></a>
<a href="https://github.com/Anorakbyte1"><img src="https://github.com/Anorakbyte1.png" width="60" style="border-radius:50%"></a>

---

<div align="center">
<sub>Built with ❤️ using TensorFlow, Streamlit & Python</sub>
</div>
