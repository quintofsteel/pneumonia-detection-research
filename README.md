 # Pneumonia Detection using Multimodal Deep Learning

## 🏛️ Project Details

* **University:** University of South Africa — Honours Research Project
* **Author:** **Pride Chamisa**
* **Year:** 2025

---

## 📖 Overview

This repository contains the implementation and analysis for the research project:

**“Development of a Machine Learning Model for Automatic Detection of Life-Threatening Diseases”**
(Focus: Multimodal pneumonia detection using chest X-rays and structured clinical metadata)

The study investigates whether multimodal deep learning, combining chest X-ray images and structured clinical metadata, can improve pneumonia detection performance and generalizability by evaluating a **DenseNet-121 + MLP fusion model** across heterogeneous datasets (**NIH ChestX-ray14**, **CheXpert**, and **RSNA Pneumonia** ).

The project rigorously evaluates model accuracy, interpretability, generalizability, and fairness to support ethical and safe AI deployment in low-resource clinical settings, particularly within South Africa’s healthcare system.

### 🧠 Research Objectives

* To investigate whether fusing image and metadata improves pneumonia detection performance.
* To assess model generalizability under domain shift.
* To ensure interpretability and fairness in medical AI decision-making.

---

## 🧩 Repository Structure

| Folder | Description |
| :--- | :--- |
| `data/` | Raw and processed datasets |
| `src/` | Core code modules (data, model, training, evaluation) |
| `experiments/` | Logs, results, GradCAM and SHAP outputs |
| `notebooks/` | Jupyter notebooks for analysis and visual results |
| `thesis/` | Final figures, tables, and written report chapters |


---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/pridechamisa/pneumonia-detection-research.git](https://github.com/pridechamisa/pneumonia-detection-research.git)
    cd pneumonia-detection-thesis
    ```
2.  **Install base dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install PyTorch with GPU support (Optional):**
    If using an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch version (e.g., for CUDA 11.8):
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

---

## 🚀 Reproduction aligned to the notebook

### 1) Setup environment (matching notebook versions)
```bash
pip install -r requirements.txt
# optional: install torch/torchvision per your CUDA
```

### 2) Prepare RSNA data (DICOM → PNG, metadata, CSV)
This mirrors the notebook preprocessing (age/sex/view_position mappings, image-level labels):
```bash
python scripts/prepare_rsna.py \
  --rsna_dir "/kaggle/input/rsna-pneumonia-detection-challenge" \
  --out_dir "chapter4_outputs/rsna_prepared" \
  --limit 0
```
Outputs:
- `chapter4_outputs/rsna_prepared/rsna_metadata.csv`
- `chapter4_outputs/rsna_prepared/rsna_classification.csv`
- `chapter4_outputs/rsna_prepared/rsna_train.csv|rsna_val.csv|rsna_test.csv`
- `chapter4_outputs/rsna_prepared/rsna_png_images/`

If your data lives elsewhere, override via env vars or flags. You can also set:
`DATA_ROOT`, `RSNA_PATH`, and `OUTPUT_ROOT` environment variables.

All outputs (metrics, plots, logs) are automatically saved to `/experiments/results/`.

1.  **Train a Model** (using the exported CSVs)
    ```bash
    python -m src.train \
      --train_csv chapter4_outputs/rsna_prepared/rsna_train.csv \
      --val_csv   chapter4_outputs/rsna_prepared/rsna_val.csv \
      --test_csv  chapter4_outputs/rsna_prepared/rsna_test.csv \
      --epochs 10 --batch_size 32 --lr 1e-4
    ```

2.  **Evaluate Performance**
    ```bash
    python src/evaluate.py
    ```

3.  **Generate GradCAM & SHAP Explanations**
    ```bash
    python src/interpretability.py
    ```

Notes:
- Paths in `src/config.py` default to notebook-style (`/kaggle/input/...`, `chapter4_outputs/`). Override with env vars if needed.
- The notebook file can be placed under `notebooks/` for reference; the scripts above reproduce its pipeline end-to-end.

---

## 📊 Results Summary

| Model | AUC | Sensitivity | Specificity | F1 | Brier |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Image-only (DenseNet121) | 0.296 | 1.000 | 0.000 | 0.374 | 0.378 |
| Metadata-only (MLP) | 0.729 | 0.000 | 1.000 | 0.000 | 0.245 |
| **Fusion (Final)** | 0.422 | 1.000 | 0.000 | 0.374 | 0.269 |

> GradCAM and SHAP analyses highlight interpretability and reveal the **fragility of multimodal fusion** in heterogeneous data settings.

### 📈 Key Findings

* Fusion models **did not outperform** unimodal baselines under domain shift.
* Metadata-only networks exhibited better **calibration and robustness**.
* Model fairness varied by sex and view position, emphasizing **ethical deployment concerns**.

---

### 🩺 Practical Implications

* AI must be locally validated before clinical use.
* Metadata-driven triage is a safer starting point for LMIC deployments.
* Fairness auditing and calibration are mandatory to prevent diagnostic bias.

---

### 🔒 Ethical Notice

* All datasets used are publicly available and de-identified.
* This repository does not include patient data from South African clinical sources.
* All model outputs must be interpreted with clinician oversight and are intended for research purposes only.

---
<!--
## 📘 Citation

If you reference this work in your own research, please use the following BibTeX entry:

```bibtex
@thesis{chamisa2025pneumoniafusion,
  title={Development of a Machine Learning Model for Automatic Detection of Pneumonia from Chest X-Rays},
  author={Chamisa, Pride},
  school={University of South Africa},
  year={2025}
}
-->

