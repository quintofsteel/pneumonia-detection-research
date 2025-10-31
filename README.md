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

The study evaluates a **DenseNet-121 + MLP fusion model** across three datasets — **NIH ChestX-ray14**, **CheXpert**, and **RSNA Pneumonia** — focusing on key metrics like accuracy, interpretability, fairness, and domain generalization.

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

## 🚀 Running the Experiments

All outputs (metrics, plots, logs) are automatically saved to `/experiments/results/`.

1.  **Train a Model**
    ```bash
    python src/train.py --config src/config.py
    ```

2.  **Evaluate Performance**
    ```bash
    python src/evaluate.py
    ```

3.  **Generate GradCAM & SHAP Explanations**
    ```bash
    python src/interpretability.py
    ```

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

## 📘 Citation

If you reference this work in your own research, please use the following BibTeX entry:

```bibtex
@thesis{chamisa2025pneumoniafusion,
  title={Development of a Machine Learning Model for Automatic Detection of Pneumonia from Chest X-Rays},
  author={Chamisa, Pride},
  school={University of Cape Town},
  year={2025}
}
