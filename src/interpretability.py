"""
Interpretability utilities:
- Grad-CAM: lightweight implementation using pytorch-grad-cam (recommended)
- SHAP: outline to compute SHAP for metadata MLP (requires CPU sklearn wrapper)
"""

import os
from pathlib import Path
import numpy as np
import torch

# Grad-CAM: we prefer the pytorch-grad-cam package if available.
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False

from . import config


def make_gradcam(model, target_layer, images: torch.Tensor, preprocess_image_fn, output_dir: str):
    """
    Generate Grad-CAM overlays for a batch of images.
    - model: FusionModel; must expose image_encoder.features or compatible layer
    - target_layer: model.image_encoder.features.denseblock4 or similar
    - images: (B, C, H, W) tensor on CPU or device
    - preprocess_image_fn: function that converts tensor image to numpy RGB [0,1] for visualization
    """
    if not GRADCAM_AVAILABLE:
        raise RuntimeError("pytorch-grad-cam not available; install with `pip install grad-cam`")

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(config.DEVICE == "cuda"))

    os.makedirs(output_dir, exist_ok=True)
    images_np = preprocess_image_fn(images)  # user-supplied conversion to numpy

    grayscale_cam = cam(input_tensor=images, eigen_smooth=True)
    outs = []
    for i in range(len(images_np)):
        visualization = show_cam_on_image(images_np[i], grayscale_cam[i], use_rgb=True)
        out_path = os.path.join(output_dir, f"gradcam_{i:03d}.png")
        import imageio
        imageio.imwrite(out_path, visualization)
        outs.append(out_path)
    return outs


def compute_shap_for_metadata(model, background_data, X):
    """
    Compute SHAP values for metadata MLP.
    - model: should be a callable that accepts numpy array and returns probability
    - background_data: numpy array used as background (e.g., 100 samples)
    - X: numpy array to explain
    Note: this function requires shap package and may be slow on large X.
    """
    try:
        import shap
    except Exception:
        raise RuntimeError("Please install shap (`pip install shap`) to compute SHAP values.")

    explainer = shap.KernelExplainer(model, background_data)
    shap_values = explainer.shap_values(X, nsamples=100)
    return shap_values
