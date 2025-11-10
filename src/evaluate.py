"""
Evaluation utilities: metric computation and bootstrap confidence intervals.
"""

from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    average_precision_score,
)
from . import config


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = float("nan")
    metrics["ap"] = float(average_precision_score(y_true, y_prob))
    metrics["acc"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    metrics["confusion_matrix"] = None
    return metrics


def bootstrap_ci(metric_fn, y_true, y_prob, n_boot=1000, alpha=0.05):
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        stats.append(metric_fn(y_true[idx], y_prob[idx]))
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return lo, hi


def evaluate_model(model, dataloader, device="cpu"):
    """
    Runs model across dataloader and returns arrays of y_true and y_prob and aggregated metrics.
    """
    model.eval()
    ys_true = []
    ys_prob = []
    meta_rows = []
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            images, metas, labels, rows = batch
            images = images.to(device)
            metas = metas.to(device)
            labels = labels.to(device)
            probs, logits = model(images, metas)
            ys_true.append(labels.cpu().numpy())
            ys_prob.append(probs.cpu().numpy())
            meta_rows.extend(rows)

    if len(ys_true) == 0:
        raise RuntimeError("No valid samples in dataloader (all skipped or missing).")

    y_true = np.concatenate(ys_true)
    y_prob = np.concatenate(ys_prob)

    metrics = compute_metrics(y_true, y_prob)
    return {"y_true": y_true, "y_prob": y_prob, "metrics": metrics, "rows": meta_rows}
