"""Metrics, threshold selection, sample weighting, and warning levels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.utils.class_weight import compute_sample_weight


def select_f2_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweep thresholds and pick the one that maximises F2 (recall-weighted)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [
        fbeta_score(y_true, y_prob >= t, beta=2, zero_division=0)
        for t in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def select_recall_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, *, min_recall: float = 0.80
) -> float:
    """Pick the highest threshold whose validation recall reaches min_recall."""
    thresholds = np.linspace(0.01, 0.99, 99)
    valid = []
    for threshold in thresholds:
        y_pred = y_prob >= threshold
        recall = recall_score(y_true, y_pred, zero_division=0)
        if recall >= min_recall:
            valid.append(float(threshold))
    return max(valid) if valid else 0.01


def metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float
) -> dict[str, float | int]:
    """Calculate scalar metrics at a fixed probability threshold."""
    y_true = y_true.astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true)) == 2

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    n = tp + fp + fn + tn

    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0

    # Heidke skill score
    expected = (((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n) if n else np.nan)
    observed = (tp + tn) / n if n else np.nan
    hss = (observed - expected) / (1 - expected) if n and expected < 1 else np.nan

    mcc_den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0

    return {
        "threshold": float(threshold),
        "AUROC": float(roc_auc_score(y_true, y_prob)) if has_both_classes else np.nan,
        "AUPRC": float(average_precision_score(y_true, y_prob)) if has_both_classes else np.nan,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)),
        "F2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "CSI": float(csi),
        "FAR": float(far),
        "HSS": float(hss),
        "MCC": float(mcc),
        "prevalence": float(np.mean(y_true)) if len(y_true) else np.nan,
        "alarm_rows": tp + fp,
        "alarm_rate": float((tp + fp) / n) if n else np.nan,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def evaluate_performance(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    thresholds_train: tuple[np.ndarray, np.ndarray] | None = None
) -> dict:
    """Consolidated performance evaluation. Returns curves and scalar metrics.
    
    If thresholds_train is provided (y_train, y_prob_train), the threshold is
    selected on training data to avoid leakage.
    """
    if thresholds_train:
        thr = select_f2_threshold(*thresholds_train)
    else:
        thr = select_f2_threshold(y_true, y_prob)

    # Scalar metrics
    metrics = metrics_at_threshold(y_true, y_prob, threshold=thr)

    if len(np.unique(y_true.astype(int))) == 2:
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        # PR curve
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    else:
        fpr = tpr = prec_arr = rec_arr = np.array([np.nan])
    
    return {
        **metrics,
        "fpr": fpr,
        "tpr": tpr,
        "pr_prec": prec_arr,
        "pr_rec": rec_arr,
        "y_prob": y_prob,
    }


def compute_warning_thresholds(prevalence: float) -> tuple[float, float, float]:
    """Province of Bolzano Allegato D 4-level thresholds anchored on prevalence.
    
    Verde/Giallo = pi
    Giallo/Arancione = 2*pi
    Arancione/Rosso = 5*pi (capped at 0.5)
    """
    t1 = min(prevalence, 0.15)
    t2 = min(2 * prevalence, 0.35)
    t3 = min(5 * prevalence, 0.60)
    t2 = max(t2, t1 + 0.01)
    t3 = max(t3, t2 + 0.01)
    return t1, t2, t3


def assign_warning_level(p: float, t1: float, t2: float, t3: float) -> str:
    """Map a probability to a warning level string."""
    if p < t1: return "Verde"
    if p < t2: return "Giallo"
    if p < t3: return "Arancione"
    return "Rosso"


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse class frequency weights."""
    return compute_sample_weight("balanced", y.astype(int))


