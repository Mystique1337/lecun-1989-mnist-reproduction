"""Small statistical helpers for comparing classifiers across seeds."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and a t-based confidence interval across independent runs."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(sps.t.ppf(0.5 + confidence / 2, len(values) - 1) * sem)
    return mean, mean - half, mean + half


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact (binomial) McNemar test on paired predictions.

    Only the discordant test images carry information about which model is
    better: those A gets right and B gets wrong (``a_only``) and the reverse
    (``b_only``). Under the null hypothesis both are equally likely.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    a_only = int(np.sum(a_right & ~b_right))
    b_only = int(np.sum(~a_right & b_right))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(sps.binomtest(a_only, n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "p_value": p}


def confusion_matrix(y_true, y_pred, n_classes: int = 10) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm
