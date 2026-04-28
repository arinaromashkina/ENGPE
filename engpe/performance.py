"""
Performance estimation from FDR q-values.

Given q-values at every decision threshold, ENGPE estimates TP, FP, TN, FN
and derives standard metrics (accuracy, precision, recall, F1) without
using true sample labels.

Notation (from paper, Section 3):
  ACC_ST : standard accuracy  = ACC(threshold = -∞), accepts all samples
  ACC_TA : threshold-adapted  = max_t ACC(t), accepts only trusted predictions
  π₀     : proportion of misclassified (false discovery) samples
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core: confusion matrix estimation
# ---------------------------------------------------------------------------

def estimate_confusion_matrix(pred_scores: np.ndarray,
                               q_values: np.ndarray,
                               pi0: float) -> dict:
    """
    Estimate TP, FP, TN, FN at every decision threshold.

    The samples must be in ascending order of pred_scores (the output of
    fdr.calculate_mixmax_qvalues / fdr.calculate_tdc_qvalues).

    Parameters
    ----------
    pred_scores : (N,) sorted ascending
    q_values    : (N,) monotone q-values aligned with pred_scores
    pi0         : estimated null proportion (fraction of misclassified samples)

    Returns
    -------
    dict with keys 'TP', 'FP', 'TN', 'FN', 'ACC', each (N,) array.
    Index i corresponds to accepting the top (N-i) highest-scored samples.
    """
    n   = len(pred_scores)
    idx = np.arange(n)

    accepted = n - idx
    FP  = accepted * q_values
    TP  = accepted * (1 - q_values)
    TN  = n * pi0  - FP
    FN  = n * (1 - pi0) - TP

    FP  = np.clip(FP,  0, None)
    TP  = np.clip(TP,  0, None)
    TN  = np.clip(TN,  0, None)
    FN  = np.clip(FN,  0, None)
    ACC = np.clip((TP + TN) / n, 0.0, 1.0)

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'ACC': ACC}


# ---------------------------------------------------------------------------
# Accuracy estimation
# ---------------------------------------------------------------------------

def estimate_accuracy(pred_scores: np.ndarray,
                       q_values: np.ndarray,
                       pi0: float,
                       return_curves: bool = False):
    """
    Estimate ACC_ST (standard) and ACC_TA (threshold-adapted) accuracy.

    Parameters
    ----------
    pred_scores   : (N,) max model scores, sorted ascending
    q_values      : (N,) q-values from fdr.compute_qvalues()
    pi0           : null proportion estimated as q_values[0]
    return_curves : if True, also return the full ACC curve

    Returns
    -------
    acc_st : float — estimated accuracy accepting all samples
    acc_ta : float — estimated best accuracy with threshold adaptation
    opt_threshold : float — decision threshold that achieves acc_ta
    curves : dict (only if return_curves=True)
    """
    cm  = estimate_confusion_matrix(pred_scores, q_values, pi0)
    acc = cm['ACC']

    acc_st = float(acc[0])
    ta_idx = int(np.argmax(acc))
    acc_ta = float(acc[ta_idx])
    opt_threshold = float(pred_scores[ta_idx]) if len(pred_scores) > 0 else 0.0

    if return_curves:
        return acc_st, acc_ta, opt_threshold, cm
    return acc_st, acc_ta, opt_threshold


# ---------------------------------------------------------------------------
# Precision / Recall
# ---------------------------------------------------------------------------

def estimate_precision_recall(pred_scores: np.ndarray,
                               q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate precision and recall at every threshold.

    precision(t) = 1 - FDR(t)
    recall(t)    = (1 - FDR(t)) * |accepted(t)| / total_positives_est

    where total_positives_est = |{f(t) ≥ -∞}| * (1 - π₀).
    """
    n        = len(pred_scores)
    pi0      = float(np.clip(q_values[0], 0, 1))
    accepted = n - np.arange(n)

    precision = np.clip(1 - q_values, 0, 1)
    n_pos_est = max(n * (1 - pi0), 1)
    recall    = np.clip(precision * accepted / n_pos_est, 0, 1)
    return precision, recall


# ---------------------------------------------------------------------------
# True accuracy (with labels, for evaluation)
# ---------------------------------------------------------------------------

def compute_true_accuracy_curve(pred_scores: np.ndarray,
                                 true_labels: np.ndarray,
                                 pred_labels: np.ndarray) -> np.ndarray:
    """
    Oracle accuracy curve computed with true labels (for benchmarking only).

    Sort samples by pred_score ascending; for threshold at index i:
      - accepted  = samples i..n-1 (highest scores)
      - rejected  = samples 0..i-1 (lowest scores)
      ACC(i) = (correct_accepted + incorrect_rejected) / n
    """
    n          = len(pred_scores)
    sort_idx   = np.argsort(pred_scores)
    correct_s  = (true_labels[sort_idx] == pred_labels[sort_idx]).astype(int)

    TP_from_i  = np.cumsum(correct_s[::-1])[::-1]
    TN_to_i    = np.cumsum((1 - correct_s))
    TN_to_i    = np.concatenate([[0], TN_to_i[:-1]])

    acc = np.clip((TP_from_i + TN_to_i) / n, 0, 1)
    return acc


def compute_true_fdr_curve(pred_scores: np.ndarray,
                            true_labels: np.ndarray,
                            pred_labels: np.ndarray) -> np.ndarray:
    """
    Oracle FDR / q-value curve computed with true labels (for benchmarking).
    """
    n         = len(pred_scores)
    sort_idx  = np.argsort(pred_scores)
    incorrect = (true_labels[sort_idx] != pred_labels[sort_idx]).astype(int)

    FD_CF  = np.cumsum(incorrect[::-1])[::-1]
    D_CF   = np.arange(n, 0, -1)
    fdr    = np.clip(FD_CF / D_CF, 0, 1)
    q_true = np.clip(np.minimum.accumulate(fdr), 0, 1)
    return q_true
