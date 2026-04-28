"""
FDR control procedures used by ENGPE.

Three methods are provided:

BH (binary)
    Benjamini-Hochberg with Storey's π₀ estimator applied to empirical
    p-values computed against the CNF-generated null distribution.

Mix-Max (multi-class, default)
    From Keich et al. (2015). Uses the mixture-of-maxima statistic to
    estimate FDR at every decision threshold without requiring valid
    p-values. With π₀ = 0 (no truly null samples), the FDP formula
    simplifies to a ratio of empirical CDFs of target vs. null scores.

TDC / Knock-off (multi-class, alternative)
    Target-Decoy Competition. A sample's prediction is treated as a
    knock-off loss if the null score exceeds the model score.
    Simpler but lower statistical power than Mix-Max.

All functions operate on 1-D max scores: f(t) = max_k F_K(t)[k].
"""

from __future__ import annotations

import numpy as np
from bisect import bisect


# ---------------------------------------------------------------------------
# Empirical p-values and π₀ estimation
# ---------------------------------------------------------------------------

def empirical_pvalue(score: float, null_scores: np.ndarray) -> float:
    """Compute one empirical p-value against a null score pool."""
    n = len(null_scores)
    sorted_null = np.sort(null_scores)
    return (n - bisect(sorted_null, score)) / n


def empirical_pvalues(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    """Vectorised empirical p-values (Eq. 2 in paper)."""
    n = len(null_scores)
    sorted_null = np.sort(null_scores)
    p = np.zeros(len(scores))
    for i, s in enumerate(scores):
        p[i] = (1 + (n - bisect(sorted_null, s))) / (n + 1)
    return p


def estimate_pi0_storey(p_values: np.ndarray,
                         lambdas: np.ndarray | None = None) -> float:
    """
    Storey's estimator for the null proportion π₀ (label-free).

    π̂₀ = mean over λ of  #{p > λ} / (n * (1 - λ))
    clamped to [0, 1].
    """
    if lambdas is None:
        lambdas = np.arange(0.05, 0.95, 0.05)
    estimates = [np.mean(p_values > lam) / (1 - lam) for lam in lambdas]
    return float(np.clip(np.mean(estimates), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Benjamini-Hochberg (binary classification, K=1)
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: np.ndarray, pi0: float = 1.0) -> np.ndarray:
    """
    BH procedure with π₀ adjustment.

    Returns q-values (monotone from below) in the original sample order.
    """
    n          = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p   = p_values[sorted_idx]
    q = np.array([min(1.0, (n * sorted_p[i] * pi0) / (i + 1))
                   for i in range(n)])
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    original_q = np.zeros(n)
    original_q[sorted_idx] = q
    return original_q


# ---------------------------------------------------------------------------
# Mix-Max FDR (multi-class classification, K >= 2)
# ---------------------------------------------------------------------------

def calculate_mixmax_qvalues(pred_scores: np.ndarray,
                              decoy_scores: np.ndarray,
                              pi0: float = 0.0) -> np.ndarray:
    """
    Mix-Max q-values (Keich et al., 2015).

    Parameters
    ----------
    pred_scores  : (N,) max model score per sample: f(t) = max_k F_K(t)[k]
    decoy_scores : (N,) max null  score per sample: g(t) = max_k G_K(t)[k]
    pi0          : proportion of truly null samples (set 0 for no-reject regime)

    Returns
    -------
    q_values : (N,) monotone FDR estimates, aligned with input order.
               Samples are implicitly sorted by pred_scores descending.
    """
    n = len(pred_scores)

    sort_idx           = np.argsort(pred_scores)
    pred_scores_sorted = pred_scores[sort_idx]
    sorted_decoys      = np.sort(decoy_scores)

    unique_z, counts_z = np.unique(decoy_scores, return_counts=True)
    n_unique_z         = len(unique_z)

    # Empirical CDFs at each unique decoy value
    counts_w_leq_z = np.searchsorted(pred_scores_sorted, unique_z, side='left')
    counts_z_leq_z = np.searchsorted(sorted_decoys,      unique_z, side='left')

    P_W = np.clip((counts_w_leq_z - pi0 * counts_z_leq_z) / ((1 - pi0) * n), 0, 1)
    P_Y = np.clip(counts_z_leq_z / n, 0, 1)
    R_j = np.clip(np.divide(P_W, P_Y, out=np.zeros_like(P_W), where=P_Y > 0), 0, 1)

    # Sweep thresholds from highest to lowest prediction score
    fdr_values = np.zeros(n)
    for i, T in enumerate(pred_scores_sorted[::-1]):
        D     = i + 1
        F_0   = pi0 * np.sum(decoy_scores > T)
        z_idx = np.searchsorted(unique_z, T, side='left')
        F_1   = 0.0 if z_idx >= n_unique_z else (
            (1 - pi0) * np.sum(R_j[z_idx:] * counts_z[z_idx:]))
        fdr_values[i] = (F_0 + F_1) / D if D > 0 else 0.0

    q_sorted = np.clip(np.minimum.accumulate(np.clip(fdr_values, 0, 1)[::-1]), 0, 1)

    # Map back to original sample order
    q_values = np.zeros(n)
    for i, orig_idx in enumerate(sort_idx):
        q_values[orig_idx] = q_sorted[i]
    return q_values


def mixmax_qvalues_from_vectors(model_scores: np.ndarray,
                                 decoy_scores: np.ndarray,
                                 pi0: float = 0.0) -> np.ndarray:
    """
    Convenience wrapper: extract max scores from (N, C) arrays then
    call calculate_mixmax_qvalues.
    """
    return calculate_mixmax_qvalues(
        pred_scores  = model_scores.max(axis=1),
        decoy_scores = decoy_scores.max(axis=1),
        pi0          = pi0,
    )


# ---------------------------------------------------------------------------
# TDC / Knock-off FDR (alternative to Mix-Max)
# ---------------------------------------------------------------------------

def calculate_tdc_qvalues(pred_scores: np.ndarray,
                           decoy_scores: np.ndarray) -> np.ndarray:
    """
    Target-Decoy Competition q-values.

    A sample is treated as a decoy win (false discovery) if the null
    score exceeds the model score. FDR is estimated as the ratio of
    decoy wins to total accepted samples.

    Returns q-values aligned with the input sort order (ascending score).
    """
    n         = len(pred_scores)
    tdc_score = np.maximum(pred_scores, decoy_scores)
    tdc_win   = (pred_scores > decoy_scores).astype(int)

    tdc_idx      = np.argsort(tdc_score)
    tdc_score_s  = tdc_score[tdc_idx]
    tdc_win_s    = tdc_win[tdc_idx]

    FD_CF = np.cumsum((1 - tdc_win_s)[::-1])[::-1]
    D_CF  = np.maximum(np.arange(n)[::-1] + 1 - FD_CF, 1)
    q     = np.clip(np.minimum.accumulate(np.clip(FD_CF / D_CF, 0, 1)), 0, 1)

    return q


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_qvalues(pred_scores: np.ndarray,
                    decoy_scores: np.ndarray,
                    method: str = 'mixmax',
                    pi0: float = 0.0) -> np.ndarray:
    """
    Compute q-values using the specified FDR method.

    Parameters
    ----------
    pred_scores  : (N,) max model scores
    decoy_scores : (N,) max null scores
    method       : 'mixmax' | 'tdc'
    pi0          : null proportion (Mix-Max only; ignored for TDC)
    """
    if method == 'mixmax':
        return calculate_mixmax_qvalues(pred_scores, decoy_scores, pi0=pi0)
    elif method == 'tdc':
        return calculate_tdc_qvalues(pred_scores, decoy_scores)
    else:
        raise ValueError(f"Unknown FDR method '{method}'. Choose 'mixmax' or 'tdc'.")
