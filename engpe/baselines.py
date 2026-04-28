"""
Baseline unsupervised accuracy estimation methods.

All methods share the same interface:

    estimate = method(source_logits, source_labels, target_logits) → float

Parameters
----------
source_logits : (N_src, C) temperature-scaled logits on the validation set
source_labels : (N_src,)   true labels on the validation set
target_logits : (N_tgt, C) temperature-scaled logits on the OOD test set

References
----------
ATC, ATC-NE : Garg et al. (2022) "Leveraging Unlabeled Data to Predict OOD
               Performance", ICLR 2022.
AC, DOC      : Guillory et al. (2021) "Predicting with Confidence on Unseen
               Distributions", ICCV 2021.
COT          : requires the POT library (pip install POT).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float() if x.dtype != torch.float32 else x


def temperature_scale(logits: torch.Tensor, labels: torch.Tensor,
                      num_bins: int = 15) -> float:
    """
    Find the temperature T* that minimises ECE on a calibration set.
    Returns the scalar temperature value.
    """
    temps    = torch.linspace(0.1, 5.0, 50)
    best_t   = 1.0
    best_ece = float('inf')
    for temp in temps:
        probs      = torch.softmax(logits / temp, dim=1)
        conf, pred = probs.max(dim=1)
        acc        = (pred == labels).float()
        bins       = torch.linspace(0, 1, num_bins + 1)
        ece        = 0.0
        for i in range(num_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if mask.sum() > 0:
                ece += mask.float().mean() * (conf[mask].mean() - acc[mask].mean()).abs()
        if ece < best_ece:
            best_ece = ece
            best_t   = temp.item()
    return best_t


# ---------------------------------------------------------------------------
# ATC — Average Thresholded Confidence (Garg et al., 2022)
# ---------------------------------------------------------------------------

def predict_ATC_maxconf(source_logits, source_labels, target_logits) -> float:
    """ATC with maximum softmax confidence as the score."""
    src = _to_tensor(source_logits)
    lbl = _to_tensor(source_labels).long()
    tgt = _to_tensor(target_logits)
    src_scores  = torch.softmax(src, dim=1).amax(1)
    tgt_scores  = torch.softmax(tgt, dim=1).amax(1)
    n_correct   = (src.argmax(1) == lbl).sum()
    threshold   = torch.sort(src_scores).values[-(n_correct)]
    return float((tgt_scores > threshold).float().mean())


def predict_ATC_negent(source_logits, source_labels, target_logits) -> float:
    """ATC with negentropy as the score."""
    src = _to_tensor(source_logits)
    lbl = _to_tensor(source_labels).long()
    tgt = _to_tensor(target_logits)

    def negentropy(logits):
        probs = torch.softmax(logits, dim=1)
        ent   = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return float(np.log(logits.shape[1])) - ent

    src_scores = negentropy(src)
    tgt_scores = negentropy(tgt)
    n_correct  = (src.argmax(1) == lbl).sum()
    threshold  = torch.sort(src_scores).values[-(n_correct)]
    return float((tgt_scores > threshold).float().mean())


# ---------------------------------------------------------------------------
# AC — Average Confidence  (Guillory et al., 2021)
# ---------------------------------------------------------------------------

def predict_AC(source_logits, source_labels, target_logits) -> float:
    """Average softmax confidence on the target set."""
    tgt = _to_tensor(target_logits)
    return float(torch.softmax(tgt, dim=1).amax(1).mean())


# ---------------------------------------------------------------------------
# DOC — Difference Of Confidences  (Guillory et al., 2021)
# ---------------------------------------------------------------------------

def predict_DOC(source_logits, source_labels, target_logits) -> float:
    """Source accuracy + (target confidence − source confidence)."""
    src = _to_tensor(source_logits)
    lbl = _to_tensor(source_labels).long()
    tgt = _to_tensor(target_logits)
    src_conf = float(torch.softmax(src, dim=1).amax(1).mean())
    tgt_conf = float(torch.softmax(tgt, dim=1).amax(1).mean())
    src_acc  = float((src.argmax(1) == lbl).float().mean())
    return src_acc + (tgt_conf - src_conf)


# ---------------------------------------------------------------------------
# COT — Confidence Optimal Transport  (optional, requires POT)
# ---------------------------------------------------------------------------

try:
    import ot as _ot

    def predict_COT(source_logits, source_labels, target_logits) -> float:
        """Confidence-OT estimator. Requires the POT library (pip install POT)."""
        src = _to_tensor(source_logits)
        lbl = _to_tensor(source_labels).long()
        tgt = _to_tensor(target_logits)

        K              = src.shape[1]
        src_label_dist = F.one_hot(lbl, K).float().mean(0)
        tgt_probs      = torch.softmax(tgt, dim=1)
        cost_matrix    = torch.stack(
            [(tgt_probs - e).abs().sum(1) / 2 for e in torch.eye(K)], dim=1)

        ot_plan  = _ot.emd(np.ones(len(tgt_probs)) / len(tgt_probs),
                           src_label_dist.numpy(),
                           cost_matrix.numpy())
        ot_cost  = float(np.sum(ot_plan * cost_matrix.numpy()))
        src_conf = float(torch.softmax(src, dim=1).amax(1).mean())
        src_acc  = float((src.argmax(1) == lbl).float().mean())
        return 1.0 - (ot_cost + src_conf - src_acc)

    COT_AVAILABLE = True

except ImportError:
    COT_AVAILABLE = False

    def predict_COT(*args, **kwargs):
        raise ImportError("COT requires the POT library. Install with: pip install POT")


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

BASELINE_METHODS: dict = {
    'ATC':    predict_ATC_maxconf,
    'ATC-NE': predict_ATC_negent,
    'AC':     predict_AC,
    'DOC':    predict_DOC,
}
if COT_AVAILABLE:
    BASELINE_METHODS['COT'] = predict_COT
