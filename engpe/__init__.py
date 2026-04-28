"""
ENGPE — Empirical Null Generation-based Performance Evaluation.

Label-free accuracy estimation under data distribution shift for
multi-class deep neural networks.

Quick start
-----------
>>> from engpe import ScoreShiftFlowWrapper, compute_qvalues, estimate_accuracy

1. Train the CNF generator on your labelled training set:
   >>> flow = ScoreShiftFlowWrapper(num_classes=K, feature_dim=D)
   >>> flow.train_flow(train_dataset, epochs=30, device='cuda')

2. Generate null scores for an OOD test set:
   >>> model_scores, decoy_scores, _ = flow.generate_decoys(test_dataset)

3. Compute FDR q-values and estimate accuracy:
   >>> pred  = model_scores.max(axis=1)
   >>> decoy = decoy_scores.max(axis=1)
   >>> q     = compute_qvalues(pred, decoy, method='mixmax')
   >>> acc_st, acc_ta, threshold = estimate_accuracy(
   ...     pred, q, pi0=float(q[0]))
"""

from engpe.flow import ScoreShiftFlowWrapper, ScoreShiftFlow
from engpe.fdr import compute_qvalues, calculate_mixmax_qvalues, calculate_tdc_qvalues
from engpe.performance import estimate_accuracy, estimate_precision_recall
from engpe.null_pool import build_null_pools, build_decoy_vectors
from engpe.dataset import ScoreFeatureDataset, build_dataset_from_scores
from engpe.baselines import BASELINE_METHODS, temperature_scale

__all__ = [
    'ScoreShiftFlowWrapper',
    'ScoreShiftFlow',
    'compute_qvalues',
    'calculate_mixmax_qvalues',
    'calculate_tdc_qvalues',
    'estimate_accuracy',
    'estimate_precision_recall',
    'build_null_pools',
    'build_decoy_vectors',
    'ScoreFeatureDataset',
    'build_dataset_from_scores',
    'BASELINE_METHODS',
    'temperature_scale',
]
