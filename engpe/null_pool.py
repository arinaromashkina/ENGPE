"""
Null score pool construction for CNF training.

For each predicted class c, the null pool consists of the logit scores
on the c-th coordinate for samples that do NOT belong to class c (negative
samples). During CNF training the true-class coordinate of each sample's
logit vector is replaced with a random draw from the corresponding pool,
producing paired (features, null_logits) training observations.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

_MIN_POOL = 50  # minimum pool size before falling back to all negatives


def build_null_pools(train_scores: np.ndarray,
                     train_labels: np.ndarray,
                     num_classes: int,
                     verbose: bool = True) -> tuple[dict, dict]:
    """
    Build per-class null score pools from training data.

    The primary pool for class c contains logit[c] values of training
    samples where argmax == c AND label != c (true errors). If fewer
    than 50 errors exist we fall back to all samples with label != c.

    Parameters
    ----------
    train_scores : (N, C) array of logit vectors
    train_labels : (N,) array of true class labels
    num_classes  : number of classes K

    Returns
    -------
    pool_score   : dict[int → np.ndarray]  scalar null scores per class
    pool_vectors : dict[int → np.ndarray]  full score vectors with label != c
    """
    pred_classes = train_scores.argmax(axis=1)
    pool_score, pool_vectors = {}, {}

    if verbose:
        acc = (pred_classes == train_labels).mean()
        print(f"Building null pools  (train acc={acc:.4f})")

    for c in range(num_classes):
        error_mask = (pred_classes == c) & (train_labels != c)
        n_err = error_mask.sum()

        if n_err >= _MIN_POOL:
            pool_score[c] = train_scores[error_mask, c]
            src = "errors only"
        else:
            neg_mask = train_labels != c
            pool_score[c] = train_scores[neg_mask, c]
            src = "all negatives"

        pool_vectors[c] = train_scores[train_labels != c]

        if verbose:
            p = pool_score[c]
            print(f"  class {c:2d}: n={len(p):7d}  "
                  f"[{p.min():.3f}, {p.max():.3f}]  mean={p.mean():.3f}"
                  f"  n_errors={n_err:5d}  ({src})")

    return pool_score, pool_vectors


def build_decoy_vectors(scores: np.ndarray,
                        pool_score: dict,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Score-coordinate decoy construction.

    For each sample predicted as class c, the c-th coordinate of the
    logit vector is replaced by a draw from pool_score[c]. All other
    coordinates are kept unchanged, preserving the multivariate
    correlation structure of the logit vector.

    Parameters
    ----------
    scores     : (N, C) model logit array
    pool_score : dict[int → np.ndarray] from build_null_pools()
    rng        : numpy random generator

    Returns
    -------
    decoys : (N, C) array with replaced c-th coordinates
    """
    pred_classes = scores.argmax(axis=1)
    decoys = scores.copy()
    for c in range(scores.shape[1]):
        mask = pred_classes == c
        if not mask.any():
            continue
        pool = pool_score.get(c, [])
        if len(pool) == 0:
            continue
        decoys[mask, c] = rng.choice(pool, size=mask.sum(), replace=True)
    return decoys


def collect_negative_scores(model, train_dataset,
                             num_classes: int = 10,
                             device: str = 'cuda') -> dict:
    """
    Collect per-class negative logit score pools by running a model
    over a labelled dataset.

    For class k, the pool contains score[k] for all samples with label != k.

    Parameters
    ----------
    model        : classifier with get_features() and linear2 attributes
    train_dataset: labelled PyTorch dataset
    num_classes  : K
    device       : torch device string

    Returns
    -------
    negative_scores_pools : dict[int → np.ndarray]
    """
    import torch
    from torch.utils.data import DataLoader

    negative_scores_pools = {i: [] for i in range(num_classes)}
    model.eval()
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Collecting negative scores'):
            images, labels = images.to(device), labels.to(device)
            features = model.get_features(images)
            scores   = model.linear2(features)
            for c in range(num_classes):
                neg_mask = labels != c
                if neg_mask.sum() > 0:
                    negative_scores_pools[c].append(scores[neg_mask, c].cpu())

    for c in range(num_classes):
        if negative_scores_pools[c]:
            negative_scores_pools[c] = torch.cat(negative_scores_pools[c]).numpy()
        else:
            negative_scores_pools[c] = np.array([])

    return negative_scores_pools
