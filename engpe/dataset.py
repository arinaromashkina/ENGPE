"""
Dataset utilities for ENGPE training and evaluation.

ScoreFeatureDataset stores the four-tuple
    (logit_scores, penultimate_features, null_decoy_scores, labels)
needed to train the conditional normalizing flow.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ScoreFeatureDataset(Dataset):
    """
    Dataset of (logit_scores, penultimate_features, null_decoys, labels).

    Attributes
    ----------
    cnn_scores   : (N, C) tensor — classifier logit vectors
    features     : (N, D) tensor — penultimate-layer feature vectors
    null_decoys  : (N, C) tensor — null logit vectors (target for CNF training)
    labels       : (N,)   tensor — true class labels
    """

    def __init__(self, cnn_scores: torch.Tensor, features: torch.Tensor,
                 null_decoys: torch.Tensor, labels: torch.Tensor):
        assert len(cnn_scores) == len(features), (
            f"score/feature length mismatch: {len(cnn_scores)} vs {len(features)}")
        self.cnn_scores  = cnn_scores
        self.features    = features
        self.null_decoys = null_decoys
        self.labels      = labels

    def __len__(self) -> int:
        return len(self.cnn_scores)

    def __getitem__(self, idx):
        return (self.cnn_scores[idx], self.features[idx],
                self.null_decoys[idx], self.labels[idx])


# ---------------------------------------------------------------------------
# Builders for standard datasets
# ---------------------------------------------------------------------------

def build_dataset_from_classifier(dataset, model,
                                   null_pools: dict,
                                   device: str = 'cuda',
                                   batch_size: int = 64) -> ScoreFeatureDataset:
    """
    Run a classifier over a labelled dataset and return a ScoreFeatureDataset.

    Expects the model to have:
      - ``get_features(x) → (B, D)`` penultimate features
      - ``linear2(h)       → (B, C)`` output logits

    The null score for each sample replaces the true-class coordinate with
    a random draw from null_pools[true_class].
    """
    cnn_list, feat_list, null_list, lbl_list = [], [], [], []
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Building dataset', leave=False):
            images  = images.to(device)
            feats   = model.get_features(images)
            scores  = model.linear2(feats)

            scores_cpu = scores.cpu()
            feats_cpu  = feats.cpu()
            null_vecs  = scores_cpu.clone()

            for i, label in enumerate(labels):
                c    = label.item()
                pool = null_pools.get(c, [])
                if len(pool) > 0:
                    null_vecs[i, c] = torch.tensor(
                        np.random.choice(pool), dtype=null_vecs.dtype)

            cnn_list.append(scores_cpu)
            feat_list.append(feats_cpu)
            null_list.append(null_vecs)
            lbl_list.append(labels)

    return ScoreFeatureDataset(
        torch.cat(cnn_list), torch.cat(feat_list),
        torch.cat(null_list), torch.cat(lbl_list),
    )


def build_dataset_from_scores(scores: np.ndarray,
                               features: np.ndarray,
                               labels: np.ndarray,
                               pool_score: dict,
                               rng: np.random.Generator | None = None) -> ScoreFeatureDataset:
    """
    Build a ScoreFeatureDataset from pre-computed numpy arrays.

    Uses the score-coordinate null construction: for each sample predicted
    as class c, the c-th coordinate of its logit vector is replaced with a
    draw from pool_score[c].

    Parameters
    ----------
    scores    : (N, C) array of logit vectors
    features  : (N, D) array of penultimate features
    labels    : (N,)   array of true class labels
    pool_score: dict[int → np.ndarray] from engpe.null_pool.build_null_pools()
    rng       : optional numpy random generator (created if None)
    """
    from engpe.null_pool import build_decoy_vectors
    if rng is None:
        rng = np.random.default_rng(42)
    null_vecs = build_decoy_vectors(scores, pool_score, rng)
    return ScoreFeatureDataset(
        torch.from_numpy(scores).float(),
        torch.from_numpy(features).float(),
        torch.from_numpy(null_vecs).float(),
        torch.from_numpy(labels).long(),
    )


def build_dataset_from_bcss_tile(tile_path: str) -> ScoreFeatureDataset:
    """
    Load one BCSS .tensor tile and return a ScoreFeatureDataset.

    The null slot is set to the original scores (placeholder); decoys are
    generated at evaluation time by the trained CNF.
    """
    data        = torch.load(tile_path, weights_only=False)
    predictions = torch.flatten(data['predictions'], start_dim=2).squeeze(0).T
    features    = torch.flatten(data['features'],    start_dim=2).squeeze(0).T
    labels      = torch.flatten(torch.tensor(data['mask']), start_dim=0)
    labels      = torch.where(labels <= 3, labels, torch.tensor(4))
    return ScoreFeatureDataset(predictions, features, predictions.clone(), labels)
