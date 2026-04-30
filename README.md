# ENGPE — Empirical Null Generation-based Performance Evaluation

**Label-free performance estimation under distribution shift for multi-class deep neural networks.**

---

## Overview

Modern AI models decay in production due to unforeseen data drift, yet measuring their actual performance requires expensive ground-truth labels. **ENGPE** estimates accuracy, precision, recall, and F1 at any decision threshold — *without using test labels*.

**Key idea.** For a classifier `F = F_C ∘ F_R`, we train a shallow conditional normalizing flow (CNF) `G` to generate *empirical null scores* conditioned on the penultimate features `F_R(t)`. Because both branches share the same feature extractor, any distribution shift in the test data propagates equally through the model and the generator — keeping the null approximation calibrated without retraining.

```
Input t ──► F_R ──►┬──► F_C ──► predicted logits  F(t)
                   │
                   └──► Encoder ──► CNF ──► null logits  G(t)
```

The null logits are then used with the **Mix-Max** FDR procedure (Keich et al., 2015) to estimate False Discovery Rate at every decision threshold, from which TP, FP, TN, FN, and all standard metrics follow analytically.

### Why ENGPE?

| Property | ATC / DOC / AC | MaNo / NucNorm | **ENGPE** |
|---|:---:|:---:|:---:|
| No test labels needed | ✓ | ✓ | ✓ |
| Threshold-adapted accuracy | ✗ | ✗ | ✓ |
| Per-sample q-values | ✗ | ✗ | ✓ |
| Works under subpopulation shift | ✗ | ✗ | ✓ |

**Results:** ENGPE reduces mean absolute error (MAE) by **36%** on average across four challenging benchmarks (Camelyon17, CIFAR-10-C, BCSS, BREEDS) compared to the best competitor.

| Dataset | Best competitor MAE | **ENGPE MAE** | Reduction |
|---|:---:|:---:|:---:|
| Camelyon-17 | 0.027 (COT) | **0.007** | 74% |
| BCSS | 0.166 (ATC) | **0.084** | 49% |
| BREEDS | 0.125 (ATC-NE) | **0.082** | 34% |
| CIFAR-10-C | 0.039 (ATC-NE) | 0.071 | — |
| **Average** | 0.095 | **0.061** | **36%** |

---

## Installation

```bash
git clone https://github.com/<your-handle>/ENGPE.git
cd ENGPE
pip install -e .
# For BREEDS experiments:
pip install git+https://github.com/MadryLab/robustness.git
# For Camelyon17 experiments:
pip install wilds
# Optional COT baseline:
pip install POT
```

---

## Quick Start

```python
import torch
from engpe import ScoreShiftFlowWrapper, compute_qvalues, estimate_accuracy

# 1. Build a ScoreFeatureDataset from your training data
from engpe import build_null_pools, build_dataset_from_scores
pool_score, _ = build_null_pools(train_scores, train_labels, num_classes=K)
train_ds = build_dataset_from_scores(train_scores, train_features,
                                      train_labels, pool_score)

# 2. Train the CNF generator (once, after your main classifier is trained)
flow = ScoreShiftFlowWrapper(num_classes=K, feature_dim=D)
flow.train_flow(train_ds, epochs=30, device='cuda')
torch.save(flow.state_dict(), 'engpe_flow.pth')

# 3. At test time: generate null scores for the OOD test set
model_scores, decoy_scores, _ = flow.generate_decoys(test_ds, device='cuda')

# 4. Estimate accuracy without labels
pred  = model_scores.max(axis=1)   # max logit per sample
decoy = decoy_scores.max(axis=1)
q     = compute_qvalues(pred, decoy, method='mixmax')
pi0   = float(q[0])                # proportion of false discoveries

import numpy as np
sort_idx = np.argsort(pred)
acc_st, acc_ta, threshold = estimate_accuracy(pred[sort_idx], q[sort_idx], pi0)
print(f"Estimated standard accuracy:          {acc_st:.4f}")
print(f"Estimated threshold-adapted accuracy: {acc_ta:.4f}")
print(f"Optimal decision threshold:           {threshold:.4f}")
```

---

## Repository Structure

```
ENGPE/
├── engpe/                     ← core library
│   ├── __init__.py
│   ├── flow.py                ← Conditional Normalizing Flow (CNF)
│   ├── null_pool.py           ← null score pool construction
│   ├── dataset.py             ← ScoreFeatureDataset + builders
│   ├── fdr.py                 ← BH, Mix-Max, TDC FDR procedures
│   ├── performance.py         ← accuracy / PR estimation from FDR
│   └── baselines.py           ← ATC, ATC-NE, AC, DOC, COT
├── models/                    ← dataset-specific classifier architectures
│   ├── breeds.py              ← ResNet-50 for BREEDS
│   ├── cifar10c.py            ← WideResNet-28-10 for CIFAR-10-C
│   └── camelyon17.py          ← EfficientNet-B0 for Camelyon17
└── experiments/               ← end-to-end evaluation scripts
    ├── run_breeds.py
    ├── run_bcss.py
    ├── run_cifar10c.py
    └── run_camelyon17.py
```

---

## Reproducing Experiments

### BREEDS (subpopulation shift)

```bash
python experiments/run_breeds.py \
    --name entity30 \
    --data_dir /path/to/imagenet \
    --device cuda
```

Available `--name` options: `entity13`, `entity30`, `living17`, `nonliving26`.

### BCSS (breast cancer segmentation)

```bash
python experiments/run_bcss.py \
    --train_data /path/to/bcss.mini.training.torch \
    --test_dir   /path/to/BCSS/test/ \
    --device cuda
```

### CIFAR-10-C (image corruptions)

```bash
python experiments/run_cifar10c.py \
    --cifar10_dir  /path/to/CIFAR-10 \
    --cifar10c_dir /path/to/CIFAR-10-C \
    --device cuda
```

### Camelyon17-WILDS (cross-hospital domain shift)

```bash
python experiments/run_camelyon17.py \
    --data_dir /path/to/wilds_data \
    --device cuda
```

---

## Method Details

### Conditional Normalizing Flow

The CNF `G_K : R^K → R^K` is trained to model the null logit distribution conditioned on the penultimate features `h = F_R(t)`:

**Architecture:**
- **Robust feature normalizer** — running median/IQR + tanh compression, bounds OOD features to (-1, 1)
- **Feature encoder** — 2-layer MLP (256 → 128), LayerNorm + GELU
- **N_f = 12 affine coupling layers** (alternating first-half / second-half masks)
- **ActNorm** after every coupling layer (except the last)

**Null logit construction (training):** For each training sample with predicted class `c`, the `c`-th coordinate of its logit vector is replaced by a random draw from the pool of logit-`c` scores of non-class-c samples. All other coordinates are kept unchanged to preserve the multivariate correlation structure.

**Objective:** Maximum log-likelihood on (null_logit_vector, features) pairs.

### Performance Estimation

Given q-values `q(t)` from Mix-Max FDR at threshold `s_th`:

```
TP(s_th) = |{f(t) ≥ s_th}| · (1 - q(s_th))
FP(s_th) = |{f(t) ≥ s_th}| · q(s_th)
TN(s_th) = N · π₀ − FP(s_th)
FN(s_th) = N · (1 − π₀) − TP(s_th)
```

- **ACC_ST** = accuracy accepting all samples (threshold = −∞)
- **ACC_TA** = max over thresholds of ACC(threshold) — finds the best operating point

---

## Acknowledgements

The baseline implementations are based on:
- [ATC / DOC](https://github.com/saurabhgarg1996/ATC_code) — Garg et al. (2022)
- [MaNo](https://github.com/Renchunzi-Xie/MaNo) — Xie et al. (2024)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
