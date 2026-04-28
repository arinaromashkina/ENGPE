"""
ENGPE evaluation on Camelyon17-WILDS (cross-hospital domain shift).

Binary classification: tumour vs. normal in histopathology patches.
Train: hospitals 0, 3, 4  →  Test: hospitals 1, 2.

Usage
-----
    python experiments/run_camelyon17.py --data_dir /path/to/wilds_data

Requires the wilds package (pip install wilds).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

from engpe import (
    ScoreShiftFlowWrapper,
    build_null_pools,
    compute_qvalues,
    estimate_accuracy,
    BASELINE_METHODS,
    ScoreFeatureDataset,
)
from engpe.fdr import empirical_pvalues, estimate_pi0_storey, benjamini_hochberg
from models.camelyon17 import Camelyon17Classifier, CAMELYON_TRAIN_TRANSFORM, CAMELYON_TEST_TRANSFORM
from models.camelyon17 import evaluate_accuracy, train


def collect_scores_binary(model: Camelyon17Classifier, loader, device: str = 'cuda'):
    """Return (N,1) logits, (N, 256) features, (N,) labels."""
    model.eval()
    sc, ft, lb = [], [], []
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            feats  = model.get_features(images)
            logits = model(images)
            sc.append(logits.cpu().numpy())
            ft.append(feats.cpu().numpy())
            lb.append(labels.cpu().numpy())
    return np.concatenate(sc), np.concatenate(ft), np.concatenate(lb)


def build_binary_null_dataset(logits: np.ndarray,
                               features: np.ndarray,
                               labels: np.ndarray) -> ScoreFeatureDataset:
    """
    Build training dataset for binary CNF.

    Null construction: for negative samples (label=0), keep score as-is.
    For positive samples, replace with a random draw from negatives' scores.
    """
    neg_pool = logits[labels == 0, 0]
    null     = logits.copy()
    pos_mask = labels == 1
    if len(neg_pool) > 0:
        null[pos_mask, 0] = np.random.choice(neg_pool, size=pos_mask.sum(), replace=True)
    return ScoreFeatureDataset(
        torch.from_numpy(logits).float(),
        torch.from_numpy(features).float(),
        torch.from_numpy(null).float(),
        torch.from_numpy(labels).long(),
    )


def main(args):
    os.makedirs('results/camelyon17', exist_ok=True)
    device = args.device

    # ── WILDS dataset ─────────────────────────────────────────────────────────
    dataset     = get_dataset('camelyon17', download=True, root_dir=args.data_dir)
    train_data  = dataset.get_subset('train',  transform=CAMELYON_TRAIN_TRANSFORM)
    val_data    = dataset.get_subset('val',    transform=CAMELYON_TEST_TRANSFORM)
    test_data   = dataset.get_subset('test',   transform=CAMELYON_TEST_TRANSFORM)
    trainloader = get_train_loader('standard', train_data, batch_size=32)
    valloader   = get_eval_loader('standard',  val_data,   batch_size=128)
    testloader  = get_eval_loader('standard',  test_data,  batch_size=128)

    # ── Classifier ───────────────────────────────────────────────────────────
    clf_path = 'results/camelyon17/classifier.pth'
    model    = Camelyon17Classifier(pretrained=True).to(device)
    if os.path.exists(clf_path):
        model.load_state_dict(torch.load(clf_path, map_location=device))
        print(f"Classifier loaded from {clf_path}")
    else:
        train(model, trainloader, valloader,
              epochs=10, device=device, save_path=clf_path)
    model.eval()
    print(f"Val accuracy:  {evaluate_accuracy(model, valloader,  device):.4f}")
    print(f"Test accuracy: {evaluate_accuracy(model, testloader, device):.4f}")

    # ── Training data for CNF ─────────────────────────────────────────────────
    train_logits, train_feats, train_labels = collect_scores_binary(
        model, DataLoader(train_data, batch_size=256), device)
    train_ds = build_binary_null_dataset(train_logits, train_feats, train_labels)

    flow_path = 'results/camelyon17/flow.pth'
    flow = ScoreShiftFlowWrapper(num_classes=1, n_flows=12,
                                  feature_dim=256, hidden_dim=256,
                                  encoder_dim=128).to(device)
    if os.path.exists(flow_path):
        flow.load_state_dict(torch.load(flow_path, map_location=device))
        print(f"Flow loaded from {flow_path}")
    else:
        print("Training CNF...")
        flow.train_flow(train_ds, epochs=45, lr=3e-4, device=device, patience=5)
        torch.save(flow.state_dict(), flow_path)
    flow.eval()

    # Source data for baselines (need 2-class logits; borrow binary logit as 2-col)
    src_logits_bin, _, src_labels = flow.generate_decoys(train_ds, device=device)

    # ── Test set evaluation (BH procedure for K=1) ───────────────────────────
    test_logits, test_feats, test_labels = collect_scores_binary(
        model, testloader, device)
    test_ds = build_binary_null_dataset(test_logits, test_feats, test_labels)
    ms, ds_flow, ls = flow.generate_decoys(test_ds, device=device)
    n = len(ls)

    pred_scores  = ms[:, 0]
    decoy_scores = ds_flow[:, 0]
    true_acc     = float((pred_scores >= 0) == ls.astype(bool)).mean()

    # Generate empirical p-values and apply BH
    null_pool = decoy_scores
    p_values  = empirical_pvalues(pred_scores, null_pool)
    pi0       = estimate_pi0_storey(p_values)
    q_bh      = benjamini_hochberg(p_values, pi0=pi0)

    # Also apply Mix-Max (treating score as 1-D)
    q_mm = compute_qvalues(pred_scores, decoy_scores, method='mixmax')

    sort_idx = np.argsort(pred_scores)
    q_sorted = q_mm[sort_idx]
    pi0_mm   = float(np.clip(q_sorted[0], 0, 1))
    acc_st, acc_ta, _ = estimate_accuracy(pred_scores[sort_idx], q_sorted, pi0_mm)

    print(f"\nCalmelyon17 results")
    print(f"  True accuracy : {true_acc:.4f}")
    print(f"  ENGPE-ST      : {acc_st:.4f}  (MAE={abs(acc_st - true_acc):.4f})")
    print(f"  ENGPE-TA      : {acc_ta:.4f}  (MAE={abs(acc_ta - true_acc):.4f})")
    print(f"  π₀ estimate   : {pi0_mm:.4f}  (true={1 - true_acc:.4f})")

    # Baselines (temperature-scaled; use sigmoid output as 2-class proxy)
    src_probs = torch.sigmoid(torch.tensor(src_logits_bin)).numpy()
    src_logits2 = np.hstack([np.log(1 - src_probs + 1e-8),
                              np.log(src_probs + 1e-8)])
    tgt_probs   = torch.sigmoid(torch.tensor(ms)).numpy()
    tgt_logits2 = np.hstack([np.log(1 - tgt_probs + 1e-8),
                              np.log(tgt_probs + 1e-8)])
    src_t = torch.tensor(src_logits2)
    tgt_t = torch.tensor(tgt_logits2)
    src_l = torch.tensor(src_labels)

    results = dict(true_acc=true_acc, acc_st=acc_st, acc_ta=acc_ta,
                   mae_st=abs(acc_st - true_acc), mae_ta=abs(acc_ta - true_acc))
    for mname, mfn in BASELINE_METHODS.items():
        try:
            est = float(np.clip(mfn(src_t, src_l, tgt_t), 0, 1))
        except Exception:
            est = float('nan')
        results[f'est_{mname}'] = est
        results[f'err_{mname}'] = abs(est - true_acc)
        print(f"  {mname:10s}: {est:.4f}  (MAE={abs(est - true_acc):.4f})")

    pd.DataFrame([results]).to_csv(
        'results/camelyon17/results.csv', index=False, float_format='%.6f')
    print("\nResults saved to results/camelyon17/results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ENGPE on Camelyon17')
    parser.add_argument('--data_dir', required=True,
                        help='Root directory for WILDS datasets')
    parser.add_argument('--device',   default='cuda')
    main(parser.parse_args())
