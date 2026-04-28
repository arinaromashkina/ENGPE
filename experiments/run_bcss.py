"""
ENGPE evaluation on BCSS (Breast Cancer Semantic Segmentation).

Each test image is treated as a separate test set; ENGPE estimates per-pixel
classification accuracy for every image without using pixel labels.

Data format: pre-computed .tensor files from the TIA toolbox FCN-ResNet50 model.
Each file contains 'predictions', 'features', and 'mask' fields.

Usage
-----
    python experiments/run_bcss.py \\
        --train_data /path/to/bcss.mini.training.torch \\
        --test_dir   /path/to/BCSS/test/
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from engpe import (
    ScoreShiftFlowWrapper,
    build_null_pools,
    compute_qvalues,
    estimate_accuracy,
    BASELINE_METHODS,
    temperature_scale,
    ScoreFeatureDataset,
)
from engpe.dataset import build_dataset_from_bcss_tile

NUM_CLASSES  = 5
FEATURE_DIM  = 64
SUBSAMPLE    = 10   # evaluate every N-th pixel to keep memory reasonable


# ---------------------------------------------------------------------------
# Build training dataset from BCSS pre-computed logits
# ---------------------------------------------------------------------------

def build_bcss_train_dataset(data: dict) -> ScoreFeatureDataset:
    """Convert BCSS training dict → ScoreFeatureDataset with null replacement."""
    total_preds    = data['total_preds']
    total_features = data['total_features']
    classes        = sorted(total_preds.keys())

    neg_pools = {}
    for c in classes:
        neg_scores = [total_preds[oc][c] for oc in classes if oc != c]
        neg_pools[c] = torch.cat(neg_scores).cpu().numpy()

    sc, ft, dc, lb = [], [], [], []
    rng = np.random.default_rng(42)
    for c in classes:
        preds = total_preds[c].T.cpu()    # (N_c, C)
        feats = total_features[c].T.cpu() # (N_c, D)
        N_c   = preds.shape[0]
        null  = preds.clone()
        null[:, c] = torch.tensor(
            rng.choice(neg_pools[c], size=N_c), dtype=null.dtype)
        sc.append(preds)
        ft.append(feats)
        dc.append(null)
        lb.append(torch.full((N_c,), c, dtype=torch.long))

    return ScoreFeatureDataset(
        torch.cat(sc), torch.cat(ft), torch.cat(dc), torch.cat(lb))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs('results/bcss', exist_ok=True)
    device = args.device

    print("Loading training data...")
    data     = torch.load(args.train_data, weights_only=False)
    train_ds = build_bcss_train_dataset(data)
    print(f"  Training pixels: {len(train_ds)}")

    # ── CNF ─────────────────────────────────────────────────────────────────
    flow_path = 'results/bcss/flow.pth'
    flow = ScoreShiftFlowWrapper(num_classes=NUM_CLASSES, n_flows=12,
                                  feature_dim=FEATURE_DIM, hidden_dim=256,
                                  encoder_dim=128).to(device)
    if os.path.exists(flow_path):
        flow.load_state_dict(torch.load(flow_path, map_location=device))
        print(f"Flow loaded from {flow_path}")
    else:
        print("Training CNF...")
        flow.train_flow(train_ds, epochs=40, lr=3e-4, device=device, patience=5)
        torch.save(flow.state_dict(), flow_path)
        print(f"Flow saved to {flow_path}")
    flow.eval()

    # Source data for baselines
    np.random.seed(42)
    sub_idx  = np.random.choice(len(train_ds), int(0.4 * len(train_ds)), replace=False)
    sub_ds   = Subset(train_ds, sub_idx.tolist())
    src_logits, _, src_labels = flow.generate_decoys(sub_ds, device=device)
    src_t    = torch.tensor(src_logits).to(device)
    src_l    = torch.tensor(src_labels).to(device)
    temp     = temperature_scale(src_t, src_l)
    print(f"Temperature: {temp:.4f}")

    # ── Evaluate per test image ───────────────────────────────────────────
    test_files = sorted(glob.glob(os.path.join(args.test_dir, '*.tensor')))
    print(f"\nFound {len(test_files)} test images")
    all_results = []

    for fpath in test_files:
        tile_name = os.path.basename(fpath)
        print(f"\n  {tile_name}")
        try:
            test_ds = build_dataset_from_bcss_tile(fpath)
        except Exception as e:
            print(f"    Failed to load: {e}")
            continue

        ms, ds_flow, ls = flow.generate_decoys(test_ds, device=device)
        ms      = ms[::SUBSAMPLE]
        ds_flow = ds_flow[::SUBSAMPLE]
        ls      = ls[::SUBSAMPLE]
        n       = len(ls)
        if n < 50:
            print(f"    Too few pixels ({n}), skipping")
            continue

        true_acc  = float((ms.argmax(1) == ls).mean())
        pred      = ms.max(axis=1)
        decoy     = ds_flow.max(axis=1)
        sort_idx  = np.argsort(pred)
        q_mm      = compute_qvalues(pred, decoy, method='mixmax')
        q_sorted  = q_mm[sort_idx]
        pi0_mm    = float(np.clip(q_sorted[0], 0, 1))
        acc_st, acc_ta, _ = estimate_accuracy(pred[sort_idx], q_sorted, pi0_mm)

        tgt_t = torch.tensor(ms).to(device) / temp
        row = dict(tile=tile_name, n_pixels=n, true_acc=true_acc,
                   acc_st_mm=acc_st, acc_ta_mm=acc_ta,
                   err_st=abs(acc_st - true_acc),
                   err_ta=abs(acc_ta - true_acc))
        for mname, mfn in BASELINE_METHODS.items():
            try:
                est = float(np.clip(mfn(src_t / temp, src_l, tgt_t), 0, 1))
            except Exception:
                est = float('nan')
            row[f'est_{mname}'] = est
            row[f'err_{mname}'] = abs(est - true_acc)
        all_results.append(row)
        print(f"    true={true_acc:.4f}  ENGPE-ST={acc_st:.4f}  ENGPE-TA={acc_ta:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = 'results/bcss/results.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nResults saved to {csv_path}")
    print(f"\n{'='*40} SUMMARY (mean MAE) {'='*40}")
    err_cols = ['err_st', 'err_ta'] + [f'err_{m}' for m in BASELINE_METHODS]
    print(df[err_cols].mean().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ENGPE on BCSS')
    parser.add_argument('--train_data', required=True,
                        help='Path to bcss.mini.training.torch')
    parser.add_argument('--test_dir',   required=True,
                        help='Directory containing test .tensor files')
    parser.add_argument('--device',     default='cuda')
    main(parser.parse_args())
