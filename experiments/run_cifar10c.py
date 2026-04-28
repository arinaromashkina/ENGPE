"""
ENGPE evaluation on CIFAR-10-C corruption benchmarks.

Evaluates 19 corruption types × 5 severity levels (= 95 test sets).
Classifier: Wide ResNet-28-10 trained on clean CIFAR-10.

Usage
-----
    python experiments/run_cifar10c.py \\
        --cifar10_dir  /path/to/CIFAR-10 \\
        --cifar10c_dir /path/to/CIFAR-10-C
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from engpe import (
    ScoreShiftFlowWrapper,
    build_null_pools,
    build_dataset_from_scores,
    compute_qvalues,
    estimate_accuracy,
    BASELINE_METHODS,
    temperature_scale,
)
from models.cifar10c import WideResNet, CIFAR10_TRAIN_TRANSFORM, CIFAR10_TEST_TRANSFORM, evaluate_accuracy, train

CORRUPTIONS = [
    "fog", "frost", "motion_blur", "brightness", "zoom_blur",
    "snow", "defocus_blur", "glass_blur", "gaussian_noise",
    "shot_noise", "impulse_noise", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression", "speckle_noise", "spatter",
    "gaussian_blur", "saturate",
]
SEVERITIES = [1, 2, 3, 4, 5]


def collect_scores(model: WideResNet, loader, device: str = 'cuda'):
    model.eval()
    sc, ft, lb = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            feats  = model.get_features(images.to(device))
            scores = model.linear2(F.relu(feats))
            sc.append(scores.cpu().numpy())
            ft.append(feats.cpu().numpy())
            lb.append(labels.numpy())
    return np.concatenate(sc), np.concatenate(ft), np.concatenate(lb)


def main(args):
    os.makedirs('results/cifar10c', exist_ok=True)
    device = args.device

    # ── Datasets ─────────────────────────────────────────────────────────────
    trainset = datasets.CIFAR10(args.cifar10_dir, train=True,
                                 transform=CIFAR10_TRAIN_TRANSFORM, download=True)
    valset   = datasets.CIFAR10(args.cifar10_dir, train=False,
                                 transform=CIFAR10_TEST_TRANSFORM, download=True)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=4)
    valloader   = DataLoader(valset,   batch_size=256, shuffle=False, num_workers=4)

    # ── Classifier ───────────────────────────────────────────────────────────
    clf_path = 'results/cifar10c/wresnet.pth'
    model    = WideResNet(num_classes=10).to(device)
    if os.path.exists(clf_path):
        model.load_state_dict(torch.load(clf_path, map_location=device))
        print(f"Classifier loaded from {clf_path}")
    else:
        train(model, trainloader, valloader,
              epochs=200, device=device, save_path=clf_path)
    model.eval()
    print(f"Clean val accuracy: {evaluate_accuracy(model, valloader, device):.4f}")

    # ── Training scores and CNF ───────────────────────────────────────────────
    train_scores, train_feats, train_labels = collect_scores(model, trainloader, device)
    pool_score, _ = build_null_pools(train_scores, train_labels, num_classes=10)
    train_ds      = build_dataset_from_scores(train_scores, train_feats,
                                               train_labels, pool_score)

    flow_path = 'results/cifar10c/flow.pth'
    flow = ScoreShiftFlowWrapper(num_classes=10, n_flows=12,
                                  feature_dim=640, hidden_dim=256,
                                  encoder_dim=128).to(device)
    if os.path.exists(flow_path):
        flow.load_state_dict(torch.load(flow_path, map_location=device))
        print(f"Flow loaded from {flow_path}")
    else:
        flow.train_flow(train_ds, epochs=30, lr=3e-4, device=device)
        torch.save(flow.state_dict(), flow_path)
    flow.eval()

    src_logits, _, src_labels = flow.generate_decoys(train_ds, device=device)
    src_t = torch.tensor(src_logits).to(device)
    src_l = torch.tensor(src_labels).to(device)
    temp  = temperature_scale(src_t, src_l)
    print(f"Temperature: {temp:.4f}")

    # ── CIFAR-10-C evaluation ─────────────────────────────────────────────────
    # CIFAR-10-C stores all corruptions as numpy arrays of shape (50000, 32, 32, 3)
    labels_c = np.load(os.path.join(args.cifar10c_dir, 'labels.npy'))

    all_results = []
    for corruption in CORRUPTIONS:
        data_path = os.path.join(args.cifar10c_dir, f'{corruption}.npy')
        if not os.path.exists(data_path):
            print(f"  Skipping {corruption} (not found)")
            continue
        data_all = np.load(data_path)   # (50000, 32, 32, 3)

        for severity in SEVERITIES:
            ds_name = f'{corruption}_sev{severity}'
            start   = (severity - 1) * 10000
            images  = data_all[start:start + 10000]
            labels  = labels_c[start:start + 10000]

            imgs_t   = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
            mean     = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std      = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            imgs_t   = (imgs_t - mean) / std
            lbl_t    = torch.from_numpy(labels).long()
            test_ds  = torch.utils.data.TensorDataset(imgs_t, lbl_t)
            loader   = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

            test_scores, test_feats, test_labels = collect_scores(model, loader, device)
            test_ds_flow = build_dataset_from_scores(test_scores, test_feats,
                                                      test_labels, pool_score)
            ms, ds_flow, ls = flow.generate_decoys(test_ds_flow, device=device)
            n        = len(ls)
            true_acc = float((ms.argmax(1) == ls).mean())
            pred     = ms.max(axis=1)
            decoy    = ds_flow.max(axis=1)
            sort_idx = np.argsort(pred)
            q_mm     = compute_qvalues(pred, decoy, method='mixmax')
            q_sorted = q_mm[sort_idx]
            pi0_mm   = float(np.clip(q_sorted[0], 0, 1))
            acc_st, acc_ta, _ = estimate_accuracy(pred[sort_idx], q_sorted, pi0_mm)

            tgt_t = torch.tensor(ms).to(device) / temp
            row   = dict(testset=ds_name, corruption=corruption, severity=severity,
                         n=n, true_acc=true_acc, acc_st_mm=acc_st, acc_ta_mm=acc_ta,
                         err_st=abs(acc_st - true_acc), err_ta=abs(acc_ta - true_acc))
            for mname, mfn in BASELINE_METHODS.items():
                try:
                    est = float(np.clip(mfn(src_t / temp, src_l, tgt_t), 0, 1))
                except Exception:
                    est = float('nan')
                row[f'est_{mname}'] = est
                row[f'err_{mname}'] = abs(est - true_acc)
            all_results.append(row)
            print(f"  {ds_name:35s}  true={true_acc:.4f}"
                  f"  ENGPE-ST={acc_st:.4f}  ENGPE-TA={acc_ta:.4f}")

    df = pd.DataFrame(all_results)
    df.to_csv('results/cifar10c/results.csv', index=False, float_format='%.6f')
    print("\nResults saved to results/cifar10c/results.csv")
    print("\n" + "="*40 + " SUMMARY " + "="*40)
    err_cols = ['err_st', 'err_ta'] + [f'err_{m}' for m in BASELINE_METHODS]
    print(df[err_cols].mean().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ENGPE on CIFAR-10-C')
    parser.add_argument('--cifar10_dir',  required=True)
    parser.add_argument('--cifar10c_dir', required=True)
    parser.add_argument('--device',       default='cuda')
    main(parser.parse_args())
