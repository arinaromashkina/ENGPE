"""
ENGPE evaluation on BREEDS subpopulation-shift benchmarks.

Datasets: Living-17 | Entity-13 | Entity-30 | Nonliving-26
Each benchmark tests both source and target subpopulation splits,
plus 19 ImageNet-C corruption types × 5 severity levels.

Usage
-----
    python experiments/run_breeds.py --name entity30 --data_dir /path/to/imagenet

Required data layout
--------------------
    <data_dir>/
      imagenetv1/train/
      imagenetv1/val/
      imagenet-c/<corruption>/<severity>/
      imagenet_class_hierarchy/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from robustness.tools.breeds_helpers import (
    make_living17, make_entity13, make_entity30, make_nonliving26,
)

from engpe import (
    ScoreShiftFlowWrapper,
    build_null_pools,
    build_dataset_from_scores,
    compute_qvalues,
    estimate_accuracy,
    BASELINE_METHODS,
    temperature_scale,
)
from models.breeds import BREEDSClassifier, BREEDS_TRANSFORM, evaluate_accuracy, train

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_C_CORRUPTIONS = [
    "fog", "frost", "motion_blur", "brightness", "zoom_blur",
    "snow", "defocus_blur", "glass_blur", "gaussian_noise",
    "shot_noise", "impulse_noise", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression", "speckle_noise", "spatter",
    "gaussian_blur", "saturate",
]
SEVERITIES = [1, 2, 3, 4, 5]

BREEDS_MAKERS = {
    'living17':    make_living17,
    'entity13':    make_entity13,
    'entity30':    make_entity30,
    'nonliving26': make_nonliving26,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_breeds(data_dir: str, name: str, batch_size: int = 64):
    hierarchy_dir = os.path.join(data_dir, 'imagenet_class_hierarchy')
    ret           = BREEDS_MAKERS[name](hierarchy_dir, split='good')
    src_map       = get_label_mapping('custom_imagenet', ret[1][0])
    tgt_map       = get_label_mapping('custom_imagenet', ret[1][1])

    trainset = folder.ImageFolder(
        root=os.path.join(data_dir, 'imagenetv1/train/'),
        transform=BREEDS_TRANSFORM, label_mapping=src_map)
    targetset = folder.ImageFolder(
        root=os.path.join(data_dir, 'imagenetv1/train/'),
        transform=BREEDS_TRANSFORM, label_mapping=tgt_map)

    idx = np.arange(len(trainset))
    np.random.seed(42)
    np.random.shuffle(idx)
    train_idx, val_idx = idx[:-10000], idx[-10000:]

    train_subset = torch.utils.data.Subset(trainset, train_idx)
    val_subset   = torch.utils.data.Subset(trainset, val_idx)
    trainloader  = DataLoader(train_subset, batch_size=batch_size,
                               shuffle=True, num_workers=4)
    valloader    = DataLoader(val_subset,   batch_size=batch_size,
                               shuffle=False, num_workers=4)

    testsets, testloaders, testset_names = [], [], []

    def add(ds, name_):
        testsets.append(ds)
        testloaders.append(DataLoader(ds, batch_size=batch_size,
                                       shuffle=False, num_workers=4))
        testset_names.append(name_)

    add(val_subset,  'val_source')
    add(targetset,   'val_target')
    add(folder.ImageFolder(os.path.join(data_dir, 'imagenetv1/val/'),
                            transform=BREEDS_TRANSFORM,
                            label_mapping=src_map), 'imagenet_val_source')
    add(folder.ImageFolder(os.path.join(data_dir, 'imagenetv1/val/'),
                            transform=BREEDS_TRANSFORM,
                            label_mapping=tgt_map), 'imagenet_val_target')

    for split, lmap in [('source', src_map), ('target', tgt_map)]:
        for corr in IMAGENET_C_CORRUPTIONS:
            for sev in SEVERITIES:
                path = os.path.join(data_dir, f'imagenet-c/{corr}/{sev}')
                if os.path.isdir(path):
                    add(folder.ImageFolder(root=path, transform=BREEDS_TRANSFORM,
                                           label_mapping=lmap),
                        f'corr_{split}_{corr}_sev{sev}')

    num_classes = len(ret[1][0])
    return (train_subset, val_subset, trainloader, valloader,
            testsets, testloaders, testset_names, num_classes)


# ---------------------------------------------------------------------------
# Score collection
# ---------------------------------------------------------------------------

def collect_scores(model: BREEDSClassifier, dataset,
                   batch_size: int = 256, device: str = 'cuda'):
    model.eval()
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    sc_list, ft_list, lb_list = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            feats  = model.get_features(images.to(device))
            scores = model.linear2(F.relu(feats))
            sc_list.append(scores.cpu().numpy())
            ft_list.append(feats.cpu().numpy())
            lb_list.append(labels.numpy())
    return (np.concatenate(sc_list),
            np.concatenate(ft_list),
            np.concatenate(lb_list))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(f'results/{args.name}', exist_ok=True)
    device = args.device

    print(f"\n{'='*60}\nBREEDS: {args.name}\n{'='*60}")
    (train_subset, val_subset, trainloader, valloader,
     testsets, testloaders, testset_names, num_classes) = load_breeds(
        args.data_dir, args.name, args.batch_size)
    print(f"Source classes: {num_classes}  |  Test sets: {len(testsets)}")

    # ── Classifier ──────────────────────────────────────────────────────────
    clf_path = f'results/{args.name}/classifier.pth'
    model    = BREEDSClassifier(num_classes=num_classes,
                                feature_dim=640, pretrained=True).to(device)
    if os.path.exists(clf_path):
        model.load_state_dict(torch.load(clf_path, map_location=device))
        print(f"Classifier loaded from {clf_path}")
    else:
        train(model, trainloader, valloader,
              epochs=30, lr=0.01, device=device, save_path=clf_path)
    model.eval()
    print(f"Val accuracy: {evaluate_accuracy(model, valloader, device):.4f}")

    # ── Null pools and CNF ───────────────────────────────────────────────────
    train_scores, train_features, train_labels = collect_scores(
        model, train_subset, device=device)
    pool_score, _ = build_null_pools(train_scores, train_labels, num_classes)
    train_ds      = build_dataset_from_scores(train_scores, train_features,
                                               train_labels, pool_score)

    flow_path = f'results/{args.name}/flow.pth'
    flow = ScoreShiftFlowWrapper(num_classes=num_classes, n_flows=12,
                                  feature_dim=640, hidden_dim=256,
                                  encoder_dim=128).to(device)
    if os.path.exists(flow_path):
        flow.load_state_dict(torch.load(flow_path, map_location=device))
        print(f"Flow loaded from {flow_path}")
    else:
        flow.train_flow(train_ds, epochs=30, lr=3e-4, device=device)
        torch.save(flow.state_dict(), flow_path)
        print(f"Flow saved to {flow_path}")
    flow.eval()

    # Temperature calibration
    src_logits, _, src_labels = flow.generate_decoys(train_ds, device=device)
    temp = temperature_scale(torch.tensor(src_logits).to(device),
                              torch.tensor(src_labels).to(device))
    print(f"Temperature: {temp:.4f}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    all_results = []
    for idx, (testset, testloader, ds_name) in enumerate(
            zip(testsets, testloaders, testset_names)):
        print(f"\n  [{idx+1}/{len(testsets)}] {ds_name}")

        test_scores, test_features, test_labels = collect_scores(
            model, testset, device=device)
        test_ds = build_dataset_from_scores(test_scores, test_features,
                                             test_labels, pool_score)
        ms, ds_flow, ls = flow.generate_decoys(test_ds, device=device)
        n = len(ls)
        if n == 0:
            continue

        true_acc = float((ms.argmax(1) == ls).mean())
        pred  = ms.max(axis=1)
        decoy = ds_flow.max(axis=1)
        sort_idx = np.argsort(pred)

        q_mm  = compute_qvalues(pred, decoy, method='mixmax')
        q_tdc = compute_qvalues(pred, decoy, method='tdc')

        q_sorted  = q_mm[sort_idx]
        pi0_mm    = float(np.clip(q_sorted[0], 0, 1))
        acc_st_mm, acc_ta_mm, _ = estimate_accuracy(pred[sort_idx], q_sorted, pi0_mm)

        q_sorted_t = q_tdc[sort_idx]
        pi0_tdc    = float(np.clip(q_sorted_t[0], 0, 1))
        acc_st_tdc, acc_ta_tdc, _ = estimate_accuracy(pred[sort_idx], q_sorted_t, pi0_tdc)

        # Baselines
        tgt_t = torch.tensor(ms).to(device) / temp
        src_t = torch.tensor(src_logits).to(device) / temp
        src_l = torch.tensor(src_labels).to(device)
        row = dict(testset=ds_name, n=n, true_acc=true_acc,
                   acc_st_mm=acc_st_mm, acc_ta_mm=acc_ta_mm,
                   acc_st_tdc=acc_st_tdc, acc_ta_tdc=acc_ta_tdc,
                   err_st_mm=abs(acc_st_mm - true_acc),
                   err_ta_mm=abs(acc_ta_mm - true_acc))
        for mname, mfn in BASELINE_METHODS.items():
            try:
                est = float(np.clip(mfn(src_t, src_l, tgt_t), 0, 1))
            except Exception:
                est = float('nan')
            row[f'est_{mname}'] = est
            row[f'err_{mname}'] = abs(est - true_acc)
        all_results.append(row)

        print(f"    true={true_acc:.4f}  ENGPE-ST={acc_st_mm:.4f}"
              f"  ENGPE-TA={acc_ta_mm:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = f'results/{args.name}/results.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nResults saved to {csv_path}")
    print(f"\n{'='*40} SUMMARY {'='*40}")
    cols = ['err_st_mm', 'err_ta_mm'] + [f'err_{m}' for m in BASELINE_METHODS]
    print(df[cols].mean().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ENGPE on BREEDS')
    parser.add_argument('--name',       default='entity30',
                        choices=list(BREEDS_MAKERS))
    parser.add_argument('--data_dir',   required=True,
                        help='Path to ImageNet data root')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device',     default='cuda')
    main(parser.parse_args())
