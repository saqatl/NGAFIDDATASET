"""
Standalone evaluation script for the CNN + Transformer multi-task models.

Loads saved checkpoints and evaluates on the corresponding test fold(s),
printing per-fold accuracy and an overall classification report.

Usage:
    python eval.py                                  # evaluate all tasks, all folds
    python eval.py --tasks binary multiclass
    python eval.py --folds 0 2
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from data_loader import NGAFIDDataset
from model import CNNTransformerClassifier

ALL_TASKS = ["binary", "multiclass", "combined"]


@torch.no_grad()
def evaluate_fold(fold, task, dataset, checkpoint_dir, batch_size, device):
    ckpt_path = os.path.join(checkpoint_dir, task, f"fold_{fold}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt["args"]

    model = CNNTransformerClassifier(
        in_channels=23, num_classes=ckpt["num_classes"],
        d_model=sa["d_model"], nhead=sa["nhead"],
        num_encoder_layers=sa["num_encoder_layers"],
        dim_feedforward=sa["dim_feedforward"],
        dropout=sa["dropout"], task=task,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_X, test_yb, test_ym = dataset.get_fold_data(fold, training=False)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(test_X),
                       torch.from_numpy(test_yb),
                       torch.from_numpy(test_ym)),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    bin_preds, bin_labels = [], []
    multi_preds, multi_labels = [], []

    for X, yb, ym in loader:
        X = X.to(device)

        if task == "binary":
            logits = model(X)
            bin_preds.append(logits.argmax(1).cpu().numpy())
            bin_labels.append(yb.numpy())
        elif task == "multiclass":
            logits = model(X)
            multi_preds.append(logits.argmax(1).cpu().numpy())
            multi_labels.append(ym.numpy())
        else:
            bl, ml = model(X)
            bin_preds.append(bl.argmax(1).cpu().numpy())
            bin_labels.append(yb.numpy())
            multi_preds.append(ml.argmax(1).cpu().numpy())
            multi_labels.append(ym.numpy())

    result = {}
    if bin_preds:
        bp, bl = np.concatenate(bin_preds), np.concatenate(bin_labels)
        result["binary_acc"] = float((bp == bl).mean())
        result["binary_preds"] = bp
        result["binary_labels"] = bl
    if multi_preds:
        mp, ml = np.concatenate(multi_preds), np.concatenate(multi_labels)
        result["multi_acc"] = float((mp == ml).mean())
        result["multi_preds"] = mp
        result["multi_labels"] = ml

    return result


def main():
    parser = argparse.ArgumentParser(description="NGAFID CNN+Transformer Evaluation")
    parser.add_argument("--data_name", type=str, default="2days")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading NGAFID dataset …")
    dataset = NGAFIDDataset(name=args.data_name, destination=args.data_dir)
    label_map = dataset.label_map
    folds = args.folds if args.folds is not None else list(range(5))

    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"  Task: {task}")
        print(f"{'=' * 60}")

        all_bin_p, all_bin_l = [], []
        all_mul_p, all_mul_l = [], []
        fold_accs_bin, fold_accs_mul = [], []

        for fold in folds:
            try:
                res = evaluate_fold(fold, task, dataset, args.checkpoint_dir,
                                    args.batch_size, device)
            except FileNotFoundError as e:
                print(f"  {e} — skipping")
                continue

            parts = [f"Fold {fold+1}:"]
            if "binary_acc" in res:
                parts.append(f"Binary {res['binary_acc']:.4f}")
                fold_accs_bin.append(res["binary_acc"])
                all_bin_p.append(res["binary_preds"])
                all_bin_l.append(res["binary_labels"])
            if "multi_acc" in res:
                parts.append(f"Multiclass {res['multi_acc']:.4f}")
                fold_accs_mul.append(res["multi_acc"])
                all_mul_p.append(res["multi_preds"])
                all_mul_l.append(res["multi_labels"])
            print("  " + "  ".join(parts))

        if fold_accs_bin:
            m, s = np.mean(fold_accs_bin), np.std(fold_accs_bin)
            print(f"\n  Binary Acc mean: {m:.4f} ± {s:.4f}")
        if fold_accs_mul:
            m, s = np.mean(fold_accs_mul), np.std(fold_accs_mul)
            print(f"  Multiclass Acc mean: {m:.4f} ± {s:.4f}")

        if all_bin_p:
            bp, bl = np.concatenate(all_bin_p), np.concatenate(all_bin_l)
            print(f"\n  Binary classification report (aggregated):\n")
            print(classification_report(bl, bp,
                  target_names=["after (0)", "before (1)"], digits=4))

        if all_mul_p:
            mp, ml = np.concatenate(all_mul_p), np.concatenate(all_mul_l)
            names = [str(label_map[i]) for i in range(dataset.num_classes)]
            print(f"\n  Multiclass classification report (aggregated):\n")
            print(classification_report(ml, mp, target_names=names, digits=4))


if __name__ == "__main__":
    main()
