"""
Standalone evaluation script.

Loads saved checkpoints and evaluates on the corresponding test fold,
printing per-fold accuracy and an overall classification report.

Usage:
    python eval.py                                  # evaluate all 5 folds
    python eval.py --folds 0 2                      # evaluate folds 0 and 2 only
    python eval.py --checkpoint_dir checkpoints     # custom checkpoint directory
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from data_loader import NGAFIDDataset
from model import CNNTransformerClassifier


@torch.no_grad()
def evaluate_fold(
    fold: int,
    dataset: NGAFIDDataset,
    checkpoint_dir: str,
    batch_size: int,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    ckpt_path = os.path.join(checkpoint_dir, f"fold_{fold}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]

    model = CNNTransformerClassifier(
        in_channels=23,
        num_classes=ckpt["num_classes"],
        d_model=saved_args["d_model"],
        nhead=saved_args["nhead"],
        num_encoder_layers=saved_args["num_encoder_layers"],
        dim_feedforward=saved_args["dim_feedforward"],
        dropout=saved_args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_X, test_y = dataset.get_fold_data(fold, training=False)
    test_ds = TensorDataset(
        torch.from_numpy(test_X), torch.from_numpy(test_y)
    )
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    acc = correct / total
    return acc, np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description="NGAFID CNN+Transformer Evaluation")
    parser.add_argument("--data_name", type=str, default="2days")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="Fold indices to evaluate (default: all available)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading NGAFID dataset …")
    dataset = NGAFIDDataset(name=args.data_name, destination=args.data_dir)
    label_map = dataset.label_map
    class_names = [str(label_map[i]) for i in range(dataset.num_classes)]

    folds = args.folds if args.folds is not None else list(range(5))
    fold_accs: list[float] = []
    all_preds_global, all_labels_global = [], []

    for fold in folds:
        print(f"\n--- Fold {fold + 1} ---")
        acc, preds, labels = evaluate_fold(
            fold, dataset, args.checkpoint_dir, args.batch_size, device
        )
        fold_accs.append(acc)
        all_preds_global.append(preds)
        all_labels_global.append(labels)
        print(f"  Accuracy: {acc:.4f}  ({int(acc * len(labels))}/{len(labels)})")

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for i, (f, acc) in enumerate(zip(folds, fold_accs)):
        print(f"  Fold {f + 1}: {acc:.4f}")
    mean, std = np.mean(fold_accs), np.std(fold_accs)
    print(f"  ────────────────────────────")
    print(f"  Mean:  {mean:.4f} ± {std:.4f}")

    all_preds_np = np.concatenate(all_preds_global)
    all_labels_np = np.concatenate(all_labels_global)
    print(f"\n  Classification Report (aggregated over evaluated folds):\n")
    print(
        classification_report(
            all_labels_np, all_preds_np, target_names=class_names, digits=4
        )
    )


if __name__ == "__main__":
    main()
