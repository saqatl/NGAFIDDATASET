"""
5-Fold Cross-Validation training script for the CNN + Transformer classifier
on the NGAFID 2-days benchmark dataset.

Usage:
    python train.py                     # default settings
    python train.py --epochs 100        # override epochs
    python train.py --batch_size 32     # override batch size
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import NGAFIDDataset
from model import CNNTransformerClassifier


def print_dataset_info(dataset: NGAFIDDataset):
    """Print dataset structure, label distribution, and the first sample."""
    header = dataset.flight_header

    print(f"\n{'=' * 60}")
    print("  Dataset Structure Overview")
    print(f"{'=' * 60}")

    # flight_header schema
    print("\n  [flight_header.csv]")
    print(f"    Shape: {header.shape}  (rows × columns)")
    print(f"    Columns & dtypes:")
    for col in header.columns:
        print(f"      {col:<25s}  {header[col].dtype}")
    print(f"\n    First 3 rows:")
    print(header.head(3).to_string(max_colwidth=30).replace("\n", "\n    "))

    # fold distribution
    fold_counts = header["fold"].value_counts().sort_index()
    print(f"\n  [Fold distribution]")
    for fold_id, cnt in fold_counts.items():
        print(f"    Fold {fold_id}: {cnt} samples")

    # label distribution
    class_counts = header["target_class"].value_counts().sort_index()
    print(f"\n  [target_class distribution]  ({dataset.num_classes} unique classes)")
    print(f"    {'original_label':<16s} → {'mapped_idx':<10s}  count")
    for orig_label, cnt in class_counts.items():
        mapped = dataset._label2idx[orig_label]
        print(f"    {str(orig_label):<16s} → {mapped:<10d}  {cnt}")

    # normalization stats
    print(f"\n  [Normalization stats]  (per-channel min / max, 23 channels)")
    print(f"    mins : {dataset.mins[:5]} ... (showing first 5)")
    print(f"    maxs : {dataset.maxs[:5]} ... (showing first 5)")

    # first raw sample
    first = dataset._data_dict[0]
    raw_data = first["data"]
    print(f"\n{'=' * 60}")
    print("  First Sample (raw, before normalization)")
    print(f"{'=' * 60}")
    print(f"    target_class : {first['target_class']}")
    print(f"    fold         : {first['fold']}")
    print(f"    data shape   : {raw_data.shape}  (time_steps × channels)")
    print(f"    data dtype   : {raw_data.dtype}")

    nonzero_mask = np.any(raw_data != 0, axis=1)
    actual_len = int(nonzero_mask.sum())
    print(f"    actual length: {actual_len}  (non-zero rows out of {raw_data.shape[0]})")
    print(f"    padding rows : {raw_data.shape[0] - actual_len}")

    print(f"\n    data[0, :5]  (first timestep, first 5 channels):")
    print(f"      {raw_data[0, :5]}")
    first_nonzero = np.argmax(nonzero_mask) if actual_len > 0 else 0
    print(f"    data[{first_nonzero}, :5]  (first non-zero row, first 5 channels):")
    print(f"      {raw_data[first_nonzero, :5]}")
    print(f"    data[-1, :5] (last timestep, first 5 channels):")
    print(f"      {raw_data[-1, :5]}")

    # first normalized sample
    X_sample, y_sample = dataset.get_fold_data(first["fold"], training=False)
    print(f"\n{'=' * 60}")
    print("  First Sample (after min-max normalization)")
    print(f"{'=' * 60}")
    print(f"    X shape : {X_sample.shape}  (num_samples_in_fold × time_steps × channels)")
    print(f"    X dtype : {X_sample.dtype}")
    print(f"    y shape : {y_sample.shape}")
    print(f"    y dtype : {y_sample.dtype}")
    print(f"    y[0]    : {y_sample[0]}  (mapped label)")
    print(f"    X[0] value range: [{X_sample[0].min():.4f}, {X_sample[0].max():.4f}]")
    print(f"    X[0, 0, :5] (first timestep, first 5 channels):")
    print(f"      {X_sample[0, 0, :5]}")
    print(f"    X[0, -1, :5] (last timestep, first 5 channels):")
    print(f"      {X_sample[0, -1, :5]}")
    print()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


def run_fold(
    fold: int,
    dataset: NGAFIDDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    print(f"\n{'=' * 60}")
    print(f"  Fold {fold + 1} / {args.num_folds}")
    print(f"{'=' * 60}")

    train_X, train_y = dataset.get_fold_data(fold, training=True)
    test_X, test_y = dataset.get_fold_data(fold, training=False)

    print(
        f"  Train: {len(train_y)} samples  |  Test: {len(test_y)} samples  |"
        f"  Classes: {dataset.num_classes}"
    )

    train_ds = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_y)
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_X), torch.from_numpy(test_y)
    )
    # num_workers=0 for Windows compatibility
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    model = CNNTransformerClassifier(
        in_channels=23,
        num_classes=dataset.num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            best_acc = max(best_acc, test_acc)
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch + 1:3d}/{args.epochs}"
                f"  | Train loss {train_loss:.4f}  acc {train_acc:.4f}"
                f"  | Test  loss {test_loss:.4f}  acc {test_acc:.4f}"
                f"  | lr {lr:.2e}"
            )

    # Final evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    best_acc = max(best_acc, test_acc)
    print(f"\n  Fold {fold + 1} — final test accuracy: {test_acc:.4f}  (best: {best_acc:.4f})")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"fold_{fold}.pt")
    torch.save(
        {
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "num_classes": dataset.num_classes,
            "label_map": dataset.label_map,
            "test_accuracy": test_acc,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"  Checkpoint saved → {ckpt_path}")

    return test_acc


def main():
    parser = argparse.ArgumentParser(description="NGAFID CNN+Transformer 5-Fold CV")

    # data
    parser.add_argument("--data_name", type=str, default="2days")
    parser.add_argument("--data_dir", type=str, default=".")

    # model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # misc
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Print test metrics every N epochs")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    print("Loading NGAFID dataset …")
    dataset = NGAFIDDataset(name=args.data_name, destination=args.data_dir)
    print(f"  Samples: {len(dataset.flight_header)}  |  Classes: {dataset.num_classes}")
    print(f"  Label mapping: {dataset.label_map}")

    print_dataset_info(dataset)

    fold_results: list[float] = []
    t0 = time.time()

    for fold in range(args.num_folds):
        acc = run_fold(fold, dataset, args, device)
        fold_results.append(acc)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print("  5-Fold Cross-Validation Results")
    print(f"{'=' * 60}")
    for i, acc in enumerate(fold_results):
        print(f"  Fold {i + 1}: {acc:.4f}")
    mean, std = np.mean(fold_results), np.std(fold_results)
    print(f"  ────────────────────────────")
    print(f"  Mean:  {mean:.4f} ± {std:.4f}")
    print(f"  Total time: {elapsed / 60:.1f} min")

    summary_path = os.path.join(args.checkpoint_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "fold_accuracies": fold_results,
                "mean_accuracy": float(mean),
                "std_accuracy": float(std),
                "elapsed_seconds": elapsed,
                "args": vars(args),
            },
            f,
            indent=2,
        )
    print(f"  Results saved → {summary_path}")


if __name__ == "__main__":
    main()
