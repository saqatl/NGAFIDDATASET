"""
5-Fold CV training for 3 tasks on the NGAFID 2-days benchmark dataset.

Tasks (following the original paper):
  1. binary      — maintenance issue detection   (before=1 / after=0)
  2. multiclass  — maintenance issue classification (after->class 0, before->class 1-19)
  3. combined    — joint detection + classification (two-head model)

Optional optimisations (all OFF by default for baseline reproducibility):
  --weighted_loss   sqrt-inverse-frequency class weights + label smoothing
  --focal_loss      Focal Loss instead of CE (better for class imbalance)
  --oversample      WeightedRandomSampler to oversample rare classes
  --deep_head       deeper classification head (hidden layer + dropout)

Data augmentation (individually selectable, or --augment to enable all):
  --aug_noise       Gaussian noise injection
  --aug_scale       per-channel random scaling
  --aug_wslice      Window Slice (random crop + resize)
  --aug_timewarp    TimeWarp (smooth non-linear time distortion)
  --augment         convenience flag — enables all four above

Usage:
    python train.py                                            # baseline
    python train.py --augment                                  # all augmentations
    python train.py --aug_noise --aug_wslice                   # pick specific ones
    python train.py --weighted_loss --augment --deep_head      # all optimisations
    python train.py --epochs 100 --batch_size 32
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from data_loader import NGAFIDDataset
from model import CNNTransformerClassifier

ALL_TASKS = ["binary", "multiclass", "combined"]


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(
    y: np.ndarray, num_classes: int, device: torch.device,
    max_ratio: float = 5.0,
) -> torch.Tensor:
    """Sqrt-inverse-frequency weights, clamped so no class exceeds
    clamp: max_ratio times the median weight (prevents gradient explosion
    on extremely rare classes)"""
    counts = Counter(y.tolist())
    weights = torch.tensor(
        [1.0 / math.sqrt(max(counts.get(i, 1), 1)) for i in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    weights = weights.clamp(max=weights.median() * max_ratio)
    return weights / weights.sum() * num_classes


def build_sampler(task: str, train_yb: np.ndarray,
                  train_ym: np.ndarray) -> WeightedRandomSampler:
    """Oversample rare classes.

    For binary/multiclass tasks the label used is obvious; for the combined
    task we use the multiclass label because it has the most severe imbalance.
    """
    if task == "binary":
        labels = train_yb
    else:
        labels = train_ym

    counts = Counter(labels.tolist())
    sample_weights = np.array([1.0 / counts[int(l)] for l in labels],
                              dtype=np.float64)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for class-imbalanced classification.

    Down-weights well-classified examples so the model focuses on hard /
    misclassified samples — especially helpful for rare classes.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


def aug_noise(X: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """Add Gaussian noise to ~50 % of samples."""
    B = X.size(0)
    mask = (torch.rand(B, 1, 1, device=X.device) < 0.5).float()
    return X + torch.randn_like(X) * sigma * mask


def aug_scale(X: torch.Tensor, lo: float = 0.9, hi: float = 1.1) -> torch.Tensor:
    """Per-channel random scaling in [lo, hi] for ~50 % of samples."""
    B, _T, C = X.shape
    mask = (torch.rand(B, 1, 1, device=X.device) < 0.5).float()
    s = lo + torch.rand(B, 1, C, device=X.device) * (hi - lo)
    return X * (1.0 - mask + mask * s)


def aug_window_slice(X: torch.Tensor, min_ratio: float = 0.8) -> torch.Tensor:
    """Randomly crop a contiguous sub-window and resize back to original length.
    Same window for the whole batch (vectorised)."""
    """50% probability of applying the augmentation"""
    if random.random() > 0.5:
        return X
    B, T, C = X.shape
    win_len = random.randint(int(T * min_ratio), T)
    start = random.randint(0, T - win_len)
    sliced = X[:, start : start + win_len, :]          # (B, win_len, C)
    sliced = sliced.permute(0, 2, 1)                   # (B, C, win_len)
    resized = F.interpolate(sliced, size=T, mode="linear", align_corners=False)
    return resized.permute(0, 2, 1)                    # (B, T, C)


def aug_time_warp(
    X: torch.Tensor, sigma: float = 0.2, num_knots: int = 4
) -> torch.Tensor:
    """Smooth non-linear time distortion via random cubic-spline-like warping.
    Each sample gets an independent warp (vectorised with grid_sample)."""
    """50% probability of applying the augmentation"""
    if random.random() > 0.5:
        return X
    B, T, C = X.shape
    device = X.device

    # Uniform knot positions + random perturbation on interior knots
    knots = torch.linspace(0, T - 1, num_knots + 2, device=device)
    knots = knots.unsqueeze(0).expand(B, -1).clone()   # (B, K)
    spacing = (T - 1) / (num_knots + 1)
    knots[:, 1:-1] += torch.randn(B, num_knots, device=device) * sigma * spacing

    # Sort to guarantee monotonicity, clamp to valid range
    knots, _ = knots.sort(dim=1)
    knots = knots.clamp(0, T - 1)

    # Interpolate knots → full-length warp path  (B, T)
    warp = F.interpolate(
        knots.unsqueeze(1), size=T, mode="linear", align_corners=True
    ).squeeze(1)

    # Build grid for grid_sample:  warp ∈ [0, T-1] → normalise to [-1, 1]
    grid = (warp / (T - 1) * 2 - 1)                    # (B, T)
    grid = grid.view(B, 1, T, 1).expand(-1, -1, -1, 2).clone()
    grid[..., 1] = 0                                    # y-coordinate unused

    X_4d = X.permute(0, 2, 1).unsqueeze(2)             # (B, C, 1, T)
    warped = F.grid_sample(
        X_4d, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped.squeeze(2).permute(0, 2, 1)           # (B, T, C)


def augment_batch(
    X: torch.Tensor,
    do_noise: bool = False,
    do_scale: bool = False,
    do_wslice: bool = False,
    do_timewarp: bool = False,
) -> torch.Tensor:
    """Apply selected augmentations sequentially."""
    if do_noise:
        X = aug_noise(X)
    if do_scale:
        X = aug_scale(X)
    if do_wslice:
        X = aug_window_slice(X)
    if do_timewarp:
        X = aug_time_warp(X)
    return X


# ──────────────────────────────────────────────────────────────
#  Train / Evaluate one epoch
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, task, criterions, optimizer, scheduler,
                    device, aug_flags: dict):
    model.train()
    loss_sum, bin_ok, multi_ok, n = 0.0, 0, 0, 0

    for X, y_bin, y_multi in loader:
        X = X.to(device)
        if any(aug_flags.values()):
            X = augment_batch(X, **aug_flags)
        y_bin, y_multi = y_bin.to(device), y_multi.to(device)
        optimizer.zero_grad()

        if task == "binary":
            logits = model(X)
            loss = criterions["binary"](logits, y_bin)
            bin_ok += (logits.argmax(1) == y_bin).sum().item()
        elif task == "multiclass":
            logits = model(X)
            loss = criterions["multi"](logits, y_multi)
            multi_ok += (logits.argmax(1) == y_multi).sum().item()
        else:
            bl, ml = model(X)
            loss = criterions["binary"](bl, y_bin) + criterions["multi"](ml, y_multi)
            bin_ok += (bl.argmax(1) == y_bin).sum().item()
            multi_ok += (ml.argmax(1) == y_multi).sum().item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += loss.item() * X.size(0)
        n += X.size(0)

    r = {"loss": loss_sum / n}
    if task in ("binary", "combined"):
        r["binary_acc"] = bin_ok / n
    if task in ("multiclass", "combined"):
        r["multi_acc"] = multi_ok / n
    return r


@torch.no_grad()
def evaluate(model, loader, task, criterions, device):
    model.eval()
    loss_sum, n = 0.0, 0
    bin_preds, bin_labels = [], []
    multi_preds, multi_labels = [], []

    for X, y_bin, y_multi in loader:
        X, y_bin, y_multi = X.to(device), y_bin.to(device), y_multi.to(device)

        if task == "binary":
            logits = model(X)
            loss = criterions["binary"](logits, y_bin)
            bin_preds.append(logits.argmax(1).cpu().numpy())
            bin_labels.append(y_bin.cpu().numpy())
        elif task == "multiclass":
            logits = model(X)
            loss = criterions["multi"](logits, y_multi)
            multi_preds.append(logits.argmax(1).cpu().numpy())
            multi_labels.append(y_multi.cpu().numpy())
        else:
            bl, ml = model(X)
            loss = criterions["binary"](bl, y_bin) + criterions["multi"](ml, y_multi)
            bin_preds.append(bl.argmax(1).cpu().numpy())
            bin_labels.append(y_bin.cpu().numpy())
            multi_preds.append(ml.argmax(1).cpu().numpy())
            multi_labels.append(y_multi.cpu().numpy())

        loss_sum += loss.item() * X.size(0)
        n += X.size(0)

    r = {"loss": loss_sum / n}
    if bin_preds:
        bp = np.concatenate(bin_preds)
        bl = np.concatenate(bin_labels)
        r["binary_acc"] = float((bp == bl).mean())
        r["binary_f1"] = float(f1_score(bl, bp, average="macro", zero_division=0))
        r["binary_recall"] = float(recall_score(bl, bp, average="macro", zero_division=0))
    if multi_preds:
        mp = np.concatenate(multi_preds)
        ml = np.concatenate(multi_labels)
        r["multi_acc"] = float((mp == ml).mean())
        r["multi_f1"] = float(f1_score(ml, mp, average="macro", zero_division=0))
        r["multi_recall"] = float(recall_score(ml, mp, average="macro", zero_division=0))
    return r


# ──────────────────────────────────────────────────────────────
#  Build loss functions
# ──────────────────────────────────────────────────────────────

def build_criterions(task, train_yb, train_ym, num_classes, device,
                     weighted_loss: bool, focal_loss: bool = False,
                     focal_gamma: float = 2.0):
    criterions = {}

    if task in ("binary", "combined"):
        weight = compute_class_weights(train_yb, 2, device) if weighted_loss else None
        ls = 0.1 if weighted_loss else 0.0
        if focal_loss:
            criterions["binary"] = FocalLoss(alpha=weight, gamma=focal_gamma,
                                             label_smoothing=ls)
        else:
            kwargs = {}
            if weighted_loss:
                kwargs["weight"] = weight
                kwargs["label_smoothing"] = ls
            criterions["binary"] = nn.CrossEntropyLoss(**kwargs)

    if task in ("multiclass", "combined"):
        weight = compute_class_weights(train_ym, num_classes, device) if weighted_loss else None
        ls = 0.05 if weighted_loss else 0.0
        if focal_loss:
            criterions["multi"] = FocalLoss(alpha=weight, gamma=focal_gamma,
                                            label_smoothing=ls)
        else:
            kwargs = {}
            if weighted_loss:
                kwargs["weight"] = weight
                kwargs["label_smoothing"] = ls
            criterions["multi"] = nn.CrossEntropyLoss(**kwargs)

    return criterions


# ──────────────────────────────────────────────────────────────
#  Run one fold
# ──────────────────────────────────────────────────────────────

def run_fold(fold, task, dataset, args, device):
    print(f"\n  ── Fold {fold + 1}/{args.num_folds} ──")

    train_X, train_yb, train_ym = dataset.get_fold_data(fold, training=True)
    test_X, test_yb, test_ym = dataset.get_fold_data(fold, training=False)
    print(f"     Train {len(train_yb)}  |  Test {len(test_yb)}")

    train_ds = TensorDataset(torch.from_numpy(train_X),
                              torch.from_numpy(train_yb),
                              torch.from_numpy(train_ym))
    if args.oversample:
        sampler = build_sampler(task, train_yb, train_ym)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=0, pin_memory=device.type == "cuda",
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=device.type == "cuda",
        )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_X),
                       torch.from_numpy(test_yb),
                       torch.from_numpy(test_ym)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    model = CNNTransformerClassifier(
        in_channels=23, num_classes=dataset.num_classes,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, task=task,
        deep_head=args.deep_head, head_dropout=0.15,
    ).to(device)

    criterions = build_criterions(
        task, train_yb, train_ym, dataset.num_classes, device,
        weighted_loss=args.weighted_loss,
        focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs,
    )

    hist = {
        "train_loss": [], "test_loss": [], "test_epochs": [],
        "train_binary_acc": [], "train_multi_acc": [],
        "test_binary_acc": [], "test_multi_acc": [],
    }

    aug_flags = {
        "do_noise": args.aug_noise,
        "do_scale": args.aug_scale,
        "do_wslice": args.aug_wslice,
        "do_timewarp": args.aug_timewarp,
    }

    fold_start = time.time()
    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, task, criterions,
                             optimizer, scheduler, device,
                             aug_flags=aug_flags)
        hist["train_loss"].append(tr["loss"])
        hist["train_binary_acc"].append(tr.get("binary_acc"))
        hist["train_multi_acc"].append(tr.get("multi_acc"))

        do_eval = (epoch + 1) % args.log_interval == 0 or epoch == 0
        if do_eval:
            te = evaluate(model, test_loader, task, criterions, device)
            hist["test_loss"].append(te["loss"])
            hist["test_epochs"].append(epoch)
            hist["test_binary_acc"].append(te.get("binary_acc"))
            hist["test_multi_acc"].append(te.get("multi_acc"))

            elapsed = time.time() - fold_start
            eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
            eta_m, eta_s = divmod(int(eta), 60)

            parts = [f"Ep {epoch+1:3d}/{args.epochs}"]
            parts.append(f"TrL {tr['loss']:.4f}")
            if tr.get("binary_acc") is not None:
                parts.append(f"TrBin {tr['binary_acc']:.4f}")
            if tr.get("multi_acc") is not None:
                parts.append(f"TrMul {tr['multi_acc']:.4f}")
            parts.append(f"TeL {te['loss']:.4f}")
            if te.get("binary_acc") is not None:
                parts.append(f"TeBin {te['binary_acc']:.4f}")
            if te.get("multi_acc") is not None:
                parts.append(f"TeMul {te['multi_acc']:.4f}")
            parts.append(f"ETA {eta_m}m{eta_s:02d}s")
            print("     " + "  ".join(parts))

    # final eval
    te_final = evaluate(model, test_loader, task, criterions, device)
    if not hist["test_epochs"] or hist["test_epochs"][-1] != args.epochs - 1:
        hist["test_loss"].append(te_final["loss"])
        hist["test_epochs"].append(args.epochs - 1)
        hist["test_binary_acc"].append(te_final.get("binary_acc"))
        hist["test_multi_acc"].append(te_final.get("multi_acc"))

    # save checkpoint
    ckpt_dir = os.path.join(args.checkpoint_dir, task)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}.pt")
    torch.save({
        "fold": fold, "task": task,
        "model_state_dict": model.state_dict(),
        "num_classes": dataset.num_classes,
        "label_map": dataset.label_map,
        "args": vars(args),
    }, ckpt_path)

    return {
        "test_loss": te_final["loss"],
        "train_loss": hist["train_loss"][-1],
        "binary_acc": te_final.get("binary_acc"),
        "binary_f1": te_final.get("binary_f1"),
        "binary_recall": te_final.get("binary_recall"),
        "multi_acc": te_final.get("multi_acc"),
        "multi_f1": te_final.get("multi_f1"),
        "multi_recall": te_final.get("multi_recall"),
        "history": hist,
    }


# ──────────────────────────────────────────────────────────────
#  Run one task (all folds)
# ──────────────────────────────────────────────────────────────

def run_task(task, dataset, args, device):
    print(f"\n{'=' * 60}")
    print(f"  Task: {task}")
    print(f"{'=' * 60}")

    fold_results = []
    for fold in range(args.num_folds):
        res = run_fold(fold, task, dataset, args, device)
        fold_results.append(res)
        parts = [f"Fold {fold+1} done — TestLoss {res['test_loss']:.4f}"]
        if res["binary_acc"] is not None:
            parts.append(f"Bin Acc={res['binary_acc']:.4f} F1={res['binary_f1']:.4f} Recall={res['binary_recall']:.4f}")
        if res["multi_acc"] is not None:
            parts.append(f"Mul Acc={res['multi_acc']:.4f} F1={res['multi_f1']:.4f} Recall={res['multi_recall']:.4f}")
        print(f"     {'  '.join(parts)}")

    return {"fold_results": fold_results}


# ──────────────────────────────────────────────────────────────
#  Summary table  (similar to paper Table 2)
# ──────────────────────────────────────────────────────────────

def _fmt(values):
    """Format mean +/- std from a list of floats (or return '-' if empty)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "—"
    m, s = np.mean(vals), np.std(vals)
    return f"{m:.4f}±{s:.4f}"


def print_summary_table(all_results):
    print(f"\n{'=' * 120}")
    print("  Results Table  (mean ± std across 5 folds)")
    print(f"{'=' * 120}")
    header = (f"  {'Task':<12s}  {'Bin Acc':<14s}  {'Bin F1':<14s}  {'Bin Recall':<14s}"
              f"  {'Mul Acc':<14s}  {'Mul F1':<14s}  {'Mul Recall':<14s}"
              f"  {'Test Loss':<14s}")
    print(header)
    print(f"  {'─'*12}" + f"  {'─'*14}" * 7)

    for task in ALL_TASKS:
        if task not in all_results:
            continue
        folds = all_results[task]["fold_results"]
        ba  = _fmt([f["binary_acc"]    for f in folds])
        bf1 = _fmt([f["binary_f1"]     for f in folds])
        br  = _fmt([f["binary_recall"] for f in folds])
        ma  = _fmt([f["multi_acc"]     for f in folds])
        mf1 = _fmt([f["multi_f1"]      for f in folds])
        mr  = _fmt([f["multi_recall"]  for f in folds])
        tl  = _fmt([f["test_loss"]     for f in folds])
        print(f"  {task:<12s}  {ba:<14s}  {bf1:<14s}  {br:<14s}"
              f"  {ma:<14s}  {mf1:<14s}  {mr:<14s}  {tl:<14s}")

    print(f"{'=' * 120}")


# ──────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────

def plot_training_curves(all_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tasks = [t for t in ALL_TASKS if t in all_results]
    n_tasks = len(tasks)

    fig, axes = plt.subplots(n_tasks, 2, figsize=(14, 4.5 * n_tasks), squeeze=False)

    for row, task in enumerate(tasks):
        histories = [f["history"] for f in all_results[task]["fold_results"]]

        # ---- Loss ----
        ax = axes[row, 0]
        train_loss = np.array([h["train_loss"] for h in histories])
        epochs = np.arange(1, train_loss.shape[1] + 1)
        mu, sigma = train_loss.mean(0), train_loss.std(0)
        ax.plot(epochs, mu, label="Train", color="tab:blue")
        ax.fill_between(epochs, mu - sigma, mu + sigma, alpha=0.15, color="tab:blue")

        te_ep = np.array(histories[0]["test_epochs"]) + 1
        te_loss = np.array([h["test_loss"] for h in histories])
        mu, sigma = te_loss.mean(0), te_loss.std(0)
        ax.plot(te_ep, mu, label="Test", color="tab:orange", marker=".", ms=4)
        ax.fill_between(te_ep, mu - sigma, mu + sigma, alpha=0.15, color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{task} — Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- Accuracy ----
        ax = axes[row, 1]
        colors = iter(["tab:green", "tab:red"])

        if task in ("binary", "combined"):
            accs = np.array([h["test_binary_acc"] for h in histories], dtype=np.float64)
            mu, sigma = accs.mean(0), accs.std(0)
            lbl = "Binary Acc" if task == "combined" else "Accuracy"
            c = next(colors)
            ax.plot(te_ep, mu, label=lbl, color=c, marker=".", ms=4)
            ax.fill_between(te_ep, mu - sigma, mu + sigma, alpha=0.15, color=c)

        if task in ("multiclass", "combined"):
            accs = np.array([h["test_multi_acc"] for h in histories], dtype=np.float64)
            mu, sigma = accs.mean(0), accs.std(0)
            lbl = "Multiclass Acc" if task == "combined" else "Accuracy"
            c = next(colors)
            ax.plot(te_ep, mu, label=lbl, color=c, marker=".", ms=4)
            ax.fill_between(te_ep, mu - sigma, mu + sigma, alpha=0.15, color=c)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task} — Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Training curves saved → {path}")


# ──────────────────────────────────────────────────────────────
#  Dataset info printer
# ──────────────────────────────────────────────────────────────

def print_dataset_info(dataset: NGAFIDDataset):
    header = dataset.flight_header

    print(f"\n{'=' * 60}")
    print("  Dataset Structure Overview")
    print(f"{'=' * 60}")

    print(f"\n  [flight_header.csv]  shape: {header.shape}")
    print(f"    Columns: {list(header.columns)}")
    print(f"\n    First 3 rows:")
    print("    " + header.head(3).to_string(max_colwidth=30).replace("\n", "\n    "))

    fold_counts = header["fold"].value_counts().sort_index()
    print(f"\n  [Fold distribution]")
    for fid, cnt in fold_counts.items():
        print(f"    Fold {fid}: {cnt}")

    ba_counts = header["before_after"].value_counts().sort_index()
    print(f"\n  [before_after distribution]")
    for val, cnt in ba_counts.items():
        label = "after maintenance" if val == 0 else "before maintenance"
        print(f"    {val} ({label}): {cnt}")

    class_counts = header["target_class"].value_counts().sort_index()
    print(f"\n  [target_class distribution]  ({dataset.num_classes} classes)")
    for orig, cnt in class_counts.items():
        print(f"    {orig:>3d} -> idx {dataset._label2idx[orig]:<3d}  ({cnt} samples)")

    first = dataset._data_dict[0]
    raw = first["data"]
    nonzero = int(np.any(raw != 0, axis=1).sum())
    print(f"\n{'=' * 60}")
    print("  First sample (raw)")
    print(f"{'=' * 60}")
    print(f"    before_after : {first['before_after']}")
    print(f"    target_class : {first['target_class']}")
    print(f"    fold         : {first['fold']}")
    print(f"    data shape   : {raw.shape}  dtype={raw.dtype}")
    print(f"    actual length: {nonzero} / {raw.shape[0]}")
    print(f"    data[-1, :5] : {raw[-1, :5]}")

    X, yb, ym = dataset.get_fold_data(first["fold"], training=False)
    print(f"\n  First sample (normalised, from fold {first['fold']} test split)")
    print(f"    X shape : {X.shape}   dtype={X.dtype}")
    print(f"    y_binary[0]={yb[0]}   y_multi[0]={ym[0]}")
    print(f"    X[0] range: [{X[0].min():.4f}, {X[0].max():.4f}]")
    print()


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NGAFID CNN+Transformer — 3-task 5-Fold CV benchmark"
    )

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
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        choices=ALL_TASKS, help="Which tasks to train")

    # optimisations (all OFF by default)
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Enable sqrt-inverse-frequency class weights + label smoothing")
    parser.add_argument("--focal_loss", action="store_true",
                        help="Use Focal Loss instead of CrossEntropyLoss (better for imbalanced classes)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma (focusing parameter); higher = more focus on hard samples")
    parser.add_argument("--oversample", action="store_true",
                        help="Oversample rare classes via WeightedRandomSampler")
    parser.add_argument("--deep_head", action="store_true",
                        help="Enable deeper classification head (Linear-GELU-Dropout-Linear)")

    # data augmentation (individually selectable)
    parser.add_argument("--augment", action="store_true",
                        help="Convenience flag: enable ALL four augmentations below")
    parser.add_argument("--aug_noise", action="store_true",
                        help="Gaussian noise injection (sigma=0.02)")
    parser.add_argument("--aug_scale", action="store_true",
                        help="Per-channel random scaling [0.9, 1.1]")
    parser.add_argument("--aug_wslice", action="store_true",
                        help="Window Slice: random crop 80-100%% + resize")
    parser.add_argument("--aug_timewarp", action="store_true",
                        help="TimeWarp: smooth non-linear time distortion")

    # misc
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()

    # --augment is a convenience shortcut for all four augmentations
    if args.augment:
        args.aug_noise = True
        args.aug_scale = True
        args.aug_wslice = True
        args.aug_timewarp = True

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    any_aug = args.aug_noise or args.aug_scale or args.aug_wslice or args.aug_timewarp
    print(f"\nOptimisations:")
    print(f"  --weighted_loss : {'ON' if args.weighted_loss else 'OFF'}")
    print(f"  --focal_loss    : {'ON (gamma=' + str(args.focal_gamma) + ')' if args.focal_loss else 'OFF'}")
    print(f"  --oversample    : {'ON' if args.oversample else 'OFF'}")
    print(f"  --deep_head     : {'ON' if args.deep_head else 'OFF'}")
    print(f"  Augmentation    : {'OFF' if not any_aug else ''}")
    if any_aug:
        print(f"    --aug_noise   : {'ON' if args.aug_noise else 'OFF'}")
        print(f"    --aug_scale   : {'ON' if args.aug_scale else 'OFF'}")
        print(f"    --aug_wslice  : {'ON' if args.aug_wslice else 'OFF'}")
        print(f"    --aug_timewarp: {'ON' if args.aug_timewarp else 'OFF'}")

    print("\nLoading NGAFID dataset …")
    dataset = NGAFIDDataset(name=args.data_name, destination=args.data_dir)
    print(f"  Samples: {len(dataset.flight_header)}  |  Classes: {dataset.num_classes}")
    print_dataset_info(dataset)

    all_results = {}
    t0 = time.time()

    for task in args.tasks:
        task_result = run_task(task, dataset, args, device)
        all_results[task] = task_result

    elapsed = time.time() - t0

    # ── Summary table ──
    print_summary_table(all_results)
    print(f"  Total time: {elapsed / 60:.1f} min")

    # ── Plot curves ──
    plot_dir = os.path.join(args.checkpoint_dir, "plots")
    plot_training_curves(all_results, plot_dir)

    # ── Save JSON ──
    summary = {"elapsed_seconds": elapsed, "args": vars(args), "tasks": {}}
    for task, tres in all_results.items():
        folds = tres["fold_results"]
        entry = {}

        for key, prefix in [("binary_acc", "binary_acc"),
                             ("binary_f1", "binary_f1"),
                             ("binary_recall", "binary_recall"),
                             ("multi_acc", "multi_acc"),
                             ("multi_f1", "multi_f1"),
                             ("multi_recall", "multi_recall")]:
            vals = [f[key] for f in folds if f.get(key) is not None]
            if vals:
                entry[f"{prefix}_mean"] = float(np.mean(vals))
                entry[f"{prefix}_std"] = float(np.std(vals))
                entry[f"{prefix}_per_fold"] = vals

        tl = [f["test_loss"] for f in folds]
        trl = [f["train_loss"] for f in folds]
        entry["test_loss_mean"] = float(np.mean(tl))
        entry["test_loss_std"] = float(np.std(tl))
        entry["train_loss_mean"] = float(np.mean(trl))
        entry["train_loss_std"] = float(np.std(trl))
        summary["tasks"][task] = entry

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    results_path = os.path.join(args.checkpoint_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {results_path}")


if __name__ == "__main__":
    main()
