"""
Print per-class sample counts for the NGAFID 2-days dataset.

Usage:
    python inspect_dataset.py
"""

from collections import Counter
from data_loader import NGAFIDDataset


def main():
    ds = NGAFIDDataset(name="2days", destination=".")
    header = ds.flight_header

    print(f"Total samples: {len(header)}\n")

    # ── before_after distribution ──
    ba = header["before_after"].value_counts().sort_index()
    print("before_after distribution")
    print("-" * 40)
    for val, cnt in ba.items():
        tag = "after maintenance" if val == 0 else "before maintenance"
        print(f"  {val}  ({tag:<20s})  {cnt:>5d}  ({cnt/len(header)*100:.1f}%)")

    # ── target_class distribution ──
    tc = header["target_class"].value_counts().sort_index()
    print(f"\ntarget_class distribution  ({len(tc)} classes)")
    print("-" * 55)
    print(f"  {'class':>5s}  {'mapped_idx':>10s}  {'count':>6s}  {'pct':>6s}")
    for orig, cnt in tc.items():
        idx = ds._label2idx[orig]
        print(f"  {orig:>5d}  {idx:>10d}  {cnt:>6d}  {cnt/len(header)*100:5.1f}%")

    # ── original 'class' column (maintenance issue type) ──
    mc = header["class"].value_counts().sort_index()
    print(f"\nclass (maintenance issue type) distribution  ({len(mc)} classes)")
    print("-" * 55)
    print(f"  {'class':>5s}  {'count':>6s}  {'pct':>6s}")
    for cls, cnt in mc.items():
        print(f"  {cls:>5d}  {cnt:>6d}  {cnt/len(header)*100:5.1f}%")

    # ── per-fold breakdown of target_class ──
    print(f"\nPer-fold target_class counts")
    print("-" * 70)
    folds = sorted(header["fold"].unique())
    classes = sorted(header["target_class"].unique())
    hdr = f"  {'class':>5s}" + "".join(f"  fold{f}" for f in folds) + "  total"
    print(hdr)
    for c in classes:
        row = f"  {c:>5d}"
        total = 0
        for f in folds:
            n = int(((header["target_class"] == c) & (header["fold"] == f)).sum())
            row += f"  {n:>5d}"
            total += n
        row += f"  {total:>5d}"
        print(row)


if __name__ == "__main__":
    main()
