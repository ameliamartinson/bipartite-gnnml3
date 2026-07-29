#!/usr/bin/env python3
"""Plot the spectral-parameter test suite results produced by test_spectral_params.py.

Reads aggregated.csv and history.csv (plain CSVs — also directly importable via
``np.genfromtxt(path, delimiter=',', names=True)`` or ``pandas.read_csv``) and
writes, for every varied axis (k, dv, nfreq):

    <dir>/plot_<axis>_metrics.png   final loss, R@20, R@50, N@20 vs. the
                                    parameter value (mean +/- std over seeds)
    <dir>/plot_<axis>_curves.png    loss, R@20, R@50, N@20 vs. training epoch,
                                    one curve per parameter value (mean over seeds)

Usage:
    python plot_spectral_params.py --dir results/spectral_test/gowalla
    python plot_spectral_params.py --dir results/spectral_test/gowalla --show
"""

import argparse
import csv
import os
from collections import defaultdict
from statistics import mean

import matplotlib

matplotlib.use("Agg")  # headless by default; --show re-enables a backend below
import matplotlib.pyplot as plt

AXES = ("k", "dv", "nfreq")
AXIS_LABEL = {"k": "SVD rank k", "dv": "kernel bandwidth dv", "nfreq": "frequency bands nfreq"}

# (column suffix, y-label) for the four quantities of interest.
METRICS = [
    ("final_loss", "BPR loss"),
    ("recall_at_20", "Recall@20"),
    ("recall_at_50", "Recall@50"),
    ("ndcg_at_20", "NDCG@20"),
]
# history.csv columns for the training curves (loss has no 'final_' prefix).
CURVE_COLS = ["loss", "recall_at_20", "recall_at_50", "ndcg_at_20"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def by_axis(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["axis"]].append(r)
    return out


def plot_metrics(axis, rows, out_dir, dataset):
    """2x2 panel: final loss / R@20 / R@50 / N@20 vs. parameter value."""
    rows = sorted(rows, key=lambda r: float(r["value"]))
    x = [float(r["value"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (col, label) in zip(axes.flat, METRICS):
        y = [float(r[f"{col}_mean"]) for r in rows]
        s = [float(r[f"{col}_std"]) for r in rows]
        ax.errorbar(x, y, yerr=s, marker="o", capsize=4)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(AXIS_LABEL[axis])
    fig.suptitle(f"GNNML3 on {dataset}: metrics vs. {AXIS_LABEL[axis]}"
                 " (mean ± std over seeds)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"plot_{axis}_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_curves(axis, rows, out_dir, dataset):
    """2x2 panel: loss / R@20 / R@50 / N@20 vs. epoch, one curve per value."""
    # (value, epoch) -> list of per-seed measurements, per metric column.
    buckets = {c: defaultdict(list) for c in CURVE_COLS}
    for r in rows:
        key = (float(r["value"]), int(r["epoch"]))
        for c in CURVE_COLS:
            if r[c] not in ("", None):
                buckets[c][key].append(float(r[c]))

    values = sorted({k[0] for k in buckets["loss"]})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, col, (_, label) in zip(axes.flat, CURVE_COLS, METRICS):
        for v in values:
            pts = sorted((ep, mean(vals)) for (val, ep), vals
                         in buckets[col].items() if val == v)
            if not pts:
                continue
            eps, ys = zip(*pts)
            ax.plot(eps, ys, marker=".", ms=4, label=f"{axis}={v:g}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    fig.suptitle(f"GNNML3 on {dataset}: training curves vs. {AXIS_LABEL[axis]}"
                 " (mean over seeds)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"plot_{axis}_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="test-suite output dir containing aggregated.csv "
                    "and history.csv, e.g. results/spectral_test/gowalla")
    ap.add_argument("--show", action="store_true",
                    help="open interactive windows instead of only saving PNGs")
    args = ap.parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)

    agg = by_axis(read_csv(os.path.join(args.dir, "aggregated.csv")))
    hist = by_axis(read_csv(os.path.join(args.dir, "history.csv")))
    dataset = next(iter(agg.values()))[0]["dataset"] if agg else "?"

    for axis in AXES:
        if axis in agg:
            plot_metrics(axis, agg[axis], args.dir, dataset)
        if axis in hist:
            plot_curves(axis, hist[axis], args.dir, dataset)


if __name__ == "__main__":
    main()
