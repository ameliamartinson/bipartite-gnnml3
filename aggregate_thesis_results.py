#!/usr/bin/env python3
"""Aggregate thesis benchmark runs into a mean +/- std results table.

Groups records in a benchmark JSONL by (model, dataset, configuration) --
i.e. everything except the seed -- and reports Recall/NDCG@K as mean +/- std
over seeds, plus timing and parameter counts. Writes Markdown (for the
summary) and a CSV next to it (for pgfplots/pandas in the thesis).

Usage:
  python aggregate_thesis_results.py --jsonl results/thesis/benchmark.jsonl
"""

import argparse
import csv
import os
from collections import defaultdict
from statistics import mean, stdev

from bench_utils import read_jsonl

# Fields that define a configuration (missing fields are treated as "-").
CONFIG_FIELDS = [
    "model", "dataset", "emb_in", "layers", "layer_combine", "struct_feats",
    "bpr_batch", "nfreq", "k_svd", "uu_topk", "biadj", "lr", "decay", "epochs",
]


def fmt(m, s, n):
    return f"{m:.4f}" if n < 2 else f"{m:.4f} ± {s:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="", help="write Markdown here (and a .csv "
                    "sibling); default: print only")
    args = ap.parse_args()

    rk, nk = f"recall@{args.topk}", f"ndcg@{args.topk}"
    try:
        rows = [r for r in read_jsonl(args.jsonl) if rk in r]
    except FileNotFoundError:
        print(f"No results at {args.jsonl}")
        return

    groups = defaultdict(list)
    for r in rows:
        key = tuple(str(r.get(f, "-")) for f in CONFIG_FIELDS)
        groups[key].append(r)

    table = []
    for key, rs in groups.items():
        cfg = dict(zip(CONFIG_FIELDS, key))
        recalls = [r[rk] for r in rs]
        ndcgs = [r[nk] for r in rs]
        n = len(rs)
        table.append({
            **cfg,
            "seeds": n,
            "recall_mean": mean(recalls),
            "recall_std": stdev(recalls) if n > 1 else 0.0,
            "ndcg_mean": mean(ndcgs),
            "ndcg_std": stdev(ndcgs) if n > 1 else 0.0,
            "best_epoch": round(mean(r.get("best_epoch", 0) for r in rs)),
            "train_time_s": round(mean(r.get("train_time_s", 0) for r in rs)),
            "n_params": rs[0].get("n_params", "-"),
        })
    table.sort(key=lambda t: (t["dataset"], -t["recall_mean"]))

    show = ["model", "dataset", "emb_in", "layers", "bpr_batch", "nfreq",
            "k_svd", "uu_topk", "seeds"]
    lines = [f"# Thesis benchmark summary (@{args.topk})", ""]
    lines.append("| " + " | ".join(show + [f"Recall@{args.topk}",
                 f"NDCG@{args.topk}", "best_ep", "time_s", "params"]) + " |")
    lines.append("|" + "---|" * (len(show) + 5))
    for t in table:
        lines.append("| " + " | ".join(
            [str(t[c]) for c in show]
            + [fmt(t["recall_mean"], t["recall_std"], t["seeds"]),
               fmt(t["ndcg_mean"], t["ndcg_std"], t["seeds"]),
               str(t["best_epoch"]), str(t["train_time_s"]),
               str(t["n_params"])]) + " |")
    md = "\n".join(lines)
    print(md)

    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")
        cpath = os.path.splitext(args.out)[0] + ".csv"
        cols = CONFIG_FIELDS + ["seeds", "recall_mean", "recall_std",
                                "ndcg_mean", "ndcg_std", "best_epoch",
                                "train_time_s", "n_params"]
        with open(cpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for t in table:
                w.writerow({c: t.get(c, "") for c in cols})
        print(f"\nWrote {args.out} and {cpath}")


if __name__ == "__main__":
    main()
