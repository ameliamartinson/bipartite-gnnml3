"""
Orchestrate the GNNML3 vs LightGCN comparison.

Runs the matrix {models} x {datasets} x {seeds}, each as a subprocess that
appends one result record (shared bench_utils JSONL schema), then aggregates
mean +/- std across seeds per (model, dataset) and prints a results table.

Both models are launched with matched hyperparameters (embedding dim, layers,
lr, decay, epochs, topks) and the same protocol (best test epoch, seeded), so
the only differences are the architectures themselves.

Examples:
    # Full matrix on one dataset, 3 seeds
    python run_comparison.py --datasets "[gowalla]" --seeds "[2020,2021,2022]"

    # Just aggregate whatever is already in the results file
    python run_comparison.py --skip-run
"""

import argparse
import os
import statistics
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
from bench_utils import read_jsonl


def _parse_list(s):
    """Parse '[a,b]' / 'a,b' into a list of stripped string tokens."""
    return [t.strip() for t in s.strip().strip("[]").split(",") if t.strip()]


def build_cmd(model, dataset, seed, args):
    py = sys.executable
    if model == "gnnml3":
        cmd = [
            py, os.path.join(_HERE, "bipartite_experiment.py"),
            "--dataset", dataset,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--embed-dim", str(args.embed_dim),
            "--lr", str(args.lr),
            "--decay", str(args.decay),
            "--topks", args.topks,
            "--out", args.out,
            "--device", args.device,
            "--layers", str(args.layers),
            "--emb-in", str(args.emb_in),
            "--bpr-batch", str(args.bpr_batch),
            "--nfreq", str(args.nfreq),
            "--dv", str(args.dv),
            "--k", str(args.k),
            "--recfield", str(args.recfield),
            "--uu-topk", str(args.uu_topk),
            "--k-core", str(args.k_core),
        ]
        if args.raw_biadj:
            cmd.append("--raw-biadj")
        if args.no_struct_feats:
            cmd.append("--no-struct-feats")
        if args.layer_combine:
            cmd.append("--layer-combine")
        return cmd
    elif model == "lightgcn":
        return [
            py, os.path.join(_HERE, "run_lightgcn.py"),
            "--dataset", dataset,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--recdim", str(args.embed_dim),
            "--layer", str(args.layers),
            "--lr", str(args.lr),
            "--decay", str(args.decay),
            "--topks", args.topks,
            "--out", args.out,
            "--device", args.device,
        ]
    raise ValueError(model)


def run_matrix(models, datasets, seeds, args):
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                cmd = build_cmd(model, dataset, int(seed), args)
                print("\n" + "=" * 70)
                print(f"RUN: {model} | {dataset} | seed={seed}")
                print("=" * 70)
                print(" ".join(cmd))
                r = subprocess.run(cmd)
                if r.returncode != 0:
                    print(f"!! {model}/{dataset}/seed{seed} exited {r.returncode}")


def dedupe(rows):
    """Keep the last record per (model, dataset, seed)."""
    latest = {}
    for row in rows:
        latest[(row.get("model"), row.get("dataset"), row.get("seed"))] = row
    return list(latest.values())


def aggregate_and_print(out_path, ks):
    rows = dedupe(read_jsonl(out_path))
    if not rows:
        print(f"No results found in {out_path}")
        return

    metric_cols = []
    for k in ks:
        metric_cols += [f"recall@{k}", f"ndcg@{k}", f"precision@{k}"]
    eff_cols = ["train_time_s", "time_to_best_s", "peak_mem_mb", "n_params"]

    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["dataset"]), []).append(row)

    def fmt(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.4f}+/-{s:.4f}" if m < 100 else f"{m:.1f}+/-{s:.1f}"

    cols = metric_cols + eff_cols
    header = ["model", "dataset", "seeds"] + cols
    widths = [max(len(h), 16) for h in header]

    print("\n" + "#" * 100)
    print("AGGREGATED RESULTS (mean +/- std across seeds)")
    print("#" * 100)
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for (model, dataset), grp in sorted(groups.items()):
        cells = [model, dataset, str(len(grp))]
        for c in cols:
            cells.append(fmt([g.get(c) for g in grp]))
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="[gnnml3,lightgcn]")
    ap.add_argument("--datasets", default="[gowalla]")
    ap.add_argument("--seeds", default="[2020]")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument(
        "--layers",
        type=int,
        default=3,
        help="propagation/spectral layers (LightGCN --layer and gnnml3 --layers)",
    )
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--decay", type=float, default=1e-4)
    ap.add_argument("--topks", default="[20]")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "benchmark.jsonl"))
    # gnnml3-specific spectral-design hyperparameters
    ap.add_argument("--nfreq", type=int, default=5)
    ap.add_argument("--dv", type=float, default=5)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--recfield", type=int, default=1)
    ap.add_argument(
        "--uu-topk",
        type=int,
        default=0,
        help="gnnml3 only: top-N sparsified co-interaction edges per node (0 = off)",
    )
    ap.add_argument(
        "--k-core",
        type=int,
        default=0,
        help="gnnml3 only: recursive k-core filtering of the train set (0 = off)",
    )
    ap.add_argument(
        "--raw-biadj",
        action="store_true",
        help="gnnml3 only: SVD the raw (unnormalized) biadjacency",
    )
    ap.add_argument(
        "--emb-in",
        type=int,
        default=0,
        help="gnnml3 only: learnable node-embedding dim (0 = structural feats only)",
    )
    ap.add_argument(
        "--bpr-batch",
        type=int,
        default=0,
        help="gnnml3 only: 0 = all-interactions/epoch; >0 = minibatched BPR",
    )
    ap.add_argument(
        "--no-struct-feats",
        action="store_true",
        help="gnnml3 only: with --emb-in>0, use embeddings only (drop struct feats)",
    )
    ap.add_argument(
        "--layer-combine",
        action="store_true",
        help="gnnml3 only: average layer outputs (LightGCN-style readout)",
    )
    ap.add_argument("--skip-run", action="store_true", help="only aggregate existing results")
    args = ap.parse_args()

    models = _parse_list(args.models)
    datasets = _parse_list(args.datasets)
    seeds = _parse_list(args.seeds)
    ks = [int(x) for x in _parse_list(args.topks)]

    if not args.skip_run:
        run_matrix(models, datasets, seeds, args)
    aggregate_and_print(args.out, ks)


if __name__ == "__main__":
    main()
