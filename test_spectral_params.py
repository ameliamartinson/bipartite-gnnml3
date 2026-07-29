#!/usr/bin/env python3
"""Test suite for the GNNML3 spectral-design hyperparameters --k, --dv, --nfreq.

Runs a *coordinate* sweep around a baseline (one parameter varied at a time,
the other two held at their baseline values), then aggregates every run into
CSV files that are trivially importable from matplotlib/numpy/pandas:

    <out-dir>/runs.jsonl        raw per-run final metrics (from the experiment)
    <out-dir>/history.jsonl     raw per-epoch loss + metrics (from the experiment)
    <out-dir>/summary.csv       one row per (axis, value, seed) with final_loss,
                                recall_at_20, recall_at_50, ndcg_at_20, ndcg_at_50
    <out-dir>/aggregated.csv    summary.csv averaged over seeds (mean + std)
    <out-dir>/history.csv       per-(axis, value, seed, epoch) loss/metric curves

The baseline configuration appears on every axis at its baseline value, so each
unique configuration is only trained once and then emitted once per axis.

Examples:
    python test_spectral_params.py --dataset gowalla --epochs 300
    python test_spectral_params.py --k-values 50,100,200,400 \
        --dv-values 1,5,10,20 --nfreq-values 2,5,8 --seeds 2020,2021
    python test_spectral_params.py --aggregate-only      # re-emit the CSVs
    python plot_spectral_params.py                       # plot from the CSVs
"""

import argparse
import csv
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from statistics import mean, stdev

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_utils import read_jsonl

EXP = os.path.join(HERE, "bipartite_experiment.py")
PY = (
    os.path.join(HERE, ".venv", "bin", "python")
    if os.path.exists(os.path.join(HERE, ".venv", "bin", "python"))
    else sys.executable
)

AXES = ("k", "dv", "nfreq")
AXIS_FLAG = {"k": "--k", "dv": "--dv", "nfreq": "--nfreq"}

SUMMARY_COLS = [
    "dataset", "axis", "value", "k_svd", "dv", "nfreq", "seed", "epochs",
    "best_epoch", "final_loss", "recall_at_20", "recall_at_50",
    "ndcg_at_20", "ndcg_at_50", "train_time_s", "run_id",
]
AGG_COLS = [
    "dataset", "axis", "value", "n_seeds",
    "final_loss_mean", "final_loss_std",
    "recall_at_20_mean", "recall_at_20_std",
    "recall_at_50_mean", "recall_at_50_std",
    "ndcg_at_20_mean", "ndcg_at_20_std",
    "ndcg_at_50_mean", "ndcg_at_50_std",
]
HISTORY_COLS = [
    "dataset", "axis", "value", "k_svd", "dv", "nfreq", "seed", "epoch",
    "loss", "recall_at_20", "recall_at_50", "ndcg_at_20", "ndcg_at_50",
    "run_id",
]


def parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def parse_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def fmt_val(v):
    """Compact value rendering for run ids (100 -> '100', 0.5 -> '0.5')."""
    return f"{v:g}"


def build_configs(args):
    """Unique (k, dv, nfreq) configurations of the coordinate sweep, in order."""
    values = {
        "k": parse_ints(args.k_values),
        "dv": parse_floats(args.dv_values),
        "nfreq": parse_ints(args.nfreq_values),
    }
    baseline = {"k": args.baseline_k, "dv": args.baseline_dv, "nfreq": args.baseline_nfreq}
    configs = {}
    for axis in AXES:
        for v in values[axis]:
            cfg = dict(baseline)
            cfg[axis] = v
            key = (cfg["k"], cfg["dv"], cfg["nfreq"])
            configs.setdefault(key, cfg)
    return list(configs.values())


def run_id_for(cfg, seed):
    return f"k{fmt_val(cfg['k'])}_dv{fmt_val(cfg['dv'])}_nf{fmt_val(cfg['nfreq'])}_s{seed}"


def run_experiments(args, configs, out_dir):
    runs_jsonl = os.path.join(out_dir, "runs.jsonl")
    history_jsonl = os.path.join(out_dir, "history.jsonl")
    seeds = parse_ints(args.seeds)
    total = len(configs) * len(seeds)
    done = 0
    failed = []
    for cfg in configs:
        for seed in seeds:
            done += 1
            rid = run_id_for(cfg, seed)
            cmd = [
                PY, EXP,
                "--dataset", args.dataset,
                "--epochs", str(args.epochs),
                "--eval-every", str(args.eval_every),
                "--device", args.device,
                "--seed", str(seed),
                "--topks", "[20,50]",
                "--k", str(cfg["k"]),
                "--dv", str(cfg["dv"]),
                "--nfreq", str(cfg["nfreq"]),
                "--emb-in", str(args.emb_in),
                "--out", runs_jsonl,
                "--history-out", history_jsonl,
                "--run-id", rid,
            ]
            if args.layer_combine:
                cmd.append("--layer-combine")
            if args.extra:
                cmd += shlex.split(args.extra)
            print(f"\n[{done}/{total}] run_id={rid}")
            print("+ " + " ".join(cmd), flush=True)
            rc = subprocess.call(cmd, cwd=HERE)
            if rc != 0:
                print(f"!! FAILED (rc={rc}): {rid} (continuing)")
                failed.append(rid)
    return runs_jsonl, history_jsonl, failed


def axis_rows(cfg_fields, base, baseline):
    """Yield one output row per sweep axis this config legitimately belongs to.

    A config lies on axis A only when the other two parameters sit at their
    baseline values (coordinate-sweep semantics) — otherwise its metrics would
    pollute the baseline point of axes it was never meant to vary.
    """
    for axis in AXES:
        others = [a for a in AXES if a != axis]
        if all(cfg_fields[a] == baseline[a] for a in others):
            row = dict(base)
            row["axis"] = axis
            row["value"] = cfg_fields[axis]
            yield row


def aggregate(args, out_dir):
    """Join runs.jsonl + history.jsonl into the matplotlib-friendly CSVs."""
    runs_jsonl = os.path.join(out_dir, "runs.jsonl")
    history_jsonl = os.path.join(out_dir, "history.jsonl")
    runs = {}
    for r in read_jsonl(runs_jsonl):
        if r.get("model") == "gnnml3" and r.get("dataset") == args.dataset:
            runs[r.get("run_id", "")] = r
    history = [
        h for h in read_jsonl(history_jsonl)
        if h.get("model") == "gnnml3" and h.get("dataset") == args.dataset
    ]
    if not runs:
        print(f"No gnnml3 runs for dataset '{args.dataset}' in {runs_jsonl}")
        return 1

    baseline = {"k": args.baseline_k, "dv": args.baseline_dv,
                "nfreq": args.baseline_nfreq}

    # Final loss per run = loss recorded at the last evaluated epoch.
    final_loss = {}
    for h in history:
        rid = h.get("run_id", "")
        if rid not in final_loss or h["epoch"] >= final_loss[rid][0]:
            final_loss[rid] = (h["epoch"], h["loss"])

    summary_rows = []
    for rid, r in sorted(runs.items()):
        cfg_fields = {"k": r.get("k_svd"), "dv": r.get("dv"), "nfreq": r.get("nfreq")}
        base = {
            "dataset": r.get("dataset"),
            "k_svd": r.get("k_svd"),
            "dv": r.get("dv"),
            "nfreq": r.get("nfreq"),
            "seed": r.get("seed"),
            "epochs": r.get("epochs"),
            "best_epoch": r.get("best_epoch"),
            "final_loss": final_loss.get(rid, ("", ""))[1],
            "recall_at_20": r.get("recall@20"),
            "recall_at_50": r.get("recall@50"),
            "ndcg_at_20": r.get("ndcg@20"),
            "ndcg_at_50": r.get("ndcg@50"),
            "train_time_s": r.get("train_time_s"),
            "run_id": rid,
        }
        summary_rows.extend(axis_rows(cfg_fields, base, baseline))

    history_rows = []
    for h in sorted(history, key=lambda x: (x.get("run_id", ""), x.get("epoch", 0))):
        cfg_fields = {"k": h.get("k_svd"), "dv": h.get("dv"), "nfreq": h.get("nfreq")}
        base = {
            "dataset": h.get("dataset"),
            "k_svd": h.get("k_svd"),
            "dv": h.get("dv"),
            "nfreq": h.get("nfreq"),
            "seed": h.get("seed"),
            "epoch": h.get("epoch"),
            "loss": h.get("loss"),
            "recall_at_20": h.get("recall@20"),
            "recall_at_50": h.get("recall@50"),
            "ndcg_at_20": h.get("ndcg@20"),
            "ndcg_at_50": h.get("ndcg@50"),
            "run_id": h.get("run_id", ""),
        }
        history_rows.extend(axis_rows(cfg_fields, base, baseline))

    groups = defaultdict(list)
    for row in summary_rows:
        groups[(row["dataset"], row["axis"], row["value"])].append(row)

    def ms(vals):
        vals = [v for v in vals if v != "" and v is not None]
        if not vals:
            return "", ""
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        return round(m, 6), round(s, 6)

    agg_rows = []
    for (ds, axis, value), rs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], AXES.index(kv[0][1]), kv[0][2])
    ):
        row = {"dataset": ds, "axis": axis, "value": value, "n_seeds": len(rs)}
        for m in ["final_loss", "recall_at_20", "recall_at_50", "ndcg_at_20", "ndcg_at_50"]:
            m_mean, m_std = ms([r[m] for r in rs])
            row[f"{m}_mean"], row[f"{m}_std"] = m_mean, m_std
        agg_rows.append(row)

    def write_csv(name, cols, rows):
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path} ({len(rows)} rows)")

    write_csv("summary.csv", SUMMARY_COLS, summary_rows)
    write_csv("aggregated.csv", AGG_COLS, agg_rows)
    write_csv("history.csv", HISTORY_COLS, history_rows)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gowalla",
                    choices=["amazon-book", "gowalla", "yelp2018"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", default="2020", help="comma-separated seeds")
    ap.add_argument("--k-values", default="50,100,200",
                    help="comma-separated values for --k (SVD rank)")
    ap.add_argument("--dv-values", default="1,5,10",
                    help="comma-separated values for --dv (kernel bandwidth)")
    ap.add_argument("--nfreq-values", default="2,5,8",
                    help="comma-separated values for --nfreq (frequency bands)")
    ap.add_argument("--baseline-k", type=int, default=100)
    ap.add_argument("--baseline-dv", type=float, default=5)
    ap.add_argument("--baseline-nfreq", type=int, default=5)
    ap.add_argument("--emb-in", type=int, default=64,
                    help="learnable node-embedding dim for all runs (0 = off)")
    ap.add_argument("--no-layer-combine", dest="layer_combine",
                    action="store_false",
                    help="use last-layer readout instead of the layer mean")
    ap.add_argument("--extra", default="",
                    help="extra args forwarded verbatim to bipartite_experiment.py, "
                    "e.g. --extra '--layers 4 --lr 0.0005'")
    ap.add_argument("--out-dir", default="",
                    help="output directory (default: results/spectral_test/<dataset>)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; rebuild the CSVs from existing JSONL")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands that would run, then exit")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        HERE, "results", "spectral_test", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if args.aggregate_only:
        sys.exit(aggregate(args, out_dir))

    configs = build_configs(args)
    print(f"Dataset: {args.dataset}  epochs={args.epochs}  seeds={args.seeds}")
    print(f"Out dir: {out_dir}")
    print(f"Unique configurations to train ({len(configs)}):")
    for cfg in configs:
        print(f"  k={cfg['k']:g}  dv={cfg['dv']:g}  nfreq={cfg['nfreq']:g}")
    if args.dry_run:
        for cfg in configs:
            flags = " ".join(
                f"{AXIS_FLAG[a]} {fmt_val(cfg[a])}" for a in AXES)
            print(f"  python bipartite_experiment.py --dataset {args.dataset} "
                  f"--topks '[20,50]' {flags} ...")
        return

    _, _, failed = run_experiments(args, configs, out_dir)
    rc = aggregate(args, out_dir)

    print("\nPlot with:  python plot_spectral_params.py --dir " + out_dir)
    if failed:
        print(f"WARNING: {len(failed)} run(s) failed: {', '.join(failed)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
