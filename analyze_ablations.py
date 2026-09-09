#!/usr/bin/env python3
"""Aggregate and interpret the GNNML3 link-prediction ablation matrix.

Reads the JSONL written by ``run_ablation_study.py`` (one record per run) and
produces:

  * ``ablation_summary.md`` -- the human-readable report (ranked tables,
    verdicts, per-seed deltas);
  * ``ablation_summary.csv`` -- one row per variant for thesis tables;
  * ``ablation_summary.json`` -- machine-readable summary;
  * optional plots (``--plots``): a ranked delta bar chart and training curves.

Metric convention: the *primary* metric is the fixed-budget ``final_*`` value,
i.e. the last evaluation of a run, which has no best-epoch selection bias. The
best-epoch metric is reported alongside because the main results table in the
thesis uses it.

Usage:
    python analyze_ablations.py --out-dir results/ablations
    python analyze_ablations.py --out-dir results/ablations --plots
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean, stdev

from bench_utils import read_jsonl

PRIMARY = "final_recall@20"
SECONDARY = "final_ndcg@20"
BEST_PRIMARY = "recall@20"


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def paired_deltas(rows_by_seed, base_by_seed, key):
    """Per-seed deltas (variant - baseline) on the seeds both ran."""
    seeds = sorted(set(rows_by_seed) & set(base_by_seed))
    out = {}
    for s in seeds:
        v, b = rows_by_seed[s].get(key), base_by_seed[s].get(key)
        if v is not None and b is not None:
            out[s] = v - b
    return out


def welch_or_paired_t(deltas):
    """Two-sided paired t-test p-value on the deltas (None if n < 2)."""
    vals = list(deltas.values())
    if len(vals) < 2:
        return None
    try:
        from scipy import stats
        if all(abs(v - vals[0]) < 1e-12 for v in vals):
            return 1.0 if abs(vals[0]) < 1e-12 else 0.0
        return float(stats.ttest_1samp(vals, 0.0).pvalue)
    except Exception:
        return None


def verdict_for(delta, delta_std, per_seed, noise):
    """Three-level verdict from effect size and per-seed sign consistency.

    * ``important``  -- mean |delta| clears 2x the seed-noise floor and every
      seed moves the same way (a component whose removal reliably changes the
      metric).
    * ``suggestive`` -- the mean clears the noise floor but the direction is not
      seed-consistent, or it sits between 1x and 2x the floor.
    * ``negligible`` -- mean |delta| below the seed-noise floor.
    """
    if delta is None:
        return "no data"
    vals = [v for v in per_seed.values() if v is not None]
    same_sign = bool(vals) and (all(v < 0 for v in vals) or all(v > 0 for v in vals))
    a = abs(delta)
    if a > 2 * noise and same_sign:
        return "**important**"
    if a > 2 * noise:
        return "**important (unstable)**"
    if a > noise:
        return "suggestive"
    return "not important"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "ablations"))
    ap.add_argument("--benchmark", default="", help="override benchmark JSONL path")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    bench = args.benchmark or os.path.join(out_dir, "benchmark.jsonl")
    rows = [r for r in read_jsonl(bench) if r.get("model") == "gnnml3"]
    if not rows:
        raise SystemExit(f"no gnnml3 rows in {bench}")

    # Non-learned floors (popularity / random) computed on the identical split,
    # if popularity_baseline.py has been run into the same directory.
    floors = {}
    for r in read_jsonl(os.path.join(out_dir, "baselines.jsonl")):
        if r.get("k_core") == rows[0].get("k_core") and PRIMARY.replace("final_", "") in r:
            floors[r["model"]] = r[PRIMARY.replace("final_", "")]

    # Group by ablation tag; keep only the study's protocol (dataset/k-core/
    # epochs) so stray exploratory runs cannot pollute the comparison.
    proto_fields = ("dataset", "k_core", "epochs")
    counts = defaultdict(int)
    for r in rows:
        counts[tuple(r.get(f) for f in proto_fields)] += 1
    proto = max(counts, key=counts.get)
    dropped = len(rows) - counts[proto]
    rows = [r for r in rows if tuple(r.get(f) for f in proto_fields) == proto]
    if dropped:
        print(f"[analyze] kept protocol {dict(zip(proto_fields, proto))} "
              f"({len(rows)} rows); dropped {dropped} rows from other "
              f"protocols -- use a separate --out-dir per dataset/k-core/epoch "
              f"setting for a clean comparison.")

    by_tag = defaultdict(dict)          # tag -> seed -> row
    meta = {}
    for r in rows:
        tag = r.get("ablation") or "untagged"
        by_tag[tag][r["seed"]] = r
        meta.setdefault(tag, r)

    base = by_tag.get(args.baseline, {})
    if not base:
        raise SystemExit(f"baseline tag {args.baseline!r} not found in {bench}")

    def stats(tag, key):
        vals = [r[key] for r in by_tag[tag].values() if key in r]
        if not vals:
            return None, None, 0
        return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0), len(vals)

    b_mean, b_std, b_n = stats(args.baseline, PRIMARY)
    b_best_mean, b_best_std, _ = stats(args.baseline, BEST_PRIMARY)
    # Noise floor: seed-to-seed spread of the reference model.
    noise = max(b_std, 1e-9)

    records = []
    for tag, seed_rows in by_tag.items():
        if tag == args.baseline:
            continue
        m, sd, n = stats(tag, PRIMARY)
        nm, nsd, _ = stats(tag, SECONDARY)
        bm, bsd, _ = stats(tag, BEST_PRIMARY)
        pd = paired_deltas(seed_rows, base, PRIMARY)
        pd_ndcg = paired_deltas(seed_rows, base, SECONDARY)
        pd_best = paired_deltas(seed_rows, base, BEST_PRIMARY)
        d_mean = mean(pd.values()) if pd else None
        d_std = stdev(pd.values()) if len(pd) > 1 else 0.0
        p = welch_or_paired_t(pd)
        rec = meta.get(tag, {})
        records.append(dict(
            tag=tag,
            group=next((v["group"] for v in _variants() if v["tag"] == tag), "?"),
            role=next((v["role"] for v in _variants() if v["tag"] == tag), "ablation"),
            note=next((v["note"] for v in _variants() if v["tag"] == tag), ""),
            seeds=n,
            final_recall=m, final_recall_std=sd,
            final_ndcg=nm, final_ndcg_std=nsd,
            best_recall=bm, best_recall_std=bsd,
            delta=d_mean, delta_std=d_std, delta_ndcg=(
                mean(pd_ndcg.values()) if pd_ndcg else None),
            delta_best=(mean(pd_best.values()) if pd_best else None),
            delta_pct=(100.0 * d_mean / b_mean) if (d_mean is not None and b_mean) else None,
            pvalue=p,
            per_seed_delta=pd,
            n_params=rec.get("n_params"),
            best_epoch=rec.get("best_epoch"),
            train_time_s=rec.get("train_time_s"),
        ))

    records.sort(key=lambda r: (r["delta"] if r["delta"] is not None else 0.0))

    lines = []
    A = lines.append
    A("# GNNML3 link-prediction ablation study")
    A("")
    A(f"- Model: `gnnml3` (bipartite spectral design + ML3 spectral layers + "
      f"BPR head), dataset `{meta[args.baseline].get('dataset')}`, "
      f"k-core `{meta[args.baseline].get('k_core')}`")
    A(f"- Protocol: {meta[args.baseline].get('epochs')} epochs full-batch BPR, "
      f"seeds {sorted(by_tag[args.baseline])}, full-ranking Recall@20 / NDCG@20 "
      f"(LightGCN-style, train items excluded)")
    A(f"- Primary metric: **{PRIMARY}** (fixed training budget, no best-epoch "
      f"selection); `{BEST_PRIMARY}` (best epoch) also reported for continuity "
      f"with the main results table")
    A(f"- Reference ({args.baseline}): {PRIMARY} = {fmt(b_mean)} ± {fmt(b_std)} "
      f"({b_n} seeds); best-epoch Recall@20 = {fmt(b_best_mean)} ± {fmt(b_best_std)}")
    A(f"- Seed noise floor (std of the reference model) = {fmt(b_std)}; "
      f"a variant is called *meaningful* when |paired delta| > 2× that floor "
      f"({fmt(2 * noise)})")
    if floors:
        A("- Non-learned floors on the same split: " + ", ".join(
            f"`{name}` Recall@20 = {fmt(v)}" for name, v in sorted(floors.items())))
    A("")
    A("## Ranked leave-one-out results")
    A("")
    A("| rank | variant | group | final Recall@20 | Δ Recall@20 | Δ% | Δ NDCG@20 | "
      "Δ best-ep R@20 | |Δ|/σ_seed | seeds | p (paired t) | verdict |")
    A("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate([x for x in records if x["role"] == "ablation"], 1):
        verdict = verdict_for(r["delta"], r["delta_std"], r["per_seed_delta"], noise)
        ratio = (abs(r["delta"]) / noise) if r["delta"] is not None else None
        A(f"| {i} | `{r['tag']}` | {r['group']} | {fmt(r['final_recall'])} ± "
          f"{fmt(r['final_recall_std'])} | {fmt(r['delta'])} | "
          f"{('%+.1f' % r['delta_pct']) if r['delta_pct'] is not None else 'n/a'} | "
          f"{fmt(r['delta_ndcg'])} | {fmt(r['delta_best'])} | "
          f"{fmt(ratio, 1)} | {r['seeds']} | "
          f"{('%g' % r['pvalue']) if r['pvalue'] is not None else 'n/a'} | "
          f"{verdict} |")
    A("")
    A("## Component groups")
    A("")
    A("| group | variants | mean Δ Recall@20 | worst Δ | best Δ | important |")
    A("|---|---|---:|---:|---:|---:|")
    groups = defaultdict(list)
    for r in records:
        if r["role"] == "ablation" and r["delta"] is not None:
            groups[r["group"]].append(r)
    for g, rs in sorted(groups.items(),
                        key=lambda kv: mean(x["delta"] for x in kv[1])):
        n_imp = sum(1 for x in rs
                    if verdict_for(x["delta"], x["delta_std"], x["per_seed_delta"],
                                   noise).startswith("**important"))
        A(f"| {g} | {len(rs)} | {fmt(mean(x['delta'] for x in rs))} | "
          f"{fmt(min(x['delta'] for x in rs))} | "
          f"{fmt(max(x['delta'] for x in rs))} | {n_imp}/{len(rs)} |")
    A("")
    A("## Addition probes (not leave-one-out)")
    A("")
    A("| variant | final Recall@20 | Δ vs baseline | Δ% | Δ NDCG@20 | seeds | verdict |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for r in [x for x in records if x["role"] != "ablation"]:
        verdict = verdict_for(r["delta"], r["delta_std"], r["per_seed_delta"], noise)
        verdict = ("helps" if (r["delta"] or 0) > 0 else "hurts") + (
            "" if verdict in ("**important**",) else f" ({verdict})")
        A(f"| `{r['tag']}` | {fmt(r['final_recall'])} ± {fmt(r['final_recall_std'])} | "
          f"{fmt(r['delta'])} | "
          f"{('%+.1f' % r['delta_pct']) if r['delta_pct'] is not None else 'n/a'} | "
          f"{fmt(r['delta_ndcg'])} | {r['seeds']} | {verdict} |")
    A("")
    A("## Per-seed paired deltas (final Recall@20)")
    A("")
    seeds_all = sorted(by_tag[args.baseline])
    A("| variant | " + " | ".join(f"seed {s}" for s in seeds_all) + " | mean | std |")
    A("|---|" + "---:|" * (len(seeds_all) + 2))
    for r in records:
        cells = []
        for s in seeds_all:
            d = r["per_seed_delta"].get(s)
            cells.append(fmt(d, 4) if d is not None else "—")
        A(f"| `{r['tag']}` | " + " | ".join(cells) +
          f" | {fmt(r['delta'])} | {fmt(r['delta_std'])} |")
    A("")
    A("## What each variant removes")
    A("")
    for r in records:
        A(f"- **`{r['tag']}`** ({r['group']}, {r['seeds']} seeds, "
          f"Δ Recall@20 {fmt(r['delta'])}): {r['note']}")
    A("")

    md = "\n".join(lines)
    print(md)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ablation_summary.md"), "w") as f:
        f.write(md + "\n")

    cols = ["tag", "group", "role", "seeds", "final_recall", "final_recall_std",
            "final_ndcg", "final_ndcg_std", "best_recall", "best_recall_std",
            "delta", "delta_std", "delta_ndcg", "delta_pct", "pvalue",
            "n_params", "best_epoch", "train_time_s", "note"]
    with open(os.path.join(out_dir, "ablation_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    with open(os.path.join(out_dir, "ablation_summary.json"), "w") as f:
        json.dump(dict(baseline=dict(mean=b_mean, std=b_std, seeds=b_n,
                                     best_mean=b_best_mean, best_std=b_best_std),
                       noise_floor=b_std, variants=records), f, indent=2)
    print(f"\nWrote ablation_summary.md / .csv / .json to {out_dir}")

    if args.plots:
        make_plots(out_dir, records, by_tag, args.baseline)


def _variants():
    """Import the variant table without importing the driver's heavy deps."""
    import run_ablation_study as ras
    return ras.VARIANTS


def make_plots(out_dir, records, by_tag, baseline):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(out_dir, ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. ranked delta bar chart
    abl = [r for r in records if r["role"] == "ablation" and r["delta"] is not None]
    abl.sort(key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(abl) + 2))
    ys = range(len(abl))
    ax.barh(list(ys), [r["delta"] for r in abl],
            xerr=[r["delta_std"] for r in abl], color="#3b6ea5",
            error_kw=dict(ecolor="#333", lw=1, capsize=2))
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["tag"] for r in abl])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ final Recall@20 vs baseline (paired, per-seed mean)")
    ax.set_title("GNNML3 component ablations — ranked by damage")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_delta_recall.png"), dpi=160)
    plt.close(fig)

    # 2. training curves
    hist_path = os.path.join(out_dir, "history.jsonl")
    hist = read_jsonl(hist_path)
    if hist:
        curves = defaultdict(lambda: defaultdict(list))
        for h in hist:
            if h.get("model") != "gnnml3":
                continue
            tag = h.get("ablation") or "untagged"
            curves[tag][h["seed"]].append((h["epoch"], h.get("recall@20")))
        fig, ax = plt.subplots(figsize=(9, 6))
        for tag in [baseline] + [r["tag"] for r in abl]:
            if tag not in curves:
                continue
            # mean over seeds, on the epoch grid of the first seed
            seeds = sorted(curves[tag])
            grid = [e for e, _ in curves[tag][seeds[0]]]
            series = []
            for s in seeds:
                d = dict(curves[tag][s])
                series.append([d.get(e, float("nan")) for e in grid])
            n = len(series)
            y = [sum(col) / n for col in zip(*series)]
            lw, alpha = (2.4, 1.0) if tag == baseline else (1.1, 0.75)
            ax.plot(grid, y, label=tag, lw=lw, alpha=alpha)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Recall@20 (test, best-so-far per eval)")
        ax.set_title("Ablation training curves (mean over seeds)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ablation_curves.png"), dpi=160)
        plt.close(fig)
        print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
