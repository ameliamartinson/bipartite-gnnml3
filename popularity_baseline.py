#!/usr/bin/env python3
"""Non-learned reference scores on the same split the ablations use.

The ablation numbers are only interpretable against a floor: how much of a
variant's Recall@20 is anything beyond "recommend the globally most popular
items"? This script computes the popularity and random-ranking references with
the identical protocol (same k-core filtering, same train-item exclusion, same
``eval_common.score`` math) so they can be quoted next to the learned models.

Usage:
    python popularity_baseline.py --dataset gowalla --k-core 20 --topks "[20]"
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnn-matlang"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bipartite_experiment import count_lines, load_edges  # noqa: E402
from bench_utils import append_jsonl  # noqa: E402
from eval_common import score  # noqa: E402
from kcore import (  # noqa: E402
    k_core_filter, load_k_core_cache, remap_k_core, save_k_core_cache,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def ranked_from_scores(item_scores, test_ui, train_ui, ni, kmax, rng=None):
    """Rank items by a fixed score, excluding each user's train items.

    Vectorized over items (a per-user Python loop over all 40k items would take
    minutes on the full Gowalla graph).
    """
    if rng is not None:
        item_scores = rng.random(ni)
    order = np.argsort(-np.asarray(item_scores))
    ranked, truth = [], []
    keep = np.ones(ni, dtype=bool)
    for u, gt in test_ui.items():
        if not gt:
            continue
        seen = train_ui.get(u)
        if seen:
            keep[:] = True
            keep[np.fromiter(seen, dtype=np.int64, count=len(seen))] = False
            top = order[keep[order]][:kmax]
        else:
            top = order[:kmax]
        ranked.append(top.tolist())
        truth.append(gt)
    return ranked, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gowalla")
    ap.add_argument("--k-core", type=int, default=20)
    ap.add_argument("--topks", default="[20]")
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "ablations",
                                                  "baselines.jsonl"))
    args = ap.parse_args()
    ks = eval(args.topks)

    dd = os.path.join(_HERE, "datasets", args.dataset)
    nu = count_lines(f"{dd}/user_list.txt") - 1
    ni = count_lines(f"{dd}/item_list.txt") - 1
    tr_e, _ = load_edges(f"{dd}/train.txt")
    te_e, _ = load_edges(f"{dd}/test.txt")

    if args.k_core > 0:
        train_path, test_path = f"{dd}/train.txt", f"{dd}/test.txt"
        cached = load_k_core_cache(train_path, test_path, args.k_core)
        if cached is not None:
            tr_e, te_e, nu, ni = cached
        else:
            tr_e = k_core_filter(tr_e, args.k_core)
            tr_e, te_e, nu, ni = remap_k_core(tr_e, te_e)
            save_k_core_cache(train_path, test_path, args.k_core, (tr_e, te_e, nu, ni))

    train_ui, test_ui = {}, {}
    deg = np.zeros(ni, dtype=np.float64)
    for u, i in tr_e:
        train_ui.setdefault(u, set()).add(i)
        deg[i] += 1
    for u, i in te_e:
        test_ui.setdefault(u, set()).add(i)

    kmax = max(ks)
    print(f"{args.dataset} k-core {args.k_core}: {nu} users, {ni} items, "
          f"{len(tr_e)} train / {len(te_e)} test interactions")

    rows = []
    for name, scores, rng in (
        ("popularity", deg, None),
        ("random", None, np.random.default_rng(args.seed)),
    ):
        ranked, truth = ranked_from_scores(scores, test_ui, train_ui, ni, kmax, rng)
        rec = score(ranked, truth, ks=ks)
        row = {"model": name, "dataset": args.dataset, "k_core": args.k_core,
               "seed": args.seed, "n_users_eval": len(truth),
               **{k: round(v, 6) for k, v in rec.items()}}
        rows.append(row)
        print(f"  {name:11s} " + "  ".join(
            f"{m}@{k}={rec[f'{m}@{k}']:.4f}" for k in ks
            for m in ("recall", "ndcg")))
    for row in rows:
        append_jsonl(args.out, row)
    print(f"Appended to {args.out}")


if __name__ == "__main__":
    main()
