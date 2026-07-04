"""
Generate example recommendations from a trained checkpoint (GNNML3 or LightGCN).

Consumes the checkpoints written by ``bipartite_experiment.py --save-model`` or
``run_lightgcn.py --save-model``. Both models ultimately score a (user, item)
pair by the inner product of their final embeddings, so recommendation is
model-agnostic here: load the saved user/item embedding tables, rank all items
per user with the user's own training interactions excluded, and take the top-K.

For each example user the report shows:
  - the training interactions the model learned the user's taste from
    (the items the recommendations were "recommended from"),
  - the top-K recommended items, flagged when they appear in the held-out
    test set (i.e. the user really did interact with them later),
  - per recommendation, the user's history items whose embeddings are most
    similar to it ("because you interacted with ..." attribution).

Item/user ids are reported both as remap ids (row indices) and the original
dataset ids from user_list.txt / item_list.txt.

Usage:
    python recommend.py --checkpoint results/gnnml3_gowalla.pt
    python recommend.py --checkpoint results/lightgcn_gowalla.pt \
        --num-users 8 --topk 10 --users 41,1024
"""

import argparse
import os
import random

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_id_map(path):
    """item_list.txt / user_list.txt: header 'org_id remap_id', then one pair
    per line. Returns a list mapping remap id -> original id (as string)."""
    org = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            p = line.split()
            if len(p) >= 2:
                org[int(p[1])] = p[0]
    return [org[i] for i in range(len(org))]


def load_ui(path):
    """train.txt / test.txt: 'user item item ...' per line -> {user: [items]}."""
    ui = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            ui[int(p[0])] = [int(x) for x in p[1:]]
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help=".pt file from --save-model")
    ap.add_argument("--topk", type=int, default=10, help="recommendations per user")
    ap.add_argument(
        "--num-users", type=int, default=5, help="number of example users to sample"
    )
    ap.add_argument(
        "--users",
        default="",
        help="comma-separated remap user ids to use instead of sampling",
    )
    ap.add_argument(
        "--min-hist", type=int, default=5,
        help="sampled users must have at least this many training interactions",
    )
    ap.add_argument(
        "--max-hist", type=int, default=25,
        help="...and at most this many (keeps the history printable)",
    )
    ap.add_argument("--seed", type=int, default=2020, help="sampling seed")
    ap.add_argument(
        "--attr-top", type=int, default=3,
        help="history items shown as the source of each recommendation",
    )
    ap.add_argument(
        "--out", default="",
        help="also write the report to this text file "
        "(default results/recommendations_<model>_<dataset>.txt; '-' = don't write)",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name, dataset = ckpt["model"], ckpt["dataset"]
    ue, ie = ckpt["user_emb"].float(), ckpt["item_emb"].float()

    dd = os.path.join(_HERE, "datasets", dataset)
    user_org = load_id_map(os.path.join(dd, "user_list.txt"))
    item_org = load_id_map(os.path.join(dd, "item_list.txt"))
    train_ui = load_ui(os.path.join(dd, "train.txt"))
    test_ui = load_ui(os.path.join(dd, "test.txt"))

    assert ue.shape[0] == len(user_org), (
        f"checkpoint has {ue.shape[0]} users but {dataset} lists {len(user_org)}"
    )
    assert ie.shape[0] == len(item_org), (
        f"checkpoint has {ie.shape[0]} items but {dataset} lists {len(item_org)}"
    )

    if args.users:
        users = [int(u) for u in args.users.split(",")]
    else:
        rng = random.Random(args.seed)
        pool = [
            u for u, hist in train_ui.items()
            if args.min_hist <= len(hist) <= args.max_hist and test_ui.get(u)
        ]
        users = sorted(rng.sample(pool, min(args.num_users, len(pool))))

    # Normalized item embeddings for the cosine-similarity attribution.
    ie_n = torch.nn.functional.normalize(ie, dim=1)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    metrics = ckpt.get("metrics", {})
    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
    emit(f"Model: {model_name}  Dataset: {dataset}  (best epoch {ckpt['epoch']})")
    if metric_str:
        emit(f"Test metrics at this checkpoint: {metric_str}")
    emit(f"Top-{args.topk} recommendations for {len(users)} example users; "
         f"training interactions are excluded from the candidates.")

    total_hits = 0
    total_recs = 0
    for u in users:
        hist = train_ui.get(u, [])
        test_items = set(test_ui.get(u, []))
        emit()
        emit("=" * 78)
        emit(f"User remap_id={u}  (original id {user_org[u]})  — "
             f"{len(hist)} training interactions, {len(test_items)} held-out test items")
        emit()
        emit("  Recommended FROM (training history the model saw):")
        for i in hist:
            emit(f"    item {i:>6}  (org {item_org[i]})")

        scores = ie @ ue[u]
        scores[hist] = -float("inf")
        top_s, top_i = torch.topk(scores, args.topk)

        emit()
        emit(f"  Top-{args.topk} recommendations:")
        hits = 0
        for rank, (s, i) in enumerate(zip(top_s.tolist(), top_i.tolist()), 1):
            hit = i in test_items
            hits += hit
            mark = "  ✓ in held-out test set" if hit else ""
            emit(f"    {rank:2d}. item {i:>6}  (org {item_org[i]})  "
                 f"score {s:7.3f}{mark}")
            if hist:
                sims = ie_n[hist] @ ie_n[i]
                k = min(args.attr_top, len(hist))
                a_s, a_i = torch.topk(sims, k)
                src = ", ".join(
                    f"{item_org[hist[j]]} (sim {v:.2f})"
                    for v, j in zip(a_s.tolist(), a_i.tolist())
                )
                emit(f"        because of: {src}")
        total_hits += hits
        total_recs += args.topk
        emit(f"  -> {hits}/{args.topk} recommendations confirmed by this "
             f"user's held-out test interactions")

    emit()
    emit("=" * 78)
    emit(f"Overall: {total_hits}/{total_recs} recommended items appear in the "
         f"users' held-out test sets "
         f"(precision@{args.topk} = {total_hits / max(total_recs, 1):.3f} "
         f"on these example users)")

    if args.out != "-":
        out = args.out or os.path.join(
            _HERE, "results", f"recommendations_{model_name}_{dataset}.txt"
        )
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
