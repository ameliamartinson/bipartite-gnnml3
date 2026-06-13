"""
GNNML3 for bipartite recommendation datasets (Amazon-Book, Gowalla, Yelp2018).

Uses BipartiteSpectralDesign (SVD of normalized biadjacency with even/odd
spectral filters) + GNNML3 spectral convolutions for link prediction.

Usage:
    python bipartite_experiment.py --dataset amazon-book
    python bipartite_experiment.py --dataset gowalla
    python bipartite_experiment.py --dataset yelp2018
"""

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import sys

# Anchor imports/paths to this file's directory so the script works from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "gnn-matlang"))
sys.path.insert(0, _HERE)
from libs.spect_conv import SpectConv, ML3Layer
from bipartite_utils import BipartiteSpectralDesign
from eval_common import score
from bench_utils import append_jsonl


def set_seed(seed):
    """Seed every RNG that affects a run (init, sampling). The SVD start vector
    is seeded separately inside BipartiteSpectralDesign."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_device():
    """Auto-detect the best available device: CUDA > ROCm > MPS > CPU."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"CUDA ({name})"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple Silicon)"
    else:
        return torch.device("cpu"), "CPU"


class GNNML3LinkPredictor(nn.Module):
    """GNNML3 for bipartite link prediction."""

    def __init__(self, ninp, ne, num_users, nout1=64, nout2=32, embed_dim=64):
        super().__init__()
        self.num_users = num_users
        nin = nout1 + nout2
        self.conv1 = ML3Layer(
            learnedge=True,
            nedgeinput=ne,
            nedgeoutput=ne,
            ninp=ninp,
            nout1=nout1,
            nout2=nout2,
        )
        self.conv2 = ML3Layer(
            learnedge=True,
            nedgeinput=ne,
            nedgeoutput=ne,
            ninp=nin,
            nout1=nout1,
            nout2=nout2,
        )
        self.conv3 = ML3Layer(
            learnedge=True,
            nedgeinput=ne,
            nedgeoutput=ne,
            ninp=nin,
            nout1=nout1,
            nout2=nout2,
        )
        self.user_head = nn.Linear(nin, embed_dim)
        self.item_head = nn.Linear(nin, embed_dim)

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index2, data.edge_attr2
        x = self.conv1(x, ei, ea)
        x = self.conv2(x, ei, ea)
        x = self.conv3(x, ei, ea)
        return self.user_head(x[: self.num_users]), self.item_head(x[self.num_users :])


@torch.no_grad()
def evaluate(model, data, test_ui, train_ui, ks=(20,), batch_size=1024):
    """Full-ranking evaluation mirroring LightGCN's ``Procedure.Test``.

    Ranks all items per test user, excludes training interactions, takes the
    top-max(ks) items, and scores them with the shared ``eval_common`` module so
    the metric math is byte-identical to LightGCN. Only users with a non-empty
    held-out test set are scored (same as LightGCN keying on ``testDict``).

    Returns a dict like ``{"recall@20": ..., "precision@20": ..., "ndcg@20": ...}``.
    """
    model.eval()
    ue, ie = model(data)
    ue, ie = ue.cpu(), ie.cpu()
    nu = ue.shape[0]
    kmax = max(ks)
    ranked_topk = []
    ground_truth = []
    for s in range(0, nu, batch_size):
        e = min(s + batch_size, nu)
        rows = [(i, u) for i, u in enumerate(range(s, e)) if test_ui.get(u)]
        if not rows:
            continue
        scores = ue[s:e] @ ie.T
        for i, u in rows:
            if u in train_ui:
                scores[i, list(train_ui[u])] = -1e10
        local_idx = [i for i, _ in rows]
        _, tk = torch.topk(scores[local_idx], kmax, dim=1)
        for pos, (_, u) in enumerate(rows):
            ranked_topk.append(tk[pos].tolist())
            ground_truth.append(test_ui[u])
    return score(ranked_topk, ground_truth, ks=ks)


def load_edges(path):
    edges = []
    ui = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            u = int(p[0])
            ui[u] = set()
            for it in p[1:]:
                i = int(it)
                edges.append((u, i))
                ui[u].add(i)
    return edges, ui


def count_lines(p):
    with open(p) as f:
        return sum(1 for _ in f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", default="gowalla", choices=["amazon-book", "gowalla", "yelp2018"]
    )
    p.add_argument("--nfreq", type=int, default=5)
    p.add_argument("--dv", type=float, default=5)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--recfield", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--decay",
        type=float,
        default=1e-4,
        help="L2 weight decay (matched to LightGCN's --decay default)",
    )
    p.add_argument("--seed", type=int, default=2020, help="random seed (LightGCN uses 2020)")
    p.add_argument(
        "--topks",
        default="[20]",
        help="Python-literal list of cutoffs, e.g. '[20]' or '[20,50]'",
    )
    p.add_argument(
        "--eval-every", type=int, default=10, help="evaluate on test every N epochs"
    )
    p.add_argument(
        "--out",
        default=os.path.join(_HERE, "results", "benchmark.jsonl"),
        help="JSONL file to append the result record to",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto' (detect), 'cuda', 'cpu', or specific device name",
    )
    p.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision (GPU only)"
    )
    args = p.parse_args()

    ks = list(eval(args.topks))
    primary_k = ks[0]
    set_seed(args.seed)

    device, device_name = get_best_device()
    if args.device != "auto":
        device = torch.device(args.device)
        device_name = str(device)
    use_amp = args.amp and device.type == "cuda"
    print(f"Device: {device_name}" + (" (AMP enabled)" if use_amp else ""))
    print(f"Seed: {args.seed}  topks: {ks}")

    dd = os.path.join(_HERE, "datasets", args.dataset)
    nu = count_lines(f"{dd}/user_list.txt") - 1
    ni = count_lines(f"{dd}/item_list.txt") - 1
    print(f"Dataset: {args.dataset}")
    print(f"  Users: {nu:,}  Items: {ni:,}  Total: {nu+ni:,}")

    print("Loading edges...")
    tr_e, tr_ui = load_edges(f"{dd}/train.txt")
    te_e, te_ui = load_edges(f"{dd}/test.txt")
    print(f"  Train: {len(tr_e):,}  Test: {len(te_e):,}")

    print("Building graph...")
    nt = nu + ni
    el = []
    for u, i in tr_e:
        el.append([u, nu + i])
        el.append([nu + i, u])
    ei = torch.tensor(el, dtype=torch.int64).T
    x = torch.zeros(nt, 2)
    x[:nu, 0] = 1.0
    x[nu:, 1] = 1.0
    data = Data(edge_index=ei, x=x, y=torch.tensor([0]))

    print(f"Spectral design (nfreq={args.nfreq}, dv={args.dv}, k={args.k})...")
    t0 = time.time()
    tf = BipartiteSpectralDesign(
        nu,
        nfreq=args.nfreq,
        dv=args.dv,
        k=args.k,
        recfield=args.recfield,
        adddegree=True,
        nmax=0,
        seed=args.seed,
    )
    data = tf(data)
    setup_time_s = time.time() - t0
    print(f"  Done in {setup_time_s:.1f}s")
    print(
        f"  edge_index2: {data.edge_index2.shape}, edge_attr2: {data.edge_attr2.shape}"
    )

    ne = data.edge_attr2.shape[1]
    model = GNNML3LinkPredictor(data.x.shape[1], ne, nu, embed_dim=args.embed_dim)
    model = model.to(device)
    data = data.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = sum(1 for m in model.modules() if isinstance(m, ML3Layer))
    print(f"  Params: {n_params:,}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scaler = torch.amp.GradScaler() if use_amp else None

    # Full-batch training: the spectral convolution always processes the whole
    # graph, so we run it exactly once per epoch instead of once per user
    # minibatch. Users with no training interactions are dropped up front, and
    # their positive item tuples are cached for fast per-epoch sampling.
    valid_users = [u for u in sorted(tr_ui.keys()) if tr_ui[u]]
    pos_pool = [tuple(tr_ui[u]) for u in valid_users]
    pos_sets = [tr_ui[u] for u in valid_users]
    ul_tensor = torch.tensor(valid_users, dtype=torch.long, device=device)
    nuv = len(valid_users)
    print(f"\nTraining {args.epochs} epochs (full-batch, {nuv:,} users)...")

    # Mirror LightGCN: evaluate on test periodically and report the best epoch.
    best = None
    best_epoch = 0
    time_to_best_s = 0.0
    train_start = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()

        # One positive and one negative item per user for this epoch's BPR step.
        # Negatives are rejection-sampled so they are never items the user has
        # actually interacted with (no false negatives).
        pi_tensor = torch.tensor(
            [random.choice(p) for p in pos_pool], dtype=torch.long, device=device
        )
        neg = []
        for ps in pos_sets:
            j = random.randrange(ni)
            while j in ps:
                j = random.randrange(ni)
            neg.append(j)
        ni_tensor = torch.tensor(neg, dtype=torch.long, device=device)

        opt.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                ue, ie = model(data)
                pos_s = (ue[ul_tensor] * ie[pi_tensor]).sum(1)
                neg_s = (ue[ul_tensor] * ie[ni_tensor]).sum(1)
                loss = -F.logsigmoid(pos_s - neg_s).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            ue, ie = model(data)
            pos_s = (ue[ul_tensor] * ie[pi_tensor]).sum(1)
            neg_s = (ue[ul_tensor] * ie[ni_tensor]).sum(1)
            loss = -F.logsigmoid(pos_s - neg_s).mean()
            loss.backward()
            opt.step()

        if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
            rec = evaluate(model, data, te_ui, tr_ui, ks=ks)
            r_primary = rec[f"recall@{primary_k}"]
            print(
                f"  Epoch {ep:4d} | Loss: {loss.item():.4f} | "
                f"R@{primary_k}: {r_primary:.4f}  N@{primary_k}: {rec[f'ndcg@{primary_k}']:.4f}"
            )
            if best is None or r_primary > best[f"recall@{primary_k}"]:
                best = rec
                best_epoch = ep
                time_to_best_s = time.time() - train_start

    train_time_s = time.time() - train_start
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )

    print(f"\nBest epoch {best_epoch} (selected by recall@{primary_k}):")
    for k in ks:
        print(
            f"  Recall@{k}: {best[f'recall@{k}']:.4f}  "
            f"NDCG@{k}: {best[f'ndcg@{k}']:.4f}  "
            f"Precision@{k}: {best[f'precision@{k}']:.4f}"
        )

    row = {
        "model": "gnnml3",
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "embed_dim": args.embed_dim,
        "layers": n_layers,
        "lr": args.lr,
        "decay": args.decay,
        "device": device_name,
        "best_epoch": best_epoch,
        "setup_time_s": round(setup_time_s, 3),
        "train_time_s": round(train_time_s, 3),
        "time_to_best_s": round(time_to_best_s, 3),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "n_params": int(n_params),
        # model-specific spectral-design hyperparameters
        "nfreq": args.nfreq,
        "dv": args.dv,
        "k_svd": args.k,
        "recfield": args.recfield,
        "amp": bool(use_amp),
    }
    for k in ks:
        row[f"recall@{k}"] = round(best[f"recall@{k}"], 6)
        row[f"ndcg@{k}"] = round(best[f"ndcg@{k}"], 6)
        row[f"precision@{k}"] = round(best[f"precision@{k}"], 6)

    append_jsonl(args.out, row)
    print(f"\nResult appended to {args.out}")


if __name__ == "__main__":
    main()
