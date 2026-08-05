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
from kcore import k_core_filter, remap_k_core


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
    """GNNML3 for bipartite link prediction.

    Optionally prepends a learnable per-node embedding table (collaborative-
    filtering style, like LightGCN/MF) to the spectral-convolution input, which
    gives the model the per-node capacity that pure structural features lack.
    Supports a configurable number of ML3 layers and an optional LightGCN-style
    averaging of layer outputs (jumping-knowledge mean) to curb over-smoothing.

    Args:
        ninp: width of the structural input features in ``data.x``.
        ne: number of edge supports (nfreq + 1).
        num_users: number of user nodes (rows 0..num_users-1).
        num_nodes: total nodes (users + items); needed for the embedding table.
        emb_in: learnable node-embedding dim. 0 disables embeddings (the model
                then runs on structural features only, the original behavior).
        n_layers: number of stacked ML3 spectral layers.
        use_struct_feats: when embeddings are on, also concatenate the structural
                features; ignored (forced True) when emb_in == 0.
        layer_combine: average all layer outputs instead of using just the last.
    """

    def __init__(self, ninp, ne, num_users, num_nodes=0, emb_in=0, n_layers=3,
                 nout1=64, nout2=32, embed_dim=64, use_struct_feats=True,
                 layer_combine=False):
        super().__init__()
        self.num_users = num_users
        self.emb_in = emb_in
        self.use_struct_feats = use_struct_feats or emb_in == 0
        self.layer_combine = layer_combine

        if emb_in > 0:
            self.node_emb = nn.Embedding(num_nodes, emb_in)
            nn.init.normal_(self.node_emb.weight, std=0.1)
        else:
            self.node_emb = None

        conv_in = (emb_in if emb_in > 0 else 0) + (ninp if self.use_struct_feats else 0)
        nin = nout1 + nout2
        self.convs = nn.ModuleList(
            ML3Layer(
                learnedge=True,
                nedgeinput=ne,
                nedgeoutput=ne,
                ninp=conv_in if li == 0 else nin,
                nout1=nout1,
                nout2=nout2,
            )
            for li in range(n_layers)
        )
        self.user_head = nn.Linear(nin, embed_dim)
        self.item_head = nn.Linear(nin, embed_dim)

    def forward(self, data):
        ei, ea = data.edge_index2, data.edge_attr2
        if self.node_emb is not None:
            x = self.node_emb.weight
            if self.use_struct_feats:
                x = torch.cat([x, data.x], dim=1)
        else:
            x = data.x
        outs = []
        for conv in self.convs:
            x = conv(x, ei, ea)
            outs.append(x)
        h = torch.stack(outs, dim=0).mean(0) if self.layer_combine else outs[-1]
        return self.user_head(h[: self.num_users]), self.item_head(h[self.num_users :])


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
    p.add_argument(
        "--k-core",
        type=int,
        default=0,
        help="keep only users/items with at least K train interactions, "
        "applied recursively (standard k-core dataset filtering; 0 = off). "
        "Test interactions involving dropped users/items are removed too",
    )
    p.add_argument("--nfreq", type=int, default=5)
    p.add_argument("--dv", type=float, default=5)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--recfield", type=int, default=1)
    p.add_argument(
        "--raw-biadj",
        action="store_true",
        help="SVD the raw binary biadjacency instead of the symmetrically "
        "normalized one (D_u^-1/2 B D_v^-1/2)",
    )
    p.add_argument(
        "--uu-topk",
        type=int,
        default=0,
        help="add sparsified user-user/item-item co-interaction edges to the "
        "receptive field: keep the N strongest co-occurrence neighbors per "
        "node (0 = off). Gives the even spectral filters real within-partition "
        "edges without the dense 2-hop memory blow-up",
    )
    p.add_argument(
        "--off-diag",
        action="store_true",
        help="add the first off-diagonal band (i, i+/-1) in the UU and VV "
        "blocks of the receptive-field mask, giving the even spectral filters "
        "within-partition edges between consecutive user/item ids",
    )
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument(
        "--emb-in",
        type=int,
        default=0,
        help="learnable node-embedding dim fed to the convs (0 = off, use only "
        "structural indicator/degree features; >0 enables CF-style embeddings)",
    )
    p.add_argument(
        "--layers", type=int, default=3, help="number of stacked ML3 spectral layers"
    )
    p.add_argument(
        "--no-struct-feats",
        action="store_true",
        help="when --emb-in>0, drop the structural features and feed the convs "
        "only the learnable embeddings",
    )
    p.add_argument(
        "--layer-combine",
        action="store_true",
        help="average all layer outputs (LightGCN-style jumping-knowledge mean) "
        "instead of using only the last layer's output",
    )
    p.add_argument(
        "--bpr-batch",
        type=int,
        default=0,
        help="0 = one full-graph forward per epoch with a BPR loss over ALL "
        "training interactions (cheap, recommended); >0 = LightGCN-style "
        "minibatched BPR with a full-graph forward per minibatch of this size "
        "(many more gradient steps, but much slower for GNNML3)",
    )
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
        "--history-out",
        default="",
        help="JSONL file to append per-evaluation epoch history records "
        "(epoch, loss, metrics) to; empty = don't record history",
    )
    p.add_argument(
        "--run-id",
        default="",
        help="identifier embedded in the result and history records so a "
        "sweep/test harness can join them",
    )
    p.add_argument(
        "--save-model",
        default="",
        help="path to save a checkpoint of the best model (state_dict + final "
        "user/item embeddings, consumable by recommend.py); empty = don't save",
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

    if args.k_core > 0:
        print(f"Applying {args.k_core}-core filtering on train interactions...")
        tr_e = k_core_filter(tr_e, args.k_core)
        if not tr_e:
            raise SystemExit(
                f"error: the {args.k_core}-core of {args.dataset} train set is empty"
            )
        # Remap surviving users/items to contiguous ids and apply the same
        # mapping to the test set, dropping test interactions that involve
        # filtered-out nodes. Shared with the LightGCN runner via kcore.py so
        # both models see the exact same filtered graph.
        tr_e, te_e, nu, ni = remap_k_core(tr_e, te_e)
        tr_ui = {}
        for u, i in tr_e:
            tr_ui.setdefault(u, set()).add(i)
        te_ui = {}
        for u, i in te_e:
            te_ui.setdefault(u, set()).add(i)
        print(
            f"  {args.k_core}-core: {nu:,} users, {ni:,} items, "
            f"{len(tr_e):,} train / {len(te_e):,} test interactions"
        )

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

    biadj_kind = "raw" if args.raw_biadj else "normalized"
    print(
        f"Spectral design (nfreq={args.nfreq}, dv={args.dv}, k={args.k}, "
        f"biadj={biadj_kind}, uu_topk={args.uu_topk}, "
        f"off_diag={args.off_diag})..."
    )
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
        normalize_biadj=not args.raw_biadj,
        uu_topk=args.uu_topk,
        off_diag=args.off_diag,
    )
    data = tf(data)
    setup_time_s = time.time() - t0
    print(f"  Done in {setup_time_s:.1f}s")
    print(
        f"  edge_index2: {data.edge_index2.shape}, edge_attr2: {data.edge_attr2.shape}"
    )

    ne = data.edge_attr2.shape[1]
    use_struct_feats = not args.no_struct_feats
    model = GNNML3LinkPredictor(
        data.x.shape[1],
        ne,
        nu,
        num_nodes=data.x.shape[0],
        emb_in=args.emb_in,
        n_layers=args.layers,
        embed_dim=args.embed_dim,
        use_struct_feats=use_struct_feats,
        layer_combine=args.layer_combine,
    )
    model = model.to(device)
    data = data.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = sum(1 for m in model.modules() if isinstance(m, ML3Layer))
    feat_desc = []
    if args.emb_in > 0:
        feat_desc.append(f"emb_in={args.emb_in}")
        if model.use_struct_feats:
            feat_desc.append("+struct")
    else:
        feat_desc.append("struct-only")
    print(
        f"  Model: {n_layers} layers, "
        f"{'mean' if args.layer_combine else 'last'}-layer readout, "
        f"input=[{', '.join(feat_desc)}]"
    )
    print(f"  Params: {n_params:,}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scaler = torch.amp.GradScaler() if use_amp else None

    # All training interactions as parallel (user, positive-item) tensors. Each
    # epoch samples one negative item per interaction. Negatives are drawn with a
    # vectorized randint without per-edge rejection: at these densities (~1e-3)
    # the false-negative rate is negligible and BPR tolerates it, so we keep the
    # sampler GPU-friendly instead of looping in Python.
    inter_u = torch.tensor([u for u, _ in tr_e], dtype=torch.long, device=device)
    inter_i = torch.tensor([i for _, i in tr_e], dtype=torch.long, device=device)
    n_inter = inter_u.shape[0]

    def compute_loss(u_idx, pos_idx, neg_idx):
        ue, ie = model(data)
        pos_s = (ue[u_idx] * ie[pos_idx]).sum(1)
        neg_s = (ue[u_idx] * ie[neg_idx]).sum(1)
        return -F.logsigmoid(pos_s - neg_s).mean()

    def optimize(u_idx, pos_idx, neg_idx):
        opt.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                loss = compute_loss(u_idx, pos_idx, neg_idx)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = compute_loss(u_idx, pos_idx, neg_idx)
            loss.backward()
            opt.step()
        return loss.item()

    if args.bpr_batch > 0:
        mode_desc = f"minibatch BPR (bsz={args.bpr_batch})"
    else:
        mode_desc = "full-batch (all interactions / epoch)"
    print(f"\nTraining {args.epochs} epochs, {mode_desc}, {n_inter:,} interactions...")

    # Mirror LightGCN: evaluate on test periodically and report the best epoch.
    best = None
    best_epoch = 0
    time_to_best_s = 0.0
    train_start = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        neg = torch.randint(0, ni, (n_inter,), dtype=torch.long, device=device)

        if args.bpr_batch > 0:
            perm = torch.randperm(n_inter, device=device)
            losses = []
            for s in range(0, n_inter, args.bpr_batch):
                idx = perm[s : s + args.bpr_batch]
                losses.append(optimize(inter_u[idx], inter_i[idx], neg[idx]))
            loss_val = sum(losses) / max(len(losses), 1)
        else:
            loss_val = optimize(inter_u, inter_i, neg)

        if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
            rec = evaluate(model, data, te_ui, tr_ui, ks=ks)
            r_primary = rec[f"recall@{primary_k}"]
            print(
                f"  Epoch {ep:4d} | Loss: {loss_val:.4f} | "
                f"R@{primary_k}: {r_primary:.4f}  N@{primary_k}: {rec[f'ndcg@{primary_k}']:.4f}"
            )
            if args.history_out:
                hrec = {
                    "model": "gnnml3",
                    "run_id": args.run_id,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "epoch": ep,
                    "loss": round(loss_val, 6),
                    "nfreq": args.nfreq,
                    "dv": args.dv,
                    "k_svd": args.k,
                }
                for k in ks:
                    hrec[f"recall@{k}"] = round(rec[f"recall@{k}"], 6)
                    hrec[f"ndcg@{k}"] = round(rec[f"ndcg@{k}"], 6)
                    hrec[f"precision@{k}"] = round(rec[f"precision@{k}"], 6)
                append_jsonl(args.history_out, hrec)
            if best is None or r_primary > best[f"recall@{primary_k}"]:
                best = rec
                best_epoch = ep
                time_to_best_s = time.time() - train_start
                if args.save_model:
                    with torch.no_grad():
                        ue, ie = model(data)
                    os.makedirs(os.path.dirname(os.path.abspath(args.save_model)),
                                exist_ok=True)
                    torch.save(
                        {
                            "model": "gnnml3",
                            "dataset": args.dataset,
                            "epoch": ep,
                            "metrics": {m: float(v) for m, v in rec.items()},
                            "config": vars(args),
                            "state_dict": model.state_dict(),
                            "user_emb": ue.cpu(),
                            "item_emb": ie.cpu(),
                        },
                        args.save_model,
                    )

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
        "run_id": args.run_id,
        "dataset": args.dataset,
        "k_core": args.k_core,
        "seed": args.seed,
        "epochs": args.epochs,
        "embed_dim": args.embed_dim,
        "emb_in": args.emb_in,
        "struct_feats": bool(model.use_struct_feats),
        "layer_combine": bool(args.layer_combine),
        "bpr_batch": args.bpr_batch,
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
        "uu_topk": args.uu_topk,
        "off_diag": args.off_diag,
        "biadj": biadj_kind,
        "amp": bool(use_amp),
    }
    for k in ks:
        row[f"recall@{k}"] = round(best[f"recall@{k}"], 6)
        row[f"ndcg@{k}"] = round(best[f"ndcg@{k}"], 6)
        row[f"precision@{k}"] = round(best[f"precision@{k}"], 6)

    append_jsonl(args.out, row)
    print(f"\nResult appended to {args.out}")
    if args.save_model:
        print(f"Best-epoch checkpoint saved to {args.save_model}")


if __name__ == "__main__":
    main()
