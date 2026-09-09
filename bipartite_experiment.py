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
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

# Anchor imports/paths to this file's directory so the script works from any CWD.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnn-matlang"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libs.spect_conv import SpectConv, ML3Layer
from bipartite_utils import BipartiteSpectralDesign
from eval_common import score
from bench_utils import append_jsonl
from kcore import (
    k_core_filter,
    load_k_core_cache,
    remap_k_core,
    save_k_core_cache,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


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
        nout2: width of the multiplicative tanh*tanh gating branch of ML3Layer;
                0 removes that branch entirely (ablation).
        learnedge: whether ML3Layer learns the edge features (its edge MLP) or
                uses the fixed spectral supports as-is (ablation).
        shared_head: use one readout projection for users and items instead of
                two partition-specific ones (ablation).
    """

    def __init__(self, ninp, ne, num_users, num_nodes=0, emb_in=0, n_layers=3,
                 nout1=64, nout2=32, embed_dim=64, use_struct_feats=True,
                 layer_combine=False, learnedge=True, shared_head=False,
                 grad_checkpoint=False):
        super().__init__()
        self.num_users = num_users
        self.emb_in = emb_in
        self.use_struct_feats = use_struct_feats or emb_in == 0
        self.layer_combine = layer_combine
        self.learnedge = learnedge
        self.shared_head = shared_head
        # Recompute each layer's activations in backward instead of storing the
        # per-edge message tensors. The supports carry millions of edges, so
        # these are the dominant memory term; checkpointing trades ~30% compute
        # for a large drop in peak memory. Numerically identical (no RNG in the
        # forward pass).
        self.grad_checkpoint = grad_checkpoint

        if emb_in > 0:
            self.node_emb = nn.Embedding(num_nodes, emb_in)
            nn.init.normal_(self.node_emb.weight, std=0.1)
        else:
            self.node_emb = None

        conv_in = (emb_in if emb_in > 0 else 0) + (ninp if self.use_struct_feats else 0)
        nin = nout1 + nout2
        self.convs = nn.ModuleList(
            ML3Layer(
                learnedge=learnedge,
                nedgeinput=ne,
                nedgeoutput=ne,
                ninp=conv_in if li == 0 else nin,
                nout1=nout1,
                nout2=nout2,
            )
            for li in range(n_layers)
        )
        if shared_head:
            # Single readout shared by both partitions (ablation: does the
            # user/item-specific projection matter?).
            self.head = nn.Linear(nin, embed_dim)
            self.user_head = self.item_head = None
        else:
            self.head = None
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
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    conv, x, ei, ea, use_reentrant=False)
            else:
                x = conv(x, ei, ea)
            outs.append(x)
        h = torch.stack(outs, dim=0).mean(0) if self.layer_combine else outs[-1]
        if self.head is not None:
            return self.head(h[: self.num_users]), self.head(h[self.num_users :])
        return self.user_head(h[: self.num_users]), self.item_head(h[self.num_users :])


def ablate_supports(data, num_users, nfreq, drop_identity=False,
                    drop_even=False, drop_odd=False):
    """Zero out selected spectral-support columns of ``edge_attr2`` in place.

    ``BipartiteSpectralDesign`` lays the supports out as columns
    ``0 .. nfreq-1`` (the Gaussian frequency bands) followed by the identity
    support at column ``nfreq`` (and, if ``addadj``, an adjacency support after
    it). Within the band columns the within-partition blocks (UU and VV) carry
    the *even* filters ``h(sigma)`` and the cross-partition blocks (UV and VU)
    carry the *odd* filters ``g(sigma)``. This helper removes exactly one of
    those mechanisms at a time, so the rest of the pipeline (masks, edge
    network, message passing) is untouched and the comparison stays paired.

    Args:
        data: Data object carrying ``edge_index2`` / ``edge_attr2``.
        num_users: user/item split point of the node ids.
        nfreq: number of frequency-band columns.
        drop_identity: zero the identity support (self-connection in the conv).
        drop_even: zero the even-filter columns on within-partition edges.
        drop_odd: zero the odd-filter columns on cross-partition edges.
    """
    if not (drop_identity or drop_even or drop_odd):
        return data
    ea = data.edge_attr2
    ei = data.edge_index2
    if drop_identity:
        ea[:, nfreq] = 0.0
    if nfreq > 0 and (drop_even or drop_odd):
        is_user = ei[0] < num_users
        col_is_user = ei[1] < num_users
        same_partition = is_user == col_is_user
        if drop_even:
            ea[same_partition, :nfreq] = 0.0
        if drop_odd:
            ea[~same_partition, :nfreq] = 0.0
    return data


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
        "--dataset",
        default="gowalla",
        choices=["amazon-book", "gowalla", "yelp2018", "spotify"],
        help="spotify requires running convert_spotify.py first",
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
        "--design-cache",
        default="",
        help="directory to cache the built spectral supports "
        "(x/edge_index2/edge_attr2) keyed by the spectral-design config; empty "
        "= rebuild every run. Architecture ablations share one design, so this "
        "removes the repeated SVD/support cost from an ablation matrix",
    )
    p.add_argument(
        "--design-seed",
        type=int,
        default=-1,
        help="seed for the truncated-SVD start vector; -1 = use --seed. Pin it "
        "to make the spectral design identical across training seeds (and "
        "therefore cacheable across an ablation matrix)",
    )
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
    # ── ablation switches (all default to the full model) ────────────────
    p.add_argument(
        "--ablation",
        default="",
        help="free-form tag for the ablation this run represents; stored in the "
        "result record so a study driver can join runs into variants",
    )
    p.add_argument(
        "--no-learnedge",
        action="store_true",
        help="ablation: drop ML3Layer's learnable edge network and use the fixed "
        "spectral supports as edge weights",
    )
    p.add_argument(
        "--nout2",
        type=int,
        default=32,
        help="width of the ML3 multiplicative tanh*tanh gating branch; "
        "0 removes that branch (ablation)",
    )
    p.add_argument(
        "--shared-head",
        action="store_true",
        help="ablation: one readout projection shared by users and items instead "
        "of partition-specific user/item heads",
    )
    p.add_argument(
        "--no-identity",
        action="store_true",
        help="ablation: zero the identity support column, removing the conv's "
        "self-connection (the gating branch still sees the raw node features)",
    )
    p.add_argument(
        "--no-even",
        action="store_true",
        help="ablation: zero the even spectral filters on within-partition "
        "(UU/VV) edges, i.e. remove user-user / item-item message passing",
    )
    p.add_argument(
        "--no-odd",
        action="store_true",
        help="ablation: zero the odd spectral filters on cross-partition "
        "(UV/VU) edges, i.e. remove user-item message passing",
    )
    p.add_argument(
        "--no-degree",
        action="store_true",
        help="ablation: drop the log-degree structural feature (indicator "
        "columns only)",
    )
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="recompute layer activations in backward (torch.utils.checkpoint) "
        "to cut peak memory on graphs with millions of support edges; ~30%% "
        "slower per epoch, numerically identical",
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
        help="L2 regularization strength: applied as Adam weight_decay and as "
        "an explicit L2 penalty on the user/item embeddings (LightGCN-style)",
    )
    p.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="max gradient norm for clipping (0 = no clipping)",
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
        train_path, test_path = f"{dd}/train.txt", f"{dd}/test.txt"
        cached = load_k_core_cache(train_path, test_path, args.k_core)
        if cached is not None:
            tr_e, te_e, nu, ni = cached
            print("  (loaded from cache)")
        else:
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
            save_k_core_cache(train_path, test_path, args.k_core,
                              (tr_e, te_e, nu, ni))
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
    design_seed = args.seed if args.design_seed < 0 else args.design_seed
    print(
        f"Spectral design (nfreq={args.nfreq}, dv={args.dv}, k={args.k}, "
        f"biadj={biadj_kind}, uu_topk={args.uu_topk}, "
        f"off_diag={args.off_diag})..."
    )
    t0 = time.time()
    # The support construction (SVD + per-edge spectral entries) depends only on
    # the spectral-design configuration, not on the architecture flags, so an
    # ablation matrix can build it once and reuse it. Keyed on every input that
    # changes the result, including the design seed.
    cache_path = ""
    if args.design_cache:
        import hashlib
        os.makedirs(args.design_cache, exist_ok=True)
        key = "|".join(map(str, [
            args.dataset, args.k_core, nu, ni, args.nfreq, args.dv, args.k,
            args.recfield, int(not args.no_degree), biadj_kind, args.uu_topk,
            int(args.off_diag), design_seed,
        ]))
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        cache_path = os.path.join(args.design_cache, f"design_{digest}.pt")

    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        data.x = blob["x"]
        data.edge_index2 = blob["edge_index2"]
        data.edge_attr2 = blob["edge_attr2"]
        print(f"  (design loaded from cache {os.path.basename(cache_path)})")
    else:
        tf = BipartiteSpectralDesign(
            nu,
            nfreq=args.nfreq,
            dv=args.dv,
            k=args.k,
            recfield=args.recfield,
            adddegree=not args.no_degree,
            nmax=0,
            seed=design_seed,
            normalize_biadj=not args.raw_biadj,
            uu_topk=args.uu_topk,
            off_diag=args.off_diag,
        )
        data = tf(data)
        if cache_path:
            torch.save({"x": data.x, "edge_index2": data.edge_index2,
                        "edge_attr2": data.edge_attr2, "key": key}, cache_path)
    setup_time_s = time.time() - t0
    ablate_supports(
        data,
        nu,
        args.nfreq,
        drop_identity=args.no_identity,
        drop_even=args.no_even,
        drop_odd=args.no_odd,
    )
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
        nout2=args.nout2,
        embed_dim=args.embed_dim,
        use_struct_feats=use_struct_feats,
        layer_combine=args.layer_combine,
        learnedge=not args.no_learnedge,
        shared_head=args.shared_head,
        grad_checkpoint=args.grad_checkpoint,
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
    if args.ablation:
        print(f"  Ablation: {args.ablation}")
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
        bpr = -F.logsigmoid(pos_s - neg_s).mean()
        # LightGCN-style L2 penalty on the involved user/item embeddings, scaled
        # by --decay, to bound embedding norms (BPR itself has no such bound).
        reg = 0.5 * (
            ue[u_idx].pow(2).sum(1).mean()
            + ie[pos_idx].pow(2).sum(1).mean()
            + ie[neg_idx].pow(2).sum(1).mean()
        )
        return bpr + args.decay * reg

    def optimize(u_idx, pos_idx, neg_idx):
        opt.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                loss = compute_loss(u_idx, pos_idx, neg_idx)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss = compute_loss(u_idx, pos_idx, neg_idx)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        return loss.item()

    if args.bpr_batch > 0:
        mode_desc = f"minibatch BPR (bsz={args.bpr_batch})"
    else:
        mode_desc = "full-batch (all interactions / epoch)"
    print(f"\nTraining {args.epochs} epochs, {mode_desc}, {n_inter:,} interactions...")

    # Mirror LightGCN: evaluate on test periodically and report the best epoch.
    # ``last`` keeps the final evaluation, i.e. the fixed-budget metric with no
    # best-epoch selection bias, which is the primary number ablation studies
    # compare (``best`` stays the headline metric for the main results table).
    best = None
    last = None
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
            last = rec
            r_primary = rec[f"recall@{primary_k}"]
            print(
                f"  Epoch {ep:4d} | Loss: {loss_val:.4f} | "
                f"R@{primary_k}: {r_primary:.4f}  N@{primary_k}: {rec[f'ndcg@{primary_k}']:.4f}"
            )
            if args.history_out:
                hrec = {
                    "model": "gnnml3",
                    "run_id": args.run_id,
                    "ablation": args.ablation,
                    "dataset": args.dataset,
                    "k_core": args.k_core,
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
        "clip": args.clip,
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
        # ablation switches (all False / default = the full model)
        "ablation": args.ablation,
        "learnedge": bool(args.no_learnedge is False),
        "nout2": args.nout2,
        "shared_head": bool(args.shared_head),
        "drop_identity": bool(args.no_identity),
        "drop_even": bool(args.no_even),
        "drop_odd": bool(args.no_odd),
        "degree_feat": bool(not args.no_degree),
        "design_seed": design_seed,
        "grad_checkpoint": bool(args.grad_checkpoint),
    }
    for k in ks:
        row[f"recall@{k}"] = round(best[f"recall@{k}"], 6)
        row[f"ndcg@{k}"] = round(best[f"ndcg@{k}"], 6)
        row[f"precision@{k}"] = round(best[f"precision@{k}"], 6)
        if last is not None:
            row[f"final_recall@{k}"] = round(last[f"recall@{k}"], 6)
            row[f"final_ndcg@{k}"] = round(last[f"ndcg@{k}"], 6)
            row[f"final_precision@{k}"] = round(last[f"precision@{k}"], 6)

    append_jsonl(args.out, row)
    print(f"\nResult appended to {args.out}")
    if args.save_model:
        print(f"Best-epoch checkpoint saved to {args.save_model}")


if __name__ == "__main__":
    main()
