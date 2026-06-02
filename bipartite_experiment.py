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
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import sys

sys.path.insert(0, "gnn-matlang")
from libs.spect_conv import SpectConv, ML3Layer
from bipartite_utils import BipartiteSpectralDesign


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
def evaluate_recall(
    model, data, test_ui, train_ui, k=20, batch_size=1024, device="cpu"
):
    model.eval()
    ue, ie = model(data)
    ue, ie = ue.cpu(), ie.cpu()
    nu = ue.shape[0]
    hits = total = 0
    for s in range(0, nu, batch_size):
        e = min(s + batch_size, nu)
        scores = ue[s:e] @ ie.T
        for i, u in enumerate(range(s, e)):
            if u in train_ui:
                scores[i, list(train_ui[u])] = -1e10
        _, tk = torch.topk(scores, k, dim=1)
        for i, u in enumerate(range(s, e)):
            if u in test_ui:
                ts = test_ui[u]
                if not ts:
                    continue
                total += len(ts)
                hits += sum(1 for t in ts if t in tk[i].tolist())
    return hits / max(total, 1)


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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto' (detect), 'cuda', 'cpu', or specific device name",
    )
    p.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision (GPU only)"
    )
    args = p.parse_args()

    device, device_name = get_best_device()
    if args.device != "auto":
        device = torch.device(args.device)
        device_name = str(device)
    use_amp = args.amp and device.type == "cuda"
    print(f"Device: {device_name}" + (" (AMP enabled)" if use_amp else ""))

    dd = f"datasets/{args.dataset}"
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
    )
    data = tf(data)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(
        f"  edge_index2: {data.edge_index2.shape}, edge_attr2: {data.edge_attr2.shape}"
    )

    ne = data.edge_attr2.shape[1]
    model = GNNML3LinkPredictor(data.x.shape[1], ne, nu, embed_dim=args.embed_dim)
    model = model.to(device)
    data = data.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler() if use_amp else None
    all_items = list(range(ni))
    tr_users = sorted(tr_ui.keys())
    print(f"\nTraining {args.epochs} epochs...")

    for ep in range(1, args.epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        perm = torch.randperm(len(tr_users))
        for s in range(0, len(tr_users), 512):
            e = min(s + 512, len(tr_users))
            bu = [tr_users[i] for i in perm[s:e].tolist()]

            # 1. Filter users safely (drop users with 0 interactions)
            valid_bu = [u for u in bu if tr_ui[u]]
            if not valid_bu:
                continue

            # 2. Fast Positive Sampling
            pi = [random.choice(tuple(tr_ui[u])) for u in valid_bu]

            # 3. Vectorized Negative Sampling
            ni_tensor = torch.randint(0, ni, (len(valid_bu),), device=device)

            # 4. Convert the user and positive item lists directly to PyTorch tensors
            ul_tensor = torch.tensor(valid_bu, dtype=torch.long, device=device)
            pi_tensor = torch.tensor(pi, dtype=torch.long, device=device)

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
            tl += loss.item()
            nb += 1

        if ep % 10 == 0 or ep == 1:
            r20 = evaluate_recall(model, data, te_ui, tr_ui, k=20, device=device)
            print(f"  Epoch {ep:3d} | Loss: {tl/max(nb,1):.4f} | R@20: {r20:.4f}")

    print("\nFinal...")
    r20 = evaluate_recall(model, data, te_ui, tr_ui, k=20, device=device)
    r50 = evaluate_recall(model, data, te_ui, tr_ui, k=50, device=device)
    print(f"  Recall@20: {r20:.4f}  Recall@50: {r50:.4f}")


if __name__ == "__main__":
    main()
