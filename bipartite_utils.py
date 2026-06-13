"""
Bipartite spectral design utilities for GNNML3.

Extends the spectral convolution approach from "Breaking the Limits of
Message Passing Graph Neural Networks" (Balcilar et al., ICML 2021) to
bipartite graphs. Instead of decomposing the full Laplacian, we SVD the
normalized biadjacency matrix and apply even/odd spectral filters to
separately handle within-partition and cross-partition message passing.

Uses truncated SVD (sp.linalg.svds) with sparse block construction for
efficiency on large graphs.
"""

import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from networkx.algorithms import bipartite
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected


# ──────────────────────────────────────────────────────────────
#  Spectral filter functions (same formulation as spectral.py)
# ──────────────────────────────────────────────────────────────

def phi(x, b=5, f_s=0.5):
    """Gaussian spectral kernel."""
    return np.exp(-b * (x - f_s) ** 2)


def h(x, b=5, f_s=0.5):
    """Even part of the spectral kernel: 0.5*(phi(x) + phi(-x))."""
    return 0.5 * (phi(x, b, f_s) + phi(-x, b, f_s))


def g(x, b=5, f_s=1):
    """Odd part of the spectral kernel: 0.5*(phi(x) - phi(-x))."""
    return 0.5 * (phi(x, b, f_s) - phi(-x, b, f_s))


def normalized_biadjacency(edge_index, num_users, num_items):
    """
    Build the row-and-column-normalized biadjacency matrix from edge_index.

    Args:
        edge_index: torch.Tensor of shape (2, num_edges), undirected edges
                    where users have ids 0..num_users-1 and items
                    have ids num_users..num_users+num_items-1.
        num_users: number of user nodes
        num_items: number of item nodes

    Returns:
        scipy.sparse.csr_matrix of shape (num_users, num_items),
        D_u^{-1/2} @ B @ D_v^{-1/2}.
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    # Extract cross-partition edges: user->item  and  item->user
    # user->item: src < num_users, dst >= num_users
    mask_u2i = (src < num_users) & (dst >= num_users)
    u_row = src[mask_u2i]
    i_col = dst[mask_u2i] - num_users

    # item->user: src >= num_users, dst < num_users  (store as (user, item))
    mask_i2u = (src >= num_users) & (dst < num_users)
    u_row2 = dst[mask_i2u]
    i_col2 = src[mask_i2u] - num_users

    all_u = np.concatenate([u_row, u_row2])
    all_i = np.concatenate([i_col, i_col2])

    B = sp.csr_matrix((np.ones(len(all_u), dtype=np.float32),
                       (all_u, all_i)), shape=(num_users, num_items))
    # Deduplicate
    B.data = np.ones_like(B.data, dtype=np.float32)

    # Normalize: D_u^{-1/2} @ B @ D_v^{-1/2}
    row_sums = np.array(B.sum(axis=1)).flatten()
    col_sums = np.array(B.sum(axis=0)).flatten()

    d_inv_sqrt_row = np.zeros_like(row_sums)
    d_inv_sqrt_col = np.zeros_like(col_sums)
    np.power(row_sums, -0.5, where=(row_sums != 0), out=d_inv_sqrt_row)
    np.power(col_sums, -0.5, where=(col_sums != 0), out=d_inv_sqrt_col)

    D_u_inv_sqrt = sp.diags(d_inv_sqrt_row)
    D_v_inv_sqrt = sp.diags(d_inv_sqrt_col)

    return D_u_inv_sqrt @ B @ D_v_inv_sqrt


def _svds_seeded(B, k, seed):
    """Truncated SVD with a reproducible start vector.

    scipy renamed the seeding kwarg over time (``random_state`` -> ``rng``); try
    the modern name first and fall back gracefully so this works across versions.
    """
    if seed is None:
        return sp.linalg.svds(B, k=k)
    try:
        return sp.linalg.svds(B, k=k, rng=seed)
    except TypeError:
        pass
    try:
        return sp.linalg.svds(B, k=k, random_state=seed)
    except TypeError:
        # Oldest scipy: pin a deterministic start vector manually.
        rng = np.random.RandomState(seed)
        v0 = rng.standard_normal(min(B.shape)).astype(np.float64)
        return sp.linalg.svds(B, k=k, v0=v0)


# ──────────────────────────────────────────────────────────────
#  BipartiteSpectralDesign – the core pre-transform
# ──────────────────────────────────────────────────────────────

class BipartiteSpectralDesign(object):
    """
    Spectral design pre-transform for bipartite graphs.

    Uses SVD of the normalized biadjacency matrix instead of full Laplacian
    eigen-decomposition. Applies even filters h(σ) to within-partition blocks
    (UU, VV) and odd filters g(σ) to cross-partition blocks (UV, VU).

    Args:
        num_users: number of user nodes in the graph.
        nfreq: number of frequency sampling points (spectral support count).
        dv: bandwidth parameter for Gaussian kernel (b in paper notation).
        k: number of singular vectors for truncated SVD (0 = full SVD,
           useful for small graphs only).
        recfield: receptive field. 0=adj only, 1=adj+I, >=2: n-hop area.
        adddegree: whether to append node degree as a feature.
        addadj: whether to append raw adjacency as an additional support.
        nmax: max nodes for PPGN (set 0 to skip PPGN tensors).
        seed: seed for the truncated-SVD start vector (reproducibility). The
              spectral features are bilinear in the singular vectors, so the
              sign ambiguity cancels; only the iterative start vector needs
              pinning.
    """

    def __init__(self, num_users, nfreq=5, dv=5, k=100, recfield=1,
                 adddegree=True, addadj=False, nmax=0, seed=None):
        self.num_users = num_users
        self.nfreq = nfreq
        self.dv = dv
        self.k = k
        self.recfield = recfield
        self.adddegree = adddegree
        self.addadj = addadj
        self.nmax = nmax
        self.seed = seed

    def __call__(self, data):
        n = data.x.shape[0]
        nf = data.x.shape[1]
        num_users = self.num_users
        num_items = n - num_users

        data.x = data.x.type(torch.float32)

        nsup = self.nfreq + 1  # +1 for identity
        if self.addadj:
            nsup += 1

        # ── build full adjacency A (sparse, binary) ──────────
        row, col = data.edge_index[0].numpy(), data.edge_index[1].numpy()
        data_vals = np.ones(len(row), dtype=np.float32)
        A_sp = sp.csr_matrix((data_vals, (row, col)), shape=(n, n))
        A_sp.data = np.ones_like(A_sp.data, dtype=np.float32)

        # ── optional degree feature ─────────────────────────
        if self.adddegree:
            deg = np.array(A_sp.sum(0)).flatten()
            data.x = torch.cat([data.x, torch.tensor(deg).unsqueeze(-1)], 1)

        # ── receptive field mask M (sparse) ─────────────────
        if self.recfield == 0:
            M_sp = A_sp.copy()
        else:
            I_sp = sp.eye(n, dtype=np.float32, format='csr')
            M_sp = (A_sp + I_sp).astype(bool).astype(np.float32)
            for _ in range(1, self.recfield):
                M_sp = (M_sp @ M_sp).astype(bool).astype(np.float32)

        # ── build normalized biadjacency B ──────────────────
        B = normalized_biadjacency(data.edge_index, num_users, num_items)

        # ── SVD of biadjacency ──────────────────────────────
        if self.k > 0 and self.k < min(num_users, num_items):
            U, S, Vt = _svds_seeded(B, self.k, self.seed)
            idx = np.argsort(S)[::-1]
            S = S[idx]
            U = U[:, idx]
            Vt = Vt[idx, :]
            V = Vt.T
        else:
            B_dense = B.toarray()
            U, S, Vt = np.linalg.svd(B_dense, full_matrices=False)
            V = Vt.T

        lambda_max = S[0]
        freqcenter = np.linspace(-lambda_max, lambda_max, self.nfreq)

        # ── split M into quadrants (COO) and build masks ────
        M_11 = M_sp[:num_users, :num_users].tocoo()
        M_12 = M_sp[:num_users, num_users:].tocoo()
        M_21 = M_sp[num_users:, :num_users].tocoo()
        M_22 = M_sp[num_users:, num_users:].tocoo()

        # ── build edge_index2 and edge_attr2 ────────────────
        M_coo = M_sp.tocoo()
        edge_index2_list = np.vstack([M_coo.row, M_coo.col])
        num_edges2 = edge_index2_list.shape[1]
        edge_attr2 = np.zeros((num_edges2, nsup), dtype=np.float32)

        # Boolean masks for each quadrant in M_coo
        is_uu = (M_coo.row < num_users) & (M_coo.col < num_users)
        is_uv = (M_coo.row < num_users) & (M_coo.col >= num_users)
        is_vu = (M_coo.row >= num_users) & (M_coo.col < num_users)
        is_vv = (M_coo.row >= num_users) & (M_coo.col >= num_users)

        # Local indices within each block (relative row/col)
        uu_row = M_coo.row[is_uu]
        uu_col = M_coo.col[is_uu]
        uv_row = M_coo.row[is_uv]
        uv_col = M_coo.col[is_uv] - num_users
        vu_row = M_coo.row[is_vu] - num_users
        vu_col = M_coo.col[is_vu]
        vv_row = M_coo.row[is_vv] - num_users
        vv_col = M_coo.col[is_vv] - num_users

        for i, f_s in enumerate(freqcenter):
            h_S = h(S, b=self.dv, f_s=f_s)
            g_S = g(S, b=self.dv, f_s=f_s)

            # UU block (even filter)
            if len(uu_row) > 0:
                edge_attr2[is_uu, i] = np.sum(
                    U[uu_row, :] * h_S * U[uu_col, :], axis=1)
            # UV block (odd filter)
            if len(uv_row) > 0:
                edge_attr2[is_uv, i] = np.sum(
                    U[uv_row, :] * g_S * V[uv_col, :], axis=1)
            # VU block (odd filter)
            if len(vu_row) > 0:
                edge_attr2[is_vu, i] = np.sum(
                    V[vu_row, :] * g_S * U[vu_col, :], axis=1)
            # VV block (even filter)
            if len(vv_row) > 0:
                edge_attr2[is_vv, i] = np.sum(
                    V[vv_row, :] * h_S * V[vv_col, :], axis=1)

        # ── identity support (column nfreq) ─────────────────
        diag_mask = M_coo.row == M_coo.col
        edge_attr2[diag_mask, self.nfreq] = 1.0

        # ── optional adjacency support ──────────────────────
        if self.addadj:
            # M_coo entries that are actual edges (A_sp nonzeros)
            A_coo = A_sp.tocoo()
            flat_a = set(zip(A_coo.row, A_coo.col))
            for j in range(num_edges2):
                if (M_coo.row[j], M_coo.col[j]) in flat_a:
                    edge_attr2[j, self.nfreq + 1] = 1.0

        data.edge_index2 = torch.tensor(edge_index2_list, dtype=torch.int64)
        data.edge_attr2 = torch.tensor(edge_attr2, dtype=torch.float32)

        # ── PPGN tensors (if nmax > 0) ──────────────────────
        if self.nmax > 0:
            H = torch.zeros(1, nf + 2, self.nmax, self.nmax)
            H[0, 0, data.edge_index[0], data.edge_index[1]] = 1
            H[0, 1, :n, :n] = torch.diag(torch.ones(n))
            for j in range(nf):
                H[0, j + 2, :n, :n] = torch.diag(data.x[:, j])
            data.X2 = H
            Mmask = torch.zeros(1, 2, self.nmax, self.nmax)
            for i in range(n):
                Mmask[0, 0, i, i] = 1
            Mmask[0, 1, :n, :n] = 1 - Mmask[0, 0, :n, :n]
            data.M = Mmask

        return data


# ──────────────────────────────────────────────────────────────
#  BipartiteDataset – loads bipartite interaction txt files
# ──────────────────────────────────────────────────────────────

class BipartiteDataset(InMemoryDataset):
    """
    Loads a bipartite graph from an edge-list .txt file.

    File format: one line per user
        user_id item_id1 item_id2 item_id3 ...

    User and item ids are re-indexed to 0..num_users-1 and
    num_users..num_users+num_items-1 respectively.

    Args:
        root: dataset root directory (should contain raw/<name>.txt).
        name: base filename (without .txt extension).
        transform: PyG transform (applied after pre_transform, at load time).
        pre_transform: PyG transform (applied before saving to disk).
    """

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(BipartiteDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.name}.txt"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        b = self.processed_paths[0]

        # ── read txt file ───────────────────────────────────
        edges = []
        user_ids = set()
        item_ids = set()
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                u = int(parts[0])
                user_ids.add(u)
                for it in parts[1:]:
                    i = int(it)
                    item_ids.add(i)
                    edges.append((u, i))

        # ── re-index ────────────────────────────────────────
        user_list = sorted(user_ids)
        item_list = sorted(item_ids)
        user2idx = {u: i for i, u in enumerate(user_list)}
        item2idx = {i: j + len(user_list) for j, i in enumerate(item_list)}

        num_users = len(user_list)
        num_items = len(item_list)
        n_total = num_users + num_items

        edge_list = []
        for u, i in edges:
            src = user2idx[u]
            dst = item2idx[i]
            edge_list.append([src, dst])
            edge_list.append([dst, src])  # undirected

        edge_index = torch.tensor(edge_list).T.type(torch.int64)
        edge_index = to_undirected(edge_index)

        # ── node features (one-hot for user vs item + degree) ──
        x = torch.zeros(n_total, 2)
        x[:num_users, 0] = 1.0  # user indicator
        x[num_users:, 1] = 1.0  # item indicator
        y = torch.tensor([0])   # placeholder

        data = Data(edge_index=edge_index, x=x, y=y)
        data.num_users = num_users
        data.num_items = num_items

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
