"""
Bipartite spectral design utilities for GNNML3.

Extends the spectral convolution approach from "Breaking the Limits of
Message Passing Graph Neural Networks" (Balcilar et al., ICML 2021) to
bipartite graphs. Instead of decomposing the full Laplacian, we SVD the
normalized biadjacency matrix and apply even/odd spectral filters to
separately handle within-partition and cross-partition message passing.

Uses truncated SVD (sp.linalg.svds) with sparse block construction for
efficiency on large graphs.

The receptive-field mask is ``M = A + I`` by default, optionally augmented with
sparsified within-partition co-interaction edges (``uu_topk``), a consecutive-id
band (``off_diag``), and/or the candidate/negative pair set of the augmented
mask ``M' = A + I + P`` (``cand_pairs``, see ``change-ref/GNNML3_LP_CF_analysis.pdf``
sec. 6.1).
"""

import warnings

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


def normalized_biadjacency(edge_index, num_users, num_items, normalize=True):
    """
    Build the biadjacency matrix from edge_index, optionally normalized.

    Args:
        edge_index: torch.Tensor of shape (2, num_edges), undirected edges
                    where users have ids 0..num_users-1 and items
                    have ids num_users..num_users+num_items-1.
        num_users: number of user nodes
        num_items: number of item nodes
        normalize: if True (default), return the symmetrically normalized
                   biadjacency D_u^{-1/2} @ B @ D_v^{-1/2}; if False, return the
                   raw binary biadjacency B.

    Returns:
        scipy.sparse.csr_matrix of shape (num_users, num_items).
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

    if not normalize:
        return B

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


def topk_cooccurrence(B, topk, block_size=2048):
    """Sparsified co-occurrence graph: top-``topk`` rows of ``B @ B.T``.

    Computes the (row-side) co-interaction graph of a biadjacency matrix in
    row blocks so the full product is never materialized at once, keeping only
    the ``topk`` strongest neighbors per row (diagonal excluded; the identity
    support covers self-loops separately). Pass the *normalized* biadjacency so
    the weights down-rank promiscuous columns instead of raw co-counts.

    The result is symmetrized with an element-wise max, so an edge survives if
    either endpoint ranks the other in its top-``topk``.

    Args:
        B: scipy sparse matrix (num_rows x num_cols); rows are the partition
           the co-occurrence graph is built over.
        topk: neighbors kept per row (before symmetrization).
        block_size: rows per spgemm block (bounds peak memory).

    Returns:
        scipy.sparse.csr_matrix of shape (num_rows, num_rows).
    """
    B = B.tocsr()
    m = B.shape[0]
    Bt = B.T.tocsr()
    rows_out, cols_out, vals_out = [], [], []
    for s in range(0, m, block_size):
        blk = (B[s : s + block_size] @ Bt).tocsr()
        for i in range(blk.shape[0]):
            lo, hi = blk.indptr[i], blk.indptr[i + 1]
            c = blk.indices[lo:hi]
            v = blk.data[lo:hi]
            keep = c != (s + i)  # drop the self-loop
            c, v = c[keep], v[keep]
            if len(v) > topk:
                sel = np.argpartition(v, -topk)[-topk:]
                c, v = c[sel], v[sel]
            if len(c):
                rows_out.append(np.full(len(c), s + i, dtype=np.int64))
                cols_out.append(c.astype(np.int64))
                vals_out.append(v.astype(np.float32))
    if not rows_out:
        return sp.csr_matrix((m, m), dtype=np.float32)
    G = sp.csr_matrix(
        (np.concatenate(vals_out),
         (np.concatenate(rows_out), np.concatenate(cols_out))),
        shape=(m, m),
    )
    return G.maximum(G.T).tocsr()


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


def _spectral_block_attr(U, rows, filt, V, cols, chunk_elems=1 << 24):
    """Row-wise ``sum_c U[rows, c] * filt[c] * V[cols, c]``, evaluated in chunks.

    The naive expression materializes a ``len(rows) x k`` temporary, which at
    recommendation scale (millions of support edges, k in the thousands) is
    hundreds of GB. Chunking bounds the temporaries to ``chunk_elems`` elements
    while computing exactly the same per-row sums (same summation order), so the
    result is numerically identical to the unchunked code.

    Args:
        U: (n_rows_total, k) singular-vector matrix (left block).
        rows: integer array of row indices into ``U``.
        filt: (k,) spectral filter values for one frequency band.
        V: (n_cols_total, k) singular-vector matrix (right block); may be ``U``.
        cols: integer array of row indices into ``V`` (same length as ``rows``).
        chunk_elems: max elements per temporary (default 16M ~ 128 MB float64).
    """
    n = len(rows)
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    k = max(1, U.shape[1])
    step = max(1, int(chunk_elems) // k)
    for s in range(0, n, step):
        e = min(s + step, n)
        out[s:e] = np.sum(U[rows[s:e], :] * filt * V[cols[s:e], :], axis=1)
    return out


def _sample_candidate_pairs(A_sp, num_users, num_items, ratio, seed):
    """Sample the candidate/negative pair set ``P`` of an augmented mask.

    Implements the ``P`` term of the augmented receptive field
    ``M' = A + I + P`` proposed in ``change-ref/GNNML3_LP_CF_analysis.pdf`` §6.1:
    ``P`` is a set of user-item *non-edges* (pairs absent from the observed
    biadjacency) drawn uniformly, with ``|P| = ratio * |E_train|``. Including
    them in the mask makes the spectral supports non-zero on unobserved pairs,
    so the per-pair edge transform (``mlp1..4`` / the ``learnedge`` branch) is
    evaluated on negatives as well as on observed edges.

    The pairs are sampled once, seeded by ``seed``, and returned as a symmetric
    ``(n, n)`` binary matrix (each pair contributes both ``(u, v)`` and
    ``(v, u)``, matching the undirected adjacency the mask is built from).

    Note: the report describes ``P`` as an epoch-local object (resampled with
    the training negatives). Rebuilding the supports every epoch is far more
    expensive than sampling here once, so this implementation fixes ``P`` at
    design-build time and is cached with the rest of the design; see the
    docstring of :class:`BipartiteSpectralDesign`.

    Args:
        A_sp: (n, n) sparse binary adjacency (undirected, users first).
        num_users: user/item split point.
        num_items: number of item nodes.
        ratio: pairs per training interaction (``1.0`` -> ``|P| = |E_train|``).
        seed: RNG seed for reproducible sampling.

    Returns:
        scipy.sparse.csr_matrix of shape (n, n), binary, symmetric.
    """
    n = A_sp.shape[0]
    n_target = int(round(float(ratio) * (A_sp.nnz // 2)))
    if n_target <= 0 or num_users == 0 or num_items == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)

    # Existing user->item pairs as flat keys u * num_items + i, for O(log E)
    # membership tests while rejection-sampling non-edges.
    A_coo = A_sp.tocoo()
    is_u2v = (A_coo.row < num_users) & (A_coo.col >= num_users)
    keys_existing = (A_coo.row[is_u2v].astype(np.int64) * num_items
                     + (A_coo.col[is_u2v] - num_users))
    keys_existing.sort()
    n_possible = int(num_users) * int(num_items) - int(keys_existing.size)
    if n_target > n_possible:
        raise ValueError(
            f"cand_pairs ratio {ratio} asks for {n_target:,} candidate pairs but "
            f"the graph has only {n_possible:,} user-item non-edges")

    rng = np.random.default_rng(seed)
    keys = np.empty(0, dtype=np.int64)
    rows_u = np.empty(0, dtype=np.int64)
    rows_v = np.empty(0, dtype=np.int64)
    while len(keys) < n_target:
        need = n_target - len(keys)
        # The graphs are sparse (density ~1e-3), so a 10% oversample suffices;
        # the +64 avoids a tight loop on tiny requests.
        draw = int(need * 1.1) + 64
        u = rng.integers(0, num_users, size=draw).astype(np.int64)
        v = rng.integers(0, num_items, size=draw).astype(np.int64)
        k = u * num_items + v

        # reject observed edges
        if keys_existing.size:
            pos = np.searchsorted(keys_existing, k)
            np.clip(pos, 0, keys_existing.size - 1, out=pos)
            keep = keys_existing[pos] != k
            k, u, v = k[keep], u[keep], v[keep]
        if k.size == 0:
            continue
        # deduplicate within this draw, then against everything collected
        uniq, first = np.unique(k, return_index=True)
        k, u, v = uniq, u[first], v[first]
        if keys.size:
            fresh = ~np.isin(k, keys, assume_unique=False)
            k, u, v = k[fresh], u[fresh], v[fresh]
        keys = np.concatenate([keys, k])
        rows_u = np.concatenate([rows_u, u])
        rows_v = np.concatenate([rows_v, v])

    # keep exactly n_target (the last draw may overshoot)
    rows_u, rows_v = rows_u[:n_target], rows_v[:n_target]
    P_sp = sp.csr_matrix(
        (np.ones(2 * n_target, dtype=np.float32),
         (np.concatenate([rows_u, rows_v + num_users]),
          np.concatenate([rows_v + num_users, rows_u]))),
        shape=(n, n),
    )
    return P_sp


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
        normalize_biadj: if True (default), SVD the symmetrically normalized
              biadjacency D_u^{-1/2} B D_v^{-1/2}; if False, SVD the raw binary
              biadjacency B instead.
        uu_topk: if > 0, augment the receptive-field mask with sparsified
              within-partition co-interaction edges: for each user (item), the
              uu_topk strongest neighbors of the normalized co-occurrence graph
              B_hat @ B_hat.T (B_hat.T @ B_hat). This gives the even spectral
              filters real user-user/item-item edges to act on -- the effect of
              recfield=2 -- while keeping the mask ~topk edges/node instead of
              the dense 2-hop blow-up (~1000+ edges/node on these datasets).
        off_diag: if True, also set the first off-diagonal band (i, i+/-1) in
              the UU and VV blocks of the receptive-field mask, giving the even
              spectral filters within-partition edges between consecutive user
              (item) ids.
        cand_pairs: candidate/negative pair ratio for the augmented mask
              ``M' = A + I + P`` (``change-ref/GNNML3_LP_CF_analysis.pdf`` §6.1).
              ``0`` (default) keeps the plain mask ``M = A + I``; ``1.0`` samples
              ``|E_train|`` user-item non-edges into the mask, so the spectral
              supports and the per-pair edge transform are evaluated on
              unobserved pairs too. ``P`` is sampled once, seeded by ``seed``,
              and cached with the rest of the design (the report's epoch-local
              resampling would require rebuilding the supports every epoch).
        flat_support: if True, overwrite every spectral band column with the
              constant ``1.0`` so all support entries are indistinguishable to
              the edge network (which then emits one weight vector per support:
              a plain learned-weight aggregation). A constant non-zero value,
              not zero, because the ML3 edge MLPs are bias-free and ``F(0) = 0``
              would silence message passing entirely. ``nsup``, the support
              graph, the identity column and the ``addadj`` column are unchanged.
        shuffle_bands: if True, evaluate the band filters ``h``/``g`` at a
              seeded permutation of the singular values
              (``edge_attr2[e, s] = sum_c U[u,c] g(sigma_{pi(c)}; f_s) V[i,c]``),
              destroying the frequency correspondence while keeping ``U``, ``V``,
              the support graph and the multiset of filter values. Both this and
              ``flat_support`` are no-ops when ``nfreq == 0`` (warned).
    """

    def __init__(self, num_users, nfreq=5, dv=5, k=100, recfield=1,
                 adddegree=True, addadj=False, nmax=0, seed=None,
                 normalize_biadj=True, uu_topk=0, off_diag=False,
                 cand_pairs=0.0, flat_support=False, shuffle_bands=False,
                 chunk_elems=1 << 24):
        self.num_users = num_users
        self.nfreq = nfreq
        self.dv = dv
        self.k = k
        self.recfield = recfield
        self.adddegree = adddegree
        self.addadj = addadj
        self.nmax = nmax
        self.seed = seed
        self.normalize_biadj = normalize_biadj
        self.uu_topk = uu_topk
        self.off_diag = off_diag
        # Candidate/negative pair ratio for the augmented mask M' = A + I + P
        # (0 = off, the plain M = A + I). See _sample_candidate_pairs.
        self.cand_pairs = float(cand_pairs)
        # Spectral-selectivity controls (both no-ops when nfreq == 0):
        # flat_support writes a constant non-zero descriptor into the band
        # columns; shuffle_bands evaluates the filters at a permutation of the
        # singular values, destroying the frequency correspondence.
        self.flat_support = bool(flat_support)
        self.shuffle_bands = bool(shuffle_bands)
        # Cap on the elements of any temporary in the support construction
        # (~128 MB of float64); keeps peak host RAM bounded on big graphs.
        self.chunk_elems = chunk_elems

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

        # ── optional degree feature (log1p-scaled) ──────────
        if self.adddegree:
            deg = np.array(A_sp.sum(0)).flatten()
            deg = np.log1p(deg).astype(np.float32)
            data.x = torch.cat([data.x, torch.tensor(deg).unsqueeze(-1)], 1)

        # ── receptive field mask M (sparse) ─────────────────
        if self.recfield == 0:
            M_sp = A_sp.copy()
        else:
            I_sp = sp.eye(n, dtype=np.float32, format='csr')
            M_sp = (A_sp + I_sp).astype(bool).astype(np.float32)
            for _ in range(1, self.recfield):
                M_sp = (M_sp @ M_sp).astype(bool).astype(np.float32)

        # ── build (optionally normalized) biadjacency B ─────
        B = normalized_biadjacency(data.edge_index, num_users, num_items,
                                   normalize=self.normalize_biadj)

        # ── sparsified co-interaction edges (uu_topk) ───────
        if self.uu_topk > 0:
            # Selection weights always use the normalized biadjacency, even if
            # the SVD runs on the raw one, so promiscuous nodes don't dominate.
            Bn = B if self.normalize_biadj else normalized_biadjacency(
                data.edge_index, num_users, num_items, normalize=True)
            UU = topk_cooccurrence(Bn, self.uu_topk)
            VV = topk_cooccurrence(Bn.T.tocsr(), self.uu_topk)
            co_sp = sp.bmat([[UU, None], [None, VV]], format='csr')
            M_sp = (M_sp + co_sp).astype(bool).astype(np.float32)

        if self.off_diag:
            off_diags_uu = sp.diags([np.ones(num_users - 1), np.ones(num_users - 1)],
                  offsets=[-1, 1], shape=(num_users, num_users), format='csr')
            off_diags_vv = sp.diags([np.ones(num_items - 1), np.ones(num_items - 1)],
                  offsets=[-1, 1], shape=(num_items, num_items), format='csr')
            co_sp = sp.bmat([[off_diags_uu, None], [None, off_diags_vv]], format='csr')
            M_sp = (M_sp + co_sp).astype(bool).astype(np.float32)

        # ── augmented mask M' = A + I + P (candidate pairs) ─
        # Adds sampled user-item non-edges to the receptive field so the
        # spectral supports (and the per-pair edge transform) are evaluated on
        # unobserved pairs as well. These are cross-partition entries, so the
        # odd filters g(sigma) fill their support values below, exactly like an
        # observed edge; the identity column stays zero off the diagonal.
        if self.cand_pairs > 0:
            P_sp = _sample_candidate_pairs(A_sp, num_users, num_items,
                                           self.cand_pairs, self.seed)
            M_sp = (M_sp + P_sp).astype(bool).astype(np.float32)

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
        # nfreq == 0 is the "no spectral bands" ablation: only the identity
        # support survives (nsup == 1), which lets the edge network turn the
        # conv into a plain learnable aggregation. Guard the division so that
        # degenerate configuration is expressible.
        freqcenter = (np.linspace(lambda_max / self.nfreq, lambda_max, self.nfreq)
                      if self.nfreq > 0 else np.empty(0, dtype=np.float64))

        # Both spectral-selectivity controls act on the band columns, which do
        # not exist at nfreq == 0; say so loudly rather than silently no-op.
        if self.nfreq == 0 and (self.flat_support or self.shuffle_bands):
            flags = ", ".join(f for f, on in
                              (("--flat-support", self.flat_support),
                               ("--shuffle-bands", self.shuffle_bands)) if on)
            warnings.warn(
                f"{flags} with --nfreq 0 is a no-op: there are no spectral band "
                f"columns to modify (the only support is the identity)",
                stacklevel=2)

        # --shuffle-bands: keep U, V and the multiset of filter values but break
        # the frequency correspondence, by evaluating h/g at sigma_{pi(c)} in
        # place of sigma_c.
        S_filt = S
        if self.shuffle_bands and self.nfreq > 0:
            perm = np.random.default_rng(self.seed).permutation(len(S))
            S_filt = S[perm]

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
            h_S = h(S_filt, b=self.dv, f_s=f_s)
            g_S = g(S_filt, b=self.dv, f_s=f_s)

            # UU block (even filter)
            if len(uu_row) > 0:
                edge_attr2[is_uu, i] = _spectral_block_attr(
                    U, uu_row, h_S, U, uu_col, self.chunk_elems)
            # UV block (odd filter)
            if len(uv_row) > 0:
                edge_attr2[is_uv, i] = _spectral_block_attr(
                    U, uv_row, g_S, V, uv_col, self.chunk_elems)
            # VU block (odd filter)
            if len(vu_row) > 0:
                edge_attr2[is_vu, i] = _spectral_block_attr(
                    V, vu_row, g_S, U, vu_col, self.chunk_elems)
            # VV block (even filter)
            if len(vv_row) > 0:
                edge_attr2[is_vv, i] = _spectral_block_attr(
                    V, vv_row, h_S, V, vv_col, self.chunk_elems)

        # ── --flat-support: constant non-zero descriptor on every band ──────
        # Every support entry gets the same band vector, so the edge network
        # emits a single weight per support and the layer degenerates to a plain
        # learned-weight aggregation -- the "no spectral selectivity" reference.
        # Constant 1.0, not 0.0: the edge MLPs are bias-free, so F(0) = 0 would
        # zero every message weight and remove message passing entirely.
        if self.flat_support and self.nfreq > 0:
            edge_attr2[:, :self.nfreq] = 1.0

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
