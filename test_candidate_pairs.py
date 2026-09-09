#!/usr/bin/env python3
"""Self-contained checks for the augmented mask ``M' = A + I + P``.

Verifies the ``--cand-pairs`` implementation of
``change-ref/GNNML3_LP_CF_analysis.pdf`` §6.1 without training anything:

  1. ``_sample_candidate_pairs`` draws exactly ``ratio * |E_train|`` distinct
     user-item non-edges, symmetric, with no observed edge, no duplicates and no
     self-loops; it is reproducible under a fixed seed and changes with the seed;
     it raises rather than looping forever when the ratio exceeds the number of
     available non-edges.
  2. ``BipartiteSpectralDesign(cand_pairs=r)`` grows the support graph by exactly
     ``2r|E_train|`` entries, and the new entries are cross-partition with
     support values equal to the odd spectral filter ``g(sigma)`` evaluated at
     the pair -- i.e. they are treated exactly like observed edges.
  3. ``cand_pairs=0`` (the default) reproduces the previous design bit-for-bit,
     and the identity support stays zero off the diagonal.

Usage:
    python test_candidate_pairs.py            # exits 1 on any failure
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "gnn-matlang"))
sys.path.insert(0, _HERE)

from torch_geometric.data import Data  # noqa: E402

from bipartite_utils import (  # noqa: E402
    BipartiteSpectralDesign, _sample_candidate_pairs, g, normalized_biadjacency,
)

FAILURES = []


def check(name, ok, detail=""):
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def synthetic_bipartite(num_users=60, num_items=45, num_edges=250, seed=0):
    rng = np.random.default_rng(seed)
    edges = set()
    while len(edges) < num_edges:
        edges.add((int(rng.integers(num_users)), int(rng.integers(num_items))))
    edges = sorted(edges)
    el = []
    for u, i in edges:
        el += [[u, num_users + i], [num_users + i, u]]
    ei = torch.tensor(el, dtype=torch.int64).T
    n = num_users + num_items
    A = sp.csr_matrix(
        (np.ones(len(el), dtype=np.float32),
         (np.array([e[0] for e in el]), np.array([e[1] for e in el]))),
        shape=(n, n))
    return ei, A, edges, num_users, num_items


def build_design(ei, num_users, n, **kw):
    x = torch.zeros(n, 2)
    x[:num_users, 0] = 1.0
    x[num_users:, 1] = 1.0
    data = Data(edge_index=ei.clone(), x=x, y=torch.tensor([0]))
    return BipartiteSpectralDesign(num_users, **kw)(data)


def main():
    ei, A, edges, nu, ni = synthetic_bipartite()
    n = nu + ni
    print("1. _sample_candidate_pairs")

    for ratio in (0.25, 1.0, 2.0):
        P = _sample_candidate_pairs(A, nu, ni, ratio, seed=7)
        Pc = P.tocoo()
        pairs = [(int(r), int(c) - nu) for r, c in zip(Pc.row, Pc.col)
                 if r < nu and c >= nu]
        expected = int(round(ratio * len(edges)))
        check(f"ratio={ratio}: |P| == {expected}", len(pairs) == expected,
              f"got {len(pairs)}")
        check(f"ratio={ratio}: no observed edge sampled",
              not (set(pairs) & set(edges)))
        check(f"ratio={ratio}: symmetric, no duplicates, no self-loops",
              (P - P.T).nnz == 0 and len(pairs) == len(set(pairs))
              and (P.diagonal() != 0).sum() == 0)

    P1 = _sample_candidate_pairs(A, nu, ni, 1.0, seed=7)
    P2 = _sample_candidate_pairs(A, nu, ni, 1.0, seed=7)
    P3 = _sample_candidate_pairs(A, nu, ni, 1.0, seed=8)
    check("reproducible under a fixed seed", (P1 - P2).nnz == 0)
    check("changes when the seed changes", (P1 - P3).nnz > 0)

    try:
        _sample_candidate_pairs(A, nu, ni, 100.0, seed=1)
        check("raises when ratio exceeds available non-edges", False)
    except ValueError:
        check("raises when ratio exceeds available non-edges", True)

    print("2. BipartiteSpectralDesign(cand_pairs=...)")
    kw = dict(nfreq=3, dv=5, k=0, recfield=1, seed=7)
    base = build_design(ei, nu, n, **kw)
    aug = build_design(ei, nu, n, cand_pairs=1.0, **kw)
    delta = aug.edge_attr2.shape[0] - base.edge_attr2.shape[0]
    check(f"support graph grows by exactly 2|E| = {2 * len(edges)}",
          delta == 2 * len(edges), f"got {delta}")

    # candidate entries carry the odd filter g(sigma) at the pair
    B = normalized_biadjacency(ei, nu, ni, normalize=True).toarray()
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    V = Vt.T
    freqcenter = np.linspace(S[0] / 3, S[0], 3)
    r, c = aug.edge_index2
    uv = (r < nu) & (c >= nu)
    rr, cc = r[uv].numpy(), (c[uv] - nu).numpy()
    worst = 0.0
    for j, f_s in enumerate(freqcenter):
        want = np.sum(U[rr, :] * g(S, b=5, f_s=f_s) * V[cc, :], axis=1)
        worst = max(worst, float(np.abs(want - aug.edge_attr2[uv, j].numpy()).max()))
    check("candidate support values equal the odd filter g(sigma)", worst < 1e-6,
          f"max|delta|={worst:.2e}")

    check("identity support stays zero off the diagonal",
          bool(torch.all(aug.edge_attr2[r != c, 3] == 0))
          and int((aug.edge_attr2[r == c, 3] != 0).sum()) == n)

    same = build_design(ei, nu, n, cand_pairs=0.0, **kw)
    check("cand_pairs=0 reproduces the default design bit-for-bit",
          torch.equal(base.edge_attr2, same.edge_attr2)
          and torch.equal(base.edge_index2, same.edge_index2))

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("All candidate-pair checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
