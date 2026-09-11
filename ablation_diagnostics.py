#!/usr/bin/env python3
"""Diagnostics that explain the ablation results mechanistically.

Two independent measurements, neither of which trains anything:

1. **Support statistics** (``--support-stats``). For a given spectral-design
   configuration, how much of the receptive field does each support actually
   touch, and how large are its entries, split by block (UU / UV / VU / VV)?
   With ``recfield=1`` and no ``uu_topk`` the within-partition blocks contain
   only the diagonal (identity) plus the optional off-diagonal band, so the
   even-filter columns are non-zero on a tiny fraction of edges -- which is the
   mechanism behind a "removing the even filters changes nothing" result.

2. **Inference-time sensitivity** (``--checkpoint``). Load a trained baseline
   checkpoint and measure the metric drop from zeroing one support column (or
   block) at inference, without retraining. This separates "the trained model
   relies on this signal" from "the model can relearn without it" (the retrained
   ablation), which is exactly the distinction a reader needs.

Usage:
    python ablation_diagnostics.py --support-stats --off-diag
    python ablation_diagnostics.py --support-stats --uu-topk 30 --out /tmp/s.json
    python ablation_diagnostics.py --support-stats --uu-topk 30 --cand-pairs 1
    python ablation_diagnostics.py --checkpoint results/ablations/baseline_s2020.pt
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "gnn-matlang"))
sys.path.insert(0, _HERE)

from torch_geometric.data import Data  # noqa: E402

from bipartite_experiment import (  # noqa: E402
    GNNML3LinkPredictor, ablate_supports, count_lines, evaluate, load_edges,
)
from bipartite_utils import BipartiteSpectralDesign  # noqa: E402
from kcore import (  # noqa: E402
    k_core_filter, load_k_core_cache, remap_k_core, save_k_core_cache,
)


def load_split(dataset, k_core):
    dd = os.path.join(_HERE, "datasets", dataset)
    nu = count_lines(f"{dd}/user_list.txt") - 1
    ni = count_lines(f"{dd}/item_list.txt") - 1
    tr_e, _ = load_edges(f"{dd}/train.txt")
    te_e, _ = load_edges(f"{dd}/test.txt")
    if k_core > 0:
        train_path, test_path = f"{dd}/train.txt", f"{dd}/test.txt"
        cached = load_k_core_cache(train_path, test_path, k_core)
        if cached is not None:
            tr_e, te_e, nu, ni = cached
        else:
            tr_e = k_core_filter(tr_e, k_core)
            tr_e, te_e, nu, ni = remap_k_core(tr_e, te_e)
            save_k_core_cache(train_path, test_path, k_core, (tr_e, te_e, nu, ni))
    return tr_e, te_e, nu, ni


def build_data(tr_e, nu, ni):
    el = []
    for u, i in tr_e:
        el.append([u, nu + i])
        el.append([nu + i, u])
    ei = torch.tensor(el, dtype=torch.int64).T
    x = torch.zeros(nu + ni, 2)
    x[:nu, 0] = 1.0
    x[nu:, 1] = 1.0
    return Data(edge_index=ei, x=x, y=torch.tensor([0]))


def offdiag_zero_descriptor_fraction(edge_index2, edge_attr2):
    """Fraction of off-diagonal support rows whose descriptor is all zeros.

    The descriptor of a support entry is the row of ``edge_attr2`` that
    ``ML3Layer``'s edge network consumes (``nsup`` values: the spectral bands,
    the identity support, and optionally the adjacency support). Every ML3 edge
    MLP is bias-free, so a zero descriptor gives ``F(0) = 0`` and a zero message
    weight. A fraction of 1.0 therefore means no message passing happens at all
    on off-diagonal entries -- exactly the degeneracy of ``--nfreq 0`` (whose
    only column is the identity, zero everywhere off the diagonal).
    """
    ei = edge_index2 if isinstance(edge_index2, torch.Tensor) \
        else torch.as_tensor(edge_index2)
    ea = edge_attr2 if isinstance(edge_attr2, torch.Tensor) \
        else torch.as_tensor(edge_attr2)
    offdiag = ei[0] != ei[1]
    n_offdiag = int(offdiag.sum())
    if n_offdiag == 0:
        return 0.0
    zero = ea.abs().sum(dim=1) == 0
    return float((offdiag & zero).sum()) / n_offdiag


def support_stats(args):
    tr_e, _, nu, ni = load_split(args.dataset, args.k_core)
    data = build_data(tr_e, nu, ni)
    tf = BipartiteSpectralDesign(
        nu, nfreq=args.nfreq, dv=args.dv, k=args.k, recfield=args.recfield,
        adddegree=True, nmax=0, seed=args.seed,
        normalize_biadj=not args.raw_biadj, uu_topk=args.uu_topk,
        off_diag=args.off_diag, cand_pairs=args.cand_pairs,
        flat_support=args.flat_support,
        flat_support_value=args.flat_support_value,
        shuffle_bands=args.shuffle_bands)
    data = tf(data)

    ea = data.edge_attr2
    ei = data.edge_index2
    n_edges = ea.shape[0]
    is_user = (ei[0] < nu).numpy()
    col_user = (ei[1] < nu).numpy()
    block = np.where(is_user & col_user, "UU",
                     np.where(is_user & ~col_user, "UV",
                              np.where(~is_user & col_user, "VU", "VV")))
    identity_col = args.nfreq

    # Mark which cross-partition support entries are actual observed edges vs
    # candidate (non-edge) pairs contributed by the augmented mask A + I + P.
    obs = data.edge_index[0].numpy().astype(np.int64) * (nu + ni) \
        + data.edge_index[1].numpy().astype(np.int64)
    obs = np.unique(obs)
    here = ei[0].numpy().astype(np.int64) * (nu + ni) + ei[1].numpy().astype(np.int64)
    pos = np.searchsorted(obs, here)
    np.clip(pos, 0, obs.size - 1, out=pos)
    is_observed = obs[pos] == here

    out = {
        "config": dict(dataset=args.dataset, k_core=args.k_core, nu=nu, ni=ni,
                       n_support_edges=int(n_edges), nfreq=args.nfreq,
                       dv=args.dv, k_svd=args.k, recfield=args.recfield,
                       uu_topk=args.uu_topk, off_diag=bool(args.off_diag),
                       cand_pairs=float(args.cand_pairs),
                       flat_support=bool(args.flat_support),
                       shuffle_bands=bool(args.shuffle_bands),
                       biadj="raw" if args.raw_biadj else "normalized"),
        "blocks": {},
        "columns": [],
    }
    # Degeneracy check: with nfreq=0 every off-diagonal descriptor is zero, so
    # (bias-free edge MLPs => F(0)=0) no message passes. See the helper.
    out["offdiag_zero_descriptor_fraction"] = round(
        offdiag_zero_descriptor_fraction(ei, ea), 6)
    for b in ("UU", "UV", "VU", "VV"):
        m = block == b
        # P is defined on user-item non-edges, so only cross-partition entries
        # can be candidate pairs; UU/VV mask entries (diagonal, co-interaction
        # edges) are not "non-edges" in that sense.
        cand = m & ~is_observed & (is_user != col_user)
        out["blocks"][b] = dict(
            n_edges=int(m.sum()),
            frac_of_support=round(float(m.mean()), 6),
            n_band_nonzero=int((ea[m, :identity_col].abs() > 0).any(1).sum()),
            frac_band_nonzero=round(
                float((ea[m, :identity_col].abs() > 0).any(1).float().mean())
                if m.sum() else 0.0, 6),
            mean_abs_band=round(float(ea[m, :identity_col].abs().mean())
                                if m.sum() else 0.0, 6),
            identity_nonzero=int((ea[m, identity_col] != 0).sum()),
            n_candidate_pairs=int(cand.sum()),
            mean_abs_band_candidate=round(
                float(ea[cand, :identity_col].abs().mean())
                if cand.sum() else 0.0, 6),
        )
    for j in range(ea.shape[1]):
        col = ea[:, j]
        nz = col != 0
        out["columns"].append(dict(
            index=j,
            kind="identity" if j == identity_col else f"band_{j}",
            n_nonzero=int(nz.sum()),
            frac_nonzero=round(float(nz.float().mean()), 6),
            mean_abs=round(float(col.abs().mean()), 6),
            max_abs=round(float(col.abs().max()), 6),
        ))

    print(f"\nSupport statistics for {args.dataset} (k-core {args.k_core}): "
          f"{nu} users, {ni} items, {n_edges:,} support edges")
    print(f"  supports: {args.nfreq} bands + identity"
          f" | cand_pairs={args.cand_pairs} (M' = A + I + P)"
          f" | flat_support={args.flat_support}"
          f" | shuffle_bands={args.shuffle_bands}")
    print(f"  {'block':6s} {'#edges':>10s} {'%ofsup':>8s} {'band nz':>10s} "
          f"{'band nz%':>9s} {'mean|band|':>11s} {'ident nz':>9s} "
          f"{'#cand':>10s} {'mean|cand|':>11s}")
    for b, s in out["blocks"].items():
        print(f"  {b:6s} {s['n_edges']:10,d} {100*s['frac_of_support']:7.2f}% "
              f"{s['n_band_nonzero']:10,d} {100*s['frac_band_nonzero']:8.2f}% "
              f"{s['mean_abs_band']:11.5f} {s['identity_nonzero']:9,d} "
              f"{s['n_candidate_pairs']:10,d} "
              f"{s['mean_abs_band_candidate']:11.5f}")
    print(f"  {'column':>8s} {'kind':>10s} {'#nonzero':>10s} {'%':>7s} "
          f"{'mean|.|':>9s} {'max|.|':>9s}")
    for c in out["columns"]:
        print(f"  {c['index']:8d} {c['kind']:>10s} {c['n_nonzero']:10,d} "
              f"{100*c['frac_nonzero']:6.2f}% {c['mean_abs']:9.5f} "
              f"{c['max_abs']:9.5f}")

    if args.nfreq > 0:
        band_abs = ea[:, :args.nfreq].abs()
        cross = is_user != col_user
        suggest = float(band_abs[cross].mean()) if cross.any() \
            else float(band_abs.mean())
        out["suggested_flat_support_value"] = round(suggest, 8)
        print(f"  suggested --flat-support-value (mean |band| on cross-partition "
              f"edges): {suggest:.6f}   [overall mean |band|: "
              f"{float(band_abs.mean()):.6f}]")

    frac_zero = out["offdiag_zero_descriptor_fraction"]
    print(f"  off-diagonal support rows with an all-zero descriptor: "
          f"{100 * frac_zero:.2f}%")
    if frac_zero > 0:
        msg = (
            f"DEGENERATE DESIGN: {100 * frac_zero:.2f}% of off-diagonal support "
            f"rows have an all-zero edge descriptor (nfreq={args.nfreq}, "
            f"flat_support={args.flat_support}, shuffle_bands="
            f"{args.shuffle_bands}). The ML3 edge MLPs are bias-free, so F(0)=0 "
            f"and those edges carry zero message weight: this configuration "
            f"removes message passing rather than ablating spectral selectivity.")
        print("\n" + "!" * 78)
        print("!! WARNING: " + msg)
        print("!" * 78 + "\n")
        warnings.warn(msg, stacklevel=2)
    return out


def inference_sensitivity(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    dataset = cfg.get("dataset", args.dataset)
    k_core = cfg.get("k_core", args.k_core)
    tr_e, te_e, nu, ni = load_split(dataset, k_core)
    data = build_data(tr_e, nu, ni)
    # Rebuild the *exact* design the checkpoint was trained with: the design
    # seed (not the training seed) drives the truncated-SVD start vector and the
    # sampled candidate pairs P, so using the wrong one would evaluate the model
    # on a different mask.
    design_seed = cfg.get("design_seed", -1)
    if design_seed is None or design_seed < 0:
        design_seed = cfg.get("seed", 2020)
    tf = BipartiteSpectralDesign(
        nu, nfreq=cfg.get("nfreq", 5), dv=cfg.get("dv", 5),
        k=cfg.get("k", 100), recfield=cfg.get("recfield", 1),
        adddegree=not cfg.get("no_degree", False), nmax=0,
        seed=design_seed,
        normalize_biadj=not cfg.get("raw_biadj", False),
        uu_topk=cfg.get("uu_topk", 0), off_diag=cfg.get("off_diag", False),
        cand_pairs=cfg.get("cand_pairs", 0.0),
        flat_support=cfg.get("flat_support", False),
        flat_support_value=cfg.get("flat_support_value", 1.0),
        shuffle_bands=cfg.get("shuffle_bands", False))
    data = tf(data)
    tr_ui, te_ui = {}, {}
    for u, i in tr_e:
        tr_ui.setdefault(u, set()).add(i)
    for u, i in te_e:
        te_ui.setdefault(u, set()).add(i)

    model = GNNML3LinkPredictor(
        data.x.shape[1], data.edge_attr2.shape[1], nu,
        num_nodes=data.x.shape[0], emb_in=cfg.get("emb_in", 0),
        n_layers=cfg.get("layers", 3), nout2=cfg.get("nout2", 32),
        embed_dim=cfg.get("embed_dim", 64),
        use_struct_feats=not cfg.get("no_struct_feats", False),
        layer_combine=cfg.get("layer_combine", False),
        learnedge=not cfg.get("no_learnedge", False),
        shared_head=cfg.get("shared_head", False))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    nfreq = cfg.get("nfreq", 5)
    ks = eval(cfg.get("topks", "[20]"))
    base = evaluate(model, data, te_ui, tr_ui, ks=ks)
    print(f"\nInference-time sensitivity (no retraining) on {dataset}:")
    print(f"  reference: " + "  ".join(
        f"{m}@{ks[0]}={base[f'{m}@{ks[0]}']:.4f}" for m in ("recall", "ndcg")))

    rows = []
    probes = [("identity", dict(drop_identity=True)),
              ("even(UU/VV bands)", dict(drop_even=True)),
              ("odd(UV/VU bands)", dict(drop_odd=True))]
    for name, kw in probes:
        d = data.clone()
        ablate_supports(d, nu, nfreq, **kw)
        rec = evaluate(model, d, te_ui, tr_ui, ks=ks)
        delta = rec[f"recall@{ks[0]}"] - base[f"recall@{ks[0]}"]
        rows.append(dict(probe=name, recall=rec[f"recall@{ks[0]}"],
                         ndcg=rec[f"ndcg@{ks[0]}"], delta_recall=delta))
        print(f"  zero {name:20s} recall@{ks[0]}={rec[f'recall@{ks[0]}']:.4f} "
              f"(Δ {delta:+.4f})")
    # all spectral bands at once (identity kept)
    d = data.clone()
    if nfreq > 0:
        d.edge_attr2[:, :nfreq] = 0.0
    rec = evaluate(model, d, te_ui, tr_ui, ks=ks)
    delta = rec[f"recall@{ks[0]}"] - base[f"recall@{ks[0]}"]
    rows.append(dict(probe="all bands", recall=rec[f"recall@{ks[0]}"],
                     ndcg=rec[f"ndcg@{ks[0]}"], delta_recall=delta))
    print(f"  zero {'all bands':20s} recall@{ks[0]}={rec[f'recall@{ks[0]}']:.4f} "
          f"(Δ {delta:+.4f})")
    return dict(reference=base, probes=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gowalla")
    ap.add_argument("--k-core", type=int, default=20)
    ap.add_argument("--nfreq", type=int, default=5)
    ap.add_argument("--dv", type=float, default=5)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--recfield", type=int, default=1)
    ap.add_argument("--uu-topk", type=int, default=0)
    ap.add_argument("--off-diag", action="store_true")
    ap.add_argument("--cand-pairs", type=float, nargs="?", const=1.0,
                    default=0.0, metavar="R",
                    help="augmented mask M' = A + I + P: sample R * |E_train| "
                    "user-item non-edges into the receptive field (0 = off)")
    ap.add_argument("--flat-support", action="store_true",
                    help="spectral-selectivity control: every band column is "
                    "the constant 1.0 (no-op with --nfreq 0)")
    ap.add_argument("--flat-support-value", type=float, default=1.0,
                    metavar="C",
                    help="constant written into the band columns by "
                    "--flat-support (default 1.0)")
    ap.add_argument("--shuffle-bands", action="store_true",
                    help="spectral-selectivity control: filters evaluated at a "
                    "seeded permutation of the singular values (no-op with "
                    "--nfreq 0)")
    ap.add_argument("--raw-biadj", action="store_true")
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--support-stats", action="store_true")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    result = {}
    if args.support_stats or not args.checkpoint:
        result["support_stats"] = support_stats(args)
    if args.checkpoint:
        result["inference_sensitivity"] = inference_sensitivity(args)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
