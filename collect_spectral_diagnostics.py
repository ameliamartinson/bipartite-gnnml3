#!/usr/bin/env python3
"""Collect the spectral-diagnosis data for the thesis figures.

Computes, per dataset, everything Diagnoses I & II need, without training:

  * top-K singular values of the (raw and normalized) biadjacency
  * exact total Frobenius energy and the cumulative energy retained at each
    truncation rank k  (Diagnosis II: information loss from truncation)
  * per-band even/odd Gaussian filter responses h(S), g(S) at the same band
    centers the model uses, plus a per-band effective occupancy
    (Diagnosis I: degenerate / redundant frequency bands)

Writes one JSON per dataset plus a flat CSV of the energy curve, so the
thesis figures can be regenerated from files instead of a notebook session.

Usage:
  python collect_spectral_diagnostics.py --dataset gowalla --k 400
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from bipartite_utils import normalized_biadjacency, _svds_seeded, h, g

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_train_edge_index(dataset):
    """Build a (2, E) user->item edge_index (item ids offset by num_users)."""
    dd = os.path.join(_HERE, "datasets", dataset)
    with open(os.path.join(dd, "user_list.txt")) as f:
        num_users = sum(1 for _ in f) - 1
    with open(os.path.join(dd, "item_list.txt")) as f:
        num_items = sum(1 for _ in f) - 1
    us, vs = [], []
    with open(os.path.join(dd, "train.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            for i in parts[1:]:
                us.append(u)
                vs.append(num_users + int(i))
    ei = torch.tensor([us, vs], dtype=torch.int64)
    return ei, num_users, num_items


def analyze(B, k, seed, nfreqs, dv):
    """Spectrum, energy curve, and band responses for one biadjacency."""
    total_energy = float((B.data.astype(np.float64) ** 2).sum())  # ||B||_F^2
    k = min(k, min(B.shape) - 1)
    U, S, Vt = _svds_seeded(B, k, seed)
    S = np.sort(S)[::-1]

    cum = np.cumsum(S.astype(np.float64) ** 2)
    energy_retained = cum / total_energy

    bands = {}
    lambda_max = float(S[0])
    for nfreq in nfreqs:
        centers = np.linspace(lambda_max / nfreq, lambda_max, nfreq)
        per_band = []
        for f_s in centers:
            h_S = h(S, b=dv, f_s=f_s)
            g_S = g(S, b=dv, f_s=f_s)
            per_band.append({
                "center": float(f_s),
                "h_response": [float(x) for x in h_S],
                "g_response": [float(x) for x in g_S],
                # modes this band actually "sees" (response above 1% of peak)
                "h_occupancy": int((h_S > 0.01 * h_S.max()).sum()) if h_S.max() > 0 else 0,
                "g_occupancy": int((g_S > 0.01 * np.abs(g_S).max()).sum()) if np.abs(g_S).max() > 0 else 0,
            })
        bands[str(nfreq)] = {"centers": [float(c) for c in centers], "bands": per_band}

    return {
        "shape": [int(B.shape[0]), int(B.shape[1])],
        "nnz": int(B.nnz),
        "total_frobenius_energy": total_energy,
        "k_computed": int(k),
        "singular_values": [float(s) for s in S],
        "energy_retained_at_rank": [float(e) for e in energy_retained],
        "band_responses": bands,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gowalla",
                    choices=["amazon-book", "gowalla", "yelp2018"])
    ap.add_argument("--k", type=int, default=400,
                    help="how many singular values to compute (>= the largest "
                    "k_svd used in any experiment, so every run's truncation "
                    "point lies on the curve)")
    ap.add_argument("--nfreqs", default="[2,5,8]",
                    help="Python-literal list of band counts to analyze")
    ap.add_argument("--dv", type=float, default=5, help="Gaussian bandwidth b")
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "results", "diagnostics"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nfreqs = list(eval(args.nfreqs))

    print(f"Loading {args.dataset}...")
    ei, nu, ni = load_train_edge_index(args.dataset)
    print(f"  users={nu:,} items={ni:,} train edges={ei.shape[1]:,}")

    out = {"dataset": args.dataset, "num_users": nu, "num_items": ni,
           "dv": args.dv, "seed": args.seed}
    for kind, normalize in [("normalized", True), ("raw", False)]:
        print(f"SVD of {kind} biadjacency (k={args.k})...")
        B = normalized_biadjacency(ei, nu, ni, normalize=normalize)
        out[kind] = analyze(B, args.k, args.seed, nfreqs, args.dv)
        er = out[kind]["energy_retained_at_rank"]
        for probe in (50, 100, 200, args.k):
            if probe <= len(er):
                print(f"  energy retained at rank {probe}: {er[probe - 1]:.4f}")

    jpath = os.path.join(args.out_dir, f"{args.dataset}_spectrum.json")
    with open(jpath, "w") as f:
        json.dump(out, f)
    print(f"Wrote {jpath}")

    cpath = os.path.join(args.out_dir, f"{args.dataset}_energy_curve.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "sigma_normalized", "energy_retained_normalized",
                    "sigma_raw", "energy_retained_raw"])
        n = max(out["normalized"]["k_computed"], out["raw"]["k_computed"])
        for r in range(n):
            row = [r + 1]
            for kind in ("normalized", "raw"):
                sv = out[kind]["singular_values"]
                er = out[kind]["energy_retained_at_rank"]
                row += [sv[r] if r < len(sv) else "",
                        er[r] if r < len(er) else ""]
            w.writerow(row)
    print(f"Wrote {cpath}")


if __name__ == "__main__":
    main()
