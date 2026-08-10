"""
Convert the Spotify Million Playlist Dataset (MPD) JSON slices into the
standard user-item .txt format used by both runners (bipartite_experiment.py
and run_lightgcn.py / the LightGCN dataloader).

Each MPD slice file holds 1000 playlists. Playlists become users, unique
tracks become items. Per playlist, a random fraction of tracks is held out
as the test split (playlists with a single track stay entirely in train).

Outputs:
    datasets/spotify/{train,test,user_list,item_list}.txt   (GNNML3 runner)
    LightGCN-PyTorch/data/spotify/{train,test}.txt          (LightGCN runner)

Because both runners read the same generated files, --k-core filtering via
kcore.py applies identically to the converted data.

Usage:
    python convert_spotify.py --slices 10        # first 10 slices = 10k playlists
    python convert_spotify.py --slices 1000      # full MPD
"""

import argparse
import glob
import json
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MPD_DIR = os.path.join(_HERE, "datasets", "spotify_million_playlist_dataset", "data")
OUT_DIR = os.path.join(_HERE, "datasets", "spotify")
LGN_DIR = os.path.join(_HERE, "LightGCN-PyTorch", "data", "spotify")


def slice_key(path):
    """Sort slices numerically by their playlist range (mpd.slice.0-999.json)."""
    m = re.search(r"mpd\.slice\.(\d+)-(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--slices",
        type=int,
        default=10,
        help="number of MPD JSON slice files to load, 1000 playlists each "
        "(slices are taken in playlist-range order; max 1000 = full MPD)",
    )
    ap.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="fraction of each playlist's tracks held out for testing "
        "(playlists with fewer than 2 tracks contribute train only)",
    )
    ap.add_argument("--seed", type=int, default=2020)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(MPD_DIR, "mpd.slice.*.json")), key=slice_key)
    if not files:
        raise SystemExit(f"error: no MPD slices found in {MPD_DIR}")
    if args.slices > len(files):
        raise SystemExit(
            f"error: --slices {args.slices} but only {len(files)} slices exist"
        )
    files = files[: args.slices]
    print(f"Loading {len(files)} slice(s) from {MPD_DIR} ...")

    # ── read slices: playlists -> users, track_uris -> items ──
    rng = np.random.RandomState(args.seed)
    item2idx = {}
    playlists = []  # per playlist: ordered list of unique item ids
    for fi, path in enumerate(files):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pl in data["playlists"]:
            seen = set()
            tracks = []
            for t in pl["tracks"]:
                uri = t["track_uri"]
                if uri in seen:
                    continue
                seen.add(uri)
                if uri not in item2idx:
                    item2idx[uri] = len(item2idx)
                tracks.append(item2idx[uri])
            if tracks:  # skip empty playlists
                playlists.append((pl["pid"], tracks))
        if (fi + 1) % 50 == 0 or fi == len(files) - 1:
            print(f"  {fi + 1}/{len(files)} slices, {len(item2idx):,} unique tracks")

    # Contiguous user ids over non-empty playlists (pid order).
    playlists.sort(key=lambda x: x[0])
    user2idx = {pid: i for i, (pid, _) in enumerate(playlists)}
    num_users = len(playlists)
    num_items = len(item2idx)

    # ── per-playlist train/test split ──
    train_ui = [[] for _ in range(num_users)]
    test_ui = {}
    for pid, tracks in playlists:
        u = user2idx[pid]
        if len(tracks) < 2:
            train_ui[u] = tracks
            continue
        n_test = max(1, int(round(len(tracks) * args.test_ratio)))
        n_test = min(n_test, len(tracks) - 1)  # always keep >= 1 train item
        test_idx = set(rng.choice(len(tracks), size=n_test, replace=False).tolist())
        for j, it in enumerate(tracks):
            if j in test_idx:
                test_ui.setdefault(u, []).append(it)
            else:
                train_ui[u].append(it)

    n_train = sum(len(v) for v in train_ui)
    n_test = sum(len(v) for v in test_ui.values())
    print(
        f"Playlists (users): {num_users:,}  Tracks (items): {num_items:,}  "
        f"Interactions: {n_train:,} train / {n_test:,} test"
    )

    # ── write the standard-format files for both runners ──
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LGN_DIR, exist_ok=True)

    train_path = os.path.join(OUT_DIR, "train.txt")
    test_path = os.path.join(OUT_DIR, "test.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        for u in range(num_users):
            f.write(" ".join([str(u)] + [str(i) for i in train_ui[u]]) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        for u in sorted(test_ui):
            f.write(" ".join([str(u)] + [str(i) for i in test_ui[u]]) + "\n")

    # user_list.txt / item_list.txt: header line + "org_id remap_id" pairs
    # (bipartite_experiment.py derives the counts from the line totals).
    with open(os.path.join(OUT_DIR, "user_list.txt"), "w", encoding="utf-8") as f:
        f.write("org_id remap_id\n")
        for pid, u in sorted(user2idx.items(), key=lambda x: x[1]):
            f.write(f"{pid} {u}\n")
    with open(os.path.join(OUT_DIR, "item_list.txt"), "w", encoding="utf-8") as f:
        f.write("org_id remap_id\n")
        for uri, i in sorted(item2idx.items(), key=lambda x: x[1]):
            f.write(f"{uri} {i}\n")

    # Same train/test for the LightGCN dataloader. Remove any cached normalized
    # adjacency so a re-conversion with different --slices is never mixed with
    # a stale graph (the dataloader only caches when k-core is off).
    for name in ("train.txt", "test.txt"):
        with open(os.path.join(OUT_DIR, name), "r", encoding="utf-8") as fin, open(
            os.path.join(LGN_DIR, name), "w", encoding="utf-8"
        ) as fout:
            fout.write(fin.read())
    cache = os.path.join(LGN_DIR, "s_pre_adj_mat.npz")
    if os.path.exists(cache):
        os.remove(cache)
        print(f"Removed stale LightGCN cache {cache}")

    print(f"Wrote {OUT_DIR}/{{train,test,user_list,item_list}}.txt")
    print(f"Wrote {LGN_DIR}/{{train,test}}.txt")
    print(
        f"Done. Run with: python bipartite_experiment.py --dataset spotify "
        f"[--k-core K]  or  python run_lightgcn.py --dataset spotify [--k-core K]"
    )


if __name__ == "__main__":
    main()
