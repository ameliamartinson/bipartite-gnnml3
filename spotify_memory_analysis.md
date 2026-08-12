# Spotify k-core memory requirements for bipartite-gnnml3

## TL;DR

On the **Spotify Million Playlist** dataset, `bipartite-gnnml3` training VRAM is
dominated by the spectral receptive-field graph (`edge_attr2` plus the per-band
message passing), which scales roughly linearly with the number of **train
edges** surviving k-core filtering.

- **Full dataset: ~560–710 GB VRAM** (will not fit).
- **k-core 30: ~390–490 GB**, **k-core 50: ~270–345 GB** — both far over 80 GB.
- **To fit 80 GB VRAM you need k-core ≈ 95** (with `--emb-in 64`) or **≈ 90**
  (default `--emb-in 0`).

The real first bottleneck is the **CPU-side spectral feature construction**,
which on a 31 GB-RAM machine already fails below about **k-core 84**.

---

## Spotify data scale (full MPD, already converted)

| | users (playlists) | items (tracks) | train edges | test edges |
|---|---|---|---|---|
| full | 1,000,000 | 2,262,292 | 52,373,152 | 13,091,624 |

Average train degree ≈ 52 (users), 23 (items), max playlist length 273.

## Measured k-core profile (train set)

The k-core of this graph is very shallow — **max core number is 106**, then it
is empty.

| k-core | users | items | nodes | train edges |
|---|---:|---:|---:|---:|
| 30 | 541,556 | 118,013 | 659,569 | 37,830,042 |
| 50 | 302,967 | 60,849 | 363,816 | 26,474,561 |
| 70 | 155,175 | 30,142 | 185,317 | 16,070,685 |
| 80 | 106,369 | 20,976 | 127,345 | 11,783,006 |
| 84 | 89,275 | 17,835 | 107,110 | 10,138,744 |
| 88 | 73,478 | 14,973 | 88,451 | 8,548,310 |
| 90 | 66,705 | 13,789 | 80,494 | 7,846,077 |
| 94 | 53,527 | 11,471 | 64,998 | 6,431,873 |
| 96 | 47,063 | 10,279 | 57,342 | 5,710,886 |
| 100 | 35,248 | 8,148 | 43,396 | 4,355,311 |
| 105 | 16,805 | 4,370 | 21,175 | 2,110,434 |

The collapse is steep: k=30 keeps 54% of users, k=100 keeps only 3.5%.

## Estimated memory (default `--nfreq 5 --recfield 1 --layers 3 --k 100`)

The GPU estimate was calibrated on the one recorded full-Gowalla run
(`emb_in=64`: 14,052 MB at 1,027,370 edges, 70,839 nodes) and modeled as
`peak = a·edges + b·nodes`. The CPU setup estimate is
`(800·nodes + 2400·edges)` bytes, driven by the `U[uv_row,:] * g * V[uv_col,:]`
intermediate, which is `edges × k × 8` float64 at `k_svd=100`.

| k-core | VRAM, `emb_in=0` | VRAM, `emb_in=64` | CPU setup RAM (`k_svd=100`) |
|---|---:|---:|---:|
| 30 | 392 GB | 493 GB | 91 GB |
| 50 | 273 GB | 344 GB | 64 GB |
| 70 | 166 GB | 208 GB | 39 GB |
| 80 | 121 GB | 153 GB | 28 GB |
| 84 | 104 GB | 131 GB | 24 GB |
| 88 | 88 GB | 111 GB | 21 GB |
| 90 | 81 GB | 102 GB | 19 GB |
| 94 | 66 GB | 83 GB | 16 GB |
| 96 | 59 GB | 74 GB | 14 GB |
| 100 | 45 GB | 56 GB | 11 GB |

These are estimates, roughly **±20–30%** — the calibration is a single data
point and activation memory may not scale perfectly linearly at 40–100×
Gowalla's size.

## Answer: what k-core fits in 80 GB VRAM

- **`--emb-in 64`** (CF embeddings, the config used in most benchmark runs):
  **k-core ≈ 95** (k=94 ≈ 83 GB, k=96 ≈ 74 GB).
- **`--emb-in 0`** (default, structural features only): **k-core ≈ 90**
  (k=90 ≈ 81 GB, k=92 ≈ 74 GB).

So **~k=95 is the safe answer for 80 GB VRAM** with embeddings on.

## Important caveats

1. **CPU setup is the binding constraint first.** The spectral design runs in
   NumPy/SciPy before anything touches VRAM, and its peak is roughly
   `2.4 GB per million train edges` at `k_svd=100`. On this machine (31 GB RAM,
   ~25 GB usable) that means setup alone OOMs below **~k=84**. If the 80 GB-VRAM
   box has plenty of CPU RAM this relaxes; if not, k-core is not the only limit.

2. **The k-core is brutally aggressive.** To hit 80 GB you are down to
   ~40–67K playlists and ~8–14K tracks (3–7% of users). Max core number is 106,
   so there is almost no headroom above k≈95.

3. **Levers that reduce memory far more than k-core tweaking:**
   - `--nfreq 1` cuts the per-band message-passing memory ~3× (the dominant term
     is `nsup = nfreq + 1` propagations).
   - `--amp` roughly halves GPU activation memory.
   - `--k` (SVD rank) only affects CPU setup, not training VRAM — lower it
     (e.g. `--k 50`) to cut the CPU `edges × k` intermediate.

## Reproducing the k-core profile

```bash
cd /home/amelia/repos/bipartite-gnnml3
.venv/bin/python - <<'PY'
# Core-number decomposition of datasets/spotify/train.txt (users 0..999999,
# items offset by 1_000_000) using a NumPy Batagelj-Zaversnik implementation,
# then count surviving users/items/edges per k.
PY
```

The exact numbers in the profile table came from a single core decomposition of
`datasets/spotify/train.txt`; the max core number observed was 106.
