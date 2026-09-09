# GNNML3 link-prediction ablation study — runbook

This is the scripted ablation study for the bipartite GNNML3 link predictor.
Everything is written and locally validated; the compute-heavy matrix is meant
to run on your CUDA server (40 GB VRAM) and then be reported back.

**Question it answers:** which parts of the model are load-bearing for
link prediction (Recall@20 / NDCG@20) and which are decorative?

## TL;DR — the four commands

```bash
# 0) floors the report will quote (~10 s each)
python popularity_baseline.py --dataset gowalla --k-core 0 --topks "[20,50]" \
    --out results/ablations_gowalla/baselines.jsonl

# 1) smoke test (~2 min)
python run_ablation_study.py --only baseline,no_odd,no_emb --seeds 2020 \
    --epochs 2 --eval-every 2 --k-core 20 --device cuda \
    --out-dir results/ablation_smoke

# 2) timing probe (~5-15 min) -- use its s/epoch to pick --epochs in step 3
python run_ablation_study.py --probe --epochs 20 --eval-every 20 \
    --k-core 0 --device cuda --grad-checkpoint \
    --out-dir results/ablations_gowalla

# 3) the main matrix (full Gowalla, 50 runs) + auto-report
python run_ablation_study.py --dataset gowalla --k-core 0 \
    --epochs 1000 --eval-every 50 \
    --device cuda --concurrency 2 --threads 8 \
    --grad-checkpoint --save-models \
    --out-dir results/ablations_gowalla
```

Then send me `results/ablations_gowalla/ablation_summary.md` (plus
`ablation_summary.csv` and `benchmark.jsonl` if convenient) and the probe output.
Optional but recommended extra tier in section 5.

---

## 0. What was added / changed

| file | role |
|---|---|
| `run_ablation_study.py` | **entry point** — variant table, parallel/resume-safe driver, probe mode, auto-report |
| `analyze_ablations.py` | builds `ablation_summary.md/.csv/.json` + figures from the JSONL |
| `popularity_baseline.py` | popularity / random floors on the identical split |
| `ablation_diagnostics.py` | support statistics + inference-time sensitivity (no retraining) |
| `bipartite_experiment.py` | *modified*: ablation flags (`--no-learnedge`, `--nout2`, `--shared-head`, `--no-identity`, `--no-even`, `--no-odd`, `--no-degree`), `--design-cache`, `--design-seed`, final-epoch metrics in the record |
| `bipartite_utils.py` | *modified*: chunked support construction (bounded host RAM) + `nfreq=0` support |

Defaults are unchanged for existing runs; the ablation flags default to the full
model.

---

## 1. Environment

```bash
cd /path/to/bipartite-gnnml3
git submodule update --init                    # gnn-matlang is required
python -m venv .venv && source .venv/bin/activate
pip install torch torch_geometric numpy scipy networkx pandas matplotlib
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` and your GPU name. `--device auto` (the default) then picks CUDA.

---

## 2. Smoke test — ~2 minutes

Verifies the whole pipeline (flags, design cache, driver, report) on the small
k-core-20 graph before spending hours:

```bash
python run_ablation_study.py --only baseline,no_odd,no_emb --seeds 2020 \
    --epochs 2 --eval-every 2 --k-core 20 --device cuda \
    --out-dir results/ablation_smoke
```

Success looks like `Finished 3 runs ... (0 failures)` and three rows in
`results/ablation_smoke/benchmark.jsonl`.

---

## 3. Timing probe — ~5–15 minutes

Measures seconds/epoch for the real full-graph configuration and projects the
whole matrix. **Run this before the main matrix** and use the printed number to
pick the epoch budget.

```bash
python run_ablation_study.py --probe --epochs 20 --eval-every 20 \
    --k-core 0 --device cuda --grad-checkpoint \
    --out-dir results/ablations_gowalla
```

Choose the main-run `--epochs` from the reported s/epoch:

| probed s/epoch | suggested `--epochs` | projected 50-run matrix |
|---|---|---|
| < 0.2 | 2000 | ~4 h |
| 0.2 – 0.5 | 1000 (default) | ~3–5 h |
| 0.5 – 1.0 | 750 | ~5–8 h |
| > 1.0 | 500, or add `--uu-topk 0` tier | ~6–10 h |

(The projection in the probe already accounts for the 7 distinct spectral-design
builds; the other 43 runs load the cached design in seconds.)

---

## 4. Main matrix — full Gowalla (the headline study)

```bash
python run_ablation_study.py \
    --dataset gowalla --k-core 0 \
    --epochs 1000 --eval-every 50 \
    --device cuda --concurrency 2 --threads 8 \
    --grad-checkpoint --save-models \
    --out-dir results/ablations_gowalla
```

* 50 runs: baseline ×5 seeds (2020–2024), 11 core ablations ×3 seeds, 6 secondary
  probes ×2 seeds.
* `--grad-checkpoint` is **recommended here**: the baseline support graph has
  4.37 M edges, and without it the per-edge message tensors need ~29 GB (host
  RAM / VRAM), which OOM-killed a 31 GB machine in local testing. With it the
  peak is ~14.5 GB and numerics are unchanged (verified bit-identical metrics,
  ~25 % slower per epoch).
* Resume-safe: re-running the same command skips finished runs, so an
  interruption (or a longer `--epochs`) is cheap to recover.
* `--concurrency 2` runs two workers on the GPU. Check `nvidia-smi`: if GPU
  utilization is already >90 % at concurrency 1, use `--concurrency 1`.
  With ~14 GB per run, concurrency 2 fits comfortably in 40 GB — but only
  **together with `--grad-checkpoint`** (without it one run alone needs ~29 GB,
  so two would not fit).
* `--save-models` writes baseline checkpoints for the sensitivity diagnostic.
* `--amp` is available (CUDA only) but **off by default** — only enable it if the
  probe showed normal training and you want ~1.3–1.8× speed; verify the first
  few epochs' loss/metrics look sane.

Baseline configuration (identical for every variant except the ablated flag):

```
nfreq=5  dv=5  k=500  recfield=1  uu_topk=30  normalized biadjacency
emb_in=64  layers=3  layer_combine  embed_dim=64
lr=1e-3  decay=1e-4  clip=1.0  full-batch BPR
```

Measured cost of this configuration (single CPU worker, for scale reference —
your GPU should be ~50–200× faster per epoch):

| stage | cost |
|---|---|
| rank-500 truncated SVD (`svds`) | 35 s |
| co-occurrence build (`uu_topk=30`) | ~1 s |
| support build (4.37 M edges × 5 bands) | ~90 s, 2.2 GB peak RSS |
| peak RSS during training | 14.5 GB with `--grad-checkpoint`, ~29 GB without |

All seven distinct spectral designs are built once and cached under
`<out-dir>/design_cache/` (~0.2–1 GB each); the other 43 runs load them in
seconds.

---

## 5. Convergence tier — k-core 20 (recommended, cheap)

The published high-recall numbers (~0.19) came from this smaller graph; running
the same matrix there (much cheaper per epoch, ~380 k support edges) shows
whether the component ranking is stable across graph scale and at a longer
budget. It also fits in a few GB of VRAM, so `--grad-checkpoint` is unnecessary.

```bash
python run_ablation_study.py \
    --dataset gowalla --k-core 20 \
    --epochs 2500 --eval-every 100 \
    --device cuda --concurrency 2 --threads 8 \
    --out-dir results/ablations_kc20
```

---

## 6. Reference floors and mechanism diagnostics

Run the floors **before** the matrix (the auto-report quotes them), ~10 s each:

```bash
python popularity_baseline.py --dataset gowalla --k-core 0  --topks "[20,50]" \
    --out results/ablations_gowalla/baselines.jsonl
python popularity_baseline.py --dataset gowalla --k-core 20 --topks "[20,50]" \
    --out results/ablations_kc20/baselines.jsonl
```

After the matrix (needs `--save-models`):

```bash
# receptive-field statistics of the baseline design (explains why some
# components cannot matter)
python ablation_diagnostics.py --support-stats --k 500 --uu-topk 30 \
    --out results/ablations_gowalla/support_stats.json

# what the *trained* model relies on, zeroing supports at inference only
python ablation_diagnostics.py \
    --checkpoint results/ablations_gowalla/checkpoints/ab_baseline_s2020.pt \
    --out results/ablations_gowalla/inference_sensitivity.json
```

---

## 7. Re-render the report at any time

```bash
python analyze_ablations.py --out-dir results/ablations_gowalla --plots
```

Produces `ablation_summary.md`, `.csv`, `.json`, `ablation_delta_recall.png`,
`ablation_curves.png` (the driver also does this automatically when the matrix
finishes).

---

## 8. What to send back

Paste or attach:

1. `results/ablations_gowalla/ablation_summary.md`  ← the ranked table + verdicts
2. `results/ablations_gowalla/ablation_summary.csv`
3. `results/ablations_gowalla/benchmark.jsonl` (one line per run)
4. `results/ablations_gowalla/history.jsonl` (training curves; optional)
5. The probe output (s/epoch + device name)
6. If you ran the k-core-20 tier: `results/ablations_kc20/ablation_summary.md`

With those I can write the interpretation (which components are important /
irrelevant, effect sizes vs the seed-noise floor, and the mechanism behind each
verdict) and fold it into the thesis's Chapter 5.5 / Chapter 6.

---

## 9. Troubleshooting

| symptom | fix |
|---|---|
| `CUDA out of memory` | keep `--grad-checkpoint` on, drop `--concurrency` to 1, and/or lower `--uu-topk` (the edge count drives memory) |
| host-RAM blow-up during setup | already bounded by chunking (`chunk_elems`), peak ≈ 2.2 GB for this config |
| a run fails, driver continues | see `logs/<run_id>.log`; re-run the same command to retry only the failed runs |
| `--epochs` changed later | resume logic keys on (variant, seed, epochs, k-core, dataset), so the new budget re-runs everything at the new length |
| want fewer runs | `--only baseline,no_emb,no_odd,...` runs a subset; `--seeds 2020 2021` shortens the seed list |
| design cache disk use | ~0.2–1 GB per distinct design under `<out-dir>/design_cache`; safe to delete (rebuilt on demand) |

---

## 10. Variant table (what each run removes)

**Reference** — `baseline`: full model.

**GNNML3 mechanisms** — `no_learnedge` (ML3 learnable edge network → fixed
spectral weights), `no_gate` (multiplicative tanh·tanh branch), `no_spectral`
(nfreq=0: all Gaussian bands removed, only the identity support), `no_identity`
(no self-connection support), `no_even` (even filters on UU/VV edges zeroed →
no user–user/item–item message passing), `no_odd` (odd filters on UV/VU edges
zeroed → no user–item message passing).

**Representation / architecture** — `no_emb` (learnable ID embeddings removed →
structural features only), `no_struct` (indicator+degree features removed →
embeddings only), `no_degree` (log-degree feature removed), `layers1` (depth 3 →
1), `no_combine` (mean readout → last layer), `shared_head` (shared user/item
readout).

**Spectral design** — `nfreq1` (5 bands → 1), `dv0.5` (bandwidth 5 → 0.5),
`k100` (rank 500 → 100), `no_uu` (co-interaction receptive field removed).

**Addition probe (not leave-one-out)** — `uu100` (doubles the co-interaction
receptive field).

---

## 11. Local CPU pilot (context only, not the study)

`results/ablations/` holds a small CPU-only pilot that was run here before the
GPU plan was chosen: Gowalla k-core 20, 500 epochs, 3 baseline seeds
(`final_recall@20 = 0.149 ± 0.007`), with popularity 0.089 and random 0.007 as
floors. It is useful only as a sanity reference for the GPU numbers.
