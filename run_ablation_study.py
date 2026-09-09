#!/usr/bin/env python3
"""Run the GNNML3 link-prediction ablation matrix.

Every variant is the full model with exactly one mechanism disabled (or, for
the two "add" probes, one mechanism added). All variants share the identical
training/evaluation code path in ``bipartite_experiment.py`` -- they differ only
by CLI flags -- so a variant's metric difference against the baseline is
attributable to the flag, not to a divergent implementation.

The driver is resume-safe: a (ablation, seed, epochs, k_core, dataset) tuple
already present in the output JSONL is skipped, so an interrupted run can be
restarted (or extended with more seeds) without recomputing finished runs.

Examples:
    # pilot: baseline seeds only
    python run_ablation_study.py --only baseline --seeds 2020 2021 2022
    # full matrix
    python run_ablation_study.py --seeds 2020 2021 2022 --concurrency 4
    # just re-render the report
    python run_ablation_study.py --aggregate-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

from bench_utils import read_jsonl

_HERE = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────────────────
#  Variant table
#
#  Each entry changes exactly one thing relative to ``baseline``. The "group"
#  field is only for reporting; "role" marks the two addition probes, which are
#  not leave-one-out ablations and must not be ranked with them.
# ──────────────────────────────────────────────────────────────────────────
VARIANTS = [
    dict(tag="baseline", group="reference", role="baseline",
         note="full model: learned edge features, gating branch, 5 spectral "
              "bands, identity + even + odd supports, co-interaction receptive "
              "field, embeddings + structural features, 3 layers, mean readout",
         args=[], seeds=[2020, 2021, 2022, 2023, 2024]),

    # ── GNNML3-specific mechanisms ───────────────────────────────────────
    dict(tag="no_learnedge", group="gnnml3", role="ablation",
         note="ML3Layer's learnable edge network removed (fixed spectral "
              "supports used as edge weights)",
         args=["--no-learnedge"]),
    dict(tag="no_gate", group="gnnml3", role="ablation",
         note="multiplicative tanh*tanh gating branch removed (nout2=0)",
         args=["--nout2", "0"]),
    dict(tag="no_spectral", group="gnnml3", role="ablation",
         note="all Gaussian spectral bands removed (nfreq=0); only the identity "
              "support remains, so the conv is a learned aggregation",
         args=["--nfreq", "0"]),
    dict(tag="no_identity", group="gnnml3", role="ablation",
         note="identity support zeroed (no self-connection in the conv)",
         args=["--no-identity"]),
    dict(tag="no_even", group="gnnml3", role="ablation",
         note="even spectral filters on within-partition (UU/VV) edges zeroed: "
              "no user-user / item-item message passing",
         args=["--no-even"]),
    dict(tag="no_odd", group="gnnml3", role="ablation",
         note="odd spectral filters on cross-partition (UV/VU) edges zeroed: "
              "no user-item message passing",
         args=["--no-odd"]),

    # ── representation / architecture ────────────────────────────────────
    dict(tag="no_emb", group="representation", role="ablation",
         note="learnable node-embedding table removed (structural features only)",
         args=["--emb-in", "0"]),
    dict(tag="no_struct", group="representation", role="ablation",
         note="structural features (indicator + log-degree) dropped; embeddings "
              "only",
         args=["--no-struct-feats"]),
    dict(tag="no_degree", group="representation", role="ablation",
         note="log-degree structural feature dropped (indicator columns only)",
         args=["--no-degree"]),
    dict(tag="layers1", group="architecture", role="ablation",
         note="single ML3 layer instead of 3",
         args=["--layers", "1"]),
    dict(tag="no_combine", group="architecture", role="ablation",
         note="last-layer readout instead of the mean over layers",
         args=["--layer-combine-off"]),
    dict(tag="shared_head", group="architecture", role="ablation",
         note="one readout projection shared by users and items",
         args=["--shared-head"], seeds=[2020, 2021]),

    # ── spectral design hyperparameters ──────────────────────────────────
    dict(tag="nfreq1", group="spectral", role="ablation",
         note="one frequency band instead of five",
         args=["--nfreq", "1"], seeds=[2020, 2021]),
    dict(tag="dv0.5", group="spectral", role="ablation",
         note="narrow Gaussian bandwidth (dv=0.5) instead of dv=5",
         args=["--dv", "0.5"], seeds=[2020, 2021]),
    dict(tag="k100", group="spectral", role="ablation",
         note="rank-100 truncated SVD instead of rank-500",
         args=["--k", "100"], seeds=[2020, 2021]),
    dict(tag="no_uu", group="spectral", role="ablation",
         note="sparsified within-partition co-interaction edges removed "
              "(uu_topk=0): the even filters see only the identity diagonal",
         args=["--uu-topk-off"], seeds=[2020, 2021]),

    # ── addition probes (not leave-one-out) ──────────────────────────────
    dict(tag="uu100", group="addition", role="addition",
         note="doubles the co-interaction receptive field (uu_topk=100); "
              "~2.3x more support edges than the baseline",
         args=["--uu-topk", "100"], seeds=[2020, 2021]),
]

# Flags that are part of the fixed protocol for every run in the study. Variants
# that must drop one of these use a negative switch (--uu-topk-off,
# --layer-combine-off) which is translated in variant_cmd().
#
# Defaults target the full Gowalla graph on a CUDA GPU: rank-500 truncated SVD
# (cheap to build, ~35 s), 5 spectral bands, and 30 sparsified co-interaction
# edges per node so the even filters have real within-partition structure.
BASE_FLAGS = [
    "--nfreq", "5",
    "--dv", "5",
    "--k", "500",
    "--uu-topk", "30",
    "--emb-in", "64",
    "--layer-combine",
    "--layers", "3",
]


def variant_cmd(py, args, tag, seed, run_id, out_jsonl, history_jsonl,
                dataset, k_core, epochs, eval_every, topks, device, amp=False,
                design_cache="", design_seed=2020, save_model="",
                grad_checkpoint=False):
    """Assemble the argv for one (variant, seed) run."""
    base = list(BASE_FLAGS)
    if "--uu-topk-off" in args:
        # translate the negative switch into "no uu_topk flag at all"
        args = [a for a in args if a != "--uu-topk-off"]
        i = base.index("--uu-topk")
        del base[i:i + 2]
    if "--layer-combine-off" in args:
        args = [a for a in args if a != "--layer-combine-off"]
        base = [a for a in base if a != "--layer-combine"]
    cmd = [
        py, os.path.join(_HERE, "bipartite_experiment.py"),
        "--dataset", dataset,
        "--k-core", str(k_core),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--eval-every", str(eval_every),
        "--topks", topks,
        "--device", device,
        "--run-id", run_id,
        "--ablation", tag,
        "--out", out_jsonl,
        "--history-out", history_jsonl,
        *base, *args,
    ]
    if design_cache:
        cmd += ["--design-cache", design_cache, "--design-seed", str(design_seed)]
    if amp:
        cmd += ["--amp"]
    if save_model:
        cmd += ["--save-model", save_model]
    if grad_checkpoint:
        cmd += ["--grad-checkpoint"]
    return cmd


def already_done(rows, key):
    return any(
        (r.get("ablation") == key[0] and r.get("seed") == key[1]
         and r.get("epochs") == key[2] and r.get("k_core") == key[3]
         and r.get("dataset") == key[4] and r.get("model") == "gnnml3")
        for r in rows
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gowalla")
    ap.add_argument("--k-core", type=int, default=0,
                    help="0 = full graph (the standard benchmark split)")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2020, 2021, 2022])
    ap.add_argument("--topks", default="[20]")
    ap.add_argument("--device", default="auto",
                    help="'auto' picks CUDA when available")
    ap.add_argument("--amp", action="store_true",
                    help="automatic mixed precision (CUDA only)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel worker processes; on one GPU 1-3 is typical")
    ap.add_argument("--threads", type=int, default=8,
                    help="OMP/MKL threads per worker process")
    ap.add_argument("--design-cache", default="",
                    help="spectral-design cache dir (default: <out-dir>/design_cache)")
    ap.add_argument("--design-seed", type=int, default=2020,
                    help="fixed design seed so all runs share one cached design")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "results", "ablations"))
    ap.add_argument("--only", default="",
                    help="comma-separated variant tags to run (default: all)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="run one baseline for --epochs and print the measured "
                    "seconds/epoch plus the projected time of the full matrix")
    ap.add_argument("--save-models", action="store_true",
                    help="save a checkpoint for each baseline run (needed by "
                    "ablation_diagnostics.py --checkpoint)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute layer activations in backward: ~2x lower "
                    "peak memory for ~25%% more time. Recommended whenever the "
                    "support graph has millions of edges (full Gowalla with "
                    "uu_topk>0) or VRAM is tight")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the analysis/report step at the end")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    if not args.design_cache:
        args.design_cache = os.path.join(out_dir, "design_cache")
    out_dir = os.path.abspath(args.out_dir)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "benchmark.jsonl")
    history_jsonl = os.path.join(out_dir, "history.jsonl")
    py = sys.executable

    if args.aggregate_only:
        return run_report(out_dir)

    wanted = set(t.strip() for t in args.only.split(",") if t.strip())
    variants = [v for v in VARIANTS if not wanted or v["tag"] in wanted]
    if wanted:
        missing = wanted - {v["tag"] for v in variants}
        if missing:
            raise SystemExit(f"unknown variant tag(s): {sorted(missing)}")

    existing = read_jsonl(out_jsonl)
    queue = deque()
    for v in variants:
        # A variant may cap its own seed list (cheaper secondary probes) via
        # ``seeds=[...]``; otherwise it runs the study's full seed list.
        seeds = v.get("seeds") or args.seeds
        for seed in seeds:
            run_id = f"ab_{v['tag']}_s{seed}"
            key = (v["tag"], seed, args.epochs, args.k_core, args.dataset)
            if already_done(existing, key):
                print(f"SKIP  {run_id} (already in {os.path.basename(out_jsonl)})")
                continue
            queue.append((v, seed, run_id))

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["MPLCONFIGDIR"] = os.path.join(out_dir, ".mplconfig")
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)

    print(f"Ablation study: dataset={args.dataset} k-core={args.k_core} "
          f"epochs={args.epochs} seeds={args.seeds}")
    print(f"Variants queued: {len(variants)}  runs queued: {len(queue)}  "
          f"concurrency={args.concurrency} x {args.threads} threads")
    print(f"Device: {args.device}  AMP: {args.amp}  "
          f"design cache: {os.path.relpath(args.design_cache, _HERE)}")

    if args.probe:
        return probe(args, py, out_jsonl, history_jsonl, env)

    if not queue:
        print("Nothing to do.")
        return run_report(out_dir) if not args.no_report else 0

    running = []          # (proc, run_id, log_path, started_at)
    failures = []
    done = 0
    t_start = time.time()

    def launch(v, seed, run_id):
        save_model = ""
        if args.save_models and v["tag"] == "baseline":
            save_model = os.path.join(out_dir, "checkpoints", f"{run_id}.pt")
            os.makedirs(os.path.dirname(save_model), exist_ok=True)
        cmd = variant_cmd(py, v["args"], v["tag"], seed, run_id, out_jsonl,
                          history_jsonl, args.dataset, args.k_core,
                          args.epochs, args.eval_every, args.topks, args.device,
                          amp=args.amp, design_cache=args.design_cache,
                          design_seed=args.design_seed, save_model=save_model,
                          grad_checkpoint=args.grad_checkpoint)
        log_path = os.path.join(log_dir, f"{run_id}.log")
        if args.dry_run:
            print("DRY", " ".join(cmd))
            return None
        log = open(log_path, "w")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                             env=env, cwd=_HERE)
        p._log = log
        print(f"START {run_id}  (log: {os.path.relpath(log_path, out_dir)})",
              flush=True)
        return p

    while queue or running:
        while queue and len(running) < args.concurrency:
            v, seed, run_id = queue.popleft()
            p = launch(v, seed, run_id)
            if p is None:            # dry run
                continue
            running.append((p, run_id, time.time()))

        time.sleep(5)
        still = []
        for p, run_id, t0 in running:
            rc = p.poll()
            if rc is None:
                still.append((p, run_id, t0))
                continue
            p._log.close()
            mins = (time.time() - t0) / 60
            if rc == 0:
                done += 1
                print(f"DONE  {run_id}  ({mins:.1f} min)", flush=True)
            else:
                failures.append((run_id, rc))
                print(f"FAIL  {run_id}  rc={rc} ({mins:.1f} min) "
                      f"see logs/{run_id}.log", flush=True)
        running = still

    elapsed_h = (time.time() - t_start) / 3600
    print(f"\nFinished {done} runs in {elapsed_h:.2f} h "
          f"({len(failures)} failures)")
    for run_id, rc in failures:
        print(f"  failed: {run_id} rc={rc}")

    if not args.no_report:
        run_report(out_dir)
    return 1 if failures else 0


def probe(args, py, out_jsonl, history_jsonl, env):
    """Time one baseline run, then project the cost of the whole matrix.

    Runs the baseline configuration for ``--epochs`` epochs (use something small
    like 20), reads the recorded setup/train times, and extrapolates. The design
    build is *not* cached during the probe, so the projection includes one
    design build plus cached loads for every later run.
    """
    import tempfile
    n_queued = sum(len(v.get("seeds") or args.seeds) for v in VARIANTS)
    probe_dir = tempfile.mkdtemp(prefix="ablation_probe_")
    probe_out = os.path.join(probe_dir, "probe.jsonl")
    probe_hist = os.path.join(probe_dir, "probe_history.jsonl")
    cmd = variant_cmd(py, [], "probe", 2020, "probe", probe_out, probe_hist,
                      args.dataset, args.k_core, args.epochs, args.eval_every,
                      args.topks, args.device, amp=args.amp,
                      design_cache="", design_seed=args.design_seed)
    print("\n=== probe: one baseline run (no design cache) ===")
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env, cwd=_HERE)
    wall = time.time() - t0
    if rc != 0:
        print(f"probe failed rc={rc}")
        return 1
    rows = read_jsonl(probe_out)
    if not rows:
        print("probe produced no record")
        return 1
    r = rows[0]
    setup = r.get("setup_time_s", 0.0)
    train = r.get("train_time_s", wall)
    n_ep = args.epochs
    per_ep = train / max(n_ep, 1)
    print(f"\n  setup (design build): {setup:.1f} s")
    print(f"  train: {train:.1f} s for {n_ep} epochs -> {per_ep:.3f} s/epoch")
    print(f"  device: {r.get('device')}")
    # Projection: the first run of each distinct spectral config pays the design
    # build, all others load it from cache (~seconds). Distinct designs in the
    # current matrix: baseline, no_spectral, nfreq1, dv0.5, k100, no_uu, uu100.
    n_designs = 7
    epochs_total = n_queued * args.epochs
    est_train = epochs_total * per_ep
    est_setup = n_designs * setup + max(0, n_queued - n_designs) * 3.0
    est = est_train + est_setup
    print(f"\n  queued runs: {n_queued}  (epochs/run: {args.epochs})")
    print(f"  projected train time:  {est_train/3600:.2f} h")
    print(f"  projected setup time:  {est_setup/60:.1f} min "
          f"({n_designs} distinct designs built, rest cached)")
    print(f"  projected total:       {est/3600:.2f} h on one worker")
    print("\n  Choose the epoch budget so this fits your window, e.g.")
    print(f"    --epochs {max(100, int(args.epochs))}  (probed)")
    print("  Re-run the probe after changing --k/--uu-topk: those change the "
          "support build and the per-epoch cost.")
    return 0


def run_report(out_dir):
    script = os.path.join(_HERE, "analyze_ablations.py")
    return subprocess.call([sys.executable, script, "--out-dir", out_dir])


if __name__ == "__main__":
    raise SystemExit(main())
