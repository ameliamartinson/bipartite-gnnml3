#!/usr/bin/env bash
#
# All experimental passes + data collection for the thesis, in one place.
#
# Passes (each maps to a thesis chapter/figure):
#   1. LightGCN baselines            (Results table)        multi-seed
#   2. GNNML3 headline configs       (Results table)        multi-seed
#   3. --bpr-batch gradient budget   (Diagnosis III)        single seed
#   4. Coordinate hyperparameter sweep (appendix)           via sweep_gnnml3.sh
#   5. Spectral diagnostics          (Diagnoses I & II)     no training
#   6. Qualitative recommendations   (case-study section)   from checkpoints
#   7. Aggregation                   summary.md + tables from the JSONL
#
# Everything lands under results/thesis/:
#   benchmark.jsonl   one record per training run (append-only)
#   logs/<tag>.log    full stdout of every run
#   checkpoints/      best-model .pt per headline config (seed ${SEEDS%% *})
#   diagnostics/      spectrum JSON + energy-curve CSV per dataset
#   recs/             example recommendation reports
#   summary.md        mean +/- std over seeds, ranked
#   .done/<tag>       resume markers -- delete one to force a re-run
#
# Usage:
#   ./run_thesis_experiments.sh                  # full pass, gowalla
#   QUICK=1 ./run_thesis_experiments.sh          # smoke test (dev box)
#   DATASETS="gowalla yelp2018" SEEDS="2020 2021 2022" EPOCHS=1000 \
#     DEVICE=cuda ./run_thesis_experiments.sh    # the real L40 run
#   RUN_SWEEP=1 ./run_thesis_experiments.sh      # also run the ~25-run sweep
#   AGGREGATE_ONLY=1 ./run_thesis_experiments.sh # just rebuild summary.md
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── knobs ────────────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-gowalla}"
SEEDS="${SEEDS:-2020 2021 2022}"
EPOCHS="${EPOCHS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-20}"
DEVICE="${DEVICE:-auto}"
TOPKS="${TOPKS:-[20]}"
TOPK_PRIMARY="${TOPK_PRIMARY:-20}"
K_SVD="${K_SVD:-100}"
DIAG_K="${DIAG_K:-400}"          # singular values for the energy curve
UU_TOPK="${UU_TOPK:-30}"
BPR_BATCHES="${BPR_BATCHES:-2048 8192}"
RUN_SWEEP="${RUN_SWEEP:-0}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
OUTDIR="${OUTDIR:-$HERE/results/thesis}"

# Extra args for the single "best sweep config" run; fill in after the sweep,
# e.g. BEST_ARGS="--emb-in 128 --layers 2 --layer-combine --nfreq 8 --k 200"
BEST_ARGS="${BEST_ARGS:-}"

if [ "${QUICK:-0}" = "1" ]; then
  echo ">>> QUICK smoke mode: tiny budgets, one seed, gowalla only <<<"
  DATASETS="gowalla"; SEEDS="2020"; EPOCHS=30; EVAL_EVERY=10; DIAG_K=32
fi

OUT="$OUTDIR/benchmark.jsonl"
mkdir -p "$OUTDIR"/{logs,checkpoints,diagnostics,recs,.done}

if [ -x "$HERE/.venv/bin/python" ]; then PY="$HERE/.venv/bin/python"; else PY="python3"; fi
FIRST_SEED="${SEEDS%% *}"

# run <tag> <cmd...>  -- logs to logs/<tag>.log, skips if .done/<tag> exists
run() {
  local tag="$1"; shift
  if [ -e "$OUTDIR/.done/$tag" ]; then
    echo "SKIP $tag (done; rm $OUTDIR/.done/$tag to re-run)"
    return 0
  fi
  echo
  echo "############################################################"
  echo "# RUN: $tag"
  echo "# $*"
  echo "############################################################"
  if "$@" 2>&1 | tee "$OUTDIR/logs/$tag.log"; then
    touch "$OUTDIR/.done/$tag"
  else
    echo "!! FAILED: $tag (continuing; see logs/$tag.log)"
  fi
}

aggregate() {
  "$PY" "$HERE/aggregate_thesis_results.py" \
    --jsonl "$OUT" --topk "$TOPK_PRIMARY" --out "$OUTDIR/summary.md"
}

if [ "$AGGREGATE_ONLY" = "1" ]; then aggregate; exit 0; fi

echo "Thesis experiments: datasets=[$DATASETS] seeds=[$SEEDS] epochs=$EPOCHS device=$DEVICE"
echo "Output: $OUTDIR"

for DS in $DATASETS; do

  # ── Pass 1: LightGCN baselines ────────────────────────────────────────────
  for SEED in $SEEDS; do
    save=(); [ "$SEED" = "$FIRST_SEED" ] && \
      save=(--save-model "$OUTDIR/checkpoints/lightgcn_${DS}.pt")
    run "p1_lightgcn_${DS}_s${SEED}" \
      "$PY" "$HERE/run_lightgcn.py" --dataset "$DS" --seed "$SEED" \
      --epochs "$EPOCHS" --eval-every "$EVAL_EVERY" --topks "$TOPKS" \
      --device "$DEVICE" --out "$OUT" "${save[@]}"
  done

  # ── Pass 2: GNNML3 headline configs ───────────────────────────────────────
  # base:    structural features only (no learnable embeddings)
  # emb:     + learnable 64-d embeddings, jumping-knowledge readout
  # emb_uu:  + sparsified co-interaction receptive field (uu_topk)
  # best:    the winning sweep config (set BEST_ARGS after Pass 4)
  gnnml3_common=(--epochs "$EPOCHS" --eval-every "$EVAL_EVERY" --topks "$TOPKS"
                 --device "$DEVICE" --k "$K_SVD" --out "$OUT")
  declare -A CONFIGS=(
    [base]=""
    [emb]="--emb-in 64 --layer-combine"
    [emb_uu]="--emb-in 64 --layer-combine --uu-topk $UU_TOPK"
  )
  [ -n "$BEST_ARGS" ] && CONFIGS[best]="$BEST_ARGS"
  for CFG in "${!CONFIGS[@]}"; do
    for SEED in $SEEDS; do
      save=(); [ "$SEED" = "$FIRST_SEED" ] && \
        save=(--save-model "$OUTDIR/checkpoints/gnnml3_${CFG}_${DS}.pt")
      # shellcheck disable=SC2086
      run "p2_gnnml3_${CFG}_${DS}_s${SEED}" \
        "$PY" "$HERE/bipartite_experiment.py" --dataset "$DS" --seed "$SEED" \
        "${gnnml3_common[@]}" ${CONFIGS[$CFG]} "${save[@]}"
    done
  done

  # ── Pass 3: gradient-step budget (--bpr-batch), Diagnosis III ────────────
  # Full-batch (bpr_batch=0) is already covered by the emb config above.
  for BB in $BPR_BATCHES; do
    run "p3_gnnml3_bprbatch${BB}_${DS}_s${FIRST_SEED}" \
      "$PY" "$HERE/bipartite_experiment.py" --dataset "$DS" --seed "$FIRST_SEED" \
      "${gnnml3_common[@]}" --emb-in 64 --layer-combine --bpr-batch "$BB"
  done

  # ── Pass 4: coordinate sweep (appendix; ~25 runs, off by default) ────────
  if [ "$RUN_SWEEP" = "1" ]; then
    run "p4_sweep_${DS}" env DATASET="$DS" EPOCHS="$EPOCHS" DEVICE="$DEVICE" \
      SEED="$FIRST_SEED" TOPK="$TOPK_PRIMARY" \
      OUT="$OUTDIR/sweep_${DS}.jsonl" "$HERE/sweep_gnnml3.sh"
  fi

  # ── Pass 5: spectral diagnostics (Diagnoses I & II; no training) ─────────
  run "p5_diagnostics_${DS}" \
    "$PY" "$HERE/collect_spectral_diagnostics.py" --dataset "$DS" \
    --k "$DIAG_K" --seed "$FIRST_SEED" --out-dir "$OUTDIR/diagnostics"

  # ── Pass 6: qualitative recommendations from the saved checkpoints ───────
  for CK in "$OUTDIR"/checkpoints/*_"$DS".pt; do
    [ -e "$CK" ] || continue
    name="$(basename "$CK" .pt)"
    run "p6_recs_${name}" \
      "$PY" "$HERE/recommend.py" --checkpoint "$CK" --seed "$FIRST_SEED" \
      --out "$OUTDIR/recs/${name}.txt"
  done

done

# ── Pass 7: aggregate ────────────────────────────────────────────────────────
aggregate

echo
echo "Done. Rebuild the summary any time with:"
echo "  AGGREGATE_ONLY=1 ./run_thesis_experiments.sh"
