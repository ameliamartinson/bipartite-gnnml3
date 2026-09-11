#!/usr/bin/env bash
#
# Spectral-selectivity study: is the GNNML3 band/filter machinery actually
# doing anything, or is the model just learnable node embeddings + a learned
# aggregation over the mask?
#
# Anchor = the best configuration found so far on Gowalla k-core 20 (see
# results/benchmark.jsonl): 1 ML3 layer, 1 spectral band, rank-100 SVD, no
# extra co-interaction edges, 64-d learnable embeddings, 3000 epochs.
#
# Arms (every arm changes exactly one thing in the descriptor the edge network
# sees; the support graph, nsup and parameter count are identical):
#   baseline      real band filters g(sigma), even filters h(sigma)
#   flat_1_0      band columns = 1.0            (the flag as first specified)
#   flat_matched  band columns = mean|band|     (scale-matched, no selectivity)
#   shuffled      filters at permuted sigma    (frequency correspondence gone)
#   no_spectral   nfreq=0                      (no bands at all -> graph-free)
#   no_odd        cross-partition band zeroed  (no user-item band signal)
#
# The two flat arms are the point: flat_1_0 vs flat_matched separates the *scale*
# effect of a constant descriptor from the *loss of selectivity*, and
# baseline vs flat_matched is the clean no-selectivity comparison.
#
# Usage:
#   ./run_selectivity_study.sh                       # k-core 20, 3 seeds, ~30-60 min
#   K_CORE=0 EPOCHS=2000 ./run_selectivity_study.sh  # full Gowalla (headline graph)
#   SEEDS="2020" ./run_selectivity_study.sh          # single seed (level check)
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-gowalla}"
K_CORE="${K_CORE:-20}"
EPOCHS="${EPOCHS:-3000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
SEEDS="${SEEDS:-2020 2021 2022}"
DEVICE="${DEVICE:-auto}"
AMP="${AMP:-0}"                     # keep 0: flat_1_0 overflows fp16 (the NaN)
OUTDIR="${OUTDIR:-$HERE/results/selectivity_kc${K_CORE}}"
FLAT_VALUE="${FLAT_VALUE:-0.0018}"  # mean|band| for nfreq=1, dv=5 (see below)

if [ -x "$HERE/.venv/bin/python" ]; then PY="$HERE/.venv/bin/python"; else PY="python3"; fi
OUT="$OUTDIR/benchmark.jsonl"
HIST="$OUTDIR/history.jsonl"
mkdir -p "$OUTDIR"

# Regenerate the suggested scale-matched constant for the anchor, if cheap.
if [ "$FLAT_VALUE" = "auto" ]; then
  FLAT_VALUE="$("$PY" "$HERE/ablation_diagnostics.py" --support-stats \
      --dataset "$DATASET" --k-core "$K_CORE" --nfreq 1 --dv 5 --k 100 \
      --uu-topk 0 2>/dev/null | grep -o 'suggested --flat-support-value.*' \
      | grep -oE '[0-9]+\.[0-9]+' | head -1)"
  echo "auto flat-support-value = $FLAT_VALUE"
fi

ANCHOR=(--dataset "$DATASET" --k-core "$K_CORE" --layers 1 --nfreq 1 --dv 5
        --k 100 --uu-topk 0 --emb-in 64 --embed-dim 64 --layer-combine
        --epochs "$EPOCHS" --eval-every "$EVAL_EVERY" --topks "[20]"
        --device "$DEVICE" --design-cache "$OUTDIR/design_cache"
        --design-seed 2020 --out "$OUT" --history-out "$HIST")
[ "$AMP" = "1" ] && ANCHOR+=(--amp)

run() {  # run <tag> <extra args...>
  local tag="$1"; shift
  local rid="${tag}_s${SEED}"
  if [ -f "$OUT" ] && grep -q "\"run_id\": \"$rid\"" "$OUT"; then
    echo "SKIP $rid (already in $OUT)"; return 0
  fi
  echo ">>> $rid  $*"
  "$PY" "$HERE/bipartite_experiment.py" "${ANCHOR[@]}" --seed "$SEED" \
      --run-id "$rid" --ablation "$tag" "$@" \
      2>&1 | tee "$OUTDIR/logs_${rid}.log" | tail -3
}

echo "Spectral-selectivity study: dataset=$DATASET k-core=$K_CORE epochs=$EPOCHS"
echo "seeds=[$SEEDS] device=$DEVICE amp=$AMP flat_value=$FLAT_VALUE"
echo "out=$OUT"

for SEED in $SEEDS; do
  run baseline
  run flat_1_0     --flat-support
  run flat_support_matched --flat-support --flat-support-value "$FLAT_VALUE"
  run shuffled     --shuffle-bands
  run no_spectral  --nfreq 0
  run no_odd       --no-odd
done

echo
echo "=== summary ==="
"$PY" "$HERE/analyze_ablations.py" --out-dir "$OUTDIR" --plots

cat <<'EOF'

Read the result as (paired over seeds, baseline = the spectral arm):
  * flat_matched ~= baseline, flat_1_0 > baseline  -> the flat_1_0 win was the
    550x message-scale change; spectral selectivity contributes nothing.
  * flat_matched <  baseline                      -> the band descriptors do
    carry signal (the single-seed flat result was a scale artifact).
  * flat_matched >  baseline                      -> per-edge spectral variation
    actively hurts (strongest form of the "band bank does not help" claim).
  * shuffled ~= baseline                          -> frequency correspondence is
    irrelevant: the bands act as random features.
  * no_spectral << baseline ~= flat_matched       -> message passing matters;
    the gain is not all in the embeddings/heads.
Send back ablation_summary.md / .csv and benchmark.jsonl.
EOF
