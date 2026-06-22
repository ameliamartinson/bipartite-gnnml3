#!/usr/bin/env bash
#
# Hyperparameter sweep for the GNNML3 bipartite link-prediction model.
#
# This is a *coordinate* sweep: it fixes a sensible baseline and varies one axis
# at a time (embedding dim, layers, readout, features, lr, spectral design,
# biadjacency normalization). That keeps it to ~25 runs instead of a full grid
# of hundreds. All runs append to one JSONL; at the end the script prints the
# configurations ranked by Recall@K so you can read off the best settings.
#
# Usage:
#   ./sweep_gnnml3.sh                # gowalla, defaults below
#   DATASET=yelp2018 ./sweep_gnnml3.sh
#   EPOCHS=1000 DEVICE=cuda ./sweep_gnnml3.sh amazon-book
#   RUN_MINIBATCH=1 ./sweep_gnnml3.sh   # also try (slow) minibatched BPR
#   AGGREGATE_ONLY=1 ./sweep_gnnml3.sh  # just re-print the ranked table
#
# Tunables (env vars, with defaults):
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${1:-${DATASET:-gowalla}}"
EPOCHS="${EPOCHS:-500}"
EVAL_EVERY="${EVAL_EVERY:-20}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-2020}"
TOPK="${TOPK:-20}"
OUT="${OUT:-$HERE/results/sweep_${DATASET}.jsonl}"
RUN_MINIBATCH="${RUN_MINIBATCH:-0}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"

# Pick the project venv if present, else fall back to python3.
if [ -x "$HERE/.venv/bin/python" ]; then
  PY="$HERE/.venv/bin/python"
else
  PY="python3"
fi
EXP="$HERE/bipartite_experiment.py"

mkdir -p "$(dirname "$OUT")"

# Baseline (each sweep below overrides exactly one axis of this).
BASE_EMB_IN=64
BASE_LAYERS=3
BASE_NFREQ=5
BASE_K=100
BASE_LR=0.001

run() {
  # run <tag> <extra args...>
  local tag="$1"; shift
  echo
  echo "############################################################"
  echo "# RUN: $tag"
  echo "############################################################"
  echo "+ $PY $EXP --dataset $DATASET (epochs=$EPOCHS) $*"
  "$PY" "$EXP" \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --eval-every "$EVAL_EVERY" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --topks "[$TOPK]" \
    --out "$OUT" \
    "$@" \
    || echo "!! FAILED: $tag (continuing)"
}

aggregate() {
  "$PY" - "$OUT" "$DATASET" "$TOPK" <<'PYEOF'
import json, sys
path, ds, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
rk, nk = f"recall@{k}", f"ndcg@{k}"
rows = []
try:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("model") == "gnnml3" and r.get("dataset") == ds and rk in r:
                rows.append(r)
except FileNotFoundError:
    print(f"No results file at {path}")
    sys.exit(0)
rows.sort(key=lambda r: r.get(rk, 0.0), reverse=True)
cols = ["emb_in", "layers", "layer_combine", "struct_feats", "bpr_batch",
        "nfreq", "k_svd", "lr", "biadj", rk, nk, "best_epoch", "train_time_s"]
print()
print("=" * 120)
print(f"RANKED gnnml3 SETTINGS on {ds} ({len(rows)} runs, sorted by {rk})")
print("=" * 120)
hdr = " ".join(c.rjust(13) for c in cols)
print(hdr)
print("-" * len(hdr))
for r in rows[:40]:
    print(" ".join(str(r.get(c, "-")).rjust(13) for c in cols))
if rows:
    b = rows[0]
    print()
    print("BEST:", {c: b.get(c) for c in cols})
PYEOF
}

if [ "$AGGREGATE_ONLY" = "1" ]; then
  aggregate
  exit 0
fi

echo "Sweeping gnnml3 on '$DATASET'  (epochs=$EPOCHS, device=$DEVICE, out=$OUT)"
echo "Python: $PY"

# ---------------------------------------------------------------------------
# 1. Learnable node-embedding dimension (0 = structural features only).
# ---------------------------------------------------------------------------
for E in 0 32 64 128; do
  args=(--emb-in "$E" --layers "$BASE_LAYERS" --nfreq "$BASE_NFREQ" --k "$BASE_K" --lr "$BASE_LR")
  [ "$E" -gt 0 ] && args+=(--layer-combine)
  run "emb_in=$E" "${args[@]}"
done

# ---------------------------------------------------------------------------
# 2. Number of spectral layers.
# ---------------------------------------------------------------------------
for L in 1 2 3 4; do
  run "layers=$L" --emb-in "$BASE_EMB_IN" --layers "$L" --layer-combine \
      --nfreq "$BASE_NFREQ" --k "$BASE_K" --lr "$BASE_LR"
done

# ---------------------------------------------------------------------------
# 3. Layer readout: mean (jumping-knowledge) vs last-layer only.
# ---------------------------------------------------------------------------
for LC in on off; do
  args=(--emb-in "$BASE_EMB_IN" --layers "$BASE_LAYERS" --nfreq "$BASE_NFREQ" --k "$BASE_K" --lr "$BASE_LR")
  [ "$LC" = on ] && args+=(--layer-combine)
  run "layer_combine=$LC" "${args[@]}"
done

# ---------------------------------------------------------------------------
# 4. Input features: embeddings + structural vs embeddings only.
# ---------------------------------------------------------------------------
for SF in with without; do
  args=(--emb-in "$BASE_EMB_IN" --layers "$BASE_LAYERS" --layer-combine --nfreq "$BASE_NFREQ" --k "$BASE_K" --lr "$BASE_LR")
  [ "$SF" = without ] && args+=(--no-struct-feats)
  run "struct_feats=$SF" "${args[@]}"
done

# ---------------------------------------------------------------------------
# 5. Learning rate.
# ---------------------------------------------------------------------------
for LR in 0.0005 0.001 0.005; do
  run "lr=$LR" --emb-in "$BASE_EMB_IN" --layers "$BASE_LAYERS" --layer-combine \
      --nfreq "$BASE_NFREQ" --k "$BASE_K" --lr "$LR"
done

# ---------------------------------------------------------------------------
# 6. Spectral design: number of frequency bands.
# ---------------------------------------------------------------------------
for NF in 2 5 8; do
  run "nfreq=$NF" --emb-in "$BASE_EMB_IN" --layers "$BASE_LAYERS" --layer-combine \
      --nfreq "$NF" --k "$BASE_K" --lr "$BASE_LR"
done

# ---------------------------------------------------------------------------
# 7. Spectral design: SVD rank k.
# ---------------------------------------------------------------------------
for KV in 50 100 200; do
  run "k_svd=$KV" --emb-in "$BASE_EMB_IN" --layers "$BASE_LAYERS" --layer-combine \
      --nfreq "$BASE_NFREQ" --k "$KV" --lr "$BASE_LR"
done

# ---------------------------------------------------------------------------
# 8. Biadjacency: symmetrically normalized vs raw.
# ---------------------------------------------------------------------------
for BA in normalized raw; do
  args=(--emb-in "$BASE_EMB_IN" --layers "$BASE_LAYERS" --layer-combine --nfreq "$BASE_NFREQ" --k "$BASE_K" --lr "$BASE_LR")
  [ "$BA" = raw ] && args+=(--raw-biadj)
  run "biadj=$BA" "${args[@]}"
done

# ---------------------------------------------------------------------------
# 9. (Optional, slow) LightGCN-style minibatched BPR.
# ---------------------------------------------------------------------------
if [ "$RUN_MINIBATCH" = "1" ]; then
  for BB in 2048 8192; do
    run "bpr_batch=$BB" --emb-in "$BASE_EMB_IN" --layers "$BASE_LAYERS" --layer-combine \
        --nfreq "$BASE_NFREQ" --k "$BASE_K" --lr "$BASE_LR" --bpr-batch "$BB"
  done
fi

aggregate

echo
echo "Done. Re-print the ranked table any time with:"
echo "  AGGREGATE_ONLY=1 DATASET=$DATASET ./sweep_gnnml3.sh"
