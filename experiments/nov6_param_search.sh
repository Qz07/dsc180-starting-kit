#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all_kfixed.sh 0.75
# If omitted, defaults to k=1.0

K="${1:-1.0}"
OUTDIR="exp_logs_k${K}"
RUNNER="python -u run_exp.py"

mkdir -p "$OUTDIR"

echo "=== Running RMU sweeps at fixed k=${K} ==="

# -----------------------------
# Default balanced-ish (epochs 4–5)
# -----------------------------
for E in 4 5; do
  tag="rmu_default_e${E}_k${K}"
  echo ">>> $tag"
  $RUNNER --method rmu --num_epochs "$E" --k "$K" | tee "${OUTDIR}/${tag}.log"
done

# -----------------------------
# Faster, stronger forgetting-ish (epochs 2–3)
# -----------------------------
for E in 2 3; do
  tag="rmu_fast_e${E}_k${K}"
  echo ">>> $tag"
  $RUNNER --method rmu --num_epochs "$E" --k "$K" | tee "${OUTDIR}/${tag}.log"
done

# -----------------------------
# High-fidelity retention-ish (epochs 6–8)
# -----------------------------
for E in 6 8; do
  tag="rmu_safe_e${E}_k${K}"
  echo ">>> $tag"
  $RUNNER --method rmu --num_epochs "$E" --k "$K" | tee "${OUTDIR}/${tag}.log"
done

# -----------------------------
# Baselines (for reference)
# -----------------------------
echo ">>> baseline_original"
$RUNNER --method original --num_epochs 1 --k "$K" | tee "${OUTDIR}/baseline_original.log" || true

echo ">>> baseline_retrain"
$RUNNER --method retrain  --num_epochs 1 --k "$K" | tee "${OUTDIR}/baseline_retrain.log"  || true

echo "=== Done. Logs in: $OUTDIR ==="
