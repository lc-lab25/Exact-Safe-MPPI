#!/usr/bin/env bash
# Full-scale reproduction of every paper result, then verification.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-1}"
EXTRA="${EXTRA:-}"
export PYTHONPATH="src:experiments:${PYTHONPATH:-}"
for e in closed_form_checks lcp_vs_scalar gap_threshold fig2_median_runs nine_constraint kmax_ablation; do
    echo "=== $e ==="
    python "experiments/$e.py" --workers "$WORKERS" $EXTRA
done
echo "=== verify ==="
python scripts/verify_results.py ${EXTRA:+--quick}
