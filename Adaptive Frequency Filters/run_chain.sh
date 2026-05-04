#!/usr/bin/env bash
# Autonomous chain: waits for step 2 to finish, then runs reports + BC-only +
# residual sweep, with .txt reports between each phase.
#
# Each step writes to its own .log file under results/sweep_chain_logs/.
# A fail in any phase stops the chain (set -e).

set -e
set -o pipefail

cd "$(dirname "$0")"
mkdir -p results/sweep_chain_logs
CHAIN_LOG=results/sweep_chain_logs/_chain.log
PYTHON="C:/Users/alona/miniconda3/envs/AFFNet/python.exe"
export PYTHONIOENCODING=utf-8

echo "[chain] $(date) - waiting for step 2 [stub] DONE marker" >> "$CHAIN_LOG"
until grep -q "^\[stub\] DONE" results/sweep_stub_logs/_orchestrator.log 2>/dev/null; do
  sleep 30
done
echo "[chain] $(date) - step 2 done; writing step1 + step2 reports" >> "$CHAIN_LOG"

"$PYTHON" -u write_reports.py step1 >> "$CHAIN_LOG" 2>&1
"$PYTHON" -u write_reports.py step2 >> "$CHAIN_LOG" 2>&1

echo "[chain] $(date) - launching BC-only top5" >> "$CHAIN_LOG"
"$PYTHON" -u run_bc_only_top5.py >> results/sweep_chain_logs/bc_only.log 2>&1

echo "[chain] $(date) - writing BC vs BC+BN report" >> "$CHAIN_LOG"
"$PYTHON" -u write_reports.py bc_nf >> "$CHAIN_LOG" 2>&1

echo "[chain] $(date) - launching residual 8x8 sweep" >> "$CHAIN_LOG"
"$PYTHON" -u run_residual_sweep.py >> results/sweep_chain_logs/residual.log 2>&1

echo "[chain] $(date) - writing residual report" >> "$CHAIN_LOG"
"$PYTHON" -u write_reports.py residual >> "$CHAIN_LOG" 2>&1

echo "[chain] $(date) - ALL DONE" >> "$CHAIN_LOG"
