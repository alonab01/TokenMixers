#!/usr/bin/env bash
# Portable 6-experiment driver for AFFNet-ET FP32 / PTQ reproduction.
#
# Usage:
#   ./scripts/run_experiments.sh             # print usage
#   ./scripts/run_experiments.sh all         # run experiments 1..6 in order
#   ./scripts/run_experiments.sh 3           # run only experiment 3
#   ./scripts/run_experiments.sh 1 3 5       # run 1, 3, 5 in that order
#
# Optional overrides:
#   CONDA_ENV=AFFnet ./scripts/run_experiments.sh all
#   PYTHON_BIN=/path/to/python ./scripts/run_experiments.sh all
#   CFG=/path/to/config.yaml CKPT=/path/to/checkpoint.pt ./scripts/run_experiments.sh 1
#   RESULTS_DIR=my_results ./scripts/run_experiments.sh all
#   AUTO_CONDA_ACTIVATE=0 ./scripts/run_experiments.sh all

set -euo pipefail

# -----------------------------
# Project root
# -----------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# -----------------------------
# Portable configuration
# -----------------------------

CFG="${CFG:-resource/config/imagenet_et/config.yaml}"
CKPT="${CKPT:-resource/model/imagenet_et/checkpoint_ema_score_73.0238.pt}"

RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/scripts_logs}"
CSV_DIR="${CSV_DIR:-$RESULTS_DIR/scripts_csv}"
SUMMARY_CSV="$CSV_DIR/_summary.csv"

mkdir -p "$LOG_DIR" "$CSV_DIR"

# -----------------------------
# Conda activation
# -----------------------------

# Default conda environment name.
# Note: conda environment names may be case-sensitive.
CONDA_ENV="${CONDA_ENV:-AFFnet}"

# Set AUTO_CONDA_ACTIVATE=0 to skip conda activation.
AUTO_CONDA_ACTIVATE="${AUTO_CONDA_ACTIVATE:-1}"

activate_conda_env() {
  if [[ "$AUTO_CONDA_ACTIVATE" != "1" ]]; then
    echo "Skipping conda activation because AUTO_CONDA_ACTIVATE=0"
    return 0
  fi

  # If the user explicitly supplied Python, do not activate conda automatically.
  if [[ -n "${PYTHON_BIN:-}" || -n "${PYTHON:-}" ]]; then
    echo "Skipping conda activation because PYTHON_BIN or PYTHON was provided."
    return 0
  fi

  local conda_base=""

  # Normal case: conda is available in PATH.
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
  fi

  # Fallback common install locations.
  if [[ -z "$conda_base" ]]; then
    local candidates=(
      "$HOME/miniconda3"
      "$HOME/anaconda3"
      "/opt/conda"
      "/c/Users/${USERNAME:-}/miniconda3"
      "/c/Users/${USERNAME:-}/anaconda3"
    )

    local base
    for base in "${candidates[@]}"; do
      if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
        conda_base="$base"
        break
      fi
    done
  fi

  if [[ -z "$conda_base" || ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: Could not find conda.sh."
    echo
    echo "Fix options:"
    echo "  1. Open a terminal where conda is available."
    echo "  2. Or pass Python directly:"
    echo "       PYTHON_BIN=/path/to/python ./scripts/run_experiments.sh all"
    echo "  3. Or disable conda activation:"
    echo "       AUTO_CONDA_ACTIVATE=0 ./scripts/run_experiments.sh all"
    exit 2
  fi

  # Required so that 'conda activate' works inside a bash script.
  # shellcheck source=/dev/null
  source "$conda_base/etc/profile.d/conda.sh"

  if ! conda activate "$CONDA_ENV"; then
    echo "ERROR: Failed to activate conda environment: $CONDA_ENV"
    echo
    echo "Available conda environments:"
    conda env list || true
    echo
    echo "You can override the environment name like this:"
    echo "  CONDA_ENV=AFFNet ./scripts/run_experiments.sh all"
    exit 2
  fi

  echo "Activated conda environment: $CONDA_ENV"
}

# -----------------------------
# Python auto-detection
# -----------------------------

PYTHON_CMD=()
find_python() {
  python_works() {
    "$@" -c "import sys; print(sys.executable)" >/dev/null 2>&1
  }

  is_windows_store_python() {
    local p="$1"
    [[ "$p" == *"/WindowsApps/python"* || "$p" == *"/Microsoft/WindowsApps/python"* ]]
  }

  # User override has highest priority.
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if python_works "$PYTHON_BIN"; then
      PYTHON_CMD=("$PYTHON_BIN")
      return 0
    fi

    echo "ERROR: PYTHON_BIN was provided but does not work:"
    echo "  $PYTHON_BIN"
    return 1
  fi

  # Backward-compatible override.
  if [[ -n "${PYTHON:-}" ]]; then
    if python_works "$PYTHON"; then
      PYTHON_CMD=("$PYTHON")
      return 0
    fi

    echo "ERROR: PYTHON was provided but does not work:"
    echo "  $PYTHON"
    return 1
  fi

  # After 'conda activate', the correct Python should usually be first in PATH.
  # Important: prefer 'python' over 'python3' on Windows/Git Bash.
  local p
  p="$(command -v python 2>/dev/null || true)"

  if [[ -n "$p" ]] && ! is_windows_store_python "$p" && python_works "$p"; then
    PYTHON_CMD=("$p")
    return 0
  fi

  # If CONDA_PREFIX exists, try direct paths inside the active conda env.
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    local candidates=(
      "$CONDA_PREFIX/bin/python"
      "$CONDA_PREFIX/Scripts/python.exe"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
      if [[ -f "$candidate" ]] && python_works "$candidate"; then
        PYTHON_CMD=("$candidate")
        return 0
      fi
    done
  fi

  # Strong fallback: use conda run directly.
  # This is slower, but very reliable across Git Bash / Windows weird path cases.
  if command -v conda >/dev/null 2>&1; then
    if conda run -n "$CONDA_ENV" python -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
      PYTHON_CMD=("conda" "run" "-n" "$CONDA_ENV" "python")
      return 0
    fi
  fi

  # Common local virtual environments.
  local venv_candidates=(
    "$PROJECT_DIR/.venv/bin/python"
    "$PROJECT_DIR/.venv/Scripts/python.exe"
    "$PROJECT_DIR/venv/bin/python"
    "$PROJECT_DIR/venv/Scripts/python.exe"
  )

  local venv_candidate
  for venv_candidate in "${venv_candidates[@]}"; do
    if [[ -f "$venv_candidate" ]] && python_works "$venv_candidate"; then
      PYTHON_CMD=("$venv_candidate")
      return 0
    fi
  done

  # System Python fallback.
  for cmd in python python3; do
    p="$(command -v "$cmd" 2>/dev/null || true)"

    if [[ -n "$p" ]] && ! is_windows_store_python "$p" && python_works "$p"; then
      PYTHON_CMD=("$p")
      return 0
    fi
  done

  # Windows Python launcher fallback.
  if command -v py >/dev/null 2>&1; then
    if py -3 -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
      PYTHON_CMD=("py" "-3")
      return 0
    fi
  fi

  return 1
}

activate_conda_env

if ! find_python; then
  echo "ERROR: Could not find Python."
  echo
  echo "Fix options:"
  echo "  1. Activate your environment first, then run this script."
  echo "  2. Or pass Python explicitly:"
  echo "       PYTHON_BIN=/path/to/python ./scripts/run_experiments.sh all"
  exit 2
fi

echo "Using Python: ${PYTHON_CMD[*]}"
echo "Project dir:   $PROJECT_DIR"
echo "Config file:   $CFG"
echo "Checkpoint:    $CKPT"
echo

# -----------------------------
# Validation
# -----------------------------

require_file() {
  local path="$1"

  if [[ ! -f "$path" ]]; then
    echo "ERROR: Required file not found:"
    echo "  $path"
    echo
    echo "You can override paths like this:"
    echo "  CFG=/path/to/config.yaml CKPT=/path/to/checkpoint.pt ./scripts/run_experiments.sh all"
    exit 2
  fi
}

require_file "main_eval.py"
require_file "main_quant.py"
require_file "$CFG"
require_file "$CKPT"

# -----------------------------
# Usage
# -----------------------------

usage() {
  cat <<EOF
Usage: $0 [all|N ...]
  all          run experiments 1..6 in order
  N           run experiment N, one or more of 1..6

Optional overrides:
  CONDA_ENV              conda environment name, default: AFFnet
  AUTO_CONDA_ACTIVATE    1 to activate conda, 0 to skip, default: 1
  PYTHON_BIN             path to Python executable
  CFG                    path to config YAML
  CKPT                   path to checkpoint
  RESULTS_DIR            output directory

Examples:
  ./scripts/run_experiments.sh all
  ./scripts/run_experiments.sh 1 3 5
  CONDA_ENV=AFFnet ./scripts/run_experiments.sh all
  PYTHON_BIN=/usr/bin/python3 ./scripts/run_experiments.sh all
  CFG=my_config.yaml CKPT=my_checkpoint.pt ./scripts/run_experiments.sh 1

Experiments:
  1  FP32 + mixed precision
  2  FP32 without mixed precision
  3  PTQ Conv+Linear
  4  exp 3 + fold-BN + bias correction
  5  exp 4 + 4 input stubs
  6  exp 5 + residual-add quant
EOF
  exit 1
}

# -----------------------------
# Summary parser
# -----------------------------

# append_summary <exp_id> <name> <log_path>
append_summary() {
  local exp_id="$1"
  local name="$2"
  local log_path="$3"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"

  local summary_line top1 top5
  summary_line="$(grep -E 'top1=.*\|\|.*top5=' "$log_path" | tail -1 || true)"

  if [[ -z "$summary_line" ]]; then
    top1="NA"
    top5="NA"
  else
    top1="$(echo "$summary_line" | sed -E 's/.*top1=([0-9.]+).*/\1/')"
    top5="$(echo "$summary_line" | sed -E 's/.*top5=([0-9.]+).*/\1/')"
  fi

  if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "exp_id,name,top1,top5,timestamp,log_path" > "$SUMMARY_CSV"
  fi

  echo "${exp_id},${name},${top1},${top5},${ts},${log_path}" >> "$SUMMARY_CSV"

  echo "[exp ${exp_id}] DONE  top1=${top1}  top5=${top5}"
}

# -----------------------------
# Common arguments
# -----------------------------

COMMON_ARGS=(
  --common.config-file "$CFG"
  --common.results-loc "$RESULTS_DIR"
  --model.classification.pretrained "$CKPT"
)

# Quant common: 8/8, weight=per_channel_min_max symmetric,
# activation=per_channel_mse asymmetric, Conv2d + Linear quantized.
QUANT_BASE_ARGS=(
  --quant.enabled
  --quant.weight-bits 8
  --quant.activation-bits 8
  --quant.weight-observer per_channel_min_max
  --quant.act-observer per_channel_mse
  --quant.weight-scheme symmetric
  --quant.act-scheme asymmetric
  --quant.quantize-linear
)

# -----------------------------
# Experiments
# -----------------------------

run_exp1() {
  local name="fp32_amp"
  local log="$LOG_DIR/exp1_${name}_$(date +%Y%m%d_%H%M%S).log"

  echo "[exp 1] FP32 + mixed precision  ->  $log"

  "${PYTHON_CMD[@]}" -u main_eval.py "${COMMON_ARGS[@]}" \
    2>&1 | tee "$log"

  append_summary 1 "$name" "$log"
}

run_exp2() {
  local name="fp32_no_amp"
  local log="$LOG_DIR/exp2_${name}_$(date +%Y%m%d_%H%M%S).log"

  echo "[exp 2] FP32 without mixed precision  ->  $log"

  "${PYTHON_CMD[@]}" -u main_eval.py "${COMMON_ARGS[@]}" \
    --common.override-kwargs common.mixed_precision=false \
    2>&1 | tee "$log"

  append_summary 2 "$name" "$log"
}

run_exp3() {
  local name="ptq_conv_linear"
  local log="$LOG_DIR/exp3_${name}_$(date +%Y%m%d_%H%M%S).log"
  local csv="$CSV_DIR/exp3_${name}.csv"

  echo "[exp 3] PTQ Conv+Linear  ->  $log"

  "${PYTHON_CMD[@]}" -u main_quant.py "${COMMON_ARGS[@]}" "${QUANT_BASE_ARGS[@]}" \
    --quant.results-csv "$csv" \
    2>&1 | tee "$log"

  append_summary 3 "$name" "$log"
}

run_exp4() {
  local name="ptq_bc_foldbn"
  local log="$LOG_DIR/exp4_${name}_$(date +%Y%m%d_%H%M%S).log"
  local csv="$CSV_DIR/exp4_${name}.csv"

  echo "[exp 4] exp 3 + fold-BN + bias correction  ->  $log"

  "${PYTHON_CMD[@]}" -u main_quant.py "${COMMON_ARGS[@]}" "${QUANT_BASE_ARGS[@]}" \
    --quant.fold-bn \
    --quant.bias-correction \
    --quant.results-csv "$csv" \
    2>&1 | tee "$log"

  append_summary 4 "$name" "$log"
}

run_exp5() {
  local name="ptq_stubs"
  local log="$LOG_DIR/exp5_${name}_$(date +%Y%m%d_%H%M%S).log"
  local csv="$CSV_DIR/exp5_${name}.csv"

  echo "[exp 5] exp 4 + 4 input stubs  ->  $log"

  "${PYTHON_CMD[@]}" -u main_quant.py "${COMMON_ARGS[@]}" "${QUANT_BASE_ARGS[@]}" \
    --quant.fold-bn \
    --quant.bias-correction \
    --quant.insert-stubs \
    --quant.stub-config "ln2d=percentile,afno2d=per_channel_percentile,swish=per_channel_min_max,globalpool=per_channel_mse" \
    --quant.stub-bits 8 \
    --quant.stub-scheme asymmetric \
    --quant.results-csv "$csv" \
    2>&1 | tee "$log"

  append_summary 5 "$name" "$log"
}

run_exp6() {
  local name="ptq_stubs_residual"
  local log="$LOG_DIR/exp6_${name}_$(date +%Y%m%d_%H%M%S).log"
  local csv="$CSV_DIR/exp6_${name}.csv"

  echo "[exp 6] exp 5 + residual-add quant  ->  $log"

  "${PYTHON_CMD[@]}" -u main_quant.py "${COMMON_ARGS[@]}" "${QUANT_BASE_ARGS[@]}" \
    --quant.fold-bn \
    --quant.bias-correction \
    --quant.insert-stubs \
    --quant.stub-config "ln2d=percentile,afno2d=per_channel_percentile,swish=per_channel_min_max,globalpool=per_channel_mse" \
    --quant.stub-bits 8 \
    --quant.stub-scheme asymmetric \
    --quant.quantize-residuals \
    --quant.residual-bits 8 \
    --quant.residual-scheme asymmetric \
    --quant.residual-main-observer per_channel_min_max \
    --quant.residual-skip-observer per_channel_mse \
    --quant.results-csv "$csv" \
    2>&1 | tee "$log"

  append_summary 6 "$name" "$log"
}

# -----------------------------
# Dispatch
# -----------------------------

dispatch() {
  case "$1" in
    1) run_exp1 ;;
    2) run_exp2 ;;
    3) run_exp3 ;;
    4) run_exp4 ;;
    5) run_exp5 ;;
    6) run_exp6 ;;
    *) echo "unknown experiment: $1"; usage ;;
  esac
}

print_summary() {
  if [[ ! -f "$SUMMARY_CSV" ]]; then
    return
  fi

  echo
  echo "===== Summary ($SUMMARY_CSV) ====="

  if command -v column >/dev/null 2>&1; then
    column -s, -t < "$SUMMARY_CSV"
  else
    cat "$SUMMARY_CSV"
  fi
}

# -----------------------------
# Main
# -----------------------------

[[ $# -eq 0 ]] && usage

if [[ "$1" == "all" ]]; then
  for i in 1 2 3 4 5 6; do
    dispatch "$i"
  done
else
  for arg in "$@"; do
    dispatch "$arg"
  done
fi

print_summary