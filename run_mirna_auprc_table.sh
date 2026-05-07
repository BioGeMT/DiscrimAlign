#!/usr/bin/env bash
set -euo pipefail

THREADS=65
STEP_SCALES="0.00001,0.00005,0.0001,0.0005"
MAX_ITERS="100"
EVAL_SPLITS="hejret_test,manakov_test,manakov_leftout"
RESULTS_DIR="results/case_study_for_mirna"

if command -v uv >/dev/null 2>&1; then
  UV="uv"
elif command -v uv.exe >/dev/null 2>&1; then
  UV="uv.exe"
elif [[ -n "${USERPROFILE:-}" && -x "$(cygpath -u "$USERPROFILE")/.local/bin/uv.exe" ]]; then
  UV="$(cygpath -u "$USERPROFILE")/.local/bin/uv.exe"
elif [[ -n "${LOCALAPPDATA:-}" && -x "$(cygpath -u "$LOCALAPPDATA")/Programs/uv/uv.exe" ]]; then
  UV="$(cygpath -u "$LOCALAPPDATA")/Programs/uv/uv.exe"
else
  echo "uv was not found by this Bash shell. Run the script from a shell where uv is on PATH, or add uv to Git Bash PATH." >&2
  exit 127
fi

mkdir -p logs

run_case () {
  local train_family="$1"
  local run_tag="$2"

  echo "Training once on ${train_family}_train; evaluating ${EVAL_SPLITS}"

  "$UV" run python case_study_for_mirna/case_study_mirna.py \
    --dataset "${train_family}" \
    --eval-splits "${EVAL_SPLITS}" \
    --aligner-modes local,global \
    --substitution-modes simple,symmetric,general \
    --gap-modes linear,affine \
    --stepfunctions power,constant \
    --step-scales "${STEP_SCALES}" \
    --max-iters "${MAX_ITERS}" \
    --num-threads "${THREADS}" \
    --results-dir "${RESULTS_DIR}" \
    --run-tag "${run_tag}"
}

run_case "hejret" "train_hejret_eval_all"
run_case "manakov" "train_manakov_eval_all"

echo "All miRNA AUPRC table runs completed."
