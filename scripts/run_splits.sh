#!/usr/bin/env bash
# Train every sigma x split for one config, then run label-free model selection.
#   bash scripts/run_splits.sh configs/casum_googlenet.yaml tvsum
#   SIGMAS="0.6" SPLITS="0" EPOCHS=5 bash scripts/run_splits.sh configs/casum_googlenet.yaml tvsum   # smoke test
set -euo pipefail
CONFIG=${1:?config path}
DATASET=${2:-tvsum}
SIGMAS=${SIGMAS:-"0.5 0.6 0.7 0.8 0.9"}
SPLITS=${SPLITS:-"0 1 2 3 4"}
EXP=${EXP:-$(basename "${CONFIG}" .yaml)}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Interpreter: $PYTHON, else the local project venv if present, else python3 from PATH (Kaggle/Colab).
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${ROOT}/../.venv/bin/python" ]]; then PYTHON="${ROOT}/../.venv/bin/python"; else PYTHON=python3; fi
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"  # works without `pip install -e .`
EPOCH_ARGS=()
[[ -n "${EPOCHS:-}" ]] && EPOCH_ARGS=(--n_epochs "${EPOCHS}")

cd "${ROOT}"
for sigma in ${SIGMAS}; do
  for split in ${SPLITS}; do
    "${PYTHON}" -m vidsum.train --config "${CONFIG}" --dataset "${DATASET}" --exp "${EXP}" \
      --split "${split}" --reg_factor "${sigma}" ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"}
  done
done
"${PYTHON}" -m vidsum.evaluate --exp "${EXP}" --dataset "${DATASET}"
