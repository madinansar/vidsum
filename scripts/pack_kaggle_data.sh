#!/usr/bin/env bash
# Bundle everything training needs (no raw videos, no legacy files) for upload as a Kaggle Dataset.
#   bash scripts/pack_kaggle_data.sh            -> dist/vidsum-data.zip  (~230 MB)
# Upload it at kaggle.com/datasets -> New Dataset; Kaggle unzips it, so official/ and features/ appear
# under /kaggle/input/<dataset-slug>/ (the notebook finds them automatically).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${VIDSUM_DATA:-"${ROOT}/data"}
OUT="${ROOT}/dist/vidsum-data.zip"
mkdir -p "${ROOT}/dist"
rm -f "${OUT}"
cd "${DATA}"
for required in official/splits/tvsum_splits.json official/tvsum/eccv16_dataset_tvsum_google_pool5.h5 \
                official/summe/eccv16_dataset_summe_google_pool5.h5 official/tvsum/ydata-anno.tsv; do
  [[ -f "${required}" ]] || { echo "missing ${DATA}/${required}" >&2; exit 1; }
done
FEATURE_DIRS=$(find features -mindepth 1 -maxdepth 1 -type d ! -name legacy | sort)
# -X: no macOS extended attributes; zip follows symlinks by default, so files are stored for real.
zip -q -r -X "${OUT}" README.md official ${FEATURE_DIRS} -x '*.DS_Store'
echo "Wrote ${OUT} ($(du -h "${OUT}" | cut -f1))"
unzip -l "${OUT}" | tail -n +4 | awk '{print $4}' | grep -v '/$' | sed '/^$/d' | head -20
