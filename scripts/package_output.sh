#!/usr/bin/env bash
# Package ZeroED outputs, logs, and configs into a downloadable zip file.
# Usage: bash scripts/package_output.sh [dataset_name] [destination_dir]

DATASET="${1:-rayyan}"
DEST_DIR="${2:-.}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ZIP_NAME="zeroed_output_${DATASET}_${TIMESTAMP}.zip"
ZIP_PATH="${DEST_DIR}/${ZIP_NAME}"

echo "Packaging ZeroED outputs for dataset '${DATASET}' into ${ZIP_PATH}..."

# Gather target files
zip -q -r "${ZIP_PATH}" result/ run_config.yaml 2>/dev/null || true
[ -f "llm_calls.jsonl" ] && zip -q -u "${ZIP_PATH}" llm_calls.jsonl 2>/dev/null || true
[ -f "run.log" ] && zip -q -u "${ZIP_PATH}" run.log 2>/dev/null || true
for f in run_stdout_*.log summary_*.csv; do
    [ -f "$f" ] && zip -q -u "${ZIP_PATH}" "$f" 2>/dev/null || true
done

if [ -f "${ZIP_PATH}" ]; then
    SIZE=$(du -h "${ZIP_PATH}" | cut -f1)
    echo "✓ Created ${ZIP_PATH} (${SIZE})"
else
    echo "✗ Failed to create zip file."
    exit 1
fi
