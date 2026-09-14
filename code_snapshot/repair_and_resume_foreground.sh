#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${COGBENCH_PROJECT:-/CogBench/CogBench_Final_Experiments_V5_2}"
OUTPUT_ARG="${COGBENCH_OUTPUT_ROOT:-outputs/final_paper_v5_2}"
CORE_PYTHON="${COGBENCH_CORE_PYTHON:-/usr/local/miniconda3/envs/py312/bin/python}"
OFFICIAL_PYTHON="${COGBENCH_OFFICIAL_PYTHON:-/usr/local/miniconda3/envs/cogbench-hinormer-cpu/bin/python}"
OFFICIAL_DEVICE="${COGBENCH_OFFICIAL_DEVICE:-cpu}"
RECOVERY_ROOT="${COGBENCH_PLATEAU_V2_RECOVERY:-/CogBench/CogBench_V5_2_Plateau_V2_Recovery}"

if [[ "$OUTPUT_ARG" = /* ]]; then
  OUTPUT_ROOT="$OUTPUT_ARG"
else
  OUTPUT_ROOT="$PROJECT/$OUTPUT_ARG"
fi

printf '[%s] starting isolated official-HINormer extension\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$CORE_PYTHON" -u "$PACKAGE_DIR/official_hinormer_repair.py" \
  --project "$PROJECT" \
  --output-root "$OUTPUT_ROOT" \
  --official-python "$OFFICIAL_PYTHON" \
  --official-device "$OFFICIAL_DEVICE" \
  --caps 8000,12000

if [[ ! -f "$RECOVERY_ROOT/start_autorepair.sh" ]]; then
  printf 'ERROR: plateau-v2 recovery controller is missing: %s\n' "$RECOVERY_ROOT" >&2
  exit 2
fi

printf '[%s] extension committed; resuming the canonical suite\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COGBENCH_PROJECT="$PROJECT" \
COGBENCH_OUTPUT_ROOT="$OUTPUT_ROOT" \
COGBENCH_CORE_PYTHON="$CORE_PYTHON" \
COGBENCH_OFFICIAL_PYTHON="$OFFICIAL_PYTHON" \
COGBENCH_OFFICIAL_DEVICE="$OFFICIAL_DEVICE" \
bash "$RECOVERY_ROOT/start_autorepair.sh"

RECOVERY_PID_FILE="$RECOVERY_ROOT/autorepair_console.pid"
while true; do
  RECOVERY_PID=""
  if [[ -f "$RECOVERY_PID_FILE" ]]; then
    RECOVERY_PID="$(tr -d '[:space:]' < "$RECOVERY_PID_FILE")"
  fi
  if [[ ! "$RECOVERY_PID" =~ ^[0-9]+$ ]] || ! kill -0 "$RECOVERY_PID" 2>/dev/null; then
    break
  fi
  printf '[%s] canonical suite still running (PID %s)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RECOVERY_PID"
  sleep 20
done

"$CORE_PYTHON" -u "$PACKAGE_DIR/verify_final.py" \
  --output-root "$OUTPUT_ROOT"
printf '[%s] COMPLETE: official repair and strict final release gate passed\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

