#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$PACKAGE_DIR/repair_and_resume.pid"
LOG_FILE="$PACKAGE_DIR/repair_and_resume.log"
PROJECT="${COGBENCH_PROJECT:-/CogBench/CogBench_Final_Experiments_V5_2}"
OUTPUT_ARG="${COGBENCH_OUTPUT_ROOT:-outputs/final_paper_v5_2}"
CORE_PYTHON="${COGBENCH_CORE_PYTHON:-/usr/local/miniconda3/envs/py312/bin/python}"
RECOVERY_ROOT="${COGBENCH_PLATEAU_V2_RECOVERY:-/CogBench/CogBench_V5_2_Plateau_V2_Recovery}"

if [[ "$OUTPUT_ARG" = /* ]]; then OUTPUT_ROOT="$OUTPUT_ARG"; else OUTPUT_ROOT="$PROJECT/$OUTPUT_ARG"; fi
PID=""
if [[ -f "$PID_FILE" ]]; then PID="$(tr -d '[:space:]' < "$PID_FILE")"; fi
if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
  printf 'REPAIR_CHAIN=RUNNING PID=%s\n' "$PID"
else
  printf 'REPAIR_CHAIN=NOT_RUNNING\n'
fi

LATEST_STATE="$(find "$PACKAGE_DIR/runs" -name state.json -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
if [[ -n "$LATEST_STATE" && -f "$LATEST_STATE" ]]; then
  printf 'REPAIR_STATE=%s\n' "$LATEST_STATE"
  cat "$LATEST_STATE"
  WORKER_PROGRESS="$(dirname "$LATEST_STATE")/worker/worker_progress.json"
  if [[ -f "$WORKER_PROGRESS" ]]; then
    printf '\nWORKER_PROGRESS=%s\n' "$WORKER_PROGRESS"
    cat "$WORKER_PROGRESS"
  fi
else
  printf 'REPAIR_STATE=NOT_CREATED\n'
fi

if [[ -f "$RECOVERY_ROOT/status_autorepair.sh" ]]; then
  printf '\nCANONICAL_SUITE_STATUS\n'
  COGBENCH_PROJECT="$PROJECT" COGBENCH_OUTPUT_ROOT="$OUTPUT_ROOT" \
    COGBENCH_CORE_PYTHON="$CORE_PYTHON" bash "$RECOVERY_ROOT/status_autorepair.sh" || true
fi

if [[ -f "$OUTPUT_ROOT/final_release_report.json" ]]; then
  printf '\nFINAL_RELEASE\n'
  "$CORE_PYTHON" "$PACKAGE_DIR/verify_final.py" --output-root "$OUTPUT_ROOT" || true
else
  printf '\nFINAL_RELEASE=NOT_CREATED_YET\n'
fi

if [[ -f "$LOG_FILE" ]]; then
  printf '\nLOG_TAIL=%s\n' "$LOG_FILE"
  tail -n 80 "$LOG_FILE"
fi

