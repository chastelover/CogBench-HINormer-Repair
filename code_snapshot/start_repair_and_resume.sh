#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$PACKAGE_DIR/repair_and_resume.pid"
LOG_FILE="$PACKAGE_DIR/repair_and_resume.log"
START_LOCK="$PACKAGE_DIR/start.lock"

is_running() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  if [[ -r "/proc/$pid/cmdline" ]]; then
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$PACKAGE_DIR/repair_and_resume_foreground.sh"
  fi
}

exec 9> "$START_LOCK"
if ! flock -n 9; then
  printf 'ERROR: another start request is active\n' >&2
  exit 2
fi

OLD_PID=""
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "$PID_FILE")"
fi
if is_running "$OLD_PID"; then
  printf 'ERROR: repair/resume is already running (PID %s)\n' "$OLD_PID" >&2
  exit 2
fi

: > "$LOG_FILE"
nohup bash "$PACKAGE_DIR/repair_and_resume_foreground.sh" >> "$LOG_FILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
sleep 1
if ! is_running "$PID"; then
  set +e
  wait "$PID"
  CODE=$?
  set -e
  printf 'ERROR: launcher exited during startup (code %s)\n' "$CODE" >&2
  tail -n 50 "$LOG_FILE" >&2 || true
  exit "$CODE"
fi
printf 'STARTED PID=%s\n' "$PID"
printf 'LOG=%s\n' "$LOG_FILE"
printf 'Follow with: tail -f %q\n' "$LOG_FILE"

