#!/bin/bash
# Stop MiroThinker Table-1 runner and optional vllm-mlx on :8001.
set -euo pipefail
WS="$HOME/.openclaw/workspace"
PIDFILE="$WS/table1/mirothinker_local_run.pid"
if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE")
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
fi
pkill -f 'table1_mirothinker_static.py' 2>/dev/null || true
rm -f "$WS/table1_mirothinker_resume.lock"
echo "Stopped MiroThinker Table-1 runner."

if [[ "${1:-}" == "--all" || "${STOP_MIRO_VLLM:-}" == "1" ]]; then
  launchctl bootout "gui/$(id -u)/ai.openclaw.vllm-mlx-mirothinker" 2>/dev/null || true
  sleep 2
  if lsof -iTCP:8001 -sTCP:LISTEN >/dev/null 2>&1; then
    pid=$(lsof -tiTCP:8001 -sTCP:LISTEN | head -1)
    kill "$pid" 2>/dev/null || true
  fi
  if lsof -iTCP:8001 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "⚠️  :8001 still listening — run: launchctl bootout gui/\$(id -u)/ai.openclaw.vllm-mlx-mirothinker"
  else
    echo "Stopped vllm-mlx-mirothinker (:8001)."
  fi
fi
