#!/bin/bash
# Stop Table 1 Qwen72B background job and release lock.
set -euo pipefail
WS="$HOME/.openclaw/workspace"
PIDFILE="$WS/table1/qwen72b_local_run.pid"

if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping table1_qwen72b PID $pid ..."
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi

pkill -f 'table1_qwen72b_static.py' 2>/dev/null && echo "Killed table1_qwen72b_static.py" || echo "No table1_qwen72b_static.py process"
rm -f "$WS/table1_qwen72b_resume.lock"
echo "Done. Qwen Table-1 runner stopped (vllm-mlx-qwen2 on :8000 is left running)."
