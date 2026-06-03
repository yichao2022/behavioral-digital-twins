#!/bin/bash
# Run Table 1 Qwen on local vllm-mlx in background (Terminal.app).
set -euo pipefail
cd "$HOME/.openclaw/workspace"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy __CURSOR_SANDBOX_ENV_RESTORE
export TABLE1_QWEN_BACKEND=local

LOG="$HOME/.openclaw/workspace/table1/qwen72b_local_run.log"
PIDFILE="$HOME/.openclaw/workspace/table1/qwen72b_local_run.pid"

if [[ -f "$PIDFILE" ]]; then
  oldpid=$(cat "$PIDFILE")
  if kill -0 "$oldpid" 2>/dev/null; then
    echo "Already running (PID $oldpid). tail -f $LOG"
    exit 0
  fi
fi

if ! curl -sf -m 5 -H "Authorization: Bearer not-needed" http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "Local vllm not up — starting..."
  "$HOME/.openclaw/workspace/table1/start_local_vllm_qwen.sh"
fi

python3 table1_qwen72b_static.py --preflight

nohup python3 -u table1_qwen72b_static.py run-llm --no-preflight >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started PID $(cat "$PIDFILE")"
echo "  log:  $LOG"
echo "  tail: tail -f $LOG"
echo "  status: python3 table1_qwen72b_static.py status"
