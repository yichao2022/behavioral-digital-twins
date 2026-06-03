from pathlib import Path
#!/usr/bin/env python3
"""
Table 1 — MiroThinker-1.7-mini (mixed-logit static frontier).

Run in Terminal.app:
  python3 table1_mirothinker_static.py build-grid
  python3 table1_mirothinker_static.py --preflight
  python3 table1_mirothinker_static.py run-llm
  python3 table1_mirothinker_static.py metrics
  python3 table1_mirothinker_static.py all

API backend (first match unless TABLE1_MIRO_BACKEND is set):
  - local (default): http://127.0.0.1:8001/v1 → Qwen2-72B-4bit via vllm-mlx
  - dashscope: DASHSCOPE_API_KEY → qwen2.5-72b-instruct
  - openrouter: OPENROUTER_API_KEY → qwen/qwen-2.5-72b-instruct

Keys may be loaded from ~/.openclaw/openclaw.json env block if not exported.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone

import certifi

from table1_metrics_lib import compute_lambda_sensitivity
from table1_llm_utils import (
    bdt_messages,
    extract_message_text,
    parse_response,
)

WORKSPACE = str(Path(__file__).resolve().parent)
INPUT_GRID = os.path.join(WORKSPACE, "bdt_eval_grid_updated.csv")
INPUT_COEF = os.path.join(WORKSPACE, "analysis_output", "mxl_coefs.csv")
STATIC_GRID = os.path.join(WORKSPACE, "bdt_eval_grid_static.csv")

RAW_PARTIAL = os.path.join(WORKSPACE, "llm_raw_outputs_mirothinker_unconstrained_partial.csv")
PARSED_PARTIAL = os.path.join(WORKSPACE, "llm_parsed_outputs_mirothinker_unconstrained_partial.csv")
RAW_FINAL = os.path.join(WORKSPACE, "llm_raw_outputs_mirothinker_unconstrained.csv")
PARSED_FINAL = os.path.join(WORKSPACE, "llm_parsed_outputs_mirothinker_unconstrained.csv")
ANCHOR_OUT = os.path.join(WORKSPACE, "static_bdt_anchor_mirothinker.csv")
METRICS_OUT = os.path.join(WORKSPACE, "metrics_table1_mirothinker_static_lambda_sensitivity.csv")
LOCK_FILE = os.path.join(WORKSPACE, "table1_mirothinker_resume.lock")
OPENCLAW_JSON = os.path.expanduser("~/.openclaw/openclaw.json")

MODEL_LABEL = "MiroThinker-1.7-mini"
CONDITION = "prompt_unconstrained"
REPEATS = 10

RAW_FIELDS = [
    "model", "condition", "repeat", "state", "timestamp",
    "prompt_text", "response_raw",
]
PARSED_FIELDS = [
    "model", "condition", "repeat", "state", "wait", "eff", "se", "P_static",
    "raw_response", "decision", "probability_0_100", "probability_0_1",
    "reasoning", "parse_success",
]

_HTTPS_OPENER: urllib.request.OpenerDirector | None = None
_API_URL = ""
_API_KEY = ""
_API_MODEL = ""
_API_TIMEOUT = 120
_API_BACKEND = ""


def _https_opener() -> urllib.request.OpenerDirector:
    global _HTTPS_OPENER
    if _HTTPS_OPENER is None:
        ctx = ssl.create_default_context(cafile=certifi.where())
        _HTTPS_OPENER = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ctx),
        )
    return _HTTPS_OPENER


def load_openclaw_env() -> dict[str, str]:
    if not os.path.isfile(OPENCLAW_JSON):
        return {}
    try:
        with open(OPENCLAW_JSON, encoding="utf-8") as f:
            data = json.load(f)
        env = data.get("env") or {}
        return {k: str(v) for k, v in env.items() if v}
    except (OSError, json.JSONDecodeError):
        return {}


def _env(key: str) -> str:
    val = (os.environ.get(key) or "").strip()
    if val:
        return val
    return load_openclaw_env().get(key, "").strip()


def _local_model_reachable(base: str, api_key: str, timeout: float = 3.0) -> bool:
    url = base.rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with _https_opener().open(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _configure_local() -> bool:
    global _API_URL, _API_KEY, _API_MODEL, _API_TIMEOUT, _API_BACKEND
    base = (os.environ.get("MIRO_API_BASE") or "http://127.0.0.1:8001/v1").rstrip("/")
    key = os.environ.get("MIRO_API_KEY", "not-needed")
    model = os.environ.get(
        "TABLE1_MODEL",
        os.environ.get("MIRO_API_MODEL", "mirothinker-1.7-mini-4bit"),
    )
    if not _local_model_reachable(base, key):
        return False
    _API_BACKEND = "local"
    _API_URL = base + "/chat/completions"
    _API_KEY = key
    _API_MODEL = model
    _API_TIMEOUT = int(os.environ.get("TABLE1_API_TIMEOUT", "600"))
    return True


def _configure_dashscope() -> bool:
    global _API_URL, _API_KEY, _API_MODEL, _API_TIMEOUT, _API_BACKEND
    key = _env("DASHSCOPE_API_KEY")
    if not key:
        return False
    _API_BACKEND = "dashscope"
    _API_URL = os.environ.get(
        "DASHSCOPE_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    if not _API_URL.endswith("/chat/completions"):
        _API_URL = _API_URL.rstrip("/") + "/chat/completions"
    _API_KEY = key
    _API_MODEL = os.environ.get("TABLE1_MODEL", "qwen2.5-72b-instruct")
    _API_TIMEOUT = int(os.environ.get("TABLE1_API_TIMEOUT", "120"))
    return True


def _configure_openrouter() -> bool:
    global _API_URL, _API_KEY, _API_MODEL, _API_TIMEOUT, _API_BACKEND
    key = _env("OPENROUTER_API_KEY")
    if not key:
        return False
    _API_BACKEND = "openrouter"
    _API_URL = "https://openrouter.ai/api/v1/chat/completions"
    _API_KEY = key
    _API_MODEL = os.environ.get("TABLE1_MODEL", "qwen/qwen-2.5-72b-instruct")
    _API_TIMEOUT = int(os.environ.get("TABLE1_API_TIMEOUT", "180"))
    return True


def resolve_api() -> None:
    """Pick backend. Default local-only — no silent cloud fallback."""
    mode = (os.environ.get("TABLE1_MIRO_BACKEND") or "local").strip().lower()

    if mode == "local":
        if _configure_local():
            return
        sys.exit(
            "\n❌ Local vllm-mlx is not running on http://127.0.0.1:8001\n"
            "Start it first (Terminal):\n"
            "  launchctl kickstart -k \"gui/$(id -u)/ai.openclaw.vllm-mlx-mirothinker\"\n"
            "  # wait until model loads, then:\n"
            "  curl -H 'Authorization: Bearer not-needed' http://127.0.0.1:8001/v1/models\n\n"
            "Or run:  ~/.openclaw/workspace/table1/start_local_vllm_mirothinker.sh\n\n"
            "Table 1 Miro will NOT fall back to OpenRouter while TABLE1_MIRO_BACKEND=local.\n"
            "Cloud partial runs should be archived before restarting locally — see table1/README_mirothinker.md\n"
        )

    if mode == "dashscope":
        if _configure_dashscope():
            return
        sys.exit("❌ TABLE1_MIRO_BACKEND=dashscope but DASHSCOPE_API_KEY is missing.")

    if mode == "openrouter":
        if _configure_openrouter():
            return
        sys.exit("❌ TABLE1_MIRO_BACKEND=openrouter but OPENROUTER_API_KEY is missing.")

    if mode == "auto":
        for fn in (_configure_local, _configure_dashscope, _configure_openrouter):
            if fn():
                return
        sys.exit("❌ No Qwen backend available (tried local → dashscope → openrouter).")

    sys.exit(f"❌ Unknown TABLE1_MIRO_BACKEND={mode!r}. Use local|dashscope|openrouter|auto.")


def detect_cursor_proxy() -> str | None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        val = (os.environ.get(key) or "").strip()
        if val and "127.0.0.1" in val:
            return f"{key}={val}"
    if os.environ.get("__CURSOR_SANDBOX_ENV_RESTORE"):
        return "Cursor sandbox (API calls will fail — use Terminal.app)"
    return None


def load_coefs(path: str) -> dict[str, float]:
    coefs: dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                coefs[row[0].strip('"')] = float(row[1])
    return coefs


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def build_static_grid() -> None:
    coefs = load_coefs(INPUT_COEF)
    rows_out = []
    with open(INPUT_GRID, newline="") as f:
        for row in csv.DictReader(f):
            wait, eff, se = float(row["wait"]), float(row["eff"]), float(row["se"])
            u = coefs["WaitTime"] * wait + coefs["VaccineEfficacy"] * eff + coefs["SideEffects"] * se
            rows_out.append({
                "state": row["state"],
                "wait": row["wait"],
                "eff": row["eff"],
                "se": row["se"],
                "U_static": f"{u:.8f}",
                "P_static": f"{sigmoid(u):.8f}",
                "prompt_unconstrained": (row.get("prompt_unconstrained") or "").strip(),
            })
    fields = ["state", "wait", "eff", "se", "U_static", "P_static", "prompt_unconstrained"]
    with open(STATIC_GRID, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {STATIC_GRID} ({len(rows_out)} rows)")


def _api_body_messages(messages: list[dict], max_tokens: int = 256) -> bytes:
    payload: dict = {
        "model": _API_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # MiroThinker chat template: disable agent thinking blocks (required for 3-line parse).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if _API_BACKEND == "local":
        payload["max_tokens"] = min(max_tokens, 256)
    return json.dumps(payload).encode()


def preflight_api() -> None:
    resolve_api()
    print(f"Using backend={_API_BACKEND} url={_API_URL} model={_API_MODEL}")
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    if _API_BACKEND == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/openclaw/bdt-table1"
        headers["X-Title"] = "BDT Table1 Qwen72B"
    req = urllib.request.Request(
        _API_URL,
        data=_api_body_messages(
            bdt_messages(
                "Wait 0 months, vaccine effectiveness 0.3, side effect risk 0.0. "
                "Would you get vaccinated now?"
            ),
            max_tokens=128,
        ),
        headers=headers,
        method="POST",
    )
    with _https_opener().open(req, timeout=_API_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())
    if "error" in data:
        raise RuntimeError(data["error"])
    raw = extract_message_text(data)
    parsed = parse_response(raw)
    if parsed["parse_success"] != "True":
        raise RuntimeError(
            f"preflight parse failed (is enable_thinking off?): {raw[:200]!r}"
        )
    if "<think>" in raw.lower() or "the user asks" in raw.lower():
        raise RuntimeError(
            "preflight still in thinking/meta mode — restart vllm or check "
            "chat_template_kwargs enable_thinking=false"
        )
    print(f"API preflight OK backend={_API_BACKEND} model={_API_MODEL} timeout={_API_TIMEOUT}s")


def _call_messages(messages: list[dict], max_retries: int = 2) -> tuple[str, bool]:
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    if _API_BACKEND == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/openclaw/bdt-table1"
        headers["X-Title"] = "BDT Table1 MiroThinker"
    req = urllib.request.Request(
        _API_URL,
        data=_api_body_messages(messages),
        headers=headers,
        method="POST",
    )
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            with _https_opener().open(req, timeout=_API_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            if "error" in data:
                raise RuntimeError(data["error"])
            return extract_message_text(data), False
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(3)
    return f"ERROR: {last_err}", True


def call_llm_for_row(row: dict) -> tuple[str, bool, str]:
    """Returns (raw_text, is_error, prompt_used)."""
    prompt = get_prompt(row)
    raw, err = _call_messages(bdt_messages(prompt))
    return raw, err, prompt


def task_key(state: str, repeat: int | str) -> tuple[str, str]:
    return (str(state), str(repeat))


def load_grid() -> list[dict]:
    if not os.path.exists(STATIC_GRID):
        build_static_grid()
    with open(STATIC_GRID, newline="") as f:
        return list(csv.DictReader(f))


def load_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_completed_successful() -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    for row in load_rows(PARSED_PARTIAL):
        if str(row.get("parse_success", "")).lower() in ("true", "1", "yes"):
            done.add(task_key(row["state"], row["repeat"]))
    return done


@contextmanager
def run_lock():
    fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        sys.exit(f"Another run holds {LOCK_FILE}.")
    fd.write(str(os.getpid()))
    fd.flush()
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass


def _flat(text: str, limit: int = 4000) -> str:
    return " ".join((text or "").split())[:limit]


def ensure_partial_headers() -> None:
    if not os.path.exists(RAW_PARTIAL) or os.path.getsize(RAW_PARTIAL) == 0:
        with open(RAW_PARTIAL, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=RAW_FIELDS).writeheader()
    if not os.path.exists(PARSED_PARTIAL) or os.path.getsize(PARSED_PARTIAL) == 0:
        with open(PARSED_PARTIAL, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=PARSED_FIELDS).writeheader()


def append_row(path: str, fields: list[str], row: dict) -> None:
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def get_prompt(row: dict) -> str:
    prompt = (row.get("prompt_unconstrained") or "").strip()
    if prompt:
        return prompt
    return (
        f"You are evaluating a vaccination decision. "
        f"The current wait time is {row['wait']} months, vaccine effectiveness is {row['eff']}, "
        f"and side effect risk is {row['se']}. Would you get vaccinated now?\n"
        "Respond exactly in this format:\n"
        "Decision: Yes or No\n"
        "Probability: a number between 0 and 100\n"
        "Reasoning: one sentence"
    )


def print_status(grid: list[dict]) -> None:
    total = len(grid) * REPEATS
    ok = load_completed_successful()
    resolve_api()
    print("=" * 60)
    print("Table 1 MiroThinker-1.7-mini — status")
    print("=" * 60)
    print(f"Backend:         {_API_BACKEND}")
    print(f"API model:       {_API_MODEL}")
    print(f"Total calls:     {total}")
    print(f"Parse OK:        {len(ok)}")
    print(f"Remaining:       {total - len(ok)}")


def run_llm(grid: list[dict], sleep_s: float | None = None, *, skip_preflight: bool = False) -> None:
    proxy = detect_cursor_proxy()
    if proxy:
        sys.exit(
            f"\n❌ Refusing to run: {proxy}\n"
            "Use Terminal.app:\n"
            "  cd ~/.openclaw/workspace && python3 table1_mirothinker_static.py run-llm\n"
        )
    resolve_api()
    if sleep_s is None:
        sleep_s = 0.0 if _API_BACKEND == "local" else 1.0

    with run_lock():
        if not skip_preflight:
            preflight_api()
        ensure_partial_headers()
        completed = load_completed_successful()
        total = len(grid) * REPEATS
        pending = [
            (row["state"], rep, row)
            for row in grid
            for rep in range(1, REPEATS + 1)
            if task_key(row["state"], rep) not in completed
        ]
        print(f"Backend={_API_BACKEND} model={_API_MODEL}")
        print(f"Pending: {len(pending)} / {total}")
        if not pending:
            print("Nothing to run.")
            return

        errors = 0
        parse_fail_streak = 0
        t0 = time.time()
        for idx, (state, rep, row) in enumerate(pending, 1):
            prompt = get_prompt(row)
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            print(f"\n[{idx}/{len(pending)}] state={state} repeat={rep}")
            sys.stdout.flush()

            raw_text, is_error, prompt_used = call_llm_for_row(row)
            parsed = parse_response(raw_text)

            append_row(RAW_PARTIAL, RAW_FIELDS, {
                "model": MODEL_LABEL,
                "condition": CONDITION,
                "repeat": str(rep),
                "state": state,
                "timestamp": ts,
                "prompt_text": _flat(prompt_used, 800),
                "response_raw": _flat(raw_text, 4000),
            })
            append_row(PARSED_PARTIAL, PARSED_FIELDS, {
                "model": MODEL_LABEL,
                "condition": CONDITION,
                "repeat": str(rep),
                "state": state,
                "wait": row["wait"],
                "eff": row["eff"],
                "se": row["se"],
                "P_static": row["P_static"],
                "raw_response": _flat(raw_text, 4000),
                "decision": parsed["decision"],
                "probability_0_100": parsed["probability_0_100"],
                "probability_0_1": parsed["probability_0_1"],
                "reasoning": parsed["reasoning"],
                "parse_success": parsed["parse_success"],
            })

            if is_error or parsed["parse_success"] != "True":
                errors += 1
                parse_fail_streak += 1
                print(f"  ⚠️  {_flat(raw_text, 120)}")
            else:
                parse_fail_streak = 0
                print(f"  ✅ {parsed['decision']} P={parsed['probability_0_100']}")

            if idx >= 15 and errors / idx > 0.5:
                sys.exit(
                    f"\n❌ Aborting: {errors}/{idx} parse failures (>50%).\n"
                    "MiroThinker likely still has thinking enabled.\n"
                    "  1) ./table1/stop_mirothinker.sh\n"
                    "  2) ./table1/reset_mirothinker_partial.sh\n"
                    "  3) Pull latest table1_mirothinker_static.py and rerun preflight\n"
                )

            if idx % 10 == 0 or idx == len(pending):
                elapsed = time.time() - t0
                rate = idx / elapsed * 60 if elapsed > 0 else 0
                print(f"--- {len(completed) + idx}/{total} | {rate:.1f} calls/min | errors={errors} ---")

            if sleep_s > 0:
                time.sleep(sleep_s)

        print(f"\nBatch done in {time.time() - t0:.0f}s ({errors} errors)")


def dedupe_rows(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    out: dict[tuple, dict] = {}
    for row in rows:
        out[tuple(row[k] for k in key_fields)] = row
    return list(out.values())


def write_final_csvs() -> None:
    raw_rows = dedupe_rows(load_rows(RAW_PARTIAL), ("state", "repeat"))
    parsed_rows = dedupe_rows(load_rows(PARSED_PARTIAL), ("state", "repeat"))
    with open(RAW_FINAL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(raw_rows)
    with open(PARSED_FINAL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PARSED_FIELDS)
        w.writeheader()
        w.writerows(parsed_rows)
    ok = sum(1 for r in parsed_rows if str(r.get("parse_success", "")).lower() in ("true", "1", "yes"))
    print(f"Wrote {RAW_FINAL} ({len(raw_rows)} rows)")
    print(f"Wrote {PARSED_FINAL} ({len(parsed_rows)} rows, {ok} parse_ok)")


def run_metrics() -> None:
    parsed_path = PARSED_FINAL if os.path.isfile(PARSED_FINAL) else PARSED_PARTIAL
    if not os.path.isfile(parsed_path):
        sys.exit(f"No parsed outputs at {parsed_path}. Run run-llm first.")
    compute_lambda_sensitivity(
        parsed_path=parsed_path,
        static_grid_path=STATIC_GRID,
        metrics_out=METRICS_OUT,
        anchor_out=ANCHOR_OUT,
        model_label=MODEL_LABEL,
        print_table=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 1 MiroThinker-1.7-mini static pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["build-grid", "run-llm", "metrics", "all", "status"],
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        proxy = detect_cursor_proxy()
        if proxy:
            sys.exit(f"❌ Proxy: {proxy}")
        preflight_api()
        return

    if args.command == "build-grid":
        build_static_grid()
        return

    grid = load_grid()

    if args.command == "status":
        print_status(grid)
        return

    if args.command in ("run-llm", "all") and not args.finalize_only:
        run_llm(grid, sleep_s=args.sleep, skip_preflight=args.no_preflight)

    if args.command in ("metrics", "all") or args.finalize_only:
        if not args.finalize_only and args.command == "metrics":
            write_final_csvs()
        elif args.finalize_only or args.command == "all":
            write_final_csvs()
        run_metrics()


if __name__ == "__main__":
    main()
