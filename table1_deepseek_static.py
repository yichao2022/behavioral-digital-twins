from pathlib import Path
#!/usr/bin/env python3
"""
Table 1 — DeepSeek V4 Pro (mixed-logit static frontier).

Steps:
  1. build-grid   — bdt_eval_grid_static.csv from mxl_coefs + bdt_eval_grid_updated
  2. run-llm      — prompt_unconstrained only, 64 states × 10 repeats (checkpoint/resume)
  3. metrics      — Static-BDT anchor (Eq. 3) + Table 1 metrics vs P_static

Run in Terminal.app (not Cursor sandbox):
  python3 table1_deepseek_static.py build-grid
  python3 table1_deepseek_static.py --preflight
  python3 table1_deepseek_static.py run-llm
  python3 table1_deepseek_static.py metrics

Or all at once:
  python3 table1_deepseek_static.py all
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
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from statistics import mean, stdev

import certifi

WORKSPACE = str(Path(__file__).resolve().parent)
INPUT_GRID = os.path.join(WORKSPACE, "bdt_eval_grid_updated.csv")
INPUT_COEF = os.path.join(WORKSPACE, "analysis_output", "mxl_coefs.csv")
STATIC_GRID = os.path.join(WORKSPACE, "bdt_eval_grid_static.csv")

RAW_PARTIAL = os.path.join(WORKSPACE, "llm_raw_outputs_deepseek_unconstrained_partial.csv")
PARSED_PARTIAL = os.path.join(WORKSPACE, "llm_parsed_outputs_deepseek_unconstrained_partial.csv")
RAW_FINAL = os.path.join(WORKSPACE, "llm_raw_outputs_deepseek_unconstrained.csv")
PARSED_FINAL = os.path.join(WORKSPACE, "llm_parsed_outputs_deepseek_unconstrained.csv")
ANCHOR_OUT = os.path.join(WORKSPACE, "static_bdt_anchor_deepseek.csv")
METRICS_OUT = os.path.join(WORKSPACE, "metrics_table1_deepseek_static.csv")
LOCK_FILE = os.path.join(WORKSPACE, "table1_deepseek_resume.lock")

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/v1/chat/completions"
API_MODEL = os.environ.get("TABLE1_MODEL", "deepseek-v4-pro")
MODEL_LABEL = "DeepSeek V4 Pro"

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


def _https_opener() -> urllib.request.OpenerDirector:
    global _HTTPS_OPENER
    if _HTTPS_OPENER is None:
        ctx = ssl.create_default_context(cafile=certifi.where())
        _HTTPS_OPENER = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ctx),
        )
    return _HTTPS_OPENER


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
        next(reader)  # header
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
    for k in ("WaitTime", "VaccineEfficacy", "SideEffects"):
        if k not in coefs:
            sys.exit(f"Missing coefficient {k}")

    rows_out = []
    with open(INPUT_GRID, newline="") as f:
        for row in csv.DictReader(f):
            wait = float(row["wait"])
            eff = float(row["eff"])
            se = float(row["se"])
            u = (
                coefs["WaitTime"] * wait
                + coefs["VaccineEfficacy"] * eff
                + coefs["SideEffects"] * se
            )
            p_static = sigmoid(u)
            rows_out.append({
                "state": row["state"],
                "wait": row["wait"],
                "eff": row["eff"],
                "se": row["se"],
                "U_static": f"{u:.8f}",
                "P_static": f"{p_static:.8f}",
                "prompt_unconstrained": (row.get("prompt_unconstrained") or "").strip(),
            })

    fields = ["state", "wait", "eff", "se", "U_static", "P_static", "prompt_unconstrained"]
    with open(STATIC_GRID, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    ps = [float(r["P_static"]) for r in rows_out]
    print(f"Wrote {STATIC_GRID} ({len(rows_out)} rows)")
    print(f"P_static ∈ [{min(ps):.4f}, {max(ps):.4f}]")


def _api_body(messages: list[dict], max_tokens: int = 512) -> bytes:
    return json.dumps({
        "model": API_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }).encode()


def _extract_content(data: dict) -> str:
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if text:
        return text
    for key in ("reasoning_content", "reasoning"):
        alt = (msg.get(key) or "").strip()
        if alt:
            return alt
    raise RuntimeError(f"empty model content; keys={list(msg.keys())}")


def preflight_api() -> None:
    req = urllib.request.Request(
        API_URL,
        data=_api_body([{"role": "user", "content": "Reply exactly: OK"}], max_tokens=8),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with _https_opener().open(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    if "error" in data:
        raise RuntimeError(data["error"])
    _extract_content(data)
    print(f"API preflight OK (model={API_MODEL})")


def call_llm(user_prompt: str, max_retries: int = 2) -> tuple[str, bool]:
    req = urllib.request.Request(
        API_URL,
        data=_api_body([{"role": "user", "content": user_prompt}]),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            with _https_opener().open(req, timeout=90) as resp:
                data = json.loads(resp.read().decode())
            if "error" in data:
                raise RuntimeError(data["error"])
            return _extract_content(data), False
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(3)
    return f"ERROR: {last_err}", True


def parse_response(text: str) -> dict:
    if text.startswith("ERROR:"):
        return {
            "decision": "",
            "probability_0_100": "",
            "probability_0_1": "",
            "reasoning": text[:500],
            "parse_success": "False",
        }

    decision = ""
    p100 = ""
    p01 = ""
    reasoning = ""

    dm = re.search(r"Decision\s*:\s*(Yes|No)", text, re.IGNORECASE)
    if dm:
        decision = dm.group(1).capitalize()

    pm = re.search(r"Probability\s*:\s*(\d+(?:\.\d+)?)", text)
    if pm:
        val = float(pm.group(1))
        if 0 <= val <= 100:
            p100 = str(val)
            p01 = str(val / 100.0)

    rm = re.search(r"Reasoning\s*:\s*(.+)", text, re.DOTALL)
    if rm:
        reasoning = rm.group(1).strip()[:500]

    ok = decision in ("Yes", "No") and p01 != ""
    return {
        "decision": decision,
        "probability_0_100": p100,
        "probability_0_1": p01,
        "reasoning": reasoning,
        "parse_success": "True" if ok else "False",
    }


def task_key(state: str, repeat: int | str) -> tuple[str, str]:
    return (str(state), str(repeat))


def load_grid() -> list[dict]:
    if not os.path.exists(STATIC_GRID):
        sys.exit(f"Missing {STATIC_GRID}. Run: python3 table1_deepseek_static.py build-grid")
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
        sys.exit(f"Another run holds {LOCK_FILE}. Stop duplicate processes first.")
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
    parsed = load_rows(PARSED_PARTIAL)
    print("=" * 60)
    print("Table 1 DeepSeek — status")
    print("=" * 60)
    print(f"Grid:            {STATIC_GRID}")
    print(f"Model:           {API_MODEL}")
    print(f"Total calls:     {total} (64 × {REPEATS})")
    print(f"Parse OK:        {len(ok)}")
    print(f"Remaining:       {total - len(ok)}")
    print(f"Partial parsed:  {PARSED_PARTIAL}")


def run_llm(grid: list[dict], sleep_s: float = 1.0, *, skip_preflight: bool = False) -> None:
    proxy = detect_cursor_proxy()
    if proxy:
        sys.exit(
            f"\n❌ Refusing to run: {proxy}\n"
            "Use Terminal.app:\n"
            "  cd ~/.openclaw/workspace && python3 table1_deepseek_static.py run-llm\n"
        )

    with run_lock():
        if not skip_preflight:
            preflight_api()
        ensure_partial_headers()
        completed = load_completed_successful()
        total = len(grid) * REPEATS

        pending: list[tuple[str, int, dict]] = []
        for row in grid:
            for rep in range(1, REPEATS + 1):
                if task_key(row["state"], rep) not in completed:
                    pending.append((row["state"], rep, row))

        print(f"Pending: {len(pending)} / {total}")
        if not pending:
            print("Nothing to run.")
            return

        errors = 0
        t0 = time.time()
        for idx, (state, rep, row) in enumerate(pending, 1):
            prompt = get_prompt(row)
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            print(f"\n[{idx}/{len(pending)}] state={state} repeat={rep}")
            sys.stdout.flush()

            raw_text, is_error = call_llm(prompt)
            parsed = parse_response(raw_text)
            p_static = row["P_static"]

            append_row(RAW_PARTIAL, RAW_FIELDS, {
                "model": MODEL_LABEL,
                "condition": CONDITION,
                "repeat": str(rep),
                "state": state,
                "timestamp": ts,
                "prompt_text": _flat(prompt, 800),
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
                "P_static": p_static,
                "raw_response": _flat(raw_text, 4000),
                "decision": parsed["decision"],
                "probability_0_100": parsed["probability_0_100"],
                "probability_0_1": parsed["probability_0_1"],
                "reasoning": parsed["reasoning"],
                "parse_success": parsed["parse_success"],
            })

            if is_error or parsed["parse_success"] != "True":
                errors += 1
                print(f"  ⚠️  {_flat(raw_text, 120)}")
            else:
                print(f"  ✅ {parsed['decision']} P={parsed['probability_0_100']}")

            if idx % 25 == 0 or idx == len(pending):
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


def write_final_csvs() -> list[dict]:
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

    print(f"Wrote {RAW_FINAL} ({len(raw_rows)} rows)")
    print(f"Wrote {PARSED_FINAL} ({len(parsed_rows)} rows)")
    return parsed_rows


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _spearman(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho)
    except ImportError:
        # Rank-based Pearson fallback
        def ranks(vals: list[float]) -> list[float]:
            order = sorted(range(n), key=lambda i: vals[i])
            r = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    r[order[k]] = avg_rank
                i = j + 1
            return r
        return _pearson(ranks(x), ranks(y))


def chr_violation_rate(
    state_preds: dict[str, float],
    state_meta: dict[str, dict],
    *,
    axis: str,
) -> float | None:
    """Fraction of comparable pairs that violate monotonicity (higher = worse)."""
    state_list = [
        (s, float(state_meta[s]["wait"]), float(state_meta[s]["eff"]), float(state_meta[s]["se"]), state_preds[s])
        for s in state_preds
        if s in state_meta
    ]
    violations = 0
    total = 0
    for i, (s1, w1, e1, se1, p1) in enumerate(state_list):
        for j, (s2, w2, e2, se2, p2) in enumerate(state_list):
            if i >= j:
                continue
            if axis == "wait" and e1 == e2 and se1 == se2 and w1 != w2:
                total += 1
                if (w1 > w2 and p1 > p2) or (w1 < w2 and p1 < p2):
                    violations += 1
            elif axis == "se" and w1 == w2 and e1 == e2 and se1 != se2:
                total += 1
                if (se1 > se2 and p1 > p2) or (se1 < se2 and p1 < p2):
                    violations += 1
    if total == 0:
        return None
    return violations / total


def select_lambda(p_llm: dict[str, float], p_static: dict[str, float]) -> tuple[float, dict[str, float]]:
    states = sorted(p_static.keys(), key=lambda s: int(s))
    best_lambda = 1.0
    best_mse = float("inf")
    best_pi: dict[str, float] = {}

    for i in range(11):
        lam = round(i * 0.1, 1)
        pi = {
            s: lam * p_llm[s] + (1.0 - lam) * p_static[s]
            for s in states
        }
        mse = mean((pi[s] - p_static[s]) ** 2 for s in states)
        if mse < best_mse - 1e-12 or (abs(mse - best_mse) <= 1e-12 and lam > best_lambda):
            best_mse = mse
            best_lambda = lam
            best_pi = pi
    return best_lambda, best_pi


def compute_metrics(parsed_rows: list[dict], grid: list[dict]) -> None:
    grid_lookup = {r["state"]: r for r in grid}
    p_static = {s: float(grid_lookup[s]["P_static"]) for s in grid_lookup}

    by_state: dict[str, list[float]] = defaultdict(list)
    for row in parsed_rows:
        if str(row.get("parse_success", "")).lower() not in ("true", "1", "yes"):
            continue
        by_state[row["state"]].append(float(row["probability_0_1"]))

    states = sorted(p_static.keys(), key=lambda s: int(s))
    p_llm_mean = {s: mean(by_state[s]) for s in states if s in by_state and by_state[s]}

    if len(p_llm_mean) < len(states):
        missing = set(states) - set(p_llm_mean)
        print(f"⚠️  Missing successful parses for states: {sorted(missing, key=int)[:10]}...")

    parse_ok = sum(
        1 for r in parsed_rows if str(r.get("parse_success", "")).lower() in ("true", "1", "yes")
    )
    parse_rate = parse_ok / len(parsed_rows) if parsed_rows else 0.0

    lam_star, pi_bdt = select_lambda(p_llm_mean, p_static)

    # Anchor output
    anchor_fields = [
        "state", "wait", "eff", "se", "P_static", "p_LLM_mean",
        "lambda_star", "pi_BDT",
    ]
    with open(ANCHOR_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=anchor_fields)
        w.writeheader()
        for s in states:
            if s not in p_llm_mean:
                continue
            g = grid_lookup[s]
            w.writerow({
                "state": s,
                "wait": g["wait"],
                "eff": g["eff"],
                "se": g["se"],
                "P_static": g["P_static"],
                "p_LLM_mean": f"{p_llm_mean[s]:.8f}",
                "lambda_star": f"{lam_star:.1f}",
                "pi_BDT": f"{pi_bdt[s]:.8f}",
            })
    print(f"Wrote {ANCHOR_OUT} (lambda*={lam_star})")

    state_meta = {
        s: {"wait": grid_lookup[s]["wait"], "eff": grid_lookup[s]["eff"], "se": grid_lookup[s]["se"]}
        for s in states
    }

    def row_metrics(method: str, preds: dict[str, float], lam: str = "") -> dict:
        ss = [s for s in states if s in preds]
        targets = [p_static[s] for s in ss]
        predv = [preds[s] for s in ss]
        mse_v = mean((p - t) ** 2 for p, t in zip(predv, targets))
        mae_v = mean(abs(p - t) for p, t in zip(predv, targets))
        chr_w = chr_violation_rate(preds, state_meta, axis="wait")
        chr_se = chr_violation_rate(preds, state_meta, axis="se")
        rho = _spearman(predv, targets)
        pr = _pearson(predv, targets)
        return {
            "model": MODEL_LABEL,
            "method": method,
            "MSE": f"{mse_v:.8f}",
            "MAE": f"{mae_v:.8f}",
            "CHR_Wait": f"{chr_w:.4f}" if chr_w is not None else "N/A",
            "CHR_SE": f"{chr_se:.4f}" if chr_se is not None else "N/A",
            "Spearman_rho": f"{rho:.4f}" if rho is not None else "N/A",
            "Pearson_r": f"{pr:.4f}" if pr is not None else "N/A",
            "selected_lambda": lam,
            "parse_success_rate": f"{parse_rate:.4f}",
        }

    metrics_rows = [
        row_metrics("Unconstrained LLM", p_llm_mean, ""),
        row_metrics("Static-BDT Anchor", pi_bdt, f"{lam_star:.1f}"),
    ]

    fields = [
        "model", "method", "MSE", "MAE", "CHR_Wait", "CHR_SE",
        "Spearman_rho", "Pearson_r", "selected_lambda", "parse_success_rate",
    ]
    with open(METRICS_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metrics_rows)
    print(f"Wrote {METRICS_OUT}")

    print_table1(metrics_rows)


def print_table1(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"TABLE 1 — {MODEL_LABEL} (static mixed-logit frontier, vs P_static)")
    print("=" * 72)
    print(f"| {'Method':<22} | {'MSE':>8} | {'MAE':>8} | {'CHR-W':>7} | {'CHR-SE':>7} | {'ρ':>6} | {'r':>6} | λ* |")
    print(f"|{'-'*24}|{'-'*10}|{'-'*10}|{'-'*9}|{'-'*9}|{'-'*8}|{'-'*8}|{'-'*4}|")
    for r in rows:
        lam = r["selected_lambda"] if r["selected_lambda"] else "—"
        print(
            f"| {r['method']:<22} | {float(r['MSE']):>8.4f} | {float(r['MAE']):>8.4f} | "
            f"{r['CHR_Wait']:>7} | {r['CHR_SE']:>7} | {r['Spearman_rho']:>6} | {r['Pearson_r']:>6} | {lam:>4} |"
        )
    print(f"| parse_success_rate     | {rows[0]['parse_success_rate']}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 1 DeepSeek V4 Pro (static frontier)")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["build-grid", "run-llm", "metrics", "all", "status"],
        help="Pipeline step (default: all)",
    )
    parser.add_argument("--preflight", action="store_true", help="Test API and exit")
    parser.add_argument("--no-preflight", action="store_true", help="Skip API preflight")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between API calls")
    parser.add_argument("--finalize-only", action="store_true", help="Rebuild finals + metrics from partials")
    args = parser.parse_args()

    if args.preflight:
        proxy = detect_cursor_proxy()
        if proxy:
            sys.exit(f"❌ Proxy: {proxy}")
        preflight_api()
        return

    cmd = args.command
    if cmd == "build-grid":
        build_static_grid()
        return

    if cmd == "status":
        grid = load_grid()
        print_status(grid)
        return

    if not os.path.exists(STATIC_GRID):
        build_static_grid()

    grid = load_grid()

    if cmd in ("run-llm", "all") and not args.finalize_only:
        run_llm(grid, sleep_s=args.sleep, skip_preflight=args.no_preflight)

    if cmd in ("metrics", "all") or args.finalize_only:
        parsed = write_final_csvs()
        compute_metrics(parsed, grid)


if __name__ == "__main__":
    main()
