"""
Created by Hermes Agent for out-of-design stress test (part 1/2).

Part 1: Generate prompts, compute P_static, and run LLM queries via OpenRouter.
Part 2: Compute metrics (run after LLM outputs are available).

Usage:
  python3 scripts/out_of_design_stress_test.py gen-prompts   # Step 1
  python3 scripts/out_of_design_stress_test.py compute-pstatic  # Step 2
  python3 scripts/out_of_design_stress_test.py run-llm       # Step 3 (needs API keys)
  python3 scripts/out_of_design_stress_test.py metrics       # Step 4 (needs LLM outputs)
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean

# ── paths ──────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
SCENARIOS_CSV = WORKSPACE / "out_of_design_scenarios.csv"
PROMPTS_JSONL = WORKSPACE / "prompts" / "out_of_design_stress_test_prompts.jsonl"
RAW_OUTPUTS = WORKSPACE / "results" / "out_of_design_raw_outputs.jsonl"
PARSED_PROBS = WORKSPACE / "results" / "out_of_design_parsed_probabilities.csv"
BDT_METRICS = WORKSPACE / "results" / "out_of_design_bdt_metrics.csv"
STRESS_TABLE_TEX = WORKSPACE / "results" / "out_of_design_stress_test_table.tex"
GRID_STATIC = WORKSPACE / "bdt_eval_grid_static.csv"
MXL_COEFS = WORKSPACE / "analysis_output" / "mxl_coefs.csv"
RUN_INSTRUCTIONS = WORKSPACE / "results" / "out_of_design_run_instructions.md"

REPEATS = 10
SEED = 2026
LAMBDA = 0.25

# Side effect mapping: text → numeric (DCE scale 0-3)
SE_MAP = {"none": 0.0, "low": 0.0, "mild": 1.0, "moderate": 2.0, "severe": 3.0}

# Models to run (OpenRouter slugs)
MODELS = {
    "Qwen2.5-72B": "qwen/qwen-2.5-72b-instruct",
    "DeepSeek V3": "deepseek/deepseek-chat-v3-0324",
    "DeepSeek V4 Pro": None,  # reasoning model returns content=null
    "MiroThinker-1.7-mini": None,  # not available on OpenRouter
}

# ── helpers ─────────────────────────────────────────────────────────
def sigmoid(x: float) -> float:
    """Logistic function."""
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def load_coefs(path: str | Path) -> dict[str, float]:
    """Load mixed-logit coefficients from mxl_coefs.csv."""
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row[""].strip()
            if name:
                yield name, float(row["Estimate"])


def compute_pstatic(wait: float, eff: float, se: float,
                    coefs: dict[str, float]) -> float:
    """Compute P_static = sigmoid(beta_w * wait + beta_e * eff + beta_se * se)."""
    u = coefs["WaitTime"] * wait + coefs["VaccineEfficacy"] * eff + coefs["SideEffects"] * se
    return sigmoid(u)


def make_prompt(scenario: dict) -> str:
    """Build LLM prompt string for an out-of-design scenario."""
    extra = scenario["extra_policy_description"]
    parts = [
        f"You are evaluating a vaccination decision."
    ]
    # DCE-measured dimensions
    parts.append(f"The current wait time is {scenario['wait_time']} months, "
                 f"vaccine effectiveness is {scenario['efficacy']}, "
                 f"and side effect risk is {scenario['side_effects']}.")
    # Richer policy description
    if extra:
        parts.append(f"Additional context: {extra}.")
    parts.append("Would you get vaccinated now?")
    parts.append("Respond exactly in this format:")
    parts.append("Decision: Yes or No")
    parts.append("Probability: a number between 0 and 100")
    parts.append("Reasoning: one short sentence")
    return "\n".join(parts)


def parse_response(text: str) -> dict:
    """Parse LLM response into decision, probability, reasoning."""
    result = {
        "decision": "",
        "probability_0_1": "",
        "reasoning": "",
        "parse_success": "False",
    }
    dm = re.search(r"Decision\s*:\s*(Yes|No)", text, re.IGNORECASE)
    if dm:
        result["decision"] = dm.group(1).capitalize()

    pm = re.search(r"Probability\s*:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if pm:
        val = float(pm.group(1))
        if 0 <= val <= 100:
            result["probability_0_1"] = str(val / 100.0)

    rm = re.search(r"Reasoning\s*:\s*(.+?)(?:\n\s*\n|$)", text, re.DOTALL | re.IGNORECASE)
    if rm:
        result["reasoning"] = rm.group(1).strip()[:500]

    ok = result["decision"] in ("Yes", "No") and result["probability_0_1"] != ""
    result["parse_success"] = "True" if ok else "False"
    return result


# ── step 1: generate prompts ──────────────────────────────────────
def gen_prompts() -> None:
    """Read scenarios, generate prompts, write JSONL."""
    PROMPTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUTS.parent.mkdir(parents=True, exist_ok=True)

    with open(SCENARIOS_CSV, newline="") as f:
        scenarios = list(csv.DictReader(f))

    count = 0
    with open(PROMPTS_JSONL, "w") as out:
        for sc in scenarios:
            prompt = make_prompt(sc)
            entry = {
                "scenario_id": sc["scenario_id"],
                "wait_time": float(sc["wait_time"]),
                "efficacy": float(sc["efficacy"]),
                "side_effects": sc["side_effects"],
                "extra_policy_description": sc["extra_policy_description"],
                "expected_wait_order_group": sc["expected_wait_order_group"],
                "prompt": prompt,
            }
            out.write(json.dumps(entry) + "\n")
            count += 1
    print(f"Wrote {count} prompts to {PROMPTS_JSONL}")


# ── step 2: compute P_static ──────────────────────────────────────
def compute_pstatic_all() -> list[dict]:
    """Compute P_static for each scenario using existing MXL coefficients."""
    coefs = dict(load_coefs(MXL_COEFS))

    with open(SCENARIOS_CSV, newline="") as f:
        scenarios = list(csv.DictReader(f))

    rows = []
    for sc in scenarios:
        wait = float(sc["wait_time"])
        eff = float(sc["efficacy"])
        se_text = sc["side_effects"].strip().lower()
        se_num = SE_MAP.get(se_text, 0.0)
        p_static = compute_pstatic(wait, eff, se_num, coefs)
        rows.append({
            "scenario_id": sc["scenario_id"],
            "wait_time": wait,
            "efficacy": eff,
            "side_effects_numeric": se_num,
            "side_effects_text": sc["side_effects"],
            "extra_policy_description": sc["extra_policy_description"],
            "expected_wait_order_group": sc["expected_wait_order_group"],
            "P_static": p_static,
        })
    return rows


# ── step 3: run LLM via OpenRouter ────────────────────────────────
def openrouter_chat(api_key: str, model: str, messages: list[dict],
                    timeout: int = 60) -> dict:
    """Call OpenRouter chat completions API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yichao2022/behavioral-digital-twins",
        "X-Title": "BDT Out-of-Design Stress Test",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                  headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        return {"error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"error": str(e)}


def _system_prompt(model_name: str) -> str:
    """System prompt matching the existing paper format."""
    if "miro" in model_name.lower() or "mirothink" in model_name.lower():
        return (
            "You simulate a survey respondent. Output ONLY the three-line format. "
            "Never explain instructions. Never repeat the question."
        )
    return (
        "You simulate one person's vaccination choice. "
        "Reply with EXACTLY three lines and nothing else:\n"
        "Decision: Yes or No\n"
        "Probability: <integer 0-100>\n"
        "Reasoning: <one short sentence>"
    )


def _messages_for(model_name: str, prompt: str) -> list[dict]:
    """Build message list matching existing per-model chat format."""
    sys = _system_prompt(model_name)
    if "miro" in model_name.lower() or "mirothink" in model_name.lower():
        # Few-shot + assistant prefill for MiroThinker
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": "Wait 0 months, vaccine effectiveness 0.5, "
                                         "side effect risk 0.0. Would you get vaccinated now?"},
            {"role": "assistant", "content": "Decision: Yes\nProbability: 60\n"
                                              "Reasoning: Moderate benefit with no wait."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Decision:"},
        ]
    return [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]


def run_llm_model(model_name: str, api_key: str) -> None:
    """Run Qwen2.5-72B on all scenarios via OpenRouter."""
    slug = MODELS[model_name]
    with open(PROMPTS_JSONL) as f:
        prompts = [json.loads(line) for line in f]

    print(f"Running {model_name} ({slug}) on {len(prompts)} scenarios, "
          f"{REPEATS} repeats each = {len(prompts) * REPEATS} calls")

    raw_rows = []
    parsed_rows = []

    random.seed(SEED + hash(model_name) % 10000)

    for sc in prompts:
        sc_id = sc["scenario_id"]
        for rep in range(1, REPEATS + 1):
            messages = _messages_for(model_name, sc["prompt"])
            resp = openrouter_chat(api_key, slug, messages)

            if "error" in resp:
                raw_text = f"ERROR: {resp['error']}"
                parsed = {"decision": "", "probability_0_1": "",
                          "reasoning": raw_text[:500], "parse_success": "False"}
            else:
                try:
                    msg = resp["choices"][0]["message"]
                    raw_text = (msg.get("content") or "").strip()
                    if not raw_text:
                        # Try reasoning field (DeepSeek V4 Pro pattern)
                        raw_text = (msg.get("reasoning") or "").strip()
                except (KeyError, IndexError):
                    raw_text = json.dumps(resp)[:500]

                parsed = parse_response(raw_text)

            raw_rows.append({
                "model": model_name,
                "repeat": rep,
                "scenario_id": sc_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "prompt_text": sc["prompt"],
                "response_raw": raw_text,
            })
            parsed_rows.append({
                "model": model_name,
                "repeat": rep,
                "scenario_id": sc_id,
                "wait_time": sc["wait_time"],
                "efficacy": sc["efficacy"],
                "side_effects_text": sc["side_effects"],
                "extra_policy_description": sc["extra_policy_description"],
                "expected_wait_order_group": sc["expected_wait_order_group"],
                "decision": parsed["decision"],
                "probability_0_1": parsed["probability_0_1"],
                "reasoning": parsed["reasoning"],
                "parse_success": parsed["parse_success"],
            })

            # Rate limiting
            time.sleep(1.0)

        print(f"  Completed {sc_id} ({rep}/{REPEATS})")

    # Append to raw outputs JSONL
    mode = "a" if RAW_OUTPUTS.exists() else "w"
    with open(RAW_OUTPUTS, mode) as f:
        for row in raw_rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Appended {len(raw_rows)} raw rows to {RAW_OUTPUTS}")

    # Append to parsed probabilities CSV
    mode = "a" if PARSED_PROBS.exists() else "w"
    fields = ["model", "repeat", "scenario_id", "wait_time", "efficacy",
              "side_effects_text", "extra_policy_description",
              "expected_wait_order_group", "decision", "probability_0_1",
              "reasoning", "parse_success"]
    with open(PARSED_PROBS, mode) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        writer.writerows(parsed_rows)
    print(f"  Appended {len(parsed_rows)} parsed rows to {PARSED_PROBS}")


# ── step 4: compute metrics ──────────────────────────────────────
def compute_metrics(parsed_path: str | Path, p_static_rows: list[dict]) -> dict:
    """Compute MVR-Wait, MAD, Spearman for each model-method."""
    # Load parsed LLM probabilities
    by_model = defaultdict(lambda: defaultdict(list))  # model -> scenario_id -> list[float]
    with open(parsed_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("parse_success", "").lower() in ("true", "1", "yes"):
                try:
                    prob = float(row["probability_0_1"])
                except (ValueError, TypeError):
                    continue
                model = row["model"]
                sid = row["scenario_id"]
                by_model[model][sid].append(prob)

    # Build scenario lookup
    sc_lookup = {r["scenario_id"]: r for r in p_static_rows}
    scenario_ids = sorted(sc_lookup.keys())
    p_static_vals = [sc_lookup[s]["P_static"] for s in scenario_ids]

    results = []

    for model in sorted(by_model.keys()):
        p_llm_by_sc = by_model[model]

        # Compute mean LLM probability per scenario
        p_llm_mean = {}
        for sid in scenario_ids:
            vals = p_llm_by_sc.get(sid, [])
            p_llm_mean[sid] = mean(vals) if vals else None

        # Initialize entries
        p_llm_list = [p_llm_mean[s] for s in scenario_ids]
        p_bdt_list = [
            (LAMBDA * p_llm_mean[s] + (1.0 - LAMBDA) * sc_lookup[s]["P_static"])
            if p_llm_mean[s] is not None else None
            for s in scenario_ids
        ]

        for method, probs in [("Unconstrained LLM", p_llm_list),
                              (f"Static-BDT (lambda={LAMBDA})", p_bdt_list)]:
            valid = [(p, ps) for p, ps in zip(probs, p_static_vals) if p is not None]
            if len(valid) < 3:
                results.append({
                    "model": model, "method": method,
                    "MVR_Wait": "N/A", "MAD": "N/A",
                    "Spearman_rho": "N/A", "n_scenarios": len(valid),
                })
                continue

            pv, psv = zip(*valid)

            # MVR-Wait: monotonicity across the 6 paired groups
            group_violations = 0
            group_total = 0
            for gname in sorted(set(r["expected_wait_order_group"] for r in p_static_rows)):
                g_scenarios = [r for r in p_static_rows
                               if r["expected_wait_order_group"] == gname]
                if len(g_scenarios) != 2:
                    continue
                s_low = next((r for r in g_scenarios if r["wait_time"] == 2), None)
                s_high = next((r for r in g_scenarios if r["wait_time"] == 6), None)
                if s_low is None or s_high is None:
                    continue
                p_low = p_llm_mean.get(s_low["scenario_id"])
                p_high = p_llm_mean.get(s_high["scenario_id"])
                if p_low is None or p_high is None:
                    continue
                group_total += 1
                if p_low < p_high:
                    group_violations += 1

            mvr_wait = group_violations / group_total if group_total > 0 else None

            # MAD from P_static
            mad = mean(abs(p - ps) for p, ps in zip(pv, psv))

            # Spearman correlation with P_static
            rho = None
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(pv, psv)
                rho = float(rho)
            except ImportError:
                # Manual Spearman
                n = len(pv)
                def rank(vals):
                    order = sorted(range(n), key=lambda i: vals[i])
                    r = [0.0] * n
                    i = 0
                    while i < n:
                        j = i
                        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                            j += 1
                        avg = (i + j) / 2.0 + 1.0
                        for k in range(i, j + 1):
                            r[order[k]] = avg
                        i = j + 1
                    return r
                from statistics import mean as m
                rx, ry = rank(list(pv)), rank(list(psv))
                mx, my = m(rx), m(ry)
                num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
                dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
                dy = math.sqrt(sum((b - my) ** 2 for b in ry))
                if dx > 0 and dy > 0:
                    rho = num / (dx * dy)

            results.append({
                "model": model,
                "method": method,
                "MVR_Wait": f"{mvr_wait:.4f}" if mvr_wait is not None else "N/A",
                "MAD": f"{mad:.6f}",
                "Spearman_rho": f"{rho:.4f}" if rho is not None else "N/A",
                "n_scenarios": len(valid),
            })

    return results


# ── main ──────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "gen-prompts":
        gen_prompts()

    elif cmd == "compute-pstatic":
        rows = compute_pstatic_all()
        # Save P_static alongside parsed probs for reference
        out = WORKSPACE / "results" / "out_of_design_pstatic.csv"
        fields = ["scenario_id", "wait_time", "efficacy", "side_effects_numeric",
                  "side_effects_text", "extra_policy_description",
                  "expected_wait_order_group", "P_static"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote P_static for {len(rows)} scenarios to {out}")
        for r in rows:
            print(f"  {r['scenario_id']}: P_static = {r['P_static']:.4f}")

    elif cmd == "run-llm":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            sys.exit(1)

        models_to_run = []
        if len(sys.argv) > 2:
            models_to_run = [sys.argv[2]]
        else:
            # Default: run all available
            models_to_run = [k for k, v in MODELS.items() if v is not None]

        for mname in models_to_run:
            slug = MODELS.get(mname)
            if slug is None:
                print(f"Skipping {mname}: no OpenRouter slug available")
                continue
            run_llm_model(mname, api_key)

        if "MiroThinker-1.7-mini" not in models_to_run:
            print(f"\nMiroThinker not run. See instructions in {RUN_INSTRUCTIONS}")

    elif cmd == "metrics":
        pstatic_rows = compute_pstatic_all()
        if not PARSED_PROBS.exists():
            print(f"ERROR: {PARSED_PROBS} not found. Run LLM generation first.")
            sys.exit(1)
        results = compute_metrics(PARSED_PROBS, pstatic_rows)
        fields = ["model", "method", "MVR_Wait", "MAD",
                  "Spearman_rho", "n_scenarios"]
        with open(BDT_METRICS, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f"Wrote {len(results)} metric rows to {BDT_METRICS}")
        for r in results:
            print(f"  {r['model']} / {r['method']}: "
                  f"MVR_Wait={r['MVR_Wait']}, MAD={r['MAD']}, "
                  f"Spearman={r['Spearman_rho']} (n={r['n_scenarios']})")

    elif cmd == "all":
        gen_prompts()
        rows = compute_pstatic_all()
        out = WORKSPACE / "results" / "out_of_design_pstatic.csv"
        fields = ["scenario_id", "wait_time", "efficacy", "side_effects_numeric",
                  "side_effects_text", "extra_policy_description",
                  "expected_wait_order_group", "P_static"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"P_static computed. Now run: python3 scripts/out_of_design_stress_test.py run-llm")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
