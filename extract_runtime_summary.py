#!/usr/bin/env python3
"""Extract Table-1 LLM runtime summary from checkpoint CSVs and run logs."""
from __future__ import annotations

import csv
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
OUT_CSV = WORKSPACE / "results" / "runtime_summary.csv"
OUT_TEX = WORKSPACE / "results" / "runtime_summary.tex"

MODELS = [
    {
        "model": "Qwen2.5-72B",
        "raw_csv": WORKSPACE / "llm_raw_outputs_qwen72b_unconstrained.csv",
        "log": WORKSPACE / "table1" / "qwen72b_local_run.log",
        "metrics": WORKSPACE / "metrics_table1_qwen72b_static_lambda_sensitivity.csv",
        "expected_n": 640,
    },
    {
        "model": "DeepSeek V4 Pro",
        "raw_csv": WORKSPACE / "llm_raw_outputs_deepseek_unconstrained.csv",
        "log": None,
        "metrics": WORKSPACE / "metrics_table1_deepseek_static_lambda_sensitivity.csv",
        "expected_n": 640,
    },
    {
        "model": "MiroThinker-1.7-mini",
        "raw_csv": WORKSPACE / "llm_raw_outputs_mirothinker_unconstrained.csv",
        "log": WORKSPACE / "table1" / "mirothinker_local_run.log",
        "metrics": WORKSPACE / "metrics_table1_mirothinker_static_lambda_sensitivity.csv",
        "expected_n": 640,
    },
]

BATCH_DONE_RE = re.compile(r"Batch done in (\d+)s", re.I)


def parse_ts(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def load_timestamps(path: Path) -> list[datetime]:
    if not path.is_file():
        return []
    out: list[datetime] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = parse_ts(row.get("timestamp", ""))
            if ts is not None:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                out.append(ts)
    return out


def inter_arrival_seconds(ts: list[datetime]) -> list[float]:
    if len(ts) < 2:
        return []
    ordered = sorted(ts)
    return [(ordered[i] - ordered[i - 1]).total_seconds() for i in range(1, len(ordered))]


def log_batch_seconds(log_path: Path | None) -> tuple[float | None, str]:
    if not log_path or not log_path.is_file():
        return None, ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, ""
    matches = [int(m.group(1)) for m in BATCH_DONE_RE.finditer(text)]
    if not matches:
        return None, ""
    # Prefer last completed batch line
    return float(matches[-1]), f"log:{log_path.name}"


def post_processing_seconds(last_ts: datetime | None, metrics_path: Path) -> tuple[float | None, str]:
    if last_ts is None or not metrics_path.is_file():
        return None, ""
    mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=timezone.utc)
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    delta = (mtime - last_ts).total_seconds()
    if delta < 0:
        return None, "mtime_before_last_gen"
    return delta, f"mtime_gap_to:{metrics_path.name}"


def summarize_model(spec: dict) -> dict:
    raw = spec["raw_csv"]
    ts = load_timestamps(raw)
    timing_quality = "exact" if ts else "approximate"
    notes: list[str] = []

    first_ts = last_ts = None
    wall_span: float | None = None
    avg_gen: float | None = None

    if ts:
        ordered = sorted(ts)
        first_ts, last_ts = ordered[0], ordered[-1]
        wall_span = (last_ts - first_ts).total_seconds()
        deltas = inter_arrival_seconds(ts)
        avg_gen = statistics.mean(deltas) if deltas else None
        data_source = f"csv_checkpoint:{raw.name}"
    else:
        data_source = f"file_mtime:{raw.name}"
        if raw.is_file():
            st = raw.stat()
            first_ts = last_ts = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            wall_span = 0.0
            notes.append("no_timestamp_column")
        else:
            notes.append("missing_raw_csv")

    n_gen = len(ts) if ts else 0
    if spec["expected_n"] and n_gen != spec["expected_n"]:
        notes.append(f"n={n_gen}_expected_{spec['expected_n']}")

    log_wall, log_src = log_batch_seconds(spec.get("log"))
    if log_wall is not None:
        # Use log wall-clock for batch timer when available (matches in-process t0)
        wall_clock = log_wall
        if wall_span is not None and abs(log_wall - wall_span) > 120:
            notes.append(f"log_vs_span_delta={log_wall - wall_span:.0f}s")
    elif wall_span is not None:
        wall_clock = wall_span
    else:
        wall_clock = None

    if avg_gen is None and wall_clock and n_gen > 0:
        avg_gen = wall_clock / n_gen
        timing_quality = "approximate"
        notes.append("avg_from_wall_over_n")

    post_sec, post_src = post_processing_seconds(last_ts, spec["metrics"])
    if post_sec is not None and post_sec > 86400:
        notes.append("post_proc>24h_maybe_manual_step")
        timing_quality = "approximate"

    if not ts:
        timing_quality = "approximate"

    return {
        "model": spec["model"],
        "data_source": data_source,
        "timing_quality": timing_quality,
        "first_timestamp_utc": first_ts.isoformat() if first_ts else "",
        "last_timestamp_utc": last_ts.isoformat() if last_ts else "",
        "n_generations": str(n_gen),
        "wall_clock_seconds": f"{wall_clock:.1f}" if wall_clock is not None else "",
        "avg_seconds_per_generation": f"{avg_gen:.3f}" if avg_gen is not None else "",
        "wall_clock_log_seconds": f"{log_wall:.1f}" if log_wall is not None else "",
        "wall_clock_log_source": log_src,
        "post_processing_seconds": f"{post_sec:.1f}" if post_sec is not None else "",
        "post_processing_source": post_src,
        "notes": "; ".join(notes),
    }


def fmt_duration(seconds: float | str) -> str:
    if seconds == "" or seconds is None:
        return "---"
    s = float(seconds)
    if s >= 3600:
        return f"{s/3600:.2f} h"
    if s >= 60:
        return f"{s/60:.1f} min"
    return f"{s:.1f} s"


def write_csv(rows: list[dict]) -> None:
    fields = [
        "model",
        "data_source",
        "timing_quality",
        "first_timestamp_utc",
        "last_timestamp_utc",
        "n_generations",
        "wall_clock_seconds",
        "avg_seconds_per_generation",
        "wall_clock_log_seconds",
        "wall_clock_log_source",
        "post_processing_seconds",
        "post_processing_source",
        "notes",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_tex(rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by extract_runtime_summary.py",
        "% Checkpoint CSVs (resume partials); no JSONL checkpoints found in workspace.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{LLM inference runtime summary (64 states $\\times$ 10 repeats = 640 generations).}",
        "\\label{tab:runtime-summary}",
        "\\small",
        "\\begin{tabular}{l r r r r r r}",
        "\\toprule",
        "Model & $N$ & First (UTC) & Last (UTC) & Wall clock & Avg / gen & Post-proc$^\\ddagger$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        first = (r["first_timestamp_utc"] or "")[:19].replace("T", " ")
        last = (r["last_timestamp_utc"] or "")[:19].replace("T", " ")
        qual = r["timing_quality"]
        mark = "$^\\dagger$" if qual == "approximate" else ""
        post = fmt_duration(r["post_processing_seconds"]) if r["post_processing_seconds"] else "---"
        lines.append(
            f"{r['model']}{mark} & {r['n_generations']} & {first} & {last} & "
            f"{fmt_duration(r['wall_clock_seconds'])} & {fmt_duration(r['avg_seconds_per_generation'])} & {post} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{0.35em}",
            "\\begin{minipage}{0.92\\textwidth}",
            "\\footnotesize",
            "Wall clock from run log when available, else first-to-last checkpoint timestamp. "
            "Avg / gen is mean inter-arrival time between consecutive checkpoint rows. "
            "$^\\dagger$Approximate timing. "
            "$^\\ddagger$Post-processing: gap from last checkpoint timestamp to metrics CSV mtime (anchor/metrics step; approximate).",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console(rows: list[dict]) -> None:
    print("=" * 78)
    print("Runtime summary (checkpoint CSV + logs)")
    print("=" * 78)
    for r in rows:
        print(f"\n{r['model']}  [{r['timing_quality']}]  {r['data_source']}")
        print(f"  generations: {r['n_generations']}")
        print(f"  first UTC:   {r['first_timestamp_utc']}")
        print(f"  last UTC:    {r['last_timestamp_utc']}")
        print(f"  wall clock:  {r['wall_clock_seconds']} s  (log: {r['wall_clock_log_seconds'] or 'n/a'})")
        print(f"  avg / gen:   {r['avg_seconds_per_generation']} s")
        if r["post_processing_seconds"]:
            print(
                f"  post-proc:   {r['post_processing_seconds']} s  ({r['post_processing_source']})"
            )
        if r["notes"]:
            print(f"  notes:       {r['notes']}")

    print("\n" + OUT_TEX.read_text(encoding="utf-8"))


def main() -> None:
    rows = [summarize_model(spec) for spec in MODELS]
    write_csv(rows)
    write_tex(rows)
    print_console(rows)
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
