"""Shared LLM response normalization and parsing for Table 1 static runs."""
from __future__ import annotations

import re

_THINK_CLOSE = re.compile(r"</(?:redacted_)?think(?:ing)?>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<(?:redacted_)?think(?:ing)?>", re.IGNORECASE)
_PREFILL_PREFIX = "Decision:"


def normalize_model_text(text: str) -> str:
    if not text:
        return ""
    parts = _THINK_CLOSE.split(text)
    if len(parts) > 1:
        tail = parts[-1].strip()
        if tail:
            return tail
    t = _THINK_OPEN.sub("", text)
    t = _THINK_CLOSE.sub("", t)
    return t.strip()


def _apply_prefill(text: str, prefill: str) -> str:
    text = normalize_model_text(text)
    if not text:
        return prefill
    if prefill and not re.search(r"Decision\s*:", text, re.I):
        text = text.lstrip()
        if text.lower().startswith(("yes", "no")):
            text = prefill + " " + text
        else:
            text = prefill + " " + text
    return text


def extract_message_text(data: dict, *, prefill: str = "") -> str:
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    ordered = [content, reasoning, f"{reasoning}\n{content}".strip()]
    best = ""
    for raw in ordered:
        if not raw:
            continue
        norm = _apply_prefill(raw, prefill)
        if re.search(r"Decision\s*:", norm, re.I) and re.search(r"Probability\s*:", norm, re.I):
            return norm
        if not best:
            best = norm
    if best:
        return best
    raise RuntimeError(f"empty model content; message keys={list(msg.keys())}")


def bdt_messages(user_prompt: str) -> list[dict]:
    """Standard chat (DeepSeek / Qwen)."""
    system = (
        "You simulate one person's vaccination choice. "
        "Reply with EXACTLY three lines and nothing else:\n"
        "Decision: Yes or No\n"
        "Probability: <integer 0-100>\n"
        "Reasoning: <one short sentence>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]


def bdt_messages_miro(user_prompt: str) -> list[dict]:
    """
    MiroThinker: few-shot + assistant prefill so completion starts at Decision:
    Same scenario text as grid; only the chat wrapper differs.
    """
    return [
        {
            "role": "system",
            "content": (
                "You simulate a survey respondent. Output ONLY the three-line format. "
                "Never explain instructions. Never repeat the question."
            ),
        },
        {
            "role": "user",
            "content": (
                "Wait 0 months, vaccine effectiveness 0.5, side effect risk 0.0. "
                "Would you get vaccinated now?"
            ),
        },
        {
            "role": "assistant",
            "content": "Decision: Yes\nProbability: 60\nReasoning: Moderate benefit with no wait.",
        },
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": _PREFILL_PREFIX},
    ]


def compact_scenario_prompt(wait: str, eff: str, se: str) -> str:
    """Retry prompt: same scenario (wait/eff/se), fewer words."""
    return (
        f"Wait {wait} months, vaccine effectiveness {eff}, side effect risk {se}. "
        "Would you get vaccinated now?"
    )


def parse_response(text: str, *, prefill: str = "") -> dict:
    if text.startswith("ERROR:"):
        return {
            "decision": "",
            "probability_0_100": "",
            "probability_0_1": "",
            "reasoning": text[:500],
            "parse_success": "False",
        }

    text = _apply_prefill(text, prefill or _PREFILL_PREFIX)
    decision = ""
    p100 = ""
    p01 = ""
    reasoning = ""

    dm = re.search(r"Decision\s*:\s*(Yes|No)\b", text, re.IGNORECASE)
    if dm:
        decision = dm.group(1).capitalize()
    else:
        tail = text[-400:]
        ym = re.search(r"\b(Yes|No)\b", tail, re.IGNORECASE)
        if ym:
            decision = ym.group(1).capitalize()

    pm = re.search(r"Probability\s*:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if pm:
        val = float(pm.group(1))
        if 0 <= val <= 100:
            p100 = str(val)
            p01 = str(val / 100.0)
    if not p01:
        tail = text[-300:]
        for m in re.finditer(r"\b(\d{1,3})\b", tail):
            val = int(m.group(1))
            if 0 <= val <= 100:
                p100 = str(val)
                p01 = str(val / 100.0)
                break

    rm = re.search(r"Reasoning\s*:\s*(.+?)(?:\n\s*\n|$)", text, re.DOTALL | re.IGNORECASE)
    if rm:
        reasoning = rm.group(1).strip()[:500]
    elif decision:
        reasoning = "Stated choice."

    meta = (
        "<think>" in text.lower()
        or "the user asks" in text.lower()
        or "we need to respond" in text.lower()
    )
    ok = decision in ("Yes", "No") and p01 != "" and not meta
    return {
        "decision": decision,
        "probability_0_100": p100,
        "probability_0_1": p01,
        "reasoning": reasoning,
        "parse_success": "True" if ok else "False",
    }
