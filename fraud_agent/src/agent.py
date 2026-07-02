"""
Agentic investigation loop using a local Ollama model with tool-calling.

The LLM client is isolated behind a single interface — change LLM_BASE_URL,
LLM_API_KEY, and LLM_MODEL to swap providers (Gemini via OpenRouter, Groq, etc.)
without touching any other code.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Generator

import httpx
from openai import OpenAI

from .tools import CLAIM_TOOL_MAP, CLAIM_TOOLS, TOOL_MAP, TOOLS

logger = logging.getLogger(__name__)

# ── LLM config — change these to swap provider ──────────────────────────────
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_API_KEY  = "ollama"        # Ollama ignores this; openai SDK requires a value
LLM_MODEL    = "qwen3:8b"      # must be a tool-calling-capable model
MAX_STEPS    = 6               # hard cap to prevent runaway loops
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a healthcare fraud investigator at an insurance company.
Your task: investigate a specific provider and produce a structured investigation memo.

You have three tools available:
- get_claims_summary: billing portfolio overview (start here)
- ml_risk_score: machine-learning fraud scores (ridge regression + isolation forest + z-score ensemble)
- check_billing_rules: symbolic billing rule checks (5 hard policy rules)

Use the tools in whatever order makes sense based on what you find.
After gathering evidence, write a structured investigation memo in this format:

## INVESTIGATION MEMO

**Provider:** [ID]
**Risk Level:** LOW | MODERATE | HIGH
**Summary:** [2–3 sentences covering the overall picture]

**ML Risk Signals:**
[Interpret the ridge probability, isolation forest score, and claim risk fraction.
Explain what level of concern each represents and how they compare to expected ranges.]

**Billing Rule Violations:**
[List each triggered rule, the violation count, and why it matters clinically/legally.
If no rules triggered, state that explicitly.]

**Key Risk Factors:**
- [bullet point 1]
- [bullet point 2]
- ...

**Recommendation:** Refer for full audit | Monitor | No action required
"""


def check_ollama() -> tuple[bool, str]:
    """Return (is_running, human-readable message)."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        base = LLM_MODEL.split(":")[0]
        if not any(base in m for m in models):
            return (
                False,
                f"Ollama running but model '{LLM_MODEL}' not found. "
                f"Run:  ollama pull {LLM_MODEL}",
            )
        return True, "OK"
    except Exception:
        return False, "Ollama is not running. Start it with:  ollama serve"


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks emitted by qwen3 reasoning mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_tool_call(tc) -> tuple[str, dict] | None:
    """Extract (name, args) from an OpenAI ToolCall object; return None on failure."""
    try:
        return tc.function.name, json.loads(tc.function.arguments)
    except Exception as exc:
        logger.warning("Malformed tool call: %s", exc)
        return None


def _execute_tool(name: str, args: dict) -> str:
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        result = TOOL_MAP[name](**args)
        return json.dumps(result, default=str)
    except Exception as exc:
        logger.error("Tool %s raised: %s", name, exc)
        return json.dumps({"error": str(exc)})


def _stream_clean(stream_iter) -> Generator[str, None, None]:
    """
    Yield text chunks from a streaming response, suppressing <think>…</think>
    blocks that qwen3 reasoning mode emits before the actual answer.
    """
    buffer = ""
    in_think = False

    for chunk in stream_iter:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta

        while buffer:
            if in_think:
                end = buffer.find("</think>")
                if end >= 0:
                    in_think = False
                    buffer = buffer[end + 8:]
                else:
                    buffer = ""   # still inside <think>, wait for more
                    break
            else:
                start = buffer.find("<think>")
                if start >= 0:
                    if start > 0:
                        yield buffer[:start]
                    in_think = True
                    buffer = buffer[start + 7:]
                else:
                    yield buffer
                    buffer = ""
                    break

    if buffer and not in_think:
        yield buffer


# ── Public API ────────────────────────────────────────────────────────────────

def run_investigation(provider_id: str) -> Generator[dict, None, None]:
    """
    Generator that drives the agentic loop and yields event dicts:

      {"type": "step",      "step": int,  "action": str}
      {"type": "tool_call", "tool": str,  "args": dict, "result": str}
      {"type": "memo_chunk","text": str}   ← streaming final memo tokens
      {"type": "done"}
      {"type": "error",     "message": str}
    """
    ok, msg = check_ollama()
    if not ok:
        yield {"type": "error", "message": msg}
        return

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Investigate provider {provider_id} for potential fraudulent billing "
                "and produce an investigation memo for a human reviewer."
            ),
        },
    ]

    for step in range(1, MAX_STEPS + 1):
        yield {"type": "step", "step": step, "action": "Calling LLM…"}

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception as exc:
            yield {"type": "error", "message": f"LLM call failed: {exc}"}
            return

        choice = response.choices[0]
        msg_out = choice.message
        content = _strip_thinking(msg_out.content or "")

        # Model finished reasoning — stream the memo
        if not msg_out.tool_calls:
            yield {"type": "step", "step": step, "action": "Writing memo…"}
            yield from _emit_memo(client, messages, content)
            yield {"type": "done"}
            return

        # Append assistant turn (keep original, including tool_calls)
        messages.append(msg_out)

        for tc in msg_out.tool_calls:
            parsed = _parse_tool_call(tc)
            if parsed is None:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": "Could not parse tool call"}),
                })
                continue

            name, args = parsed
            yield {"type": "step", "step": step, "action": f"Calling tool: {name}"}

            result_str = _execute_tool(name, args)
            yield {
                "type": "tool_call",
                "tool": name,
                "args": args,
                "result": result_str[:600],   # truncate for UI display
            }

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # MAX_STEPS exhausted — force a memo from accumulated context
    yield {"type": "step", "step": MAX_STEPS, "action": "Max steps — synthesising memo…"}
    messages.append({
        "role": "user",
        "content": "You have reached the step limit. Write the investigation memo now.",
    })
    yield from _emit_memo(client, messages, "")
    yield {"type": "done"}


CLAIM_SYSTEM_PROMPT = """You are a healthcare fraud investigator assessing a SINGLE MEDICAL CLAIM
from a provider with no prior billing history in our system.

CRITICAL CONSTRAINT: This is a cold-start scenario.
- Layer 1 (ridge regression, isolation forest) requires aggregated provider history — NOT AVAILABLE.
- Do NOT attempt to infer or estimate a provider-level score. State explicitly it is N/A.
- Layer 2 (claim-level z-score) and Layer 3 (symbolic rules) ARE available and are your tools.

Key framing: the symbolic rules are the PRIMARY fraud-detection capability here.
They are deterministic, claim-level, and require zero provider history — this is
exactly the cold-start strength of the neural-symbolic architecture over pure ML.

You have two tools:
- score_single_claim_layer2: z-score anomaly check vs training reference distributions
- check_single_claim_rules: 5 symbolic billing-rule checks

Use both. Then write:

## CLAIM ASSESSMENT MEMO

**Assessment Scope:** Single claim — no provider history
**Risk Level:** LOW | MODERATE | HIGH
**Summary:** [2–3 sentences]

**Layer 1 — Provider ML (Ridge + Isolation Forest):** N/A — requires provider billing history. Not available for new/unknown providers.

**Layer 2 — Claim Anomaly Score (Z-Score Ensemble):**
[Interpret the risk fraction: how many peer groups flagged it, what that implies about the reimbursement amount relative to comparable claims in training data.]

**Layer 3 — Billing Rule Violations:**
[List each triggered rule, its violation count, and clinical/legal significance.
If no rules triggered, say so clearly.]

**Key Findings:**
- [bullets]

**Recommendation:** Flag for follow-up | Monitor if more claims arrive | No concern
**Note:** This is a preliminary single-claim assessment. Provider fraud determination requires multi-claim provider-level analysis.
"""


def run_claim_investigation(claim_dict: dict) -> Generator[dict, None, None]:
    """
    Agentic loop scoped to a single claim from an unknown provider.
    Only claim-level tools are available (Layer 2 + symbolic rules).
    Layer 1 provider scoring is explicitly excluded.

    The claim_dict is passed to tools via closure — the LLM does not need
    to supply it as a tool argument.
    """
    ok, msg = check_ollama()
    if not ok:
        yield {"type": "error", "message": msg}
        return

    # Bind claim_dict into the tool implementations via closure
    from .tools import score_single_claim_layer2, check_single_claim_rules

    def _bound_score(_confirm: str = "") -> dict:
        return score_single_claim_layer2(claim_dict)

    def _bound_rules(_confirm: str = "") -> dict:
        return check_single_claim_rules(claim_dict)

    bound_tool_map = {
        "score_single_claim_layer2": _bound_score,
        "check_single_claim_rules":  _bound_rules,
    }

    # Build a concise claim summary for the opening user message
    reimb  = claim_dict.get("InscClaimAmtReimbursed", "unknown")
    ctype  = "Inpatient" if claim_dict.get("claim_type") == 1 else "Outpatient"
    diag1  = claim_dict.get("ClmDiagnosisCode_1", "none")
    proc1  = claim_dict.get("ClmProcedureCode_1", "none")
    admit  = claim_dict.get("AdmissionDt", "")
    disch  = claim_dict.get("DischargeDt", "")
    dates  = f", admission {admit} → discharge {disch}" if admit and disch else ""
    claim_summary = (
        f"Claim details: {ctype}, reimbursement ${reimb:,.0f}"
        if isinstance(reimb, (int, float)) else
        f"Claim details: {ctype}, reimbursement {reimb}"
    )
    claim_summary += f"{dates}. Primary diagnosis: {diag1}. Procedure: {proc1}."

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    messages: list[dict] = [
        {"role": "system", "content": CLAIM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Assess this claim for fraud indicators. Provider has no prior history.\n\n"
                f"{claim_summary}\n\n"
                "Use your tools to run the z-score check and billing rule checks, "
                "then write the assessment memo."
            ),
        },
    ]

    for step in range(1, MAX_STEPS + 1):
        yield {"type": "step", "step": step, "action": "Calling LLM…"}

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=CLAIM_TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception as exc:
            yield {"type": "error", "message": f"LLM call failed: {exc}"}
            return

        choice = response.choices[0]
        msg_out = choice.message
        content = _strip_thinking(msg_out.content or "")

        if not msg_out.tool_calls:
            yield {"type": "step", "step": step, "action": "Writing memo…"}
            yield from _emit_memo(client, messages, content, "## CLAIM ASSESSMENT MEMO")
            yield {"type": "done"}
            return

        messages.append(msg_out)

        for tc in msg_out.tool_calls:
            parsed = _parse_tool_call(tc)
            if parsed is None:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": "Could not parse tool call"}),
                })
                continue

            name, _ = parsed
            yield {"type": "step", "step": step, "action": f"Calling tool: {name}"}

            fn = bound_tool_map.get(name)
            if fn is None:
                result_str = json.dumps({"error": f"Tool '{name}' not available in claim mode"})
            else:
                try:
                    result_str = json.dumps(fn(), default=str)
                except Exception as exc:
                    result_str = json.dumps({"error": str(exc)})

            yield {
                "type": "tool_call",
                "tool": name,
                "args": {"claim": "(provided in session)"},
                "result": result_str[:600],
            }
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    yield {"type": "step", "step": MAX_STEPS, "action": "Max steps — synthesising memo…"}
    messages.append({
        "role": "user",
        "content": "Write the claim assessment memo now.",
    })
    yield from _emit_memo(client, messages, "", "## CLAIM ASSESSMENT MEMO")
    yield {"type": "done"}


def _emit_memo(client: OpenAI, messages: list, existing: str,
               header: str = "## INVESTIGATION MEMO") -> Generator[dict, None, None]:
    """Stream the final memo, either from existing content or by asking the LLM again."""
    if existing and header in existing:
        yield {"type": "memo_chunk", "text": existing}
        return

    prompt_messages = messages + [
        {
            "role": "user",
            "content": "Now write the complete investigation memo based on everything you found.",
        }
    ]

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=prompt_messages,
            temperature=0.2,
            stream=True,
        )
        for chunk in _stream_clean(stream):
            if chunk:
                yield {"type": "memo_chunk", "text": chunk}
    except Exception as exc:
        yield {"type": "memo_chunk", "text": f"\n\n[Streaming error: {exc}]"}
