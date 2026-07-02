"""
llm_judge.py

Uses an external LLM (Claude or OpenAI) to judge whether a generated
BraneScript workflow is functionally equivalent to a reference implementation.

Usage:
    from llm_judge import judge
    result = judge(intent, reference_code, generated_code)
    print(result.verdict, result.score, result.reasoning)

Env vars:
    JUDGE_API      "anthropic" (default) or "openai"
    JUDGE_MODEL    model name (default: claude-3-5-sonnet-20241022 / gpt-4o)
    ANTHROPIC_API_KEY or OPENAI_API_KEY
"""

import json
import os
from dataclasses import dataclass, field

JUDGE_API   = os.environ.get("JUDGE_API",   "anthropic")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")  # empty → use per-api default

_ANTHROPIC_DEFAULT = "claude-3-5-sonnet-20241022"
_OPENAI_DEFAULT    = "gpt-4o"

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    verdict:    str         # "EQUIVALENT" | "PARTIAL" | "NOT_EQUIVALENT"
    equivalent: bool        # True when verdict == "EQUIVALENT"
    score:      float       # 0.0 – 1.0  (EQUIV=1.0, PARTIAL≈0.5, NOT_EQUIV=0.0)
    reasoning:  str         # one-line explanation
    issues:     list[str] = field(default_factory=list)

    @classmethod
    def error(cls, msg: str) -> "JudgeResult":
        return cls(verdict="ERROR", equivalent=False, score=0.0,
                   reasoning=msg, issues=[msg])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert evaluator of BraneScript workflow code.

BraneScript orchestrates computations in the Brane distributed framework.
Key constructs: import <pkg>, let x := expr;, if/for/while, class, new ClassName { ... },
#[on("node")] for routing, commit_result("name", result) for persisting output.

TASK: given a user intent and two BraneScript implementations, judge whether
the GENERATED code is functionally equivalent to the REFERENCE — i.e. would it
accomplish the same task, invoke the same package functions, process the same
data, and produce equivalent results when executed.

Rules:
- Variable names, class field order, whitespace → do NOT matter.
- Extra print() statements → do NOT affect equivalence.
- The generated code may be structured differently as long as the outcome is the same.
- Missing import, wrong function name, wrong dataset name, wrong argument type → NOT equivalent.
- Calling the right function with wrong or missing fields → PARTIAL.

Return a JSON object and NOTHING else:
{
  "verdict":   "EQUIVALENT" | "PARTIAL" | "NOT_EQUIVALENT",
  "score":     <float 0.0–1.0>,
  "reasoning": "<one sentence>",
  "issues":    ["<issue>", ...]
}"""

_USER_TEMPLATE = """\
USER INTENT:
{intent}

REFERENCE (known correct BraneScript):
```
{reference}
```

GENERATED (model output to evaluate):
```
{generated}
```

Is the GENERATED code functionally equivalent to the REFERENCE?
Respond with JSON only — no prose, no markdown fences."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge(
    intent:    str,
    reference: str,
    generated: str,
    api:       str = None,
    model:     str = None,
) -> JudgeResult:
    """
    Ask an external LLM whether *generated* is functionally equivalent
    to *reference* for the given *intent*.

    Parameters
    ----------
    intent    : original natural-language user request
    reference : known-correct BraneScript (LLM-generated ground truth)
    generated : model output to evaluate
    api       : "anthropic" or "openai" (overrides JUDGE_API env var)
    model     : model name (overrides JUDGE_MODEL env var)
    """
    api   = api   or JUDGE_API
    model = model or JUDGE_MODEL

    user_msg = _USER_TEMPLATE.format(
        intent=intent,
        reference=reference.strip(),
        generated=generated.strip(),
    )

    try:
        if api == "anthropic":
            raw = _call_anthropic(model or _ANTHROPIC_DEFAULT, user_msg)
        elif api == "openai":
            raw = _call_openai(model or _OPENAI_DEFAULT, user_msg)
        else:
            return JudgeResult.error(f"Unknown JUDGE_API: {api!r}")

        return _parse_result(raw)
    except Exception as exc:
        return JudgeResult.error(f"Judge call failed: {exc}")


def judge_batch(
    items: list[dict],
    api:   str = None,
    model: str = None,
) -> list[JudgeResult]:
    """
    Judge a list of dicts, each with keys: intent, reference, generated.
    Returns one JudgeResult per item in the same order.
    """
    return [
        judge(d["intent"], d["reference"], d["generated"], api=api, model=model)
        for d in items
    ]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_anthropic(model: str, user_msg: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return msg.content[0].text


def _call_openai(model: str, user_msg: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_result(raw: str) -> JudgeResult:
    # Strip any accidental markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return JudgeResult.error(f"Could not parse judge response: {text[:200]}")

    verdict = d.get("verdict", "NOT_EQUIVALENT").upper()
    score   = float(d.get("score", 0.0))
    score   = max(0.0, min(1.0, score))

    return JudgeResult(
        verdict    = verdict,
        equivalent = verdict == "EQUIVALENT",
        score      = score,
        reasoning  = d.get("reasoning", ""),
        issues     = d.get("issues", []),
    )
