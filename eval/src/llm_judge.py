"""GPT-5.4-mini judge for matching predicted vs ground-truth proactive reasons.

Cache layout: eval/runs/.cache_judge/{ab}/{full_hash}.json
Separated from main API cache so survey predictions and judges don't collide.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

JUDGE_CACHE_DIR = Path(__file__).resolve().parents[1] / "runs" / ".cache_judge"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"

JUDGE_SYSTEM = (
    "You are a careful evaluator. Given a ground-truth (GT) reason for why a user "
    "wanted proactive AI help in a specific situation and a model-predicted reason, "
    "decide if the predicted reason describes the SAME underlying motivation. "
    "Match on intent and trigger, not exact wording. Synonyms, paraphrase, or more/less "
    "specific phrasings still count as a match if the core motivation is preserved. "
    "Output strict JSON: {\"match\": true|false, \"explanation\": \"<one sentence>\"}."
)


def _judge_key(model_id: str, gt_reason: str, pred_reason: str) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x1f")
    h.update(gt_reason.strip().encode("utf-8"))
    h.update(b"\x1f")
    h.update(pred_reason.strip().encode("utf-8"))
    return h.hexdigest()


def _judge_path(key: str) -> Path:
    return JUDGE_CACHE_DIR / key[:2] / f"{key}.json"


def judge_reason_match(
    *,
    gt_reason: str,
    pred_reason: str,
    judge_model_id: str = DEFAULT_JUDGE_MODEL,
    use_cache: bool = True,
    max_tokens: int = 8000,
) -> dict[str, Any]:
    """Return {match: bool, explanation: str, cache_key, cached, raw}.

    Empty inputs return match=False without calling the API.
    """
    gt = (gt_reason or "").strip()
    pr = (pred_reason or "").strip()
    if not gt or not pr:
        return {
            "match": False,
            "explanation": "empty input (no gt or pred reason)",
            "cache_key": None,
            "cached": False,
            "raw": None,
        }

    key = _judge_key(judge_model_id, gt, pr)
    cpath = _judge_path(key)
    if use_cache and cpath.exists():
        rec = json.load(open(cpath, encoding="utf-8"))
        rec["cached"] = True
        return rec

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set. Required to call the LLM judge."
        )

    from openai import OpenAI

    client = OpenAI()
    user_msg = (
        f"GT reason: {gt}\n\n"
        f"Predicted reason: {pr}\n\n"
        "Output JSON now."
    )
    create_kwargs: dict[str, Any] = {
        "model": judge_model_id,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    def _is_unsupported(msg: str, k: str) -> bool:
        m = msg.lower()
        return k.lower() in m and (
            "unsupported" in m or "not supported" in m or "invalid" in m
        )

    resp = None
    last_err: Exception | None = None
    rl = 0
    for _ in range(8):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            adjusted = False
            if "max_tokens" in create_kwargs and (
                "max_completion_tokens" in msg or _is_unsupported(msg, "max_tokens")
            ):
                create_kwargs["max_completion_tokens"] = create_kwargs.pop("max_tokens")
                adjusted = True
            elif "temperature" in create_kwargs and _is_unsupported(msg, "temperature"):
                create_kwargs.pop("temperature", None)
                adjusted = True
            elif "response_format" in create_kwargs and (
                "response_format" in msg or "json" in msg.lower()
            ):
                create_kwargs.pop("response_format", None)
                adjusted = True
            elif "rate_limit" in msg.lower() or "429" in msg:
                rl += 1
                if rl > 5:
                    raise
                wait = min(60, 5 * (2 ** (rl - 1)))
                print(f"  [judge rate-limit] sleep {wait}s", flush=True)
                time.sleep(wait)
                adjusted = True
            elif "timeout" in msg.lower() or "connection" in msg.lower():
                rl += 1
                if rl > 3:
                    raise
                time.sleep(5 * rl)
                adjusted = True
            if not adjusted:
                raise
    if resp is None:
        raise last_err if last_err else RuntimeError("judge call failed")

    content = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
    except Exception:
        # try to grab first {...}
        s = content.find("{")
        e = content.rfind("}")
        try:
            parsed = json.loads(content[s : e + 1]) if 0 <= s < e else {}
        except Exception:
            parsed = {}

    match = bool(parsed.get("match", False))
    explanation = str(parsed.get("explanation", "")).strip()
    rec = {
        "match": match,
        "explanation": explanation,
        "cache_key": key,
        "cached": False,
        "raw": content,
    }
    if content:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec
