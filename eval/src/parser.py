"""Parse LLM raw text into {question_no: 'A'|'B'|'C'|'D'|None}."""
from __future__ import annotations

import json
import re
from typing import Iterable

_LETTER_RE = re.compile(r"[ABCD]")
_KV_RE = re.compile(r'"(\d{1,3})"\s*:\s*"?([ABCD])"?', re.IGNORECASE)


def _coerce_letter(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    m = _LETTER_RE.search(s)
    return m.group(0) if m else None


def parse_letters(content: str, expected_qnos: Iterable[str]) -> dict[str, str | None]:
    """Robustly extract {qno: letter} from raw LLM content.

    Strategy:
      1. Try strict json.loads -> dict
      2. Fall back to regex on `"\d+"\s*:\s*"[A-D]"` pairs
      3. Fill missing keys with None
    """
    out: dict[str, str | None] = {}

    # 1. Strict JSON.
    parsed = None
    try:
        parsed = json.loads(content)
    except Exception:
        # Try to find the first {...} block.
        start = content.find("{")
        end = content.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(content[start : end + 1])
            except Exception:
                parsed = None

    if isinstance(parsed, dict):
        for k, v in parsed.items():
            ks = str(k).strip()
            if ks.isdigit():
                letter = _coerce_letter(v)
                if letter:
                    out[ks] = letter

    # 2. Regex fallback for any keys still missing.
    if not out or any(q not in out for q in expected_qnos):
        for m in _KV_RE.finditer(content):
            qno = m.group(1)
            letter = m.group(2).upper()
            out.setdefault(qno, letter)

    # 3. Fill missing keys with None to keep downstream stable.
    for q in expected_qnos:
        out.setdefault(q, None)
    return out
