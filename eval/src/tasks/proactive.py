"""Proactive-prediction task — v3 with TEXT and IMAGE settings sharing one pool.

Pool definition (both settings)
  - "Positive day" = day having >=1 anchor with files != [] AND
    agentHelpReason non-empty AND image present on disk.
  - On every positive day, ALL image-bearing anchors become candidates,
    regardless of whether they themselves carry agentHelpReason.
  - GT.is_proactive = bool(agentHelpReason) for each candidate.

LLM call unit
  - One call per positive day. The model judges every image-bearing anchor
    on that day at once and returns multi-event JSON.

TEXT setting (proactive_text)
  - Prior days: full detail.
  - Target day candidates: rendered as facts-only text (domain + time_bucket
    + structured payload, no description / mood / agentHelpReason).
  - No images attached.

IMAGE setting (proactive_image)
  - Prior days: full detail (text).
  - Target day candidates: only id + domain + time_bucket as a header line per
    anchor; images are attached IN ORDER matching the listed anchor ids.
  - The model sees images and minimal text headers.

Output (both settings)
  {"<event_id>": {"needs_proactive": bool, "reason": "..." | null}, ...}

Metrics
  - Binary cls per anchor: tp/fp/fn/tn -> accuracy/precision/recall/F1
  - Reason match (compound): denom = #{GT=true};
                              num = #{GT=true ^ pred=true ^ judge YES}.
"""
from __future__ import annotations

import json
from typing import Any

from .. import loader, serializer

# ---- Prompts -------------------------------------------------------------

PROACTIVE_TEXT_SYSTEM = (
    "You are a proactive assistance evaluator. Given a user's prior daily activity "
    "(full detail) and a list of TODAY's anchor events shown as facts only "
    "(domain, time_bucket, structured payload — no descriptions, no mood, no "
    "agent-help annotations), decide for each anchor whether the user would "
    "clearly benefit from proactive AI assistance at that moment. If yes, give "
    "a one-sentence reason describing the user's underlying motivation or "
    "trigger condition (focus on the specific need, not generic helpfulness).\n\n"
    "Output STRICT JSON of the form:\n"
    '{"<event_id>": {"needs_proactive": true|false, "reason": "<one sentence or null>"}, ...}\n\n'
    "Rules:\n"
    "- Use exactly the event_id strings provided in the prompt.\n"
    "- needs_proactive must be true OR false (not null).\n"
    "- If needs_proactive=false, set reason to null.\n"
    "- Be selective; not every event needs proactive help."
)

PROACTIVE_IMAGE_SYSTEM = (
    "You are a proactive assistance evaluator. Given a user's prior daily activity "
    "(full detail, text) and a list of TODAY's anchor events represented ONLY "
    "by attached images (one image per anchor, in the order listed), decide for "
    "each anchor whether the user would clearly benefit from proactive AI "
    "assistance at that moment. If yes, give a one-sentence reason describing "
    "the user's underlying motivation or trigger condition.\n\n"
    "Output STRICT JSON of the form:\n"
    '{"<event_id>": {"needs_proactive": true|false, "reason": "<one sentence or null>"}, ...}\n\n'
    "Rules:\n"
    "- Use exactly the event_id strings provided in the prompt.\n"
    "- The N attached images, in order, correspond to the N anchor_ids listed.\n"
    "- needs_proactive must be true OR false. If false, reason must be null.\n"
    "- Be selective; not every anchor needs proactive help."
)


# ---- Core pool extraction (shared by text and image) ---------------------

def _is_positive_anchor(e: dict[str, Any]) -> bool:
    """An anchor is positive iff it has image AND agentHelpReason."""
    if not e.get("is_real_log"):
        return False
    files = e.get("files") or []
    if not files:
        return False
    if serializer.image_path_for(files[0]) is None:
        return False
    ahr = (e.get("payload") or {}).get("agentHelpReason") or ""
    return bool(ahr)


def _is_image_bearing_anchor(e: dict[str, Any]) -> bool:
    """An anchor we can predict on iff it has an image (regardless of label)."""
    if not e.get("is_real_log"):
        return False
    files = e.get("files") or []
    if not files:
        return False
    return serializer.image_path_for(files[0]) is not None


def extract_targets_proactive(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one entry per positive day. Each entry has all image-bearing
    anchors on that day labeled with GT.

    [
      {
        "day_index": int,
        "anchors": [
          {event_id, seq, domain, time_bucket,
           image_path, image_relpath,
           is_proactive_gt: bool, agentHelpReason: str|None}, ...
        ]
      }, ...
    ]
    """
    days = traj.get("days") or {}
    sorted_dks = sorted(days.keys(), key=lambda x: int(x.replace("Day", "")))

    out: list[dict[str, Any]] = []
    for dk in sorted_dks:
        day = days[dk]
        di = day.get("day_index")
        events = day.get("events") or {}

        # Step 1: skip days without any positive anchor
        has_pos = any(_is_positive_anchor(e) for e in events.values())
        if not has_pos:
            continue

        # Step 2: collect ALL image-bearing anchors as candidates
        anchors: list[dict[str, Any]] = []
        for ek in sorted(events.keys(), key=lambda x: int(x.replace("event", ""))):
            e = events[ek]
            if not _is_image_bearing_anchor(e):
                continue
            files = e.get("files") or []
            ipath = serializer.image_path_for(files[0])
            ahr = (e.get("payload") or {}).get("agentHelpReason") or ""
            anchors.append(
                {
                    "event_id": e.get("event_id"),
                    "seq": e.get("seq"),
                    "domain": e.get("domain"),
                    "time_bucket": e.get("time_bucket"),
                    "image_path": str(ipath),
                    "image_relpath": files[0].get("original_storage_path"),
                    "is_proactive_gt": bool(ahr),
                    "agentHelpReason": ahr if ahr else None,
                }
            )
        if anchors:
            out.append({"day_index": di, "anchors": anchors})
    return out


# ---- Eligibility ---------------------------------------------------------

def discover_eligible_users_proactive(min_days: int = 1) -> list[str]:
    """Users with at least `min_days` positive days."""
    out = []
    for u in loader.discover_users():
        try:
            traj = loader.load_trajectory(u)
        except Exception:
            continue
        if len(extract_targets_proactive(traj)) >= min_days:
            out.append(u)
    return out


def count_positive_days(traj: dict[str, Any]) -> int:
    return len(extract_targets_proactive(traj))


def count_total_samples_proactive(traj: dict[str, Any]) -> int:
    return sum(len(t["anchors"]) for t in extract_targets_proactive(traj))


def count_positive_samples_proactive(traj: dict[str, Any]) -> int:
    return sum(
        1
        for t in extract_targets_proactive(traj)
        for a in t["anchors"]
        if a["is_proactive_gt"]
    )


# ---- Aliases for existing callers ----------------------------------------

def extract_targets_text(traj: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    """v3: the text setting uses the same positive-day pool as image. The
    `k` parameter is ignored (kept for backward call signatures)."""
    return extract_targets_proactive(traj)


def extract_targets_image(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """v3: image setting also returns per-day groups (not per-event).
    Same pool as text."""
    return extract_targets_proactive(traj)


def discover_eligible_users_text(min_anchors: int = 1) -> list[str]:
    return discover_eligible_users_proactive(min_days=1)


def discover_eligible_users_image(min_samples: int = 1) -> list[str]:
    return discover_eligible_users_proactive(min_days=1)


def count_samples_text(traj: dict[str, Any], k: int = 5) -> int:
    return count_total_samples_proactive(traj)


def count_samples_image(traj: dict[str, Any]) -> int:
    return count_total_samples_proactive(traj)


# ---- Prompt building -----------------------------------------------------

def _prior_indices(traj: dict[str, Any], target_di: int) -> list[int]:
    out: list[int] = []
    for dk in (traj.get("days") or {}):
        di = traj["days"][dk].get("day_index")
        if isinstance(di, int) and di < target_di:
            out.append(di)
    return out


def _find_day_events(traj: dict[str, Any], target_di: int) -> dict[str, Any]:
    for dk, d in (traj.get("days") or {}).items():
        if d.get("day_index") == target_di:
            return d.get("events") or {}
    return {}


def build_prompt_text(
    traj: dict[str, Any], target_day: dict[str, Any]
) -> tuple[str, str]:
    """(system, user_text) for one target day in TEXT setting.

    Renders only the IMAGE-BEARING anchors on that day, as facts-only text.
    """
    target_di = int(target_day["day_index"])
    anchors = target_day["anchors"]
    anchor_ids = [a["event_id"] for a in anchors]
    eid_set = set(anchor_ids)

    sections: list[str] = ["=== USER PRIOR ACTIVITY (full detail) ==="]
    sections.append(serializer.serialize_trajectory_window(traj, _prior_indices(traj, target_di)))

    sections.append(
        f"\n=== TODAY (Day{target_di}) — IMAGE-BEARING ANCHORS, FACTS ONLY ==="
    )
    sections.append(
        "(Only the image-bearing anchors below are candidates. Each is listed "
        "with structured payload facts only — no description, no mood, no "
        "agent-help annotation.)"
    )
    day_events = _find_day_events(traj, target_di)
    rendered_for_eid: dict[str, str] = {}
    for ev in day_events.values():
        if ev.get("event_id") in eid_set:
            rendered_for_eid[ev["event_id"]] = serializer.serialize_event_facts_only(ev)
    for a in anchors:
        eid = a["event_id"]
        if eid in rendered_for_eid:
            sections.append(rendered_for_eid[eid])
        else:
            sections.append(
                f"- id={eid} [seq{a['seq']}, time={a['time_bucket']}, "
                f"domain={a['domain']}] (no structured data)"
            )

    sections.append("\n=== QUESTION ===")
    sections.append(
        "For each anchor event listed above, decide whether the user would "
        "clearly benefit from proactive AI assistance at that moment. If yes, "
        "give a one-sentence reason describing the underlying motivation or "
        "trigger.\nAnchor event_ids:"
    )
    for aid in anchor_ids:
        sections.append(f"  - {aid}")
    sections.append(
        '\nReturn STRICT JSON: {"<event_id>": {"needs_proactive": bool, '
        '"reason": "..." | null}, ...}\n'
        "All listed event_ids MUST appear as keys."
    )

    return PROACTIVE_TEXT_SYSTEM, "\n".join(sections)


def build_prompt_image(
    traj: dict[str, Any], target_day: dict[str, Any]
) -> tuple[str, str, list[str]]:
    """(system, user_text, image_paths) for one target day in IMAGE setting.

    Attaches one image per anchor, in order. The text only contains anchor
    headers (id + domain + time_bucket) — no payload details.
    """
    target_di = int(target_day["day_index"])
    anchors = target_day["anchors"]
    anchor_ids = [a["event_id"] for a in anchors]

    sections: list[str] = ["=== USER PRIOR ACTIVITY (full detail) ==="]
    sections.append(serializer.serialize_trajectory_window(traj, _prior_indices(traj, target_di)))

    sections.append(
        f"\n=== TODAY (Day{target_di}) — IMAGE-BEARING ANCHORS ==="
    )
    sections.append(
        f"There are {len(anchors)} anchors today. Each has ONE image attached "
        "below in the order listed. No other detail of these anchors is given."
    )
    for i, a in enumerate(anchors, start=1):
        sections.append(
            f"  Anchor {i}: id={a['event_id']} "
            f"[seq{a['seq']}, time={a['time_bucket']}, domain={a['domain']}]"
        )

    sections.append("\n=== QUESTION ===")
    sections.append(
        "Based on the user's prior pattern and the attached images (in order), "
        "decide for each anchor whether the user would clearly benefit from "
        "proactive AI assistance. If yes, give a one-sentence reason.\n"
        "Anchor event_ids in the same order as the attached images:"
    )
    for aid in anchor_ids:
        sections.append(f"  - {aid}")
    sections.append(
        '\nReturn STRICT JSON: {"<event_id>": {"needs_proactive": bool, '
        '"reason": "..." | null}, ...}\n'
        "All listed event_ids MUST appear as keys."
    )

    image_paths = [a["image_path"] for a in anchors]
    return PROACTIVE_IMAGE_SYSTEM, "\n".join(sections), image_paths


# ---- Response parsing ----------------------------------------------------

def _try_parse_json(content: str) -> Any:
    try:
        return json.loads(content)
    except Exception:
        s = content.find("{")
        e = content.rfind("}")
        if 0 <= s < e:
            try:
                return json.loads(content[s : e + 1])
            except Exception:
                return None
    return None


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    if isinstance(v, (int, float)):
        return bool(v)
    return False


def _clean_reason(r: Any) -> str | None:
    if r is None:
        return None
    s = str(r).strip()
    if not s or s.lower() == "null":
        return None
    return s


def parse_response_text(content: str, anchor_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Parse multi-event JSON. Missing keys default to false / null / missing."""
    out: dict[str, dict[str, Any]] = {}
    parsed = _try_parse_json(content)
    if isinstance(parsed, dict):
        for aid in anchor_ids:
            v = parsed.get(aid)
            if not isinstance(v, dict):
                out[aid] = {"needs_proactive": False, "reason": None, "missing": True}
                continue
            out[aid] = {
                "needs_proactive": _coerce_bool(v.get("needs_proactive")),
                "reason": _clean_reason(v.get("reason")),
                "missing": False,
            }
    else:
        for aid in anchor_ids:
            out[aid] = {"needs_proactive": False, "reason": None, "missing": True}
    return out


def parse_response_image(content: str, anchor_ids: list[str]) -> dict[str, dict[str, Any]]:
    """v3: image setting now also returns multi-event JSON. Same as text."""
    return parse_response_text(content, anchor_ids)


# ---- Per-day metrics -----------------------------------------------------

def compute_binary_metrics_day(
    target_day: dict[str, Any], parsed: dict[str, dict[str, Any]]
) -> dict[str, int]:
    tp = fp = fn = tn = 0
    n_anchors = 0
    for a in target_day["anchors"]:
        gt = bool(a["is_proactive_gt"])
        p = parsed.get(a["event_id"]) or {}
        pred = bool(p.get("needs_proactive"))
        n_anchors += 1
        if gt and pred:
            tp += 1
        elif (not gt) and pred:
            fp += 1
        elif gt and (not pred):
            fn += 1
        else:
            tn += 1
    return {
        "n_anchors": n_anchors,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_pos_gt": tp + fn,
    }


# ---- Backwards-compat names ---------------------------------------------

def extract_targets(traj: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    return extract_targets_proactive(traj)


def build_prompt(traj: dict[str, Any], target: dict[str, Any]) -> tuple[str, str]:
    return build_prompt_text(traj, target)


def parse_response(content: str, anchor_ids: list[str]) -> dict[str, dict[str, Any]]:
    return parse_response_text(content, anchor_ids)


def compute_binary_metrics(target: dict[str, Any], parsed: dict[str, dict[str, Any]]) -> dict[str, int]:
    return compute_binary_metrics_day(target, parsed)


def discover_eligible_users(min_days: int = 1) -> list[str]:
    return discover_eligible_users_proactive(min_days=min_days)
