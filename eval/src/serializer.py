"""Render trajectory/persona JSON into compact text suitable for LLM context.

Design goals:
- Keep total length predictable (truncate description_long to ~2 sentences).
- Preserve the most decision-relevant signals: domain, time_bucket, payload, derived,
  emotional/proactive flags, day theme & dominant mood.
- Same serializer is used for all conditions; conditions only toggle which sections
  end up in the final prompt.

Facts-only mode (used for target days/events in proactive_text / emotion_text):
- Strips description_long, summary_text, thematic_continuity (synth narrator can
  leak GT signals it was conditioned on).
- Strips payload free-text fields (mealDesc, exerciseDetail, moodNote,
  disturbanceNote, eventDescription) AND ALL leak fields (agentHelpReason,
  moodKeywords, agentHelp).
- Keeps only domain, time_bucket, and structured numeric/enum/bool payload fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_MAX_DESC_SENTENCES = 2

# Project root: eval/src/serializer.py -> JarvisBench/
_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = _ROOT / "Data" / "processed" / "image"

# Payload fields to ALWAYS strip in facts-only mode (leak GT or are free-text prose).
_LEAK_PAYLOAD_FIELDS = {
    # Direct GT leakage:
    "agentHelpReason",
    "moodKeywords",
    "agentHelp",  # legacy bool that often co-occurs with proactive
    # Free-text user input that frequently embeds mood/intent prose:
    "mealDesc",
    "exerciseDetail",
    "eventDescription",
    "moodNote",
    "disturbanceNote",
    "agentHelpNote",
    "noteForAgent",
    "comment",
    "notes",
    "remark",
    "reflection",
}

# Max length for a payload string value to be considered "structured" (anything
# longer is treated as free-text prose and dropped in facts-only mode).
_FACTS_MAX_STR_LEN = 60


def _truncate_description(desc: str, max_sentences: int = _MAX_DESC_SENTENCES) -> str:
    if not desc:
        return ""
    # naive sentence split; OK for log-style prose
    parts = [p.strip() for p in desc.replace("\n", " ").split(". ") if p.strip()]
    snippet = ". ".join(parts[:max_sentences]).strip()
    if snippet and not snippet.endswith("."):
        snippet += "."
    return snippet


def _format_payload(payload: dict[str, Any]) -> str:
    """Flatten one-level scalar fields of payload into a compact 'k=v; k=v' string."""
    if not payload:
        return ""
    items = []
    for k, v in payload.items():
        if v is None or v == "":
            continue
        if isinstance(v, (dict, list)):
            # collapse small dicts/lists
            try:
                import json as _j
                s = _j.dumps(v, ensure_ascii=False)
            except Exception:
                s = str(v)
            if len(s) > 160:
                s = s[:157] + "..."
            items.append(f"{k}={s}")
        else:
            items.append(f"{k}={v}")
    return "; ".join(items)


def serialize_trajectory(
    traj: dict[str, Any],
    max_days: int | None = None,
    event_domain_filter=None,
) -> str:
    """Convert trajectory JSON to a markdown-ish text block.

    event_domain_filter: optional callable(event_dict)->bool; events for which
    it returns False are skipped. Used by Domain Transfer ablations.
    """
    days = traj.get("days", {})
    day_keys = sorted(days.keys(), key=lambda k: int(k.replace("Day", "")))
    if max_days is not None:
        day_keys = day_keys[:max_days]

    lines: list[str] = []
    lines.append(f"User: {traj.get('user_slug','?')} | Days included: {len(day_keys)}/{traj.get('day_count','?')}")

    for dk in day_keys:
        day = days[dk]
        weekend = "weekend" if day.get("is_weekend") else "weekday"
        mood = day.get("dominant_mood", "")
        header = f"\n## {dk} ({weekend}"
        if mood:
            header += f", mood: {mood}"
        header += ")"
        lines.append(header)

        if day.get("day_theme"):
            lines.append(f"Theme: {day['day_theme']}")

        env = day.get("environment", {}) or {}
        if env:
            env_bits = []
            for k in ("setting", "weather_hint", "ambient_notes"):
                if env.get(k):
                    env_bits.append(f"{k}={env[k]}")
            if env_bits:
                lines.append("Env: " + "; ".join(env_bits))

        derived = day.get("derived", {}) or {}
        if derived:
            d_str = ", ".join(f"{k}={v}" for k, v in derived.items() if v not in (None, ""))
            if d_str:
                lines.append(f"Derived: {d_str}")

        events = day.get("events", {}) or {}
        ev_keys = sorted(events.keys(), key=lambda x: int(x.replace("event", "")))
        for ek in ev_keys:
            e = events[ek]
            if event_domain_filter is not None and not event_domain_filter(e):
                continue
            tag = "REAL" if e.get("is_real_log") else "synth"
            flags = []
            if e.get("is_emotional"):
                flags.append("emo")
            if e.get("is_proactive"):
                flags.append("proactive")
            if e.get("has_image"):
                flags.append("img")
            flag_str = f" [{','.join(flags)}]" if flags else ""

            head = (
                f"  - [{tag}] seq{e.get('seq')} {e.get('time_bucket','')} "
                f"<{e.get('domain','')}>{flag_str}: {e.get('summary_text','')}"
            )
            lines.append(head)

            desc = _truncate_description(e.get("description_long", ""))
            if desc:
                lines.append(f"    {desc}")

            payload_str = _format_payload(e.get("payload", {}) or {})
            if payload_str:
                lines.append(f"    Data: {payload_str}")

            ev_derived = e.get("derived", {}) or {}
            if ev_derived:
                d_str = "; ".join(f"{k}={v}" for k, v in ev_derived.items() if v not in (None, ""))
                if d_str:
                    lines.append(f"    Derived: {d_str}")

    return "\n".join(lines)


def serialize_trajectory_window(
    traj: dict[str, Any],
    day_indices: list[int],
    event_domain_filter=None,
) -> str:
    """Serialize only the days whose day_index is in `day_indices` (full detail).

    event_domain_filter: optional event-level filter, see serialize_trajectory.
    """
    days = traj.get("days", {})
    target = set(day_indices)
    keep = {dk: d for dk, d in days.items() if d.get("day_index") in target}
    if not keep:
        return f"User: {traj.get('user_slug','?')} | (no days in window)"
    sub = dict(traj)
    sub["days"] = keep
    # Re-use main serializer (no max_days truncation; we already filtered).
    return serialize_trajectory(sub, max_days=None, event_domain_filter=event_domain_filter)


def serialize_day_descriptions_only(
    day: dict[str, Any],
    max_seq_exclusive: int | None = None,
    include_anchor_ids: bool = True,
) -> str:
    """Render ONE day's events using ONLY description_long / summary_text.

    No payload, no derived, no flags, no domain — pure prose so the model can't
    cheat by reading proactive markers or moodKeywords.
    Used for proactive task day-D and emotion task day-D.

    Args:
      max_seq_exclusive: only emit events with seq < this value. None = all.
      include_anchor_ids: prefix each line with event_id (needed by proactive).
    """
    dk = f"Day{day.get('day_index','?')}"
    weekend = "weekend" if day.get("is_weekend") else "weekday"
    lines = [f"## {dk} ({weekend})"]

    events = day.get("events", {}) or {}
    ev_keys = sorted(events.keys(), key=lambda x: int(x.replace("event", "")))
    n_emitted = 0
    for ek in ev_keys:
        e = events[ek]
        seq = e.get("seq")
        if max_seq_exclusive is not None and isinstance(seq, int) and seq >= max_seq_exclusive:
            break
        anchor_tag = ""
        if e.get("is_real_log"):
            anchor_tag = " [ANCHOR]"
            if include_anchor_ids:
                anchor_tag = f" [ANCHOR id={e.get('event_id')}]"
        prose = (e.get("description_long") or e.get("summary_text") or "").strip()
        if not prose:
            continue
        lines.append(f"- (seq{seq}){anchor_tag} {prose}")
        n_emitted += 1
    if n_emitted == 0:
        lines.append("- (no events)")
    return "\n".join(lines)


def _format_payload_facts_only(payload: dict[str, Any]) -> str:
    """Like _format_payload but drops leak fields and free-text strings."""
    if not payload:
        return ""
    items = []
    for k, v in payload.items():
        if k in _LEAK_PAYLOAD_FIELDS:
            continue
        if v is None or v == "" or v == [] or v == {}:
            continue
        if isinstance(v, str):
            if len(v) > _FACTS_MAX_STR_LEN:
                continue  # likely free-text prose
            items.append(f"{k}={v}")
        elif isinstance(v, (int, float, bool)):
            items.append(f"{k}={v}")
        elif isinstance(v, (list, dict)):
            try:
                import json as _j
                s = _j.dumps(v, ensure_ascii=False)
            except Exception:
                s = str(v)
            if len(s) > _FACTS_MAX_STR_LEN:
                continue
            items.append(f"{k}={s}")
    return "; ".join(items)


def serialize_event_facts_only(event: dict[str, Any]) -> str:
    """Render ONE event with only structural facts (no description / summary / mood / reason).

    Output format:
      - id=<eid> [seq{N}, time=<bucket>, domain=<d>] [tags...]
          Data: k1=v1; k2=v2; ...
    """
    eid = event.get("event_id", "?")
    seq = event.get("seq", "?")
    tb = event.get("time_bucket", "?")
    domain = event.get("domain", "?")
    tags = []
    if event.get("is_real_log"):
        tags.append("ANCHOR")
    if event.get("has_image"):
        tags.append("img")
    tag_str = f" [{','.join(tags)}]" if tags else ""
    head = f"- id={eid} [seq{seq}, time={tb}, domain={domain}]{tag_str}"

    payload = event.get("payload") or {}
    bits = _format_payload_facts_only(payload)
    if bits:
        return head + "\n    Data: " + bits
    return head + " (no structured data)"


def serialize_day_facts_only(
    day: dict[str, Any],
    anchors_only: bool = True,
    max_seq_exclusive: int | None = None,
) -> str:
    """Render ONE day's events using facts-only mode (no prose at all)."""
    di = day.get("day_index", "?")
    weekend = "weekend" if day.get("is_weekend") else "weekday"
    lines = [f"## Day{di} ({weekend})"]
    events = day.get("events", {}) or {}
    n_emitted = 0
    for ek in sorted(events.keys(), key=lambda x: int(x.replace("event", ""))):
        e = events[ek]
        seq = e.get("seq")
        if max_seq_exclusive is not None and isinstance(seq, int) and seq >= max_seq_exclusive:
            break
        if anchors_only and not e.get("is_real_log"):
            continue
        lines.append(serialize_event_facts_only(e))
        n_emitted += 1
    if n_emitted == 0:
        lines.append("- (no qualifying events)")
    return "\n".join(lines)


def image_path_for(file_entry: dict[str, Any]) -> Path | None:
    """Resolve `original_storage_path` to an absolute on-disk path. Returns None
    if the path is missing or the file does not exist."""
    if not isinstance(file_entry, dict):
        return None
    osp = file_entry.get("original_storage_path") or ""
    if not osp:
        return None
    p = (IMAGE_ROOT / osp).resolve()
    if not p.exists():
        return None
    return p


def serialize_persona(persona_user: dict[str, Any]) -> str:
    """persona_data[user] = {structured: str, summary: str, updatedAt: ...}"""
    parts: list[str] = []
    structured = persona_user.get("structured")
    if structured:
        parts.append("--- Structured Profile ---\n" + str(structured).strip())
    summary = persona_user.get("summary")
    if summary:
        parts.append("--- Persona Summary ---\n" + str(summary).strip())
    return "\n\n".join(parts)


def _stringify(val: Any) -> str:
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (int, float, bool)) or val is None:
        return str(val)
    if isinstance(val, list):
        return "; ".join(_stringify(x) for x in val)
    if isinstance(val, dict):
        return "; ".join(f"{k}: {_stringify(v)}" for k, v in val.items())
    return str(val)


def _render_enhanced_profile(ep: dict[str, Any]) -> str:
    """Render the enhanced_profile dict (each leaf is {value, source, confidence}).

    We strip source/confidence for prompt compactness; just emit `key: value`.
    """
    lines: list[str] = []
    for section, fields in ep.items():
        if not isinstance(fields, dict):
            lines.append(f"{section}: {_stringify(fields)}")
            continue
        section_pretty = section.replace("_", " ").title()
        lines.append(f"\n[{section_pretty}]")
        for fkey, fval in fields.items():
            if isinstance(fval, dict) and "value" in fval:
                v = fval["value"]
            else:
                v = fval
            lines.append(f"  - {fkey}: {_stringify(v)}")
    return "\n".join(lines)


def serialize_profile_persona_no_insight(profile_persona: dict[str, Any] | None) -> str:
    """For C3. Renders profile_persona MINUS survey_insight.

    Includes: enhanced_profile + uniqueness_signature + persona_narrative.
    """
    if not profile_persona:
        return ""

    parts: list[str] = []

    ep = profile_persona.get("enhanced_profile")
    if ep and isinstance(ep, dict):
        parts.append("--- Enhanced Profile ---")
        parts.append(_render_enhanced_profile(ep))

    us = profile_persona.get("uniqueness_signature")
    if us:
        parts.append("\n--- Uniqueness Signature ---")
        parts.append(_stringify(us))

    narr = profile_persona.get("persona_narrative")
    if narr:
        parts.append("\n--- Persona Narrative ---")
        parts.append(_stringify(narr))

    return "\n".join(parts)


def serialize_survey_insight(profile_persona: dict[str, Any] | None) -> str:
    """For C4 skyline only. Renders profile_persona.survey_insight to text."""
    if not profile_persona:
        return ""
    si = profile_persona.get("survey_insight")
    if not si:
        return ""

    parts: list[str] = ["--- Survey Insight (Distilled User Preferences) ---"]

    summary = si.get("implicit_preference_summary")
    if summary:
        parts.append(_stringify(summary))

    di_list = si.get("deep_inferences")
    if di_list:
        parts.append("\nDeep inferences:")
        for di in di_list:
            if isinstance(di, dict):
                claim = di.get("claim", "")
                did = di.get("id", "")
                support = di.get("supporting_questions") or []
                support_str = (
                    f" [supports Qs: {', '.join(map(str, support))}]" if support else ""
                )
                parts.append(f"- [{did}] {claim}{support_str}")
            else:
                parts.append(f"- {_stringify(di)}")

    div = si.get("domain_indexed_view")
    if div and isinstance(div, dict):
        parts.append("\nDomain-indexed view:")
        for dom, content in div.items():
            parts.append(f"### {dom}")
            parts.append(_stringify(content))

    return "\n".join(parts)
