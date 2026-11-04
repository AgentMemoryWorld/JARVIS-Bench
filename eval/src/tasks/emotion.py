"""Emotion-prediction task — v3 with TEXT and IMAGE settings sharing one pool.

Pool definition (both settings)
  - Sample = anchor event in the LAST 5 days of a user's trajectory that has
    BOTH files != [] (image present on disk) AND moodKeywords != [].
  - Mixed-polarity GT (positive + negative moods together) is allowed.

TEXT setting (emotion_text)
  - Context: prior days (full detail) + day-D events with seq < target_seq
    as facts-only text.
  - Target event itself: facts-only (domain + time_bucket + structured payload
    minus moodKeywords / moodNote / etc.).
  - No images attached.
  - One LLM call per sample.

IMAGE setting (emotion_image)
  - Context: prior days (full detail) + day-D events with seq < target_seq
    as facts-only text.
  - Target event: IMAGE only, no payload text.
  - One LLM call per sample.

Output (both settings)
  {"polarity": "positive"|"negative"|"mixed", "moods": ["A","B"...]}
  - moods chosen from FULL 24 candidate list (no per-polarity restriction).
  - polarity is derived from selected moods (positive-only / negative-only /
    has both = mixed). The model also has to declare polarity; we score both
    declared and effective.

Metrics (3-way polarity)
  polarity_accuracy: predicted polarity == GT polarity (3-way)
  mood_f1:           2|P∩G| / (|P|+|G|) over the FULL 24-candidate space
  mood_jaccard:      |P∩G| / |P∪G|
  top1_accuracy:     |P∩G| >= 1
"""
from __future__ import annotations

import json
from typing import Any

from .. import loader, serializer

POSITIVE_MOODS = [
    "Fulfilled", "Calm", "Happy",
    "Excited", "Content", "Relaxed",
    "Grateful", "Confident", "Joyful",
    "Focused", "Warm", "Hopeful",
    "Proud",
]
NEGATIVE_MOODS = [
    "Anxious", "Exhausted", "Down",
    "Nervous", "Bored", "Lonely", "Lost",
    "Irritated", "Lazy", "Stressed",
    "Disappointed",
]
POSITIVE_SET = set(POSITIVE_MOODS)
NEGATIVE_SET = set(NEGATIVE_MOODS)
ALL_MOODS = POSITIVE_MOODS + NEGATIVE_MOODS
ALL_MOODS_LOWER = {m.lower(): m for m in ALL_MOODS}


def polarity_of(moods: list[str]) -> str:
    """Return 'positive' / 'negative' / 'mixed' / 'empty'."""
    s = set(moods)
    if not s:
        return "empty"
    pos = bool(s & POSITIVE_SET)
    neg = bool(s & NEGATIVE_SET)
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    if pos and neg:
        return "mixed"
    return "empty"


# ---- Prompts -------------------------------------------------------------

EMOTION_TEXT_SYSTEM = (
    "You are an emotional state classifier. Given a user's prior daily activity "
    "(full detail) and the events leading up to a target event today (FACTS ONLY: "
    "domain, time_bucket, structured payload — no descriptions, no mood prose), "
    "predict the user's mood DURING the target event.\n\n"
    "You must:\n"
    "1. Choose 1-3 mood keywords from the candidate list below.\n"
    "2. Declare the polarity of your chosen set: 'positive' (all positive moods), "
    "'negative' (all negative), or 'mixed' (contains both).\n"
    "Your declared polarity must be CONSISTENT with the moods you chose.\n\n"
    f"Positive candidates ({len(POSITIVE_MOODS)}): {', '.join(POSITIVE_MOODS)}\n"
    f"Negative candidates ({len(NEGATIVE_MOODS)}): {', '.join(NEGATIVE_MOODS)}\n\n"
    'Output STRICT JSON: {"polarity": "positive"|"negative"|"mixed", '
    '"moods": ["...", "..."]}.'
)

EMOTION_IMAGE_SYSTEM = (
    "You are an emotional state classifier. You will be given a user's prior daily "
    "activity (full detail, text) and an IMAGE attached to a single target anchor "
    "event from a later day. Based on the prior pattern and the image alone, predict "
    "the user's mood DURING the target event.\n\n"
    "You must:\n"
    "1. Choose 1-3 mood keywords from the candidate list below.\n"
    "2. Declare the polarity of your chosen set: 'positive' (all positive moods), "
    "'negative' (all negative), or 'mixed' (contains both).\n\n"
    f"Positive candidates ({len(POSITIVE_MOODS)}): {', '.join(POSITIVE_MOODS)}\n"
    f"Negative candidates ({len(NEGATIVE_MOODS)}): {', '.join(NEGATIVE_MOODS)}\n\n"
    'Output STRICT JSON: {"polarity": "positive"|"negative"|"mixed", '
    '"moods": ["...", "..."]}.'
)


# ---- Eligibility ---------------------------------------------------------

def discover_eligible_users_emotion(k_days: int = 5, min_samples: int = 1) -> list[str]:
    """Users with at least `min_samples` events in the last `k_days` having
    BOTH moodKeywords AND a present image."""
    out = []
    for u in loader.discover_users():
        try:
            traj = loader.load_trajectory(u)
        except Exception:
            continue
        if len(extract_targets_emotion(traj, k_days=k_days)) >= min_samples:
            out.append(u)
    return out


def count_samples_emotion(traj: dict[str, Any], k_days: int = 5) -> int:
    return len(extract_targets_emotion(traj, k_days=k_days))


# ---- Aliases (callers / aggregator continue to work) ---------------------

def discover_eligible_users_text(min_events: int = 1) -> list[str]:
    return discover_eligible_users_emotion(min_samples=1)


def discover_eligible_users_image(min_samples: int = 1) -> list[str]:
    return discover_eligible_users_emotion(min_samples=1)


def count_samples_text(traj: dict[str, Any], k: int = 5) -> int:
    return count_samples_emotion(traj, k_days=k)


def count_samples_image(traj: dict[str, Any]) -> int:
    return count_samples_emotion(traj, k_days=5)


# ---- Target extraction ---------------------------------------------------

def _normalize_mood_list(raw: list[Any]) -> list[str]:
    out: list[str] = []
    for m in raw:
        if not isinstance(m, str):
            continue
        ml = m.strip().lower()
        if ml in ALL_MOODS_LOWER:
            cm = ALL_MOODS_LOWER[ml]
            if cm not in out:
                out.append(cm)
    return out


def extract_targets_emotion(traj: dict[str, Any], k_days: int = 5) -> list[dict[str, Any]]:
    """v3: pool = anchors in the LAST `k_days` days that have BOTH non-empty
    moodKeywords AND a present image. Both text and image setting use this."""
    days = traj.get("days") or {}
    sorted_dks = sorted(days.keys(), key=lambda x: int(x.replace("Day", "")))
    last_k = sorted_dks[-k_days:]

    out: list[dict[str, Any]] = []
    for dk in last_k:
        day = days[dk]
        di = day.get("day_index")
        events = day.get("events") or {}
        for ek in sorted(events.keys(), key=lambda x: int(x.replace("event", ""))):
            e = events[ek]
            if not e.get("is_real_log"):
                continue
            payload = e.get("payload") or {}
            kws_raw = payload.get("moodKeywords") or []
            files = e.get("files") or []
            if not (kws_raw and files):
                continue
            ipath = serializer.image_path_for(files[0])
            if ipath is None:
                continue
            kws = _normalize_mood_list(kws_raw)
            if not kws:
                continue
            out.append(
                {
                    "event_id": e.get("event_id"),
                    "day_index": di,
                    "seq": e.get("seq"),
                    "domain": e.get("domain"),
                    "time_bucket": e.get("time_bucket"),
                    "image_path": str(ipath),
                    "image_relpath": files[0].get("original_storage_path"),
                    "gt_moods": kws,
                    "gt_polarity": polarity_of(kws),
                }
            )
    return out


def extract_targets_text(traj: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    """Alias of extract_targets_emotion(traj, k_days=k). Both settings share
    the same pool now (anchors in last k days with mood + image)."""
    return extract_targets_emotion(traj, k_days=k)


def extract_targets_image(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Same pool as extract_targets_text. (k_days=5)"""
    return extract_targets_emotion(traj, k_days=5)


# ---- Prompt building -----------------------------------------------------

def build_prompt_text(
    traj: dict[str, Any], target: dict[str, Any]
) -> tuple[str, str]:
    target_di = int(target["day_index"])
    target_seq = int(target["seq"])

    prior_indices: list[int] = []
    for dk in (traj.get("days") or {}):
        di = traj["days"][dk].get("day_index")
        if isinstance(di, int) and di < target_di:
            prior_indices.append(di)

    sections: list[str] = ["=== USER PRIOR ACTIVITY (full detail) ==="]
    sections.append(serializer.serialize_trajectory_window(traj, prior_indices))

    day_d = None
    for dk, d in (traj.get("days") or {}).items():
        if d.get("day_index") == target_di:
            day_d = d
            break

    sections.append(
        f"\n=== TODAY (Day{target_di}) — EVENTS BEFORE TARGET, FACTS ONLY ==="
    )
    if day_d is not None:
        sections.append(
            serializer.serialize_day_facts_only(
                day_d, anchors_only=False, max_seq_exclusive=target_seq
            )
        )
    else:
        sections.append("(day not found)")

    sections.append(
        f"\n=== TARGET EVENT (Day{target_di}, seq{target_seq}) — FACTS ONLY ==="
    )
    if day_d is not None:
        events = day_d.get("events") or {}
        target_ev = None
        for ek, ev in events.items():
            if ev.get("seq") == target_seq:
                target_ev = ev
                break
        if target_ev is not None:
            sections.append(serializer.serialize_event_facts_only(target_ev))
        else:
            sections.append("(target event not found)")
    else:
        sections.append("(day not found)")

    sections.append(
        "\n=== QUESTION ===\n"
        "Classify the user's mood during the TARGET EVENT.\n"
        "Choose 1-3 mood keywords from the FULL candidate list (positive AND/OR "
        "negative). Then declare the polarity of your chosen set ('positive', "
        "'negative', or 'mixed'). The polarity MUST be consistent with the moods "
        "you picked.\n"
        'Output JSON: {"polarity": "positive"|"negative"|"mixed", "moods": ["...","..."]}'
    )

    return EMOTION_TEXT_SYSTEM, "\n".join(sections)


def build_prompt_image(
    traj: dict[str, Any], target: dict[str, Any]
) -> tuple[str, str, list[str]]:
    target_di = int(target["day_index"])

    prior_indices: list[int] = []
    for dk in (traj.get("days") or {}):
        di = traj["days"][dk].get("day_index")
        if isinstance(di, int) and di < target_di:
            prior_indices.append(di)

    sections: list[str] = ["=== USER PRIOR ACTIVITY (full detail) ==="]
    sections.append(serializer.serialize_trajectory_window(traj, prior_indices))

    sections.append(
        f"\n=== TARGET ANCHOR EVENT (Day{target_di}, seq{target['seq']}) ==="
    )
    sections.append(
        f"Event id: {target['event_id']}\n"
        f"Domain: {target['domain']}\n"
        f"Time bucket: {target['time_bucket']}\n"
        "(One image is attached below. No other detail of this event is provided.)"
    )

    sections.append(
        "\n=== QUESTION ===\n"
        "Based on the user's prior pattern and the attached image, classify the "
        "user's mood during the TARGET EVENT.\n"
        "Choose 1-3 mood keywords from the FULL candidate list (positive AND/OR "
        "negative). Declare polarity ('positive', 'negative', or 'mixed').\n"
        'Output JSON: {"polarity": "positive"|"negative"|"mixed", "moods": ["...","..."]}'
    )

    return EMOTION_IMAGE_SYSTEM, "\n".join(sections), [target["image_path"]]


# ---- Response parsing ----------------------------------------------------

def parse_response(content: str) -> dict[str, Any]:
    parsed = None
    try:
        parsed = json.loads(content)
    except Exception:
        s = content.find("{")
        e = content.rfind("}")
        if 0 <= s < e:
            try:
                parsed = json.loads(content[s : e + 1])
            except Exception:
                parsed = None

    if not isinstance(parsed, dict):
        return {"polarity": "unknown", "moods": [], "raw_polarity": None,
                "missing": True}

    raw_pol = parsed.get("polarity")
    pol = "unknown"
    if isinstance(raw_pol, str):
        rp = raw_pol.strip().lower()
        if rp.startswith("pos"):
            pol = "positive"
        elif rp.startswith("neg"):
            pol = "negative"
        elif rp.startswith("mix") or rp.startswith("both"):
            pol = "mixed"

    raw_moods = parsed.get("moods") or []
    if not isinstance(raw_moods, list):
        raw_moods = []
    canon: list[str] = []
    for m in raw_moods:
        if not isinstance(m, str):
            continue
        ml = m.strip().lower()
        if ml in ALL_MOODS_LOWER:
            cm = ALL_MOODS_LOWER[ml]
            if cm not in canon:
                canon.append(cm)
    canon = canon[:3]

    return {
        "polarity": pol,
        "moods": canon,
        "raw_polarity": raw_pol,
        "missing": False,
    }


# ---- Per-event metrics ---------------------------------------------------

def compute_event_metrics(
    target: dict[str, Any], parsed: dict[str, Any]
) -> dict[str, Any]:
    gt_set = set(target["gt_moods"])
    pred_set = set(parsed.get("moods") or [])
    gt_pol = target["gt_polarity"]
    pred_pol_eff = polarity_of(parsed.get("moods") or [])
    pred_pol_declared = parsed.get("polarity") or "unknown"

    polarity_correct = int(
        pred_pol_declared == gt_pol and pred_pol_eff == gt_pol
    )

    inter = len(gt_set & pred_set)
    union = len(gt_set | pred_set)
    f1 = (2 * inter / (len(gt_set) + len(pred_set))) if (gt_set or pred_set) else 0.0
    jacc = (inter / union) if union else 0.0
    top1 = 1 if inter >= 1 else 0

    return {
        "polarity_correct": polarity_correct,
        "mood_f1": round(f1, 4),
        "mood_jaccard": round(jacc, 4),
        "top1": top1,
        "pred_polarity_declared": pred_pol_declared,
        "pred_polarity_effective": pred_pol_eff,
        "n_pred": len(pred_set),
        "n_gt": len(gt_set),
        "n_inter": inter,
    }


# ---- Backwards-compat names ----------------------------------------------

def extract_targets(traj: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:  # noqa: D401
    """Alias of extract_targets_text(traj, k)."""
    return extract_targets_text(traj, k=k)


def build_prompt(traj: dict[str, Any], target: dict[str, Any]) -> tuple[str, str]:  # noqa: D401
    """Alias of build_prompt_text(traj, target)."""
    return build_prompt_text(traj, target)


def discover_eligible_users(min_events: int = 1) -> list[str]:  # noqa: D401
    """Alias of discover_eligible_users_text(min_events=min_events)."""
    return discover_eligible_users_text(min_events=min_events)
