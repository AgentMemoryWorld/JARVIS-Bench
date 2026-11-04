"""Compose (system, user) prompts for a given (user, condition)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import loader, serializer

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _read_system() -> str:
    return (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")


def build_user_context(
    user_slug: str,
    cond_cfg: dict[str, Any],
    persona_data: dict[str, Any],
    event_domain_filter=None,
) -> str:
    """Render the user-context block that varies by condition."""
    sections: list[str] = []

    if cond_cfg.get("include_trajectory"):
        traj = loader.load_trajectory(user_slug)
        max_days = cond_cfg.get("trajectory_days")
        sections.append("=== USER DAILY ACTIVITY LOG ===")
        sections.append(serializer.serialize_trajectory(
            traj, max_days=max_days, event_domain_filter=event_domain_filter
        ))

    if cond_cfg.get("include_persona"):
        pu = (persona_data or {}).get(user_slug)
        if pu:
            sections.append("\n=== USER PROFILE ===")
            sections.append(serializer.serialize_persona(pu))

    # Load profile_persona once if either flag needs it.
    pp = None
    if cond_cfg.get("include_profile_persona") or cond_cfg.get("include_survey_insight"):
        pp = loader.load_profile_persona(user_slug)

    if cond_cfg.get("include_profile_persona"):
        pp_text = serializer.serialize_profile_persona_no_insight(pp)
        if pp_text:
            sections.append("\n=== USER PROFILE (ENHANCED) ===")
            sections.append(pp_text)

    if cond_cfg.get("include_survey_insight"):
        si_text = serializer.serialize_survey_insight(pp)
        if si_text:
            sections.append("\n=== DISTILLED PREFERENCE SUMMARY ===")
            sections.append(si_text)

    if not sections:
        sections.append(
            "(No user-specific context provided. Answer with your best generic prior.)"
        )

    return "\n".join(sections)


def build_questions_block(questions: dict[str, dict[str, str]]) -> str:
    lines: list[str] = ["=== SURVEY QUESTIONS (answer all 100) ==="]
    for qno in sorted(questions.keys(), key=int):
        q = questions[qno]
        lines.append(f"\n{qno}. [{q['domain']} | {q['interaction_type']}] {q['question']}")
        lines.append(f"   A. {q['A']}")
        lines.append(f"   B. {q['B']}")
        lines.append(f"   C. {q['C']}")
        lines.append(f"   D. {q['D']}")
    lines.append(
        '\nReturn the JSON object now: {"1":"A","2":"B",...,"100":"D"}. '
        "All 100 keys must be present."
    )
    return "\n".join(lines)


def build_prompt(
    user_slug: str,
    cond_cfg: dict[str, Any],
    questions: dict[str, dict[str, str]],
    persona_data: dict[str, Any],
    event_domain_filter=None,
) -> tuple[str, str]:
    system = _read_system()
    user_ctx = build_user_context(user_slug, cond_cfg, persona_data, event_domain_filter=event_domain_filter)
    qblock = build_questions_block(questions)
    user = f"{user_ctx}\n\n{qblock}"
    return system, user
