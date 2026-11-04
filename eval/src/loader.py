"""Data IO for the eval pipeline.

All paths are resolved relative to the JarvisBench project root
(two levels up from this file: eval/src/loader.py -> JarvisBench/).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_CSV = ROOT / "JarvisBench_problem.csv"
SURVEY_JSON = ROOT / "Data" / "processed" / "survey.json"
PERSONA_JSON = ROOT / "Data" / "processed" / "persona.json"
SYNTHETIC_DIR = ROOT / "synthetic"


def load_questions() -> dict[str, dict[str, str]]:
    """Return {question_no: {no, domain, interaction_type, question, A, B, C, D}}."""
    out: dict[str, dict[str, str]] = {}
    with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            no = row["No."].strip()
            out[no] = {
                "no": no,
                "domain": row["Domain"].strip(),
                "interaction_type": row["Interaction Type"].strip(),
                "question": row["Question"].strip(),
                "A": row["Option A"].strip(),
                "B": row["Option B"].strip(),
                "C": row["Option C"].strip(),
                "D": row["Option D"].strip(),
            }
    return out


def load_ground_truth() -> dict[str, Any]:
    return json.load(open(SURVEY_JSON, encoding="utf-8"))


def load_persona() -> dict[str, Any]:
    return json.load(open(PERSONA_JSON, encoding="utf-8"))


def load_trajectory(user_slug: str) -> dict[str, Any]:
    path = SYNTHETIC_DIR / user_slug / "synthetic_trajectory.json"
    return json.load(open(path, encoding="utf-8"))


def load_profile_persona(user_slug: str) -> dict[str, Any] | None:
    """Includes survey_insight (used only for C4 skyline)."""
    path = SYNTHETIC_DIR / user_slug / "profile_persona.json"
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf-8"))


def load_skeleton(user_slug: str) -> dict[str, Any] | None:
    """Has clue_plan, used by aggregator for planted/difficulty/clue_type slices."""
    path = SYNTHETIC_DIR / user_slug / "skeleton.json"
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf-8"))


def discover_users() -> list[str]:
    """List user slugs that have a synthetic_trajectory.json on disk."""
    if not SYNTHETIC_DIR.exists():
        return []
    out = []
    for d in sorted(SYNTHETIC_DIR.iterdir()):
        if d.is_dir() and (d / "synthetic_trajectory.json").exists():
            out.append(d.name)
    return out
