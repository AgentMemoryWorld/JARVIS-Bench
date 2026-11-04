"""Aggregate emotion-task results across users x models, both TEXT and IMAGE.

Run folder layout (new):
  eval/runs/emotion_text__{user}__{model}/
    config.snapshot.json
    targets.json
    parsed/{event_id}.json
  eval/runs/emotion_image__{user}__{model}/
    same layout

Outputs (eval/runs/_summary/):
  emotion_text__per_user.csv
  emotion_text__per_event.csv
  emotion_text__average.csv     # macro-mean across users
  emotion_text__pool.csv        # micro-mean over all events
  emotion_text__report.{md,txt}

  emotion_image__per_user.csv
  emotion_image__per_event.csv
  emotion_image__pool.csv
  emotion_image__report.{md,txt}

  emotion__eligibility.csv
  emotion__eligibility_report.{md,txt}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import loader, render
from .tasks import emotion as etask

EVAL_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = EVAL_ROOT / "runs"
SUMMARY_DIR = RUNS_DIR / "_summary"


def _iter_runs(prefix: str):
    if not RUNS_DIR.exists():
        return
    for d in sorted(RUNS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(prefix):
            yield d


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# ---- Eligibility --------------------------------------------------------

def collect_eligibility() -> list[dict[str, Any]]:
    """v3: text and image share the same pool (last 5 days + image + mood)."""
    rows: list[dict[str, Any]] = []
    for u in loader.discover_users():
        try:
            traj = loader.load_trajectory(u)
        except Exception:
            continue
        targets = etask.extract_targets_emotion(traj, k_days=5)
        n = len(targets)
        n_pos = sum(1 for t in targets if t.get("gt_polarity") == "positive")
        n_neg = sum(1 for t in targets if t.get("gt_polarity") == "negative")
        n_mix = sum(1 for t in targets if t.get("gt_polarity") == "mixed")
        rows.append(
            {
                "user_slug": u,
                "n_samples": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_mix": n_mix,
                "eligible": int(n >= 1),
            }
        )
    rows.sort(key=lambda r: r["user_slug"])
    return rows


# ---- Per-event collection (works for both variants) ---------------------

def collect_variant(prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """prefix = 'emotion_text__' or 'emotion_image__'.
    Returns (per_user_rows, per_event_rows)."""
    per_event: list[dict[str, Any]] = []
    per_user: list[dict[str, Any]] = []

    for run_dir in _iter_runs(prefix):
        snap_path = run_dir / "config.snapshot.json"
        if not snap_path.exists():
            continue
        snap = json.load(open(snap_path, encoding="utf-8"))
        user_slug = snap.get("user_slug")
        model_key = snap.get("model")
        if not user_slug or not model_key:
            continue
        targets_path = run_dir / "targets.json"
        if not targets_path.exists():
            continue
        targets = json.load(open(targets_path, encoding="utf-8"))

        parsed_dir = run_dir / "parsed"
        n = 0
        sum_pol = 0
        sum_f1 = 0.0
        sum_jacc = 0.0
        sum_top1 = 0

        for tgt in targets:
            eid = tgt["event_id"]
            parsed_path = parsed_dir / f"{eid}.json"
            parsed: dict[str, Any] = {}
            if parsed_path.exists():
                try:
                    parsed = json.load(open(parsed_path, encoding="utf-8"))
                except Exception:
                    parsed = {}
            m = etask.compute_event_metrics(tgt, parsed)
            n += 1
            sum_pol += m["polarity_correct"]
            sum_f1 += m["mood_f1"]
            sum_jacc += m["mood_jaccard"]
            sum_top1 += m["top1"]
            per_event.append(
                {
                    "user_slug": user_slug,
                    "model": model_key,
                    "event_id": eid,
                    "day_index": tgt["day_index"],
                    "seq": tgt["seq"],
                    "gt_polarity": tgt["gt_polarity"],
                    "gt_moods": ",".join(tgt["gt_moods"]),
                    "pred_polarity_declared": m["pred_polarity_declared"],
                    "pred_polarity_effective": m["pred_polarity_effective"],
                    "pred_moods": ",".join(parsed.get("moods") or []),
                    "polarity_correct": m["polarity_correct"],
                    "mood_f1": m["mood_f1"],
                    "mood_jaccard": m["mood_jaccard"],
                    "top1": m["top1"],
                    "missing": int(bool(parsed.get("missing"))),
                }
            )

        if n > 0:
            per_user.append(
                {
                    "user_slug": user_slug,
                    "model": model_key,
                    "n_events": n,
                    "polarity_accuracy": round(sum_pol / n, 4),
                    "mood_f1_mean": round(sum_f1 / n, 4),
                    "mood_jaccard_mean": round(sum_jacc / n, 4),
                    "top1_accuracy": round(sum_top1 / n, 4),
                }
            )

    per_event.sort(key=lambda r: (r["model"], r["user_slug"], r["day_index"], r["seq"]))
    per_user.sort(key=lambda r: (r["model"], r["user_slug"]))
    return per_user, per_event


# ---- Pool computation ---------------------------------------------------

def _pool_metrics(per_event: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, float]] = {}
    for row in per_event:
        m = row["model"]
        d = by_model.setdefault(m, {"n": 0, "pol": 0, "f1": 0.0,
                                     "jacc": 0.0, "top1": 0})
        d["n"] += 1
        d["pol"] += row["polarity_correct"]
        d["f1"] += row["mood_f1"]
        d["jacc"] += row["mood_jaccard"]
        d["top1"] += row["top1"]
    out: list[dict[str, Any]] = []
    for m, d in sorted(by_model.items()):
        n = d["n"]
        out.append({
            "model": m,
            "n_events": n,
            "polarity_accuracy": round(_safe_div(d["pol"], n), 4),
            "mood_f1_mean": round(_safe_div(d["f1"], n), 4),
            "mood_jaccard_mean": round(_safe_div(d["jacc"], n), 4),
            "top1_accuracy": round(_safe_div(d["top1"], n), 4),
        })
    return out


# ---- Public entry -------------------------------------------------------

def write_summary() -> Path:
    try:
        import pandas as pd
    except ImportError:
        print("[aggregate-emotion] pandas not installed.")
        return SUMMARY_DIR

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    elig_rows = collect_eligibility()
    if elig_rows:
        pd.DataFrame(elig_rows).to_csv(
            SUMMARY_DIR / "emotion__eligibility.csv", index=False
        )
        elig_headers = list(elig_rows[0].keys())
        elig_table = [[r[h] for h in elig_headers] for r in elig_rows]
        n_elig = sum(1 for r in elig_rows if r["eligible"])
        n_total = sum(r["n_samples"] for r in elig_rows)
        n_pos = sum(r["n_pos"] for r in elig_rows)
        n_neg = sum(r["n_neg"] for r in elig_rows)
        n_mix = sum(r["n_mix"] for r in elig_rows)
        elig_summary = [
            ["Eligible users (>=1 sample in last 5d)", n_elig, ""],
            ["Total samples (last 5d + image + mood)",
             n_total, f"{n_pos} pos / {n_neg} neg / {n_mix} mixed"],
        ]
        render.write_report(
            path_md=SUMMARY_DIR / "emotion__eligibility_report.md",
            path_txt=SUMMARY_DIR / "emotion__eligibility_report.txt",
            title="Emotion Task — Eligibility (v3 last-5-days + image + mood)",
            sections=[
                ("Pool summary",
                 ["metric", "value", "note"],
                 elig_summary),
                ("Per-user breakdown", elig_headers, elig_table),
            ],
        )

    for prefix, label in [("emotion_text__", "Emotion Text"),
                          ("emotion_image__", "Emotion Image")]:
        per_user, per_event = collect_variant(prefix)
        if not per_user:
            print(f"[aggregate-emotion] No {prefix.rstrip('_')} runs found.")
            continue

        pe_df = pd.DataFrame(per_event)
        pu_df = pd.DataFrame(per_user)

        suffix = prefix.rstrip("_")
        pe_df.to_csv(SUMMARY_DIR / f"{suffix}__per_event.csv", index=False)
        pu_df.to_csv(SUMMARY_DIR / f"{suffix}__per_user.csv", index=False)

        if prefix == "emotion_text__":
            avg = pu_df.groupby("model").agg(
                n_users=("user_slug", "nunique"),
                polarity_accuracy_mean=("polarity_accuracy", "mean"),
                polarity_accuracy_std=("polarity_accuracy", "std"),
                mood_f1_mean=("mood_f1_mean", "mean"),
                mood_f1_std=("mood_f1_mean", "std"),
                mood_jaccard_mean=("mood_jaccard_mean", "mean"),
                top1_accuracy_mean=("top1_accuracy", "mean"),
            ).reset_index()
            for c in ("polarity_accuracy_std", "mood_f1_std"):
                avg[c] = avg[c].fillna(0.0)
            for c in avg.columns:
                if avg[c].dtype.kind == "f":
                    avg[c] = avg[c].round(4)
            avg.to_csv(SUMMARY_DIR / f"{suffix}__average.csv", index=False)
            avg_section = ("Average across users (macro-mean)",
                           list(avg.columns), avg.values.tolist())
        else:
            avg_section = None

        pool = _pool_metrics(per_event)
        pd.DataFrame(pool).to_csv(SUMMARY_DIR / f"{suffix}__pool.csv", index=False)

        sections = [
            ("Pool metrics (micro across all events)",
             list(pool[0].keys()) if pool else [],
             [list(r.values()) for r in pool]),
        ]
        if avg_section is not None:
            sections.append(avg_section)
        sections.append(
            ("Per-user breakdown", list(pu_df.columns), pu_df.values.tolist())
        )
        render.write_report(
            path_md=SUMMARY_DIR / f"{suffix}__report.md",
            path_txt=SUMMARY_DIR / f"{suffix}__report.txt",
            title=f"{label} — Summary",
            sections=sections,
        )

    print(f"[aggregate-emotion] Wrote summary to {SUMMARY_DIR}")
    return SUMMARY_DIR
